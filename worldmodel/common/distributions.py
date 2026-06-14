"""Distributions & transforms shared by DreamerV3 and TD-MPC2.

Implements the numerically-robust pieces that make modern world models train with
fixed hyper-parameters across reward/observation scales:

* ``symlog`` / ``symexp`` -- bi-symmetric log compression of wide-range targets.
* :class:`SymlogTwoHot` -- discrete (two-hot) regression of scalars in symlog space,
  used for reward and value heads (DreamerV3, TD-MPC2).
* :class:`SymlogMSE` -- Gaussian-in-symlog-space head for reconstructing observations.
* :class:`TanhNormal` -- squashed Gaussian policy over actions in ``[-1, 1]``.
* :func:`unimix_logits` / :func:`onehot_straight_through` -- categorical latents with
  straight-through gradients for the RSSM.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


# --------------------------------------------------------------------------- #
# symlog                                                                       #
# --------------------------------------------------------------------------- #
def symlog(x: Tensor) -> Tensor:
    """Bi-symmetric log: ``sign(x) * log(|x| + 1)``."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: Tensor) -> Tensor:
    """Inverse of :func:`symlog`."""
    return torch.sign(x) * torch.expm1(torch.abs(x))


# --------------------------------------------------------------------------- #
# two-hot discrete regression (reward / value)                                #
# --------------------------------------------------------------------------- #
class SymlogTwoHot:
    """Discrete regression of a scalar via two-hot targets over symlog-spaced bins.

    The network outputs ``logits`` over ``num_bins`` buckets uniformly spaced in
    symlog space ``[low, high]``. The predicted scalar is ``symexp`` of the softmax
    expectation; the loss is the cross-entropy against the two-hot encoding of the
    (symlog-transformed) target. This makes value/reward learning insensitive to
    magnitude and far more stable than plain MSE.
    """

    def __init__(self, logits: Tensor, low: float = -20.0, high: float = 20.0):
        self.logits = logits
        self.num_bins = logits.shape[-1]
        self.bins = torch.linspace(low, high, self.num_bins, device=logits.device, dtype=logits.dtype)
        self.probs = torch.softmax(logits, dim=-1)

    @property
    def mean(self) -> Tensor:
        """Expected value in *real* space (shape == logits.shape[:-1])."""
        return symexp((self.probs * self.bins).sum(dim=-1))

    mode = mean  # the expectation is the natural point estimate for a unimodal head

    def two_hot(self, x: Tensor) -> Tensor:
        """Two-hot encode ``symlog(x)`` over ``self.bins``."""
        y = symlog(x).clamp(self.bins[0], self.bins[-1])
        below = (self.bins <= y[..., None]).sum(dim=-1) - 1
        below = below.clamp(0, self.num_bins - 1)
        above = (below + 1).clamp(0, self.num_bins - 1)
        equal = below == above
        d_below = torch.where(equal, torch.ones_like(y), (y - self.bins[below]).abs())
        d_above = torch.where(equal, torch.ones_like(y), (self.bins[above] - y).abs())
        total = d_below + d_above
        w_below = (d_above / total)[..., None]
        w_above = (d_below / total)[..., None]
        target = (
            F.one_hot(below, self.num_bins) * w_below + F.one_hot(above, self.num_bins) * w_above
        )
        return target.to(self.logits.dtype)

    def log_prob(self, x: Tensor) -> Tensor:
        """Log-likelihood of target ``x`` under the two-hot distribution."""
        target = self.two_hot(x)
        log_probs = self.logits - torch.logsumexp(self.logits, dim=-1, keepdim=True)
        return (target * log_probs).sum(dim=-1)


class SymlogMSE:
    """Gaussian (unit-variance) head in symlog space -- used to reconstruct vectors."""

    def __init__(self, mean: Tensor):
        self._mean = mean

    @property
    def mode(self) -> Tensor:
        return symexp(self._mean)

    mean = mode

    def log_prob(self, x: Tensor) -> Tensor:
        # Up to an additive constant this is -0.5 * ||pred - symlog(x)||^2, summed
        # over the feature dimension (a diagonal unit-variance Gaussian).
        return -0.5 * ((self._mean - symlog(x)) ** 2).sum(dim=-1)


# --------------------------------------------------------------------------- #
# squashed Gaussian policy                                                     #
# --------------------------------------------------------------------------- #
class TanhNormal:
    """Tanh-squashed diagonal Gaussian over actions in ``[-1, 1]`` (SAC-style)."""

    def __init__(self, mean: Tensor, std: Tensor):
        self.normal = torch.distributions.Normal(mean, std)

    @property
    def mode(self) -> Tensor:
        return torch.tanh(self.normal.mean)

    def rsample(self) -> tuple[Tensor, Tensor]:
        """Reparameterised sample plus its log-prob (with the tanh correction)."""
        pre = self.normal.rsample()
        action = torch.tanh(pre)
        # log_prob with change-of-variables for tanh; the 1e-6 guards log(0).
        log_prob = self.normal.log_prob(pre) - torch.log(1.0 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=-1)

    def log_prob(self, action: Tensor) -> Tensor:
        action = action.clamp(-0.999999, 0.999999)
        pre = torch.atanh(action)
        log_prob = self.normal.log_prob(pre) - torch.log(1.0 - action.pow(2) + 1e-6)
        return log_prob.sum(dim=-1)

    def entropy(self) -> Tensor:
        # Entropy of the base Gaussian (a cheap, monotone proxy for the squashed one).
        return self.normal.entropy().sum(dim=-1)


# --------------------------------------------------------------------------- #
# categorical latents (RSSM)                                                   #
# --------------------------------------------------------------------------- #
def unimix_logits(logits: Tensor, unimix: float = 0.01) -> Tensor:
    """Mix categorical probabilities with a uniform prior, return new logits.

    Following DreamerV3, blending 1% uniform mass prevents the latent posterior from
    collapsing to deterministic codes and keeps the KL well-behaved.
    """
    if unimix <= 0.0:
        return logits
    probs = torch.softmax(logits, dim=-1)
    uniform = torch.ones_like(probs) / probs.shape[-1]
    probs = (1.0 - unimix) * probs + unimix * uniform
    return torch.log(probs)


def onehot_straight_through(logits: Tensor) -> Tensor:
    """Sample a one-hot from ``logits`` with a straight-through gradient estimator."""
    dist = torch.distributions.OneHotCategoricalStraightThrough(logits=logits)
    return dist.rsample()


def categorical_kl(logits_q: Tensor, logits_p: Tensor) -> Tensor:
    """KL(q || p) for grouped categoricals, summed over the group dimension."""
    q = torch.distributions.OneHotCategorical(logits=logits_q)
    p = torch.distributions.OneHotCategorical(logits=logits_p)
    return torch.distributions.kl_divergence(q, p).sum(dim=-1)
