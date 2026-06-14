"""Recurrent State-Space Model (RSSM) -- the core of the DreamerV3 world model.

Each latent state has two parts:

* a **deterministic** recurrent state ``h`` carried by a LayerNorm GRU, and
* a **stochastic** state ``z`` drawn from ``groups`` independent categoricals
  (``classes`` each), with straight-through gradients and 1% uniform mixing.

The model exposes a *prior* ``p(z_t | h_t)`` (its imagination of what comes next) and a
*posterior* ``q(z_t | h_t, x_t)`` (corrected by the actual observation). Training pulls
the prior toward the posterior (KL) so the model can later dream forward without inputs.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ...common.distributions import categorical_kl, onehot_straight_through, unimix_logits
from ...common.nets import LayerNormGRUCell, activation


class RSSM(nn.Module):
    def __init__(self, embed_dim: int, action_dim: int, deter: int = 512, groups: int = 32,
                 classes: int = 32, hidden: int = 256, act: str = "silu", unimix: float = 0.01):
        super().__init__()
        self.deter = deter
        self.groups = groups
        self.classes = classes
        self.stoch_dim = groups * classes
        self.unimix = unimix
        self.feat_dim = deter + self.stoch_dim

        self.img_in = nn.Sequential(
            nn.Linear(self.stoch_dim + action_dim, hidden), nn.LayerNorm(hidden), activation(act)
        )
        self.cell = LayerNormGRUCell(hidden, deter)
        self.prior_net = nn.Sequential(
            nn.Linear(deter, hidden), nn.LayerNorm(hidden), activation(act),
            nn.Linear(hidden, self.stoch_dim),
        )
        self.post_net = nn.Sequential(
            nn.Linear(deter + embed_dim, hidden), nn.LayerNorm(hidden), activation(act),
            nn.Linear(hidden, self.stoch_dim),
        )

    # -- state helpers ----------------------------------------------------
    def initial(self, batch: int, device: torch.device) -> dict[str, Tensor]:
        return {
            "deter": torch.zeros(batch, self.deter, device=device),
            "stoch": torch.zeros(batch, self.groups, self.classes, device=device),
        }

    def get_feat(self, state: dict[str, Tensor]) -> Tensor:
        stoch = state["stoch"].reshape(*state["stoch"].shape[:-2], self.stoch_dim)
        return torch.cat([state["deter"], stoch], dim=-1)

    def _stoch_flat(self, state: dict[str, Tensor]) -> Tensor:
        return state["stoch"].reshape(*state["stoch"].shape[:-2], self.stoch_dim)

    def _to_dist(self, raw_logits: Tensor) -> tuple[Tensor, Tensor]:
        """Reshape head output to grouped logits, apply unimix, sample straight-through."""
        logits = raw_logits.reshape(*raw_logits.shape[:-1], self.groups, self.classes)
        logits = unimix_logits(logits, self.unimix)
        return onehot_straight_through(logits), logits

    # -- transitions ------------------------------------------------------
    def img_step(self, state: dict[str, Tensor], action: Tensor):
        """Prior transition: predict the next state from the current state + action."""
        x = self.img_in(torch.cat([self._stoch_flat(state), action], dim=-1))
        deter = self.cell(x, state["deter"])
        stoch, logits = self._to_dist(self.prior_net(deter))
        return {"deter": deter, "stoch": stoch}, logits

    def obs_step(self, state: dict[str, Tensor], prev_action: Tensor, embed: Tensor,
                 is_first: Tensor):
        """Posterior transition: prior step, then correct the latent with observation ``embed``."""
        keep = (1.0 - is_first)[..., None]
        state = {"deter": state["deter"] * keep, "stoch": state["stoch"] * keep[..., None]}
        prev_action = prev_action * keep

        prior, prior_logits = self.img_step(state, prev_action)
        stoch, post_logits = self._to_dist(self.post_net(torch.cat([prior["deter"], embed], dim=-1)))
        post = {"deter": prior["deter"], "stoch": stoch}
        return post, post_logits, prior_logits

    def observe(self, embed: Tensor, action: Tensor, is_first: Tensor,
                state: dict[str, Tensor] | None = None):
        """Roll the posterior over a sequence. ``embed/action/is_first`` are ``[B, T, ...]``.

        ``action[:, t]`` is the action taken *at* obs ``t``; the posterior for step ``t``
        is conditioned on the action that led into it (``action[:, t-1]``), so we shift.
        """
        b, t = embed.shape[:2]
        if state is None:
            state = self.initial(b, embed.device)
        zeros = torch.zeros_like(action[:, :1])
        prev_actions = torch.cat([zeros, action[:, :-1]], dim=1)

        deters, stochs, post_logits, prior_logits = [], [], [], []
        for i in range(t):
            state, post_l, prior_l = self.obs_step(
                state, prev_actions[:, i], embed[:, i], is_first[:, i]
            )
            deters.append(state["deter"])
            stochs.append(state["stoch"])
            post_logits.append(post_l)
            prior_logits.append(prior_l)

        states = {"deter": torch.stack(deters, 1), "stoch": torch.stack(stochs, 1)}
        return states, torch.stack(post_logits, 1), torch.stack(prior_logits, 1), state

    def imagine(self, actor, state: dict[str, Tensor], horizon: int):
        """Dream ``horizon`` steps forward from ``state`` using ``actor``.

        Returns stacked features ``[H+1, N, feat]`` and actions ``[H, N, act]`` (with
        gradients flowing through the dynamics for the policy update).
        """
        feats, actions = [self.get_feat(state)], []
        for _ in range(horizon):
            action = actor(feats[-1])
            actions.append(action)
            state, _ = self.img_step(state, action)
            feats.append(self.get_feat(state))
        return torch.stack(feats, 0), torch.stack(actions, 0)

    # -- losses -----------------------------------------------------------
    def kl_loss(self, post_logits: Tensor, prior_logits: Tensor, free: float = 1.0,
                dyn_scale: float = 0.5, rep_scale: float = 0.1):
        """KL with free bits and DreamerV3's dyn/rep balancing."""
        dyn = categorical_kl(post_logits.detach(), prior_logits)  # train prior toward posterior
        rep = categorical_kl(post_logits, prior_logits.detach())  # train posterior toward prior
        dyn = torch.clamp(dyn, min=free).mean()
        rep = torch.clamp(rep, min=free).mean()
        loss = dyn_scale * dyn + rep_scale * rep
        return loss, {"kl_dyn": dyn.detach(), "kl_rep": rep.detach()}
