"""Actor-critic learned entirely inside imagined RSSM rollouts (DreamerV3 behavior)."""

from __future__ import annotations

import copy

import torch
from torch import Tensor, nn

from ...common.distributions import SymlogTwoHot, TanhNormal
from ...common.nets import build_mlp, zero_init_
from ...common.utils import Moments, soft_update


class Actor(nn.Module):
    def __init__(self, feat_dim: int, act_dim: int, hidden: int, layers: int,
                 min_std: float = 0.1, max_std: float = 1.0):
        super().__init__()
        self.net = build_mlp(feat_dim, hidden, 2 * act_dim, layers)
        self.min_std, self.max_std = min_std, max_std

    def dist(self, feat: Tensor) -> TanhNormal:
        mean, std = self.net(feat).chunk(2, dim=-1)
        std = (self.max_std - self.min_std) * torch.sigmoid(std) + self.min_std
        return TanhNormal(mean, std)

    def forward(self, feat: Tensor) -> Tensor:
        # Reparameterized action used during imagination (gradients flow to the policy).
        action, _ = self.dist(feat).rsample()
        return action


class Critic(nn.Module):
    def __init__(self, feat_dim: int, hidden: int, layers: int, num_bins: int):
        super().__init__()
        self.net = build_mlp(feat_dim, hidden, num_bins, layers)
        zero_init_(self.net)

    def dist(self, feat: Tensor) -> SymlogTwoHot:
        return SymlogTwoHot(self.net(feat))

    def value(self, feat: Tensor) -> Tensor:
        return self.dist(feat).mean


class Behavior(nn.Module):
    """Holds the actor, critic and EMA target critic, and computes their losses."""

    def __init__(self, feat_dim: int, act_dim: int, cfg):
        super().__init__()
        self.actor = Actor(feat_dim, act_dim, cfg.hidden, cfg.mlp_layers,
                           cfg.actor_min_std, cfg.actor_max_std)
        self.critic = Critic(feat_dim, cfg.hidden, cfg.mlp_layers, cfg.num_bins)
        self.slow_critic = copy.deepcopy(self.critic)
        for p in self.slow_critic.parameters():
            p.requires_grad_(False)
        self.moments = Moments()
        self.gamma = cfg.gamma
        self.lam = cfg.lam
        self.ent_coef = cfg.ent_coef
        self.slow_tau = cfg.critic_slow_tau

    @staticmethod
    def lambda_return(reward: Tensor, value: Tensor, disc: Tensor, lam: float) -> Tensor:
        """λ-returns for states s_0..s_{H-1}. All inputs are over the *next* states, [H, N]."""
        horizon = reward.shape[0]
        out = torch.empty_like(reward)
        last = value[-1]
        for t in reversed(range(horizon)):
            last = reward[t] + disc[t] * ((1.0 - lam) * value[t] + lam * last)
            out[t] = last
        return out

    def learn(self, world_model, start: dict[str, Tensor], horizon: int):
        """One behavior update. Returns ``(actor_loss, critic_loss, metrics)``."""
        feats, _ = world_model.rssm.imagine(self.actor, start, horizon)  # [H+1, N, F]

        reward = world_model.reward_dist(feats).mean             # [H+1, N]
        cont = world_model.cont_dist(feats).mean                 # continue probability
        disc = self.gamma * cont
        with torch.no_grad():
            target_value = self.slow_critic.value(feats)         # [H+1, N]

        returns = self.lambda_return(reward[1:], target_value[1:], disc[1:], self.lam)  # [H, N]
        weights = torch.cumprod(
            torch.cat([torch.ones_like(disc[:1]), disc[1:-1]], dim=0), dim=0
        ).detach()                                               # [H, N], weights[0] = 1

        # -- actor: maximize normalized returns + entropy (dynamics backprop) --
        self.moments.update(returns)
        normed = (returns - self.moments.offset()) / self.moments.scale()
        entropy = self.actor.dist(feats[:-1]).entropy()
        actor_loss = -(weights * normed).mean() - self.ent_coef * (weights * entropy).mean()

        # -- critic: regress two-hot value toward λ-returns --
        value_dist = self.critic.dist(feats[:-1].detach())
        critic_loss = -(weights * value_dist.log_prob(returns.detach())).mean()

        metrics = {
            "actor_loss": actor_loss.detach(),
            "critic_loss": critic_loss.detach(),
            "actor_entropy": entropy.mean().detach(),
            "imag_return": returns.mean().detach(),
            "imag_value": value_dist.mean.mean().detach(),
        }
        return actor_loss, critic_loss, metrics

    def update_slow(self) -> None:
        soft_update(self.slow_critic, self.critic, self.slow_tau)
