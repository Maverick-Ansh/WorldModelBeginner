"""TD-MPC2 latent world model: encoder, dynamics, reward, Q-ensemble and policy prior.

Unlike DreamerV3 there is *no* observation decoder: the latent is shaped purely by a
self-consistency loss (predicted latent must match the encoding of the next obs), a
reward loss and a TD value loss. The latent uses SimNorm so it stays bounded and sparse.
A target network (EMA of encoder/dynamics/Q) stabilizes the TD targets.
"""

from __future__ import annotations

import copy

import torch
from torch import Tensor, nn

from ...common.distributions import SymlogTwoHot, TanhNormal, symlog
from ...common.nets import SimNorm, build_mlp, zero_init_


def _latent_head(in_dim: int, hidden: int, latent_dim: int, layers: int, group: int) -> nn.Module:
    return nn.Sequential(build_mlp(in_dim, hidden, latent_dim, layers), SimNorm(group))


class TDMPC2WorldModel(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, cfg):
        super().__init__()
        self.act_dim = act_dim
        self.num_q = cfg.num_q
        latent, hidden, layers = cfg.latent_dim, cfg.hidden, cfg.mlp_layers
        group = cfg.simnorm_group

        self.encoder = _latent_head(obs_dim, hidden, latent, layers, group)
        self.dynamics = _latent_head(latent + act_dim, hidden, latent, layers, group)
        self.reward = build_mlp(latent + act_dim, hidden, cfg.num_bins, layers)
        self.qs = nn.ModuleList(
            [build_mlp(latent + act_dim, hidden, cfg.num_bins, layers) for _ in range(self.num_q)]
        )
        for q in self.qs:
            zero_init_(q)
        self.pi = build_mlp(latent, hidden, 2 * act_dim, layers)
        self.pi_min_std, self.pi_max_std = cfg.actor_min_std, cfg.actor_max_std

        # EMA target network (encoder + dynamics + Q).
        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_dynamics = copy.deepcopy(self.dynamics)
        self.target_qs = copy.deepcopy(self.qs)
        for module in (self.target_encoder, self.target_dynamics, self.target_qs):
            for p in module.parameters():
                p.requires_grad_(False)

    # -- forward pieces ---------------------------------------------------
    def encode(self, obs: Tensor, target: bool = False) -> Tensor:
        enc = self.target_encoder if target else self.encoder
        return enc(symlog(obs))

    def next(self, z: Tensor, action: Tensor, target: bool = False) -> Tensor:
        dyn = self.target_dynamics if target else self.dynamics
        return dyn(torch.cat([z, action], dim=-1))

    def reward_dist(self, z: Tensor, action: Tensor) -> SymlogTwoHot:
        return SymlogTwoHot(self.reward(torch.cat([z, action], dim=-1)))

    def q_dists(self, z: Tensor, action: Tensor) -> list[SymlogTwoHot]:
        za = torch.cat([z, action], dim=-1)
        return [SymlogTwoHot(q(za)) for q in self.qs]

    def q_values(self, z: Tensor, action: Tensor, target: bool = False) -> Tensor:
        """Return per-ensemble mean values, shape ``[..., num_q]``."""
        za = torch.cat([z, action], dim=-1)
        modules = self.target_qs if target else self.qs
        return torch.stack([SymlogTwoHot(q(za)).mean for q in modules], dim=-1)

    def q_reduced(self, z: Tensor, action: Tensor, target: bool = False) -> Tensor:
        """Min over two random ensemble members (clipped double-Q)."""
        values = self.q_values(z, action, target=target)
        idx = torch.randperm(self.num_q, device=values.device)[:2]
        return values[..., idx].min(dim=-1).values

    # -- policy prior -----------------------------------------------------
    def pi_dist(self, z: Tensor) -> TanhNormal:
        mean, std = self.pi(z).chunk(2, dim=-1)
        std = (self.pi_max_std - self.pi_min_std) * torch.sigmoid(std) + self.pi_min_std
        return TanhNormal(mean, std)

    @torch.no_grad()
    def soft_update_target(self, tau: float) -> None:
        for online, target in (
            (self.encoder, self.target_encoder),
            (self.dynamics, self.target_dynamics),
            (self.qs, self.target_qs),
        ):
            for op, tp in zip(online.parameters(), target.parameters()):
                tp.mul_(1.0 - tau).add_(op, alpha=tau)
