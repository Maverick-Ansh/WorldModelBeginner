"""MPPI / iterated-CEM planner for TD-MPC2.

At each decision the planner imagines many action sequences through the learned latent
dynamics, scores them by discounted predicted reward plus a terminal Q-value, and
refits a per-step Gaussian toward the high-return "elite" trajectories. A fraction of
the candidates are drawn from the policy prior, which both warm-starts planning and
guarantees the planner is never worse than the amortized policy. Fully vectorized over
the ``num_envs`` parallel environments.
"""

from __future__ import annotations

import torch
from torch import Tensor


class MPPIPlanner:
    def __init__(self, cfg):
        self.horizon = cfg.plan_horizon
        self.iterations = cfg.plan_iterations
        self.num_samples = cfg.plan_samples
        self.num_elites = cfg.plan_elites
        self.num_pi = cfg.plan_pi_samples
        self.temperature = cfg.plan_temperature
        self.min_std = cfg.min_plan_std
        self.max_std = cfg.max_plan_std
        self.gamma = cfg.gamma

    @torch.no_grad()
    def _pi_trajectories(self, model, z0: Tensor) -> Tensor:
        """Roll ``num_pi`` policy-prior action sequences. Returns ``[E, num_pi, H, A]``."""
        e = z0.shape[0]
        z = z0[:, None].expand(e, self.num_pi, z0.shape[-1]).reshape(e * self.num_pi, -1)
        actions = []
        for _ in range(self.horizon):
            a = model.pi_dist(z).rsample()[0]
            actions.append(a)
            z = model.next(z, a)
        traj = torch.stack(actions, dim=1)  # [E*num_pi, H, A]
        return traj.reshape(e, self.num_pi, self.horizon, model.act_dim)

    @torch.no_grad()
    def _evaluate(self, model, z0: Tensor, actions: Tensor) -> Tensor:
        """Estimate the return of each candidate. ``actions`` is ``[E, M, H, A]`` -> ``[E, M]``."""
        e, m = actions.shape[:2]
        z = z0[:, None].expand(e, m, z0.shape[-1]).reshape(e * m, -1)
        g = torch.zeros(e * m, device=z0.device)
        discount = 1.0
        for t in range(self.horizon):
            a = actions[:, :, t].reshape(e * m, -1)
            g = g + discount * model.reward_dist(z, a).mean
            z = model.next(z, a)
            discount *= self.gamma
        terminal = model.q_reduced(z, model.pi_dist(z).mode)
        g = g + discount * terminal
        return g.reshape(e, m)

    @torch.no_grad()
    def plan(self, model, z0: Tensor, prev_mean: Tensor | None, training: bool):
        e, device = z0.shape[0], z0.device
        a_dim = model.act_dim
        mean = prev_mean if prev_mean is not None else torch.zeros(e, self.horizon, a_dim, device=device)
        std = torch.full((e, self.horizon, a_dim), self.max_std, device=device)
        pi_traj = self._pi_trajectories(model, z0)

        for _ in range(self.iterations):
            eps = torch.randn(e, self.num_samples, self.horizon, a_dim, device=device)
            gaussian = (mean[:, None] + std[:, None] * eps).clamp(-1.0, 1.0)
            actions = torch.cat([pi_traj, gaussian], dim=1)  # [E, M, H, A]

            value = self._evaluate(model, z0, actions)       # [E, M]
            elite_val, elite_idx = value.topk(self.num_elites, dim=1)
            idx = elite_idx[:, :, None, None].expand(-1, -1, self.horizon, a_dim)
            elite_actions = torch.gather(actions, 1, idx)    # [E, Ne, H, A]

            weights = torch.softmax(
                (elite_val - elite_val.max(dim=1, keepdim=True).values) / self.temperature, dim=1
            )
            w = weights[:, :, None, None]
            mean = (w * elite_actions).sum(dim=1)
            var = (w * (elite_actions - mean[:, None]) ** 2).sum(dim=1)
            std = var.sqrt().clamp(self.min_std, self.max_std)

        action = mean[:, 0]
        if training:
            action = (action + std[:, 0] * torch.randn_like(action)).clamp(-1.0, 1.0)
        return action, mean
