"""TD-MPC2 agent: joint latent-model + value learning, plus MPPI planning to act."""

from __future__ import annotations

import itertools

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ...common.utils import to_tensor
from ..base import Agent
from .planner import MPPIPlanner
from .world_model import TDMPC2WorldModel


class TDMPC2Agent(Agent):
    def __init__(self, obs_dim: int, act_dim: int, cfg, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        algo = cfg.algo
        self.model = TDMPC2WorldModel(obs_dim, act_dim, algo).to(device)
        self.planner = MPPIPlanner(algo)

        model_params = itertools.chain(
            self.model.encoder.parameters(),
            self.model.dynamics.parameters(),
            self.model.reward.parameters(),
            self.model.qs.parameters(),
        )
        self.model_opt = torch.optim.Adam(model_params, lr=algo.lr)
        self.pi_opt = torch.optim.Adam(self.model.pi.parameters(), lr=algo.lr)

        self.horizon = algo.horizon
        self.gamma = algo.gamma
        self.rho = algo.rho
        self.batch_size = algo.batch_size
        self.grad_clip = algo.grad_clip
        self.consistency_coef = algo.consistency_coef
        self.reward_coef = algo.reward_coef
        self.value_coef = algo.value_coef
        self.entropy_coef = algo.entropy_coef
        self.tau = algo.tau

        self._plan_mean = None

    # -- acting -----------------------------------------------------------
    @torch.no_grad()
    def act(self, obs: np.ndarray, is_first: np.ndarray, training: bool = True) -> np.ndarray:
        obs_t = to_tensor(obs, self.device)
        num_envs = obs_t.shape[0]
        z0 = self.model.encode(obs_t)

        prev = self._plan_mean
        if prev is None or prev.shape[0] != num_envs:
            prev = None
        else:  # warm-start: shift the previous plan forward one step
            prev = torch.cat([prev[:, 1:], torch.zeros_like(prev[:, :1])], dim=1)
            reset = to_tensor(is_first, self.device).reshape(num_envs, 1, 1) > 0.5
            prev = torch.where(reset, torch.zeros_like(prev), prev)

        action, mean = self.planner.plan(self.model, z0, prev, training)
        self._plan_mean = mean
        return action.cpu().numpy().astype(np.float32)

    # -- learning ---------------------------------------------------------
    def update(self, buffer) -> dict:
        seq_len = self.horizon + 1
        if not buffer.can_sample(seq_len):
            return {}
        batch = buffer.sample(self.batch_size, seq_len)
        obs, action = batch["obs"], batch["action"]
        reward, cont = batch["reward"], batch["cont"]

        with torch.no_grad():
            z_target_all = self.model.encode(obs, target=True)  # [B, H+1, L]

        z = self.model.encode(obs[:, 0])
        consistency = reward_loss = value_loss = 0.0
        latents = []
        rho = 1.0
        for t in range(self.horizon):
            a = action[:, t]
            latents.append(z.detach())

            reward_pred = self.model.reward_dist(z, a)
            q_dists = self.model.q_dists(z, a)
            z_next = self.model.next(z, a)
            z_tgt = z_target_all[:, t + 1]

            with torch.no_grad():
                a_next = self.model.pi_dist(z_tgt).mode
                q_next = self.model.q_reduced(z_tgt, a_next, target=True)
                td_target = reward[:, t] + self.gamma * cont[:, t] * q_next

            consistency = consistency + rho * F.mse_loss(z_next, z_tgt)
            reward_loss = reward_loss + rho * (-reward_pred.log_prob(reward[:, t]).mean())
            value_loss = value_loss + rho * torch.stack(
                [-qd.log_prob(td_target).mean() for qd in q_dists]
            ).mean()

            z = z_next
            rho *= self.rho

        model_loss = (
            self.consistency_coef * consistency
            + self.reward_coef * reward_loss
            + self.value_coef * value_loss
        ) / self.horizon

        self.model_opt.zero_grad(set_to_none=True)
        model_loss.backward()
        nn.utils.clip_grad_norm_(
            itertools.chain(
                self.model.encoder.parameters(), self.model.dynamics.parameters(),
                self.model.reward.parameters(), self.model.qs.parameters(),
            ),
            self.grad_clip,
        )
        self.model_opt.step()

        # -- policy improvement on the (detached) imagined latents --
        pi_loss = 0.0
        rho = 1.0
        for z_t in latents:
            dist = self.model.pi_dist(z_t)
            a, logp = dist.rsample()
            q = self.model.q_reduced(z_t, a)
            pi_loss = pi_loss + rho * (-q + self.entropy_coef * logp).mean()
            rho *= self.rho
        pi_loss = pi_loss / self.horizon

        self.pi_opt.zero_grad(set_to_none=True)
        pi_loss.backward()
        nn.utils.clip_grad_norm_(self.model.pi.parameters(), self.grad_clip)
        self.pi_opt.step()

        self.model.soft_update_target(self.tau)

        return {
            "model_loss": model_loss.detach().item(),
            "consistency": (consistency / self.horizon).detach().item(),
            "reward_loss": (reward_loss / self.horizon).detach().item(),
            "value_loss": (value_loss / self.horizon).detach().item(),
            "pi_loss": pi_loss.detach().item(),
        }

    # -- persistence ------------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "model": self.model.state_dict(),
            "model_opt": self.model_opt.state_dict(),
            "pi_opt": self.pi_opt.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.model.load_state_dict(state["model"])
        self.model_opt.load_state_dict(state["model_opt"])
        self.pi_opt.load_state_dict(state["pi_opt"])
