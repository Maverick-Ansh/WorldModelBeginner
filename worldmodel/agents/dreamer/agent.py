"""DreamerV3 agent: world-model learning + imagination behavior + recurrent acting."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ...common.utils import to_tensor
from ..base import Agent
from .behavior import Behavior
from .world_model import WorldModel


class DreamerAgent(Agent):
    def __init__(self, obs_dim: int, act_dim: int, cfg, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        algo = cfg.algo
        self.wm = WorldModel(obs_dim, act_dim, algo).to(device)
        self.behavior = Behavior(self.wm.rssm.feat_dim, act_dim, algo).to(device)

        self.wm_opt = torch.optim.Adam(self.wm.parameters(), lr=algo.wm_lr, eps=1e-8)
        self.actor_opt = torch.optim.Adam(self.behavior.actor.parameters(), lr=algo.actor_lr)
        self.critic_opt = torch.optim.Adam(self.behavior.critic.parameters(), lr=algo.critic_lr)

        self.batch_size = algo.batch_size
        self.seq_len = algo.seq_len
        self.horizon = algo.horizon
        self.grad_clip = algo.grad_clip

        self._state = None        # RSSM recurrent state, per env
        self._prev_action = None  # last action taken, per env

    # -- acting -----------------------------------------------------------
    def _ensure_state(self, num_envs: int) -> None:
        if self._state is None or self._state["deter"].shape[0] != num_envs:
            self._state = self.wm.rssm.initial(num_envs, self.device)
            self._prev_action = torch.zeros(num_envs, self.act_dim, device=self.device)

    @torch.no_grad()
    def act(self, obs: np.ndarray, is_first: np.ndarray, training: bool = True) -> np.ndarray:
        obs_t = to_tensor(obs, self.device)
        num_envs = obs_t.shape[0]
        self._ensure_state(num_envs)
        is_first_t = to_tensor(is_first, self.device).reshape(num_envs)

        embed = self.wm.encoder(obs_t)
        self._state, _, _ = self.wm.rssm.obs_step(
            self._state, self._prev_action, embed, is_first_t
        )
        feat = self.wm.rssm.get_feat(self._state)
        dist = self.behavior.actor.dist(feat)
        action = dist.rsample()[0] if training else dist.mode
        self._prev_action = action
        return action.cpu().numpy().astype(np.float32)

    # -- learning ---------------------------------------------------------
    def _clip(self, module: nn.Module) -> None:
        nn.utils.clip_grad_norm_(module.parameters(), self.grad_clip)

    def update(self, buffer) -> dict:
        if not buffer.can_sample(self.seq_len):
            return {}
        batch = buffer.sample(self.batch_size, self.seq_len)

        wm_loss, start, metrics = self.wm.loss(batch)
        self.wm_opt.zero_grad(set_to_none=True)
        wm_loss.backward()
        self._clip(self.wm)
        self.wm_opt.step()

        # Use every observed latent as an independent imagination start state.
        start = {k: v.reshape(-1, *v.shape[2:]) for k, v in start.items()}
        actor_loss, critic_loss, beh_metrics = self.behavior.learn(self.wm, start, self.horizon)
        self.actor_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        (actor_loss + critic_loss).backward()
        self._clip(self.behavior.actor)
        self._clip(self.behavior.critic)
        self.actor_opt.step()
        self.critic_opt.step()
        self.behavior.update_slow()

        metrics.update(beh_metrics)
        return {k: float(v) for k, v in metrics.items()}

    # -- persistence ------------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "wm": self.wm.state_dict(),
            "behavior": self.behavior.state_dict(),
            "wm_opt": self.wm_opt.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.wm.load_state_dict(state["wm"])
        self.behavior.load_state_dict(state["behavior"])
        self.wm_opt.load_state_dict(state["wm_opt"])
        self.actor_opt.load_state_dict(state["actor_opt"])
        self.critic_opt.load_state_dict(state["critic_opt"])
