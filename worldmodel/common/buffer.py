"""Vectorized sequence replay buffer shared by both agents.

World models train on short *trajectories*, not independent transitions: DreamerV3
unrolls the RSSM over a window, and TD-MPC2 rolls its latent dynamics over a planning
horizon. This buffer therefore samples contiguous sub-sequences along the time axis.

It is vectorized over ``num_envs`` parallel environments (the Isaac Lab use case):
storage is ``[capacity, num_envs, ...]`` and all envs advance the shared time cursor
together, so each env keeps its own clean temporal stream. ``is_first`` flags mark
episode boundaries inside a window so the model can reset recurrent state where needed
(windows may legitimately span a boundary, exactly as in DreamerV3). With ``num_envs=1``
this degenerates to an ordinary single-stream replay.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


class SequenceReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int, device: torch.device,
                 num_envs: int = 1, seed: int = 0):
        # capacity is the per-env number of timesteps stored.
        self.capacity = max(1, int(capacity) // max(1, num_envs))
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = device
        self._rng = np.random.default_rng(seed)

        shape = (self.capacity, num_envs)
        self.obs = np.zeros((*shape, obs_dim), dtype=np.float32)
        self.action = np.zeros((*shape, act_dim), dtype=np.float32)
        self.reward = np.zeros(shape, dtype=np.float32)
        self.cont = np.zeros(shape, dtype=np.float32)  # 1 - terminal
        self.is_first = np.zeros(shape, dtype=np.float32)

        self._idx = 0
        self._full = False

    def __len__(self) -> int:
        """Total number of stored transitions across all envs."""
        steps = self.capacity if self._full else self._idx
        return steps * self.num_envs

    @property
    def steps_per_env(self) -> int:
        return self.capacity if self._full else self._idx

    def add(self, obs, action, reward, cont, is_first) -> None:
        """Append one timestep for every env. Leading dim of each arg is ``num_envs``."""
        i = self._idx
        self.obs[i] = obs
        self.action[i] = action
        self.reward[i] = np.reshape(reward, (self.num_envs,))
        self.cont[i] = np.reshape(cont, (self.num_envs,))
        self.is_first[i] = np.reshape(is_first, (self.num_envs,))
        self._idx = (i + 1) % self.capacity
        self._full = self._full or self._idx == 0

    def can_sample(self, seq_len: int) -> bool:
        return self.steps_per_env >= seq_len

    def sample(self, batch_size: int, seq_len: int) -> dict[str, Tensor]:
        """Return a batch of contiguous sequences, each tensor shaped ``[B, T, ...]``."""
        n = self.steps_per_env
        if n < seq_len:
            raise ValueError(f"Need {seq_len} steps to sample, buffer holds {n} per env.")
        base = self._idx if self._full else 0  # logical 0 == oldest valid step
        starts = self._rng.integers(0, n - seq_len + 1, size=batch_size)
        envs = self._rng.integers(0, self.num_envs, size=batch_size)
        offsets = np.arange(seq_len)
        time = (base + starts[:, None] + offsets[None, :]) % self.capacity  # [B, T]
        env_col = envs[:, None]  # [B, 1]

        def gather(arr):
            return torch.as_tensor(arr[time, env_col], device=self.device)

        return {
            "obs": gather(self.obs),
            "action": gather(self.action),
            "reward": gather(self.reward),
            "cont": gather(self.cont),
            "is_first": gather(self.is_first),
        }

    # -- persistence ------------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "obs": self.obs, "action": self.action, "reward": self.reward,
            "cont": self.cont, "is_first": self.is_first,
            "idx": self._idx, "full": self._full,
        }

    def load_state_dict(self, state: dict) -> None:
        for key in ("obs", "action", "reward", "cont", "is_first"):
            getattr(self, key)[:] = state[key]
        self._idx = state["idx"]
        self._full = state["full"]
