"""Common agent interface for the unified training loop."""

from __future__ import annotations

import abc

import numpy as np
import torch


class Agent(abc.ABC):
    device: torch.device

    @abc.abstractmethod
    def act(self, obs: np.ndarray, is_first: np.ndarray, training: bool = True) -> np.ndarray:
        """Map observations ``[num_envs, obs_dim]`` to actions ``[num_envs, act_dim]`` in [-1,1].

        ``is_first`` (``[num_envs]`` bool) marks envs that just reset, so stateful agents
        (DreamerV3) can reset the corresponding recurrent state.
        """

    @abc.abstractmethod
    def update(self, buffer) -> dict:
        """Run one gradient update by sampling from ``buffer``; return scalar metrics."""

    @abc.abstractmethod
    def state_dict(self) -> dict: ...

    @abc.abstractmethod
    def load_state_dict(self, state: dict) -> None: ...

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path, map_location=self.device))
