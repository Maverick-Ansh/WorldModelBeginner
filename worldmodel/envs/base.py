"""The single environment interface every backend implements.

Design choices that keep one training loop working from a toy env up to Isaac Lab:

* **Vectorized.** Everything carries a leading ``num_envs`` dimension (1 for toy/Gym).
* **Auto-resetting.** ``step`` resets finished envs internally and returns the *next*
  episode's first observation, so the training driver never special-cases resets.
* **Normalized actions.** Agents always emit actions in ``[-1, 1]^act_dim``; each
  backend rescales to the true actuator range. This is what makes a policy portable
  from sim to your real hand -- only the rescaling changes.

Observations/rewards are returned as ``float32`` NumPy arrays; ``terminated`` and
``truncated`` are ``bool`` arrays of shape ``[num_envs]`` (Gymnasium semantics:
``terminated`` = real episode end, ``truncated`` = time limit).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np


@dataclass
class EnvSpec:
    obs_dim: int
    act_dim: int
    act_low: np.ndarray   # true low actuator bounds (informational; agent acts in [-1,1])
    act_high: np.ndarray  # true high actuator bounds
    max_episode_steps: int | None = None

    @property
    def summary(self) -> str:
        return f"obs_dim={self.obs_dim} act_dim={self.act_dim} horizon={self.max_episode_steps}"


class Env(abc.ABC):
    """Vectorized, auto-resetting environment with normalized continuous actions."""

    spec: EnvSpec
    num_envs: int

    @abc.abstractmethod
    def reset(self) -> np.ndarray:
        """Return initial observations, shape ``[num_envs, obs_dim]`` (float32)."""

    @abc.abstractmethod
    def step(self, action: np.ndarray):
        """Step every env with ``action`` in ``[-1, 1]``, shape ``[num_envs, act_dim]``.

        Returns ``(obs, reward, terminated, truncated, info)`` where finished envs have
        already been reset and ``obs`` holds the new episode's first observation.
        """

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release any backend resources."""
