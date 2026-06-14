"""ToyHandEnv: a dependency-free, vectorized proxy for dexterous-hand control.

Each "hand" has ``num_joints`` torque-controlled joints (a damped double integrator)
that must be driven to a random target configuration. The observation is
``[q, qdot, goal]`` and the action is a per-joint torque in ``[-1, 1]``.

It deliberately mirrors the *shape* of the real problem -- proprioceptive state in,
joint commands out, dense shaped reward -- so DreamerV3 and TD-MPC2 can be exercised
end-to-end on CPU with zero external dependencies (used by the unit tests and the
quickstart). Swap it for ``isaaclab`` / a Gym hand by changing one config line.
"""

from __future__ import annotations

import numpy as np

from .base import Env, EnvSpec


class ToyHandEnv(Env):
    def __init__(self, num_envs: int = 1, num_joints: int = 6, episode_steps: int = 100,
                 dt: float = 0.05, damping: float = 0.1, torque_scale: float = 2.0,
                 q_limit: float = 1.0, v_limit: float = 5.0, reach_sigma: float = 0.3,
                 ctrl_cost: float = 0.01, seed: int = 0):
        self.num_envs = num_envs
        self.d = num_joints
        self.episode_steps = episode_steps
        self.dt = dt
        self.damping = damping
        self.torque_scale = torque_scale
        self.q_limit = q_limit
        self.v_limit = v_limit
        self.reach_sigma = reach_sigma
        self.ctrl_cost = ctrl_cost
        self._rng = np.random.default_rng(seed)

        self.spec = EnvSpec(
            obs_dim=3 * self.d,
            act_dim=self.d,
            act_low=-np.ones(self.d, dtype=np.float32),
            act_high=np.ones(self.d, dtype=np.float32),
            max_episode_steps=episode_steps,
        )
        self.q = np.zeros((num_envs, self.d), dtype=np.float32)
        self.qdot = np.zeros((num_envs, self.d), dtype=np.float32)
        self.goal = np.zeros((num_envs, self.d), dtype=np.float32)
        self.t = np.zeros((num_envs,), dtype=np.int64)

    def _obs(self) -> np.ndarray:
        return np.concatenate([self.q, self.qdot, self.goal], axis=-1).astype(np.float32)

    def _reset_envs(self, mask: np.ndarray) -> None:
        n = int(mask.sum())
        if n == 0:
            return
        self.q[mask] = 0.0
        self.qdot[mask] = 0.0
        self.goal[mask] = self._rng.uniform(-self.q_limit, self.q_limit, size=(n, self.d))
        self.t[mask] = 0

    def reset(self) -> np.ndarray:
        self._reset_envs(np.ones(self.num_envs, dtype=bool))
        return self._obs()

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        torque = action * self.torque_scale
        self.qdot = np.clip(
            self.qdot + self.dt * (torque - self.damping * self.qdot),
            -self.v_limit, self.v_limit,
        )
        self.q = np.clip(self.q + self.dt * self.qdot, -self.q_limit, self.q_limit)

        dist2 = np.sum((self.q - self.goal) ** 2, axis=-1)
        reach = np.exp(-dist2 / (2 * self.reach_sigma ** 2))
        reward = (reach - self.ctrl_cost * np.sum(action ** 2, axis=-1)).astype(np.float32)

        self.t += 1
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = self.t >= self.episode_steps

        # Auto-reset finished envs, returning the new episode's first observation.
        self._reset_envs(terminated | truncated)
        return self._obs(), reward, terminated, truncated, {}
