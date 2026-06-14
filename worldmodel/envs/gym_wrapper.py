"""Adapter from a single Gymnasium env (incl. MuJoCo dexterous hands) to :class:`Env`.

Handles flat ``Box`` and goal-conditioned ``Dict`` observation spaces (e.g.
Gymnasium-Robotics ``AdroitHand*`` / ``HandManipulate*``) by flattening, and rescales
normalized ``[-1, 1]`` actions to the env's true actuator range. Presented as a
``num_envs=1`` vector env so it shares the training loop with Isaac Lab.
"""

from __future__ import annotations

import numpy as np

from .base import Env, EnvSpec


class GymEnv(Env):
    def __init__(self, task: str, seed: int = 0, **make_kwargs):
        import gymnasium as gym

        try:  # registers AdroitHand*/HandManipulate* tasks if installed
            import gymnasium_robotics  # noqa: F401

            gym.register_envs(gymnasium_robotics)
        except Exception:
            pass

        self.num_envs = 1
        self._env = gym.make(task, **make_kwargs)
        self._gym = gym
        self._seed = seed

        obs_space = self._env.observation_space
        act_space = self._env.action_space
        if not isinstance(act_space, gym.spaces.Box):
            raise TypeError("Only continuous (Box) action spaces are supported.")

        self._obs_dim = int(gym.spaces.flatdim(obs_space))
        self._obs_space = obs_space
        self._act_low = act_space.low.astype(np.float32)
        self._act_high = act_space.high.astype(np.float32)
        self.spec = EnvSpec(
            obs_dim=self._obs_dim,
            act_dim=int(act_space.shape[0]),
            act_low=self._act_low,
            act_high=self._act_high,
            max_episode_steps=getattr(self._env.spec, "max_episode_steps", None),
        )

    def _flat_obs(self, obs) -> np.ndarray:
        flat = self._gym.spaces.flatten(self._obs_space, obs).astype(np.float32)
        return flat[None]  # add num_envs dimension

    def _scale_action(self, action: np.ndarray) -> np.ndarray:
        a = np.clip(action[0], -1.0, 1.0)  # drop num_envs dim
        return self._act_low + (a + 1.0) * 0.5 * (self._act_high - self._act_low)

    def reset(self) -> np.ndarray:
        obs, _ = self._env.reset(seed=self._seed)
        self._seed += 1
        return self._flat_obs(obs)

    def step(self, action: np.ndarray):
        obs, reward, terminated, truncated, info = self._env.step(self._scale_action(action))
        if terminated or truncated:
            obs, _ = self._env.reset(seed=self._seed)
            self._seed += 1
        return (
            self._flat_obs(obs),
            np.array([reward], dtype=np.float32),
            np.array([terminated], dtype=bool),
            np.array([truncated], dtype=bool),
            info,
        )

    def close(self) -> None:
        self._env.close()
