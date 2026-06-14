"""Adapter from an NVIDIA Isaac Lab task to :class:`Env`.

Isaac Lab runs thousands of GPU-parallel environments and is the recommended path to
real hardware (its sim-to-real tooling is first-class). This wrapper launches the
Omniverse app, builds a registered task, and maps its torch/GPU API onto our NumPy
:class:`Env` contract. Isaac Lab already auto-resets finished envs inside ``step``,
which matches our convention exactly.

Requires a CUDA GPU and a working Isaac Sim / Isaac Lab install -- see
``docs/sim2real.md``. The heavy imports happen lazily inside ``__init__`` so the rest
of the package (and the test suite) imports fine on machines without Isaac Lab.

Note: for maximum throughput you would keep tensors on the GPU end-to-end; this first
version round-trips through NumPy for a single, uniform code path. See docs for the
zero-copy optimization.
"""

from __future__ import annotations

import numpy as np

from .base import Env, EnvSpec


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


class IsaacLabEnv(Env):
    def __init__(self, task: str, num_envs: int = 4096, device: str = "cuda",
                 headless: bool = True, seed: int = 0, **cfg_overrides):
        # 1) Launch the simulator app BEFORE importing any isaaclab.* modules.
        from isaaclab.app import AppLauncher

        self._app_launcher = AppLauncher(headless=headless, device=device)
        self._sim_app = self._app_launcher.app

        # 2) Now the task framework can be imported and the env constructed.
        import gymnasium as gym
        import isaaclab_tasks  # noqa: F401  (registers tasks)
        from isaaclab_tasks.utils import parse_env_cfg

        env_cfg = parse_env_cfg(task, device=device, num_envs=num_envs)
        for key, value in cfg_overrides.items():
            setattr(env_cfg, key, value)
        self._env = gym.make(task, cfg=env_cfg)
        self._torch = __import__("torch")
        self._device = device

        unwrapped = self._env.unwrapped
        self.num_envs = int(getattr(unwrapped, "num_envs", num_envs))

        act_space = self._env.action_space
        act_dim = int(np.prod(act_space.shape) // self.num_envs) if act_space.shape else 0
        low = np.broadcast_to(np.asarray(act_space.low, dtype=np.float32), (act_dim,)).copy()
        high = np.broadcast_to(np.asarray(act_space.high, dtype=np.float32), (act_dim,)).copy()
        self._act_low, self._act_high = low, high
        self._finite = np.isfinite(low) & np.isfinite(high)

        obs = self.reset()
        self.spec = EnvSpec(
            obs_dim=obs.shape[-1],
            act_dim=act_dim,
            act_low=low,
            act_high=high,
            max_episode_steps=getattr(unwrapped, "max_episode_length", None),
        )

    def _extract_obs(self, obs_dict) -> np.ndarray:
        obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict
        return _to_numpy(obs).astype(np.float32).reshape(self.num_envs, -1)

    def _scale_action(self, action: np.ndarray):
        a = np.clip(action, -1.0, 1.0)
        scaled = self._act_low + (a + 1.0) * 0.5 * (self._act_high - self._act_low)
        a = np.where(self._finite, scaled, a)  # pass through unbounded actuators
        return self._torch.as_tensor(a, dtype=self._torch.float32, device=self._device)

    def reset(self) -> np.ndarray:
        obs_dict, _ = self._env.reset()
        return self._extract_obs(obs_dict)

    def step(self, action: np.ndarray):
        obs_dict, reward, terminated, truncated, info = self._env.step(self._scale_action(action))
        return (
            self._extract_obs(obs_dict),
            _to_numpy(reward).astype(np.float32).reshape(self.num_envs),
            _to_numpy(terminated).astype(bool).reshape(self.num_envs),
            _to_numpy(truncated).astype(bool).reshape(self.num_envs),
            info if isinstance(info, dict) else {},
        )

    def close(self) -> None:
        try:
            self._env.close()
        finally:
            self._sim_app.close()
