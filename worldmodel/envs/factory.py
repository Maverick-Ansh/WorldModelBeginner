"""Build an :class:`Env` from config: ``backend`` in {toy, gym, isaaclab}."""

from __future__ import annotations

from .base import Env


def make_env(env_cfg, seed: int = 0) -> Env:
    backend = env_cfg.get("backend", "toy")
    num_envs = int(env_cfg.get("num_envs", 1))

    if backend == "toy":
        from .toy import ToyHandEnv

        return ToyHandEnv(
            num_envs=num_envs,
            num_joints=int(env_cfg.get("num_joints", 6)),
            episode_steps=int(env_cfg.get("episode_steps", 100)),
            seed=seed,
        )

    if backend == "gym":
        from .gym_wrapper import GymEnv

        if num_envs != 1:
            raise ValueError("The Gym backend is single-env; set env.num_envs=1.")
        return GymEnv(task=env_cfg["task"], seed=seed, **dict(env_cfg.get("make_kwargs", {})))

    if backend == "isaaclab":
        from .isaaclab import IsaacLabEnv

        return IsaacLabEnv(
            task=env_cfg["task"],
            num_envs=num_envs,
            device=env_cfg.get("device", "cuda"),
            headless=bool(env_cfg.get("headless", True)),
            seed=seed,
        )

    raise ValueError(f"Unknown env backend '{backend}' (expected toy|gym|isaaclab).")
