"""Unified training entry point for both agents and all env backends.

Usage::

    python -m worldmodel.train --algo dreamerv3 --env toy
    python -m worldmodel.train --algo tdmpc2 --env toy seed=1 algo.horizon=3
    python -m worldmodel.train --algo dreamerv3 --env isaaclab env.task=Isaac-Repose-Cube-Shadow-Direct-v0

Any ``key.subkey=value`` token overrides the merged YAML config.
"""

from __future__ import annotations

import argparse
import collections
import time
from pathlib import Path

import numpy as np

from .agents import make_agent
from .common.config import load_configs
from .common.buffer import SequenceReplayBuffer
from .common.logger import Logger
from .common.utils import get_device, set_seed
from .envs import make_env

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train a world-model agent.")
    parser.add_argument("--algo", default="dreamerv3", choices=["dreamerv3", "tdmpc2"])
    parser.add_argument("--env", default="toy", help="env config name under configs/env/")
    parser.add_argument("--config", action="append", default=[], help="extra YAML to merge")
    parser.add_argument("overrides", nargs="*", help="key.subkey=value overrides")
    return parser.parse_args(argv)


def build_config(args):
    paths = [
        CONFIG_DIR / "default.yaml",
        CONFIG_DIR / "algo" / f"{args.algo}.yaml",
        CONFIG_DIR / "env" / f"{args.env}.yaml",
        *[Path(p) for p in args.config],
    ]
    return load_configs(paths, args.overrides)


def main(argv=None) -> None:
    args = parse_args(argv)
    cfg = build_config(args)
    set_seed(cfg.seed)
    device = get_device(cfg.device)

    env = make_env(cfg.env, seed=cfg.seed)
    obs_dim, act_dim = env.spec.obs_dim, env.spec.act_dim
    print(f"[train] algo={cfg.algo.name} env={cfg.env.backend} {env.spec.summary} "
          f"num_envs={env.num_envs} device={device}")

    agent = make_agent(cfg, obs_dim, act_dim, device)
    buffer = SequenceReplayBuffer(cfg.replay_capacity, obs_dim, act_dim, device,
                                  num_envs=env.num_envs, seed=cfg.seed)

    run_name = f"{cfg.algo.name}-{cfg.env.backend}-{int(time.time())}"
    logger = Logger(Path(cfg.logdir) / run_name)
    ckpt_dir = Path(cfg.logdir) / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    obs = env.reset()
    is_first = np.ones(env.num_envs, dtype=bool)
    ep_return = np.zeros(env.num_envs, dtype=np.float64)
    ep_len = np.zeros(env.num_envs, dtype=np.int64)
    returns = collections.deque(maxlen=100)
    lengths = collections.deque(maxlen=100)

    global_step = 0
    it = 0
    rng = np.random.default_rng(cfg.seed)
    while global_step < cfg.total_steps:
        if global_step < cfg.prefill_steps:
            action = rng.uniform(-1.0, 1.0, size=(env.num_envs, act_dim)).astype(np.float32)
        else:
            action = agent.act(obs, is_first, training=True)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated | truncated
        buffer.add(obs, action, reward, 1.0 - terminated.astype(np.float32), is_first)

        ep_return += reward
        ep_len += 1
        for i in np.nonzero(done)[0]:
            returns.append(ep_return[i])
            lengths.append(ep_len[i])
            ep_return[i] = 0.0
            ep_len[i] = 0

        obs = next_obs
        is_first = done
        global_step += env.num_envs
        it += 1

        if global_step >= cfg.prefill_steps and it % cfg.train_every == 0:
            for _ in range(cfg.updates_per_train):
                metrics = agent.update(buffer)
                if metrics:
                    logger.add(metrics, prefix="train/")

        if it % cfg.log_every == 0:
            if returns:
                logger.add({"return": float(np.mean(returns)),
                            "ep_len": float(np.mean(lengths))}, prefix="rollout/")
            logger.scalar("rollout/buffer_steps", len(buffer), global_step)
            logger.write(global_step)

        if cfg.checkpoint_every and it % cfg.checkpoint_every == 0:
            agent.save(str(ckpt_dir / "checkpoint.pt"))

    agent.save(str(ckpt_dir / "checkpoint.pt"))
    logger.write(global_step)
    logger.close()
    env.close()
    print(f"[train] done. checkpoint: {ckpt_dir / 'checkpoint.pt'}")


if __name__ == "__main__":
    main()
