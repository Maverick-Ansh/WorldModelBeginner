"""Evaluate a trained checkpoint with the deterministic (greedy) policy.

Usage::

    python -m worldmodel.eval --algo tdmpc2 --env toy --checkpoint logs/.../checkpoint.pt --episodes 10
"""

from __future__ import annotations

import argparse

import numpy as np

from .agents import make_agent
from .common.utils import get_device, set_seed
from .envs import make_env
from .train import build_config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate a world-model agent.")
    parser.add_argument("--algo", default="dreamerv3", choices=["dreamerv3", "tdmpc2"])
    parser.add_argument("--env", default="toy")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--config", action="append", default=[])
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args(argv)


def evaluate(agent, env, episodes: int) -> dict:
    obs = env.reset()
    is_first = np.ones(env.num_envs, dtype=bool)
    ep_return = np.zeros(env.num_envs, dtype=np.float64)
    returns: list[float] = []
    while len(returns) < episodes:
        action = agent.act(obs, is_first, training=False)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated | truncated
        ep_return += reward
        for i in np.nonzero(done)[0]:
            returns.append(float(ep_return[i]))
            ep_return[i] = 0.0
        is_first = done
    returns = returns[:episodes]
    return {"episodes": len(returns), "return_mean": float(np.mean(returns)),
            "return_std": float(np.std(returns)), "return_min": float(np.min(returns)),
            "return_max": float(np.max(returns))}


def main(argv=None) -> None:
    args = parse_args(argv)
    cfg = build_config(args)
    set_seed(cfg.seed)
    device = get_device(cfg.device)

    env = make_env(cfg.env, seed=cfg.seed + 1000)
    agent = make_agent(cfg, env.spec.obs_dim, env.spec.act_dim, device)
    agent.load(args.checkpoint)
    print(f"[eval] loaded {args.checkpoint}")

    stats = evaluate(agent, env, args.episodes)
    env.close()
    print("[eval] " + "  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                for k, v in stats.items()))


if __name__ == "__main__":
    main()
