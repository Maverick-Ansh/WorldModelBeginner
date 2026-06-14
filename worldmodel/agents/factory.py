"""Build an agent from config: ``algo.name`` in {dreamerv3, tdmpc2}."""

from __future__ import annotations

import torch

from .base import Agent


def make_agent(cfg, obs_dim: int, act_dim: int, device: torch.device) -> Agent:
    name = cfg.algo.name
    if name == "dreamerv3":
        from .dreamer import DreamerAgent

        return DreamerAgent(obs_dim, act_dim, cfg, device)
    if name == "tdmpc2":
        from .tdmpc2 import TDMPC2Agent

        return TDMPC2Agent(obs_dim, act_dim, cfg, device)
    raise ValueError(f"Unknown algo '{name}' (expected dreamerv3|tdmpc2).")
