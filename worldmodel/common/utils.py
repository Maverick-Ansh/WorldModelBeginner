"""Small, dependency-light training utilities."""

from __future__ import annotations

import random
from typing import Iterable

import numpy as np
import torch
from torch import Tensor, nn


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and Torch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(prefer: str = "auto") -> torch.device:
    """Resolve a torch device string, falling back to CPU when CUDA is absent."""
    if prefer not in ("auto", "cuda", "cpu"):
        return torch.device(prefer)
    if prefer == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


@torch.no_grad()
def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """Polyak averaging: ``target = (1 - tau) * target + tau * source``."""
    for tgt, src in zip(target.parameters(), source.parameters()):
        tgt.mul_(1.0 - tau).add_(src, alpha=tau)


@torch.no_grad()
def hard_update(target: nn.Module, source: nn.Module) -> None:
    target.load_state_dict(source.state_dict())


def to_tensor(x, device: torch.device, dtype=torch.float32) -> Tensor:
    if isinstance(x, Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(x), device=device, dtype=dtype)


class Moments(nn.Module):
    """EMA of robust return percentiles, for DreamerV3-style return normalization.

    Tracks exponential moving averages of the low/high percentiles of returns and
    exposes an ``offset`` and ``scale`` so the actor sees returns on a stable, unit
    scale regardless of the task's reward magnitude.
    """

    def __init__(self, decay: float = 0.99, low: float = 0.05, high: float = 0.95):
        super().__init__()
        self.decay = decay
        self.low_q = low
        self.high_q = high
        self.register_buffer("low", torch.zeros(()))
        self.register_buffer("high", torch.zeros(()))

    @torch.no_grad()
    def update(self, x: Tensor) -> None:
        flat = x.detach().flatten().float()
        low = torch.quantile(flat, self.low_q)
        high = torch.quantile(flat, self.high_q)
        self.low.mul_(self.decay).add_(low, alpha=1 - self.decay)
        self.high.mul_(self.decay).add_(high, alpha=1 - self.decay)

    def offset(self) -> Tensor:
        return self.low

    def scale(self) -> Tensor:
        # Never divide by < 1 so small-return tasks are not blown up.
        return torch.clamp(self.high - self.low, min=1.0)


def linear_schedule(step: int, start: float, end: float, duration: int) -> float:
    """Linearly interpolate from ``start`` to ``end`` over ``duration`` steps."""
    if duration <= 0:
        return end
    frac = min(max(step / duration, 0.0), 1.0)
    return start + frac * (end - start)


def every(step: int, period: int) -> bool:
    """True once per ``period`` environment steps (period <= 0 disables)."""
    return period > 0 and step % period == 0


def chunks(seq, size: int) -> Iterable:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]
