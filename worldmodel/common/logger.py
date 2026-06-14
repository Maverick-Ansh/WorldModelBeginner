"""Lightweight metric logging: console + TensorBoard, with mean aggregation."""

from __future__ import annotations

import collections
import time
from pathlib import Path
from typing import Mapping

import numpy as np


class Logger:
    def __init__(self, logdir: str | Path, use_tensorboard: bool = True):
        self.logdir = Path(logdir)
        self.logdir.mkdir(parents=True, exist_ok=True)
        self._metrics: dict[str, list[float]] = collections.defaultdict(list)
        self._start = time.time()
        self._writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._writer = SummaryWriter(log_dir=str(self.logdir))
            except Exception as exc:  # pragma: no cover - optional dependency
                print(f"[logger] TensorBoard unavailable ({exc}); console only.")

    def add(self, metrics: Mapping[str, float], prefix: str = "") -> None:
        for key, value in metrics.items():
            if value is None:
                continue
            self._metrics[f"{prefix}{key}"].append(float(value))

    def scalar(self, key: str, value: float, step: int) -> None:
        if self._writer is not None:
            self._writer.add_scalar(key, value, step)

    def write(self, step: int, prefix: str = "") -> dict[str, float]:
        """Flush aggregated (mean) metrics to TensorBoard and the console."""
        out: dict[str, float] = {}
        for key, values in self._metrics.items():
            mean = float(np.mean(values))
            out[key] = mean
            self.scalar(key, mean, step)
        self._metrics.clear()
        if out:
            elapsed = time.time() - self._start
            summary = "  ".join(f"{k}={v:.3g}" for k, v in sorted(out.items()))
            print(f"[{prefix}step {step:>8d} | {elapsed:6.0f}s] {summary}")
        return out

    def close(self) -> None:
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
