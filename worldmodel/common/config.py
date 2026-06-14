"""A tiny, dependency-light config system.

We deliberately avoid Hydra/OmegaConf so the project stays easy to read and tweak.
Configs are plain YAML files that are deep-merged in order, then overridden by
``key.subkey=value`` CLI arguments with automatic type inference.

Example
-------
>>> cfg = load_configs(["configs/default.yaml", "configs/algo/dreamerv3.yaml"],
...                    overrides=["seed=1", "algo.horizon=15", "env.name=toy"])
>>> cfg.algo.horizon
15
>>> cfg.seed
1
"""

from __future__ import annotations

import ast
import copy
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


class Config(Mapping):
    """Read/write nested config with both attribute and item access.

    Nested ``dict`` values are wrapped as :class:`Config` on access so that
    ``cfg.algo.horizon`` and ``cfg["algo"]["horizon"]`` are equivalent.
    """

    def __init__(self, data: Mapping[str, Any] | None = None):
        object.__setattr__(self, "_data", dict(data or {}))

    # -- mapping protocol -------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        value = self._data[key]
        if isinstance(value, dict):
            return Config(value)
        return value

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    # -- attribute access -------------------------------------------------
    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:  # pragma: no cover - mirrors normal attr error
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self._data[key] = value.to_dict() if isinstance(value, Config) else value

    def __contains__(self, key: object) -> bool:
        return key in self._data

    # -- helpers ----------------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        return self[key] if key in self._data else default

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    def update(self, other: Mapping[str, Any]) -> "Config":
        object.__setattr__(self, "_data", deep_merge(self._data, dict(other)))
        return self

    def flat(self, _prefix: str = "") -> dict[str, Any]:
        """Flatten to ``{"algo.horizon": 15, ...}`` for logging."""
        out: dict[str, Any] = {}
        for key, value in self._data.items():
            full = f"{_prefix}{key}"
            if isinstance(value, dict):
                out.update(Config(value).flat(_prefix=full + "."))
            else:
                out[full] = value
        return out

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Config({self._data!r})"


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (returns a new dict)."""
    out = copy.deepcopy(dict(base))
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _coerce(value: str) -> Any:
    """Infer the natural Python type of a CLI override string."""
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value  # leave as plain string


def _apply_override(data: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    node = data
    for key in keys[:-1]:
        node = node.setdefault(key, {})
        if not isinstance(node, dict):
            raise TypeError(f"Cannot override into non-dict at '{key}' in '{dotted_key}'")
    node[keys[-1]] = value


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_configs(
    paths: Iterable[str | Path],
    overrides: Iterable[str] | None = None,
) -> Config:
    """Deep-merge YAML files in order, then apply ``key=value`` CLI overrides."""
    merged: dict[str, Any] = {}
    for path in paths:
        merged = deep_merge(merged, load_yaml(path))
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override '{item}' must be of the form key.subkey=value")
        key, raw = item.split("=", 1)
        _apply_override(merged, key.strip(), _coerce(raw.strip()))
    return Config(merged)
