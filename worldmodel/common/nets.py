"""Reusable network components (MLPs, encoders/decoders, SimNorm, LayerNorm-GRU)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .distributions import symlog

ACTIVATIONS = {"silu": nn.SiLU, "relu": nn.ReLU, "elu": nn.ELU, "tanh": nn.Tanh}


def activation(name: str) -> nn.Module:
    return ACTIVATIONS[name]()


def build_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    num_layers: int = 2,
    act: str = "silu",
    norm: bool = True,
    out_act: bool = False,
) -> nn.Sequential:
    """A LayerNorm + activation MLP (the DreamerV3 / TD-MPC2 default block)."""
    layers: list[nn.Module] = []
    last = in_dim
    for _ in range(num_layers):
        layers.append(nn.Linear(last, hidden_dim))
        if norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(activation(act))
        last = hidden_dim
    layers.append(nn.Linear(last, out_dim))
    if out_act:
        if norm:
            layers.append(nn.LayerNorm(out_dim))
        layers.append(activation(act))
    return nn.Sequential(*layers)


@torch.no_grad()
def zero_init_(module: nn.Module) -> None:
    """Zero the parameters of the final Linear in a head (stabilises early training)."""
    last_linear = None
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            last_linear = layer
    if last_linear is not None:
        last_linear.weight.zero_()
        if last_linear.bias is not None:
            last_linear.bias.zero_()


class SimNorm(nn.Module):
    """Simplicial normalization (TD-MPC2): softmax over fixed-size latent groups.

    Projecting the latent onto a product of simplices keeps it bounded and sparse,
    which is a big part of why TD-MPC2 trains stably with a single set of
    hyper-parameters across very different tasks.
    """

    def __init__(self, group: int = 8):
        super().__init__()
        self.group = group

    def forward(self, x: Tensor) -> Tensor:
        shape = x.shape
        x = x.view(*shape[:-1], -1, self.group)
        x = F.softmax(x, dim=-1)
        return x.reshape(*shape)

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return f"group={self.group}"


class VectorEncoder(nn.Module):
    """Encode a (symlog-compressed) state vector into a feature embedding."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2,
                 act: str = "silu", norm: bool = True):
        super().__init__()
        self.net = build_mlp(in_dim, hidden_dim, out_dim, num_layers, act, norm, out_act=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(symlog(x))


class VectorDecoder(nn.Module):
    """Reconstruct a state vector from features (Gaussian-in-symlog-space head)."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2,
                 act: str = "silu", norm: bool = True):
        super().__init__()
        self.net = build_mlp(in_dim, hidden_dim, out_dim, num_layers, act, norm)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class LayerNormGRUCell(nn.Module):
    """GRU cell with layer normalization on the gate pre-activations.

    DreamerV3 uses a normalized recurrent cell for the deterministic part of the
    RSSM; the LayerNorm makes the recurrence robust to input scale.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(input_size + hidden_size, 3 * hidden_size, bias=False)
        self.norm = nn.LayerNorm(3 * hidden_size)

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        parts = self.norm(self.linear(torch.cat([x, h], dim=-1)))
        reset, cand, update = parts.chunk(3, dim=-1)
        reset = torch.sigmoid(reset)
        cand = torch.tanh(reset * cand)
        update = torch.sigmoid(update - 1.0)  # bias toward remembering, as in Dreamer
        return update * cand + (1.0 - update) * h
