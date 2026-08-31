"""Float64 MLP — the value network both learners share."""

import torch
from torch import nn

ACTIVATIONS = {"silu": nn.SiLU, "tanh": nn.Tanh, "softplus": nn.Softplus}


def make_mlp(n_in: int, hidden: tuple[int, ...], n_out: int = 1,
             activation: str = "silu") -> nn.Sequential:
    """Plain float64 MLP. Smooth activations keep input-gradients benign."""
    if activation not in ACTIVATIONS:
        raise ValueError(f"unknown activation {activation!r}")
    act = ACTIVATIONS[activation]
    dims = [n_in, *hidden]
    layers: list[nn.Module] = []
    for d_in, d_out in zip(dims[:-1], dims[1:], strict=True):
        layers += [nn.Linear(d_in, d_out), act()]
    layers += [nn.Linear(dims[-1], n_out)]
    return nn.Sequential(*layers).to(torch.float64)
