"""Differential loss — the Huge & Savine (2020) construction, torch-native.

The paper's twin network implements the value network's backpropagated
gradients (their eq. 4) and supervises them against differentiated labels.
In PyTorch the twin IS ``torch.autograd.grad`` of the model output w.r.t.
the inputs, so the loss below is the faithful translation — no separate
derivative head exists or is needed.

Work in normalized space: derivative labels are pre-scaled by
``Standardizer.grad_scale`` (the role of ``lambda_j`` in the reference
implementation, which divides by the training-set RMS of the derivative
labels — same objective, O(1) scales).

Cost follows the paper's twin split:

    C = alpha * MSE(values) + beta * MSE(gradients)
    alpha = 1 / (1 + lam * n_grad_cols),  beta = 1 - alpha
"""

import torch
from torch import nn


def value_and_grad(model: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor,
                                                               torch.Tensor]:
    """Forward pass plus d(output)/dx, the twin network in one autograd call.

    ``x`` is used as the differentiable leaf (batch, n_in); the returned
    gradient keeps its graph so the loss differentiates through it
    (second-order backprop).
    """
    x = x.requires_grad_(True)
    y = model(x)
    g = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
    return y, g


def differential_loss(y_pred: torch.Tensor, g_pred: torch.Tensor,
                      y_label: torch.Tensor, g_label: torch.Tensor,
                      lam: float = 1.0) -> torch.Tensor:
    """Weighted value/gradient cost; ``lam`` trades the two terms (paper: 1)."""
    alpha = 1.0 / (1.0 + lam * g_label.shape[1])
    beta = 1.0 - alpha
    return (alpha * nn.functional.mse_loss(y_pred, y_label)
            + beta * nn.functional.mse_loss(g_pred, g_label))
