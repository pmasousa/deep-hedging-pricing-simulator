"""Accuracy and Greeks-quality evaluation for trained learners — SP4.

All metrics are in RAW units (dollars of price, delta per dollar of spot)
so they read on business terms. ``predict_raw`` is the single door from a
trained ``TrainResult`` back to market coordinates; every evaluator here
and the dashboard model page go through it.
"""

import torch

from dhps.pricing.black_scholes import bs_greeks, bs_price
from dhps.train.trainer import TrainResult

# spots outside the training range [50, 200] — extrapolation stress
OOD_SPOTS = (25.0, 35.0, 45.0, 210.0, 225.0, 240.0)


def predict_raw(result: TrainResult, x_raw: torch.Tensor,
                with_grad: bool = True) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Model prediction in market coordinates.

    Returns ``(price_raw, grad_raw)`` where grad columns are d(price)/d(feature)
    in FEATURES order; gamma is not provided here (second order — see
    ``greeks_curve``).
    """
    model, scaler = result.model, result.scaler
    x = scaler.transform_x(x_raw)
    if not with_grad:
        with torch.no_grad():
            y = model(x)
        return y * scaler.y_std[0] + scaler.y_mean[0], None
    x = x.requires_grad_(True)
    y = model(x)
    g = torch.autograd.grad(y.sum(), x)[0]
    # detached: callers want values, not graphs
    price = (y * scaler.y_std[0] + scaler.y_mean[0]).detach()
    grad = (g / scaler.grad_scale[0]).detach()
    return price, grad


def evaluate_learner(result: TrainResult,
                     data: dict[str, torch.Tensor]) -> dict[str, float]:
    """ID metrics on the val split: price MAE and delta MAE, raw units."""
    price, grad = predict_raw(result, data["x_val"])
    true_price = data["y_val"][:, 0]
    true_delta = data["g_val"][:, 0]
    return {"price_mae": float((price[:, 0] - true_price).abs().mean()),
            "delta_mae": float((grad[:, 0] - true_delta).abs().mean())}


def ood_metrics(result: TrainResult, strike: float = 100.0, t_maturity: float = 1.0,
                sigma: float = 0.2, r: float = 0.05, q: float = 0.01) -> dict[str, float]:
    """Price/delta MAE on spots outside the training range (extrapolation)."""
    spots = torch.tensor(OOD_SPOTS, dtype=torch.float64)
    x = torch.stack([spots, torch.full_like(spots, strike),
                     torch.full_like(spots, t_maturity),
                     torch.full_like(spots, sigma)], dim=1)
    price, grad = predict_raw(result, x)
    true_price = bs_price(spots, strike, r, q, sigma, t_maturity)
    true_delta = bs_greeks(spots, strike, r, q, sigma, t_maturity)["delta"]
    return {"ood_price_mae": float((price[:, 0] - true_price).abs().mean()),
            "ood_delta_mae": float((grad[:, 0] - true_delta).abs().mean())}


def greeks_curve(result: TrainResult, spots: torch.Tensor, strike: float,
                 t_maturity: float, sigma: float, r: float = 0.05,
                 q: float = 0.01) -> dict[str, torch.Tensor]:
    """Model delta and gamma along a spot grid, against the analytic curves.

    Gamma comes from a second autograd pass through the trained model — it
    was never in the labels, so it measures what the network internalized.
    """
    x = torch.stack([spots, torch.full_like(spots, strike),
                     torch.full_like(spots, t_maturity),
                     torch.full_like(spots, sigma)], dim=1)
    scaler = result.scaler
    z = scaler.transform_x(x).requires_grad_(True)
    y = result.model(z)
    g = torch.autograd.grad(y.sum(), z, create_graph=True)[0]
    hess = torch.autograd.grad(g[:, 0].sum(), z)[0]
    # un-normalize: delta = g/gs, gamma = hess * y_std / x_std^2 (chain rule)
    gs = scaler.grad_scale[0]
    delta = (g[:, 0] / gs[0]).detach()
    gamma = (hess[:, 0] * scaler.y_std[0] / scaler.x_std[0] ** 2).detach()
    true = bs_greeks(spots, strike, r, q, sigma, t_maturity)
    return {"spot": spots, "delta": delta, "gamma": gamma,
            "delta_true": true["delta"], "gamma_true": true["gamma"]}
