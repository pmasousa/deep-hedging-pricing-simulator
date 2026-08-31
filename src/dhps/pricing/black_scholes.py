"""Closed-form Black-Scholes prices and Greeks — the SP1 reference pricer.

Torch-native (float64) so the same call sites later swap in the DML core;
Greeks here are the analytic ground truth the DML Greeks must hit. Every
parameter accepts a float or a broadcastable tensor: the dataset builder
prices (n,) parameter batches in a single call.
"""

import math

import torch

TensorOrFloat = float | torch.Tensor


def _f64(x: TensorOrFloat) -> torch.Tensor:
    """No-copy float64 tensor view of a float or tensor."""
    return torch.as_tensor(x, dtype=torch.float64)


def _d1_d2(s: TensorOrFloat, strike: TensorOrFloat, r: TensorOrFloat,
           q: TensorOrFloat, sigma: TensorOrFloat,
           t_maturity: TensorOrFloat) -> tuple[torch.Tensor, torch.Tensor]:
    s, strike = _f64(s), _f64(strike)
    r, q, sig, t = _f64(r), _f64(q), _f64(sigma), _f64(t_maturity)
    vol_t = sig * t.sqrt()
    d1 = (torch.log(s / strike) + (r - q + 0.5 * sig * sig) * t) / vol_t
    return d1, d1 - vol_t


def _norm_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(-0.5 * x**2) / math.sqrt(2.0 * math.pi)


def bs_price(
    s: TensorOrFloat,
    strike: TensorOrFloat,
    r: TensorOrFloat = 0.05,
    q: TensorOrFloat = 0.0,
    sigma: TensorOrFloat = 0.2,
    t_maturity: TensorOrFloat = 1.0,
    call: bool = True,
) -> torch.Tensor:
    """Black-Scholes price for spot ``s`` (q = continuous dividend yield)."""
    t, sig = _f64(t_maturity), _f64(sigma)
    if bool((t <= 0).any()) or bool((sig <= 0).any()):
        raise ValueError("require sigma > 0, t_maturity > 0")
    d1, d2 = _d1_d2(s, strike, r, q, sigma, t_maturity)
    df_r = torch.exp(-_f64(r) * t)
    df_q = torch.exp(-_f64(q) * t)
    if call:
        return _f64(s) * df_q * _norm_cdf(d1) - _f64(strike) * df_r * _norm_cdf(d2)
    return _f64(strike) * df_r * _norm_cdf(-d2) - _f64(s) * df_q * _norm_cdf(-d1)


def bs_greeks(
    s: TensorOrFloat,
    strike: TensorOrFloat,
    r: TensorOrFloat = 0.05,
    q: TensorOrFloat = 0.0,
    sigma: TensorOrFloat = 0.2,
    t_maturity: TensorOrFloat = 1.0,
    call: bool = True,
) -> dict[str, torch.Tensor]:
    """Analytic delta / gamma / vega / theta / rho for a European vanilla."""
    t, sig = _f64(t_maturity), _f64(sigma)
    d1, d2 = _d1_d2(s, strike, r, q, sigma, t_maturity)
    df_r = torch.exp(-_f64(r) * t)
    df_q = torch.exp(-_f64(q) * t)
    pdf_d1 = _norm_pdf(d1)
    s_t, k_t = _f64(s), _f64(strike)
    r_t, q_t = _f64(r), _f64(q)

    if call:
        delta = df_q * _norm_cdf(d1)
        theta = (-s_t * df_q * pdf_d1 * sig / (2 * t.sqrt())
                 - r_t * k_t * df_r * _norm_cdf(d2)
                 + q_t * s_t * df_q * _norm_cdf(d1))
        rho = k_t * t * df_r * _norm_cdf(d2)
    else:
        delta = df_q * (_norm_cdf(d1) - 1.0)
        theta = (-s_t * df_q * pdf_d1 * sig / (2 * t.sqrt())
                 + r_t * k_t * df_r * _norm_cdf(-d2)
                 - q_t * s_t * df_q * _norm_cdf(-d1))
        rho = -k_t * t * df_r * _norm_cdf(-d2)

    gamma = df_q * pdf_d1 / (s_t * sig * t.sqrt())
    vega = s_t * df_q * pdf_d1 * t.sqrt()
    return {"delta": delta, "gamma": gamma, "vega": vega,
            "theta": theta, "rho": rho}
