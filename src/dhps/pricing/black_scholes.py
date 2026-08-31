"""Closed-form Black-Scholes prices and Greeks — the SP1 reference pricer.

Torch-native (vectorized, float64) so the same call sites later swap in the
DML core; Greeks here are the analytic ground truth the DML Greeks must hit.
"""

import math

import torch


def _d1_d2(s: torch.Tensor, strike: float, r: float, q: float,
           sigma: float, t_maturity: float) -> tuple[torch.Tensor, torch.Tensor]:
    vol_t = sigma * math.sqrt(t_maturity)
    d1 = (torch.log(s / strike) + (r - q + 0.5 * sigma**2) * t_maturity) / vol_t
    return d1, d1 - vol_t


def _norm_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(-0.5 * x**2) / math.sqrt(2.0 * math.pi)


def bs_price(
    s: torch.Tensor,
    strike: float,
    r: float = 0.05,
    q: float = 0.0,
    sigma: float = 0.2,
    t_maturity: float = 1.0,
    call: bool = True,
) -> torch.Tensor:
    """Black-Scholes price for spot tensor ``s`` (q = continuous dividend yield)."""
    if t_maturity <= 0 or sigma <= 0:
        raise ValueError("require sigma > 0, t_maturity > 0")
    d1, d2 = _d1_d2(s, strike, r, q, sigma, t_maturity)
    df_r = math.exp(-r * t_maturity)
    df_q = math.exp(-q * t_maturity)
    if call:
        return s * df_q * _norm_cdf(d1) - strike * df_r * _norm_cdf(d2)
    return strike * df_r * _norm_cdf(-d2) - s * df_q * _norm_cdf(-d1)


def bs_greeks(
    s: torch.Tensor,
    strike: float,
    r: float = 0.05,
    q: float = 0.0,
    sigma: float = 0.2,
    t_maturity: float = 1.0,
    call: bool = True,
) -> dict[str, torch.Tensor]:
    """Analytic delta / gamma / vega / theta / rho for a European vanilla."""
    d1, d2 = _d1_d2(s, strike, r, q, sigma, t_maturity)
    df_r = math.exp(-r * t_maturity)
    df_q = math.exp(-q * t_maturity)
    pdf_d1 = _norm_pdf(d1)

    if call:
        delta = df_q * _norm_cdf(d1)
        theta = (-s * df_q * pdf_d1 * sigma / (2 * math.sqrt(t_maturity))
                 - r * strike * df_r * _norm_cdf(d2)
                 + q * s * df_q * _norm_cdf(d1))
        rho = strike * t_maturity * df_r * _norm_cdf(d2)
    else:
        delta = df_q * (_norm_cdf(d1) - 1.0)
        theta = (-s * df_q * pdf_d1 * sigma / (2 * math.sqrt(t_maturity))
                 + r * strike * df_r * _norm_cdf(-d2)
                 - q * s * df_q * _norm_cdf(-d1))
        rho = -strike * t_maturity * df_r * _norm_cdf(-d2)

    gamma = df_q * pdf_d1 / (s * sigma * math.sqrt(t_maturity))
    vega = s * df_q * pdf_d1 * math.sqrt(t_maturity)
    return {"delta": delta, "gamma": gamma, "vega": vega,
            "theta": theta, "rho": rho}
