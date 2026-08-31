"""AAD gates: autograd Greeks must reproduce the analytic ones bit-tight."""

import torch

from dhps.pricing.aad import bs_greeks_ad
from dhps.pricing.black_scholes import bs_greeks, bs_price

K, R, Q, SIGMA, T = 100.0, 0.05, 0.01, 0.2, 1.0


def test_ad_greeks_match_analytic():
    """The repo's thesis in one assertion: autograd == derived formulas."""
    s = torch.linspace(60.0, 150.0, 37, dtype=torch.float64)
    ad = bs_greeks_ad(s, K, R, Q, SIGMA, T)
    an = bs_greeks(s, K, R, Q, SIGMA, T)
    for name in ("delta", "gamma", "vega", "theta", "rho"):
        assert torch.allclose(ad[name], an[name], atol=1e-10), name


def test_ad_greeks_match_analytic_put():
    s = torch.linspace(70.0, 130.0, 25, dtype=torch.float64)
    ad = bs_greeks_ad(s, K, R, Q, SIGMA, T, call=False)
    an = bs_greeks(s, K, R, Q, SIGMA, T, call=False)
    for name in ("delta", "gamma", "vega", "theta", "rho"):
        assert torch.allclose(ad[name], an[name], atol=1e-10), name


def test_dual_delta_matches_finite_difference():
    s = torch.tensor([95.0, 100.0, 105.0], dtype=torch.float64)
    ad = bs_greeks_ad(s, K, R, Q, SIGMA, T)
    h = 1e-4
    fd = (bs_price(s, K + h, R, Q, SIGMA, T) - bs_price(s, K - h, R, Q, SIGMA, T)) / (2 * h)
    assert torch.allclose(ad["dual_delta"], fd, atol=1e-6)
