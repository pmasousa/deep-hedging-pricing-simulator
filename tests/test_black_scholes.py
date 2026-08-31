"""Black-Scholes reference pricer gates: parity + Greeks vs finite differences."""

import math

import torch

from dhps.pricing.black_scholes import bs_greeks, bs_price

K, R, Q, SIGMA, T = 100.0, 0.05, 0.01, 0.2, 1.0


def _spots() -> torch.Tensor:
    return torch.linspace(60.0, 150.0, 37, dtype=torch.float64)


def test_put_call_parity_grid():
    s = _spots()
    c = bs_price(s, K, R, Q, SIGMA, T, call=True)
    p = bs_price(s, K, R, Q, SIGMA, T, call=False)
    lhs = c - p
    rhs = s * math.exp(-Q * T) - K * math.exp(-R * T)
    assert torch.allclose(lhs, rhs, atol=1e-10)


def test_greeks_match_finite_differences():
    s = torch.tensor([90.0, 100.0, 110.0], dtype=torch.float64)
    g = bs_greeks(s, K, R, Q, SIGMA, T, call=True)
    h = 1e-4

    up = bs_price(s + h, K, R, Q, SIGMA, T, call=True)
    dn = bs_price(s - h, K, R, Q, SIGMA, T, call=True)
    assert torch.allclose(g["delta"], (up - dn) / (2 * h), atol=1e-6)
    assert torch.allclose(g["gamma"], (up - 2 * bs_price(s, K, R, Q, SIGMA, T) + dn) / h**2,
                          atol=1e-4)

    vega_fd = (bs_price(s, K, R, Q, SIGMA + h, T) - bs_price(s, K, R, Q, SIGMA - h, T)) / (2 * h)
    assert torch.allclose(g["vega"], vega_fd, atol=1e-5)

    rho_fd = (bs_price(s, K, R + h, Q, SIGMA, T) - bs_price(s, K, R - h, Q, SIGMA, T)) / (2 * h)
    assert torch.allclose(g["rho"], rho_fd, atol=1e-5)


def test_broadcast_matches_scalar_loop():
    """Per-sample tensor parameters must equal elementwise scalar calls."""
    s = torch.tensor([80.0, 100.0, 120.0], dtype=torch.float64)
    k = torch.tensor([90.0, 105.0, 95.0], dtype=torch.float64)
    sig = torch.tensor([0.15, 0.25, 0.35], dtype=torch.float64)
    t = torch.tensor([0.3, 1.0, 1.7], dtype=torch.float64)

    vec = bs_price(s, k, R, Q, sig, t)
    loop = torch.stack([bs_price(s[i:i + 1], float(k[i]), R, Q,
                                 float(sig[i]), float(t[i]))[0] for i in range(3)])
    assert torch.allclose(vec, loop, atol=1e-12)

    gv = bs_greeks(s, k, R, Q, sig, t)
    gl = [bs_greeks(s[i:i + 1], float(k[i]), R, Q, float(sig[i]), float(t[i]))
          for i in range(3)]
    for name in gv:
        stacked = torch.stack([g[name][0] for g in gl])
        assert torch.allclose(gv[name], stacked, atol=1e-12)


def test_theta_sign_convention():
    """Theta is per-year and negative for a long ATM call."""
    s = torch.tensor([100.0], dtype=torch.float64)
    g = bs_greeks(s, K, R, Q, SIGMA, T, call=True)
    assert float(g["theta"]) < 0.0
    # daily theta = annual / 365 sanity: between -1 and 0 cents/day for this config
    assert -1.0 < float(g["theta"]) / 365 < 0.0
