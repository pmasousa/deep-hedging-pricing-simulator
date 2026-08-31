"""Statistical gates for the GBM simulator (SP1).

Every tolerance is CLT-aware: with the seeded antithetic engine at
n_paths = 400_000, the standard error of the terminal-mean estimator is
~sigma_T / sqrt(n), so 5-sigma bands are far tighter than any plausible
bug drift yet loose enough to never flake.
"""

import math

import torch

from dhps.pricing.black_scholes import bs_price
from dhps.simulators.gbm import mc_european_price, simulate_gbm

N_PATHS = 400_000
N_STEPS = 64
S0, R, Q, SIGMA, T, K = 100.0, 0.05, 0.01, 0.2, 1.0, 100.0


def _paths(seed: int = 7) -> torch.Tensor:
    return simulate_gbm(
        n_paths=N_PATHS, n_steps=N_STEPS, s0=S0, r=R, q=Q,
        sigma=SIGMA, t_maturity=T, antithetic=True, seed=seed,
    )


def test_terminal_lognormal_moments():
    """E[log S_T] and Var[log S_T] must match the closed-form moments."""
    s_t = _paths()[:, -1]
    log_s = torch.log(s_t)
    m = math.log(S0) + (R - Q - 0.5 * SIGMA**2) * T
    v = SIGMA**2 * T
    se_mean = math.sqrt(v / N_PATHS)
    assert abs(float(log_s.mean()) - m) < 5 * se_mean
    # sample variance of ~400k normals: relative tolerance 2% is ~30 sigma out
    assert abs(float(log_s.var(unbiased=True)) - v) / v < 0.02


def test_mc_european_matches_black_scholes():
    """MC call price inside a 5-sigma CLT band around the closed form."""
    paths = _paths()
    mc = mc_european_price(paths, strike=K, r=R, t_maturity=T, call=True)
    bs = float(bs_price(torch.tensor([S0]), strike=K, r=R, q=Q,
                        sigma=SIGMA, t_maturity=T, call=True)[0])
    payoff = torch.clamp(paths[:, -1] - K, min=0.0)
    disc = math.exp(-R * T)
    se = disc * float(payoff.std(unbiased=True)) / math.sqrt(N_PATHS)
    assert abs(mc - bs) < 5 * se
    assert abs(mc - bs) < 0.05  # absolute sanity floor


def test_put_call_parity_on_mc_prices():
    """No-arb identity C - P = S0 e^{-qT} - K e^{-rT} on MC estimates."""
    paths = _paths()
    call = mc_european_price(paths, strike=K, r=R, t_maturity=T, call=True)
    put = mc_european_price(paths, strike=K, r=R, t_maturity=T, call=False)
    lhs = call - put
    rhs = S0 * math.exp(-Q * T) - K * math.exp(-R * T)
    # antithetic pairing shares Z across call/put, so the parity error is
    # much smaller than independent-sampling CLT would suggest; 1 cent is safe
    assert abs(lhs - rhs) < 0.01


def test_forward_is_a_martingale_under_rn_measure():
    """E[S_T] = S0 e^{(r-q)T} — no drift leak from a wrong discretization."""
    s_t = _paths()[:, -1]
    expected = S0 * math.exp((R - Q) * T)
    se = float(s_t.std(unbiased=True)) / math.sqrt(N_PATHS)
    assert abs(float(s_t.mean()) - expected) < 5 * se


def test_antithetic_seeded_reproducible_and_paired():
    a = simulate_gbm(n_paths=10_000, n_steps=32, s0=S0, sigma=SIGMA,
                     t_maturity=0.5, antithetic=True, seed=123)
    b = simulate_gbm(n_paths=10_000, n_steps=32, s0=S0, sigma=SIGMA,
                     t_maturity=0.5, antithetic=True, seed=123)
    assert torch.equal(a, b)  # same seed -> bit-identical
    # rows i and i+half use +Z / -Z: paired log-increments must sum to 2*drift
    half = 5_000
    log_inc = torch.log(a[:, 1:] / a[:, :-1])
    drift = (0.05 - 0.0 - 0.5 * SIGMA**2) * (0.5 / 32)
    pair_sum = log_inc[:half] + log_inc[half:]
    assert torch.allclose(pair_sum, torch.full_like(pair_sum, 2 * drift), atol=1e-10)
    assert torch.allclose(a[:, 0], torch.full((10_000,), S0, dtype=torch.float64))  # s0 anchor


def test_shapes_and_validation():
    p = simulate_gbm(n_paths=8, n_steps=4, antithetic=True, seed=1)
    assert p.shape == (8, 5) and p.dtype == torch.float64
    try:
        simulate_gbm(n_paths=7, n_steps=4, antithetic=True)
        raise AssertionError("odd n_paths with antithetic must raise")
    except ValueError:
        pass
