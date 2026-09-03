"""Heston statistical gates: mean reversion, martingale, reproducibility.

Params are Feller-satisfied (2*kappa*theta = 0.16 >= xi^2 = 0.09) so
negative variance excursions are rare and the full-truncation bias stays
under the 2%-of-theta slack in the mean-reversion band.
"""

import math

import torch

from dhps.simulators.heston import simulate_heston

N, STEPS = 200_000, 128
PARAMS = dict(s0=100.0, r=0.05, q=0.01, v0=0.09, kappa=2.0, theta=0.04,
              xi=0.3, rho=-0.7, t_maturity=1.0)


def _sim(seed: int = 7):
    return simulate_heston(n_paths=N, n_steps=STEPS, seed=seed, **PARAMS)


def test_shapes_and_positivity():
    spots, variances = _sim()
    assert spots.shape == variances.shape == (N, STEPS + 1)
    assert spots.dtype == variances.dtype == torch.float64
    assert bool((variances >= 0).all())  # reported v+ by construction
    assert torch.allclose(spots[:, 0],
                          torch.full((N,), 100.0, dtype=torch.float64))
    assert torch.allclose(variances[:, 0],
                          torch.full((N,), 0.09, dtype=torch.float64))


def test_variance_mean_reversion():
    """E[v_T] = theta + (v0 - theta) e^{-kappa T}, CLT band + 2% of theta
    slack for full-truncation discretization bias."""
    _, variances = _sim()
    v_t = variances[:, -1]
    theory = PARAMS["theta"] + (PARAMS["v0"] - PARAMS["theta"]) * math.exp(
        -PARAMS["kappa"] * PARAMS["t_maturity"])
    se = float(v_t.std(unbiased=True)) / math.sqrt(N)
    band = max(5 * se, 0.02 * PARAMS["theta"])
    assert abs(float(v_t.mean()) - theory) < band, (float(v_t.mean()), theory)


def test_spot_is_a_martingale():
    """E[S_T] = S0 e^{(r-q)T} — exact under log-Euler conditional on the
    variance path, so the band is pure CLT."""
    spots, _ = _sim()
    s_t = spots[:, -1]
    expected = PARAMS["s0"] * math.exp(
        (PARAMS["r"] - PARAMS["q"]) * PARAMS["t_maturity"])
    se = float(s_t.std(unbiased=True)) / math.sqrt(N)
    assert abs(float(s_t.mean()) - expected) < 5 * se


def test_seeded_reproducible_and_validation():
    a = simulate_heston(n_paths=10_000, n_steps=32, seed=99, **PARAMS)
    b = simulate_heston(n_paths=10_000, n_steps=32, seed=99, **PARAMS)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    try:
        simulate_heston(n_paths=7, n_steps=4, **PARAMS)
        raise AssertionError("odd n_paths with antithetic must raise")
    except ValueError:
        pass
    try:
        bad = {**PARAMS, "rho": 1.0}
        simulate_heston(n_paths=8, n_steps=4, **bad)
        raise AssertionError("rho=1 must raise")
    except ValueError:
        pass
