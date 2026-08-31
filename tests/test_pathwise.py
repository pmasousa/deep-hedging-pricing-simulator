"""Pathwise MC gates: CLT bands around the analytic Black-Scholes values."""

import torch

from dhps.pricing.black_scholes import bs_greeks, bs_price
from dhps.pricing.pathwise_mc import pathwise_european_greeks
from dhps.simulators.gbm import mc_european_price, simulate_gbm

N_PATHS = 200_000
CFG = dict(s0=100.0, strike=100.0, r=0.05, q=0.01, sigma=0.2, t_maturity=1.0)


def _pw(seed: int = 21) -> dict[str, float]:
    return pathwise_european_greeks(n_paths=N_PATHS, n_steps=64, seed=seed, **CFG)


def _analytic() -> dict[str, float]:
    s = torch.tensor([CFG["s0"]])
    p = float(bs_price(s, **{k: CFG[k] for k in ("strike", "r", "q", "sigma", "t_maturity")})[0])
    g = bs_greeks(s, **{k: CFG[k] for k in ("strike", "r", "q", "sigma", "t_maturity")})
    return {"price": p, "delta": float(g["delta"][0]), "vega": float(g["vega"][0])}


def test_pathwise_delta_and_vega_within_clt_bands():
    pw, an = _pw(), _analytic()
    assert abs(pw["delta"] - an["delta"]) < 5 * pw["delta_se"]
    assert abs(pw["vega"] - an["vega"]) < 5 * pw["vega_se"]
    assert pw["delta_se"] > 0.0 and pw["vega_se"] > 0.0


def test_pathwise_price_matches_plain_mc():
    """Tensor leaves must reproduce the plain float simulation bit-close."""
    pw = _pw()
    sim_cfg = {k: CFG[k] for k in ("s0", "r", "q", "sigma", "t_maturity")}
    paths = simulate_gbm(n_paths=N_PATHS, n_steps=64, seed=21, **sim_cfg)
    plain = mc_european_price(paths, strike=CFG["strike"], r=CFG["r"],
                              t_maturity=CFG["t_maturity"], call=True)
    assert abs(pw["price"] - plain) < 1e-10


def test_pathwise_greeks_deterministic_under_seed():
    a = pathwise_european_greeks(n_paths=50_000, n_steps=32, seed=99, **CFG)
    b = pathwise_european_greeks(n_paths=50_000, n_steps=32, seed=99, **CFG)
    for k in a:
        assert a[k] == b[k]
