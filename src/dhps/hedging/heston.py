"""Hedging under Heston dynamics — the spec's second demo arena.

The deep hedger was trained on GBM paths; this module evaluates it
zero-shot under stochastic volatility, against a VARIANCE-AWARE delta:
Black-Scholes delta computed on the simulated variance path. Feeding
the delta the realized variance is the standard practitioner baseline
under stochastic volatility — it is stronger than naive
delta-at-v0, so beating it is the honest claim.

The premium is charged fairly per dynamics: pass QuantLib's Heston
price when available, or use the Monte Carlo premium from this
simulator (self-contained; the QuantLib gate lives in the test suite).
"""

import math

import torch

from dhps.hedging.simulator import cvar, hedge_pnl
from dhps.pricing.black_scholes import bs_price
from dhps.simulators.gbm import european_payoff, simulate_gbm
from dhps.simulators.heston import simulate_heston

DEFAULT_HESTON = dict(v0=0.09, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
_NORMAL = torch.distributions.Normal(0.0, 1.0)


def variance_aware_delta_positions(
    paths: torch.Tensor, variances: torch.Tensor, strike: float,
    r: float, q: float, t_maturity: float,
) -> torch.Tensor:
    """BS delta along each path with vol = sqrt(simulated variance).

    Reduces exactly to the constant-vol BS delta when the variance path
    is flat — pinned by a test.
    """
    m = paths.shape[1] - 1
    ttms = t_maturity * (1.0 - torch.arange(m, dtype=paths.dtype) / m)
    ttms = ttms.repeat(paths.shape[0], 1)
    vols = variances[:, :m].sqrt()
    d1 = ((torch.log(paths[:, :m] / strike)
           + (r - q + 0.5 * vols.square()) * ttms)
          / (vols * ttms.sqrt()))
    # BS delta with dividends carries the df_q factor — same as the
    # constant-vol delta_positions it must reduce to
    return _NORMAL.cdf(d1) * torch.exp(-q * ttms)


def heston_premium_mc(s0: float = 100.0, strike: float = 100.0,
                      r: float = 0.05, q: float = 0.01,
                      t_maturity: float = 1.0, n_paths: int = 200_000,
                      seed: int = 13, **heston_params: float) -> float:
    """Fair short-call premium from this simulator (Monte Carlo)."""
    params = heston_params or DEFAULT_HESTON
    paths, _ = simulate_heston(n_paths=n_paths, n_steps=128, s0=s0, r=r,
                               q=q, t_maturity=t_maturity, seed=seed,
                               **params)
    payoff = european_payoff(paths, strike)
    return math.exp(-r * t_maturity) * float(payoff.mean())


def evaluate_on_heston(policy, premium: float, cost_rate: float = 0.01,
                       n_paths: int = 16_384, seed: int = 11,
                       include_pnl: bool = False,
                       **heston_params: float) -> dict[str, dict]:
    """Per-strategy P&L stats on Heston paths; the policy is the one
    trained on GBM — zero-shot transfer, nothing tuned here.
    ``include_pnl`` adds a subsampled per-path P&L list per strategy
    (for distribution plots)."""
    from dhps.hedging.policy import run_policy

    params = heston_params or DEFAULT_HESTON
    paths, variances = simulate_heston(n_paths=n_paths, n_steps=26, s0=100.0,
                                       r=0.05, q=0.01, t_maturity=1.0,
                                       antithetic=True, seed=seed, **params)
    delta = variance_aware_delta_positions(paths, variances, 100.0,
                                           0.05, 0.01, 1.0)
    with torch.no_grad():
        pos_policy = run_policy(policy, paths, 100.0, 1.0)

    def stats(pnl: torch.Tensor) -> dict:
        out = {"mean": float(pnl.mean()), "std": float(pnl.std()),
               "cvar95": cvar(pnl)}
        if include_pnl:
            out["pnl"] = pnl[::4].tolist()
        return out

    return {
        "no hedge": stats(hedge_pnl(paths, 100.0, premium,
                                    torch.zeros_like(delta), cost_rate)),
        "delta (var-aware)": stats(hedge_pnl(paths, 100.0, premium, delta,
                                             cost_rate)),
        "deep hedge (policy)": stats(hedge_pnl(paths, 100.0, premium,
                                               pos_policy, cost_rate)),
    }


def gbm_flat_delta_check() -> float:
    """Exposed for the reduction test: max |variance-aware − BS delta|
    when the variance path is pinned to sigma^2 (must be 0)."""
    paths = simulate_gbm(n_paths=64, n_steps=26, s0=100.0, r=0.05, q=0.01,
                         sigma=0.2, t_maturity=1.0, seed=5)
    flat_var = torch.full_like(paths, 0.04)
    va = variance_aware_delta_positions(paths, flat_var, 100.0, 0.05,
                                        0.01, 1.0)
    from dhps.hedging.simulator import delta_positions
    bs = delta_positions(paths, 100.0, 0.05, 0.01, 0.2, 1.0)
    return float((va - bs).abs().max())


def bs_call_price(s0: float = 100.0, strike: float = 100.0,
                  r: float = 0.05, q: float = 0.01, sigma: float = 0.2,
                  t_maturity: float = 1.0) -> float:
    return float(bs_price(torch.tensor([s0], dtype=torch.float64), strike,
                          r, q, sigma, t_maturity)[0])
