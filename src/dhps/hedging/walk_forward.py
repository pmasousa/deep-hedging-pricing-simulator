"""Walk-forward hedging evaluation — frozen policy, rolling windows.

Deployment-honest evaluation: the trained policy is frozen (no
retraining per window — that is what shipping means), then rolled
across out-of-window conditions it never saw: the training regime, a
volatility shock (same model, sigma 0.30), and a structure break
(Heston dynamics). The delta baseline gets the same information it
would have in production: BS delta on GBM windows, variance-aware
delta under Heston. Premiums are fair per window (closed form on GBM,
Monte Carlo under Heston).
"""

import torch

from dhps.hedging.heston import DEFAULT_HESTON, heston_premium_mc, variance_aware_delta_positions
from dhps.hedging.policy import run_policy
from dhps.hedging.simulator import cvar, delta_positions, hedge_pnl, premium_bs
from dhps.simulators.gbm import simulate_gbm
from dhps.simulators.heston import simulate_heston

WINDOWS = (
    ("GBM sigma 0.20 (training regime)", "gbm", {"sigma": 0.20}),
    ("GBM sigma 0.30 (vol shock)", "gbm", {"sigma": 0.30}),
    ("Heston rho -0.7 (structure break)", "heston", DEFAULT_HESTON),
)


def walk_forward_eval(policy, cost_rate: float = 0.01, n_paths: int = 16_384,
                      seed: int = 11) -> list[dict]:
    """One row per window: policy vs delta CVaR95/mean, plus an
    aggregate verdict row (equal-weight mean across windows)."""
    rows = []
    for name, kind, params in WINDOWS:
        if kind == "gbm":
            sigma = params["sigma"]
            paths = simulate_gbm(n_paths=n_paths, n_steps=26, s0=100.0,
                                 r=0.05, q=0.01, sigma=sigma,
                                 t_maturity=1.0, antithetic=True, seed=seed)
            delta = delta_positions(paths, 100.0, 0.05, 0.01, sigma, 1.0)
            premium = premium_bs(100.0, 100.0, 0.05, 0.01, sigma, 1.0)
        else:
            paths, variances = simulate_heston(
                n_paths=n_paths, n_steps=26, s0=100.0, r=0.05, q=0.01,
                t_maturity=1.0, antithetic=True, seed=seed, **params)
            delta = variance_aware_delta_positions(paths, variances, 100.0,
                                                   0.05, 0.01, 1.0)
            premium = heston_premium_mc(**params)
        with torch.no_grad():
            pos_policy = run_policy(policy, paths, 100.0, 1.0)
        pnl_p = hedge_pnl(paths, 100.0, premium, pos_policy, cost_rate)
        pnl_d = hedge_pnl(paths, 100.0, premium, delta, cost_rate)
        rows.append({
            "window": name,
            "policy_cvar": cvar(pnl_p), "delta_cvar": cvar(pnl_d),
            "policy_mean": float(pnl_p.mean()),
            "delta_mean": float(pnl_d.mean()),
            "edge": cvar(pnl_p) - cvar(pnl_d),  # > 0: policy safer
        })
    n = len(rows)
    rows.append({
        "window": "aggregate (mean across windows)",
        "policy_cvar": sum(r["policy_cvar"] for r in rows) / n,
        "delta_cvar": sum(r["delta_cvar"] for r in rows) / n,
        "policy_mean": sum(r["policy_mean"] for r in rows) / n,
        "delta_mean": sum(r["delta_mean"] for r in rows) / n,
        "edge": sum(r["edge"] for r in rows) / n,
    })
    return rows
