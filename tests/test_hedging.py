"""Deep-hedging gates: P&L mechanics, risk measures, and the money test —
a policy trained under costs must beat weekly delta hedging on the tail.
"""

import math

import torch

from dhps.hedging.policy import DeepHedgeConfig, delta_baseline_metrics, train_deep_hedge
from dhps.hedging.simulator import (
    banded_positions,
    cvar,
    delta_positions,
    entropic_risk,
    hedge_pnl,
    premium_bs,
)
from dhps.simulators.gbm import simulate_gbm

S0, K, R, Q, SIG, T, M, N = 100.0, 100.0, 0.05, 0.01, 0.2, 1.0, 52, 100_000


def _paths() -> torch.Tensor:
    return simulate_gbm(n_paths=N, n_steps=M, s0=S0, r=R, q=Q, sigma=SIG,
                        t_maturity=T, antithetic=True, seed=11)


def test_delta_hedge_cuts_variance():
    """Zero-cost weekly delta must crush unhedged std (discrete residual only)."""
    paths = _paths()
    prem = premium_bs(S0, K, R, Q, SIG, T)
    delta = delta_positions(paths, K, R, Q, SIG, T)
    pnl = hedge_pnl(paths, K, prem, delta, 0.0)
    naked = hedge_pnl(paths, K, prem, torch.zeros_like(delta), 0.0)
    assert float(pnl.std()) < 0.2 * float(naked.std())
    assert float(pnl.mean()) > float(naked.mean())  # carry, no cost drag


def test_costs_drag_is_monotone():
    paths = _paths()
    prem = premium_bs(S0, K, R, Q, SIG, T)
    delta = delta_positions(paths, K, R, Q, SIG, T)
    means = [float(hedge_pnl(paths, K, prem, delta, c).mean())
             for c in (0.0, 0.01, 0.05)]
    assert means[0] > means[1] > means[2]


def test_banded_trades_less_keeps_more():
    """The cost-awareness mechanism: fewer trades, better mean under costs."""
    paths = _paths()
    prem = premium_bs(S0, K, R, Q, SIG, T)
    delta = delta_positions(paths, K, R, Q, SIG, T)
    banded = banded_positions(delta, 0.05)
    pnl_d = hedge_pnl(paths, K, prem, delta, 0.01)
    pnl_b = hedge_pnl(paths, K, prem, banded, 0.01)

    def trade_count(pos: torch.Tensor) -> float:
        steps = torch.cat([pos[:, :1], pos[:, 1:] - pos[:, :-1]], dim=1)
        return float((steps != 0).float().sum(dim=1).mean())

    assert trade_count(banded) < trade_count(delta)
    assert float(pnl_b.mean()) > float(pnl_d.mean())


def test_risk_measure_formulas():
    pnl = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
    # worst 33.3% -> ceil(1) element? alpha=1/2 -> k=2: mean of [-1, 0]
    assert abs(cvar(pnl, 0.5) - (-0.5)) < 1e-12
    # entropic with lambd -> 0 converges to -mean(pnl)
    val = float(entropic_risk(pnl, 1e-6))
    assert abs(val - 0.0) < 1e-5
    # hand value at lambd=1: log(mean(e^{1}, 1, e^{-1})) = log((e + 1 + 1/e)/3)
    expected = math.log((math.e + 1.0 + 1.0 / math.e) / 3.0)
    assert abs(float(entropic_risk(pnl, 1.0)) - expected) < 1e-12


def test_policy_learns_and_beats_delta_under_costs():
    """The SP5 money gate: trained policy must beat weekly delta on CVaR95
    and entropic risk under the training cost regime, on shared eval paths."""
    cfg = DeepHedgeConfig(n_paths=16_384, n_steps=26, cost_rate=0.01,
                          epochs=150, seed=7, eval_paths=8_192)
    res = train_deep_hedge(cfg)
    base = delta_baseline_metrics(cfg)

    assert res.history["train_risk"][-1] < res.history["train_risk"][0]
    assert res.metrics["cvar95"] > base["cvar95"], (res.metrics, base)
    assert res.metrics["traded_volume"] < base["traded_volume"], (
        "cost-aware policy should move less volume than full rebalancing")
