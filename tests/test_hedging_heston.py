"""Heston and walk-forward hedging gates — the Mission 2 spec demo.

The spec sentence: the deep hedger beats delta on CVaR95 on BOTH GBM
and Heston, walk-forward. All gates share one session-trained policy
(conftest) evaluated zero-shot outside its training regime.
"""

from dhps.hedging.heston import evaluate_on_heston, gbm_flat_delta_check, heston_premium_mc
from dhps.hedging.walk_forward import walk_forward_eval


def test_variance_aware_delta_reduces_to_bs_delta():
    """With a flat variance path the variance-aware delta IS the BS delta."""
    assert gbm_flat_delta_check() < 1e-10


def test_policy_beats_variance_aware_delta_on_heston(trained_policy):
    """Zero-shot transfer: the GBM-trained policy still beats a delta that
    sees the simulated variance path, under stochastic volatility."""
    premium = heston_premium_mc()
    stats = evaluate_on_heston(trained_policy.policy, premium)
    policy = stats["deep hedge (policy)"]
    delta = stats["delta (var-aware)"]
    assert policy["cvar95"] > delta["cvar95"], (policy, delta)


def test_walk_forward_policy_wins_on_aggregate(trained_policy):
    """Frozen policy rolled across: training regime, vol shock, structure
    break. Gate is the equal-weight aggregate tail — individual windows
    may go either way (near-frictionless regimes favor delta)."""
    rows = walk_forward_eval(trained_policy.policy)
    aggregate = rows[-1]
    assert aggregate["edge"] > 0.0, rows
    # per-window sanity: every number finite, three windows plus aggregate
    assert len(rows) == 4
    assert all(abs(r["policy_cvar"]) < 100 for r in rows)
