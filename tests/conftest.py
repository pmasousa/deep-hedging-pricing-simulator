"""Shared fixtures for the hedging gates: one policy training per session."""

import pytest

from dhps.hedging.policy import DeepHedgeConfig, train_deep_hedge


@pytest.fixture(scope="session")
def trained_policy():
    """The GBM policy reused by the GBM, Heston, and walk-forward gates —
    training it once keeps the suite at one training run."""
    cfg = DeepHedgeConfig(n_paths=16_384, n_steps=26, cost_rate=0.01,
                          epochs=150, seed=7, eval_paths=8_192)
    return train_deep_hedge(cfg)
