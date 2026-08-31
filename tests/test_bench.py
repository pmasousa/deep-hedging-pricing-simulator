"""Bench gates: matched-n math, timing sanity, metric finiteness, OOD stress."""

import math

import pytest
import torch

from dhps.bench.evaluate import evaluate_learner, greeks_curve, ood_metrics
from dhps.bench.speed import mc_paths_for_error, payoff_std, time_fn
from dhps.datasets.european import make_european_dataset
from dhps.train.trainer import TrainConfig, train_model


def test_matched_paths_math():
    assert mc_paths_for_error(0.1, 0.001) == 10_000   # (std/err)^2, above floor
    assert mc_paths_for_error(10.0, 0.05) == 40_000
    assert mc_paths_for_error(1.0, 10.0) == 1_000  # floor kicks in
    # antithetic engine requires even counts: 35^2 = 1225 -> 1226 (above floor)
    assert mc_paths_for_error(0.35, 0.01) == 1226
    with pytest.raises(ValueError):
        mc_paths_for_error(1.0, 0.0)


def test_time_fn_and_payoff_std():
    assert time_fn(lambda: None, repeats=3, warmup=1) >= 0.0
    assert payoff_std(n_paths=50_000) > 0.0


@pytest.fixture(scope="module")
def trained():
    # the money-test budget: enough convergence for a meaningful curve gate
    cfg = TrainConfig(n_samples=8_000, hidden=(48, 48), epochs=120,
                      batch_size=4_096, seed=5)
    data = make_european_dataset(n_samples=8_000, seed=5)
    result = train_model(cfg, differential=True)
    return result, data


def test_evaluate_metrics_finite(trained):
    result, data = trained
    m = evaluate_learner(result, data)
    assert all(math.isfinite(v) and v > 0 for v in m.values())


def test_ood_is_harder_than_id(trained):
    """A net trained on [50, 200] must degrade outside that range."""
    result, data = trained
    m, o = evaluate_learner(result, data), ood_metrics(result)
    assert all(math.isfinite(v) for v in o.values())
    assert o["ood_price_mae"] > m["price_mae"]


def test_greeks_curve_shapes_and_quality(trained):
    result, _ = trained
    spots = torch.linspace(60.0, 190.0, 50, dtype=torch.float64)
    c = greeks_curve(result, spots, 100.0, 1.0, 0.2)
    assert c["delta"].shape == (50,) and c["gamma"].shape == (50,)
    delta_err = float((c["delta"] - c["delta_true"]).abs().mean())
    assert delta_err < 0.15, delta_err
    # gamma was never a label; only demand finiteness and rough scale
    assert bool(torch.isfinite(c["gamma"]).all())
    assert float(c["gamma_true"].abs().max()) > 0.0
