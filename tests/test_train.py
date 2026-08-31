"""Trainer gates: canary overfit, gradient flow, run folders, and the
framework's headline claim — the differential learner beats the baseline
on delta, not just on price.
"""

import json

import torch

from dhps.models.losses import differential_loss, value_and_grad
from dhps.models.mlp import make_mlp
from dhps.train.trainer import TrainConfig, save_run, train_model

SMALL = dict(n_samples=8_000, hidden=(48, 48), epochs=120, batch_size=4_096,
             seed=7)


def test_canary_dml_overfits():
    """Training loss must collapse on a small problem (learning works)."""
    cfg = TrainConfig(n_samples=2_000, hidden=(32, 32), epochs=80,
                      batch_size=512, seed=3)
    res = train_model(cfg, differential=True)
    first, last = res.history["train_loss"][0], res.history["train_loss"][-1]
    assert last < 0.05 * first, (first, last)


def test_differential_loss_matches_components():
    torch.manual_seed(0)
    y = torch.randn(32, 1, dtype=torch.float64)
    g = torch.randn(32, 4, dtype=torch.float64)
    loss = differential_loss(y, g, y, g)
    assert float(loss) == 0.0
    # alpha = 1/(1+lam*n_cols) = 1/5, beta = 4/5 with lam=1, 4 columns
    y2 = torch.full((32, 1), 1.0, dtype=torch.float64)
    g2 = torch.zeros(32, 4, dtype=torch.float64)
    manual = 0.2 * 1.0 + 0.8 * 0.0
    assert abs(float(differential_loss(y2, g2, y * 0, g2)) - manual) < 1e-12


def test_value_and_grad_shapes_and_flow():
    torch.manual_seed(1)
    model = make_mlp(n_in=4, hidden=(16, 16))
    x = torch.randn(64, 4, dtype=torch.float64)
    y, g = value_and_grad(model, x)
    assert y.shape == (64, 1) and g.shape == (64, 4)
    loss = differential_loss(y, g, torch.zeros_like(y), torch.zeros_like(g))
    loss.backward()
    grads = [p.grad for p in model.parameters()]
    assert all(p.grad is not None and float(p.grad.abs().sum()) > 0
               for p in model.parameters())
    assert all(torch.isfinite(g_).all() for g_ in grads)


def test_dml_beats_baseline_on_delta():
    """The framework's claim, as a regression gate: differential training
    produces materially better delta than values-only training. Observed
    ratio at this budget ~0.65; a broken differential signal (mis-scaled
    or mis-signed labels, dead gradient path) degrades DML to the baseline
    ratio ~1.0, far above the 0.8 gate."""
    cfg = TrainConfig(**SMALL)
    dml = train_model(cfg, differential=True)
    base = train_model(cfg, differential=False)
    assert dml.metrics["val_delta_mae"] < 0.8 * base.metrics["val_delta_mae"], (
        dml.metrics, base.metrics)


def test_save_run_writes_folder(tmp_path):
    cfg = TrainConfig(n_samples=1_000, hidden=(16,), epochs=2, batch_size=512,
                      seed=1)
    res = train_model(cfg, differential=True)
    run_dir = save_run(res, cfg, differential=True, run_dir=tmp_path / "run1")
    assert (run_dir / "run.json").exists()
    assert (run_dir / "history.json").exists()
    assert (run_dir / "weights.pt").exists()
    payload = json.loads((run_dir / "run.json").read_text())
    assert payload["differential"] is True
    assert payload["metrics"]["val_delta_mae"] >= 0.0
