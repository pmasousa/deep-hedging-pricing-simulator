"""Standardizer gates: stats, roundtrip, and the chain-rule gradient scale."""

import torch

from dhps.datasets.european import make_european_dataset
from dhps.datasets.normalize import Standardizer
from dhps.pricing.black_scholes import bs_greeks, bs_price

R, Q = 0.05, 0.01


def test_fit_transform_stats_and_roundtrip():
    d = make_european_dataset(n_samples=5_000, seed=5)
    sc = Standardizer.fit(d["x_train"], d["y_train"])
    xt = sc.transform_x(d["x_train"])
    yt = sc.transform_y(d["y_train"])
    assert torch.allclose(xt.mean(dim=0), torch.zeros(4, dtype=torch.float64), atol=1e-12)
    assert torch.allclose(xt.std(dim=0), torch.ones(4, dtype=torch.float64), atol=1e-12)
    assert torch.allclose(yt.mean(dim=0), torch.zeros(6, dtype=torch.float64), atol=1e-12)
    assert torch.allclose(sc.inverse_x(xt), d["x_train"], atol=1e-12)
    assert torch.allclose(sc.inverse_y(yt), d["y_train"], atol=1e-12)


def test_val_uses_train_stats():
    """Transforming val with train stats must NOT re-center val to N(0,1)."""
    d = make_european_dataset(n_samples=5_000, seed=5)
    sc = Standardizer.fit(d["x_train"], d["y_train"])
    xv = sc.transform_x(d["x_val"])
    assert not torch.allclose(xv.mean(dim=0), torch.zeros(4, dtype=torch.float64), atol=1e-3)


def test_grad_scale_is_the_chain_rule():
    """Jacobian of scaled price w.r.t. scaled spot = grad_scale * raw delta."""
    d = make_european_dataset(n_samples=2_000, seed=9)
    sc = Standardizer.fit(d["x_train"], d["y_train"])
    raw = d["x_val"][[0]]
    z = sc.transform_x(raw).detach().requires_grad_(True)
    x_back = sc.inverse_x(z)
    price = bs_price(x_back[:, 0], x_back[:, 1], R, Q, x_back[:, 3], x_back[:, 2])
    y_scaled = sc.transform_y(price.unsqueeze(1))
    # column 0 only: summing all six would mix the other labels' scales in
    jac = torch.autograd.grad(y_scaled[:, 0].sum(), z)[0]

    s0, k, t, sig = (float(v) for v in raw[0])
    # dtype is load-bearing: torch.tensor([s0]) alone truncates the spot to float32
    spot = torch.tensor([s0], dtype=torch.float64)
    delta_raw = float(bs_greeks(spot, k, R, Q, sig, t)["delta"][0])
    expected = delta_raw * float(sc.grad_scale[0, 0])  # price row, s0 column
    assert abs(float(jac[0, 0]) - expected) < 1e-12
    assert sc.grad_scale.shape == (6, 4)
