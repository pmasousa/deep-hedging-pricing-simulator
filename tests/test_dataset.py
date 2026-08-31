"""Dataset builder gates: shapes, split hygiene, determinism, label truth."""

import torch

from dhps.datasets.european import FEATURES, LABELS, make_european_dataset
from dhps.pricing.black_scholes import bs_greeks, bs_price

N = 20_000


def _ds(seed: int = 3) -> dict[str, torch.Tensor]:
    return make_european_dataset(n_samples=N, seed=seed)


def test_shapes_ranges_and_dtypes():
    d = _ds()
    n_tr, n_va = d["x_train"].shape[0], d["x_val"].shape[0]
    assert n_tr + n_va == N
    assert d["x_train"].shape == (n_tr, len(FEATURES))
    assert d["y_train"].shape == (n_tr, len(LABELS))
    assert d["x_val"].shape == (n_va, len(FEATURES))
    assert all(t.dtype == torch.float64 for t in d.values())

    s0, strike, t, sig = d["x_train"].T
    assert bool((s0 >= 50.0).all() and (s0 <= 200.0).all())
    assert bool((t >= 0.1).all() and (t <= 2.0).all())
    assert bool((sig >= 0.05).all() and (sig <= 0.6).all())
    moneyness = strike / s0
    assert bool(((moneyness >= 0.6) & (moneyness <= 1.4)).all())


def test_split_is_disjoint_and_complete():
    d = _ds()
    all_rows = torch.cat([d["x_train"], d["x_val"]])
    # scrambled Sobol draws are unique as 4-tuples; duplicates mean leakage
    assert torch.unique(all_rows, dim=0).shape[0] == N


def test_deterministic_under_seed():
    a, b = _ds(seed=11), _ds(seed=11)
    for key in a:
        assert torch.equal(a[key], b[key])
    c = _ds(seed=12)
    assert not torch.equal(a["x_train"], c["x_train"])


def test_labels_match_scalar_pricer():
    """Broadcast labels must equal per-sample scalar calls (vectorization gate)."""
    d = _ds()
    n = 256
    xs, ys = d["x_val"][:n], d["y_val"][:n]
    for i in range(n):
        s0, k, t, sig = (float(v) for v in xs[i])
        p = float(bs_price(torch.tensor([s0]), strike=k, r=0.05, q=0.01,
                           sigma=sig, t_maturity=t)[0])
        g = bs_greeks(torch.tensor([s0]), strike=k, r=0.05, q=0.01,
                      sigma=sig, t_maturity=t)
        row = torch.tensor([p, float(g["delta"][0]), float(g["gamma"][0]),
                            float(g["vega"][0]), float(g["theta"][0]),
                            float(g["rho"][0])], dtype=torch.float64)
        assert torch.allclose(ys[i], row, atol=1e-10)


def test_label_invariants():
    d = _ds()
    price, delta, gamma, vega = [d["y_train"][:, j] for j in (0, 1, 2, 3)]
    # deep-OTM cancellation can round a zero price to ~-1e-15 in the closed form
    assert bool((price >= -1e-9).all())
    assert bool(((delta >= 0.0) & (delta <= 1.0)).all())  # long call
    assert bool((gamma >= 0.0).all() and (vega >= 0.0).all())


def test_gradient_labels_match_finite_differences():
    """g columns must be d(price)/d(feature), FEATURES order, theta negated.

    Tolerance covers central-FD truncation, dominated by the maturity column
    (theta's T-derivative grows fast as T -> 0); sign or column swaps would
    miss by O(1).
    """
    d = _ds()
    h = 1e-5
    n = 64
    xs, gs = d["x_val"][:n], d["g_val"][:n]
    for i in range(n):
        s0, k, t, sig = (float(v) for v in xs[i])
        cols = [
            (bs_price(torch.tensor([s0 + h], dtype=torch.float64), k, 0.05, 0.01, sig, t)
             - bs_price(torch.tensor([s0 - h], dtype=torch.float64), k, 0.05, 0.01, sig, t))
            / (2 * h),
            (bs_price(torch.tensor([s0], dtype=torch.float64), k + h, 0.05, 0.01, sig, t)
             - bs_price(torch.tensor([s0], dtype=torch.float64), k - h, 0.05, 0.01, sig, t))
            / (2 * h),
            (bs_price(torch.tensor([s0], dtype=torch.float64), k, 0.05, 0.01, sig, t + h)
             - bs_price(torch.tensor([s0], dtype=torch.float64), k, 0.05, 0.01, sig, t - h))
            / (2 * h),
            (bs_price(torch.tensor([s0], dtype=torch.float64), k, 0.05, 0.01, sig + h, t)
             - bs_price(torch.tensor([s0], dtype=torch.float64), k, 0.05, 0.01, sig - h, t))
            / (2 * h),
        ]
        fd = torch.stack([c[0] for c in cols])
        assert torch.allclose(gs[i], fd, atol=1e-5), i


def test_validation_errors():
    try:
        make_european_dataset(train_frac=1.5)
        raise AssertionError("train_frac out of (0,1) must raise")
    except ValueError:
        pass
    try:
        make_european_dataset(sigma_range=(0.6, 0.05))
        raise AssertionError("inverted range must raise")
    except ValueError:
        pass
