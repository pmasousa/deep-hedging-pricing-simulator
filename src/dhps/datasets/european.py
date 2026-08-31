"""European vanilla dataset builder — the SP2 label factory.

Scrambled-Sobol coverage of (s0, moneyness, T, sigma); strike is derived as
moneyness * s0 so the parameter grid stays meaningful across spot levels.
Value labels are analytic Black-Scholes prices + Greeks; gradient labels
(g_train/g_val) are d(price)/d(feature) from autograd through the pricer —
the differential-ML training signal. float64 end to end; deterministic
under ``seed``.
"""

import torch

from dhps.pricing.aad import bs_greeks_ad
from dhps.pricing.black_scholes import bs_greeks, bs_price

FEATURES = ("s0", "strike", "t_maturity", "sigma")
LABELS = ("price", "delta", "gamma", "vega", "theta", "rho")


def make_european_dataset(
    n_samples: int = 100_000,
    train_frac: float = 0.9,
    *,
    s0_range: tuple[float, float] = (50.0, 200.0),
    moneyness_range: tuple[float, float] = (0.6, 1.4),
    t_range: tuple[float, float] = (0.1, 2.0),
    sigma_range: tuple[float, float] = (0.05, 0.6),
    r: float = 0.05,
    q: float = 0.01,
    call: bool = True,
    seed: int = 7,
) -> dict[str, torch.Tensor]:
    """Sample a European dataset; returns float64 tensors keyed by split.

    ``x_train``/``x_val`` have shape ``(n, len(FEATURES))`` with columns in
    FEATURES order; ``y_train``/``y_val`` have shape ``(n, len(LABELS))``
    with columns in LABELS order; ``g_train``/``g_val`` have shape
    ``(n, len(FEATURES))`` holding d(price)/d(feature) in FEATURES order
    (theta enters negated: d(price)/d(t_maturity) = -theta).
    """
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be in (0, 1)")
    ranges = {"s0": s0_range, "moneyness": moneyness_range,
              "t_maturity": t_range, "sigma": sigma_range}
    for name, (lo, hi) in ranges.items():
        if not lo < hi:
            raise ValueError(f"invalid range for {name}: [{lo}, {hi}]")

    sobol = torch.quasirandom.SobolEngine(dimension=4, scramble=True, seed=seed)
    u = sobol.draw(n_samples, dtype=torch.float64)

    def _uniform(i: int, name: str) -> torch.Tensor:
        lo, hi = ranges[name]
        return lo + (hi - lo) * u[:, i]

    s0 = _uniform(0, "s0")
    strike = _uniform(1, "moneyness") * s0
    t_maturity = _uniform(2, "t_maturity")
    sigma = _uniform(3, "sigma")

    x = torch.stack([s0, strike, t_maturity, sigma], dim=1)
    price = bs_price(s0, strike, r, q, sigma, t_maturity, call=call)
    greeks = bs_greeks(s0, strike, r, q, sigma, t_maturity, call=call)
    y = torch.stack([price, greeks["delta"], greeks["gamma"], greeks["vega"],
                     greeks["theta"], greeks["rho"]], dim=1)
    # differential labels: the price's input-gradient, FEATURES order
    ad = bs_greeks_ad(s0, strike, r, q, sigma, t_maturity, call=call)
    g = torch.stack([ad["delta"], ad["dual_delta"], -ad["theta"], ad["vega"]],
                    dim=1)

    gen = torch.Generator().manual_seed(seed + 1)
    perm = torch.randperm(n_samples, generator=gen)
    n_train = int(round(train_frac * n_samples))
    idx_tr, idx_va = perm[:n_train], perm[n_train:]
    return {"x_train": x[idx_tr], "y_train": y[idx_tr],
            "x_val": x[idx_va], "y_val": y[idx_va],
            "g_train": g[idx_tr], "g_val": g[idx_va]}
