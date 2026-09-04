"""Model registry and inference for the pricing/hedging API — Sprint C.

The API serves two frozen learners: the differential-ML pricer (Savine) and
the deep-hedging policy (Buehler). This module trains or loads them once
per budget and exposes single-option inference in market units.

Budgets: ``test`` (CI smoke), ``live`` (dashboard parity), ``full``
(benchmark headline numbers). Bundles persist under
``reports/api/<budget>/`` so restarts and Docker builds skip training;
each bundle records its config and refuses to load under a different one.

The training box (spot, moneyness, maturity, volatility) is enforced at
the API edge, not silently clamped: out-of-distribution degradation is
measured and real (reports/benchmarks), so the honest contract is a 422.
"""

import math
import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from dhps.bench.evaluate import evaluate_learner, ood_metrics, predict_raw
from dhps.datasets.european import make_european_dataset
from dhps.datasets.normalize import Standardizer
from dhps.hedging.policy import DeepHedgeConfig, HedgePolicy, train_deep_hedge
from dhps.models.mlp import make_mlp
from dhps.train.trainer import TrainConfig, train_model

ROOT = Path(__file__).resolve().parents[3]
CACHE_ROOT = Path(os.environ.get("DHPS_API_CACHE",
                                 str(ROOT / "reports" / "api")))

# dataset constants the learners were fit under — /meta discloses them
R_RATE = 0.05
Q_RATE = 0.01
OPTION_TYPE = "European call"

BOX = {"spot": (50.0, 200.0), "moneyness": (0.6, 1.4),
       "t_maturity": (0.1, 2.0), "sigma": (0.05, 0.6)}

BUDGETS: dict[str, dict] = {
    "test": {
        "pricer": dict(n_samples=2_000, hidden=(32, 32), epochs=40,
                       batch_size=512, seed=7),
        "policy": dict(n_paths=1_024, n_steps=26, cost_rate=0.01, epochs=10,
                       seed=7, eval_paths=1_024),
    },
    "live": {
        "pricer": dict(n_samples=20_000, hidden=(64, 64, 64), epochs=300,
                       batch_size=8_192, seed=7),
        "policy": dict(n_paths=8_192, n_steps=26, cost_rate=0.01, epochs=80,
                       seed=7, eval_paths=4_096),
    },
    "full": {
        "pricer": dict(n_samples=100_000, epochs=300, batch_size=8_192,
                       seed=7),
        "policy": dict(n_paths=32_768, n_steps=26, cost_rate=0.01,
                       epochs=200, seed=7, eval_paths=8_192),
    },
}


@dataclass(frozen=True)
class PricerBundle:
    """A trained pricer in market coordinates; duck-types ``TrainResult``
    for ``predict_raw`` / ``evaluate_learner``."""

    model: torch.nn.Module
    scaler: Standardizer
    metrics: dict[str, float]
    config: dict


@dataclass(frozen=True)
class PolicyBundle:
    policy: HedgePolicy
    metrics: dict[str, float]
    config: dict


def _save_pricer(bundle: PricerBundle, budget: str) -> None:
    out = CACHE_ROOT / budget
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"config": bundle.config, "metrics": bundle.metrics,
                "state": bundle.model.state_dict(),
                "scaler": {k: getattr(bundle.scaler, k)
                           for k in ("x_mean", "x_std", "y_mean", "y_std")}},
               out / "pricer.pt")


def _load_pricer(budget: str) -> PricerBundle | None:
    path = CACHE_ROOT / budget / "pricer.pt"
    if not path.exists():
        return None
    blob = torch.load(path, weights_only=True)
    if blob["config"] != asdict(TrainConfig(**BUDGETS[budget]["pricer"])):
        return None  # bundle from a different config: retrain
    cfg = TrainConfig(**blob["config"])
    model = make_mlp(n_in=4, hidden=cfg.hidden, activation=cfg.activation)
    model.load_state_dict(blob["state"])
    model.eval()
    return PricerBundle(model=model, scaler=Standardizer(**blob["scaler"]),
                        metrics=blob["metrics"], config=blob["config"])


def _save_policy(bundle: PolicyBundle, budget: str) -> None:
    out = CACHE_ROOT / budget
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"config": bundle.config, "metrics": bundle.metrics,
                "state": bundle.policy.state_dict()},
               out / "policy.pt")


def _load_policy(budget: str) -> PolicyBundle | None:
    path = CACHE_ROOT / budget / "policy.pt"
    if not path.exists():
        return None
    blob = torch.load(path, weights_only=True)
    if blob["config"] != asdict(DeepHedgeConfig(**BUDGETS[budget]["policy"])):
        return None
    policy = HedgePolicy(tuple(DeepHedgeConfig(**blob["config"]).hidden))
    policy.load_state_dict(blob["state"])
    policy.eval()
    return PolicyBundle(policy=policy, metrics=blob["metrics"],
                        config=blob["config"])


def _train_pricer(budget: str) -> PricerBundle:
    cfg = TrainConfig(**BUDGETS[budget]["pricer"])
    res = train_model(cfg, differential=True)
    data = make_european_dataset(n_samples=cfg.n_samples, seed=cfg.seed)
    metrics = {**evaluate_learner(res, data), **ood_metrics(res)}
    return PricerBundle(model=res.model.eval(), scaler=res.scaler,
                        metrics=metrics, config=asdict(cfg))


def _train_policy(budget: str) -> PolicyBundle:
    cfg = DeepHedgeConfig(**BUDGETS[budget]["policy"])
    res = train_deep_hedge(cfg)
    return PolicyBundle(policy=res.policy.eval(), metrics=res.metrics,
                        config=asdict(cfg))


_cache: dict[str, tuple[PricerBundle, PolicyBundle]] = {}
_lock = threading.Lock()


def active_budget() -> str:
    budget = os.environ.get("DHPS_API_BUDGET", "live")
    if budget not in BUDGETS:
        raise KeyError(f"unknown DHPS_API_BUDGET {budget!r}; "
                       f"choose from {sorted(BUDGETS)}")
    return budget


def get_models_for(budget: str) -> tuple[PricerBundle, PolicyBundle]:
    """Load-or-train both learners for ``budget``; cached in-process."""
    if budget not in BUDGETS:
        raise KeyError(f"unknown budget {budget!r}; choose from {sorted(BUDGETS)}")
    with _lock:
        if budget not in _cache:
            pricer = _load_pricer(budget)
            if pricer is None:
                pricer = _train_pricer(budget)
                _save_pricer(pricer, budget)
            policy = _load_policy(budget)
            if policy is None:
                policy = _train_policy(budget)
                _save_policy(policy, budget)
            _cache[budget] = (pricer, policy)
        return _cache[budget]


def get_models() -> tuple[PricerBundle, PolicyBundle]:
    return get_models_for(active_budget())


def validate_pricer_box(spot: float, strike: float, t_maturity: float,
                        sigma: float) -> None:
    """Raise ValueError when a request leaves the training box."""
    m = strike / spot
    checks = (
        (BOX["spot"][0] <= spot <= BOX["spot"][1],
         f"spot outside training box {BOX['spot']}: {spot:g}"),
        (BOX["moneyness"][0] <= m <= BOX["moneyness"][1],
         f"moneyness strike/spot outside training box {BOX['moneyness']}: "
         f"{m:.3g}"),
        (BOX["t_maturity"][0] <= t_maturity <= BOX["t_maturity"][1],
         f"t_maturity outside training box {BOX['t_maturity']}: {t_maturity:g}"),
        (BOX["sigma"][0] <= sigma <= BOX["sigma"][1],
         f"sigma outside training box {BOX['sigma']}: {sigma:g}"),
    )
    for ok, msg in checks:
        if not ok:
            raise ValueError(msg)


def price_one(spot: float, strike: float, t_maturity: float,
              sigma: float) -> float:
    pricer, _ = get_models()
    x = torch.tensor([[spot, strike, t_maturity, sigma]],
                     dtype=torch.float64)
    price, _ = predict_raw(pricer, x, with_grad=False)
    return float(price[0, 0])


def greeks_one(spot: float, strike: float, t_maturity: float,
               sigma: float) -> dict[str, float]:
    """Price and Greeks by one/two autograd passes through the trained net.

    Gamma was never a label — the second pass measures what the network
    internalized (same math as ``bench.evaluate.greeks_curve``).
    """
    pricer, _ = get_models()
    x = torch.tensor([[spot, strike, t_maturity, sigma]],
                     dtype=torch.float64)
    scaler = pricer.scaler
    z = scaler.transform_x(x).requires_grad_(True)
    y = pricer.model(z)
    g = torch.autograd.grad(y.sum(), z, create_graph=True)[0]
    hess = torch.autograd.grad(g[:, 0].sum(), z)[0]
    gs = scaler.grad_scale[0]
    price = float((y[0, 0] * scaler.y_std[0] + scaler.y_mean[0]).detach())
    return {
        "price": price,
        "delta": float((g[0, 0] / gs[0]).detach()),
        "dual_delta": float((g[0, 1] / gs[1]).detach()),
        # repo convention: theta = -d(price)/d(t_maturity), per year
        "theta": float(-(g[0, 2] / gs[2]).detach()),
        "vega": float((g[0, 3] / gs[3]).detach()),
        "gamma": float((hess[0, 0] * scaler.y_std[0]
                        / scaler.x_std[0] ** 2).detach()),
    }


def hedge_one(spot: float, strike: float, time_to_maturity: float,
              position: float) -> float:
    """Target stock position for the short-call book, in [0, 1]."""
    _, policy = get_models()
    with torch.no_grad():
        target = policy.policy(
            torch.tensor([math.log(spot / strike)], dtype=torch.float64),
            torch.tensor([time_to_maturity], dtype=torch.float64),
            torch.tensor([position], dtype=torch.float64))
    return float(target[0])


def meta() -> dict:
    pricer, policy = get_models()
    return {
        "budget": active_budget(),
        "option": OPTION_TYPE,
        "constants": {"r": R_RATE, "q": Q_RATE,
                      "note": "rates are dataset constants the learners "
                              "were fit under; not request parameters"},
        "pricer": {"config": pricer.config, "metrics": pricer.metrics},
        "hedge_policy": {"config": policy.config, "metrics": policy.metrics},
        "input_box": {**BOX,
                      "note": "requests outside the box get 422 — OOD error "
                              "is measured, not clamped"},
        "hedge_grid": {"time_to_maturity": "(0, 1]", "position": "[0, 1]",
                       "note": "policy trained on 26 rebalance dates, "
                               "GBM sigma 0.20, 1% costs; evaluated "
                               "zero-shot on Heston in the test suite"},
    }
