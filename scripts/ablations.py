"""Ablation study: risk knob, cost knob, rebalance frequency, features.

Four sweeps, one frozen protocol (12,288 paths, 26 steps, 120 epochs,
seed 7):

1. lambda sweep — entropic risk aversion at 1% costs
2. cost sweep — policy vs weekly delta across cost rates (lambda 1)
3. rebalance frequency — frozen policy and delta at m in {8..104}
4. feature ablation — does an observable trailing-realized-vol feature
   close the vol-shock loss documented in walk_forward.py?

Outputs reports/ablations/<timestamp>/{results.json, ablations.png,
summary.md} plus a convenience copy at reports/ablations/latest.json.
~10 minutes on CPU (eleven policy trainings).
"""

import json
import math
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from dhps.hedging.heston import heston_premium_mc, variance_aware_delta_positions  # noqa: E402
from dhps.hedging.policy import (  # noqa: E402
    DeepHedgeConfig,
    delta_baseline_metrics,  # noqa: E402
    run_policy,  # noqa: E402
    train_deep_hedge,
)
from dhps.hedging.simulator import cvar, delta_positions, hedge_pnl, premium_bs  # noqa: E402
from dhps.simulators.gbm import simulate_gbm  # noqa: E402
from dhps.simulators.heston import simulate_heston  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "reports" / "ablations"

LAMBDAS = (0.25, 0.5, 1.0, 2.0, 4.0)
COSTS = (0.0, 0.0025, 0.005, 0.01, 0.02)
FREQS = (8, 13, 26, 52, 104)


def trailing_rvol(paths: torch.Tensor, t_maturity: float,
                  window: int = 8) -> torch.Tensor:
    """Causal annualized realized vol observable at each decision date.

    Column j uses only returns realized strictly before decision j (no
    lookahead); the first dates carry the training-sigma prior.
    """
    m = paths.shape[1] - 1
    dt = t_maturity / m
    rets = torch.log(paths[:, 1:] / paths[:, :-1])
    out = torch.full((paths.shape[0], m), 0.2, dtype=paths.dtype)
    for j in range(2, m):
        seg = rets[:, max(0, j - window):j]
        if seg.shape[1] >= 2:
            out[:, j] = seg.std(dim=1, unbiased=True) * (1.0 / dt) ** 0.5
    return out


def train_and_report(lambd: float, cost: float) -> dict:
    cfg = DeepHedgeConfig(n_paths=12_288, n_steps=26, cost_rate=cost,
                          lambd=lambd, epochs=120, seed=7, eval_paths=8_192)
    res = train_deep_hedge(cfg)
    base = delta_baseline_metrics(cfg)
    return {
        "policy": res.metrics,
        "delta": base,
        "train_risk_first": res.history["train_risk"][0],
        "train_risk_last": res.history["train_risk"][-1],
    }


def sweep_lambda() -> dict:
    print("lambda sweep (cost fixed at 1%)...")
    out = {}
    for lam in LAMBDAS:
        out[str(lam)] = train_and_report(lam, 0.01)
        m = out[str(lam)]["policy"]
        print(f"  lambda {lam:>4}: CVaR95 {m['cvar95']:+.2f}  "
              f"mean {m['mean']:+.2f}  volume {m['traded_volume']:.1f}")
    return out


def sweep_costs() -> dict:
    print("cost sweep (lambda fixed at 1)...")
    out = {}
    for cost in COSTS:
        out[str(cost)] = train_and_report(1.0, cost)
        p, d = out[str(cost)]["policy"], out[str(cost)]["delta"]
        print(f"  cost {cost * 100:>5.2f}%: policy CVaR95 {p['cvar95']:+.2f}"
              f"  delta {d['cvar95']:+.2f}  "
              f"edge {p['cvar95'] - d['cvar95']:+.2f}")
    return out


def sweep_frequency(policy) -> list[dict]:
    print("rebalance-frequency sweep (frozen policy, zero-shot)...")
    rows = []
    for m in FREQS:
        paths = simulate_gbm(n_paths=12_288, n_steps=m, s0=100.0, r=0.05,
                             q=0.01, sigma=0.20, t_maturity=1.0,
                             antithetic=True, seed=31)
        premium = premium_bs(100.0, 100.0, 0.05, 0.01, 0.20, 1.0)
        delta = delta_positions(paths, 100.0, 0.05, 0.01, 0.20, 1.0)
        with torch.no_grad():
            pos_policy = run_policy(policy, paths, 100.0, 1.0)
        pnl_p = hedge_pnl(paths, 100.0, premium, pos_policy, 0.01)
        pnl_d = hedge_pnl(paths, 100.0, premium, delta, 0.01)
        rows.append({"m": m, "policy_cvar": cvar(pnl_p),
                     "delta_cvar": cvar(pnl_d),
                     "policy_std": float(pnl_p.std()),
                     "delta_std": float(pnl_d.std())})
        print(f"  m={m:>3}: policy CVaR95 {rows[-1]['policy_cvar']:+.2f}  "
              f"delta {rows[-1]['delta_cvar']:+.2f}")
    return rows


def sweep_features(cost: float = 0.01) -> dict:
    """Can an observable vol feature close the vol-shock loss?

    Trains a 4-feature policy (adds trailing realized vol) on the same
    GBM regime as the base policy, then compares base vs vol-aware vs
    delta across the three walk-forward windows. Falsifiable prediction:
    the vol-aware policy improves the vol-shock window relative to base.
    """
    print("feature ablation (base policy + trailing-rvol twin)...")
    from dhps.models.mlp import make_mlp

    def run4(policy, paths, rvol, strike, t_maturity):
        m = paths.shape[1] - 1
        position = torch.zeros(paths.shape[0], dtype=paths.dtype)
        out = []
        for j in range(m):
            ttm = t_maturity * (1.0 - j / m)
            x = torch.stack([torch.log(paths[:, j] / strike),
                             torch.full_like(paths[:, j], ttm),
                             position, rvol[:, j]], dim=1)
            position = torch.sigmoid(policy(x)).squeeze(1)
            out.append(position)
        return torch.stack(out, dim=1)

    torch.manual_seed(7)
    pol4 = make_mlp(n_in=4, hidden=(32, 32), activation="silu")
    opt = torch.optim.Adam(pol4.parameters(), lr=1e-2)  # matches the DeepHedgeConfig default
    tpaths = simulate_gbm(n_paths=12_288, n_steps=26, s0=100.0, r=0.05,
                          q=0.01, sigma=0.20, t_maturity=1.0,
                          antithetic=True, seed=7)
    t_rvol = trailing_rvol(tpaths, 1.0) / 0.2  # normalized: O(1) around the training sigma
    # the objective must carry the FULL book (premium - payoff): without
    # it, any position is a mean-zero spread that only adds risk, and the
    # risk-optimal policy collapses to doing nothing
    t_premium = premium_bs(100.0, 100.0, 0.05, 0.01, 0.20, 1.0)
    t_payoff = torch.clamp(tpaths[:, -1] - 100.0, min=0.0)
    for _epoch in range(120):
        position = torch.zeros(tpaths.shape[0], dtype=tpaths.dtype)
        schedule = []
        pnls = torch.zeros_like(position)
        for j in range(26):
            ttm = 1.0 * (1.0 - j / 26)
            x = torch.stack([torch.log(tpaths[:, j] / 100.0),
                             torch.full_like(tpaths[:, j], ttm),
                             position, t_rvol[:, j]], dim=1)
            position = torch.sigmoid(pol4(x)).squeeze(1)
            schedule.append(position)
            pnls += position * (tpaths[:, j + 1] - tpaths[:, j])
        pos2d = torch.stack(schedule, dim=1)  # (n, m)
        costs = 0.01 * torch.cat(
            [pos2d[:, :1], pos2d[:, 1:] - pos2d[:, :-1]],
            dim=1).abs().mul(tpaths[:, :-1]).sum(dim=1)
        terminal = t_premium - t_payoff + pnls - costs
        loss = (torch.logsumexp(-terminal, dim=0)
                - math.log(float(terminal.shape[0])))
        opt.zero_grad()
        loss.backward()
        opt.step()

    def gbm_window(sigma: float):
        paths = simulate_gbm(n_paths=12_288, n_steps=26, s0=100.0, r=0.05,
                             q=0.01, sigma=sigma, t_maturity=1.0,
                             antithetic=True, seed=11)
        return paths, None

    def heston_window():
        return simulate_heston(n_paths=12_288, n_steps=26, s0=100.0, r=0.05,
                               q=0.01, t_maturity=1.0, antithetic=True,
                               seed=11, v0=0.09, kappa=2.0, theta=0.04,
                               xi=0.3, rho=-0.7)

    windows = [
        ("training regime (σ 0.20)", *gbm_window(0.20), 0.20),
        ("vol shock (σ 0.30)", *gbm_window(0.30), 0.30),
        ("Heston (ρ −0.7)", *heston_window(), None),
    ]
    base_res = train_deep_hedge(DeepHedgeConfig(
        n_paths=12_288, n_steps=26, cost_rate=cost, lambd=1.0, epochs=120,
        seed=7, eval_paths=8_192))

    out = {"windows": []}
    for name, paths, variances, sigma in windows:
        if variances is None:  # GBM window: classical BS delta, BS premium
            delta = delta_positions(paths, 100.0, 0.05, 0.01, sigma, 1.0)
            premium = premium_bs(100.0, 100.0, 0.05, 0.01, sigma, 1.0)
        else:  # Heston window: variance-aware delta, Monte Carlo premium
            delta = variance_aware_delta_positions(paths, variances, 100.0,
                                                   0.05, 0.01, 1.0)
            premium = heston_premium_mc()
        rvol = trailing_rvol(paths, 1.0) / 0.2  # same normalization as training
        with torch.no_grad():
            pos_base = run_policy(base_res.policy, paths, 100.0, 1.0)
            pos_vol = run4(pol4, paths, rvol, 100.0, 1.0)
        pnl_base = hedge_pnl(paths, 100.0, premium, pos_base, cost)
        pnl_vol = hedge_pnl(paths, 100.0, premium, pos_vol, cost)
        pnl_delta = hedge_pnl(paths, 100.0, premium, delta, cost)
        row = {"window": name,
               "base_cvar": cvar(pnl_base), "vol_cvar": cvar(pnl_vol),
               "delta_cvar": cvar(pnl_delta),
               "base_mean": float(pnl_base.mean()),
               "vol_mean": float(pnl_vol.mean())}
        out["windows"].append(row)
        print(f"  {name}: base {row['base_cvar']:+.2f}  "
              f"vol-aware {row['vol_cvar']:+.2f}  "
              f"delta {row['delta_cvar']:+.2f}")
    return out


def main() -> None:
    out_dir = OUT_ROOT / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    lam_results = sweep_lambda()
    cost_results = sweep_costs()

    print("lambda/cost policies trained — frequency sweep next...")
    freq_policy_cfg = DeepHedgeConfig(n_paths=12_288, n_steps=26,
                                      cost_rate=0.01, lambd=1.0, epochs=120,
                                      seed=7, eval_paths=8_192)
    freq_policy = train_deep_hedge(freq_policy_cfg).policy
    freq_results = sweep_frequency(freq_policy)

    feature_results = sweep_features()

    results = {
        "lambdas": lam_results,
        "costs": cost_results,
        "frequency": freq_results,
        "features": feature_results,
        "protocol": {"n_paths": 12_288, "n_steps": 26, "epochs": 120,
                     "seed": 7, "eval_paths": 8_192, "eval_seed": 1_234},
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    (ROOT / "reports" / "ablations" / "latest.json").write_text(
        json.dumps(results, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    lams = list(lam_results)
    x = np.arange(len(lams))
    axes[0].bar(x - 0.2, [lam_results[k]["policy"]["cvar95"] for k in lams],
                0.4, label="CVaR95", color="#00CC96")
    axes[0].bar(x + 0.2, [lam_results[k]["policy"]["mean"] for k in lams],
                0.4, label="mean", color="#636EFA")
    axes[0].set_xticks(x, [f"λ={k}" for k in lams])
    axes[0].set_title("Risk-measure knob (entropic λ, 1% costs)")
    axes[0].legend()
    for ax in axes:
        ax.axhline(0.0, color="#9ba1ad", lw=0.8)

    costs = list(cost_results)
    x = np.arange(len(costs))
    axes[1].bar(x - 0.2, [cost_results[k]["policy"]["cvar95"] for k in costs],
                0.4, label="deep hedge", color="#00CC96")
    axes[1].bar(x + 0.2, [cost_results[k]["delta"]["cvar95"] for k in costs],
                0.4, label="delta (weekly)", color="#636EFA")
    axes[1].set_xticks(x, [f"{float(k) * 100:.2f}%" for k in costs])
    axes[1].set_title("Cost sweep — CVaR95 (higher is safer)")
    axes[1].set_xlabel("transaction cost rate")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "ablations.png", dpi=150)
    plt.close(fig)

    lines = ["# Ablations", "",
             "## Entropic lambda (cost 1%)", "",
             "| lambda | CVaR95 | mean | std | volume |", "|---|---|---|---|---|"]
    for k in lams:
        m = lam_results[k]["policy"]
        lines.append(f"| {k} | {m['cvar95']:.2f} | {m['mean']:+.2f} | "
                     f"{m['std']:.2f} | {m['traded_volume']:.1f} |")
    lines += ["", "## Cost sweep (lambda 1)", "",
              "| cost | policy CVaR95 | delta CVaR95 | edge | policy mean | "
              "delta mean |", "|---|---|---|---|---|---|"]
    for k in costs:
        p, d = cost_results[k]["policy"], cost_results[k]["delta"]
        lines.append(f"| {float(k) * 100:.2f}% | {p['cvar95']:.2f} | "
                     f"{d['cvar95']:.2f} | "
                     f"{p['cvar95'] - d['cvar95']:+.2f} | {p['mean']:+.2f} | "
                     f"{d['mean']:+.2f} |")
    lines += ["", "## Rebalance frequency (frozen policy, 1% costs)", "",
              "| m | policy CVaR95 | delta CVaR95 | policy std | delta std |",
              "|---|---|---|---|---|"]
    for r in freq_results:
        lines.append(f"| {r['m']} | {r['policy_cvar']:.2f} | "
                     f"{r['delta_cvar']:.2f} | {r['policy_std']:.2f} | "
                     f"{r['delta_std']:.2f} |")
    lines += ["", "## Feature ablation — trailing realized vol", "",
              "| window | base policy | vol-aware policy | delta |",
              "|---|---|---|---|"]
    for w in feature_results["windows"]:
        lines.append(f"| {w['window']} | {w['base_cvar']:.2f} | "
                     f"{w['vol_cvar']:.2f} | {w['delta_cvar']:.2f} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
