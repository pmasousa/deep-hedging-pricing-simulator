"""Ablation study: entropic-lambda sweep and cost sweep (Sprint B).

Two questions, answered with the same shared eval paths and frozen
protocol:

1. Risk-measure knob — what does the entropic lambda buy? Higher lambda
   penalizes the loss tail harder: expect CVaR95 to improve and the mean
   to give something up.
2. Cost knob — where does deep hedging earn its keep? The cost sweep
   trains one policy per cost level and compares against weekly delta.
   Honest expectation from the dashboard: delta wins near zero cost,
   the policy's edge grows with the cost rate.

Outputs reports/ablations/<timestamp>/{results.json, ablations.png,
summary.md} plus a convenience copy at reports/ablations/latest.json.
~6-8 minutes on CPU (ten small policy trainings).
"""

import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from dhps.hedging.policy import (  # noqa: E402
    DeepHedgeConfig,
    delta_baseline_metrics,
    train_deep_hedge,
)

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "reports" / "ablations"

LAMBDAS = (0.25, 0.5, 1.0, 2.0, 4.0)
COSTS = (0.0, 0.0025, 0.005, 0.01, 0.02)


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


def main() -> None:
    out_dir = OUT_ROOT / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("lambda sweep (cost fixed at 1%)...")
    lam_results = {}
    for lam in LAMBDAS:
        lam_results[str(lam)] = train_and_report(lam, 0.01)
        m = lam_results[str(lam)]
        print(f"  lambda {lam:>4}: CVaR95 {m['policy']['cvar95']:+.2f}  "
              f"mean {m['policy']['mean']:+.2f}  "
              f"volume {m['policy']['traded_volume']:.1f}")

    print("cost sweep (lambda fixed at 1)...")
    cost_results = {}
    for cost in COSTS:
        cost_results[str(cost)] = train_and_report(1.0, cost)
        m = cost_results[str(cost)]
        edge = m["policy"]["cvar95"] - m["delta"]["cvar95"]
        print(f"  cost {cost * 100:>5.2f}%: policy CVaR95 "
              f"{m['policy']['cvar95']:+.2f}  delta {m['delta']['cvar95']:+.2f}"
              f"  edge {edge:+.2f}")

    results = {
        "lambdas": lam_results,
        "costs": cost_results,
        "protocol": {"n_paths": 12_288, "n_steps": 26, "epochs": 120,
                     "seed": 7, "eval_paths": 8_192, "eval_seed": 1_234},
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    (ROOT / "reports" / "ablations" / "latest.json").write_text(
        json.dumps(results, indent=2))

    # two-panel figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    lams = list(lam_results)
    cvars = [lam_results[k]["policy"]["cvar95"] for k in lams]
    means = [lam_results[k]["policy"]["mean"] for k in lams]
    x = np.arange(len(lams))
    axes[0].bar(x - 0.2, cvars, 0.4, label="CVaR95", color="#00CC96")
    axes[0].bar(x + 0.2, means, 0.4, label="mean", color="#636EFA")
    axes[0].set_xticks(x, [f"λ={k}" for k in lams])
    axes[0].set_title("Risk-measure knob (entropic λ, 1% costs)")
    axes[0].legend()
    for ax in axes:
        ax.axhline(0.0, color="#9ba1ad", lw=0.8)

    costs = list(cost_results)
    pc = [cost_results[k]["policy"]["cvar95"] for k in costs]
    dc = [cost_results[k]["delta"]["cvar95"] for k in costs]
    x = np.arange(len(costs))
    axes[1].bar(x - 0.2, pc, 0.4, label="deep hedge", color="#00CC96")
    axes[1].bar(x + 0.2, dc, 0.4, label="delta (weekly)", color="#636EFA")
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
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
