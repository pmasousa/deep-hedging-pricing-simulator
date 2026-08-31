"""Run the SP4 benchmark suite: accuracy (ID + OOD), Greeks quality, speed.

Writes results.json, summary.md, and PNG plots under reports/benchmarks/.
Optionally includes CUDA timings with --device cuda.

    python scripts/benchmark.py --samples 100000 --epochs 300
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from dhps.bench.evaluate import evaluate_learner, greeks_curve, ood_metrics  # noqa: E402
from dhps.bench.speed import (  # noqa: E402
    mc_paths_for_error,
    payoff_std,
    price_one_option_mc,
    time_fn,
)
from dhps.datasets.european import make_european_dataset  # noqa: E402
from dhps.train.trainer import TrainConfig, train_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "reports" / "benchmarks"


def run_accuracy(cfg: TrainConfig) -> tuple[dict, dict, dict]:
    data = make_european_dataset(n_samples=cfg.n_samples, seed=cfg.seed)
    dml = train_model(cfg, differential=True)
    base = train_model(cfg, differential=False)
    acc = {"dml": {**evaluate_learner(dml, data), **ood_metrics(dml)},
           "baseline": {**evaluate_learner(base, data), **ood_metrics(base)}}
    return acc, dml, base


def run_speed(dml_price_mae: float, devices: list[str]) -> list[dict]:
    rows = []
    p_std = payoff_std()
    n_matched = mc_paths_for_error(p_std, dml_price_mae)
    # MC reference is timed once on CPU — the simulator has no CUDA path here,
    # so a per-device MC row would be a mislabeled CPU timing
    t_mc = time_fn(lambda: price_one_option_mc(n_matched), repeats=3)
    rows.append({"device": "cpu", "method": f"monte carlo ({n_matched:,} paths, "
                 f"matched to {dml_price_mae:.4f} err)",
                 "seconds": t_mc, "us_per_price": t_mc * 1e6})
    for device in devices:
        x1 = torch.randn(1, 4, dtype=torch.float64, device=device)
        xbig = torch.randn(100_000, 4, dtype=torch.float64, device=device)
        model_like = torch.nn.Sequential(
            torch.nn.Linear(4, 64), torch.nn.SiLU(),
            torch.nn.Linear(64, 64), torch.nn.SiLU(),
            torch.nn.Linear(64, 64), torch.nn.SiLU(),
            torch.nn.Linear(64, 1)).to(torch.float64).to(device)
        with torch.no_grad():
            t1 = time_fn(lambda m=model_like, x=x1: m(x))
            tbig = time_fn(lambda m=model_like, x=xbig: m(x))
        rows += [
            {"device": device, "method": "dml forward (1 option)", "seconds": t1,
             "us_per_price": t1 * 1e6},
            {"device": device, "method": "dml forward (100k options)",
             "seconds": tbig, "us_per_price": tbig * 1e6 / 100_000},
        ]
    return rows


def write_plots(acc: dict, dml, base, out_dir: Path) -> None:
    spots = torch.linspace(50.0, 200.0, 301, dtype=torch.float64)
    curve_dml = greeks_curve(dml, spots, 100.0, 1.0, 0.2)
    curve_base = greeks_curve(base, spots, 100.0, 1.0, 0.2)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    names = ("delta", "gamma")
    titles = ("Delta vs spot (hero chart)", "Gamma vs spot — never in the labels")
    for ax, greek, title in zip(axes, names, titles, strict=True):
        ax.plot(spots, curve_dml[f"{greek}_true"].numpy(), "k--",
                label="analytic", linewidth=2)
        ax.plot(spots, curve_dml[greek].numpy(), label="DML", linewidth=1.5)
        ax.plot(spots, curve_base[greek].numpy(), label="baseline", linewidth=1.5,
                alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel("spot")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "greeks_curves.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples", type=int, default=100_000)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=8_192)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default=None,
                    help="extra device for speed rows (e.g. cuda)")
    args = ap.parse_args()

    cfg = TrainConfig(n_samples=args.samples, epochs=args.epochs,
                      batch_size=args.batch, seed=args.seed)
    print(f"training both learners ({args.samples:,} samples, {args.epochs} epochs)...")
    acc, dml, base = run_accuracy(cfg)
    for name in ("dml", "baseline"):
        print(f"  {name:9s} " + "  ".join(f"{k}={v:.5f}" for k, v in acc[name].items()))

    devices = ["cpu"] + ([args.device] if args.device else [])
    speed = run_speed(acc["dml"]["price_mae"], devices)

    out_dir = OUT_ROOT / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(
        {"config": vars(args), "accuracy": acc, "speed": speed}, indent=2))

    lines = ["# Benchmark results", "",
             "| learner | price MAE | delta MAE | OOD price MAE | OOD delta MAE |",
             "|---|---|---|---|---|"]
    for name in ("dml", "baseline"):
        m = acc[name]
        lines.append(f"| {name} | {m['price_mae']:.5f} | {m['delta_mae']:.5f} "
                     f"| {m['ood_price_mae']:.5f} | {m['ood_delta_mae']:.5f} |")
    lines += ["", "## Speed (µs per price)", "",
              "| device | method | µs/price |", "|---|---|---|"]
    for row in speed:
        lines.append(f"| {row['device']} | {row['method']} | "
                     f"{row['us_per_price']:.3g} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")

    write_plots(acc, dml, base, out_dir)
    print(f"results written to {out_dir}")


if __name__ == "__main__":
    main()
