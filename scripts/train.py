"""Train a learner from the CLI and write a run folder.

Examples:
    python scripts/train.py --baseline --tag base
    python scripts/train.py --dml --epochs 300 --samples 200000
"""

import argparse
from datetime import datetime
from pathlib import Path

from dhps.train.trainer import TrainConfig, save_run, train_model

RUNS_ROOT = Path(__file__).resolve().parents[1] / "reports" / "runs"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--baseline", action="store_true",
                      help="values only, no differential signal")
    mode.add_argument("--dml", action="store_true",
                      help="differential ML (values + input gradients)")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--samples", type=int, default=100_000)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tag", default=None, help="subfolder name under reports/runs")
    args = ap.parse_args()

    differential = args.dml
    cfg = TrainConfig(n_samples=args.samples, epochs=args.epochs,
                      batch_size=args.batch, lr=args.lr, lam=args.lam,
                      seed=args.seed)
    result = train_model(cfg, differential=differential)

    tag = args.tag or ("dml" if differential else "baseline")
    run_dir = RUNS_ROOT / tag / datetime.now().strftime("%Y%m%d-%H%M%S")
    save_run(result, cfg, differential, run_dir)

    print(f"[{tag}] best epoch {result.best_epoch} of {cfg.epochs}, "
          f"{result.seconds:.1f}s")
    for key, value in result.metrics.items():
        print(f"  {key}: {value:.6g}")
    print(f"  run folder: {run_dir}")


if __name__ == "__main__":
    main()
