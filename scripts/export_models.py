"""Train and cache the API models for one budget — Sprint C.

    uv run python scripts/export_models.py --budget full

Writes reports/api/<budget>/{pricer.pt, policy.pt} so the API (and the
Docker image, which runs this at build time) starts warm. Re-runs reuse
an existing matching bundle unless --force.
"""

import argparse

from dhps.api import service


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--budget", default="full", choices=sorted(service.BUDGETS))
    ap.add_argument("--force", action="store_true",
                    help="retrain even if a matching bundle exists")
    args = ap.parse_args()

    if args.force:
        for name in ("pricer.pt", "policy.pt"):
            path = service.CACHE_ROOT / args.budget / name
            if path.exists():
                path.unlink()

    pricer, policy = service.get_models_for(args.budget)
    print(f"budget {args.budget!r} -> {service.CACHE_ROOT / args.budget}")
    print("pricer metrics:",
          {k: round(v, 5) for k, v in pricer.metrics.items()})
    print("policy metrics:",
          {k: round(v, 3) for k, v in policy.metrics.items()})


if __name__ == "__main__":
    main()
