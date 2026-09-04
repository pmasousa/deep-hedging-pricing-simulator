"""Latency benchmark for the API at matched Monte Carlo error — Sprint C.

    uv run --extra api python scripts/benchmark_api.py --budget live

Times /price, /greeks, /hedge through an in-process ASGI client (the
full request stack minus network), then times Monte Carlo at the path
count its CLT error needs to match the SERVED model's price MAE — the
same matched-error doctrine as scripts/benchmark.py. Gates the spec:
/price p50 AND p99 must each beat Monte Carlo by >= 10x. Writes
reports/api/latency-<budget>.json.
"""

import argparse
import asyncio
import json
import math
import platform
import time
from datetime import datetime
from pathlib import Path

import httpx
from httpx import ASGITransport

from dhps.api import service
from dhps.api.app import create_app
from dhps.bench.speed import mc_paths_for_error, payoff_std, price_one_option_mc, time_fn

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "reports" / "api"

ATM = {"spot": 100.0, "strike": 100.0, "t_maturity": 1.0, "sigma": 0.2}
HEDGE = {"spot": 100.0, "strike": 100.0, "time_to_maturity": 0.5, "position": 0.5}


def _pct(sorted_us: list[float], q: float) -> float:
    """Nearest-rank percentile of an already-sorted list."""
    return sorted_us[min(len(sorted_us) - 1, math.ceil(q * len(sorted_us)) - 1)]


async def _time_endpoint(client: httpx.AsyncClient, path: str,
                         payload: dict, warmup: int, n: int) -> dict:
    for _ in range(warmup):
        r = await client.post(path, json=payload)
        r.raise_for_status()
    us = []
    for _ in range(n):
        t0 = time.perf_counter()
        r = await client.post(path, json=payload)
        r.raise_for_status()
        us.append((time.perf_counter() - t0) * 1e6)
    s = sorted(us)
    return {"p50_us": _pct(s, 0.50), "p99_us": _pct(s, 0.99), "n": n}


async def run(budget: str, n: int, warmup: int) -> dict:
    pricer, _ = service.get_models_for(budget)
    price_mae = pricer.metrics["price_mae"]

    transport = ASGITransport(app=create_app())
    async with httpx.AsyncClient(transport=transport,
                                 base_url="http://api") as client:
        rows = {
            "price": await _time_endpoint(client, "/price", ATM, warmup, n),
            "greeks": await _time_endpoint(client, "/greeks", ATM, warmup, n),
            "hedge": await _time_endpoint(client, "/hedge", HEDGE, warmup, n),
        }
        # determinism: identical requests must return identical bytes
        texts = [(await client.post("/greeks", json=ATM)).text for _ in range(2)]
        deterministic = texts[0] == texts[1]

    n_mc = mc_paths_for_error(payoff_std(), price_mae)
    mc_us = time_fn(lambda: price_one_option_mc(n_mc), repeats=3) * 1e6
    # gate like for like: /price latency vs matched-error MC PRICING.
    # greeks/hedge are reported without an MC counterpart — pathwise-MC
    # Greeks cost more than the MC price itself
    speedup = {"p50": mc_us / rows["price"]["p50_us"],
               "p99": mc_us / rows["price"]["p99_us"]}

    return {"budget": budget, "price_mae": price_mae, "mc_paths": n_mc,
            "mc_us_per_price": mc_us, "endpoints": rows,
            "speedup_vs_mc": speedup,
            "deterministic": deterministic,
            "host": platform.platform(),
            "method": "in-process ASGI (httpx), CPU; percentiles by "
                      "nearest rank; MC timed with repeats=3 at matched "
                      "CLT error"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--budget", default="full", choices=sorted(service.BUDGETS))
    ap.add_argument("--requests", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args()

    res = asyncio.run(run(args.budget, args.requests, args.warmup))

    print(f"budget {res['budget']}  (served model price MAE "
          f"{res['price_mae']:.4f}, matched MC = {res['mc_paths']:,} paths, "
          f"{res['mc_us_per_price']:,.0f} us/price)")
    for name, row in res["endpoints"].items():
        print(f"  /{name:6s} p50 {row['p50_us']:8.0f} us   "
              f"p99 {row['p99_us']:8.0f} us")
    print(f"gate: /price p50 {res['speedup_vs_mc']['p50']:,.0f}x, "
          f"p99 {res['speedup_vs_mc']['p99']:,.0f}x vs matched-error MC "
          f"{res['mc_us_per_price']:,.0f} us/price (spec: >= 10x both)")
    assert min(res["speedup_vs_mc"].values()) >= 10.0, res
    assert res["deterministic"], "identical requests returned different bytes"

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out = OUT_ROOT / f"latency-{args.budget}.json"
    payload = {"generated": datetime.now().isoformat(timespec="seconds"),
               **res}
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
