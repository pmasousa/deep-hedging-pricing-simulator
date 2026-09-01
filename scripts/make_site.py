"""Generate the static results site (index.html) for the gh-pages branch.

Reads the newest reports/benchmarks/*/results.json for the accuracy and
speed tables, embeds the run's greeks_curves.png, and trains the hedging
policy once to build the P&L distribution chart. Output is a single
self-contained HTML file (plotly from CDN).
"""

import base64
import json
from datetime import date
from pathlib import Path

import torch

from dhps.hedging.policy import DeepHedgeConfig, train_deep_hedge
from dhps.hedging.simulator import cvar, delta_positions, hedge_pnl, premium_bs
from dhps.simulators.gbm import simulate_gbm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".notes" / "site" / "index.html"

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, 'Segoe UI', Roboto, sans-serif;
       color: #1f2933; background: #fafafa; max-width: 1080px;
       margin: 0 auto; padding: 32px 24px 64px; }
h1 { font-size: 1.9rem; margin-bottom: 6px; }
h2 { font-size: 1.15rem; margin: 40px 0 12px; color: #1f2933; }
p.lead { color: #52606d; max-width: 46rem; margin-bottom: 8px; }
p.note { color: #7b8794; font-size: 0.85rem; margin-top: 8px; }
.cards { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px;
         margin: 20px 0 8px; }
.card { background: #fff; border: 1px solid #e4e7eb; border-radius: 8px;
        padding: 14px; }
.card .k { color: #7b8794; font-size: 0.75rem; text-transform: uppercase;
           letter-spacing: 0.04em; }
.card .v { font-size: 1.25rem; font-weight: 600; margin-top: 4px; }
.card .s { color: #00875a; font-size: 0.8rem; margin-top: 2px; }
table { border-collapse: collapse; background: #fff; width: 100%;
        border: 1px solid #e4e7eb; border-radius: 8px; overflow: hidden; }
th, td { text-align: left; padding: 8px 12px; font-size: 0.9rem; }
th { background: #f5f7fa; color: #52606d; font-weight: 600; }
td { border-top: 1px solid #e4e7eb; }
img.curve { width: 100%; border: 1px solid #e4e7eb; border-radius: 8px;
            background: #fff; }
.chart { height: 420px; }
@media (max-width: 800px) { .cards { grid-template-columns: 1fr 1fr; } }
"""


def load_latest_run() -> tuple[dict, Path]:
    runs = sorted((ROOT / "reports" / "benchmarks").glob("*"))
    if not runs:
        raise SystemExit("no benchmark runs under reports/benchmarks/ — run "
                         "scripts/benchmark.py first")
    run_dir = runs[-1]
    return json.loads((run_dir / "results.json").read_text()), run_dir


def hedging_data() -> dict:
    cfg = DeepHedgeConfig(n_paths=16_384, n_steps=26, cost_rate=0.01,
                          epochs=150, seed=7, eval_paths=8_192)
    res = train_deep_hedge(cfg)
    sim = dict(s0=cfg.s0, r=cfg.r, q=cfg.q, sigma=cfg.sigma,
               t_maturity=cfg.t_maturity)
    paths = simulate_gbm(n_paths=cfg.eval_paths, n_steps=cfg.n_steps,
                         antithetic=True, seed=cfg.eval_seed, **sim)
    premium = premium_bs(cfg.s0, cfg.strike, cfg.r, cfg.q, cfg.sigma,
                         cfg.t_maturity)
    delta = delta_positions(paths, cfg.strike, cfg.r, cfg.q, cfg.sigma,
                            cfg.t_maturity)
    zero = torch.zeros_like(delta)

    def pnl_of(pos):
        return hedge_pnl(paths, cfg.strike, premium, pos, cfg.cost_rate)

    out = {}
    for name, pnl in (("no hedge", pnl_of(zero)),
                      ("delta (weekly)", pnl_of(delta)),
                      ("deep hedge (policy)", res.eval_pnl)):
        out[name] = {"pnl": pnl[::4].tolist(),
                     "mean": float(pnl.mean()), "std": float(pnl.std()),
                     "cvar95": cvar(pnl)}
    return out


def main() -> None:
    run, run_dir = load_latest_run()
    acc = run["accuracy"]
    speed = run["speed"]
    hedging = hedging_data()
    png_b64 = base64.b64encode((run_dir / "greeks_curves.png").read_bytes())

    rows_acc = "".join(
        f"<tr><td>{name}</td>"
        f"<td>{acc[name]['price_mae']:.3f}</td>"
        f"<td>{acc[name]['delta_mae']:.4f}</td>"
        f"<td>{acc[name]['ood_price_mae']:.2f}</td>"
        f"<td>{acc[name]['ood_delta_mae']:.3f}</td></tr>"
        for name in ("dml", "baseline"))
    rows_speed = "".join(
        f"<tr><td>{r['device']}</td><td>{r['method']}</td>"
        f"<td>{r['us_per_price']:,.4g}</td></tr>" for r in speed)
    mc_us = next(r["us_per_price"] for r in speed
                 if r["method"].startswith("monte carlo"))
    dml_us = next(r["us_per_price"] for r in speed
                  if "100k" in r["method"] and r["device"] == "cpu")

    hed_rows = "".join(
        f"<tr><td>{n}</td><td>{v['mean']:+.2f}</td><td>{v['std']:.2f}</td>"
        f"<td>{v['cvar95']:.2f}</td></tr>" for n, v in hedging.items())

    violins = [{"type": "violin", "y": v["pnl"], "name": n,
                "box": {"visible": True}, "meanline": {"visible": True},
                "points": False} for n, v in hedging.items()]
    speed_trace = [{"type": "bar",
                    "x": [f"{r['method']} ({r['device']})" for r in speed],
                    "y": [r["us_per_price"] for r in speed],
                    "marker": {"color": ["#636efa", "#00cc96", "#ef553b"]}}]

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Deep Hedging &amp; Pricing Simulator — results</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>{CSS}</style></head><body>
<h1>Deep Hedging &amp; Pricing Simulator</h1>
<p class="lead">European option pricing and hedging in PyTorch. Greeks are
computed three independent ways (analytic Black-Scholes, autograd, pathwise
Monte Carlo) and the tests check they agree. On top: a differential ML
pricer (Huge &amp; Savine, 2020) and a deep hedging policy under transaction
costs (Buehler et al., 2019).</p>

<div class="cards">
<div class="card"><div class="k">Autograd vs analytic</div>
<div class="v">1e-13</div><div class="s">five Greeks, call and put</div></div>
<div class="card"><div class="k">DML price MAE</div>
<div class="v">{acc['dml']['price_mae']:.3f}</div>
<div class="s">{acc['baseline']['price_mae'] / acc['dml']['price_mae']:.1f}x
better than baseline</div></div>
<div class="card"><div class="k">Inference at matched error</div>
<div class="v">{mc_us / dml_us:,.0f}x</div>
<div class="s">vs Monte Carlo on CPU</div></div>
<div class="card"><div class="k">Deep hedging CVaR95</div>
<div class="v">{hedging['deep hedge (policy)']['cvar95']:.2f}</div>
<div class="s">vs {hedging['delta (weekly)']['cvar95']:.2f} weekly delta
at 1% costs</div></div>
</div>

<h2>Differential training vs price-only baseline</h2>
<p class="lead">Same network and data; the DML model also fits the price's
input-gradients. {run['config']['samples']:,} Sobol samples,
{run['config']['epochs']} epochs, float64. OOD = spots 25–45 and 210–240,
outside the training range.</p>
<table><tr><th>learner</th><th>price MAE</th><th>delta MAE</th>
<th>OOD price MAE</th><th>OOD delta MAE</th></tr>{rows_acc}</table>

<h2>Speed at matched error</h2>
<p class="lead">Monte Carlo is timed with the path count its CLT standard
error needs to match the learner's price accuracy.</p>
<table><tr><th>device</th><th>method</th><th>µs per price</th></tr>
{rows_speed}</table>
<div id="speed-chart" class="chart"></div>

<h2>Greeks from the trained network</h2>
<p class="lead">Delta and gamma vs spot against the analytic pricer. Gamma
is not a training label — it is a second autograd pass through the trained
network.</p>
<img class="curve" alt="delta and gamma curves vs analytic"
     src="data:image/png;base64,{png_b64.decode()}">

<h2>Deep hedging under 1% transaction costs</h2>
<p class="lead">Short one at-the-money call, stock traded at 26 dates. The
policy network is trained on entropic risk of hedging P&amp;L. A fairly
priced book has zero expected P&amp;L by construction; the improvement is
cost efficiency, not alpha.</p>
<table><tr><th>strategy</th><th>mean</th><th>std</th><th>CVaR95</th></tr>
{hed_rows}</table>
<div id="hedging-chart" class="chart"></div>

<p class="note">Generated {date.today().isoformat()} by
scripts/make_site.py from benchmark run {run_dir.name} (reproduce:
scripts/benchmark.py) and a seeded policy training run. Source:
github.com — see repository README.</p>

<script>
Plotly.newPlot('speed-chart', {json.dumps(speed_trace)},
  {{y:{{type:'log', title:'µs per price'}}, margin:{{t:20, b:120}},
    x:{{tickangle: -30}}}});
Plotly.newPlot('hedging-chart', {json.dumps(violins)},
  {{y:{{title: 'P&amp;L per year'}}, margin:{{t: 20}}, violingap: 0.3}});
</script>
</body></html>
"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
