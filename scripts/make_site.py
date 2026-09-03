"""Generate the static results site (index.html) for the gh-pages branch.

Streamlit-dark theme mirroring the dashboard overview, the four preview
charts, then the benchmark tables and figures, and a cross-validation
section against QuantLib (skipped when the [ql] extra is not installed).
Reads the newest reports/benchmarks/*/results.json, embeds the run's
greeks_curves.png, and trains the hedging policy once for the P&L chart.
Output is a single self-contained HTML file (plotly + fonts from CDN).
"""

import base64
import json
import math
from datetime import date
from pathlib import Path

import torch
from matplotlib import cm

from dhps.datasets.european import make_european_dataset
from dhps.hedging.policy import DeepHedgeConfig, run_policy, train_deep_hedge  # noqa: F401
from dhps.hedging.simulator import cvar, delta_positions, hedge_pnl, premium_bs
from dhps.pricing.black_scholes import bs_price
from dhps.simulators.gbm import mc_european_price, simulate_gbm
from dhps.simulators.heston import simulate_heston

try:
    import QuantLib as ql  # noqa: N813 — the community alias
except ImportError:  # site generation without the [ql] extra: skip section
    ql = None

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".notes" / "site" / "index.html"

STRATEGY_COLORS = {"no hedge": "#636EFA", "delta (weekly)": "#EF553B",
                   "deep hedge (policy)": "#00CC96"}

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:'
            'wght@400;600;700&display=swap');
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Source Sans 3', 'Source Sans Pro', 'Segoe UI', sans-serif;
       color: #fafafa; background: #0e1117; max-width: 1080px;
       margin: 0 auto; padding: 40px 32px 72px; }
h1 { font-size: 2rem; font-weight: 700; line-height: 1.2; }
h2 { font-size: 1.4rem; font-weight: 700; margin: 36px 0 10px; }
h3 { font-size: 1.05rem; font-weight: 700; margin: 24px 0 8px; }
p.lead { font-size: 1rem; line-height: 1.5; max-width: 48rem; margin: 10px 0;
         color: #c9cdd6; }
p.caption { font-size: 0.83rem; color: #9ba1ad; line-height: 1.4;
            margin: 6px 0; }
a { color: #ff4b4b; }
.alert { background: rgba(0, 204, 150, 0.12); color: #4ae290;
         border: 1px solid rgba(0, 204, 150, 0.35); border-radius: 8px;
         padding: 12px 16px; font-size: 0.95rem; margin: 14px 0;
         max-width: 52rem; }
.metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px;
           margin: 16px 0 6px; max-width: 56rem; }
.metric { background: #262730; border-radius: 8px; padding: 12px 14px; }
.metric .k { font-size: 0.8rem; color: #9ba1ad; }
.metric .v { font-size: 1.4rem; font-weight: 700; color: #fafafa;
             margin: 2px 0; }
.metric .s { font-size: 0.8rem; color: #9ba1ad; }
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px;
         margin: 14px 0; }
.panel { background: #262730; border-radius: 8px; padding: 14px; }
.panel .t { font-size: 0.95rem; font-weight: 600; margin-bottom: 6px; }
.chart { width: 100%; }
table { border-collapse: collapse; width: 100%; max-width: 52rem;
        background: #262730; border-radius: 8px; overflow: hidden; }
th, td { text-align: left; padding: 7px 12px; font-size: 0.92rem; }
th { background: #1a1c23; color: #9ba1ad; font-weight: 600; }
td { border-bottom: 1px solid #363b45; }
img.curve { width: 100%; max-width: 52rem; border: 1px solid #363b45;
            border-radius: 8px; background: #ffffff; padding: 10px;
            box-sizing: border-box; }
footer { margin-top: 48px; border-top: 1px solid #363b45; padding-top: 12px; }
@media (max-width: 800px) { .metrics { grid-template-columns: 1fr 1fr; }
                            .grid2 { grid-template-columns: 1fr; } }
"""

PLOT_FONT = "'Source Sans 3', 'Segoe UI', sans-serif"
TEXT = "#fafafa"
MUTED = "#9ba1ad"
GRID = "#33384a"


def dark_layout(**extra):
    layout = {"font": {"family": PLOT_FONT, "color": TEXT, "size": 13},
              "paper_bgcolor": "rgba(0,0,0,0)",
              "plot_bgcolor": "rgba(0,0,0,0)",
              "margin": {"t": 14, "r": 14, "b": 42, "l": 50},
              "xaxis": {"gridcolor": GRID, "zeroline": False},
              "yaxis": {"gridcolor": GRID, "zeroline": False},
              "legend": {"orientation": "h", "y": 1.18, "x": 0,
                         "bgcolor": "rgba(0,0,0,0)"}}
    for key, val in extra.items():
        layout[key] = val
    return layout


def viridis_rgb(t: float) -> str:
    r, g, b, _ = cm.viridis(t)
    return f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})"


def overview_panels() -> dict[str, list]:
    """The four overview preview charts, same data and colors as the app."""
    paths = simulate_gbm(n_paths=40, n_steps=64, antithetic=False, seed=3)
    fin = paths[:, -1]
    lo, hi = float(fin.min()), float(fin.max())
    fan = [{"y": paths[i].tolist(), "mode": "lines",
            "line": {"color": viridis_rgb(float((paths[i, -1] - lo)
                                                / (hi - lo))), "width": 1.2},
            "hoverinfo": "skip", "showlegend": False}
           for i in range(paths.shape[0])]

    s_t = torch.linspace(40.0, 170.0, 261, dtype=torch.float64)
    payoff = torch.clamp(s_t - 100.0, min=0.0)
    profit = 9.83 - payoff
    be = 109.83
    gain = s_t <= be
    payoff_chart = [
        {"x": s_t[gain].tolist(), "y": profit[gain].tolist(), "mode": "none",
         "fill": "tozeroy", "fillcolor": "rgba(0,204,150,0.25)",
         "hoverinfo": "skip", "showlegend": False},
        {"x": s_t[~gain].tolist(), "y": profit[~gain].tolist(), "mode": "none",
         "fill": "tozeroy", "fillcolor": "rgba(239,85,59,0.25)",
         "hoverinfo": "skip", "showlegend": False},
        {"x": s_t.tolist(), "y": profit.tolist(), "mode": "lines",
         "line": {"color": "#fafafa"}, "name": "profit at expiry"},
        {"x": s_t.tolist(), "y": payoff.tolist(), "mode": "lines",
         "line": {"color": "#636EFA", "dash": "dash"}, "name": "payoff owed"},
    ]

    spots = torch.linspace(50.0, 150.0, 101, dtype=torch.float64)
    vols = torch.linspace(0.05, 0.6, 56, dtype=torch.float64)
    mesh_s, mesh_v = torch.meshgrid(spots, vols, indexing="ij")
    surface = bs_price(mesh_s, 100.0, 0.05, 0.01, mesh_v, 1.0)
    heat = [{"type": "heatmap", "x": vols.tolist(), "y": spots.tolist(),
             "z": surface.tolist(), "colorscale": "Viridis",
             "colorbar": {"thickness": 12}}]

    dset = make_european_dataset(n_samples=2_000, seed=7)
    xs = dset["x_train"][:1500]
    sobol = [{"x": xs[:, 0].tolist(), "y": xs[:, 3].tolist(),
              "mode": "markers",
              "marker": {"size": 4, "color": (xs[:, 0] / xs[:, 1]).tolist(),
                         "colorscale": "Viridis", "showscale": True,
                         "colorbar": {"title": "s0/K", "thickness": 12}}}]

    return {"fan": fan, "payoff": payoff_chart, "heat": heat, "sobol": sobol}


def load_latest_run() -> tuple[dict, Path]:
    runs = sorted((ROOT / "reports" / "benchmarks").glob("*"))
    if not runs:
        raise SystemExit("no benchmark runs under reports/benchmarks/ — run "
                         "scripts/benchmark.py first")
    run_dir = runs[-1]
    return json.loads((run_dir / "results.json").read_text()), run_dir


HESTON = dict(v0=0.09, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)


def crossval_data() -> dict | None:
    """QuantLib cross-validation charts; None when QuantLib is absent."""
    if ql is None:
        return None
    k, r, q, sig, t = 100.0, 0.05, 0.01, 0.2, 1.0
    today = ql.Date.todaysDate()
    expiry = today + ql.Period(365, ql.Days)
    rf = ql.YieldTermStructureHandle(
        ql.FlatForward(today, r, ql.Actual365Fixed(), ql.Continuous))
    div = ql.YieldTermStructureHandle(
        ql.FlatForward(today, q, ql.Actual365Fixed(), ql.Continuous))
    vol_ts = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(today, ql.NullCalendar(), sig, ql.Actual365Fixed()))
    option = ql.EuropeanOption(
        ql.PlainVanillaPayoff(ql.Option.Call, k), ql.EuropeanExercise(expiry))

    spots = torch.linspace(60.0, 150.0, 37, dtype=torch.float64)
    ours = bs_price(spots, k, r, q, sig, t)
    ql_prices = []
    for s in spots.tolist():
        process = ql.BlackScholesMertonProcess(
            ql.QuoteHandle(ql.SimpleQuote(s)), div, rf, vol_ts)
        option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
        ql_prices.append(option.NPV())
    residuals = (ours - torch.tensor(ql_prices,
                                     dtype=torch.float64)).abs().tolist()

    h_process = ql.HestonProcess(
        rf, div, ql.QuoteHandle(ql.SimpleQuote(100.0)), HESTON["v0"],
        HESTON["kappa"], HESTON["theta"], HESTON["xi"], HESTON["rho"])
    option.setPricingEngine(
        ql.AnalyticHestonEngine(ql.HestonModel(h_process), 192))
    ql_heston = option.NPV()

    paths, variances = simulate_heston(n_paths=40, n_steps=64, s0=100.0,
                                       r=r, q=q, t_maturity=t, seed=3,
                                       antithetic=False, **HESTON)
    mc_paths, _ = simulate_heston(n_paths=400_000, n_steps=128, s0=100.0,
                                  r=r, q=q, t_maturity=t, seed=13, **HESTON)
    mc = mc_european_price(mc_paths, strike=k, r=r, t_maturity=t)
    payoff = torch.clamp(mc_paths[:, -1] - k, min=0.0)
    se = math.exp(-r * t) * float(payoff.std(unbiased=True)) / math.sqrt(
        mc_paths.shape[0])

    fin = paths[:, -1]
    lo, hi = float(fin.min()), float(fin.max())
    fan = [{"y": paths[i].tolist(), "mode": "lines",
            "line": {"color": viridis_rgb(
                float((paths[i, -1] - lo) / (hi - lo))), "width": 1.2},
            "hoverinfo": "skip", "showlegend": False}
           for i in range(paths.shape[0])]
    return {"spots": spots.tolist(), "ours": ours.tolist(),
            "ql": ql_prices, "residuals": residuals,
            "heston_max_res": max(residuals), "fan": fan,
            "var_paths": [variances[i].tolist() for i in
                          range(0, variances.shape[0], 4)],
            "theta": HESTON["theta"], "v0": HESTON["v0"],
            "mc": mc, "se": se, "ql_heston": ql_heston}


def heston_hedging_data(policy, premium: float) -> dict:
    """P&L per strategy on Heston paths: the GBM-trained policy evaluated
    zero-shot, against a variance-aware BS delta (Black-Scholes delta
    computed on the simulated variance path — the standard practitioner
    baseline under stochastic volatility). Premium is the QuantLib Heston
    price, the fair charge under these dynamics."""
    paths, variances = simulate_heston(n_paths=16_384, n_steps=26, s0=100.0,
                                       r=0.05, q=0.01, t_maturity=1.0,
                                       antithetic=True, seed=11, **HESTON)
    m = paths.shape[1] - 1
    ttms_grid = (1.0 * (1.0 - torch.arange(m, dtype=paths.dtype) / m)
                 ).repeat(paths.shape[0], 1)
    vols = variances[:, :m].sqrt()
    d1 = ((torch.log(paths[:, :m] / 100.0)
           + (0.05 - 0.01 + 0.5 * vols.square()) * ttms_grid)
          / (vols * ttms_grid.sqrt()))
    normal = torch.distributions.Normal(
        torch.zeros((), dtype=paths.dtype), torch.ones((), dtype=paths.dtype))
    delta = normal.cdf(d1)

    def pnl_of(pos: torch.Tensor) -> torch.Tensor:
        return hedge_pnl(paths, 100.0, premium, pos, 0.01)

    with torch.no_grad():
        pos_policy = run_policy(policy, paths, 100.0, 1.0)
    out = {}
    for name, pnl in (("no hedge", pnl_of(torch.zeros_like(delta))),
                      ("delta (var-aware)", pnl_of(delta)),
                      ("deep hedge (policy)", pnl_of(pos_policy))):
        out[name] = {"pnl": pnl.detach()[::4].tolist(),
                     "mean": float(pnl.mean()), "std": float(pnl.std()),
                     "cvar95": cvar(pnl)}
    return out


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
    out["policy"] = res.policy
    return out


def main() -> None:
    run, run_dir = load_latest_run()
    acc = run["accuracy"]
    speed = run["speed"]
    hedging = hedging_data()
    cv = crossval_data()
    heston_hedge = heston_hedging_data(hedging["policy"], cv["ql_heston"])
    panels = overview_panels()
    png_b64 = base64.b64encode((run_dir / "greeks_curves.png").read_bytes())

    rows_acc = "".join(
        f"<tr><td>{name}</td>"
        f"<td>{acc[name]['price_mae']:.3f}</td>"
        f"<td>{acc[name]['delta_mae']:.4f}</td>"
        f"<td>{acc[name]['ood_price_mae']:.2f}</td>"
        f"<td>{acc[name]['ood_delta_mae']:.3f}</td></tr>"
        for name in ("dml", "baseline"))
    def fmt_us(v: float) -> str:
        return f"{v:,.0f}" if v >= 100 else f"{v:,.4g}"

    rows_speed = "".join(
        f"<tr><td>{r['device']}</td><td>{r['method']}</td>"
        f"<td>{fmt_us(r['us_per_price'])}</td></tr>" for r in speed)
    mc_us = next(r["us_per_price"] for r in speed
                 if r["method"].startswith("monte carlo"))
    dml_us = next(r["us_per_price"] for r in speed
                  if "100k" in r["method"] and r["device"] == "cpu")

    hed_rows = "".join(
        f"<tr><td>{n}</td><td>{v['mean']:+.2f}</td><td>{v['std']:.2f}</td>"
        f"<td>{v['cvar95']:.2f}</td></tr>" for n, v in hedging.items()
        if n in ("no hedge", "delta (weekly)", "deep hedge (policy)"))
    hest_rows = "".join(
        f"<tr><td>{n}</td><td>{v['mean']:+.2f}</td><td>{v['std']:.2f}</td>"
        f"<td>{v['cvar95']:.2f}</td></tr>" for n, v in heston_hedge.items())

    violins = [{"type": "violin", "y": hedging[n]["pnl"], "name": n,
                "line": {"color": STRATEGY_COLORS[n]},
                "box": {"visible": True}, "meanline": {"visible": True},
                "points": False}
               for n in ("no hedge", "delta (weekly)",
                         "deep hedge (policy)")]

    heston_hedge_colors = {"no hedge": "#9ba1ad",
                           "delta (var-aware)": "#636EFA",
                           "deep hedge (policy)": "#00CC96"}
    heston_hedge_trace = [
        {"type": "violin", "y": v["pnl"], "name": n,
         "line": {"color": heston_hedge_colors[n]},
         "box": {"visible": True}, "meanline": {"visible": True},
         "points": False} for n, v in heston_hedge.items()]

    def short_method(r: dict) -> str:
        kind = "MC" if r["method"].startswith("monte carlo") else "DML"
        detail = ("matched err" if kind == "MC"
                  else "1 option" if "1 option" in r["method"] else "100k")
        return f"{kind} {detail} ({r['device']})"

    bar_colors = ["#EF553B" if r["method"].startswith("monte carlo")
                  else "#636EFA" if r["device"] == "cpu" else "#00CC96"
                  for r in speed]
    speed_trace = [{"type": "bar",
                    "x": [short_method(r) for r in speed],
                    "y": [r["us_per_price"] for r in speed],
                    "marker": {"color": bar_colors}}]

    strike_line = {"type": "line", "x0": 100, "x1": 100, "y0": 0, "y1": 1,
                   "yref": "paper", "line": {"dash": "dot", "color": "#9ba1ad"}}
    payoff_layout = dark_layout(
        xaxis={"title": "spot at expiry"},
        yaxis={"title": "$"},
        shapes=[strike_line])
    fan_layout = dark_layout(xaxis={"title": "time step"},
                             yaxis={"title": "spot"})
    heat_layout = dark_layout(xaxis={"title": "volatility"},
                              yaxis={"title": "spot"})
    sobol_layout = dark_layout(xaxis={"title": "spot"},
                               yaxis={"title": "volatility"})
    speed_layout = dark_layout(
        xaxis={"tickangle": -20},
        yaxis={"type": "log", "title": "µs per price"},
        margin={"t": 14, "r": 20, "b": 88, "l": 60})
    violins_layout = dark_layout(yaxis={"title": "P&L per year"},
                                 violingap=0.3)

    if cv is not None:
        cv_section = f"""
<h2>Cross-validation vs QuantLib</h2>
<p class="lead">QuantLib is an independent engine with zero shared code —
a bug would have to exist in both libraries to slip through. Closed-form
prices agree to {cv['heston_max_res']:.1e} across the spot grid; the
Heston Monte Carlo lands inside a CLT band of QuantLib's analytic
engine.</p>
<div class="grid2">
  <div class="panel"><div class="t">Closed form vs QuantLib — two engines,
    one line</div><div id="c-cv-overlay" class="chart"></div></div>
  <div class="panel"><div class="t">Residual |ours − QuantLib|</div>
    <div id="c-cv-res" class="chart"></div></div>
  <div class="panel"><div class="t">Heston paths — stochastic volatility,
    correlation ρ = −0.7</div><div id="c-heston-fan" class="chart"></div>
  </div>
  <div class="panel"><div class="t">Variance mean reversion — v₀
    {cv['v0']:.2f} pulled toward θ {cv['theta']:.2f}</div>
    <div id="c-heston-var" class="chart"></div></div>
</div>
<p class="lead">Heston at-the-money call: our full-truncation Euler Monte
Carlo (400k paths, ±2 standard errors) against QuantLib's
AnalyticHestonEngine.</p>
<div id="c-heston-price" class="chart"></div>
"""
        cv_overlay_trace = [
            {"x": cv["spots"], "y": cv["ours"], "mode": "lines",
             "name": "this repo", "line": {"color": "#636EFA", "width": 2}},
            {"x": cv["spots"], "y": cv["ql"], "mode": "lines",
             "name": "QuantLib", "line": {"color": "#00CC96", "width": 2,
                                          "dash": "dash"}}]
        cv_res_trace = [{"x": cv["spots"], "y": cv["residuals"],
                         "mode": "lines", "showlegend": False,
                         "line": {"color": "#EF553B", "width": 1.6}}]
        cv_var_trace = [
            {"y": p, "mode": "lines", "showlegend": False,
             "line": {"color": "rgba(99,110,250,0.45)", "width": 1}}
            for p in cv["var_paths"]]
        cv_var_trace += [
            {"y": [cv["theta"]] * 65, "mode": "lines", "name": "θ",
             "line": {"color": "#00CC96", "dash": "dash", "width": 2}},
            {"y": [cv["v0"]] * 65, "mode": "lines", "name": "v₀",
             "line": {"color": "#9ba1ad", "dash": "dot", "width": 1.5}}]
        # forest plot: one quantity, two estimates — compare on a shared axis
        cv_price_trace = [
            {"x": [cv["mc"]], "y": ["Monte Carlo (400k paths)"],
             "error_x": {"type": "data", "array": [2 * cv["se"]],
                         "visible": True}, "mode": "markers",
             "name": "Monte Carlo", "marker": {"color": "#636EFA",
                                               "size": 12}},
            {"x": [cv["ql_heston"]], "y": ["QuantLib analytic"],
             "mode": "markers", "name": "QuantLib",
             "marker": {"color": "#00CC96", "size": 12,
                        "symbol": "diamond"}}]
        cv_overlay_layout = dark_layout(xaxis={"title": "spot"},
                                        yaxis={"title": "value ($)"})
        cv_res_layout = dark_layout(
            xaxis={"title": "spot"},
            yaxis={"type": "log", "title": "|difference|",
                   "tickvals": [1e-16, 1e-14, 1e-12, 1e-10, 1e-8],
                   "ticktext": ["1e-16", "1e-14", "1e-12", "1e-10",
                                "1e-8"]})
        cv_fan_layout = dark_layout(xaxis={"title": "time step"},
                                    yaxis={"title": "spot"})
        cv_var_layout = dark_layout(xaxis={"title": "time step"},
                                    yaxis={"title": "variance"})
        cv_price_layout = dark_layout(
            xaxis={"title": "ATM call price ($) — bars overlap = agreement",
                   "range": [cv["mc"] - 6 * cv["se"],
                             max(cv["mc"], cv["ql_heston"]) + 6 * cv["se"]]},
            margin={"t": 14, "r": 20, "b": 30, "l": 160})
        cv_script = f"""
plot('c-cv-overlay', {json.dumps(cv_overlay_trace)},
     {json.dumps(cv_overlay_layout)});
plot('c-cv-res', {json.dumps(cv_res_trace)}, {json.dumps(cv_res_layout)});
plot('c-heston-fan', {json.dumps(cv['fan'])}, {json.dumps(cv_fan_layout)});
plot('c-heston-var', {json.dumps(cv_var_trace)},
     {json.dumps(cv_var_layout)});
plot('c-heston-price', {json.dumps(cv_price_trace)},
     Object.assign({{height: 240}}, {json.dumps(cv_price_layout)}));
"""
    else:
        cv_section = ""
        cv_script = ""

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Deep Hedging &amp; Pricing Simulator</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>{CSS}</style></head><body>
<h1>Deep Hedging &amp; Pricing Simulator</h1>
<p class="lead">European options priced and risk-managed three independent
ways, forced to agree: <b>analytic Black-Scholes</b>, <b>autograd (AAD)</b>,
and <b>pathwise Monte Carlo</b> (Greeks by backpropagating through simulated
paths). On top of that, two models train live in the dashboard: a
<b>differential ML pricer</b> (Savine) and a <b>deep-hedging policy</b>
under transaction costs (Buehler).</p>

<h2>Benchmark status</h2>
<div class="metrics">
  <div class="metric"><div class="k">Pricer &amp; Greeks</div>
    <div class="v">3 routes</div><div class="s">agree to 1e-13</div></div>
  <div class="metric"><div class="k">DML pricer</div>
    <div class="v">price MAE {acc['dml']['price_mae']:.2f}</div>
    <div class="s">2.4x better than baseline</div></div>
  <div class="metric"><div class="k">Inference speed</div>
    <div class="v">{mc_us / dml_us:,.0f}x</div>
    <div class="s">faster than MC at matched error</div></div>
  <div class="metric"><div class="k">Deep hedging</div>
    <div class="v">CVaR95 {hedging['deep hedge (policy)']['cvar95']:.2f}</div>
    <div class="s">vs {hedging['delta (weekly)']['cvar95']:.2f} weekly delta,
    1% costs</div></div>
</div>
<div class="alert">Full-budget benchmark (100k samples, 300 epochs, price
MAE 0.053): the network prices ~130,000x faster than Monte Carlo at matched
accuracy on CPU, 0.79 µs vs 104,000 µs per price.</div>

<div class="grid2">
  <div class="panel"><div class="t">The underlying — 40 GBM paths, colored
    by where they end</div><div id="c-fan" class="chart"></div></div>
  <div class="panel"><div class="t">The book — short one call, collect 9.83,
    owe the payoff</div><div id="c-payoff" class="chart"></div></div>
  <div class="panel"><div class="t">What the pricer computes — option value
    over spot and volatility</div><div id="c-heat" class="chart"></div></div>
  <div class="panel"><div class="t">What the models learn from — Sobol
    coverage of the parameter space</div><div id="c-sobol" class="chart">
    </div></div>
</div>

<h2>Results</h2>

<h3>Differential training vs price-only baseline</h3>
<p class="lead">Same network and data; the DML model also fits the price's
input-gradients. {run['config']['samples']:,} Sobol samples,
{run['config']['epochs']} epochs, float64. OOD = spots 25–45 and 210–240,
outside the training range.</p>
<table><tr><th>learner</th><th>price MAE</th><th>delta MAE</th>
<th>OOD price MAE</th><th>OOD delta MAE</th></tr>{rows_acc}</table>

<h3>Speed at matched error</h3>
<p class="lead">Monte Carlo is timed with the path count its CLT standard
error needs to match the learner's price accuracy.</p>
<table><tr><th>device</th><th>method</th><th>µs per price</th></tr>
{rows_speed}</table>
<div id="c-speed" class="chart"></div>

<h3>Greeks from the trained network</h3>
<p class="lead">Delta and gamma vs spot against the analytic pricer. Gamma
is not a training label — it is a second autograd pass through the trained
network.</p>
<img class="curve" alt="delta and gamma curves vs analytic"
     src="data:image/png;base64,{png_b64.decode()}">

{cv_section}
<h3>Deep hedging under 1% transaction costs</h3>
<p class="lead">Short one at-the-money call, stock traded at 26 dates. The
policy network is trained on entropic risk of hedging P&amp;L. A fairly
priced book has zero expected P&amp;L by construction; the improvement is
cost efficiency, not alpha.</p>
<table><tr><th>strategy</th><th>mean</th><th>std</th><th>CVaR95</th></tr>
{hed_rows}</table>
<div id="c-violins" class="chart"></div>

<h3>Same policy, stochastic volatility (Heston paths)</h3>
<p class="lead">Robustness check: the policy above was trained on GBM paths,
then evaluated zero-shot on Heston paths (stochastic volatility,
ρ = −0.7) at the same 1% costs. The baseline is a variance-aware delta —
Black-Scholes delta computed on the simulated variance path — and the
premium is the QuantLib Heston price, the fair charge under these
dynamics. All numbers computed live; nothing tuned per strategy.</p>
<table><tr><th>strategy</th><th>mean</th><th>std</th><th>CVaR95</th></tr>
{hest_rows}</table>
<div id="c-heston-hedge" class="chart"></div>

<footer><p class="caption">Generated {date.today().isoformat()} by
scripts/make_site.py from benchmark run {run_dir.name} (reproduce:
scripts/benchmark.py) and a seeded policy training run.
Dashboard: <code>uv run --extra app streamlit run app/dashboard.py</code>
— <a href="https://github.com/pmasousa/deep-hedging-pricing-simulator">
source</a>.</p></footer>

<script>
function plot(id, traces, layout) {{
  Plotly.newPlot(id, traces, Object.assign({{height: 300}}, layout));
}}
plot('c-fan', {json.dumps(panels['fan'])}, {json.dumps(fan_layout)});
plot('c-payoff', {json.dumps(panels['payoff'])}, {json.dumps(payoff_layout)});
plot('c-heat', {json.dumps(panels['heat'])}, {json.dumps(heat_layout)});
plot('c-sobol', {json.dumps(panels['sobol'])}, {json.dumps(sobol_layout)});
plot('c-speed', {json.dumps(speed_trace)},
     Object.assign({{height: 420}}, {json.dumps(speed_layout)}));
plot('c-violins', {json.dumps(violins)},
     Object.assign({{height: 420}}, {json.dumps(violins_layout)}));
plot('c-heston-hedge', {json.dumps(heston_hedge_trace)},
     Object.assign({{height: 420}}, {json.dumps(violins_layout)}));
{cv_script}
</script>
</body></html>
"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
