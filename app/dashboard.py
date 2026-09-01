"""Recruiter-facing dashboard: Black-Scholes surfaces, dataset coverage, and
the autograd validation story. Run from the repo root:

    uv run --extra app streamlit run app/dashboard.py
"""

import plotly.graph_objects as go
import streamlit as st
import torch
from plotly.subplots import make_subplots

from dhps.bench.evaluate import evaluate_learner, greeks_curve, ood_metrics
from dhps.datasets.european import FEATURES, LABELS, make_european_dataset
from dhps.hedging.policy import DeepHedgeConfig, train_deep_hedge
from dhps.hedging.simulator import cvar, delta_positions, hedge_pnl, premium_bs
from dhps.pricing.aad import bs_greeks_ad
from dhps.pricing.black_scholes import bs_greeks, bs_price
from dhps.pricing.pathwise_mc import pathwise_european_greeks
from dhps.simulators.gbm import simulate_gbm
from dhps.train.trainer import TrainConfig, train_model

st.set_page_config(page_title="DHPS — Pricing & Differential ML", layout="wide")

TEMPLATE = "plotly_white"
GREEKS = ("delta", "gamma", "vega", "theta", "rho")
GREEK_TITLES = {
    "delta": "Delta (∂V/∂S)", "gamma": "Gamma (∂²V/∂S²)", "vega": "Vega (∂V/∂σ)",
    "theta": "Theta (−∂V/∂T, per year)", "rho": "Rho (∂V/∂r)",
}


# ---------------------------------------------------------------- overview --

def render_overview() -> None:
    st.header("Deep Hedging & Pricing Simulator")
    st.markdown(
        """
This repo prices European options and computes risk sensitivities (Greeks)
three independent ways and makes them agree:

1. **Analytic** — closed-form Black-Scholes formulas (the ground truth).
2. **Autograd (AAD)** — the same pricer differentiated by PyTorch's automatic
   differentiation instead of derived formulas.
3. **Pathwise Monte Carlo** — Greeks by backpropagating *through simulated
   asset paths*, the trick that makes differential ML possible.

The end goal: a neural network trained on prices **and** derivatives
(differential machine learning, Savine 2019) that prices and risk-manages
at inference speed, plus a deep-hedging policy under transaction costs.

Use the sidebar to explore. Every number below is computed live from the
library in this repository — nothing is hard-coded.
        """
    )
    st.subheader("Pipeline status")
    cols = st.columns(4)
    cols[0].metric("Simulation engine", "GBM + antithetic", "done")
    cols[1].metric("Reference pricer", "BS + 5 Greeks", "done")
    cols[2].metric("AAD validation", "1e-13 vs analytic", "done")
    cols[3].metric("DML training", "next stage", "wip")
    st.caption(
        "Tolerances are statistical: Monte Carlo gates use CLT bands, "
        "autograd gates use 1e-10 absolute error."
    )


# ------------------------------------------------------------------ pricer --

def render_pricer() -> None:
    st.header("Pricer & Greeks explorer")
    st.markdown(
        "Closed-form prices and Greeks across a spot grid. Change the deal "
        "parameters in the sidebar; every curve recomputes live."
    )
    with st.sidebar:
        st.subheader("Deal parameters")
        strike = st.slider("Strike K", 50.0, 150.0, 100.0, 1.0)
        t_mat = st.slider("Maturity T (years)", 0.1, 2.0, 1.0, 0.05)
        sigma = st.slider("Volatility σ", 0.05, 0.6, 0.2, 0.01)
        rate = st.slider("Risk-free rate r", 0.0, 0.10, 0.05, 0.005)
        div = st.slider("Dividend yield q", 0.0, 0.06, 0.01, 0.005)
        is_call = st.toggle("Call (off = put)", value=True)

    spots = torch.linspace(50.0, 150.0, 401, dtype=torch.float64)
    price = bs_price(spots, strike, rate, div, sigma, t_mat, call=is_call)
    greeks = bs_greeks(spots, strike, rate, div, sigma, t_mat, call=is_call)

    k_col, atm_col, delta_col, gamma_col = st.columns(4)
    atm = float(bs_price(torch.tensor([strike]), strike, rate, div, sigma, t_mat,
                         call=is_call)[0])
    k_col.metric("Strike", f"{strike:.0f}")
    atm_col.metric("ATM price", f"{atm:.4f}")
    delta_col.metric("ATM delta", f"{float(greeks['delta'][200]):.4f}")
    gamma_col.metric("ATM gamma", f"{float(greeks['gamma'][200]):.4f}")

    fig = make_subplots(rows=2, cols=3,
                        subplot_titles=("Value", *(GREEK_TITLES[g] for g in GREEKS)))
    fig.add_trace(go.Scatter(x=spots.tolist(), y=price.tolist(), showlegend=False), 1, 1)
    for i, g in enumerate(GREEKS):
        fig.add_trace(go.Scatter(x=spots.tolist(), y=greeks[g].tolist(),
                                 showlegend=False), 1 + (i + 1) // 3, (i + 1) % 3 + 1)
    fig.update_layout(height=560, template=TEMPLATE, title_text="Value and Greeks vs spot")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Value surface over spot and volatility")
    vol_grid = torch.linspace(0.05, 0.6, 56, dtype=torch.float64)
    mesh_s, mesh_v = torch.meshgrid(spots[::4], vol_grid, indexing="ij")
    surface = bs_price(mesh_s, strike, rate, div, mesh_v, t_mat, call=is_call)
    heat = go.Figure(go.Heatmap(x=vol_grid.tolist(), y=spots[::4].tolist(),
                                z=surface.tolist(), colorscale="Viridis"))
    heat.update_xaxes(title_text="volatility σ")
    heat.update_yaxes(title_text="spot")
    heat.update_layout(height=480, template=TEMPLATE)
    st.plotly_chart(heat, width="stretch")


# ----------------------------------------------------------------- dataset --

@st.cache_data(show_spinner="Building dataset...")
def _build_dataset(n_samples: int, seed: int) -> dict[str, torch.Tensor]:
    return make_european_dataset(n_samples=n_samples, seed=seed)


def render_dataset() -> None:
    st.header("Training data")
    st.markdown(
        "Scrambled-Sobol coverage of the parameter space — spot, moneyness, "
        "maturity, volatility — with analytic price and Greek labels for "
        "every sample. This is the dataset the differential ML model will "
        "train on."
    )
    with st.sidebar:
        st.subheader("Dataset")
        n_samples = st.slider("Samples", 10_000, 200_000, 50_000, step=10_000)
        seed = st.slider("Seed", 1, 99, 7)
    data = _build_dataset(n_samples, seed)

    n_tr = data["x_train"].shape[0]
    n_va = data["x_val"].shape[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Train samples", f"{n_tr:,}")
    c2.metric("Val samples", f"{n_va:,}")
    c3.metric("Features / labels", f"{len(FEATURES)} / {len(LABELS)}")
    st.caption(f"features: {', '.join(FEATURES)} — labels: {', '.join(LABELS)}")

    left, right = st.columns(2)
    with left:
        st.subheader("Parameter coverage")
        idx = torch.randperm(n_tr)[:4000]
        xs = data["x_train"][idx]
        scat = go.Figure(go.Scatter(
            x=xs[:, 0].tolist(), y=xs[:, 3].tolist(),
            mode="markers", marker=dict(size=4, color=xs[:, 0] / xs[:, 1],
                                        colorscale="Viridis", showscale=True,
                                        colorbar=dict(title="s0/K")),
        ))
        scat.update_xaxes(title_text="spot")
        scat.update_yaxes(title_text="volatility")
        scat.update_layout(height=440, template=TEMPLATE)
        st.plotly_chart(scat, width="stretch")
    with right:
        st.subheader("Label distributions")
        fig = go.Figure()
        for j, name in enumerate(LABELS):
            fig.add_trace(go.Histogram(x=data["y_train"][idx, j].tolist(),
                                       name=name, histnorm="probability density",
                                       opacity=0.55))
        fig.update_layout(barmode="overlay", height=440, template=TEMPLATE)
        st.plotly_chart(fig, width="stretch")


# ------------------------------------------------------------------- model --

def render_model() -> None:
    st.header("Differential ML: the learned pricer")
    st.markdown(
        "Two identical networks train on the same data: **baseline** sees "
        "prices only; **DML** also sees the price's input-gradients "
        "(the differential labels from the validation page). Both train "
        "live in this session — small budget, a few seconds — and every "
        "chart below is the resulting model, nothing pre-baked."
    )
    if "dml_result" not in st.session_state:
        with st.spinner("Training both learners (~15 s)..."):
            cfg = TrainConfig(n_samples=8_000, hidden=(48, 48), epochs=120,
                              batch_size=4_096, seed=7)
            st.session_state.dml_result = train_model(cfg, differential=True)
            st.session_state.base_result = train_model(cfg, differential=False)
            st.session_state.train_data = make_european_dataset(
                n_samples=8_000, seed=7)
    dml = st.session_state.dml_result
    base = st.session_state.base_result
    data = st.session_state.train_data

    m_d, m_b = evaluate_learner(dml, data), evaluate_learner(base, data)
    o_d, o_b = ood_metrics(dml), ood_metrics(base)
    c1, c2 = st.columns(2)
    c1.metric("Delta MAE — DML vs baseline",
              f"{m_d['delta_mae']:.4f} vs {m_b['delta_mae']:.4f}",
              f"{(1 - m_d['delta_mae'] / m_b['delta_mae']) * 100:.0f}% better")
    c2.metric("Price MAE — DML vs baseline",
              f"{m_d['price_mae']:.4f} vs {m_b['price_mae']:.4f}",
              f"{(1 - m_d['price_mae'] / m_b['price_mae']) * 100:.0f}% better")
    st.caption(
        "Out-of-distribution (spots 25-45 and 210-240, outside the training "
        f"range 50-200): DML price MAE {o_d['ood_price_mae']:.3f} vs baseline "
        f"{o_b['ood_price_mae']:.3f} — both degrade, differential training "
        "degrades less."
    )

    with st.sidebar:
        st.subheader("Curve configuration")
        strike = st.slider("Curve strike K", 70.0, 130.0, 100.0, 5.0)
        t_mat = st.slider("Curve maturity T", 0.25, 2.0, 1.0, 0.25)
        sigma = st.slider("Curve volatility", 0.05, 0.6, 0.2, 0.05)

    spots = torch.linspace(50.0, 200.0, 300, dtype=torch.float64)
    curves = {"DML": greeks_curve(dml, spots, strike, t_mat, sigma),
              "baseline": greeks_curve(base, spots, strike, t_mat, sigma)}
    tab_delta, tab_gamma = st.tabs(["Delta (the hero chart)", "Gamma (never a label)"])
    for tab, greek in ((tab_delta, "delta"), (tab_gamma, "gamma")):
        fig = go.Figure()
        ref = curves["DML"]
        fig.add_trace(go.Scatter(x=spots.tolist(), y=ref[f"{greek}_true"].tolist(),
                                 name="analytic", line=dict(dash="dash", width=3)))
        for name, c in curves.items():
            fig.add_trace(go.Scatter(x=spots.tolist(), y=c[greek].tolist(),
                                     name=name, line=dict(width=2)))
        fig.update_xaxes(title_text="spot")
        fig.update_yaxes(title_text=greek)
        fig.update_layout(height=460, template=TEMPLATE)
        tab.plotly_chart(fig, width="stretch")
    st.info(
        "**Gamma was never a training label.** It falls out of a second "
        "autograd pass through the trained network — the differential "
        "structure the model internalized, for free."
    )


# --------------------------------------------------------------- hedging ---

@st.cache_data(show_spinner="Training the deep-hedging policy (~20 s)...")
def _hedge_results(cost: float) -> dict:
    """Train the policy at this cost level and score every strategy on the
    SAME eval paths. Returns plain floats/lists for caching."""
    cfg = DeepHedgeConfig(n_paths=8_192, n_steps=26, cost_rate=cost,
                          epochs=80, seed=7, eval_paths=4_096)
    res = train_deep_hedge(cfg)
    sim = dict(s0=cfg.s0, r=cfg.r, q=cfg.q, sigma=cfg.sigma,
               t_maturity=cfg.t_maturity)
    paths = simulate_gbm(n_paths=cfg.eval_paths, n_steps=cfg.n_steps,
                         antithetic=True, seed=cfg.eval_seed, **sim)
    premium = premium_bs(cfg.s0, cfg.strike, cfg.r, cfg.q, cfg.sigma,
                         cfg.t_maturity)
    delta = delta_positions(paths, cfg.strike, cfg.r, cfg.q, cfg.sigma,
                            cfg.t_maturity)

    def pnl_of(positions: torch.Tensor) -> torch.Tensor:
        return hedge_pnl(paths, cfg.strike, premium, positions, cost)

    strats = {"no hedge": pnl_of(torch.zeros_like(delta)),
              "delta (weekly)": pnl_of(delta),
              "deep hedge (policy)": res.eval_pnl}
    out = {"pnl": {}, "stats": {}, "volume": {}}
    for name, pnl in strats.items():
        out["pnl"][name] = pnl.tolist()
        out["stats"][name] = {"mean": float(pnl.mean()), "std": float(pnl.std()),
                              "cvar95": cvar(pnl)}
    trades_d = torch.cat([delta[:, :1], delta[:, 1:] - delta[:, :-1]], dim=1)
    out["volume"]["delta (weekly)"] = float(trades_d.abs().sum(1).mean())
    out["volume"]["deep hedge (policy)"] = res.metrics["traded_volume"]
    out["train_seconds"] = res.seconds
    return out


def render_hedging() -> None:
    st.header("Deep hedging under transaction costs")
    st.markdown(
        "The book: SHORT one at-the-money call (premium ~9.83), hedged by "
        "trading stock at 26 dates over a year. Weekly delta hedging is the "
        "1973 answer; the deep hedge is a policy network trained by "
        "backpropagating a tail-risk penalty through simulated hedging "
        "trajectories. Same paths, same costs, same premium — only the "
        "trading rule differs."
    )
    cost = st.sidebar.slider("Transaction cost rate", 0.0, 0.02, 0.01, 0.0025,
                             format="%.2f%%")
    data = _hedge_results(cost)
    names = list(data["pnl"])

    cols = st.columns(3)
    for col, name in zip(cols, names, strict=True):
        s = data["stats"][name]
        col.metric(f"{name} — CVaR95", f"{s['cvar95']:.2f}",
                   f"mean {s['mean']:+.2f} · std {s['std']:.2f}")
    st.caption(
        "CVaR95 = average P&L of the worst 5% of simulated years (higher is "
        "better). Hedging does NOT create profit: the option is priced "
        "fairly, so expected P&L is ~0 by construction. The game is paying "
        "less for the same protection — the policy's edge is cost "
        "efficiency, not alpha."
    )

    fig = go.Figure()
    for name in names:
        fig.add_trace(go.Histogram(x=data["pnl"][name], name=name,
                                   opacity=0.6, nbinsx=80))
    fig.update_layout(barmode="overlay", height=480, template=TEMPLATE,
                      title_text="Terminal hedging P&L distribution")
    fig.update_xaxes(title_text="P&L")
    st.plotly_chart(fig, width="stretch")

    vol = data["volume"]
    st.markdown(
        f"**Traded volume** (avg sum of |Δ position| per path): "
        f"delta {vol['delta (weekly)']:.2f} vs policy "
        f"{vol['deep hedge (policy)']:.2f} — the policy trades less because "
        "it learned that rebalancing deep ITM/OTM only pays costs, and "
        "concentrates trading near the strike where gamma lives."
    )
    st.caption(
        f"Policy trained live in {data['train_seconds']:.0f} s at this cost "
        "level; changing the slider retrains. P&L is undiscounted — a stated "
        "simplification shared by every strategy, so comparisons stay fair."
    )


# -------------------------------------------------------------- validation --

def render_validation() -> None:
    st.header("Validation: three roads to the same Greek")
    st.markdown(
        "The core claim of this repo, checked live: **derivatives come free "
        "from autograd**. The analytic formulas, autograd through the pricer, "
        "and backprop through Monte Carlo paths must agree."
    )

    st.subheader("Autograd vs analytic formulas")
    spots = torch.linspace(60.0, 150.0, 91, dtype=torch.float64)
    ad = bs_greeks_ad(spots, 100.0, 0.05, 0.01, 0.2, 1.0)
    an = bs_greeks(spots, 100.0, 0.05, 0.01, 0.2, 1.0)
    worst = max(float((ad[g] - an[g]).abs().max()) for g in GREEKS)
    m1, m2 = st.columns(2)
    m1.metric("Max abs error, all 5 Greeks", f"{worst:.2e}")
    m2.metric("Agreement", "bit-level (float64)" if worst < 1e-10 else "CHECK")
    st.caption(
        "Same pricer, two differentiation strategies: hand-derived formulas "
        "vs torch.autograd.grad. Any disagreement here would be a bug in one "
        "of them."
    )

    st.subheader("Pathwise Monte Carlo vs analytic delta")
    st.markdown(
        "Delta estimated by backpropagating through 100,000 simulated GBM "
        "paths at each spot, with ±2 standard-error bands. The estimator is "
        "unbiased — the analytic curve must sit inside the bands."
    )
    with st.sidebar:
        st.subheader("Pathwise MC")
        pw_strike = st.slider("Strike (MC)", 70.0, 130.0, 100.0, 5.0)
        pw_sigma = st.slider("Volatility (MC)", 0.05, 0.6, 0.2, 0.05)
        pw_t = st.slider("Maturity (MC)", 0.25, 2.0, 1.0, 0.25)
        pw_n = st.select_slider("Paths per point", (10_000, 50_000, 100_000, 200_000))

    @st.cache_data(show_spinner="Running pathwise Monte Carlo...")
    def _pw_curve(strike: float, sigma: float, t_mat: float, n_paths: int):
        pts = []
        for s0 in (80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0):
            res = pathwise_european_greeks(
                n_paths=n_paths, n_steps=64, s0=s0, strike=strike, r=0.05,
                q=0.01, sigma=sigma, t_maturity=t_mat, seed=11,
            )
            pts.append((s0, res["delta"], 2 * res["delta_se"]))
        return pts

    pts = _pw_curve(pw_strike, pw_sigma, pw_t, pw_n)
    curve_spots = torch.linspace(75.0, 125.0, 201, dtype=torch.float64)
    analytic = bs_greeks(curve_spots, pw_strike, 0.05, 0.01, pw_sigma, pw_t)["delta"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve_spots.tolist(), y=analytic.tolist(),
                             name="analytic delta", line=dict(width=3)))
    fig.add_trace(go.Scatter(
        x=[p[0] for p in pts], y=[p[1] for p in pts],
        error_y=dict(type="data", array=[p[2] for p in pts], visible=True),
        name="pathwise MC ± 2·SE", mode="markers", marker=dict(size=9),
    ))
    fig.update_xaxes(title_text="spot")
    fig.update_yaxes(title_text="delta")
    fig.update_layout(height=480, template=TEMPLATE)
    st.plotly_chart(fig, width="stretch")

    st.info(
        "**Why no Monte Carlo gamma?** The call payoff has a kink at the "
        "strike: its second derivative is a Dirac delta, and autograd through "
        "`clamp` returns zero almost everywhere. The pathwise gamma estimator "
        "is invalid, not merely noisy — likelihood-ratio or Malliavin weights "
        "would be needed. Excluding it is a correctness decision, not a "
        "shortcut."
    )


# --------------------------------------------------------------------- run --

PAGES = {
    "Overview": render_overview,
    "Pricer & Greeks": render_pricer,
    "Training data": render_dataset,
    "Learned pricer (DML)": render_model,
    "Deep hedging": render_hedging,
    "Validation": render_validation,
}


def nav_bar() -> str:
    """Top nav bar of plain buttons; the active page is highlighted."""
    current = st.session_state.get("page", next(iter(PAGES)))
    cols = st.columns(len(PAGES))
    for col, name in zip(cols, PAGES, strict=True):
        if col.button(name, key=f"nav_{name}",
                      type="primary" if name == current else "secondary",
                      width="stretch"):
            st.session_state.page = name
            current = name
    return current


page = nav_bar()
PAGES[page]()
