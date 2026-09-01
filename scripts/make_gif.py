"""Render assets/demo.gif — animated README demo from library output.

Five scenes: GBM paths drawing in, option value sweeping volatility,
Greeks recomputing across strikes, a DML network learning the delta
epoch by epoch, and hedging P&L drifting as transaction costs rise.
Requires pillow; run from the repo root:

    uv run --extra dev --with pillow python scripts/make_gif.py
"""

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from dhps.datasets.european import make_european_dataset  # noqa: E402
from dhps.datasets.normalize import Standardizer  # noqa: E402
from dhps.hedging.simulator import cvar, delta_positions, hedge_pnl, premium_bs  # noqa: E402
from dhps.models.losses import value_and_grad  # noqa: E402
from dhps.simulators.gbm import simulate_gbm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "demo.gif"

ACCENT, BLUE, GREEN, RED = "#31333f", "#636EFA", "#00CC96", "#EF553B"
mpl.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#d5d9de", "axes.grid": True,
    "grid.color": "#ebebeb", "font.size": 11,
    "text.color": ACCENT, "axes.labelcolor": ACCENT,
    "xtick.color": ACCENT, "ytick.color": ACCENT,
    "axes.titlesize": 13, "axes.titleweight": "bold",
})

W, H = 8.0, 4.5


def new_ax(title: str):
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_title(title)
    return fig, ax


def snap(fig: plt.Figure, frames: list[Image.Image], ms: int = 90) -> None:
    fig.canvas.draw()
    img = Image.frombytes(
        "RGBA", fig.canvas.get_width_height(),
        fig.canvas.buffer_rgba()).convert("RGB")
    frames.append((img, ms))
    plt.close(fig)


def scene_paths(frames: list[Image.Image]) -> None:
    paths = simulate_gbm(n_paths=40, n_steps=64, antithetic=False, seed=3)
    fin = paths[:, -1]
    lo, hi = float(fin.min()), float(fin.max())
    colors = [plt.cm.viridis(float((p - lo) / (hi - lo))) for p in fin]
    for k in np.linspace(4, 64, 14).astype(int):
        fig, ax = new_ax("1 · simulate — GBM paths under the risk-neutral measure")
        for i in range(paths.shape[0]):
            ax.plot(paths[i, :k], color=colors[i], lw=1.1)
        ax.set_xlim(0, 64)
        ax.set_xlabel("time step")
        ax.set_ylabel("spot")
        snap(fig, frames)


def scene_value(frames: list[Image.Image]) -> None:
    spots = torch.linspace(50.0, 150.0, 200, dtype=torch.float64)
    for sigma in np.linspace(0.05, 0.6, 16):
        from dhps.pricing.black_scholes import bs_price
        fig, ax = new_ax(f"2 · price — closed form, volatility σ = {sigma:.2f}")
        price = bs_price(spots, 100.0, 0.05, 0.01, float(sigma), 1.0)
        ax.plot(spots, price, color=BLUE, lw=2.2)
        ax.fill_between(spots, price, alpha=0.15, color=BLUE)
        ax.set_title(f"2 · price — closed form, volatility σ = {sigma:.2f}")
        ax.set_xlabel("spot")
        ax.set_ylabel("value ($)")
        snap(fig, frames)


def scene_greeks(frames: list[Image.Image]) -> None:
    spots = torch.linspace(60.0, 150.0, 200, dtype=torch.float64)
    for strike in np.linspace(80.0, 120.0, 14):
        fig, ax = new_ax(f"3 · Greeks — strike K = {strike:.0f}")
        from dhps.pricing.black_scholes import bs_greeks
        g = bs_greeks(spots, float(strike), 0.05, 0.01, 0.2, 1.0)
        ax.plot(spots, g["delta"], color=BLUE, lw=2.2, label="delta")
        ax.set_xlabel("spot")
        ax.set_ylabel("delta", color=BLUE)
        ax.set_ylim(-0.05, 1.05)
        ax2 = ax.twinx()
        ax2.plot(spots, g["gamma"], color=GREEN, lw=2.2, label="gamma")
        ax2.set_ylabel("gamma", color=GREEN)
        ax2.grid(False)
        ax.legend(loc="upper left")
        ax2.legend(loc="upper right")
        snap(fig, frames)


def scene_training(frames: list[Image.Image]) -> None:
    cfg_epochs, cfg_samples = 300, 20_000
    data = make_european_dataset(n_samples=cfg_samples, seed=7)
    scaler = Standardizer.fit(data["x_train"], data["y_train"])
    g_scale = scaler.grad_scale[0]
    from dhps.bench.evaluate import greeks_curve
    from dhps.models.mlp import make_mlp
    from dhps.train.trainer import TrainResult
    torch.manual_seed(7)
    model = make_mlp(n_in=4, hidden=(64, 64, 64))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    xt = scaler.transform_x(data["x_train"])
    yt = scaler.transform_y(data["y_train"])[:, :1]
    gt = data["g_train"] * g_scale
    spots = torch.linspace(72.0, 166.0, 200, dtype=torch.float64)
    from dhps.pricing.black_scholes import bs_greeks
    true_delta = bs_greeks(spots, 100.0, 0.05, 0.01, 0.2, 1.0)["delta"]

    for epoch in range(cfg_epochs):
        perm = torch.randperm(xt.shape[0])
        for idx in perm.split(4_096):
            yp, gp = value_and_grad(model, xt[idx])
            alpha = 1.0 / 5.0
            loss = (alpha * torch.nn.functional.mse_loss(yp, yt[idx])
                    + (1 - alpha) * torch.nn.functional.mse_loss(gp, gt[idx]))
            opt.zero_grad()
            loss.backward()
            opt.step()
        # early epochs predict price-scale values far above the delta axis;
        # pre-roll 60 epochs before the first snapshot, then every 20
        if epoch < 60 or (epoch - 19) % 20:
            continue
        model.eval()
        # plot the model's input-gradient (delta), not its price output —
        # greeks_curve un-normalizes through the chain rule
        curve = greeks_curve(TrainResult(model=model, scaler=scaler),
                             spots, 100.0, 1.0, 0.2)
        xv = scaler.transform_x(data["x_val"]).requires_grad_(True)
        gv_pred = torch.autograd.grad(model(xv).sum(), xv)[0][:, 0]
        val_mae = float((gv_pred / g_scale[0] - data["g_val"][:, 0])
                        .abs().mean())
        fig, ax = new_ax(f"4 · differential ML — training, epoch {epoch + 1}")
        ax.plot(spots, true_delta, "k--", lw=2, label="analytic delta")
        ax.plot(spots, curve["delta"], color=RED, lw=2,
                label=f"DML (val delta MAE {val_mae:.3f})")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("spot")
        ax.set_ylabel("delta")
        ax.legend(loc="upper left")
        snap(fig, frames)


def scene_hedging(frames: list[Image.Image]) -> None:
    sim = dict(s0=100.0, r=0.05, q=0.01, sigma=0.2, t_maturity=1.0)
    paths = simulate_gbm(n_paths=16_384, n_steps=26, antithetic=True,
                         seed=11, **sim)
    premium = premium_bs(**{"s0": sim["s0"], "strike": 100.0, "r": sim["r"],
                            "q": sim["q"], "sigma": sim["sigma"],
                            "t_maturity": sim["t_maturity"]})
    delta = delta_positions(paths, 100.0, sim["r"], sim["q"], sim["sigma"],
                            sim["t_maturity"])
    naked = hedge_pnl(paths, 100.0, premium, torch.zeros_like(delta), 0.0)
    for cost in np.linspace(0.0, 0.02, 13):
        fig, ax = new_ax(f"5 · deep hedging — cost rate {cost * 100:.2f}%")
        ax.hist(naked.numpy(), bins=70, alpha=0.45, color="#9ba1ad",
                label="no hedge")
        pnl = hedge_pnl(paths, 100.0, premium, delta, float(cost))
        ax.hist(pnl.numpy(), bins=70, alpha=0.6, color=BLUE,
                label="delta hedged")
        ax.axvline(cvar(pnl), color=GREEN, lw=2,
                   label=f"CVaR95 {cvar(pnl):.2f}")
        ax.set_xlim(-80, 25)
        ax.set_xlabel("P&L per year ($)")
        ax.set_ylabel("years")
        ax.legend(loc="upper left", fontsize=9)
        snap(fig, frames)


def end_card(frames: list[Image.Image]) -> None:
    fig = plt.figure(figsize=(W, H))
    fig.text(0.5, 0.58, "Deep Hedging & Pricing Simulator",
             ha="center", fontsize=20, fontweight="bold", color=ACCENT)
    fig.text(0.5, 0.44,
             "$ uv run --extra app streamlit run app/dashboard.py",
             ha="center", fontsize=12, family="monospace", color=BLUE)
    fig.text(0.5, 0.34, "every chart above is computed live from this repo",
             ha="center", fontsize=10, color="#7f8c99")
    snap(fig, frames, ms=2200)


def main() -> None:
    frames: list[tuple[Image.Image, int]] = []
    scene_paths(frames)
    scene_value(frames)
    scene_greeks(frames)
    scene_training(frames)
    scene_hedging(frames)
    end_card(frames)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    images = [img.convert("RGB").quantize(colors=256, method=Image.MEDIANCUT)
              for img, _ in frames]
    durations = [ms for _, ms in frames]
    images[0].save(OUT, save_all=True, append_images=images[1:],
                   duration=durations, loop=0, optimize=True)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KiB, "
          f"{len(frames)} frames)")


if __name__ == "__main__":
    main()
