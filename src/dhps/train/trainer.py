"""Trainer for the baseline and differential learners — SP3.

One loop, two learners: ``differential=False`` trains on values only;
``differential=True`` adds the model's input-gradients to the cost (the
Savine twin). Everything runs in normalized space; headline metrics are
reported back in RAW units (delta MAE in spot-cents per dollar of spot)
so the two learners compare on business terms.

Run-folder doctrine: ``save_run`` writes config, history, and weights under
``reports/runs/<name>/<timestamp>/`` (gitignored).
"""

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import nn

from dhps.datasets.european import make_european_dataset
from dhps.datasets.normalize import Standardizer
from dhps.models.losses import differential_loss, value_and_grad
from dhps.models.mlp import make_mlp


@dataclass(frozen=True)
class TrainConfig:
    n_samples: int = 100_000
    hidden: tuple[int, ...] = (64, 64, 64)
    activation: str = "silu"
    lr: float = 1e-3
    batch_size: int = 8192
    epochs: int = 200
    lam: float = 1.0
    seed: int = 7


@dataclass
class TrainResult:
    model: nn.Module
    scaler: Standardizer
    history: dict[str, list[float]] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    best_epoch: int = 0
    seconds: float = 0.0


def train_model(cfg: TrainConfig, differential: bool) -> TrainResult:
    """Train one learner on the Sobol European dataset, seeded end to end."""
    data = make_european_dataset(n_samples=cfg.n_samples, seed=cfg.seed)
    scaler = Standardizer.fit(data["x_train"], data["y_train"])
    # g_scale[j] converts the raw d(price)/ds0 column to normalized space
    g_scale = scaler.grad_scale[0]

    xt = scaler.transform_x(data["x_train"])
    xv = scaler.transform_x(data["x_val"])
    yt = scaler.transform_y(data["y_train"])[:, :1]
    yv = scaler.transform_y(data["y_val"])[:, :1]
    gt = data["g_train"] * g_scale

    torch.manual_seed(cfg.seed)
    model = make_mlp(n_in=xt.shape[1], hidden=cfg.hidden,
                     activation=cfg.activation)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [],
                                       "val_value_mse": [], "val_delta_mae": []}
    best_val, best_state, best_epoch = float("inf"), None, 0
    gen = torch.Generator().manual_seed(cfg.seed + 1)
    t0 = time.perf_counter()

    for _epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(xt.shape[0], generator=gen)
        losses = []
        for idx in perm.split(cfg.batch_size):
            xb, yb, gb = xt[idx], yt[idx], gt[idx]
            if differential:
                yp, gp = value_and_grad(model, xb)
                loss = differential_loss(yp, gp, yb, gb, lam=cfg.lam)
            else:
                loss = nn.functional.mse_loss(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))

        model.eval()
        with torch.no_grad():
            val_value = float(nn.functional.mse_loss(model(xv), yv))
        # delta MAE in raw units for both learners (baseline gets it for free
        # via one extra autograd pass — no training signal, metric only)
        gp_val = torch.autograd.grad(model(xv.requires_grad_(True)).sum(), xv)[0]
        delta_mae = float((gp_val[:, 0] / g_scale[0] - data["g_val"][:, 0])
                          .abs().mean())
        val_loss = val_value + (delta_mae if differential else 0.0)

        history["train_loss"].append(sum(losses) / len(losses))
        history["val_loss"].append(val_loss)
        history["val_value_mse"].append(val_value)
        history["val_delta_mae"].append(delta_mae)
        if val_loss < best_val:
            best_val, best_epoch = val_loss, _epoch
            best_state = {k: v.detach().clone() for k, v in
                          model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    gp_best = torch.autograd.grad(model(xv.requires_grad_(True)).sum(), xv)[0]
    metrics = {
        "val_value_mse_norm": history["val_value_mse"][best_epoch],
        "val_delta_mae": float((gp_best[:, 0] / g_scale[0] - data["g_val"][:, 0])
                               .abs().mean()),
        "best_epoch": float(best_epoch),
    }
    return TrainResult(model=model, scaler=scaler, history=history,
                       metrics=metrics, best_epoch=best_epoch,
                       seconds=time.perf_counter() - t0)


def save_run(result: TrainResult, cfg: TrainConfig, differential: bool,
             run_dir: Path) -> Path:
    """Write config, history, and weights to a run folder."""
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {"config": asdict(cfg), "differential": differential,
               "metrics": result.metrics, "best_epoch": result.best_epoch,
               "seconds": result.seconds}
    (run_dir / "run.json").write_text(json.dumps(payload, indent=2))
    (run_dir / "history.json").write_text(json.dumps(result.history, indent=2))
    torch.save(result.model.state_dict(), run_dir / "weights.pt")
    return run_dir
