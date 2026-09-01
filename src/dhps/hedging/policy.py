"""Deep-hedging policy network and its training loop — SP5.

Buehler-style: a time-homogeneous policy maps (log-moneyness, time to
maturity, current position) to a target stock position in [0, 1] (sigmoid —
we are short a call, hedged long stock). The policy is trained by
backpropagating the entropic risk of terminal hedging P&L through entire
simulated trajectories — which is only possible because the SP2 simulator
keeps the whole path construction differentiable.

What the policy is free to learn that a fixed rule cannot: a STATE-
DEPENDENT no-trade band — rebalance aggressively near the strike where
gamma is large and delta moves fast, sit still deep ITM/OTM where delta
is flat and trading only pays costs.
"""

import time
from dataclasses import dataclass, field

import torch
from torch import nn

from dhps.hedging.simulator import cvar, delta_positions, entropic_risk, hedge_pnl, premium_bs
from dhps.models.mlp import make_mlp
from dhps.simulators.gbm import simulate_gbm


class HedgePolicy(nn.Module):
    """Per-date trading policy; shared weights across all rebalance dates."""

    def __init__(self, hidden: tuple[int, ...] = (32, 32)) -> None:
        super().__init__()
        self.net = make_mlp(n_in=3, hidden=hidden, n_out=1, activation="silu")

    def forward(self, log_moneyness: torch.Tensor, ttm: torch.Tensor,
                position: torch.Tensor) -> torch.Tensor:
        """(n,) features -> (n,) target position in [0, 1]."""
        x = torch.stack([log_moneyness, ttm, position], dim=1)
        return torch.sigmoid(self.net(x)).squeeze(1)


@dataclass(frozen=True)
class DeepHedgeConfig:
    n_paths: int = 32_768
    n_steps: int = 26
    s0: float = 100.0
    strike: float = 100.0
    r: float = 0.05
    q: float = 0.01
    sigma: float = 0.2
    t_maturity: float = 1.0
    cost_rate: float = 0.005
    lambd: float = 1.0
    hidden: tuple[int, ...] = (32, 32)
    lr: float = 1e-2
    epochs: int = 200
    seed: int = 7
    eval_paths: int = 8_192
    eval_seed: int = 1_234


@dataclass
class DeepHedgeResult:
    policy: HedgePolicy
    history: dict[str, list[float]] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    seconds: float = 0.0


def run_policy(policy: HedgePolicy, paths: torch.Tensor, strike: float,
               t_maturity: float) -> torch.Tensor:
    """Unroll the policy along every path -> positions (n, m)."""
    n, cols = paths.shape
    m = cols - 1
    position = torch.zeros(n, dtype=paths.dtype)
    out = []
    for j in range(m):
        ttm = t_maturity * (1.0 - j / m)
        tgt = policy(torch.log(paths[:, j] / strike),
                     torch.full_like(paths[:, j], ttm), position)
        position = tgt
        out.append(position)
    return torch.stack(out, dim=1)


def evaluate_policy(pnl: torch.Tensor) -> dict[str, float]:
    return {"mean": float(pnl.mean()), "std": float(pnl.std()),
            "cvar95": cvar(pnl, 0.05)}


def train_deep_hedge(cfg: DeepHedgeConfig) -> DeepHedgeResult:
    """Train the policy to minimize entropic risk of hedging P&L."""
    sim = dict(s0=cfg.s0, r=cfg.r, q=cfg.q, sigma=cfg.sigma,
               t_maturity=cfg.t_maturity)
    paths = simulate_gbm(n_paths=cfg.n_paths, n_steps=cfg.n_steps,
                         antithetic=True, seed=cfg.seed, **sim)
    eval_paths = simulate_gbm(n_paths=cfg.eval_paths, n_steps=cfg.n_steps,
                              antithetic=True, seed=cfg.eval_seed, **sim)
    premium = premium_bs(cfg.s0, cfg.strike, cfg.r, cfg.q, cfg.sigma,
                         cfg.t_maturity)

    torch.manual_seed(cfg.seed)
    policy = HedgePolicy(cfg.hidden)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    history: dict[str, list[float]] = {"train_risk": [], "eval_risk": [],
                                       "eval_cvar": []}
    best_risk, best_state = float("inf"), None
    t0 = time.perf_counter()

    for _epoch in range(cfg.epochs):
        positions = run_policy(policy, paths, cfg.strike, cfg.t_maturity)
        pnl = hedge_pnl(paths, cfg.strike, premium, positions, cfg.cost_rate)
        risk = entropic_risk(pnl, cfg.lambd)
        opt.zero_grad()
        risk.backward()
        opt.step()
        history["train_risk"].append(float(risk.detach()))

        # eval is not the bottleneck-free pass it looks like: run it sparsely
        if _epoch % 5 == 0 or _epoch == cfg.epochs - 1:
            with torch.no_grad():
                pos_e = run_policy(policy, eval_paths, cfg.strike, cfg.t_maturity)
                pnl_e = hedge_pnl(eval_paths, cfg.strike, premium, pos_e,
                                  cfg.cost_rate)
                risk_e = float(entropic_risk(pnl_e, cfg.lambd))
            history["eval_risk"].append(risk_e)
            history["eval_cvar"].append(cvar(pnl_e))
            if risk_e < best_risk:
                best_risk = risk_e
                best_state = {k: v.detach().clone() for k, v in
                              policy.state_dict().items()}

    if best_state is not None:
        policy.load_state_dict(best_state)
    with torch.no_grad():
        pos_e = run_policy(policy, eval_paths, cfg.strike, cfg.t_maturity)
        pnl_e = hedge_pnl(eval_paths, cfg.strike, premium, pos_e, cfg.cost_rate)
        trades = torch.cat([pos_e[:, :1], pos_e[:, 1:] - pos_e[:, :-1]], dim=1)
    metrics = {**evaluate_policy(pnl_e),
               # traded volume, not trade count: a continuous policy moves a
               # little every date; cost awareness is total |Δposition|
               "traded_volume": float(trades.abs().sum(dim=1).mean())}
    return DeepHedgeResult(policy=policy, history=history, metrics=metrics,
                           seconds=time.perf_counter() - t0)


def delta_baseline_metrics(cfg: DeepHedgeConfig) -> dict[str, float]:
    """Weekly BS delta on the SAME eval paths, for a fair comparison."""
    sim = dict(s0=cfg.s0, r=cfg.r, q=cfg.q, sigma=cfg.sigma,
               t_maturity=cfg.t_maturity)
    eval_paths = simulate_gbm(n_paths=cfg.eval_paths, n_steps=cfg.n_steps,
                              antithetic=True, seed=cfg.eval_seed, **sim)
    premium = premium_bs(cfg.s0, cfg.strike, cfg.r, cfg.q, cfg.sigma,
                         cfg.t_maturity)
    pos = delta_positions(eval_paths, cfg.strike, cfg.r, cfg.q, cfg.sigma,
                          cfg.t_maturity)
    pnl = hedge_pnl(eval_paths, cfg.strike, premium, pos, cfg.cost_rate)
    trades = torch.cat([pos[:, :1], pos[:, 1:] - pos[:, :-1]], dim=1)
    return {**evaluate_policy(pnl),
            "traded_volume": float(trades.abs().sum(dim=1).mean())}
