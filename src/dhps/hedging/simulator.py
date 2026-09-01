"""Discrete-time hedging simulator — SP5.

Book: SHORT one European call, hedged by trading the underlying at ``m``
equally spaced dates with proportional transaction costs ``c`` on traded
notional. P&L convention: higher is better, undiscounted (no cash interest
— stated simplification, fine at these horizons).

    pnl = premium - payoff + sum_i d_i (S_{i+1} - S_i)
          - c * [ S_0 |d_0| + sum_{i>=1} S_i |d_i - d_{i-1}| ]

``d_i`` is the stock position held over [t_i, t_{i+1}). Everything is a
(n_paths, m) tensor op — no python path loop — so the same code serves
hand-written strategies (delta, banded) and the learned policy.
"""

import math

import torch

from dhps.pricing.black_scholes import bs_price


def hedge_pnl(paths: torch.Tensor, strike: float, premium: float,
              deltas: torch.Tensor, cost_rate: float) -> torch.Tensor:
    """Terminal P&L per path for a given position schedule ``deltas`` (n, m)."""
    gains = (deltas * (paths[:, 1:] - paths[:, :-1])).sum(dim=1)
    trades = torch.cat([deltas[:, :1], deltas[:, 1:] - deltas[:, :-1]], dim=1)
    costs = cost_rate * (trades.abs() * paths[:, :-1]).sum(dim=1)
    payoff = torch.clamp(paths[:, -1] - strike, min=0.0)
    return premium - payoff + gains - costs


def delta_positions(paths: torch.Tensor, strike: float, r: float, q: float,
                    sigma: float, t_maturity: float) -> torch.Tensor:
    """BS delta at each rebalance date, (n, m) — the classical strategy."""
    from dhps.pricing.black_scholes import bs_greeks

    m = paths.shape[1] - 1
    ttms = t_maturity * (1.0 - torch.arange(m, dtype=paths.dtype) / m)
    greeks = bs_greeks(paths[:, :-1], strike, r, q, sigma, ttms)
    return greeks["delta"]


def banded_positions(target: torch.Tensor, band: float) -> torch.Tensor:
    """Hold-and-rebalance-only-past-``band``: the hand-crafted cousin of a
    learned policy — trades only when the target deviates enough to pay
    for it."""
    n, m = target.shape
    current = torch.zeros(n, dtype=target.dtype)
    out = []
    for j in range(m):
        trade = (target[:, j] - current).abs() > band
        current = torch.where(trade, target[:, j], current)
        out.append(current)
    return torch.stack(out, dim=1)


def cvar(pnl: torch.Tensor, alpha: float = 0.05) -> float:
    """Mean of the worst ``alpha`` tail of P&L (average of worst 5% = CVaR95)."""
    k = max(1, int(math.ceil(alpha * pnl.shape[0])))
    return float(pnl.sort().values[:k].mean())


def entropic_risk(pnl: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    """(1/lambd) log E[exp(-lambd * P&L)] — penalizes the loss tail.

    Differentiable (logsumexp); this is the training objective for the
    deep-hedging policy. Value <= CVaR in pessimism, >= mean in sign
    convention: SMALLER is safer (a certain P&L of x maps to -x).
    """
    n = pnl.shape[0]
    return (torch.logsumexp(-lambd * pnl, dim=0) - math.log(n)) / lambd


def premium_bs(s0: float, strike: float, r: float, q: float, sigma: float,
               t_maturity: float) -> float:
    """Fair premium charged for the short call (analytic BS)."""
    return float(bs_price(torch.tensor([s0], dtype=torch.float64), strike,
                          r, q, sigma, t_maturity)[0])
