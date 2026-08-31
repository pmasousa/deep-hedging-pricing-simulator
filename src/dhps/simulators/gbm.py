"""Geometric Brownian Motion path simulator.

Exact (log-space) discretization under the risk-neutral measure:

    dS_t = (r - q) S_t dt + sigma S_t dW_t
    =>  log S_{t+dt} = log S_t + (r - q - sigma^2 / 2) dt + sigma sqrt(dt) Z

Everything is torch-native so the same tensors feed the DML core later
(AAD comes for free with autograd) and the GPU is one ``device=`` away.
Antithetic variates are built by mirroring the normal draws, which also
makes seeded runs exactly reproducible.
"""

import torch


def simulate_gbm(
    n_paths: int,
    n_steps: int,
    s0: float = 100.0,
    r: float = 0.05,
    q: float = 0.0,
    sigma: float = 0.2,
    t_maturity: float = 1.0,
    antithetic: bool = True,
    device: str | torch.device = "cpu",
    seed: int | None = 42,
) -> torch.Tensor:
    """Simulate GBM paths; returns a float64 tensor of shape
    ``(n_paths, n_steps + 1)`` with column 0 equal to ``s0``.

    With ``antithetic=True`` the effective sample count is still ``n_paths``:
    the first ``n_paths / 2`` normals Z are paired with ``-Z``. ``n_paths``
    must therefore be even when antithetic is on.

    ``seed=None`` draws from the global generator (OS entropy): calls are
    NOT reproducible. Pass an int for bit-identical reruns.
    """
    if n_paths < 1 or n_steps < 1:
        raise ValueError("n_paths and n_steps must be >= 1")
    if sigma <= 0 or t_maturity <= 0 or s0 <= 0:
        raise ValueError("require s0 > 0, sigma > 0, t_maturity > 0")
    if antithetic and n_paths % 2 != 0:
        raise ValueError("antithetic=True requires an even n_paths")

    device = torch.device(device)
    dtype = torch.float64
    dt = t_maturity / n_steps

    half = n_paths // 2 if antithetic else n_paths
    if seed is not None:
        gen = torch.Generator(device="cpu").manual_seed(seed)
        z = torch.randn((half, n_steps), generator=gen, dtype=dtype)
    else:
        z = torch.randn((half, n_steps), dtype=dtype)
    z = z.to(device)
    if antithetic:
        z = torch.cat([z, -z], dim=0)

    drift = (r - q - 0.5 * sigma**2) * dt
    log_increments = drift + sigma * (dt**0.5) * z

    log_s0 = torch.log(torch.tensor(s0, dtype=dtype, device=device))
    log_s = torch.cat(
        [torch.zeros((n_paths, 1), dtype=dtype, device=device),
         torch.cumsum(log_increments, dim=1)],
        dim=1,
    )
    return torch.exp(log_s0 + log_s)


def european_payoff(paths: torch.Tensor, strike: float, call: bool = True) -> torch.Tensor:
    """European vanilla payoff on terminal prices ``paths[:, -1]``."""
    terminal = paths[:, -1]
    if call:
        return torch.clamp(terminal - strike, min=0.0)
    return torch.clamp(strike - terminal, min=0.0)


def mc_european_price(
    paths: torch.Tensor,
    strike: float,
    r: float = 0.05,
    t_maturity: float = 1.0,
    call: bool = True,
) -> float:
    """Discounted Monte Carlo estimate of a European vanilla price."""
    payoff = european_payoff(paths, strike, call=call)
    return float(torch.exp(torch.tensor(-r * t_maturity)) * payoff.mean())
