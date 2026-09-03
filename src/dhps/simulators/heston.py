"""Heston stochastic-volatility simulator — Mission 2, SP1 residue.

Full-truncation Euler (Lord, Koekkoek, van Dijk 2010) under the
risk-neutral measure:

    dS_t = (r - q) S_t dt + sqrt(v_t) S_t dW^S_t
    dv_t = kappa (theta - v_t) dt + xi sqrt(v_t) dW^V_t
    corr(dW^S, dW^V) = rho

The variance recursion evolves the hidden state (which may go negative)
while every coefficient uses v+ = max(v, 0); reported variances are v+,
positive by construction. That truncation IS the positivity handling —
with Feller-satisfied parameters (2 kappa theta >= xi^2) negative
excursions are rare and small.

Log-spot increments keep E[S_T] = S0 e^{(r-q)T} exact at any dt:
conditional on the variance path, each increment is N(-v dt/2, v dt),
so the martingale gate below is tight. Torch-native float64 like the
GBM engine; ``s0`` may carry requires_grad (gradients flow, with
subgradients at the truncation boundary — do not trust pathwise Greeks
through the boundary without reading Lord et al. on the bias).
"""

import torch


def simulate_heston(
    n_paths: int,
    n_steps: int,
    s0: float | torch.Tensor = 100.0,
    r: float = 0.05,
    q: float = 0.0,
    v0: float = 0.04,
    kappa: float = 2.0,
    theta: float = 0.04,
    xi: float = 0.3,
    rho: float = -0.7,
    t_maturity: float = 1.0,
    antithetic: bool = True,
    device: str | torch.device = "cpu",
    seed: int | None = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Simulate Heston paths; returns ``(spots, variances)``, both
    ``(n_paths, n_steps + 1)`` float64 with column 0 at inception.

    Antithetic mirrors both driving normals (valid for the correlated
    system), so ``n_paths`` must be even when ``antithetic=True``.
    """
    if n_paths < 1 or n_steps < 1:
        raise ValueError("n_paths and n_steps must be >= 1")
    bad = (v0 <= 0 or theta <= 0 or kappa <= 0 or xi <= 0
           or t_maturity <= 0 or not -1.0 < rho < 1.0)
    if bad:
        raise ValueError("require v0>0, theta>0, kappa>0, xi>0, "
                         "t_maturity>0, -1<rho<1")
    if torch.as_tensor(s0).numel() > 1:
        raise ValueError("s0 must be a scalar or 0-dim tensor here")
    if antithetic and n_paths % 2 != 0:
        raise ValueError("antithetic=True requires an even n_paths")

    device = torch.device(device)
    dtype = torch.float64
    dt = t_maturity / n_steps

    half = n_paths // 2 if antithetic else n_paths
    if seed is not None:
        gen = torch.Generator(device="cpu").manual_seed(seed)
        zv = torch.randn((half, n_steps), generator=gen, dtype=dtype)
        zo = torch.randn((half, n_steps), generator=gen, dtype=dtype)
    else:
        zv = torch.randn((half, n_steps), dtype=dtype)
        zo = torch.randn((half, n_steps), dtype=dtype)
    zv, zo = zv.to(device), zo.to(device)
    if antithetic:
        zv = torch.cat([zv, -zv], dim=0)
        zo = torch.cat([zo, -zo], dim=0)
    zs = rho * zv + (1.0 - rho**2) ** 0.5 * zo

    log_s = torch.log(torch.as_tensor(s0, dtype=dtype,
                                      device=device)).reshape(())
    v = torch.full((n_paths,), float(v0), dtype=dtype, device=device)
    spots = torch.empty((n_paths, n_steps + 1), dtype=dtype, device=device)
    variances = torch.empty_like(spots)
    spots[:, 0] = torch.exp(log_s)
    variances[:, 0] = v

    for j in range(n_steps):
        vp = torch.clamp(v, min=0.0)
        v = v + kappa * (theta - vp) * dt + xi * (vp * dt).sqrt() * zv[:, j]
        log_s = log_s + (r - q - 0.5 * vp) * dt \
            + (vp * dt).sqrt() * zs[:, j]
        spots[:, j + 1] = torch.exp(log_s)
        variances[:, j + 1] = torch.clamp(v, min=0.0)
    return spots, variances
