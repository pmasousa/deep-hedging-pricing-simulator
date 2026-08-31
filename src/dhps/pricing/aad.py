"""Autograd (AAD) Greeks for the Black-Scholes closed form — SP2 labels v2.

Five lines of autograd replace five pages of derived formulas: every Greek
here is ``torch.autograd.grad`` on ``bs_price``, and the test suite pins
the results to the analytic ``bs_greeks`` at 1e-10. This module is the
repo's thesis in miniature — derivatives come free from autograd.
"""

import torch

from dhps.pricing.black_scholes import bs_price


def bs_greeks_ad(
    s: float | torch.Tensor,
    strike: float | torch.Tensor,
    r: float | torch.Tensor = 0.05,
    q: float | torch.Tensor = 0.0,
    sigma: float | torch.Tensor = 0.2,
    t_maturity: float | torch.Tensor = 1.0,
    call: bool = True,
) -> dict[str, torch.Tensor]:
    """Black-Scholes Greeks via autograd on the closed-form price.

    Returns the same keys as the analytic ``bs_greeks`` plus ``dual_delta``
    (dV/dK), which the analytic pricer does not bother to expose. Inputs may
    be floats or broadcastable tensors; theta follows the same convention
    (per year, = -dV/dT).
    """
    params = dict(s=s, strike=strike, r=r, q=q, sigma=sigma, t=t_maturity)
    # every leaf broadcast to the full batch shape: scalar params shared across
    # samples would otherwise accumulate the SUM of per-sample grads, not the
    # per-sample values themselves
    shape = torch.broadcast_shapes(
        *(torch.as_tensor(v).shape for v in params.values()))
    leaf = {name: torch.broadcast_to(torch.as_tensor(v, dtype=torch.float64), shape)
            .clone().requires_grad_(True) for name, v in params.items()}
    price = bs_price(leaf["s"], leaf["strike"], leaf["r"], leaf["q"],
                     leaf["sigma"], leaf["t"], call=call)
    # grad of the sum = per-sample grads (each price depends only on its own inputs)
    first = torch.autograd.grad(price.sum(), list(leaf.values()), create_graph=True)
    gamma = torch.autograd.grad(first[0].sum(), leaf["s"])[0]
    delta, dual_delta, rho, _, vega, dv_dt = first
    # detached: these are labels/values, not graphs to keep differentiating
    return {"delta": delta.detach(), "gamma": gamma.detach(),
            "vega": vega.detach(), "theta": -dv_dt.detach(),
            "rho": rho.detach(), "dual_delta": dual_delta.detach()}
