"""Pathwise MC Greeks — autograd through the simulated paths (SP2, labels v3).

The Savine trick in its purest form: the payoff is computed on simulated
paths that are a differentiable function of the inputs, so
``torch.autograd.grad`` of the discounted payoff mean yields the pathwise
Greek estimator. Delta and vega come out unbiased with per-path standard
errors (the ``_se`` keys) for CLT bands.

Pathwise gamma is deliberately NOT reported: the call payoff has a kink at
the strike, its second derivative is a Dirac, and autograd through
``clamp`` returns zero almost everywhere — the estimator is invalid, not
merely noisy. Likelihood-ratio or Malliavin weights would be required;
out of scope here.
"""

import math

import torch

from dhps.simulators.gbm import european_payoff, simulate_gbm


def pathwise_european_greeks(
    n_paths: int = 200_000,
    n_steps: int = 64,
    s0: float = 100.0,
    strike: float = 100.0,
    r: float = 0.05,
    q: float = 0.0,
    sigma: float = 0.2,
    t_maturity: float = 1.0,
    call: bool = True,
    antithetic: bool = True,
    seed: int | None = 42,
) -> dict[str, float]:
    """MC price + pathwise delta/vega with standard errors, one graph."""
    s0_leaf = torch.full((n_paths,), float(s0), dtype=torch.float64, requires_grad=True)
    sig_leaf = torch.full((n_paths,), float(sigma), dtype=torch.float64,
                          requires_grad=True)
    paths = simulate_gbm(
        n_paths=n_paths, n_steps=n_steps, s0=s0_leaf, r=r, q=q, sigma=sig_leaf,
        t_maturity=t_maturity, antithetic=antithetic, seed=seed,
    )
    payoff = european_payoff(paths, strike, call=call)
    df = math.exp(-r * t_maturity)
    # per-path leaves => grad of the sum is the per-path gradient vector
    per_delta, per_vega = torch.autograd.grad(payoff.sum(), (s0_leaf, sig_leaf))

    def _stat(per_path: torch.Tensor) -> tuple[float, float]:
        est = df * float(per_path.mean())
        se = df * float(per_path.std(unbiased=True)) / math.sqrt(n_paths)
        return est, se

    delta, delta_se = _stat(per_delta)
    vega, vega_se = _stat(per_vega)
    return {"price": df * float(payoff.detach().mean()),
            "delta": delta, "delta_se": delta_se,
            "vega": vega, "vega_se": vega_se}
