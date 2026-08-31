"""Speed harness — µs/price for DML inference vs closed form vs Monte Carlo.

The matched-error framing: a Monte Carlo estimate needs
n = (payoff_std / target_error)^2 paths for its CLT standard error to match
a learner's price MAE, so that is the honest n to time against. Antithetic
pairing tightens the real MC error further — n here is conservative.
"""

import math
import time


def time_fn(fn, repeats: int = 5, warmup: int = 2) -> float:
    """Median wall-clock seconds per call, after warmup."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return sorted(times)[len(times) // 2]


def mc_paths_for_error(payoff_std: float, target_error: float,
                       min_paths: int = 1_000) -> int:
    """CLT n so that SE(discounted payoff mean) <= target_error.

    Returned n is even: the reference engine runs antithetic, which requires
    an even path count.
    """
    if target_error <= 0:
        raise ValueError("target_error must be positive")
    n = max(min_paths, math.ceil((payoff_std / target_error) ** 2))
    return n + (n % 2)


def price_one_option_mc(n_paths: int, s0: float = 100.0, strike: float = 100.0,
                        r: float = 0.05, q: float = 0.01, sigma: float = 0.2,
                        t_maturity: float = 1.0, seed: int = 42) -> float:
    """MC reference price — the workload being timed (import kept local so
    this module has no simulator dependency at import time)."""
    from dhps.simulators.gbm import mc_european_price, simulate_gbm

    paths = simulate_gbm(n_paths=n_paths, n_steps=64, s0=s0, r=r, q=q,
                         sigma=sigma, t_maturity=t_maturity, antithetic=True,
                         seed=seed)
    return mc_european_price(paths, strike=strike, r=r, t_maturity=t_maturity)


def payoff_std(n_paths: int = 200_000, s0: float = 100.0, strike: float = 100.0,
               r: float = 0.05, q: float = 0.01, sigma: float = 0.2,
               t_maturity: float = 1.0, seed: int = 42) -> float:
    """Discounted per-path payoff std, unpaired — conservative on purpose:
    antithetic pairing tightens the real MC error below the CLT band used
    here, so the matched n overstates MC's cost rather than understating it.
    """
    from dhps.simulators.gbm import european_payoff, simulate_gbm

    paths = simulate_gbm(n_paths=n_paths, n_steps=64, s0=s0, r=r, q=q,
                         sigma=sigma, t_maturity=t_maturity, antithetic=True,
                         seed=seed)
    payoff = european_payoff(paths, strike)
    return math.exp(-r * t_maturity) * float(payoff.std(unbiased=True))
