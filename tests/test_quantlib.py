"""QuantLib cross-validation: independent engine vs our implementations.

BS: closed-form prices against QuantLib's AnalyticEuropeanEngine at 1e-8.
Heston: our full-truncation Euler Monte Carlo against QuantLib's
AnalyticHestonEngine, CLT band plus a discretization slack.

Both tests skip when QuantLib is absent (the [ql] extra is optional; CI
does not install it).
"""

import math

import pytest
import torch

ql = pytest.importorskip("QuantLib")

from dhps.pricing.black_scholes import bs_price  # noqa: E402
from dhps.simulators.gbm import mc_european_price  # noqa: E402
from dhps.simulators.heston import simulate_heston  # noqa: E402

K, R, Q, SIG, T = 100.0, 0.05, 0.01, 0.2, 1.0
HP = dict(v0=0.09, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)


def _term_structures():
    today = ql.Date.todaysDate()
    expiry = today + ql.Period(365, ql.Days)  # Act365 -> T exactly 1.0
    rf = ql.YieldTermStructureHandle(
        ql.FlatForward(today, R, ql.Actual365Fixed(), ql.Continuous))
    div = ql.YieldTermStructureHandle(
        ql.FlatForward(today, Q, ql.Actual365Fixed(), ql.Continuous))
    return today, expiry, rf, div


def _vanilla(today, expiry):
    payoff = ql.PlainVanillaPayoff(ql.Option.Call, K)
    return ql.EuropeanOption(payoff, ql.EuropeanExercise(expiry))


def test_bs_matches_quantlib_analytic():
    today, expiry, rf, div = _term_structures()
    vol_ts = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(today, ql.NullCalendar(), SIG,
                            ql.Actual365Fixed()))
    process = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(100.0)), div, rf, vol_ts)
    option = _vanilla(today, expiry)
    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))

    spots = torch.linspace(60.0, 150.0, 19, dtype=torch.float64)
    ours = bs_price(spots, K, R, Q, SIG, T)
    for i, s in enumerate(spots.tolist()):
        process = ql.BlackScholesMertonProcess(
            ql.QuoteHandle(ql.SimpleQuote(s)), div, rf, vol_ts)
        option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
        assert abs(float(ours[i]) - option.NPV()) < 1e-8, s


def test_heston_mc_matches_quantlib_analytic():
    today, expiry, rf, div = _term_structures()
    process = ql.HestonProcess(rf, div, ql.QuoteHandle(ql.SimpleQuote(100.0)),
                               HP["v0"], HP["kappa"], HP["theta"],
                               HP["xi"], HP["rho"])
    option = _vanilla(today, expiry)
    option.setPricingEngine(
        ql.AnalyticHestonEngine(ql.HestonModel(process), 192))
    ql_price = option.NPV()

    spots, _ = simulate_heston(n_paths=400_000, n_steps=128, s0=100.0,
                               r=R, q=Q, t_maturity=T, seed=13, **HP)
    mc = mc_european_price(spots, strike=K, r=R, t_maturity=T)
    payoff = torch.clamp(spots[:, -1] - K, min=0.0)
    se = math.exp(-R * T) * float(payoff.std(unbiased=True)) / math.sqrt(
        spots.shape[0])
    assert abs(mc - ql_price) < 5 * se + 0.10, (mc, ql_price, se)
