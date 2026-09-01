# Deep Hedging & Pricing Simulator

European options priced and risk-managed three independent ways, forced to
agree — then two models trained on top: a **differential ML pricer** (Huge &
Savine, 2020) and a **deep-hedging policy** under transaction costs
(Buehler et al., 2019). Torch-native end to end, float64, statistically
gated.

## The three routes to one Greek

1. **Analytic** — closed-form Black-Scholes (ground truth).
2. **Autograd (AAD)** — the same pricer differentiated by
   `torch.autograd.grad` instead of derived formulas.
3. **Pathwise Monte Carlo** — Greeks by backpropagating *through simulated
   GBM paths*, the mechanism differential ML and deep hedging both train
   with.

The validation suite pins routes 2 and 3 to route 1 at 1e-10 absolute
(CLT bands for anything Monte Carlo) — three unrelated code paths, so a bug
would have to exist in all three to slip through.

## Results

Differential training vs a price-only baseline, identical networks and data
(100k Sobol samples, 300 epochs, float64; full run reproducible via
`scripts/benchmark.py`):

| learner | price MAE | delta MAE | OOD price MAE | OOD delta MAE |
|---|---|---|---|---|
| **DML** | **0.053** | **0.0042** | **0.45** | **0.027** |
| baseline | 0.127 | 0.0122 | 1.33 | 0.032 |

Inference speed at matched error (Monte Carlo given the path count its CLT
standard error needs to match the learner's price accuracy — 73,544 paths):

| device | method | µs per price |
|---|---|---|
| cpu | Monte Carlo, matched error | 104,000 |
| cpu | DML, batched 100k options | 0.79 |
| cuda | DML, batched 100k options | 0.0039 |

Gamma comes out of the trained network despite never being a training label
— a second autograd pass through the differential structure it internalized.

Deep hedging: a policy network trained by backpropagating entropic risk
through simulated hedging trajectories beats weekly delta hedging on CVaR95
(−3.57 vs −4.24 at 1% costs) while moving less traded volume. Hedging does
not create profit — a fairly priced book has zero expected P&L by
construction; the policy's edge is paying less for the same protection.

## Quickstart

```bash
# install (CPU) — or add --extra gpu for CUDA builds
uv sync --extra dev

# test suite (42 gates: parity, CLT bands, regression gates)
uv run pytest -q

# interactive dashboard
uv run --extra app streamlit run app/dashboard.py

# train a learner
uv run python scripts/train.py --dml --epochs 300

# full benchmark suite (writes reports/benchmarks/)
uv run python scripts/benchmark.py --samples 100000 --epochs 300 --device cuda
```

## Structure

```
src/dhps/
  simulators/   GBM paths, differentiable end to end
  pricing/      Black-Scholes, autograd Greeks, pathwise MC Greeks
  datasets/     Sobol sampler, gradient labels, normalization
  models/       MLP, differential loss (the twin, torch-native)
  train/        one loop, two learners; run folders
  bench/        accuracy/OOD metrics, Greeks curves, matched-error speed
  hedging/      cost-aware P&L simulator, deep-hedging policy
app/            Streamlit dashboard (six sections, everything live)
scripts/        train.py, benchmark.py CLIs
tests/          statistical gates and regression gates
```

Building blocks are validated against ground truth (the closed form);
learned products are validated against baselines (delta hedging, price-only
netting) — regression gates in `tests/` enforce both on every commit.

## License

MIT — see [LICENSE](LICENSE).
