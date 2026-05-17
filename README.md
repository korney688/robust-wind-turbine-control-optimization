# Robust Wind Turbine Control Optimization

This project optimizes wind turbine control parameters under stochastic wind.
The implementation uses a shared physical model, deterministic Monte Carlo wind
sampling, reproducible optimizer runs, structured JSON export, and diagnostic
plots.

## Problem Statement

The control policy maps wind speed to pitch angle and rotor speed:

```text
beta(v) = a0 + a1 * (v - v_rated) + a2 * (v - v_rated)^2
omega(v) = b0 + b1 * v
theta = (a0, a1, a2, b0, b1)
```

Wind speed is sampled from a Weibull distribution. The optimization task is to
find robust control parameters for expected power production while penalizing
variance, rated-power violations, and invalid parameter behavior.

## Objective

All optimizers use the same objective:

```text
J(theta) =
    - E[P / P_rated]
    + alpha * Var(P / P_rated)
    + gamma * E[max(0, P / P_rated - 1)^2]
    + delta * penalty(theta)
```

The objective receives fixed Monte Carlo wind samples generated once from
`MC_SEED`.

## Implemented Optimizers

- `L-SHADE`: current-to-pbest/1 mutation, external archive, adaptive F/CR
  success-history memories, linear population size reduction, diagnostics, and
  JSON export.
- `Random Search`: sanity-check baseline and reference for convergence quality.

## Reproducibility

Canonical constants and bounds live in `src/wind_robust_opt/problem/`.

- `MC_SEED = 999` controls Monte Carlo wind sampling.
- `configs/experiment_config.json` controls `n_runs`, `max_evals`, and
  optimizer run seeds.
- Optimizers use local NumPy generators seeded per run.
- Every run exports one JSON file through the shared `OptimizerResult` schema.

## Project Structure

```text
configs/
  experiment_config.json
results/
  figures/
  raw/
  summary/
scripts/
  run_final_lshade.py
  run_random_search_baseline.py
src/wind_robust_opt/
  analysis/
  experiments/
  io/
  optimizers/
  problem/
tests/
```

## Run Final L-SHADE Experiment

```powershell
python scripts/run_final_lshade.py
```

Generated outputs:

- `results/raw/lshade/`
- `results/figures/lshade/`
- `results/summary/lshade_summary.json`

## Run Diagnostics and Analysis

```powershell
python scripts/run_final_lshade.py
```

The final L-SHADE command runs the experiment, exports JSON, builds all
diagnostic plots, builds physical validation plots, and writes the final summary.

## Baseline Random Search

```powershell
python scripts/run_random_search_baseline.py
```

Random Search uses the same objective, bounds, `MC_SEED`, run seeds, and
evaluation budget as L-SHADE.

## Run CMA-ES Pipeline

## Run SPSO-2011 Pipeline
