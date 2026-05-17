# Robust Wind Turbine Control Optimization

Robust optimization project for wind turbine control under stochastic wind. The codebase is being moved from notebooks into a package with one shared problem definition, one objective, deterministic Monte Carlo sampling, and comparable optimizer outputs.

## Problem

Power model:

```text
P(v) = 0.5 * rho * A * Cp(lambda, beta) * v^3
lambda = omega * R / v
v ~ Weibull(k, c)
```

Control policy:

```text
beta(v) = a0 + a1 * (v - v_rated) + a2 * (v - v_rated)^2
omega(v) = b0 + b1 * v
theta = (a0, a1, a2, b0, b1)
```

## Objective

All methods must optimize exactly:

```text
J(theta) =
    - E[P]
    + alpha * Var(P)
    + gamma * E[max(0, P - P_rated)^2]
    + delta * penalty
```

The objective must receive a fixed Monte Carlo sample:

```text
wind_samples = generate_wind_samples(seed=MC_SEED)
J = objective(theta, wind_samples)
```

Optimizers must not sample wind internally.

## Structure

```text
configs/
  experiment_config.json
  methods_config.json
  problem_config.json
external/
  cmaes.ipynb
  SPSO-2011.ipynb
  cmaes_results.json
  spso2011_results.json
results/
scripts/
tests/
src/
  wind_robust_opt/
    problem/
      constants.py
      bounds.py
      objective.py
      wind_distribution.py
      parameter_names.py
    optimizers/
      base.py
      lshade_optimizer.py
    analysis/
    experiments/
    io/
      result_schema.py
```

## Methods

- `CMA-ES`: legacy notebook result exists in `external/cmaes_results.json`; package integration is still needed.
- `SPSO-2011`: legacy notebook result exists in `external/spso2011_results.json`; package integration is still needed.
- `L-SHADE`: placeholder file exists at `src/wind_robust_opt/optimizers/lshade_optimizer.py`; implementation is pending.
- `LLM optimizer`: planned comparison method; no package implementation is present yet.

Every method must use shared constants, shared bounds, shared objective, shared `MC_SEED`, shared history format, and shared JSON schema.

## Reproducibility

Canonical values live in `src/wind_robust_opt/problem/constants.py` and `src/wind_robust_opt/problem/bounds.py`.

```text
RHO = 1.225
ROTOR_RADIUS = 50.0
P_RATED = 5_000_000
V_RATED = 12.0
CP_MAX = 16/27
WEIBULL_K = 2.0
WEIBULL_C = 8.0
V_CUT_IN = 3.0
V_CUT_OUT = 25.0
BETA_MIN = 0.0
BETA_MAX = 30.0
OMEGA_MIN = 0.0
OMEGA_MAX = 2.0
N_MC_SAMPLES = 1000
MC_SEED = 999
ALPHA = 0.05
GAMMA = 10.0
DELTA = 10.0
```

```text
BOUNDS =
[[ 0.0, 30.0],
 [-5.0,  5.0],
 [-1.0,  1.0],
 [ 0.0,  2.0],
 [ 0.0,  0.2]]
```

Run seeds control optimizer randomness only. `MC_SEED` controls wind sampling only.

Known audit issue: `configs/problem_config.json` currently duplicates and diverges from canonical constants (`n_mc_samples`, `alpha`, `delta`). Treat `constants.py` and `bounds.py` as authoritative until the config is removed or synchronized.

## JSON Result Schema

Each optimizer run must export one `OptimizerResult`:

```json
{
  "method": "L-SHADE",
  "run_id": 0,
  "seed": 202401,
  "mc_seed": 999,
  "dimension": 5,
  "max_evals": 5000,
  "n_evals": 5000,
  "runtime_sec": 0.0,
  "final_J": 0.0,
  "theta_best": [0.0, 0.0, 0.0, 0.0, 0.0],
  "history": [
    {
      "eval": 1,
      "best_J": 0.0,
      "theta_best": [0.0, 0.0, 0.0, 0.0, 0.0]
    }
  ],
  "metadata": {}
}
```

Legacy files in `external/` are aggregate notebook outputs and do not match this per-run schema yet.

## Run Experiments

The unified runner is not implemented yet. Intended command:

```powershell
python -m wind_robust_opt.experiments.run_benchmark
```

Runner requirements:

- generate `wind_samples` once from `MC_SEED`;
- pass the same objective callable to every optimizer;
- pass canonical `BOUNDS` to every optimizer;
- write one JSON file per run using the unified schema.

## Run Final L-SHADE Pipeline

Run the frozen L-SHADE optimizer, export per-run JSON, generate all diagnostic
and physical validation plots, and write the final summary:

```powershell
python scripts/run_final_lshade.py
```

Generated outputs:

- `results/raw/lshade/`: per-run JSON files
- `results/figures/lshade/`: convergence, archive, diversity, population size,
  adaptive F/CR, power curve, and control-law plots
- `results/summary/lshade_summary.json`: best-run summary and validation checks

## Plot Convergence

The convergence analysis module is not implemented yet. Intended command:

```powershell
python -m wind_robust_opt.analysis.plot_convergence
```

Plots must consume only unified `history` entries with `eval`, `best_J`, and `theta_best`.

## Statistical Comparison

The statistical comparison module is not implemented yet. Intended command:

```powershell
python -m wind_robust_opt.analysis.compare_methods
```

Comparison must use final `J` values from repeated runs produced by the same benchmark runner, same objective, same bounds, and same Monte Carlo sample.
