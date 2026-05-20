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
- `CMA-ES`: covariance matrix adaptation evolution strategy, gradient-free,
  self-adaptive scaling, increased population size for Monte Carlo noise
  robustness, notebook-based implementation.
- `SPSO-2011`: standard particle swarm optimisation with adaptive random
  topology and rotational invariance, notebook-based implementation.
- `LLM-OPRO`: prompt-based optimiser using a large language model,
  history of best (θ, J) pairs passed as context, no gradient or covariance
  model, notebook-based implementation.

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
  cmaes/
  lshade/
  opro/
  random_search/
  spso2011/
  summary/
notebooks/
  cmaes.ipynb
  LLMOpt.ipynb
  lshade_analysis.ipynb
  SPSO-2011.ipynb
scripts/
  run_final_lshade.py
  run_random_search_baseline.py
src/wind_robust_opt/
  analysis/
  experiments/
  io/
  optimizers/
  problem/
analysis/
  statistical_comparison.py
  plots/
  tables/
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

CMA-ES is implemented in `notebooks/cmaes.ipynb`. Open and run all cells
sequentially.

**Reference:** Hansen, N., & Ostermeier, A. (2001). *Completely Derandomized
Self-Adaptation in Evolution Strategies.* Evolutionary Computation, 9(2),
159–195. https://doi.org/10.1162/106365601750190398

CMA-ES adapts the full covariance matrix of the search distribution at each
generation, allowing the algorithm to learn the variable scaling and
correlations without gradient information.

Generated outputs:

- `results/cmaes/cmaes_results.json`

## Run SPSO-2011 Pipeline

SPSO-2011 is implemented in `notebooks/SPSO-2011.ipynb`. Open and run all
cells sequentially.

**Reference:** Zambrano-Bigiarini, M., Clerc, M., & Rojas, R. (2013).
*Standard Particle Swarm Optimisation 2011 at CEC-2013: A baseline for future
PSO improvements.* 2013 IEEE Congress on Evolutionary Computation, 2337–2344.
https://doi.org/10.1109/CEC.2013.6557848

SPSO-2011 is standardised version of Particle Swarm Optimisation.
Its two main advances over earlier PSO variants are an adaptive random topology
and rotational invariance: the next particle position is sampled from a
hypersphere centred at the gravity point G = (X + p̃ + l̃) / 3 rather than from
axis-aligned rectangles.

Generated outputs:

- `results/spso2011/spso2011_results.json`

## Run LLM-OPRO Pipeline

LLM-OPRO is implemented in `notebooks/LLMOpt.ipynb`. Open and run all cells
sequentially. Requires a local GPU to run the quantised LLM (4-bit NF4).

**Reference:** Yang, C., Wang, X., Lu, Y., Liu, H., Le, Q. V., Zhou, D., &
Chen, X. (2023). Large Language Models as Optimizers. arXiv:2309.03409.
https://arxiv.org/abs/2309.03409

LLM-OPRO (Optimization by PROmpting) uses a large language model as the
optimiser. At each iteration the algorithm passes the best solutions found so
far to the LLM inside a structured prompt and asks it to propose new candidate
parameters θ. No gradient or covariance model is maintained — the LLM itself
acts as the search heuristic based on the history of (θ, J) pairs.

Generated outputs:

- `results/opro/opro_results.json`

## Statistical Analysis

Comparison across all optimisers is performed in `analysis/`:

```powershell
python analysis/statistical_comparison.py
```

`statistical_comparison.py` runs
Shapiro-Wilk, Friedman, Nemenyi post-hoc, and Wilcoxon pairwise tests and
exports tables to `analysis/tables/`.