# Final Pipeline

Roadmap and audit state after moving notebook code into the package structure.

## Stage 1: Problem Formalization

Goal: define the physical model, control parameterization, wind model, constants, objective weights, and optimization bounds.

Status: partially complete.

Ready: canonical constants are centralized in `src/wind_robust_opt/problem/constants.py`; canonical bounds are centralized in `src/wind_robust_opt/problem/bounds.py`; parameter names are isolated in `parameter_names.py`.

Remaining: remove or synchronize `configs/problem_config.json`, because it duplicates problem values and currently diverges from canonical constants.

Risks: duplicate config values can silently change experiments if a runner reads JSON instead of `constants.py`.

## Stage 2: Core Physics Implementation

Goal: implement the turbine power model, `Cp(lambda, beta)`, control policy evaluation, operating range handling, clipping, and penalty terms in `problem/`.

Status: not complete.

Ready: package location exists; constants and bounds are available.

Remaining: implement physics functions in shared problem modules, not inside optimizers.

Risks: if physics is copied into optimizers, methods will no longer be comparable.

## Stage 3: Unified Objective

Goal: implement exactly one shared objective:

```text
J(theta) =
    - E[P]
    + alpha * Var(P)
    + gamma * E[max(0, P - P_rated)^2]
    + delta * penalty
```

Status: not complete.

Ready: `src/wind_robust_opt/problem/objective.py` exists.

Remaining: replace the placeholder with the canonical objective using precomputed `wind_samples`; keep Monte Carlo sampling outside the optimizer.

Risks: method-specific objectives or internal resampling would invalidate the benchmark.

## Stage 4: CMA-ES Integration

Goal: integrate CMA-ES as a package optimizer using the shared objective, canonical bounds, optimizer seed, history entries, and JSON schema.

Status: not complete.

Ready: notebook and legacy aggregate result exist in `external/cmaes.ipynb` and `external/cmaes_results.json`.

Remaining: move only optimizer orchestration into `optimizers/`; adapt output to one `OptimizerResult` per run.

Risks: the legacy JSON contains `n_runs`, `algo_seeds`, aggregate `final_J`, and raw histories, which do not match the required per-run schema.

## Stage 5: SPSO-2011 Integration

Goal: integrate SPSO-2011 as a package optimizer using the shared objective, canonical bounds, optimizer seed, history entries, and JSON schema.

Status: not complete.

Ready: notebook and legacy aggregate result exist in `external/SPSO-2011.ipynb` and `external/spso2011_results.json`.

Remaining: move optimizer logic into `optimizers/`; convert raw best-value histories to `HistoryEntry(eval, best_J, theta_best)`.

Risks: missing `theta_best` in raw histories can make convergence analysis incomplete unless the package runner records it during evaluation.

## Stage 6: L-SHADE Implementation

Goal: implement L-SHADE with the same objective callable, bounds, evaluation budget, seed policy, and result interface as all other methods.

Status: not complete.

Ready: placeholder file exists at `src/wind_robust_opt/optimizers/lshade_optimizer.py`.

Remaining: implement optimizer state, bounded candidate repair, archive logic, adaptive memories, evaluation counting, history recording, runtime measurement, and metadata export.

Risks: adaptive population and archive logic can hide extra objective calls unless `n_evals` is counted centrally.

## Stage 7: Unified Benchmark Runner

Goal: run all enabled methods across the same run seeds, same `MC_SEED`, same `BOUNDS`, same objective, and same `max_evals`.

Status: not complete.

Ready: `configs/experiment_config.json` and `configs/methods_config.json` exist.

Remaining: implement `src/wind_robust_opt/experiments/run_benchmark.py`; generate wind samples once per benchmark configuration; pass an objective callable into every optimizer.

Risks: `experiment_config.json` currently says `n_runs = 30` but lists only 5 `run_seeds`.

## Stage 8: JSON Export Consistency

Goal: export every run as one `OptimizerResult` with the required fields and typed histories.

Status: partially complete.

Ready: `HistoryEntry` and `OptimizerResult` dataclasses exist in `src/wind_robust_opt/optimizers/base.py`.

Remaining: add missing `runtime_sec` to `OptimizerResult`; implement `src/wind_robust_opt/io/result_schema.py`; import `OptimizerResult` before using it in type annotations; convert dataclasses and NumPy values to JSON-safe types.

Risks: `result_schema.py` currently references `OptimizerResult` without importing it and contains only a placeholder body.

## Stage 9: Convergence Analysis

Goal: plot comparable convergence curves from unified histories.

Status: not complete.

Ready: `src/wind_robust_opt/analysis/` directory exists.

Remaining: implement loaders and plotting functions that consume only unified JSON records.

Risks: mixing legacy raw histories with structured histories will make evaluation axes and best-theta traces ambiguous.

## Stage 10: Statistical Comparison

Goal: compare methods statistically using repeated final objective values produced under the same benchmark protocol.

Status: not complete.

Ready: legacy CMA-ES and SPSO-2011 final values exist as external reference data.

Remaining: implement comparison after unified result export is stable; use only package-produced per-run JSON for final conclusions.

Risks: legacy notebook outputs may have hidden differences in objective, bounds, or sampling and should not be mixed with final package runs without validation.

## Stage 11: Presentation Preparation

Goal: prepare final tables, convergence plots, method descriptions, reproducibility notes, and limitations for the course presentation.

Status: not complete.

Ready: project statement, package skeleton, and legacy notebook outputs exist.

Remaining: finish the unified pipeline, rerun all methods through the package runner, generate plots, generate statistical tables, and document the final reproducibility setup.

Risks: presenting notebook-derived and package-derived results together without schema and objective validation can lead to inconsistent conclusions.
