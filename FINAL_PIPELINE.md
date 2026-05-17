# Final L-SHADE Pipeline

This guide describes the final execution path for the frozen L-SHADE optimizer
and the Random Search baseline.

## Experiment Configuration

Experiment parameters are loaded from:

```text
configs/experiment_config.json
```

The config controls:

- `n_runs`
- `max_evals`
- `run_seeds`

The benchmark uses the canonical bounds, the shared robust objective, and one
Monte Carlo wind sample generated from `MC_SEED = 999`.

## Run Final Experiment

```powershell
python scripts/run_final_lshade.py
```

The command runs L-SHADE, saves per-run JSON, builds diagnostics and physical
validation plots, and writes a summary JSON.

## Generated Outputs

JSON results:

```text
results/raw/lshade/
```

Figures:

```text
results/figures/lshade/
```

Summary:

```text
results/summary/lshade_summary.json
```

## Generated Diagnostics

- convergence
- archive dynamics
- diversity
- adaptive F
- adaptive CR
- successful updates
- population reduction
- power curve
- control laws

## Validation Checks

- `best_J_nonincreasing`
- `archive_used`
- `adaptive_updates_present`
- `theta_inside_bounds`
- `power_curve_finite`

## Baseline

```powershell
python scripts/run_random_search_baseline.py
```

Random Search uses the same experiment configuration, objective, Monte Carlo
seed, bounds, run seeds, and evaluation budget as L-SHADE. It exports JSON,
minimal convergence and physical validation plots, and a summary JSON.

Baseline outputs:

```text
results/raw/random_search/
results/figures/random_search/
results/summary/random_search_summary.json
```

## Reserved Pipelines

## CMA-ES

## SPSO-2011
