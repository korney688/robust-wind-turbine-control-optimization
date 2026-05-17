from pathlib import Path

from wind_robust_opt.experiments.config_loader import load_experiment_config
from wind_robust_opt.io.result_schema import save_result_json
from wind_robust_opt.optimizers.lshade_optimizer import LShadeOptimizer
from wind_robust_opt.problem.bounds import BOUNDS
from wind_robust_opt.problem.constants import MC_SEED
from wind_robust_opt.problem.objective import RobustWindObjective
from wind_robust_opt.problem.wind_distribution import generate_wind_samples


OUTPUT_DIR = Path("results/raw/lshade")


def run_benchmark(
    output_dir: str | Path = OUTPUT_DIR,
    run_seeds: list[int] | None = None,
    max_evals: int | None = None,
):
    config = load_experiment_config()
    output_dir = Path(output_dir)

    if run_seeds is None:
        n_runs = int(config["n_runs"])
        run_seeds = list(config["run_seeds"][:n_runs])
    else:
        run_seeds = list(run_seeds)

    if max_evals is None:
        max_evals = int(config["max_evals"])

    wind_samples = generate_wind_samples(seed=MC_SEED)
    objective = RobustWindObjective(wind_samples)
    optimizer = LShadeOptimizer(mc_seed=MC_SEED)

    results = []
    for run_id, seed in enumerate(run_seeds):
        result = optimizer.optimize(
            objective=objective,
            bounds=BOUNDS,
            seed=seed,
            max_evals=max_evals,
            run_id=run_id,
        )

        save_result_json(
            result,
            output_dir / f"run_{run_id:03d}.json",
        )
        results.append(result)

    return results


if __name__ == "__main__":
    run_benchmark()
