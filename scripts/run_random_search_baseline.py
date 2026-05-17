from pathlib import Path
import json
import sys

import matplotlib
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from wind_robust_opt.analysis.lshade_diagnostics import (  # noqa: E402
    plot_control_laws,
    plot_power_curve,
)
from wind_robust_opt.experiments.config_loader import (  # noqa: E402
    load_experiment_config,
)
from wind_robust_opt.io.result_schema import (  # noqa: E402
    optimizer_result_to_dict,
    save_result_json,
)
from wind_robust_opt.optimizers.random_search import RandomSearchOptimizer  # noqa: E402
from wind_robust_opt.problem.bounds import BOUNDS  # noqa: E402
from wind_robust_opt.problem.constants import MC_SEED, V_CUT_IN, V_CUT_OUT  # noqa: E402
from wind_robust_opt.problem.objective import RobustWindObjective  # noqa: E402
from wind_robust_opt.problem.physics import turbine_power  # noqa: E402
from wind_robust_opt.problem.wind_distribution import generate_wind_samples  # noqa: E402


RAW_DIR = REPO_ROOT / "results" / "raw" / "random_search"
FIGURE_DIR = REPO_ROOT / "results" / "figures" / "random_search"
SUMMARY_DIR = REPO_ROOT / "results" / "summary"
SUMMARY_PATH = SUMMARY_DIR / "random_search_summary.json"

GENERATED_FIGURES = [
    "convergence.png",
    "power_curve.png",
    "control_laws.png",
]


def _plot_convergence(result: dict, save_path: str | Path) -> None:
    history = result["history"]
    evals = [entry["eval"] for entry in history]
    best_J = [entry["best_J"] for entry in history]

    fig, ax = plt.subplots()
    ax.plot(evals, best_J, label="best_J")
    ax.set_xlabel("Objective evaluations")
    ax.set_ylabel("Objective value")
    ax.set_title("Random Search convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def _max_power_mw(result: dict) -> float:
    theta_best = np.asarray(result["theta_best"], dtype=float)
    wind_speed = np.linspace(V_CUT_IN, V_CUT_OUT, 200)
    power = turbine_power(wind_speed, theta_best)
    return float(np.max(power) / 1_000_000.0)


def _validation_checks(result: dict) -> dict:
    history = result["history"]
    best_values = [entry["best_J"] for entry in history]
    theta_best = np.asarray(result["theta_best"], dtype=float)
    wind_speed = np.linspace(V_CUT_IN, V_CUT_OUT, 200)
    power = turbine_power(wind_speed, theta_best)

    return {
        "best_J_nonincreasing": all(
            next_value <= value
            for value, next_value in zip(best_values, best_values[1:])
        ),
        "final_theta_inside_bounds": bool(
            np.all(theta_best >= BOUNDS[:, 0])
            and np.all(theta_best <= BOUNDS[:, 1])
        ),
        "power_curve_finite": bool(np.all(np.isfinite(power))),
    }


def _save_plots(result: dict) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    _plot_convergence(result, FIGURE_DIR / "convergence.png")

    fig, _ = plot_power_curve(result, FIGURE_DIR / "power_curve.png")
    plt.close(fig)

    fig, _ = plot_control_laws(result, FIGURE_DIR / "control_laws.png")
    plt.close(fig)


def _summary_payload(best_result: dict, config: dict) -> dict:
    return {
        "best_final_J": float(best_result["final_J"]),
        "theta_best": [float(value) for value in best_result["theta_best"]],
        "runtime_sec": float(best_result["runtime_sec"]),
        "best_run_idx": int(best_result["run_id"]),
        "max_power_MW": _max_power_mw(best_result),
        "validation_checks": _validation_checks(best_result),
        "generated_figures": list(GENERATED_FIGURES),
        "n_runs": int(config["n_runs"]),
        "max_evals": int(config["max_evals"]),
        "run_seeds": [
            int(seed)
            for seed in config["run_seeds"][: int(config["n_runs"])]
        ],
    }


def _print_summary(summary: dict) -> None:
    print("=" * 60)
    print("RANDOM SEARCH SUMMARY")
    print("=" * 60)
    print(f"best_run_idx: {summary['best_run_idx']}")
    print(f"best_final_J: {summary['best_final_J']}")
    print(f"theta_best: {summary['theta_best']}")
    print(f"runtime_sec: {summary['runtime_sec']}")
    print(f"max_power_MW: {summary['max_power_MW']}")
    print()
    print("Validation checks:")
    for name, value in summary["validation_checks"].items():
        print(f"- {name}: {value}")


def main() -> None:
    config = load_experiment_config()
    run_seeds = config["run_seeds"][: config["n_runs"]]

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    for old_result in RAW_DIR.glob("run_*.json"):
        old_result.unlink()

    wind_samples = generate_wind_samples(seed=MC_SEED)
    objective = RobustWindObjective(wind_samples)
    optimizer = RandomSearchOptimizer(mc_seed=MC_SEED)

    results = []
    for run_id, seed in enumerate(run_seeds):
        result = optimizer.optimize(
            objective=objective,
            bounds=BOUNDS,
            seed=seed,
            max_evals=config["max_evals"],
            run_id=run_id,
        )
        save_result_json(result, RAW_DIR / f"run_{run_id:03d}.json")
        results.append(result)

    best_result = min(results, key=lambda item: item.final_J)
    best_payload = optimizer_result_to_dict(best_result)

    _save_plots(best_payload)

    summary = _summary_payload(best_payload, config)
    with SUMMARY_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    _print_summary(summary)
    print()
    print(f"JSON results: {RAW_DIR}")
    print(f"Figures: {FIGURE_DIR}")
    print(f"Summary JSON: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
