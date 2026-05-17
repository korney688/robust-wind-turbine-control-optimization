from pathlib import Path
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
    plot_adaptive_parameters,
    plot_archive_dynamics,
    plot_control_laws,
    plot_convergence,
    plot_population_diversity,
    plot_power_curve,
)
from wind_robust_opt.experiments.run_benchmark import run_benchmark  # noqa: E402
from wind_robust_opt.io.result_schema import optimizer_result_to_dict  # noqa: E402
from wind_robust_opt.problem.bounds import BOUNDS  # noqa: E402
from wind_robust_opt.problem.constants import V_CUT_IN, V_CUT_OUT  # noqa: E402
from wind_robust_opt.problem.physics import turbine_power  # noqa: E402


FIGURE_DIR = REPO_ROOT / "results" / "figures" / "lshade"
RAW_DIR = REPO_ROOT / "results" / "raw" / "lshade_skeleton"


def _validation_checks(result: dict) -> dict:
    diagnostics = result["metadata"]["generation_diagnostics"]
    best_values = [entry["best_J"] for entry in diagnostics]
    archive_sizes = [entry["archive_size"] for entry in diagnostics]
    diversities = [entry["population_diversity"] for entry in diagnostics]
    mean_F = [entry["mean_F"] for entry in diagnostics]
    mean_CR = [entry["mean_CR"] for entry in diagnostics]
    successful_updates = [entry["successful_updates"] for entry in diagnostics]
    theta_best = np.asarray(result["theta_best"], dtype=float)

    wind_speed = np.linspace(V_CUT_IN, V_CUT_OUT, 200)
    power = turbine_power(wind_speed, theta_best)

    return {
        "best_J_nonincreasing": all(
            next_value <= value
            for value, next_value in zip(best_values, best_values[1:])
        ),
        "archive_size_increases_or_stays": archive_sizes[-1] >= archive_sizes[0],
        "diversity_decreased": diversities[-1] <= diversities[0],
        "diversity_not_collapsed_to_zero": diversities[-1] > 1e-12,
        "theta_best_inside_bounds": bool(
            np.all(theta_best >= BOUNDS[:, 0])
            and np.all(theta_best <= BOUNDS[:, 1])
        ),
        "power_curve_finite_nonnegative": bool(
            np.all(np.isfinite(power)) and np.all(power >= 0.0)
        ),
        "F_dynamics_finite": bool(np.all(np.isfinite(mean_F))),
        "CR_dynamics_finite": bool(np.all(np.isfinite(mean_CR))),
        "has_successful_updates": any(value > 0 for value in successful_updates),
        "max_power_W": float(np.max(power)),
        "initial_diversity": float(diversities[0]),
        "final_diversity": float(diversities[-1]),
        "initial_archive_size": int(archive_sizes[0]),
        "final_archive_size": int(archive_sizes[-1]),
        "initial_mean_F": float(mean_F[0]),
        "final_mean_F": float(mean_F[-1]),
        "initial_mean_CR": float(mean_CR[0]),
        "final_mean_CR": float(mean_CR[-1]),
    }


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    results = run_benchmark(output_dir=RAW_DIR)
    best_result = min(results, key=lambda item: item.final_J)
    best_payload = optimizer_result_to_dict(best_result)

    plot_specs = [
        ("convergence.png", plot_convergence),
        ("archive_dynamics.png", plot_archive_dynamics),
        ("population_diversity.png", plot_population_diversity),
        ("power_curve.png", plot_power_curve),
        ("control_laws.png", plot_control_laws),
    ]

    for filename, plotter in plot_specs:
        fig, _ = plotter(best_payload, FIGURE_DIR / filename)
        plt.close(fig)

    for fig, _ in plot_adaptive_parameters(best_payload, FIGURE_DIR):
        plt.close(fig)

    checks = _validation_checks(best_payload)
    print(f"Best run: {best_result.run_id}")
    print(f"Best final_J: {best_result.final_J}")
    for name, value in checks.items():
        print(f"{name}: {value}")
    print(f"Figures saved to: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
