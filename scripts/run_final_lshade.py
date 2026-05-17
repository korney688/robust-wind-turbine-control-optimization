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
    plot_adaptive_parameters,
    plot_archive_dynamics,
    plot_control_laws,
    plot_convergence,
    plot_population_diversity,
    plot_population_size,
    plot_power_curve,
)
from wind_robust_opt.experiments.run_benchmark import run_benchmark  # noqa: E402
from wind_robust_opt.io.result_schema import optimizer_result_to_dict  # noqa: E402
from wind_robust_opt.problem.bounds import BOUNDS  # noqa: E402
from wind_robust_opt.problem.constants import V_CUT_IN, V_CUT_OUT  # noqa: E402
from wind_robust_opt.problem.physics import turbine_power  # noqa: E402


RAW_DIR = REPO_ROOT / "results" / "raw" / "lshade"
FIGURE_DIR = REPO_ROOT / "results" / "figures" / "lshade"
SUMMARY_DIR = REPO_ROOT / "results" / "summary"
SUMMARY_PATH = SUMMARY_DIR / "lshade_summary.json"

GENERATED_FIGURES = [
    "convergence.png",
    "archive_dynamics.png",
    "population_diversity.png",
    "population_size.png",
    "adaptive_F.png",
    "adaptive_CR.png",
    "successful_updates.png",
    "power_curve.png",
    "control_laws.png",
]


def _validation_checks(result: dict) -> dict:
    diagnostics = result["metadata"]["generation_diagnostics"]
    best_values = [entry["best_J"] for entry in diagnostics]
    diversities = [entry["population_diversity"] for entry in diagnostics]
    archive_sizes = [entry["archive_size"] for entry in diagnostics]
    successful_updates = [entry["successful_updates"] for entry in diagnostics]
    theta_best = np.asarray(result["theta_best"], dtype=float)

    wind_speed = np.linspace(V_CUT_IN, V_CUT_OUT, 200)
    power = turbine_power(wind_speed, theta_best)

    return {
        "best_J_nonincreasing": all(
            next_value <= value
            for value, next_value in zip(best_values, best_values[1:])
        ),
        "diversity_nonzero": bool(diversities[-1] > 1e-12),
        "archive_used": bool(max(archive_sizes) > 0),
        "adaptive_updates_present": bool(
            any(value > 0 for value in successful_updates)
        ),
        "final_theta_inside_bounds": bool(
            np.all(theta_best >= BOUNDS[:, 0])
            and np.all(theta_best <= BOUNDS[:, 1])
        ),
        "power_curve_finite": bool(np.all(np.isfinite(power))),
    }


def _max_power_mw(result: dict) -> float:
    theta_best = np.asarray(result["theta_best"], dtype=float)
    wind_speed = np.linspace(V_CUT_IN, V_CUT_OUT, 200)
    power = turbine_power(wind_speed, theta_best)
    return float(np.max(power) / 1_000_000.0)


def _save_plots(result: dict) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    plot_specs = [
        ("convergence.png", plot_convergence),
        ("archive_dynamics.png", plot_archive_dynamics),
        ("population_diversity.png", plot_population_diversity),
        ("population_size.png", plot_population_size),
        ("power_curve.png", plot_power_curve),
        ("control_laws.png", plot_control_laws),
    ]

    for filename, plotter in plot_specs:
        fig, _ = plotter(result, FIGURE_DIR / filename)
        plt.close(fig)

    for fig, _ in plot_adaptive_parameters(result, FIGURE_DIR):
        plt.close(fig)


def _summary_payload(best_result: dict) -> dict:
    diagnostics = best_result["metadata"]["generation_diagnostics"]
    validation_checks = _validation_checks(best_result)

    return {
        "best_final_J": float(best_result["final_J"]),
        "theta_best": [float(value) for value in best_result["theta_best"]],
        "runtime_sec": float(best_result["runtime_sec"]),
        "best_run_idx": int(best_result["run_id"]),
        "final_population_size": int(
            best_result["metadata"]["final_population_size"]
        ),
        "archive_final_size": int(diagnostics[-1]["archive_size"]),
        "max_power_MW": _max_power_mw(best_result),
        "validation_checks": validation_checks,
        "generated_figures": list(GENERATED_FIGURES),
    }


def _print_summary(summary: dict, best_result: dict) -> None:
    diagnostics = best_result["metadata"]["generation_diagnostics"]
    total_population_reductions = sum(
        entry["population_reductions"]
        for entry in diagnostics
    )

    print("=" * 60)
    print("L-SHADE FINAL SUMMARY")
    print("=" * 60)
    print(f"best_run_idx: {summary['best_run_idx']}")
    print(f"best_final_J: {summary['best_final_J']}")
    print(f"theta_best: {summary['theta_best']}")
    print(f"runtime_sec: {summary['runtime_sec']}")
    print(
        "initial_population_size: "
        f"{best_result['metadata']['initial_population_size']}"
    )
    print(f"final_population_size: {summary['final_population_size']}")
    print(f"total_population_reductions: {total_population_reductions}")
    print(f"archive_final_size: {summary['archive_final_size']}")
    print(f"max_power_MW: {summary['max_power_MW']}")
    print()
    print("Validation checks:")
    for name, value in summary["validation_checks"].items():
        print(f"- {name}: {value}")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    results = run_benchmark(output_dir=RAW_DIR)
    best_result = min(results, key=lambda item: item.final_J)
    best_payload = optimizer_result_to_dict(best_result)

    _save_plots(best_payload)

    summary = _summary_payload(best_payload)
    with SUMMARY_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    _print_summary(summary, best_payload)
    print()
    print(f"JSON results: {RAW_DIR}")
    print(f"Figures: {FIGURE_DIR}")
    print(f"Summary JSON: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
