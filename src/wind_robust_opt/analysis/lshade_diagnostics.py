from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from wind_robust_opt.io.result_schema import load_result_json
from wind_robust_opt.problem.constants import V_CUT_IN, V_CUT_OUT
from wind_robust_opt.problem.control_laws import beta_control, omega_control
from wind_robust_opt.problem.physics import turbine_power


def load_result(path: str | Path) -> dict:
    return load_result_json(path)


def _diagnostics(result: dict) -> list[dict]:
    diagnostics = result.get("metadata", {}).get("generation_diagnostics", [])
    if not diagnostics:
        raise ValueError("result metadata does not contain generation_diagnostics")
    return diagnostics


def _save_if_requested(fig, save_path: str | Path | None) -> None:
    if save_path is None:
        return

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")


def plot_convergence(result: dict, save_path: str | Path | None = None):
    diagnostics = _diagnostics(result)
    evals = [entry["eval"] for entry in diagnostics]
    best_J = [entry["best_J"] for entry in diagnostics]
    mean_J = [entry["mean_J"] for entry in diagnostics]

    fig, ax = plt.subplots()
    ax.plot(evals, best_J, label="best_J")
    ax.plot(evals, mean_J, label="mean_J")
    ax.set_xlabel("Objective evaluations")
    ax.set_ylabel("Objective value")
    ax.set_title("L-SHADE convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)

    _save_if_requested(fig, save_path)
    return fig, ax


def plot_archive_dynamics(result: dict, save_path: str | Path | None = None):
    diagnostics = _diagnostics(result)
    evals = [entry["eval"] for entry in diagnostics]
    archive_size = [entry["archive_size"] for entry in diagnostics]

    fig, ax = plt.subplots()
    ax.plot(evals, archive_size)
    ax.set_xlabel("Objective evaluations")
    ax.set_ylabel("Archive size")
    ax.set_title("L-SHADE archive dynamics")
    ax.grid(True, alpha=0.3)

    _save_if_requested(fig, save_path)
    return fig, ax


def plot_population_diversity(result: dict, save_path: str | Path | None = None):
    diagnostics = _diagnostics(result)
    evals = [entry["eval"] for entry in diagnostics]
    diversity = [entry["population_diversity"] for entry in diagnostics]

    fig, ax = plt.subplots()
    ax.plot(evals, diversity)
    ax.set_xlabel("Objective evaluations")
    ax.set_ylabel("Mean per-dimension std")
    ax.set_title("L-SHADE population diversity")
    ax.grid(True, alpha=0.3)

    _save_if_requested(fig, save_path)
    return fig, ax


def plot_adaptive_parameters(result: dict, save_dir: str | Path | None = None):
    diagnostics = _diagnostics(result)
    evals = [entry["eval"] for entry in diagnostics]
    mean_F = [entry["mean_F"] for entry in diagnostics]
    memory_F_mean = [entry["memory_F_mean"] for entry in diagnostics]
    mean_CR = [entry["mean_CR"] for entry in diagnostics]
    memory_CR_mean = [entry["memory_CR_mean"] for entry in diagnostics]
    successful_updates = [entry["successful_updates"] for entry in diagnostics]

    save_dir = None if save_dir is None else Path(save_dir)

    fig_F, ax_F = plt.subplots()
    ax_F.plot(evals, mean_F, label="mean_F")
    ax_F.plot(evals, memory_F_mean, label="memory_F_mean")
    ax_F.set_xlabel("Objective evaluations")
    ax_F.set_ylabel("F")
    ax_F.set_title("Adaptive F dynamics")
    ax_F.legend()
    ax_F.grid(True, alpha=0.3)
    _save_if_requested(
        fig_F,
        None if save_dir is None else save_dir / "adaptive_F.png",
    )

    fig_CR, ax_CR = plt.subplots()
    ax_CR.plot(evals, mean_CR, label="mean_CR")
    ax_CR.plot(evals, memory_CR_mean, label="memory_CR_mean")
    ax_CR.set_xlabel("Objective evaluations")
    ax_CR.set_ylabel("CR")
    ax_CR.set_title("Adaptive CR dynamics")
    ax_CR.legend()
    ax_CR.grid(True, alpha=0.3)
    _save_if_requested(
        fig_CR,
        None if save_dir is None else save_dir / "adaptive_CR.png",
    )

    fig_updates, ax_updates = plt.subplots()
    ax_updates.plot(evals, successful_updates)
    ax_updates.set_xlabel("Objective evaluations")
    ax_updates.set_ylabel("Successful updates")
    ax_updates.set_title("Successful parameter updates")
    ax_updates.grid(True, alpha=0.3)
    _save_if_requested(
        fig_updates,
        None if save_dir is None else save_dir / "successful_updates.png",
    )

    return [
        (fig_F, ax_F),
        (fig_CR, ax_CR),
        (fig_updates, ax_updates),
    ]


def plot_power_curve(result: dict, save_path: str | Path | None = None):
    theta_best = np.asarray(result["theta_best"], dtype=float)
    wind_speed = np.linspace(V_CUT_IN, V_CUT_OUT, 200)
    power = turbine_power(wind_speed, theta_best)

    fig, ax = plt.subplots()
    ax.plot(wind_speed, power)
    ax.set_xlabel("Wind speed, v [m/s]")
    ax.set_ylabel("Power, P(v) [W]")
    ax.set_title("Power curve for best theta")
    ax.grid(True, alpha=0.3)

    _save_if_requested(fig, save_path)
    return fig, ax


def plot_control_laws(result: dict, save_path: str | Path | None = None):
    theta_best = np.asarray(result["theta_best"], dtype=float)
    wind_speed = np.linspace(V_CUT_IN, V_CUT_OUT, 200)
    beta = beta_control(wind_speed, theta_best)
    omega = omega_control(wind_speed, theta_best)

    fig, ax_beta = plt.subplots()
    ax_omega = ax_beta.twinx()

    beta_line = ax_beta.plot(wind_speed, beta, label="beta(v)")
    omega_line = ax_omega.plot(wind_speed, omega, label="omega(v)", linestyle="--")

    ax_beta.set_xlabel("Wind speed, v [m/s]")
    ax_beta.set_ylabel("Pitch angle, beta(v) [deg]")
    ax_omega.set_ylabel("Rotor speed, omega(v) [rad/s]")
    ax_beta.set_title("Control laws for best theta")
    ax_beta.grid(True, alpha=0.3)

    lines = beta_line + omega_line
    labels = [line.get_label() for line in lines]
    ax_beta.legend(lines, labels)

    _save_if_requested(fig, save_path)
    return fig, (ax_beta, ax_omega)
