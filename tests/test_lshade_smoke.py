import math

import numpy as np

from wind_robust_opt.io.result_schema import load_result_json, save_result_json
from wind_robust_opt.optimizers.base import OptimizerResult
from wind_robust_opt.optimizers.lshade_optimizer import LShadeOptimizer


def test_lshade_smoke_and_json_export(tmp_path):
    bounds = np.array(
        [
            [0.0, 30.0],
            [-5.0, 5.0],
            [-1.0, 1.0],
            [0.0, 2.0],
            [0.0, 0.2],
        ],
        dtype=float,
    )

    objective_calls = 0

    def flat_objective(theta):
        nonlocal objective_calls
        objective_calls += 1
        theta = np.asarray(theta, dtype=float)
        return float(np.sum(theta * 0.0))

    max_evals = 25
    optimizer = LShadeOptimizer(
        population_size=10,
        F=0.5,
        CR=0.9,
        p_best_rate=0.2,
        memory_size=6,
        mc_seed=123,
    )

    result = optimizer.optimize(
        objective=flat_objective,
        bounds=bounds,
        seed=202401,
        max_evals=max_evals,
        run_id=0,
    )

    assert isinstance(result, OptimizerResult)
    assert result.n_evals == objective_calls
    assert result.n_evals == max_evals
    assert result.history
    assert optimizer.debug_info["archive_size"] > 0
    assert optimizer.debug_info["archive_size"] <= optimizer.population_size
    assert optimizer.debug_info["pbest_selection_count"] > 0
    assert math.isfinite(result.final_J)

    theta_best = np.asarray(result.theta_best, dtype=float)
    assert np.all(theta_best >= bounds[:, 0])
    assert np.all(theta_best <= bounds[:, 1])

    output_path = tmp_path / "lshade_result.json"
    save_result_json(result, output_path)
    loaded = load_result_json(output_path)

    assert loaded["method"] == "LShade"
    assert loaded["n_evals"] == result.n_evals
    assert loaded["theta_best"] == result.theta_best
    assert loaded["metadata"]["algorithm"] == "L-SHADE-stage3-no-LPSR"
    assert loaded["metadata"]["mutation"] == "current-to-pbest/1"
    assert loaded["metadata"]["archive_enabled"] is True
    assert loaded["metadata"]["adaptive_F"] is True
    assert loaded["metadata"]["adaptive_CR"] is True
    assert loaded["metadata"]["memory_size"] == 6
    diagnostics = loaded["metadata"]["generation_diagnostics"]
    assert diagnostics
    assert set(diagnostics[0]) == {
        "generation",
        "eval",
        "best_J",
        "mean_J",
        "std_J",
        "archive_size",
        "population_diversity",
        "mean_F",
        "std_F",
        "mean_CR",
        "std_CR",
        "memory_F_mean",
        "memory_CR_mean",
        "memory_index",
        "successful_updates",
    }
    assert diagnostics[0]["generation"] == 0
    assert diagnostics[-1]["eval"] == result.n_evals
    assert any(entry["successful_updates"] > 0 for entry in diagnostics)
    assert all(math.isfinite(entry["mean_F"]) for entry in diagnostics)
    assert all(math.isfinite(entry["std_F"]) for entry in diagnostics)
    assert all(math.isfinite(entry["mean_CR"]) for entry in diagnostics)
    assert all(math.isfinite(entry["std_CR"]) for entry in diagnostics)
    assert all(math.isfinite(entry["memory_F_mean"]) for entry in diagnostics)
    assert all(math.isfinite(entry["memory_CR_mean"]) for entry in diagnostics)
    assert all(
        next_entry["best_J"] <= entry["best_J"]
        for entry, next_entry in zip(diagnostics, diagnostics[1:])
    )


def test_lshade_stage3_reproducibility_and_archive_determinism():
    bounds = np.array(
        [
            [0.0, 30.0],
            [-5.0, 5.0],
            [-1.0, 1.0],
            [0.0, 2.0],
            [0.0, 0.2],
        ],
        dtype=float,
    )

    def sphere(theta):
        return float(np.sum(np.asarray(theta, dtype=float) ** 2))

    first_optimizer = LShadeOptimizer(population_size=10, mc_seed=123)
    second_optimizer = LShadeOptimizer(population_size=10, mc_seed=123)

    first_result = first_optimizer.optimize(
        objective=sphere,
        bounds=bounds,
        seed=202401,
        max_evals=40,
        run_id=0,
    )
    second_result = second_optimizer.optimize(
        objective=sphere,
        bounds=bounds,
        seed=202401,
        max_evals=40,
        run_id=0,
    )

    assert first_result.final_J == second_result.final_J
    assert first_result.theta_best == second_result.theta_best
    assert first_result.metadata["final_M_F"] == second_result.metadata["final_M_F"]
    assert first_result.metadata["final_M_CR"] == second_result.metadata["final_M_CR"]
    assert first_optimizer.debug_info == second_optimizer.debug_info
    assert len(first_optimizer.archive) == len(second_optimizer.archive)
    for first_vector, second_vector in zip(
        first_optimizer.archive,
        second_optimizer.archive,
    ):
        assert np.array_equal(first_vector, second_vector)
