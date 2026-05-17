from wind_robust_opt.experiments.run_benchmark import run_benchmark
from wind_robust_opt.io.result_schema import load_result_json


def test_random_search_benchmark_pipeline(tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    first_results = run_benchmark(
        output_dir=first_dir,
        run_seeds=[202401],
        max_evals=10,
    )
    second_results = run_benchmark(
        output_dir=second_dir,
        run_seeds=[202401],
        max_evals=10,
    )

    assert first_results[0].final_J == second_results[0].final_J
    assert first_results[0].theta_best == second_results[0].theta_best

    loaded = load_result_json(first_dir / "run_000.json")

    assert loaded["method"] == "RandomSearch"
    assert loaded["n_evals"] == 10
    assert loaded["theta_best"] == first_results[0].theta_best
    assert len(loaded["history"]) == 10
