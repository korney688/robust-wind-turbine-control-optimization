import json

import numpy as np

from wind_robust_opt.io.result_schema import (
    load_result_json,
    optimizer_result_to_dict,
    save_result_json,
)
from wind_robust_opt.optimizers.base import HistoryEntry, OptimizerResult


def test_optimizer_result_json_roundtrip(tmp_path):
    result = OptimizerResult(
        method="TEST",
        run_id=np.int32(0),
        seed=np.int64(123),
        mc_seed=999,
        dimension=5,
        max_evals=100,
        n_evals=100,
        runtime_sec=np.float32(1.23),
        final_J=np.float64(-0.5),
        theta_best=np.array([1.0, 2.0, 3.0, 0.8, 0.05]),
        history=[
            HistoryEntry(
                eval=np.int32(50),
                best_J=np.float64(-0.25),
                theta_best=np.array([1.0, 1.5, 2.0, 0.7, 0.04]),
                mean_J=np.float32(0.1),
                std_J=np.float64(0.2),
                population_size=np.int64(10),
            ),
            HistoryEntry(
                eval=100,
                best_J=-0.5,
                theta_best=[1.0, 2.0, 3.0, 0.8, 0.05],
            ),
        ],
        metadata={
            "array": np.array([1, 2, 3], dtype=np.int64),
            "scalar": np.float32(2.5),
        },
    )

    payload = optimizer_result_to_dict(result)
    json.dumps(payload)

    path = tmp_path / "nested" / "result.json"
    save_result_json(result, path)
    loaded = load_result_json(path)

    assert loaded["method"] == "TEST"
    assert loaded["theta_best"] == [1.0, 2.0, 3.0, 0.8, 0.05]
    assert isinstance(loaded["theta_best"], list)
    assert len(loaded["history"]) == 2
    assert loaded["history"][0]["eval"] == 50
    assert loaded["history"][0]["theta_best"] == [1.0, 1.5, 2.0, 0.7, 0.04]
    assert loaded["metadata"]["array"] == [1, 2, 3]
    assert loaded["metadata"]["scalar"] == float(np.float32(2.5))
