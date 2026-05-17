import json
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from wind_robust_opt.optimizers.base import HistoryEntry, OptimizerResult


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, dict):
        return {
            str(_json_safe(key)): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_safe(getattr(value, field.name))
            for field in fields(value)
        }

    return value


def history_entry_to_dict(entry: HistoryEntry) -> dict:
    return {
        "eval": _json_safe(entry.eval),
        "best_J": _json_safe(entry.best_J),
        "theta_best": _json_safe(entry.theta_best),
        "mean_J": _json_safe(entry.mean_J),
        "std_J": _json_safe(entry.std_J),
        "population_size": _json_safe(entry.population_size),
    }


def optimizer_result_to_dict(result: OptimizerResult) -> dict:
    return {
        "method": _json_safe(result.method),
        "run_id": _json_safe(result.run_id),
        "seed": _json_safe(result.seed),
        "mc_seed": _json_safe(result.mc_seed),
        "dimension": _json_safe(result.dimension),
        "max_evals": _json_safe(result.max_evals),
        "n_evals": _json_safe(result.n_evals),
        "runtime_sec": _json_safe(result.runtime_sec),
        "final_J": _json_safe(result.final_J),
        "theta_best": _json_safe(result.theta_best),
        "history": [
            history_entry_to_dict(entry)
            for entry in result.history
        ],
        "metadata": _json_safe(result.metadata),
    }


def save_result_json(result: OptimizerResult, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            optimizer_result_to_dict(result),
            file,
            indent=2,
            ensure_ascii=False,
        )


def load_result_json(path: str | Path) -> dict:
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)
