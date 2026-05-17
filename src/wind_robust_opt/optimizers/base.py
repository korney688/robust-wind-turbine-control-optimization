from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class HistoryEntry:
    eval: int
    best_J: float
    theta_best: list[float]

    mean_J: float | None = None
    std_J: float | None = None

    population_size: int | None = None


@dataclass
class OptimizerResult:
    method: str

    run_id: int
    seed: int
    mc_seed: int

    dimension: int
    max_evals: int
    n_evals: int

    runtime_sec: float
    final_J: float
    theta_best: list[float]

    history: list[HistoryEntry]

    metadata: dict[str, Any] = field(default_factory=dict)


class BaseOptimizer(ABC):
    @abstractmethod
    def optimize(
        self,
        objective,
        bounds,
        seed: int,
        max_evals: int,
        run_id: int,
    ) -> OptimizerResult:
        ...
