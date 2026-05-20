import time

import numpy as np

from wind_robust_opt.optimizers.base import (
    BaseOptimizer,
    HistoryEntry,
    OptimizerResult,
)


def _build_generation_diagnostic(
    generation: int,
    n_evals: int,
    best_J: float,
    population_array: np.ndarray,
    fitness_array: np.ndarray,
    archive_size: int,
    sampled_F: list[float],
    sampled_CR: list[float],
    memory_F: np.ndarray,
    memory_CR: np.ndarray,
    memory_index: int,
    successful_updates: int,
    target_population_size: int,
    population_reductions: int,
) -> dict:
    if sampled_F:
        mean_F = float(np.mean(sampled_F))
        std_F = float(np.std(sampled_F))
    else:
        mean_F = float(np.mean(memory_F))
        std_F = 0.0

    if sampled_CR:
        mean_CR = float(np.mean(sampled_CR))
        std_CR = float(np.std(sampled_CR))
    else:
        mean_CR = float(np.mean(memory_CR))
        std_CR = 0.0

    return {
        "generation": int(generation),
        "eval": int(n_evals),
        "best_J": float(best_J),
        "mean_J": float(np.mean(fitness_array)),
        "std_J": float(np.std(fitness_array)),
        "archive_size": int(archive_size),
        "population_size": int(population_array.shape[0]),
        "target_population_size": int(target_population_size),
        "population_reductions": int(population_reductions),
        "population_diversity": float(np.std(population_array, axis=0).mean()),
        "mean_F": mean_F,
        "std_F": std_F,
        "mean_CR": mean_CR,
        "std_CR": std_CR,
        "memory_F_mean": float(np.mean(memory_F)),
        "memory_CR_mean": float(np.mean(memory_CR)),
        "memory_index": int(memory_index),
        "successful_updates": int(successful_updates),
    }


def _sample_F(rng: np.random.Generator, memory_value: float) -> float:
    for _ in range(100):
        sampled = float(memory_value + 0.1 * rng.standard_cauchy())
        if sampled > 0.0:
            return min(sampled, 1.0)

    return 0.5


def _target_population_size(
    initial_population_size: int,
    min_population_size: int,
    n_evals: int,
    max_evals: int,
) -> int:
    target = round(
        initial_population_size
        - (initial_population_size - min_population_size)
        * (n_evals / max_evals)
    )
    return max(min_population_size, int(target))


class LShadeOptimizer(BaseOptimizer):
    def __init__(
        self,
        population_size: int = 50,
        initial_population_size: int | None = None,
        min_population_size: int = 4,
        F: float = 0.5,
        CR: float = 0.9,
        p_best_rate: float = 0.2,
        memory_size: int = 6,
        mc_seed: int = 0,
    ):
        if initial_population_size is None:
            initial_population_size = population_size
        if initial_population_size < 4:
            raise ValueError("initial_population_size must be at least 4")
        if min_population_size < 4:
            raise ValueError("min_population_size must be at least 4")
        if min_population_size > initial_population_size:
            raise ValueError(
                "min_population_size must be <= initial_population_size"
            )
        if F < 0.0:
            raise ValueError("F must be non-negative")
        if not 0.0 <= CR <= 1.0:
            raise ValueError("CR must be in [0, 1]")
        if not 0.0 < p_best_rate <= 1.0:
            raise ValueError("p_best_rate must be in (0, 1]")
        if memory_size <= 0:
            raise ValueError("memory_size must be positive")

        self.population_size = int(initial_population_size)
        self.initial_population_size = int(initial_population_size)
        self.min_population_size = int(min_population_size)
        self.F = float(F)
        self.CR = float(CR)
        self.p_best_rate = float(p_best_rate)
        self.memory_size = int(memory_size)
        self.M_F = np.full(self.memory_size, 0.5, dtype=float)
        self.M_CR = np.full(self.memory_size, 0.5, dtype=float)
        self.memory_index = 0
        self.mc_seed = int(mc_seed)
        self.archive: list[np.ndarray] = []
        self.debug_info: dict[str, int] = {}

    def optimize(
        self,
        objective,
        bounds,
        seed: int,
        max_evals: int,
        run_id: int,
    ) -> OptimizerResult:
        bounds = np.asarray(bounds, dtype=float)
        if bounds.ndim != 2 or bounds.shape[1] != 2:
            raise ValueError("bounds must have shape (dimension, 2)")
        if np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError("each lower bound must be smaller than upper bound")
        if max_evals <= 0:
            raise ValueError("max_evals must be positive")

        lower = bounds[:, 0]
        upper = bounds[:, 1]
        dimension = int(bounds.shape[0])

        rng = np.random.default_rng(seed)
        history: list[HistoryEntry] = []

        population: list[np.ndarray] = []
        fitness: list[float] = []
        archive: list[np.ndarray] = []
        generation_diagnostics: list[dict] = []
        pbest_selection_count = 0
        best_J = float("inf")
        theta_best: np.ndarray | None = None
        n_evals = 0
        memory_F = np.full(self.memory_size, 0.5, dtype=float)
        memory_CR = np.full(self.memory_size, 0.5, dtype=float)
        memory_index = 0

        start = time.perf_counter()

        initial_population_size = self.initial_population_size
        min_population_size = self.min_population_size

        while n_evals < max_evals and len(population) < initial_population_size:
            theta = rng.uniform(lower, upper)
            current_J = float(objective(theta))
            n_evals += 1

            population.append(theta)
            fitness.append(current_J)

            if theta_best is None or current_J < best_J:
                best_J = current_J
                theta_best = theta.copy()

            history.append(
                HistoryEntry(
                    eval=n_evals,
                    best_J=float(best_J),
                    theta_best=theta_best.astype(float).tolist(),
                    population_size=len(population),
                )
            )

        population_array = np.asarray(population, dtype=float)
        fitness_array = np.asarray(fitness, dtype=float)

        generation = 0
        generation_diagnostics.append(
            _build_generation_diagnostic(
                generation=generation,
                n_evals=n_evals,
                best_J=best_J,
                population_array=population_array,
                fitness_array=fitness_array,
                archive_size=len(archive),
                sampled_F=[],
                sampled_CR=[],
                memory_F=memory_F,
                memory_CR=memory_CR,
                memory_index=memory_index,
                successful_updates=0,
                target_population_size=int(population_array.shape[0]),
                population_reductions=0,
            )
        )

        while n_evals < max_evals and population_array.shape[0] >= 4:
            current_population_size = int(population_array.shape[0])
            generation += 1
            generation_start_evals = n_evals
            sampled_F: list[float] = []
            sampled_CR: list[float] = []
            successful_F: list[float] = []
            successful_CR: list[float] = []
            fitness_improvements: list[float] = []

            for target_idx in range(current_population_size):
                if n_evals >= max_evals:
                    break

                memory_slot = int(rng.integers(self.memory_size))
                F_i = _sample_F(rng, memory_F[memory_slot])
                CR_i = float(rng.normal(memory_CR[memory_slot], 0.1))
                CR_i = float(np.clip(CR_i, 0.0, 1.0))
                sampled_F.append(F_i)
                sampled_CR.append(CR_i)

                pbest_pool_size = max(
                    2,
                    int(np.ceil(self.p_best_rate * current_population_size)),
                )
                pbest_pool_size = min(pbest_pool_size, current_population_size)
                sorted_indices = np.argsort(fitness_array, kind="stable")
                pbest_idx = int(rng.choice(sorted_indices[:pbest_pool_size]))
                pbest_selection_count += 1

                population_candidates = np.delete(
                    np.arange(current_population_size),
                    target_idx,
                )
                r1_idx = int(rng.choice(population_candidates))

                r2_population_indices = [
                    int(idx)
                    for idx in range(current_population_size)
                    if idx not in (target_idx, r1_idx)
                ]
                r2_pool = [
                    population_array[idx]
                    for idx in r2_population_indices
                ] + archive
                r2_idx = int(rng.integers(len(r2_pool)))
                x_r2 = r2_pool[r2_idx]

                mutant = (
                    population_array[target_idx]
                    + F_i
                    * (population_array[pbest_idx] - population_array[target_idx])
                    + F_i * (population_array[r1_idx] - x_r2)
                )

                crossover_mask = rng.random(dimension) < CR_i
                crossover_mask[rng.integers(dimension)] = True

                trial = np.where(
                    crossover_mask,
                    mutant,
                    population_array[target_idx],
                )
                trial = np.clip(trial, lower, upper)

                trial_J = float(objective(trial))
                n_evals += 1

                if trial_J <= fitness_array[target_idx]:
                    target_J = float(fitness_array[target_idx])
                    archive.append(population_array[target_idx].copy())
                    population_array[target_idx] = trial
                    fitness_array[target_idx] = trial_J
                    successful_F.append(F_i)
                    successful_CR.append(CR_i)
                    fitness_improvements.append(abs(target_J - trial_J))

                    while len(archive) > current_population_size:
                        remove_idx = int(rng.integers(len(archive)))
                        del archive[remove_idx]

                    if trial_J < best_J:
                        best_J = trial_J
                        theta_best = trial.copy()

                history.append(
                    HistoryEntry(
                        eval=n_evals,
                        best_J=float(best_J),
                        theta_best=theta_best.astype(float).tolist(),
                        population_size=current_population_size,
                    )
                )

            if n_evals > generation_start_evals:
                population_reductions = 0

                if successful_F:
                    improvements = np.asarray(fitness_improvements, dtype=float)
                    if float(np.sum(improvements)) > 0.0:
                        weights = improvements / np.sum(improvements)
                    else:
                        weights = np.full(
                            len(successful_F),
                            1.0 / len(successful_F),
                            dtype=float,
                        )

                    successful_F_array = np.asarray(successful_F, dtype=float)
                    successful_CR_array = np.asarray(successful_CR, dtype=float)

                    memory_CR[memory_index] = float(
                        np.sum(weights * successful_CR_array)
                    )

                    denominator = float(np.sum(weights * successful_F_array))
                    if denominator > 0.0:
                        memory_F[memory_index] = float(
                            np.sum(weights * successful_F_array**2) / denominator
                        )

                    memory_index = (memory_index + 1) % self.memory_size

                target_population_size = _target_population_size(
                    initial_population_size=initial_population_size,
                    min_population_size=min_population_size,
                    n_evals=n_evals,
                    max_evals=max_evals,
                )
                current_population_size = int(population_array.shape[0])

                if current_population_size > target_population_size:
                    population_reductions = (
                        current_population_size - target_population_size
                    )
                    keep_indices = np.argsort(fitness_array, kind="stable")[
                        :target_population_size
                    ]
                    population_array = population_array[keep_indices].copy()
                    fitness_array = fitness_array[keep_indices].copy()

                while len(archive) > population_array.shape[0]:
                    remove_idx = int(rng.integers(len(archive)))
                    del archive[remove_idx]

                generation_diagnostics.append(
                    _build_generation_diagnostic(
                        generation=generation,
                        n_evals=n_evals,
                        best_J=best_J,
                        population_array=population_array,
                        fitness_array=fitness_array,
                        archive_size=len(archive),
                        sampled_F=sampled_F,
                        sampled_CR=sampled_CR,
                        memory_F=memory_F,
                        memory_CR=memory_CR,
                        memory_index=memory_index,
                        successful_updates=len(successful_F),
                        target_population_size=target_population_size,
                        population_reductions=population_reductions,
                    )
                )

        runtime_sec = time.perf_counter() - start
        print(f"Total objective evaluations: {n_evals}")
        print(f"Configured max_evals: {max_evals}")

        self.archive = [entry.copy() for entry in archive]
        self.M_F = memory_F.copy()
        self.M_CR = memory_CR.copy()
        self.memory_index = int(memory_index)
        self.debug_info = {
            "archive_size": len(self.archive),
            "pbest_selection_count": pbest_selection_count,
            "memory_index": self.memory_index,
            "final_population_size": int(population_array.shape[0]),
        }

        return OptimizerResult(
            method="LShade",
            run_id=int(run_id),
            seed=int(seed),
            mc_seed=self.mc_seed,
            dimension=dimension,
            max_evals=int(max_evals),
            n_evals=int(n_evals),
            runtime_sec=float(runtime_sec),
            final_J=float(best_J),
            theta_best=theta_best.astype(float).tolist(),
            history=history,
            metadata={
                "algorithm": "L-SHADE",
                "mutation": "current-to-pbest/1",
                "population_size": self.population_size,
                "linear_population_reduction": True,
                "initial_population_size": initial_population_size,
                "min_population_size": min_population_size,
                "final_population_size": int(population_array.shape[0]),
                "F": self.F,
                "CR": self.CR,
                "p_best_rate": self.p_best_rate,
                "archive_enabled": True,
                "adaptive_F": True,
                "adaptive_CR": True,
                "memory_size": self.memory_size,
                "initial_M_F": 0.5,
                "initial_M_CR": 0.5,
                "final_M_F": self.M_F.astype(float).tolist(),
                "final_M_CR": self.M_CR.astype(float).tolist(),
                "generation_diagnostics": generation_diagnostics,
            },
        )
