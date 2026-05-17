import time

import numpy as np

from wind_robust_opt.optimizers.base import (
    BaseOptimizer,
    HistoryEntry,
    OptimizerResult,
)


class RandomSearchOptimizer(BaseOptimizer):
    def __init__(self, mc_seed: int = 0):
        self.mc_seed = int(mc_seed)

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
        if max_evals <= 0:
            raise ValueError("max_evals must be positive")

        lower = bounds[:, 0]
        upper = bounds[:, 1]
        dimension = int(bounds.shape[0])

        rng = np.random.default_rng(seed)
        history: list[HistoryEntry] = []

        best_J = float("inf")
        theta_best: list[float] | None = None

        start = time.perf_counter()

        for eval_id in range(1, max_evals + 1):
            theta = rng.uniform(lower, upper)
            current_J = float(objective(theta))

            if theta_best is None or current_J < best_J:
                best_J = current_J
                theta_best = theta.astype(float).tolist()

            history.append(
                HistoryEntry(
                    eval=eval_id,
                    best_J=float(best_J),
                    theta_best=list(theta_best),
                )
            )

        runtime_sec = time.perf_counter() - start

        return OptimizerResult(
            method="RandomSearch",
            run_id=int(run_id),
            seed=int(seed),
            mc_seed=self.mc_seed,
            dimension=dimension,
            max_evals=int(max_evals),
            n_evals=int(max_evals),
            runtime_sec=float(runtime_sec),
            final_J=float(best_J),
            theta_best=list(theta_best),
            history=history,
            metadata={
                "algorithm": "RandomSearch",
            },
        )
