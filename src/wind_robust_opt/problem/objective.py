import numpy as np

from wind_robust_opt.problem.constants import ALPHA, DELTA, GAMMA, P_RATED
from wind_robust_opt.problem.penalties import penalty_bounds
from wind_robust_opt.problem.physics import turbine_power


class RobustWindObjective:
    def __init__(self, wind_samples):
        self.wind_samples = np.asarray(wind_samples, dtype=float)

    def __call__(self, theta) -> float:
        theta = np.asarray(theta, dtype=float)

        p = turbine_power(self.wind_samples, theta) / P_RATED

        objective = (
            -np.mean(p)
            + ALPHA * np.var(p)
            + GAMMA * np.mean(np.maximum(0.0, p - 1.0) ** 2)
            + DELTA * penalty_bounds(theta)
        )

        return float(objective)
