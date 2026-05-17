import numpy as np

from wind_robust_opt.problem.bounds import LOWER_BOUNDS, UPPER_BOUNDS


def penalty_bounds(theta) -> float:
    theta = np.asarray(theta, dtype=float)

    lower_violation = np.maximum(0.0, LOWER_BOUNDS - theta)
    upper_violation = np.maximum(0.0, theta - UPPER_BOUNDS)

    return float(np.sum(lower_violation**2 + upper_violation**2))
