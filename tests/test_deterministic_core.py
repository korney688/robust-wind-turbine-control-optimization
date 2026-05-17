import numpy as np

from wind_robust_opt.problem.objective import RobustWindObjective
from wind_robust_opt.problem.wind_distribution import generate_wind_samples


def test_deterministic_wind_sampling_and_objective():
    samples_1 = generate_wind_samples(seed=999)
    samples_2 = generate_wind_samples(seed=999)

    assert np.array_equal(samples_1, samples_2)

    theta = np.array([10.0, 0.1, 0.01, 0.8, 0.05], dtype=float)
    objective = RobustWindObjective(samples_1)

    assert objective(theta) == objective(theta)
