import numpy as np

from wind_robust_opt.problem.constants import (
    WEIBULL_K,
    WEIBULL_C,
    V_CUT_IN,
    V_CUT_OUT,
    V_RATED,
    N_MC_SAMPLES,
    MC_SEED,
)


def generate_wind_samples(
    n_samples: int = N_MC_SAMPLES,
    seed: int = MC_SEED,
) -> np.ndarray:
    """
    Deterministic Monte Carlo wind sampling.

    Wind speed is sampled from a Weibull distribution
    and filtered to the operational wind range.
    """

    rng = np.random.default_rng(seed)

    v_all = WEIBULL_C * rng.weibull(
        WEIBULL_K,
        size=n_samples,
    )

    mask = (
        (v_all >= V_CUT_IN)
        & (v_all <= V_CUT_OUT)
    )

    v_valid = v_all[mask]

    if len(v_valid) == 0:
        return np.array([V_RATED], dtype=float)

    return v_valid.astype(float)