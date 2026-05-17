import numpy as np

from wind_robust_opt.problem.constants import (
    BETA_MAX,
    BETA_MIN,
    OMEGA_MAX,
    OMEGA_MIN,
    V_RATED,
)


def beta_control(v, theta):
    a0, a1, a2, _, _ = theta
    v = np.asarray(v, dtype=float)

    beta = (
        a0
        + a1 * (v - V_RATED)
        + a2 * (v - V_RATED) ** 2
    )

    return np.clip(beta, BETA_MIN, BETA_MAX)


def omega_control(v, theta):
    _, _, _, b0, b1 = theta
    v = np.asarray(v, dtype=float)

    omega = b0 + b1 * v

    return np.clip(omega, OMEGA_MIN, OMEGA_MAX)
