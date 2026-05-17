import numpy as np

from wind_robust_opt.problem.constants import (
    CP_MAX,
    RHO,
    ROTOR_AREA,
    ROTOR_RADIUS,
)
from wind_robust_opt.problem.control_laws import beta_control, omega_control


def compute_cp(lam, beta):
    lam = np.maximum(np.asarray(lam, dtype=float), 1e-6)
    beta = np.asarray(beta, dtype=float)

    inv = (
        1.0 / (lam + 0.08 * beta)
        - 0.035 / (beta**3 + 1.0)
    )

    cp = (
        0.5176
        * (116.0 * inv - 0.4 * beta - 5.0)
        * np.exp(-21.0 * inv)
        + 0.0068 * lam
    )

    return np.clip(cp, 0.0, CP_MAX)


def turbine_power(v, theta):
    v = np.asarray(v, dtype=float)
    beta = beta_control(v, theta)
    omega = omega_control(v, theta)
    lam = omega * ROTOR_RADIUS / np.maximum(v, 1e-6)
    cp = compute_cp(lam, beta)

    return 0.5 * RHO * ROTOR_AREA * cp * v**3
