import numpy as np

BOUNDS = np.array([
    [0.0, 30.0],
    [-5.0, 5.0],
    [-1.0, 1.0],
    [0.0, 2.0],
    [0.0, 0.2],
], dtype=float)

LOWER_BOUNDS = BOUNDS[:, 0]
UPPER_BOUNDS = BOUNDS[:, 1]

N_DIMS = BOUNDS.shape[0]

