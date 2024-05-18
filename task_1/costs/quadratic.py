import numpy as np


class QuadraticCost:
    def __init__(self, nn, d):
        self.Q = np.random.uniform(size=(nn, d, d))
        self.R = np.random.uniform(size=(nn, d))

    def __call__(self, ii, z):
        return 0.5 * z.T @ self.Q[ii, :, :] @ z + self.R[ii, :] @ z, self.Q[ii, :, :] @ z + self.R[ii, :]
