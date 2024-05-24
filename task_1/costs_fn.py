import numpy as np
from dataset_line import cost, cost_gradient


class QuadraticCost:
    def __init__(self, nn, d):
        self.Q = np.random.uniform(size=(nn, d, d))
        self.R = np.random.uniform(size=(nn, d))

    def __call__(self, ii, z):
        return 0.5 * z.T @ self.Q[ii, :, :] @ z + self.R[ii, :] @ z, self.Q[ii, :, :] @ z + self.R[ii, :]


class LogisticRegressionCost:
    def __init__(self, dataset):
        self.dataset = dataset

    def __call__(self, ii, theta):
        return cost(theta, self.dataset[ii]), cost_gradient(theta, self.dataset[ii])
