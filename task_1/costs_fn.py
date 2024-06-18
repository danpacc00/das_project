import numpy as np
from dataset import cost, cost_gradient


class QuadraticCost:
    def __init__(self, nn, d):
        self.Q = []
        for _ in range(nn):
            Q = np.random.uniform(size=(d, d))
            self.Q.append(np.dot(Q.T, Q))

        self.Q = np.array(self.Q)
        self.R = np.random.uniform(size=(nn, d))

    def __call__(self, ii, z):
        return 0.5 * z.T @ self.Q[ii, :, :] @ z + self.R[ii, :] @ z, self.Q[ii, :, :] @ z + self.R[ii, :]

    def optimal(self):
        ZZ_opt = -np.sum(self.R, axis=0) @ np.linalg.inv(np.sum(self.Q, axis=0))
        return 0.5 * ZZ_opt.T @ np.sum(self.Q, axis=0) @ ZZ_opt + np.sum(self.R, axis=0) @ ZZ_opt


class LogisticRegressionCost:
    def __init__(self, dataset):
        self.dataset = dataset

    def __call__(self, ii, theta):
        return cost(theta, self.dataset[ii]), cost_gradient(theta, self.dataset[ii])
