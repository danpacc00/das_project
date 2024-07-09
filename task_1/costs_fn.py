import numpy as np
from dataset import cost, cost_gradient

# Cost functions are defined as callable classes because they typically need 
# to store some data (such as the cost matrices or the dataset)

class QuadraticCost:
    def __init__(self, nn, d):
        self.Q = []

        # Generate random positive definite matrices
        for _ in range(nn):
            Q = np.eye(d)
            self.Q.append(np.dot(Q.T, Q))

        self.Q = np.array(self.Q)
        self.R = np.zeros((nn, d))

    def __call__(self, ii, z):
        return 0.5 * z.T @ self.Q[ii, :, :] @ z + self.R[ii, :] @ z, self.Q[ii, :, :] @ z + self.R[ii, :]

    def optimal(self):
        ZZ_opt = -np.sum(self.R, axis=0) @ np.linalg.inv(np.sum(self.Q, axis=0))
        return 0.5 * ZZ_opt.T @ np.sum(self.Q, axis=0) @ ZZ_opt + np.sum(self.R, axis=0) @ ZZ_opt


class LogisticRegressionCost:
    def __init__(self, dataset):
        self.dataset = dataset

    def __call__(self, ii, theta):
        # The cost and gradient for the logistic regression function are defined in the dataset module
        # so we just call them here to compute the cost and gradient for the i-th dataset (each 
        # node in the graph has a dataset associated with it)
        return cost(theta, self.dataset[ii]), cost_gradient(theta, self.dataset[ii])
