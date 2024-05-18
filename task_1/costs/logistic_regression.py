from ..dataset import cost, cost_gradient


class LogisticRegressionCost:
    def __init__(self, dataset):
        self.dataset = dataset

    def __call__(self, ii, theta):
        return cost(theta, self.dataset[ii]), cost_gradient(theta, self.dataset[ii])
