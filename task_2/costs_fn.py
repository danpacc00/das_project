import numpy as np


class SurveillanceCost:
    def __init__(self, targets, tradeoff):
        self.targets = targets
        self.tradeoff = tradeoff

    def __call__(self, ii, zz, sigma):
        target = self.targets[ii]

        li = self.tradeoff * np.linalg.norm(zz - target) ** 2 + np.linalg.norm(zz - sigma) ** 2
        nabla_1 = 2 * self.tradeoff * (zz - target) + 2 * (zz - sigma)
        nabla_2 = 2 * (zz - sigma)

        return li, nabla_1, nabla_2
