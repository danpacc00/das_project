import numpy as np


class SurveillanceCost:
    def __init__(self, tradeoff):
        self.tradeoff = tradeoff

    def __call__(self, target, zz, sigma):
        li = self.tradeoff * np.linalg.norm(zz - target) ** 2 + np.linalg.norm(zz - sigma) ** 2
        nabla_1 = 2 * self.tradeoff * (zz - target) + 2 * (zz - sigma)
        nabla_2 = 2 * (zz - sigma)

        return li, nabla_1, nabla_2


class CorridorCost:
    def __init__(self, tradeoff, corridor):
        self.tradeoff = tradeoff
        self.corridor = corridor

    def __call__(self, target, zz, sigma):
        li = (
            self.tradeoff * np.linalg.norm(zz - target) ** 2
            + np.linalg.norm(zz - sigma) ** 2
            + (self.corridor[1] - zz[1]) ** 2
        )

        nabla_1 = 2 * self.tradeoff * (zz - target) + 2 * (zz - sigma) + 2 * (self.corridor[1] - zz[1])
        nabla_2 = 2 * (zz - sigma)

        return li, nabla_1, nabla_2
