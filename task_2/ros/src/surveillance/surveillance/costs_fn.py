import numpy as np


class SurveillanceCost:
    def __init__(self, tradeoff):
        self.tradeoff = tradeoff

    def __call__(self, target, zz, sigma, _):
        li = self.tradeoff * np.linalg.norm(zz - target) ** 2 + np.linalg.norm(zz - sigma) ** 2
        nabla_1 = 2 * self.tradeoff * (zz - target) + 2 * (zz - sigma)
        nabla_2 = 2 * (zz - sigma)

        return li, nabla_1, nabla_2


class CorridorCost:
    def __init__(self, alpha):
        self.alpha = alpha
        self.epsilon = 1.0

    def __call__(self, target, zz, sigma, kk):
        li_target = self.alpha * np.linalg.norm(zz - target) ** 2
        li_sigma = np.linalg.norm(zz - sigma) ** 2

        g_1 = 1e-5 * zz[0] ** 4 + 2 - zz[1]
        g_2 = 1e-5 * zz[0] ** 4 + 2 + zz[1]
        li_barrier = -np.log(g_1) + -np.log(g_2)

        li = li_target + li_sigma + li_barrier

        nabla_1 = 2 * self.alpha * (zz - target) + 2 * (zz - sigma)
        nabla_1 += np.array([-1e-5 * 4 * zz[0] ** 3, 1]) / g_1
        nabla_1 += np.array([-1e-5 * 4 * zz[0] ** 3, -1]) / g_2
        nabla_2 = 2 * (zz - sigma)

        return li, nabla_1, nabla_2

    def constraints(self, zz):
        g_1 = 1e-5 * zz[0] ** 4 + 2 - zz[1]
        g_2 = 1e-5 * zz[0] ** 4 + 2 + zz[1]

        return np.array([-g_1, -g_2])
