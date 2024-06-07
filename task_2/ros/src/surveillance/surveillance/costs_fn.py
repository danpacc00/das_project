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
    def __init__(self, alpha, beta, gamma, dd, obstacles):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.dd = dd
        self.obstacles = obstacles

    def __call__(self, target, zz, sigma):
        li_target = self.alpha * np.linalg.norm(zz - target) ** 2
        li_sigma = self.gamma * np.linalg.norm(zz - sigma) ** 2

        li_obstacles = 0
        nabla_1 = 0

        for i in range(self.obstacles.shape[0]):
            # li_obstacles += -np.log(np.linalg.norm(zz - self.obstacles[i]) ** 2 - dd)
            # nabla_1 += -2 * (zz - self.obstacles[i]) / (np.linalg.norm(zz - self.obstacles[i]) ** 2 - dd)

            li_obstacles += self.beta / (np.linalg.norm(zz - self.obstacles[i]) ** 2 - self.dd)
            nabla_1 += -2 * self.beta * (zz - self.obstacles[i]) / (np.linalg.norm(zz - self.obstacles[i]) ** 3)

        li = li_target + li_sigma + li_obstacles
        nabla_1 += 2 * self.alpha * (zz - target) + 2 * self.gamma * (zz - sigma)
        nabla_2 = 2 * self.gamma * (zz - sigma)

        return li, nabla_1, nabla_2
