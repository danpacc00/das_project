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

        self.corridor = np.array([-15, 0])

    def __call__(self, target, zz, sigma):
        li_target = self.alpha * np.linalg.norm(zz - target) ** 2 + 10 * np.linalg.norm(zz - self.corridor) ** 2
        li_sigma = self.gamma * np.linalg.norm(zz - sigma) ** 2

        li_obstacles = 0
        nabla_1 = 0

        q = 4

        obstacle_seen = list(
            map(lambda x: x[1], sorted([(np.linalg.norm(zz - o) ** 2, o) for o in self.obstacles], key=lambda x: x[0]))
        )

        for obstacle in obstacle_seen:
            li_obstacles += self.beta / (np.linalg.norm(zz - obstacle) ** 2) ** q
            nabla_1 += -2 * q * self.beta * (zz - obstacle) / (np.linalg.norm(zz - obstacle) ** 2) ** (q + 1)

        li = li_target + li_sigma + li_obstacles
        nabla_1 += 2 * self.alpha * (zz - target) + 10 * (zz - self.corridor) + 2 * self.gamma * (zz - sigma)
        nabla_2 = 2 * self.gamma * (zz - sigma)

        return li, nabla_1, nabla_2


class CorridorCostV2:
    def __init__(self, alpha, beta, gamma, dd, obstacles):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.dd = dd
        self.obstacles = obstacles

        # find maximum value of obstacles along y
        max_y = np.max(self.obstacles[:, 1])
        min_y = np.min(self.obstacles[:, 1])

        # self.mean_y = abs((max_y + min_y)) / 2
        self.mean_y = 0

    def __call__(self, target, zz, sigma):
        li_target = self.alpha * np.linalg.norm(zz - target) ** 2
        li_sigma = self.gamma * np.linalg.norm(zz - sigma) ** 2

        li_corridor = np.linalg.norm(sigma[1] - self.mean_y) ** 2

        li_obstacles = 0
        nabla_1 = 0

        # find the obstacles which have the minimum distance to the agent

        for i in range(self.obstacles.shape[0]):
            li_obstacles_tmp = -np.log(np.linalg.norm(zz - self.obstacles[i]) ** 2 - self.dd)

            li_obstacles += li_obstacles_tmp

            nabla_1 += -2 * (zz - self.obstacles[i]) / (np.linalg.norm(zz - self.obstacles[i]) ** 2 - self.dd)

        li = li_target + li_sigma + self.beta * li_obstacles + li_corridor
        nabla_1 += 2 * self.alpha * (zz - target) + 2 * self.gamma * (zz - sigma)
        nabla_2 = 2 * self.gamma * (zz - sigma) + 2 * np.linalg.norm(sigma[1] - self.mean_y)

        return li, nabla_1, nabla_2
