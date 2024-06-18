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
    def __init__(self, alpha, beta, gamma, dd, obstacles):
        self.alpha = alpha  # 1
        self.beta = beta  # 10
        self.gamma = gamma
        self.dd = dd
        self.obstacles = obstacles

        self.corridor = np.array([-15, 0])

    def __call__(self, target, zz, sigma):
        li_target = self.alpha * np.linalg.norm(zz - target) ** 2  # + 10 * np.linalg.norm(zz - self.corridor) ** 2
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
        nabla_1 += 2 * self.alpha * (zz - target) + 2 * self.gamma * (zz - sigma)  # + 10 * (zz - self.corridor)
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

        obstacle_seen = list(
            map(lambda x: x[1], sorted([(np.linalg.norm(zz - o) ** 2, o) for o in self.obstacles], key=lambda x: x[0]))
        )

        for obstacle in obstacle_seen:
            li_obstacles_tmp = -0.5 * np.log(np.linalg.norm(zz - obstacle) ** 2 - self.dd)

            li_obstacles += li_obstacles_tmp

            nabla_1 += -2 * 0.5 * (zz - obstacle) / (np.linalg.norm(zz - obstacle) ** 2 - self.dd)

        li = li_target + li_sigma + self.beta * li_obstacles + li_corridor
        nabla_1 += 2 * self.alpha * (zz - target) + 2 * self.gamma * (zz - sigma)
        nabla_2 = 2 * self.gamma * (zz - sigma) + 2 * np.linalg.norm(sigma[1] - self.mean_y)

        return li, nabla_1, nabla_2


class CorridorCostV3:
    def __init__(self, alpha, beta, gamma, dd, obstacles):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.dd = dd
        self.obstacles = obstacles

    def __call__(self, target, zz, sigma):
        li_target = self.alpha * np.linalg.norm(zz - target) ** 2
        li_sigma = self.gamma * np.linalg.norm(zz - sigma) ** 2

        nabla_1 = 0

        c = 1

        li_feasible = c * zz[1] ** 2

        li = li_target + li_sigma + li_feasible

        nabla_1 += 2 * self.alpha * (zz - target) + 2 * self.gamma * (zz - sigma) + 2 * c * zz[1]
        nabla_2 = 2 * self.gamma * (zz - sigma)

        return li, nabla_1, nabla_2


class CorridorCostV4:
    def __init__(self, alpha, beta, gamma, dd, obstacles):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.dd = dd
        self.obstacles = obstacles
        self.kk = 0

    def __call__(self, target, zz, sigma):
        li_target = self.alpha * np.linalg.norm(zz - target) ** 2
        li_sigma = self.gamma * np.linalg.norm(zz - sigma) ** 2

        height = 10
        width = 1e-6
        exp = 8

        # dd = 0

        # self.kk += 1

        # make a slowly varying alpha
        # self.alpha += 3e-6
        # print("self.alpha: ", self.alpha)

        # self.alpha = 0.01 + 1 / (1 + np.exp(1000 - self.kk))

        top_barrier = -(width * zz[0] ** exp) - height + zz[1]
        bottom_barrier = -(width * zz[0] ** exp) - height - zz[1]

        li_barrier = -np.log(-top_barrier) + -np.log(-bottom_barrier)

        # li_barrier = 1 / (-top_barrier - dd) ** 2

        li = li_target + li_sigma + li_barrier

        nabla_1 = 2 * self.alpha * (zz - target) + 2 * self.gamma * (zz - sigma)
        nabla_2 = 2 * self.gamma * (zz - sigma)

        nabla_1_1_top = -(-exp * width * zz[0] ** (exp - 1)) / -top_barrier
        nabla_1_2_top = -(1) / -top_barrier

        nabla_1_1_bottom = -(-exp * width * zz[0] ** (exp - 1)) / -bottom_barrier
        nabla_1_2_bottom = -(-1) / -bottom_barrier

        # nabla_1_1_top = -2 * width * exp * zz[0] ** (exp - 1) / (-top_barrier - dd) ** 3
        # nabla_1_2_top = 2 / (-top_barrier - dd) ** 3

        nabla_1 += np.array([nabla_1_1_top, nabla_1_2_top]) + np.array([nabla_1_1_bottom, nabla_1_2_bottom])

        return li, nabla_1, nabla_2


class CorridorCostV5:
    def __init__(self, alpha, beta, gamma, dd, obstacles):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.dd = dd
        self.obstacles = obstacles
        self.kk = 0

    def __call__(self, target, zz, sigma):
        li_target = self.alpha * np.linalg.norm(zz - target) ** 2
        li_sigma = self.gamma * np.linalg.norm(zz - sigma) ** 2

        height = 10
        width = 1e-6
        exp = 8

        dd = 0

        self.kk += 1

        # make a slowly varying alpha
        self.alpha += 3e-6
        # print("self.alpha: ", self.alpha)

        top_barrier = -(width * zz[0] ** exp) - height + zz[1]
        bottom_barrier = -(width * zz[0] ** exp) - height - zz[1]

        li_barrier = 1 / (-top_barrier - dd) ** 2

        nabla_1 = 2 * self.alpha * (zz - target) + 2 * self.gamma * (zz - sigma)
        nabla_2 = 2 * self.gamma * (zz - sigma)

        nabla_1_1_top = -2 * width * exp * zz[0] ** (exp - 1) / (-top_barrier - dd) ** 3
        nabla_1_2_top = 2 / (-top_barrier - dd) ** 3

        nabla_1 += np.array([nabla_1_1_top, nabla_1_2_top])  # + np.array([nabla_1_1_bottom, nabla_1_2_bottom])

        li_obstacles = 0

        obstacle_seen = list(
            map(lambda x: x[1], sorted([(np.linalg.norm(zz - o) ** 2, o) for o in self.obstacles], key=lambda x: x[0]))
        )

        q = 4

        for obstacle in obstacle_seen:
            li_obstacles += self.beta / (np.linalg.norm(zz - obstacle) ** 2) ** q
            nabla_1 += -2 * q * self.beta * (zz - obstacle) / (np.linalg.norm(zz - obstacle) ** 2) ** (q + 1)

        li = li_target + li_sigma + li_barrier + li_obstacles

        return li, nabla_1, nabla_2


# Works well with more than 10000 iterations
class CorridorCostV6:
    def __init__(self, alpha, obstacles):
        self.alpha = alpha
        self.kk = 0
        self.epsilon = 0.1
        self.obstacles = obstacles

    def __call__(self, target, zz, sigma):
        li_target = self.alpha * np.linalg.norm(zz - target) ** 2
        li_sigma = np.linalg.norm(zz - sigma) ** 2

        li_barrier = 0
        for obstacle in self.obstacles:
            li_barrier += -np.log(np.linalg.norm(zz - obstacle) ** 2 - 1)

        li = li_target + li_sigma + li_barrier
        if self.kk % 10 == 0:
            self.epsilon = self.epsilon / 10

        nabla_1 = 2 * self.alpha * (zz - target) + 2 * (zz - sigma)
        for obstacle in self.obstacles:
            nabla_1 += -2 * self.epsilon * (zz - obstacle) / (np.linalg.norm(zz - obstacle) ** 2 - 1)
        nabla_2 = 2 * (zz - sigma)

        self.kk += 1
        return li, nabla_1, nabla_2


class CorridorCostV6:
    def __init__(self, alpha, obstacles):
        self.alpha = alpha
        self.kk = 0
        self.epsilon = 0.1
        self.obstacles = obstacles

    def __call__(self, target, zz, sigma):
        li_target = self.alpha * np.linalg.norm(zz - target) ** 2
        li_sigma = np.linalg.norm(zz - sigma) ** 2

        li_barrier = 0
        for obstacle in self.obstacles:
            li_barrier += -np.log(np.linalg.norm(zz - obstacle) ** 2 - 1)

        li = li_target + li_sigma + li_barrier
        self.epsilon = self.epsilon / 10

        nabla_1 = 2 * self.alpha * (zz - target) + 2 * (zz - sigma)
        for obstacle in self.obstacles:
            nabla_1 += -2 * (zz - obstacle) / (np.linalg.norm(zz - obstacle) ** 2 - 1)
        nabla_2 = 2 * (zz - sigma)

        self.kk += 1
        return li, nabla_1, nabla_2


class CorridorCostV7:
    def __init__(self, alpha):
        self.alpha = alpha
        self.kk = 0
        self.epsilon = 0.1

    def __call__(self, target, zz, sigma):
        li_target = self.alpha * np.linalg.norm(zz - target) ** 2
        li_sigma = np.linalg.norm(zz - sigma) ** 2

        g = 0.01 * zz[0] ** 2 + 10 - zz[1]
        li_barrier = -np.log(g)

        li = li_target + li_sigma + li_barrier
        self.epsilon = self.epsilon / 10

        nabla_1 = 2 * self.alpha * (zz - target) + 2 * (zz - sigma)
        nabla_1 += np.array([-0.02 * zz[0], 1]) / g
        nabla_2 = 2 * (zz - sigma)

        self.kk += 1
        return li, nabla_1, nabla_2


class CorridorCostV8:
    def __init__(self, alpha):
        self.alpha = alpha
        self.epsilon = 0.1

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
