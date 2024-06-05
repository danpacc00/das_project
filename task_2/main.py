import argparse

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import ros.src.surveillance.surveillance.phi as phi
from ros.src.surveillance.surveillance.aggregative_tracking import AggregativeTracking
from ros.src.surveillance.surveillance.costs_fn import SurveillanceCost
from ros.src.surveillance.surveillance.functions import animation

np.random.seed(0)


def main():
    # Task 2.1
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-n", "--nodes", type=int, default=10)
    argparser.add_argument("-i", "--iters", type=int, default=1000)
    argparser.add_argument("--no-plots", action="store_true", default=False)

    args = argparser.parse_args()

    # targets = np.random.rand(args.nodes, 2) * 10 - 5
    targets = np.array(
        [[0.48813504, 2.15189366], [1.02763376, 0.44883183], [-0.76345201, 1.45894113], [-0.62412789, 3.91773001]]
    )

    # cost = SurveillanceCost(tradeoff=1.0)
    # algo = AggregativeTracking(cost, phi.Identity(), max_iters=args.iters, alpha=1e-2)

    # graph = nx.path_graph(args.nodes)
    # zz, cost, gradient_magnitude, kk = algo.run(graph, targets, d=2)

    # if not args.no_plots:
    #     _, ax = plt.subplots(3, 1, figsize=(10, 10))
    #     ax[0].plot(np.arange(zz.shape[0]), zz[:, :, 0])
    #     ax[0].grid()
    #     ax[0].set_title("Aggregative tracking")

    #     ax[1].plot(np.arange(zz.shape[0] - 1), cost[:-1])
    #     ax[1].grid()
    #     ax[1].set_title("Cost")

    #     ax[2].semilogy(np.arange(zz.shape[0] - 1), gradient_magnitude[1:])
    #     ax[2].grid()
    #     ax[2].set_title("Gradient magnitude")

    #     plt.show()

    #     plt.figure("Animation")
    #     animation(zz, np.linspace(0, kk, kk), nx.adjacency_matrix(graph).toarray(), targets)

    # The corridor is defined by two walls: up_wall and down_wall. They are defined by a set of points between a start and an end
    up_wall_start = np.array([-5, 2])
    up_wall_end = np.array([5, 2])

    down_wall_start = np.array([-5, -2])
    down_wall_end = np.array([5, -2])

    up_wall = np.arange(0, 1.01, 0.01)[:, None] * (up_wall_end - up_wall_start) + up_wall_start
    down_wall = np.arange(0, 1.01, 0.01)[:, None] * (down_wall_end - down_wall_start) + down_wall_start

    # Define the targets at the right of the corridor
    targets = np.random.rand(args.nodes, 2) * 10 + up_wall_end

    # Show the corridor plotting all the points as dots
    plt.plot(up_wall[:, 0], up_wall[:, 1], "k")
    plt.plot(down_wall[:, 0], down_wall[:, 1], "k")
    plt.plot(targets[:, 0], targets[:, 1], "ro")
    plt.show()


if __name__ == "__main__":
    main()
