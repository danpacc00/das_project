import argparse
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from gradient_tracking import GradientTracking

np.random.seed(0)
NN = 10

I_NN = np.eye(NN)


class QuadraticCost:
    def __init__(self, nn):
        self.Q = np.random.uniform(size=(nn))
        self.R = np.random.uniform(size=(nn))

    def __call__(self, ii, z):
        return 0.5 * self.Q[ii] * z * z + self.R[ii] * z, self.Q[ii] * z + self.R[ii]


# Task 1.1
def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-n", "--nodes", type=int, default=10)
    argparser.add_argument("-i", "--iters", type=int, default=1000)
    argparser.add_argument("--no-plots", action="store_true", default=False)

    args = argparser.parse_args()

    graphs = [
        {"name": "cycle", "fn": nx.cycle_graph},
        {"name": "complete", "fn": nx.complete_graph},
        {"name": "star", "fn": nx.star_graph},
    ]

    for graph_opt in graphs:
        graph = graph_opt["fn"](args.nodes)

        cost_fn = QuadraticCost(args.nodes)
        gt = GradientTracking(cost_fn, max_iters=args.iters, alpha=1e-2)

        zz, cost, gradient_magnitude = gt.run(graph)

        if not args.no_plots:
            _, ax = plt.subplots()
            ax.plot(np.arange(zz.shape[0]), zz)
            ax.grid()
            ax.set_title(f"{graph_opt['name']} - Gradient tracking")

            _, ax = plt.subplots()
            ax.plot(np.arange(zz.shape[0] - 1), cost[:-1])
            ax.grid()
            ax.set_title(f"{graph_opt['name']} - Cost")

            _, ax = plt.subplots()
            ax.semilogy(np.arange(zz.shape[0] - 1), gradient_magnitude[1:])
            ax.grid()
            ax.set_title(f"{graph_opt['name']} - Gradient magnitude")

            plt.show()


if __name__ == "__main__":
    main()
