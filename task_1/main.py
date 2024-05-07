import argparse

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from gradient_tracking import GradientTracking
from Dataset import create_labeled_dataset

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
        graph = (
            graph_opt["fn"](args.nodes)
            if graph_opt["name"] != "star"
            else nx.star_graph(args.nodes - 1)
        )

        # graph = graph_opt["fn"](args.nodes)
        # if graph_opt["name"] == "star":
        #     print("graph", graph)

        cost_fn = QuadraticCost(args.nodes)
        gt = GradientTracking(cost_fn, max_iters=args.iters, alpha=1e-2)

        zz, cost, gradient_magnitude = gt.run(graph)

        if not args.no_plots:
            _, ax = plt.subplots(3, 1, figsize=(10, 10))
            ax[0].plot(np.arange(zz.shape[0]), zz)
            ax[0].grid()
            ax[0].set_title(f"{graph_opt['name']} - Gradient tracking")

            ax[1].plot(np.arange(zz.shape[0] - 1), cost[:-1])
            ax[1].grid()
            ax[1].set_title(f"{graph_opt['name']} - Cost")

            ax[2].semilogy(np.arange(zz.shape[0] - 1), gradient_magnitude[1:])
            ax[2].grid()
            ax[2].set_title(f"{graph_opt['name']} - Gradient magnitude")

            plt.show()

    # Task 1.2
    labeled_dataset = create_labeled_dataset()


if __name__ == "__main__":
    main()
