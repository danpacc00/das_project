import argparse

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from costs_fn import LogisticRegressionCost, QuadraticCost
from dataset import centralized_gradient, create_labeled_dataset, plot_results
from dataset import cost as opt_cost_fn
from gradient_tracking import GradientTracking

np.random.seed(0)
NN = 30


def main():
    # Task 1.1
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-n", "--nodes", type=int, default=10)
    argparser.add_argument("-i", "--iters", type=int, default=1000)
    argparser.add_argument("-d", "--dataset", type=str, choices=["line", "ellipse"], default="ellipse")
    argparser.add_argument("--no-plots", action="store_true", default=False)

    args = argparser.parse_args()

    # graphs = [
    #     {"name": "cycle", "fn": nx.cycle_graph},
    #     {"name": "complete", "fn": nx.complete_graph},
    #     {"name": "star", "fn": nx.star_graph},
    # ]

    # for graph_opt in graphs:
    #     graph = graph_opt["fn"](args.nodes) if graph_opt["name"] != "star" else nx.star_graph(args.nodes - 1)

    #     cost_fn = QuadraticCost(args.nodes, d=1)
    #     gt = GradientTracking(cost_fn, max_iters=args.iters, alpha=1e-2)

    #     zz, cost, gradient_magnitude = gt.run(graph, d=1, zz0=np.random.uniform(-5, 5, size=(args.nodes, 1)))

    #     if not args.no_plots:
    #         _, ax = plt.subplots(3, 1, figsize=(10, 10))
    #         ax[0].plot(np.arange(zz.shape[0]), zz[:, :, 0])
    #         ax[0].grid()
    #         ax[0].set_title(f"{graph_opt['name']} - Gradient tracking")

    #         ax[1].plot(np.arange(zz.shape[0] - 1), cost[:-1])
    #         ax[1].grid()
    #         ax[1].set_title(f"{graph_opt['name']} - Cost")

    #         ax[2].semilogy(np.arange(zz.shape[0] - 1), gradient_magnitude[1:])
    #         ax[2].grid()
    #         ax[2].set_title(f"{graph_opt['name']} - Gradient magnitude")

    #         plt.show()

    # Task 1.2
    theta_list = [
        np.random.uniform(0, 20, size=5),
        np.array((9.0, 2.0, 1.0, 5.0, 0.5)),
        np.array((9.0, 2.0, 1.0, -5.0, 0.5)),
    ]
    dimension = theta_list[0].shape[0]
    datasets = []

    for theta in theta_list:
        datasets.append(create_labeled_dataset(theta, M=np.random.randint(500, 1000), show_plot=False))

    for dataset in datasets:
        centralized_gradient(dataset)

    # Task 1.3
    for i, dataset in enumerate(datasets):
        # Split the dataset into NN groups
        datasets = np.array_split(dataset, NN)

        cost_fn = LogisticRegressionCost(datasets)
        gt = GradientTracking(cost_fn, max_iters=args.iters, alpha=1e-4)

        graph = nx.complete_graph(NN)

        zz0 = np.zeros(dimension)
        # zz0 = np.array([9, 2, 1, 5, 0.5]) + np.array([0.1, 0.1, 0.1, 0.1, 0.1]) #Perturbare random con magnitude abbastanza importante, anche 50/60%

        zz, cost, gradient_magnitude = gt.run(graph, d=dimension, zz0=zz0)

        if not args.no_plots:
            _, ax = plt.subplots(4, 1, figsize=(10, 10))
            ax[0].plot(np.arange(zz.shape[0]), zz[:, :, 0])
            ax[0].grid()
            ax[0].set_title("Gradient tracking of the Logistic Regression Cost Function")
            ax[0].set_xlabel("Iterations")
            ax[0].set_ylabel("$\\theta[0]$")

            ax[1].plot(np.arange(zz.shape[0]), zz[:, :, 1])
            ax[1].grid()
            ax[1].set_title("Gradient tracking of the Logistic Regression Cost Function")
            ax[1].set_xlabel("Iterations")
            ax[1].set_ylabel("$\\theta[1]$")

            ax[2].plot(np.arange(zz.shape[0] - 1), cost[:-1])
            ax[2].grid()
            ax[2].set_title("Evolution of the Cost Function")
            ax[2].set_xlabel("Iterations")
            ax[2].set_ylabel("Cost")

            ax[3].semilogy(np.arange(zz.shape[0] - 1), gradient_magnitude[1:])
            ax[3].grid()
            ax[3].set_title("Gradient magnitude")
            ax[3].set_xlabel("Iterations")
            ax[3].set_ylabel("Evolution of the Norm of the Gradient")
            plt.show()

            opt_cost = opt_cost_fn(theta_list[i], dataset)
            plt.semilogy(np.arange(args.iters - 2), np.abs(cost[:-1] - opt_cost))
            plt.title("Difference between optimal and gradient tracking cost")
            plt.xlabel("Iterations")
            plt.ylabel("Cost")
            plt.grid()
            plt.show()

            plot_results(dataset, zz[-1, 0, :], title="Result using parameters by Gradient Tracking")


if __name__ == "__main__":
    main()
