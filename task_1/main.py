import argparse
import signal

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import plot
from costs_fn import LogisticRegressionCost, QuadraticCost
from dataset import centralized_gradient, create_labeled_dataset
from gradient_tracking import GradientTracking

signal.signal(signal.SIGINT, signal.SIG_DFL)
np.random.seed(0)


def main():
    # Task 1.1
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-n", "--nodes", type=int, default=10)
    argparser.add_argument("-d", "--dimension", type=int, default=2)
    argparser.add_argument("-i", "--iters", type=int, default=1000)
    argparser.add_argument("-p", "--max-points", type=int, default=1000)
    argparser.add_argument("--no-plots", action="store_true", default=False)
    argparser.add_argument("--skip", type=str, default="")

    args = argparser.parse_args()

    skipped = [int(item) for item in args.skip.split(",")] if args.skip else []

    graphs = [
        {"name": "path", "fn": nx.path_graph},
        {"name": "cycle", "fn": nx.cycle_graph},
        {"name": "complete", "fn": nx.complete_graph},
        {"name": "star", "fn": nx.star_graph},
    ]

    # Task 1.1
    if 1 not in skipped:
        for graph_opt in graphs:
            graph = graph_opt["fn"](args.nodes - 1 if graph_opt["name"] == "star" else args.nodes)

            cost_fn = QuadraticCost(args.nodes, d=args.dimension)
            gt = GradientTracking(cost_fn, max_iters=args.iters, alpha=1e-2)

            zz, cost, gradient_magnitude = gt.run(
                graph, d=args.dimension, zz0=np.random.uniform(-5, 5, size=(args.nodes, args.dimension))
            )

            print("Optimal value: ", cost_fn.optimal())

            if not args.no_plots:
                fig, ax = plt.subplots(2, 2, figsize=(10, 10))
                fig.suptitle(
                    f"Gradient tracking with Quadratic Cost Function (Graph = {graph_opt['name']}, N = {args.nodes}, d = {args.dimension}, Iterations = {args.iters})"
                )

                ax[0, 0].plot(np.arange(zz.shape[0]), zz[:, :, 0])
                ax[0, 0].grid()
                ax[0, 0].set_title("x0")

                ax[0, 1].plot(np.arange(zz.shape[0]), zz[:, :, 1])
                ax[0, 1].grid()
                ax[0, 1].set_title("x1")

                ax[1, 0].semilogy(np.arange(zz.shape[0] - 1), cost[:-1])
                ax[1, 0].grid()
                ax[1, 0].set_title("Cost")

                ax[1, 1].semilogy(np.arange(zz.shape[0] - 1), gradient_magnitude[1:])
                ax[1, 1].grid()
                ax[1, 1].set_title("Gradient magnitude")

                plt.show()

    params_list = [
        np.array((9.0, 2.0, 1.0, 5.0, 0.5)),
        np.array((9.0, 2.0, 1.0, -5.0, 0.5)),
        np.random.uniform(1, 10, size=5).round(),
    ]
    dimension = params_list[0].shape[0]
    datasets = []

    for params in params_list:
        npoints = np.random.randint(200, args.max_points)
        dataset = create_labeled_dataset(params, M=npoints)
        datasets.append(dataset)

        a, b, c, d, e = params
        real_theta = np.array((a, b, c, d, -(e**2)))
        initial_theta = np.zeros(dimension)  # real_theta + real_theta * 0.7
        real_classifier = {
            "params": params,
            "color": "green",
            "label": f"Real Separating Function (${a}x+{b}y+{c}x^2+{d}y^2={e}^2$)",
        }

        if not args.no_plots:
            plot.dataset(
                f"Dataset with Nonlinear Separating Function. Number of points: {npoints}", dataset, real_classifier
            )

        # Task 1.2
        if 2 not in skipped:
            theta_hat, costs, gradient_magnitude = centralized_gradient(
                dataset, initial_theta=initial_theta.copy(), max_iters=args.iters, alpha=1e-5, d=dimension
            )
            plot.results(dataset, theta_hat, real_theta, costs, gradient_magnitude)

        # Task 1.3
        if 3 not in skipped:
            datasets = np.array_split(dataset, args.nodes)
            cost_fn = LogisticRegressionCost(datasets)
            gt = GradientTracking(cost_fn, max_iters=args.iters, alpha=1e-4)

            for graph_opt in graphs:
                graph = graph_opt["fn"](args.nodes - 1 if graph_opt["name"] == "star" else args.nodes)

                zz, costs, gradient_magnitude = gt.run(graph, d=dimension, zz0=initial_theta)

                if not args.no_plots:
                    axes = [
                        plt.subplot2grid(shape=(2, 6), loc=(0, 0), colspan=2),
                        plt.subplot2grid((2, 6), (0, 2), colspan=2),
                        plt.subplot2grid((2, 6), (0, 4), colspan=2),
                        plt.subplot2grid((2, 6), (1, 1), colspan=2),
                        plt.subplot2grid((2, 6), (1, 3), colspan=2),
                    ]
                    for i, label in enumerate(["a", "b", "c", "d", "bias"]):
                        axes[i].semilogx(np.arange(zz.shape[0]), zz[:, :, i])
                        axes[i].grid()
                        axes[i].set_xlabel("Iterations")
                        axes[i].set_ylabel(label)

                    plt.subplots_adjust(hspace=0.25, wspace=1.0)
                    plt.suptitle(
                        f"Gradient tracking with Logistic Regression Cost Function (graph = {graph_opt['name']}, nodes = {args.nodes}, d = {dimension}, iters = {args.iters})"
                    )
                    plt.show()

                    theta_hat = zz[-1, 0, :]
                    plot.results(dataset, theta_hat, real_theta, costs, gradient_magnitude, no_plots=args.no_plots)


if __name__ == "__main__":
    main()
