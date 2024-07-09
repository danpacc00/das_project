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
# np.random.seed(0)

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-n", "--nodes", type=int, default=10)
    argparser.add_argument("-i", "--iters", type=int, default=1000)
    argparser.add_argument("-p", "--max-points", type=int, default=1000)
    argparser.add_argument("--no-plots", action="store_true", default=False)
    argparser.add_argument("--skip", type=str, default="")

    args = argparser.parse_args()

    # Compute the list of tasks to skip
    skipped = [int(item) for item in args.skip.split(",")] if args.skip else []

    graphs = [
        {"name": "path", "fn": nx.path_graph},
        {"name": "cycle", "fn": nx.cycle_graph},
        {"name": "star", "fn": nx.star_graph},
        {"name": "complete", "fn": nx.complete_graph},
    ]

    # Task 1.1
    if 1 not in skipped:
        dim = 2
        zz0 = np.random.uniform(-5, 5, size=(args.nodes, dim))
        for graph_opt in graphs:
            # The the nx library, the star graph central node does not count in the total number of nodes
            # so to obtain the desired number of nodes, we need to subtract 1
            graph = graph_opt["fn"](args.nodes - 1 if graph_opt["name"] == "star" else args.nodes)

            cost_fn = QuadraticCost(args.nodes, d=dim)
            gt = GradientTracking(cost_fn, max_iters=args.iters, alpha=1e-2)

            zz, cost, gradient_magnitude = gt.run(graph, d=dim, zz0=zz0.copy())
            print("Optimal value: ", cost_fn.optimal())

            if not args.no_plots:
                fig, ax = plt.subplots(1, 2, figsize=(10, 10))
                fig.suptitle(
                    f"Gradient tracking with Quadratic Cost Function (Graph = {graph_opt['name']}, N = {args.nodes}, d = {dim}, Iterations = {len(cost)})"
                )

                ax[0].semilogx(np.arange(zz.shape[0]), zz[:, :, 0])
                ax[0].grid()
                ax[0].set_title("x0")
                ax[1].semilogx(np.arange(zz.shape[0]), zz[:, :, 1])
                ax[1].grid()
                ax[1].set_title("x1")

                ax[0].set_xlabel("Iterations (logarithmic scale)")
                ax[0].set_ylabel("x0")

                ax[1].set_xlabel("Iterations (logarithmic scale)")
                ax[1].set_ylabel("x1")

                plt.show()

                fig, ax = plt.subplots(1, 2, figsize=(10, 10))
                fig.suptitle(
                    f"Gradient tracking with Quadratic Cost Function (Graph = {graph_opt['name']}, N = {args.nodes}, d = {dim}, Iterations = {len(cost)})"
                )
                ax[0].semilogy(np.arange(zz.shape[0] - 1), cost[:-1])
                ax[0].grid()
                ax[0].set_title("Cost")
                ax[1].semilogy(np.arange(zz.shape[0] - 1), gradient_magnitude[1:])
                ax[1].grid()
                ax[1].set_title("Gradient magnitude")

                ax[0].set_xlabel("Iterations")
                ax[0].set_ylabel("Cost (logarithmic scale)")

                ax[1].set_xlabel("Iterations")
                ax[1].set_ylabel("Gradient magnitude (logarithmic scale)")

                plt.show()

    # Classification cases
    params_list = [
        {"values": np.array((1.5, -0.5, 1.5, 0.5, 1.0)), "stepsize": 1e-3, "max_iters": 3500},  # Vertical ellipse
        {"values": np.array((1.0, 2.0, 1.0, 2.5, 1)), "stepsize": 1e-2, "max_iters": 1500},  # Horizontal ellipse -OK
        {"values": np.array((3.5, 2.0, 1.0, -2.5, 0.5)), "stepsize": 5e-3, "max_iters": 3500},  # Hyperbola
    ]

    dimension = params_list[0]["values"].shape[0] # dimension of the parameter vector theta
    datasets = []

    for params in params_list:
        #TODO: Uncomment/remove next line
        # npoints = np.random.randint(500, args.max_points)
        npoints = 1000
        dataset = create_labeled_dataset(params["values"], M=npoints)
        datasets.append(dataset)

        a, b, c, d, e = params["values"]
        real_theta = np.array((a, b, c, d, -(e**2)))
        #TODO: Uncomment/remove next line
        # initial_theta = real_theta + real_theta * 0.7
        initial_theta = np.random.uniform(-5, 5, size=dimension)

        label = f"Real Separating Function (${a}x+{b}y+{c}x^2+{d}y^2={e}^2$)"

        if b < 0:
            label = f"Real Separating Function (${a}x{b}y+{c}x^2+{d}y^2={e}^2$)"
        elif d < 0:
            label = f"Real Separating Function (${a}x+{b}y+{c}x^2{d}y^2={e}^2$)"

        real_classifier = {
            "params": params["values"],
            "color": "green",
            "label": label,
        }

        if not args.no_plots:
            plot.dataset(
                f"Dataset with Nonlinear Separating Function. Number of points: {npoints}", dataset, real_classifier
            )

        # Task 1.2
        if 2 not in skipped:
            # We use .copy() to avoid modifying the initial_theta array since it is used in task 1.3 as well
            theta_hat, costs, gradient_magnitude = centralized_gradient(
                dataset, initial_theta=initial_theta.copy(), max_iters=args.iters, alpha=1e-2, d=dimension
            )
            plot.results(
                dataset, theta_hat, real_theta, costs, gradient_magnitude, title="Centralised gradient classification"
            )

        # Task 1.3
        if 3 not in skipped:
            datasets = np.array_split(dataset, args.nodes)
            cost_fn = LogisticRegressionCost(datasets)
            gt = GradientTracking(cost_fn, max_iters=params["max_iters"], alpha=params["stepsize"])

            graph = nx.cycle_graph(args.nodes)
            zz, costs, gradient_magnitude = gt.run(graph, d=dimension, zz0=initial_theta)

            if not args.no_plots:
                fig, ax = plt.subplots(1, 2, figsize=(20, 10))
                ax[0].semilogx(np.arange(zz.shape[0]), zz[:, :, 0])
                ax[0].grid()
                ax[0].set_title("a")
                ax[1].semilogx(np.arange(zz.shape[0]), zz[:, :, 1])
                ax[1].grid()
                ax[1].set_title("b")
                plt.suptitle(
                    f"Gradient tracking with Logistic Regression Cost Function (graph = Cycle, nodes = {args.nodes}, d = {dimension}, iters = {len(costs)})"
                )
                plt.show()

                fig, ax = plt.subplots(1, 2, figsize=(20, 10))
                ax[0].semilogx(np.arange(zz.shape[0]), zz[:, :, 2])
                ax[0].grid()
                ax[0].set_title("c")
                ax[1].semilogx(np.arange(zz.shape[0]), zz[:, :, 3])
                ax[1].grid()
                ax[1].set_title("d")
                plt.suptitle(
                    f"Gradient tracking with Logistic Regression Cost Function (graph = Cycle, nodes = {args.nodes}, d = {dimension}, iters = {len(costs)})"
                )
                plt.show()

                plt.semilogx(np.arange(zz.shape[0]), zz[:, :, 4])
                plt.ylabel("bias")
                plt.grid()
                plt.title(
                    f"Gradient tracking with Logistic Regression Cost Function (graph = Cycle, nodes = {args.nodes}, d = {dimension}, iters = {len(costs)})"
                )
                plt.show()

                theta_hat = zz[-1, 0, :]
                plot.results(
                    dataset,
                    theta_hat,
                    real_theta,
                    costs,
                    gradient_magnitude,
                    no_plots=args.no_plots,
                    title="Gradient Tracking classification",
                )


if __name__ == "__main__":
    main()
