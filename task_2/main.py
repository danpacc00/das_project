import argparse

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import ros.src.surveillance.surveillance.phi as phi
from ros.src.surveillance.surveillance.aggregative_tracking import AggregativeTracking
from ros.src.surveillance.surveillance.costs_fn import CorridorCost, SurveillanceCost
from ros.src.surveillance.surveillance.functions import animation2

# np.random.seed(0)


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-n", "--nodes", type=int, default=4)
    argparser.add_argument("-i", "--iters", type=int, default=1000)
    argparser.add_argument("--no-plots", action="store_true", default=False)

    args = argparser.parse_args()

    # Corridor

    top_wall = {"x_start": -15, "x_end": 15, "y": 5, "res": 1000}
    bottom_wall = {"x_start": -15, "x_end": 15, "y": -5, "res": 1000}

    middle = np.array((0, (top_wall["y"] - bottom_wall["y"]) / 2 + bottom_wall["y"]))

    x_offset = 10
    y_offset = 20
    random_pos_target = np.array(
        (
            np.random.rand(args.nodes) * top_wall["x_end"] + top_wall["x_end"] + x_offset,
            np.random.rand(args.nodes) * np.abs(top_wall["y"]) * 2,
        )
    ).T

    targets_list = [
        random_pos_target,
        np.column_stack((random_pos_target[:, 0], random_pos_target[:, 1] - y_offset)),
        np.column_stack((random_pos_target[:, 0], random_pos_target[:, 1] + y_offset)),
        np.column_stack((random_pos_target[:, 0], random_pos_target[:, 1] + y_offset)),
    ]

    random_initial_poses = np.array(
        (
            np.random.rand(args.nodes) * top_wall["x_start"] + top_wall["x_start"] - x_offset,
            np.random.rand(args.nodes) * np.abs(top_wall["y"]) * 2,
        )
    ).T

    initial_poses_list = [
        random_initial_poses,
        np.column_stack((random_initial_poses[:, 0], random_initial_poses[:, 1] + y_offset)),
        np.column_stack((random_initial_poses[:, 0], random_initial_poses[:, 1] - y_offset)),
        np.column_stack((random_initial_poses[:, 0], random_initial_poses[:, 1] - y_offset)),
    ]

    nobstacles = 250
    obstacles = np.column_stack(
        (
            np.array(
                (
                    np.tile(top_wall["x_start"], nobstacles),
                    np.linspace(top_wall["y"], top_wall["y"] + y_offset * 4, nobstacles),
                )
            ),
            np.array(
                (
                    np.tile(bottom_wall["x_start"], nobstacles),
                    np.linspace(bottom_wall["y"], bottom_wall["y"] - y_offset * 4, nobstacles),
                )
            ),
            np.array(
                (
                    np.linspace(top_wall["x_start"], top_wall["x_end"], nobstacles),
                    np.tile(top_wall["y"], nobstacles),
                )
            ),
            np.array(
                (
                    np.linspace(bottom_wall["x_start"], bottom_wall["x_end"], nobstacles),
                    np.tile(bottom_wall["y"], nobstacles),
                )
            ),
        )
    ).T

    for i in range(1, len(targets_list)):
        plt.figure("Corridor", figsize=(20, 20))

        plt.plot(targets_list[i][:, 0], targets_list[i][:, 1], "bx")
        plt.plot(initial_poses_list[i][:, 0], initial_poses_list[i][:, 1], "ro")
        plt.plot(obstacles[:, 0], obstacles[:, 1], "o")

        plt.xlim(-50, 50)  # Set the x-axis limits
        plt.ylim(-50, 50)  # Set the y-axis limits
        plt.show()

        cost = CorridorCost(obstacles=obstacles, alpha=0.8, beta=10, gamma=0.5, dd=0.1)
        algo = AggregativeTracking(cost, phi.Identity(), max_iters=args.iters, alpha=1e-2)

        graph = nx.path_graph(args.nodes)
        zz, cost, gradient_magnitude, kk = algo.run(graph, initial_poses_list[i], targets_list[i], d=2)

        if not args.no_plots:
            _, ax = plt.subplots(3, 1, figsize=(10, 10))
            ax[0].plot(np.arange(zz.shape[0]), zz[:, :, 0])
            ax[0].grid()
            ax[0].set_title("Aggregative tracking")

            ax[1].plot(np.arange(zz.shape[0] - 1), cost[:-1])
            ax[1].grid()
            ax[1].set_title("Cost")

            ax[2].semilogy(np.arange(zz.shape[0] - 1), gradient_magnitude[1:])
            ax[2].grid()
            ax[2].set_title("Gradient magnitude")

            plt.show()

            plt.figure("Animation")
            animation2(
                zz,
                np.linspace(0, kk, kk),
                nx.adjacency_matrix(graph).toarray(),
                targets_list[i],
                top_wall,
                bottom_wall,
                middle,
                obstacles,
            )


if __name__ == "__main__":
    main()
