import argparse
import signal

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import ros.src.surveillance.surveillance.phi as phi
from ros.src.surveillance.surveillance.aggregative_tracking import AggregativeTracking
from ros.src.surveillance.surveillance.costs_fn import (
    CorridorCost,
    SurveillanceCost,
)
from ros.src.surveillance.surveillance.functions import simple_animation, corridor_animation

signal.signal(signal.SIGINT, signal.SIG_DFL)
np.random.seed(0)


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-n", "--nodes", type=int, default=4)
    argparser.add_argument("-i", "--iters", type=int, default=40000)
    argparser.add_argument("--no-plots", action="store_true", default=False)
    argparser.add_argument("--skip", type=str, default="")
    argparser.add_argument("--skip-animations", action="store_true", default=False)

    args = argparser.parse_args()

    # Compute the list of tasks to skip
    skipped = [int(item) for item in args.skip.split(",")] if args.skip else []

    # Task 2.1
    if 1 not in skipped:
        # Represent a sort of initial distance between the agents and their targets
        distances = [10, 1]

        targets = np.random.rand(args.nodes, 2) * 10
        zz_init = np.random.rand(args.nodes, 2) * 10

        init_targets_list = [(zz_init - distance / 2, targets + distance / 2) for distance in distances]

        # Represent the importance given to the miniiization of the distance between the agent and its target.
        # The higher the value, the more the agent will try to reach its target no matter if it goes far from
        # the barycenter of the formation.
        tradeoff_list = [1.0, 0.1, 10.0]

        for case, init_targets in enumerate(init_targets_list):

            # For each case, we will try different tradeoff values
            for tradeoff in tradeoff_list:
                zz_init, targets = init_targets

                cost = SurveillanceCost(tradeoff)
                algo = AggregativeTracking(cost, phi.Identity(), max_iters=args.iters, alpha=1e-2)

                graph = nx.path_graph(args.nodes)
                zz, ss, cost, gradient_magnitude, kk = algo.run(graph, zz_init, targets, d=2)

                if not args.no_plots:
                    _, ax = plt.subplots(1, 2, figsize=(10, 10))
                    ax[0].semilogx(np.arange(ss.shape[0]), ss[:, :, 0])
                    ax[0].grid()
                    ax[0].set_title("$s_x$")

                    ax[1].semilogx(np.arange(ss.shape[0]), ss[:, :, 1])
                    ax[1].grid()
                    ax[1].set_title("$s_y$")
                    plt.suptitle(f"Barycenter estimation with tradeoff = {tradeoff}")
                    plt.show()

                    _, ax = plt.subplots(1, 2, figsize=(10, 10))
                    ax[0].semilogy(np.arange(zz.shape[0] - 1), cost[:-1])
                    ax[0].grid()
                    ax[0].set_title("Cost")

                    ax[1].semilogy(np.arange(zz.shape[0] - 1), gradient_magnitude[1:])
                    ax[1].grid()
                    ax[1].set_title("Gradient magnitude")
                    plt.suptitle(f"Aggregative tracking with tradeoff = {tradeoff}")
                    plt.show()

                    for jj in range(args.nodes):
                        plt.plot(
                            zz[:, jj, 0],
                            zz[:, jj, 1],
                            linewidth=1,
                            color="black",
                            linestyle="dashed",
                            label=f"Trajectory {jj}",
                        )

                        # Plot the final position of the agent
                        plt.scatter(zz[-1, jj, 0], zz[-1, jj, 1], color="orange", marker="x")

                        # Annotate the initial position of the agent
                        plt.annotate(
                            f"$z_{jj}^0$",
                            xy=(zz[0, jj, 0], zz[0, jj, 1]),
                            xytext=(zz[0, jj, 0] + 0.2, zz[0, jj, 1] + 0.2),
                            fontsize=12,
                            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="red"),
                        )

                        plt.plot(targets[:, 0], targets[:, 1], "bx")
                        plt.plot(zz_init[:, 0], zz_init[:, 1], "ro")

                        if case == 0:
                            label_offsets = [(0.2, 0.2), (-0.2, -0.7), (-0.8, -0.7), (0.2, -0.2)]
                        else:
                            label_offsets = [(0.1, 0.2), (0.1, 0.2), (-0.55, -0.35), (-0.55, -0.35)]

                        # Annotate the target position
                        plt.annotate(
                            f"Target {jj}",
                            xy=(targets[jj, 0], targets[jj, 1]),
                            xytext=(targets[jj, 0] + label_offsets[jj][0], targets[jj, 1] + label_offsets[jj][1]),
                            fontsize=12,
                            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="blue"),
                        )

                        plt.title(f"Agents trajectories (tradeoff = {tradeoff})")

                        print(f"Final distance from target node {jj}: ", np.linalg.norm(zz[-1, jj] - targets))

                    plt.show()

                if not args.skip_animations:
                    simple_animation(zz, np.linspace(0, kk, kk), nx.adjacency_matrix(graph).toarray(), targets)

    # Task 2.3 (Python version)
    if 3 not in skipped:
        # Define the walls of the corridor ("res" is the number of points to plot the wall)
        top_wall = {"x_start": -15, "x_end": 15, "y": 5, "res": 1000}
        bottom_wall = {"x_start": -15, "x_end": 15, "y": -5, "res": 1000}

        # Distances of the agents and targets from the corridor, both
        # in the x and y directions
        x_offset = 50
        y_offset = 20

        # First a random position for each target is generated
        random_pos_target = np.array(
            (
                np.random.rand(args.nodes) * top_wall["x_end"] * 5 + x_offset,
                np.random.rand(args.nodes) * 40 - 20,
            )
        ).T

        # Then, we create a set of experiments in which we shift the position along the y-axis
        # in order to test the robustness of the algorithm in different scenarios
        targets_list = [
            random_pos_target,
            np.column_stack((random_pos_target[:, 0], random_pos_target[:, 1] + y_offset)),
            np.column_stack((random_pos_target[:, 0], random_pos_target[:, 1] - y_offset)),
            np.column_stack((random_pos_target[:, 0], random_pos_target[:, 1] + y_offset)),
        ]

        # Generate a random initial position for each agent
        random_initial_poses = np.array(
            (
                np.random.rand(args.nodes) * top_wall["x_start"] * 5 - x_offset,
                np.random.rand(args.nodes) * 10 - 5,
            )
        ).T

        # Do the same of line 151 but for the agents
        initial_poses_list = [
            random_initial_poses,
            np.column_stack((random_initial_poses[:, 0], random_initial_poses[:, 1] + y_offset)),
            np.column_stack((random_initial_poses[:, 0], random_initial_poses[:, 1] - y_offset)),
            np.column_stack((random_initial_poses[:, 0], random_initial_poses[:, 1] - y_offset)),
        ]

        for i in range(len(targets_list)):
            targets = targets_list[i]
            x = np.linspace(-60, 60, 100)

            # Define the barrier functions used to avoid collisions with the corridor walls
            g_1 = 1e-5 * x**4 + 2
            g_2 = -(1e-5 * x**4 + 2)
            
            cost = CorridorCost(alpha=0.8)
            algo = AggregativeTracking(cost, phi.Identity(), max_iters=args.iters, alpha=1e-3)

            graph = nx.path_graph(args.nodes)
            zz, ss, cost, gradient_magnitude, kk = algo.run(graph, initial_poses_list[i], targets, d=2)

            if not args.no_plots:
                _, ax = plt.subplots(1, 2, figsize=(10, 10))
                ax[0].semilogx(np.arange(ss.shape[0]), ss[:, :, 0])
                ax[0].grid()
                ax[0].set_title("$s_x$")

                ax[1].semilogx(np.arange(ss.shape[0]), ss[:, :, 1])
                ax[1].grid()
                ax[1].set_title("$s_y$")
                plt.suptitle("Barycenter estimation (obstacles case)")
                plt.show()

                _, ax = plt.subplots(1, 2, figsize=(10, 10))
                ax[0].semilogy(np.arange(zz.shape[0] - 1), cost[:-1])
                ax[0].grid()
                ax[0].set_title("Cost")

                ax[1].semilogy(np.arange(zz.shape[0] - 1), gradient_magnitude[1:])
                ax[1].grid()
                ax[1].set_title("Gradient magnitude")
                plt.suptitle("Aggregative tracking (obstacles case)")
                plt.show()

                for jj in range(args.nodes):
                    plt.plot(
                        zz[:, jj, 0],
                        zz[:, jj, 1],
                        linewidth=1,
                        color="black",
                        linestyle="dashed",
                        label=f"Trajectory {jj}",
                    )

                    # Plot the final position of the agents
                    plt.scatter(zz[-1, jj, 0], zz[-1, jj, 1], color="orange", label=f"Final position {jj}", marker="x")

                    plt.annotate(
                        f"$z_{jj}^0$",
                        xy=(zz[0, jj, 0], zz[0, jj, 1]),
                        xytext=(zz[0, jj, 0] - 2.0, zz[0, jj, 1] + 3.5),
                        fontsize=12,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="red"),
                    )

                    plt.plot(targets[:, 0], targets[:, 1], "bx")
                    plt.plot(initial_poses_list[i][:, 0], initial_poses_list[i][:, 1], "ro")

                    # Plot the corridor walls
                    plt.plot(
                        np.linspace(top_wall["x_start"], top_wall["x_end"], top_wall["res"]),
                        np.tile(top_wall["y"], top_wall["res"]),
                        "k",
                    )
                    plt.plot(
                        np.linspace(bottom_wall["x_start"], bottom_wall["x_end"], bottom_wall["res"]),
                        np.tile(bottom_wall["y"], bottom_wall["res"]),
                        "k",
                    )
                    plt.plot(
                        np.tile(top_wall["x_start"], top_wall["res"]),
                        np.linspace(top_wall["y"], top_wall["y"] + y_offset * 4, top_wall["res"]),
                        "k",
                    )
                    plt.plot(
                        np.tile(bottom_wall["x_start"], bottom_wall["res"]),
                        np.linspace(bottom_wall["y"], bottom_wall["y"] - y_offset * 4, bottom_wall["res"]),
                        "k",
                    )
                    plt.plot(
                        np.tile(top_wall["x_end"], top_wall["res"]),
                        np.linspace(top_wall["y"], top_wall["y"] + y_offset * 4, top_wall["res"]),
                        "k",
                    )
                    plt.plot(
                        np.tile(bottom_wall["x_end"], bottom_wall["res"]),
                        np.linspace(bottom_wall["y"], bottom_wall["y"] - y_offset * 4, bottom_wall["res"]),
                        "k",
                    )

                    plt.annotate(
                        f"Target {jj}",
                        xy=(targets[jj, 0], targets[jj, 1]),
                        xytext=(targets[jj, 0] + 3.5, targets[jj, 1] + 2.5),
                        fontsize=12,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="blue"),
                    )

                    plt.title("Agents trajectories")

                    # Plot the barrier functions (they are "inside" the corridor since they are used to 
                    # keep the agents away from the walls while they are moving inside the corridor)
                    plt.plot(x, g_1, color="green", linestyle="dashed")
                    plt.plot(x, g_2, color="green", linestyle="dashed")

                plt.ylim(-60, 60)
                plt.show()

            if not args.skip_animations:
                corridor_animation(
                    zz,
                    np.linspace(0, kk, kk),
                    nx.adjacency_matrix(graph).toarray(),
                    targets_list[i],
                    top_wall,
                    bottom_wall,
                    y_offset,
                )


if __name__ == "__main__":
    main()
