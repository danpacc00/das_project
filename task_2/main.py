import argparse
import signal

import networkx as nx
import numpy as np
import ros.src.surveillance.surveillance.phi as phi
import ros.src.surveillance.surveillance.plot as plot
from ros.src.surveillance.surveillance.aggregative_tracking import AggregativeTracking
from ros.src.surveillance.surveillance.costs_fn import (
    CorridorCost,
    SurveillanceCost,
)
from ros.src.surveillance.surveillance.functions import corridor_animation, simple_animation

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
                zz, ss, cost, gradient_magnitude, diff_barycenter_s, v_nabla2_diff = algo.run(
                    graph, zz_init, targets, d=2
                )

                if not args.no_plots:
                    plot.ss_estimates(ss)

                    plot.convergence(diff_barycenter_s, v_nabla2_diff)

                    plot.cost_gradient(cost, gradient_magnitude, title_suffix=f"($\\gamma = {tradeoff}$)")

                    plot.trajectories(zz, targets, zz_init, case, tradeoff)

                if not args.skip_animations:
                    simple_animation(
                        zz,
                        np.linspace(0, zz.shape[0] - 1, zz.shape[0] - 1),
                        nx.adjacency_matrix(graph).toarray(),
                        targets,
                    )

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

            alpha = 0.8
            cost = CorridorCost(alpha)
            algo = AggregativeTracking(cost, phi.Identity(), max_iters=args.iters, alpha=1e-3)

            graph = nx.path_graph(args.nodes)
            zz, ss, cost, gradient_magnitude, diff_barycenter_s, v_nabla2_diff = algo.run(
                graph, initial_poses_list[i], targets, d=2
            )

            if not args.no_plots:
                plot.ss_estimates(ss)

                plot.convergence(diff_barycenter_s, v_nabla2_diff)

                plot.cost_gradient(cost, gradient_magnitude, title_suffix="($\\alpha = 0.8$)")

                plot.trajectories(
                    zz,
                    targets,
                    initial_poses_list[i],
                    i,
                    alpha,
                    additional_elements=[lambda: plot.corridor(top_wall, bottom_wall, y_offset, x, g_1, g_2)],
                )

                corridor_animation(
                    zz,
                    np.linspace(0, zz.shape[0] - 1, zz.shape[0] - 1),
                    nx.adjacency_matrix(graph).toarray(),
                    targets_list[i],
                    top_wall,
                    bottom_wall,
                    y_offset,
                )


if __name__ == "__main__":
    main()
