import argparse

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import ros.src.surveillance.surveillance.phi as phi
from ros.src.surveillance.surveillance.aggregative_tracking import AggregativeTracking
from ros.src.surveillance.surveillance.costs_fn import CorridorCost, CorridorCostV2, SurveillanceCost
from ros.src.surveillance.surveillance.functions import animation, animation2

np.random.seed(0)


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-n", "--nodes", type=int, default=4)
    argparser.add_argument("-i", "--iters", type=int, default=1000)
    argparser.add_argument("--no-plots", action="store_true", default=False)

    args = argparser.parse_args()

    # Task 2.1: different tuning parameter of cost function, different targets
    """
    Case 1 - Targets far, tradeoff = 1.0
    Case 2 - Same targets, tradeoff = 0.1
    Case 3 - Same targets, tradeoff = 10.0
    Case 4 - Targets close, tradeoff = 1.0
    Case 5 - Targets close, tradeoff = 0.1
    Case 6 - Targets close, tradeoff = 10.0

    Comment: If the tradeoff tends to 0, the robots are very close to each other but they struggle to reach the targets.
    If the tradeoff tends to be large, the robots are very close to the targets but they are far from each other.
    Tradeoff = 1.0 shows a good behaviour.

    If the initial distance between the targets diminishes, the behaviour is always the same.
    However, with larger distance and low tradeoff, the robots also with 1000 iterations are very far from targets.

    """

    distances = [10, 1]

    targets = np.random.rand(args.nodes, 2) * 10
    zz_init = np.random.rand(args.nodes, 2) * 10

    init_targets_list = [(zz_init - distance / 2, targets + distance / 2) for distance in distances]

    tradeoff_list = [1.0, 0.1, 10.0]

    for init_targets in init_targets_list:
        for tradeoff in tradeoff_list:
            zz_init, targets = init_targets

            cost = SurveillanceCost(tradeoff)
            algo = AggregativeTracking(cost, phi.Identity(), max_iters=args.iters, alpha=1e-2)

            graph = nx.path_graph(args.nodes)
            zz, cost, gradient_magnitude, kk = algo.run(graph, zz_init, targets, d=2)

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

                # plot trajectories
                print(f"Tradeoff {tradeoff}")
                for jj in range(args.nodes):
                    plt.plot(
                        zz[:, jj, 0],
                        zz[:, jj, 1],
                        linewidth=1,
                        color="black",
                        linestyle="dashed",
                        label=f"Trajectory {jj}",
                    )

                    plt.scatter(zz[-1, jj, 0], zz[-1, jj, 1], color="orange", label=f"Final position {jj}", marker="x")

                    plt.plot(targets[:, 0], targets[:, 1], "bx")
                    plt.plot(zz_init[:, 0], zz_init[:, 1], "ro")

                    plt.title(f"Tradeoff = {tradeoff}")

                    print(f"Final distance from target node {jj}: ", np.linalg.norm(zz[-1, jj] - targets))

                plt.show()

    # Task 2.1.3: Animation

    # plt.figure("Animation")
    # animation(zz, np.linspace(0, kk, kk), nx.adjacency_matrix(graph).toarray(), targets)

    # # Corridor

    # top_wall = {"x_start": -15, "x_end": 15, "y": 5, "res": 1000}
    # bottom_wall = {"x_start": -15, "x_end": 15, "y": -5, "res": 1000}

    # middle = np.array((0, (top_wall["y"] - bottom_wall["y"]) / 2 + bottom_wall["y"]))

    # x_offset = 10
    # y_offset = 20
    # random_pos_target = np.array(
    #     (
    #         np.random.rand(args.nodes) * top_wall["x_end"] * 10 + top_wall["x_end"] + x_offset,
    #         np.random.rand(args.nodes) * np.abs(top_wall["y"]) * 5 * 2,
    #     )
    # ).T

    # targets_list = [
    #     random_pos_target,
    #     np.column_stack((random_pos_target[:, 0], random_pos_target[:, 1] - y_offset)),
    #     np.column_stack((random_pos_target[:, 0], random_pos_target[:, 1] + y_offset)),
    #     np.column_stack((random_pos_target[:, 0], random_pos_target[:, 1] + y_offset)),
    # ]

    # random_initial_poses = np.array(
    #     (
    #         np.random.rand(args.nodes) * top_wall["x_start"] * 10 + top_wall["x_start"] - x_offset,
    #         np.random.rand(args.nodes) * np.abs(top_wall["y"]) * 5 * 2,
    #     )
    # ).T

    # initial_poses_list = [
    #     np.column_stack((random_initial_poses[:, 0], random_initial_poses[:, 1] - y_offset)),
    #     random_initial_poses,
    #     np.column_stack((random_initial_poses[:, 0], random_initial_poses[:, 1] + y_offset)),
    #     np.column_stack((random_initial_poses[:, 0], random_initial_poses[:, 1] - y_offset)),
    # ]

    # nobstacles = 150
    # obstacles = np.column_stack(
    #     (
    #         np.array(
    #             (
    #                 np.tile(top_wall["x_start"], nobstacles),
    #                 np.linspace(top_wall["y"], top_wall["y"] + y_offset * 4, nobstacles),
    #             )
    #         ),
    #         np.array(
    #             (
    #                 np.tile(bottom_wall["x_start"], nobstacles),
    #                 np.linspace(bottom_wall["y"], bottom_wall["y"] - y_offset * 4, nobstacles),
    #             )
    #         ),
    #         # np.array(
    #         #     (
    #         #         np.linspace(top_wall["x_start"], top_wall["x_end"], nobstacles),
    #         #         np.tile(top_wall["y"], nobstacles),
    #         #     )
    #         # ),
    #         # np.array(
    #         #     (
    #         #         np.linspace(bottom_wall["x_start"], bottom_wall["x_end"], nobstacles),
    #         #         np.tile(bottom_wall["y"], nobstacles),
    #         #     )
    #         # ),
    #     )
    # ).T

    # for i in range(1, 2):
    #     # plt.figure("Corridor", figsize=(20, 20))

    #     # plt.plot(targets_list[i][:, 0], targets_list[i][:, 1], "bx")
    #     # plt.plot(initial_poses_list[i][:, 0], initial_poses_list[i][:, 1], "ro")
    #     # plt.plot(obstacles[:, 0], obstacles[:, 1], "o")

    #     # plt.xlim(-50, 50)  # Set the x-axis limits
    #     # plt.ylim(-50, 50)  # Set the y-axis limits
    #     # plt.show()

    #     cost = CorridorCost(obstacles=obstacles, alpha=0.2, beta=1.0, gamma=1.0, dd=1)
    #     algo = AggregativeTracking(cost, phi.Identity(), max_iters=args.iters, alpha=1e-2)

    #     graph = nx.path_graph(args.nodes)
    #     zz, cost, gradient_magnitude, kk = algo.run(graph, initial_poses_list[i], targets_list[i], d=2)

    #     if not args.no_plots:
    #         _, ax = plt.subplots(3, 1, figsize=(10, 10))
    #         ax[0].plot(np.arange(zz.shape[0]), zz[:, :, 0])
    #         ax[0].grid()
    #         ax[0].set_title("Aggregative tracking")

    #         ax[1].plot(np.arange(zz.shape[0] - 1), cost[:-1])
    #         ax[1].grid()
    #         ax[1].set_title("Cost")

    #         ax[2].semilogy(np.arange(zz.shape[0] - 1), gradient_magnitude[1:])
    #         ax[2].grid()
    #         ax[2].set_title("Gradient magnitude")

    #         plt.show()

    #         # plot trajectories
    #         for j in range(args.nodes):
    #             plt.plot(
    #                 zz[:, j, 0],
    #                 zz[:, j, 1],
    #                 linewidth=1,
    #                 color="black",
    #                 linestyle="dashed",
    #                 label=f"Trajectory {j}",
    #             )

    #             plt.scatter(zz[-1, j, 0], zz[-1, j, 1], color="orange", label=f"Final position {j}", marker="x")

    #             plt.plot(targets_list[i][:, 0], targets_list[i][:, 1], "bx")
    #             plt.plot(initial_poses_list[i][:, 0], initial_poses_list[i][:, 1], "ro")
    #             plt.plot(obstacles[:, 0], obstacles[:, 1], "o", color="gray")

    #         plt.show()

    #         # animation2(
    #         #     zz,
    #         #     np.linspace(0, kk, kk),
    #         #     nx.adjacency_matrix(graph).toarray(),
    #         #     targets_list[i],
    #         #     top_wall,
    #         #     bottom_wall,
    #         #     middle,
    #         #     obstacles,
    #         # )


if __name__ == "__main__":
    main()
