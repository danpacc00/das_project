import networkx as nx
import numpy as np
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

np.random.seed(0)

LAUNCH_ARGS_NAMES = [
    "nodes",
    "timer_period",
    "max_iters",
    "case",
]

LAUNCH_ARGS_TYPES = {
    "nodes": int,
    "timer_period": float,
    "max_iters": int,
    "case": int,
}

LAUNCH_ARGS = [
    DeclareLaunchArgument("nodes", default_value="4"),
    DeclareLaunchArgument("timer_period", default_value="0.01"),
    DeclareLaunchArgument("max_iters", default_value="10000"),
    DeclareLaunchArgument("case", default_value="0"),
]


def launch_setup(context):
    args = {}
    for name in LAUNCH_ARGS_NAMES:
        args[name] = LAUNCH_ARGS_TYPES[name](LaunchConfiguration(name).perform(context))

    AA = np.zeros(shape=(args["nodes"], args["nodes"]))
    graph = nx.path_graph(args["nodes"])
    Adj = nx.adjacency_matrix(graph).toarray()

    for ii in range(args["nodes"]):
        N_ii = np.nonzero(Adj[ii])[0]
        deg_ii = len(N_ii)
        for jj in N_ii:
            deg_jj = len(np.nonzero(Adj[jj])[0])
            AA[ii, jj] = 1 / (1 + max([deg_ii, deg_jj]))

    AA += np.eye(args["nodes"]) - np.diag(np.sum(AA, axis=0))

    top_wall = {"x_start": -15, "x_end": 15, "y": 5, "res": 1000}

    x_offset = 50
    y_offset = 20
    random_pos_target = np.array(
        (
            np.random.rand(args["nodes"]) * top_wall["x_end"] * 10 + x_offset,
            np.random.rand(args["nodes"]) * 40 - 20,
        )
    ).T

    targets_list = [
        random_pos_target,
        np.column_stack((random_pos_target[:, 0], random_pos_target[:, 1] + y_offset)),
        np.column_stack((random_pos_target[:, 0], random_pos_target[:, 1] - y_offset)),
        np.column_stack((random_pos_target[:, 0], random_pos_target[:, 1] + y_offset)),
    ]

    random_initial_poses = np.array(
        (
            np.random.rand(args["nodes"]) * top_wall["x_start"] * 10 - x_offset,
            np.random.rand(args["nodes"]) * 10 - 5,
        )
    ).T

    initial_poses_list = [
        random_initial_poses,
        np.column_stack((random_initial_poses[:, 0], random_initial_poses[:, 1] + y_offset)),
        np.column_stack((random_initial_poses[:, 0], random_initial_poses[:, 1] - y_offset)),
        np.column_stack((random_initial_poses[:, 0], random_initial_poses[:, 1] - y_offset)),
    ]

    plotter_node = Node(
        package="surveillance",
        namespace="plotter",
        executable="plotter",
        parameters=[
            {
                "id": "Plotter",
                "nodes": args["nodes"],
                "Adj": [int(adj) for adj in Adj.flatten()],
                "zz_init": [float(init) for init in initial_poses_list[args["case"]].flatten()],
                "targets": [float(target) for target in targets_list[args["case"]].flatten()],
                "timer_period": args["timer_period"],
                "max_iters": args["max_iters"],
                "cost_type": "corridor",
            },
        ],
        output="screen",
    )

    nodes = []
    nodes.append(plotter_node)

    for i in range(args["nodes"]):
        N_ii = list(graph.neighbors(i))

        node = Node(
            package="surveillance",
            namespace=f"warden{i}",
            executable="warden",
            parameters=[
                {
                    "id": i,
                    "neighbors": N_ii,
                    "weights": [float(weight) for weight in AA[i]],
                    "initial_pose": list(initial_poses_list[args["case"]][i]),
                    "target": list(targets_list[args["case"]][i]),
                    "timer_period": args["timer_period"],
                    "max_iters": args["max_iters"],
                    "cost_type": "corridor",
                    "alpha": 1e-3,
                },
            ],
            output="screen",
        )

        nodes.append(node)

    return nodes


def generate_launch_description():
    opfunc = OpaqueFunction(function=launch_setup)
    ld = LaunchDescription(LAUNCH_ARGS)
    ld.add_action(opfunc)
    return ld
