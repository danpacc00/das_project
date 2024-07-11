import networkx as nx
import numpy as np
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

np.random.seed(1)

LAUNCH_ARGS_NAMES = [
    "nodes",
    "tradeoff",
    "distance",
    "timer_period",
    "max_iters",
]

LAUNCH_ARGS_TYPES = {
    "nodes": int,
    "tradeoff": float,
    "distance": float,
    "timer_period": float,
    "max_iters": int,
}

LAUNCH_ARGS = [
    DeclareLaunchArgument("nodes", default_value="4"),
    DeclareLaunchArgument("tradeoff"),
    DeclareLaunchArgument("distance"),
    DeclareLaunchArgument("timer_period", default_value="0.01"),
    DeclareLaunchArgument("max_iters", default_value="40000"),
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

    targets = np.array(
        (
            np.random.rand(args["nodes"]) * 10 + args["distance"] / 2,
            np.random.rand(args["nodes"]) * 50 - args["distance"] / 2,
        )
    ).T

    zz_init = np.array(
        (
            np.random.rand(args["nodes"]) * 10 - args["distance"] / 2,
            np.random.rand(args["nodes"]) * 50 + args["distance"] / 2,
        )
    ).T

    plotter_node = Node(
        package="surveillance",
        namespace="plotter",
        executable="plotter",
        parameters=[
            {
                "id": "Plotter",
                "nodes": args["nodes"],
                "Adj": [int(adj) for adj in Adj.flatten()],
                "targets": [float(target) for target in targets.flatten()],
                "zz_init": [float(init) for init in zz_init.flatten()],
                "timer_period": args["timer_period"],
                "max_iters": args["max_iters"],
                "cost_type": "surveillance",
                "tradeoff": args["tradeoff"],
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
                    "initial_pose": list(zz_init[i]),
                    "target": list(targets[i]),
                    "timer_period": args["timer_period"],
                    "max_iters": args["max_iters"],
                    "cost_type": "surveillance",
                    "tradeoff": args["tradeoff"],
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
