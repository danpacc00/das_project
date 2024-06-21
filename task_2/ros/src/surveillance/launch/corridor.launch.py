import networkx as nx
import numpy as np
from launch import LaunchDescription
from launch_ros.actions import Node

np.random.seed(0)

N = 4  # Number of nodes (they represent the agents in this example)
G = nx.path_graph(N)  # Create a graph

# initial_pose = np.random.rand(N, 2) * 10 - 5
initial_pose = np.zeros((N, 2))

Adj = nx.adjacency_matrix(G).toarray()

top_wall = {"x_start": -15, "x_end": 15, "y": 10, "res": 1000}
bottom_wall = {"x_start": -15, "x_end": 15, "y": -10, "res": 1000}

middle = np.array((0, (top_wall["y"] - bottom_wall["y"]) / 2 + bottom_wall["y"]))

x_offset = 10
y_offset = 20
random_pos_target = np.array(
    (
        np.random.rand(N) * top_wall["x_end"] * 10,
        np.random.rand(N) * 40 - 20,
    )
).T

targets_list = [
    np.column_stack((random_pos_target[:, 0], random_pos_target[:, 1] + y_offset)),
    random_pos_target,
    np.column_stack((random_pos_target[:, 0], random_pos_target[:, 1] - y_offset)),
    np.column_stack((random_pos_target[:, 0], random_pos_target[:, 1] + y_offset)),
]

random_initial_poses = np.array(
    (
        np.random.rand(N) * top_wall["x_start"] * 10 + top_wall["x_start"] - x_offset,
        np.random.rand(N) * 10 - 5,
    )
).T

initial_poses_list = [
    np.column_stack((random_initial_poses[:, 0], random_initial_poses[:, 1] + y_offset)),
    random_initial_poses,
    np.column_stack((random_initial_poses[:, 0], random_initial_poses[:, 1] - y_offset)),
    np.column_stack((random_initial_poses[:, 0], random_initial_poses[:, 1] - y_offset)),
]

timer_period = 0.01

max_iters = 50000


def generate_launch_description():
    nodes = []
    AA = np.zeros(shape=(N, N))

    for ii in range(N):
        N_ii = np.nonzero(Adj[ii])[0]
        deg_ii = len(N_ii)
        for jj in N_ii:
            deg_jj = len(np.nonzero(Adj[jj])[0])
            AA[ii, jj] = 1 / (1 + max([deg_ii, deg_jj]))

    AA += np.eye(N) - np.diag(np.sum(AA, axis=0))

    plotter_node = Node(
        package="surveillance",
        namespace="plotter",
        executable="plotter",
        parameters=[
            {
                "id": "Plotter",
                "nodes": N,
                "Adj": [int(adj) for adj in Adj.flatten()],
                "zz_init": [float(init) for init in initial_poses_list[0].flatten()],
                "targets": [float(target) for target in targets_list[0].flatten()],
                "timer_period": timer_period,
                "max_iters": max_iters,
                "cost_type": "corridor",
            },
        ],
        output="screen",
    )

    nodes.append(plotter_node)

    for i in range(N):
        N_ii = list(G.neighbors(i))

        node = Node(
            package="surveillance",
            namespace=f"warden{i}",
            executable="warden",
            parameters=[
                {
                    "id": i,
                    "neighbors": N_ii,
                    "weights": [float(weight) for weight in AA[i]],
                    "initial_pose": list(initial_poses_list[0][i]),
                    "target": list(targets_list[0][i]),
                    "timer_period": timer_period,
                    "max_iters": max_iters,
                    "cost_type": "corridor",
                    "alpha": 1e-3,
                },
            ],
            output="screen",
        )

        nodes.append(node)

    return LaunchDescription(nodes)
