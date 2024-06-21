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

distances = [10, 1]

targets = np.random.rand(N, 2) * 10
zz_init = np.random.rand(N, 2) * 10

init_agents_list = [zz_init - distance / 2 for distance in distances]
init_targets_list = [targets + distance / 2 for distance in distances]

timer_period = 0.01

max_iters = 1000


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
                "targets": [float(target) for target in init_targets_list[0].flatten()],
                "zz_init": [float(init) for init in init_agents_list[0].flatten()],
                "timer_period": timer_period,
                "max_iters": max_iters,
                "cost_type": "surveillance",
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
                    "initial_pose": list(init_agents_list[0][i]),
                    "target": list(init_targets_list[0][i]),
                    "timer_period": timer_period,
                    "max_iters": max_iters,
                    "cost_type": "surveillance",
                },
            ],
            output="screen",
        )

        nodes.append(node)

    return LaunchDescription(nodes)
