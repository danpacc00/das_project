import networkx as nx
import numpy as np
from launch import LaunchDescription
from launch_ros.actions import Node

# import sys

# sys.path.insert(0, "/home/danielepc/.asdf/installs/python/3.10.13/lib/python3.10/site-packages")

import networkx as nx

N = 4  # Number of nodes (they represent the agents in this example)
G = nx.path_graph(N)  # Create a graph

# initial_pose = np.random.rand(N, 2) * 10 - 5
initial_pose = np.zeros((N, 2))

Adj = nx.adjacency_matrix(G).toarray()

# targets = np.random.rand(N, 2) * 10 - 5
targets = np.array(
    [[0.48813504, 2.15189366], [1.02763376, 0.44883183], [-0.76345201, 1.45894113], [-0.62412789, 3.91773001]]
)

timer_period = 0.01


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
                "targets": [float(target) for target in targets.flatten()],
                "timer_period": timer_period,
                "max_iters": 150,
            },
        ],
        output="screen",
        # prefix='xterm -title "Plotter" -hold -e',
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
                    "initial_pose": list(initial_pose[i]),
                    "target": list(targets[i]),
                    "timer_period": timer_period,
                },
            ],
            # output="screen",
            # prefix=f'gnome-terminal -- bash -c "echo -ne \'\\033]0;Agent {i}\\007\'; $0; read -p \'Press any key to close this terminal...\'"'
            # prefix=f'xterm -title "Agent {i}" -hold -e',
        )

        nodes.append(node)

    return LaunchDescription(nodes)
