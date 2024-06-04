import matplotlib.pyplot as plt
import numpy as np
from launch import LaunchDescription
from launch_ros.actions import Node

# import sys

# sys.path.insert(0, "/home/danielepc/.asdf/installs/python/3.10.13/lib/python3.10/site-packages")

import networkx as nx

N = 4  # Number of nodes (they represent the agents in this example)
G = nx.path_graph(N)  # Create a graph

initial_pose = np.random.rand(N, 2) * 10 - 5

Adj = nx.adjacency_matrix(G).toarray()

targets = np.random.rand(N, 2) * 10 - 5


def generate_launch_description():
    node_list = []

    for i in range(N):
        N_ii = list(G.neighbors(i))
        N_ii.append(i)
        print("initial_pose", initial_pose[i])
        node = Node(
            package="surveillance",
            namespace=f"warden{i}",
            executable="warden",
            parameters=[
                {
                    "id": i,
                    "neighbors": int(N_ii),
                    "weights": list((Adj[i])),
                    "initial_pose": list((initial_pose[i])),
                    "target": list((targets[i])),
                },
            ],
            output="screen",
            # prefix=f'gnome-terminal -- bash -c "echo -ne \'\\033]0;Agent {i}\\007\'; $0; read -p \'Press any key to close this terminal...\'"'
            prefix=f'xterm -title "Agent {i}" -hold -e',
        )
        node_list.append(node)

    return LaunchDescription(node_list)
