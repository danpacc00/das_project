import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from launch import LaunchDescription
from launch_ros.actions import Node

N = 4  # Number of nodes (they represent the agents in this example)
G = nx.path_graph(N)  # Create a graph
# G = nx.cycle_graph(N) # Create a graph
G.add_edge(0, 3)  # Add an edge

initial_pose = np.random.rand(N, 2) * 10 - 5

Adj = nx.adjacency_matrix(G).toarray()

targets = np.random.rand(N, 2) * 10 - 5


def generate_launch_description():
    node_list = []

    for i in range(N):
        N_ii = list(G.neighbors(i))

        node = Node(
            package="surveillance",
            namespace=f"warden{i}",
            executable="warden",
            parameters=[
                {
                    "id": i,
                    "neighbors": {j: Adj[i, j] for j in N_ii + [i]},
                    "initial_pose": initial_pose[i],
                    "target": targets[i],
                },
            ],
            output="screen",
            # prefix=f'gnome-terminal -- bash -c "echo -ne \'\\033]0;Agent {i}\\007\'; $0; read -p \'Press any key to close this terminal...\'"'
            prefix=f'xterm -title "Agent {i}" -hold -e',
        )
        node_list.append(node)

    return LaunchDescription(node_list)
