from launch import LaunchDescription
from launch_ros.actions import Node
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

N = 4 # Number of nodes (they represent the agents in this example)
G = nx.path_graph(N) # Create a graph
# G = nx.cycle_graph(N) # Create a graph
G.add_edge(0,3) # Add an edge
xzero = [10,2,3,5]

def generate_launch_description():

    node_list = []

    for i in range(N):
        N_ii = list(G.neighbors(i))

        node = Node(
            package='distributed_algs',
            namespace=f"agent{i}",
            executable='generic_agent', #generic_agent is inside setup.py
            #The differences between the nodes reside in the parameters
            parameters=[
                {
                    "id":i,
                    "Nii":N_ii,
                    "x_0":xzero[i],
                },
                
            ],
            output='screen',
            # prefix=f'gnome-terminal -- bash -c "echo -ne \'\\033]0;Agent {i}\\007\'; $0; read -p \'Press any key to close this terminal...\'"'
            prefix=f'xterm -title "Agent {i}" -hold -e'
        )
        node_list.append(node)


    return LaunchDescription(node_list)
