from utils import load_from_pickle, save_as_pickle



# graph = load_from_pickle(f'../snap_dataset/test/Skitter')

# print(f'Number of nodes: {graph.number_of_nodes()}')
# print(f'Number of edges: {graph.number_of_edges()}')

import os
import networkx as nx
import pickle
import numpy as np

# ==== CONFIG ====
dataset_name = "HK"   # name of saved file
save_root = "../snap_dataset"
splits = [
    # "train",
    "val", 
        #    "test"
           ]


# Parameters for Holme–Kim graph
n = 2000000     # ~1.7 million nodes
m = 3           # edges each new node attaches to
p = 0.4         # probability of triad formation

for split in splits:
    os.makedirs(os.path.join(save_root, split), exist_ok=True)


for split in splits:
    print(f"Generating {split} graph with {n:,} nodes...")
    G = nx.powerlaw_cluster_graph(n=n, m=m, p=p)  # Holme–Kim

    

    # save_as_pickle(G, os.path.join(save_root, split, dataset_name))