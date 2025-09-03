from utils import load_from_pickle, save_as_pickle



# graph = load_from_pickle(f'../snap_dataset/test/Skitter')

# print(f'Number of nodes: {graph.number_of_nodes()}')
# print(f'Number of edges: {graph.number_of_edges()}')

import os
import networkx as nx
import pickle
import numpy as np

# train_graph = load_from_pickle(f'../snap_dataset/train/{"HK"}')

# print(f'Number of nodes: {train_graph.number_of_nodes()}')

# ==== CONFIG ====
dataset_name = "ER"   # name of saved file
save_root = "../snap_dataset"
splits = [
    "train",
    "val", 
    "test"
           ]


# Parameters for Holme–Kim graph
# n = 10000     # ~1.7 million nodes

n = {"train": 100000, "val": 100000, "test": 2000000}  # ~1.7 million nodes
# m = 3           # edges each new node attaches to
p = {"train": 0.00001, "val": 0.00001, "test": 0.000001}           # probability of adding a triangle after adding a random edge

for split in splits:
    os.makedirs(os.path.join(save_root, split), exist_ok=True)


for split in splits:
    print(f"Generating {split} graph with {n[split]:,} nodes...")
    # G = nx.powerlaw_cluster_graph(n=n[split], m=m, p=p)  # Holme–Kim

    # G = nx.erdos_renyi_graph(n=n[split], p=p[split])  # ER
    G = nx.fast_gnp_random_graph(n=n[split], p=p[split])  # ER

    

    save_as_pickle(G, os.path.join(save_root, split, dataset_name))