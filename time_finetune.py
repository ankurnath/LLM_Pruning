"""
Timing wrapper for finetune.py — runs Maximum Coverage finetuning for all datasets
without saving the model, and reports wall-clock time per dataset.
"""

import os, sys, time
import numpy as np
import torch
from torch_geometric.utils import from_networkx

sys.path.insert(0, os.path.dirname(__file__))

from utils import load_from_pickle, train_test_split, relabel_graph
from max_cover import greedy_max_cover
from utils import assign_normalized_degree_weights

# GCN must be importable from utils (same as finetune.py)
try:
    from utils import GCN
except ImportError:
    from gnn import GCN

DATA_DIR   = "/home/grads/a/anath/snap_dataset/train"
PROBLEM    = "Maximum Coverage"
BASE_DS    = "HK"
BUDGET     = 100
EPOCHS     = 10000

datasets = ["DBLP", "Deezer", "Facebook", "Skitter", "Slashdot", "Twitter", "Wiki", "YouTube"]

best_model_data = load_from_pickle(f"{PROBLEM}/{BASE_DS}/best_model_data.pkl")
print(f"Loaded best_model_data from {PROBLEM}/{BASE_DS}  features: {list(best_model_data['codes'].keys())}\n")

results = {}

for dataset in datasets:
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset}")
    print(f"{'='*60}")

    t0 = time.time()

    train_graph = load_from_pickle(os.path.join(DATA_DIR, dataset))
    print(f"  nodes={train_graph.number_of_nodes()}  edges={train_graph.number_of_edges()}")

    train_graph, val_graph = train_test_split(
        graph=train_graph, ratio=0.8, edge_level_split=True, seed=42
    )
    train_graph, _, _ = relabel_graph(train_graph)
    val_graph,   _, _ = relabel_graph(val_graph)

    # Build features
    train_X, val_X = [], []
    for feat, code in best_model_data["codes"].items():
        ns = {}
        exec(code, ns)
        train_X.append(ns["extract_feature"](G=train_graph))
        val_X.append(ns["extract_feature"](G=val_graph))

    train_X = torch.tensor(np.array(train_X).T, dtype=torch.float)
    val_X   = torch.tensor(np.array(val_X).T,   dtype=torch.float)

    train_data = from_networkx(train_graph); train_data.x = train_X
    val_data   = from_networkx(val_graph);   val_data.x   = val_X

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model     = GCN(input_channels=train_data.x.shape[1], hidden_channels=16, out_channels=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    obj_val, _, solution = greedy_max_cover(train_graph, budget=BUDGET, ground_set=None)
    mapping = dict(zip(train_graph.nodes(), range(train_graph.number_of_nodes())))
    train_mask = torch.tensor([mapping[n] for n in solution], dtype=torch.long)
    y = torch.zeros(train_graph.number_of_nodes(), dtype=torch.long)
    for n in solution:
        y[mapping[n]] = 1
    train_data.y = y

    train_data = train_data.to(device)
    val_data   = val_data.to(device)
    model      = model.to(device)

    obj_val_full, _, _ = greedy_max_cover(val_graph, budget=BUDGET, ground_set=None)

    best_C = float("-inf")
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        mask = torch.cat([train_mask, torch.randint(0, train_mask.size(0), (train_mask.size(0),))], dim=0)
        out  = model(train_data.x, train_data.edge_index)
        loss = criterion(out[mask], train_data.y[mask])
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                y_pred  = torch.argmax(model(val_data.x, val_data.edge_index), dim=1).cpu().numpy()
                indices = np.where(y_pred == 1)[0]
                if indices.size == 0:
                    continue
                obj_pruned, _, _ = greedy_max_cover(val_graph, budget=BUDGET, ground_set=indices)
                ratio          = obj_pruned / obj_val_full if obj_val_full else 0
                size_reduction = 1 - len(indices) / train_graph.number_of_nodes()
                C = ratio * size_reduction
                if C > best_C:
                    best_C = C
                    # *** model NOT saved (as requested) ***

    elapsed = time.time() - t0
    results[dataset] = elapsed
    print(f"  Done — best_C={best_C:.4f}  time={elapsed:.1f}s")

# ── Summary table ────────────────────────────────────────────
print("\n\n" + "="*45)
print(f"{'Dataset':<15} {'Time (s)':>10} {'Time (min)':>12}")
print("-"*45)
for ds, t in sorted(results.items(), key=lambda x: x[1]):
    print(f"{ds:<15} {t:>10.1f} {t/60:>12.2f}")
print("="*45)
