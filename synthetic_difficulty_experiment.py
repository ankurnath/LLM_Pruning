"""
Synthetic Difficulty Experiment for Maximum Coverage (LLM2Prune rebuttal).

Three difficulty axes:
  1. Constraint tightness: Fixed HK graph (n=10,000, m=3), vary budget k
  2. Graph density:        Fixed n=10,000, vary HK parameter m
  3. Modularity:           SBM graphs, fixed n=10,000, vary inter-community prob q

Uses the pre-trained HK model from Maximum Coverage/HK/
"""

import os
import time
import numpy as np
import networkx as nx
import torch
import pandas as pd
import matplotlib.pyplot as plt

from utils import load_from_pickle, save_as_pickle, relabel_graph, GCN
from torch_geometric.utils.convert import from_networkx
from max_cover import greedy_max_cover

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_DIR        = "Maximum Coverage/HK"
DATA_TEST_DIR    = "data/test"
RESULTS_DIR      = "results/synthetic_difficulty"
BEST_K_CANDIDATES = [500, 1000, 2000, 5000]

N_VALUES   = [10_000, 50_000, 100_000]   # graph sizes to sweep
P_TRI      = 0.1      # triangle-closure probability for HK
BUDGET_FIX = 100      # fixed budget for density & modularity experiments

os.makedirs(DATA_TEST_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR,   exist_ok=True)

# ---------------------------------------------------------------------------
# Graph generators
# ---------------------------------------------------------------------------

def make_hk_graph(n, m, p_tri=P_TRI, seed=42):
    return nx.powerlaw_cluster_graph(n=n, m=m, p=p_tri, seed=seed)


# ---------------------------------------------------------------------------
# Evaluation (mirrors test.py)
# ---------------------------------------------------------------------------

def evaluate(graph, budget, model_data, model, device):
    graph, _, _ = relabel_graph(graph)
    n_nodes = graph.number_of_nodes()

    # Feature extraction
    t0 = time.time()
    features = []
    for code in model_data["codes"].values():
        ns = {}
        exec(code, ns)
        features.append(ns["extract_feature"](G=graph))
    feat_time = time.time() - t0

    X    = torch.tensor(np.array(features).T, dtype=torch.float)
    data = from_networkx(graph)
    data.x = X
    data  = data.to(device)

    # GNN inference
    t0 = time.time()
    with torch.no_grad():
        probs = torch.softmax(model(data.x, data.edge_index), dim=1)[:, 1]
    select_time = time.time() - t0

    # Pick best candidate set size (same logic as test.py)
    topk_vals, topk_idx = torch.topk(probs, min(max(BEST_K_CANDIDATES), n_nodes))
    best_k   = BEST_K_CANDIDATES[0]
    best_idx = topk_idx[:best_k].cpu().tolist()
    best_obj, _, _ = greedy_max_cover(graph, budget=budget, ground_set=best_idx)

    for k in BEST_K_CANDIDATES[1:]:
        if k > n_nodes:
            break
        idx = topk_idx[:k].cpu().tolist()
        obj, _, _ = greedy_max_cover(graph, budget=budget, ground_set=idx)
        if best_obj == 0 or (obj - best_obj) / best_obj > 0.01:
            best_obj, best_k, best_idx = obj, k, idx

    # Pruned heuristic
    t0 = time.time()
    obj_pruned, _, _ = greedy_max_cover(graph, budget=budget, ground_set=best_idx)
    time_pruned = time.time() - t0

    # Full heuristic (no pruning)
    t0 = time.time()
    obj_full, _, _  = greedy_max_cover(graph, budget=budget)
    time_full = time.time() - t0

    ratio    = obj_pruned / obj_full if obj_full > 0 else 0.0
    size_red = 1 - len(best_idx) / n_nodes

    return {
        "Ratio":          ratio,
        "Size Reduction": size_red,
        "C":              ratio * size_red,
        "Time Full":      time_full,
        "Time Pruned":    time_pruned,
        "Time Ratio":     time_full / time_pruned if time_pruned > 0 else 0,
        "Inference Time": feat_time + select_time,
        "Obj Full":       obj_full,
        "Obj Pruned":     obj_pruned,
        "Best K":         best_k,
        "N Nodes":        n_nodes,
    }

# ---------------------------------------------------------------------------
# Plotting — one line per n value
# ---------------------------------------------------------------------------

def plot_multi_n(df, x_col, x_label, title, filename):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    metrics = [("C", "C (Combined Metric)"),
               ("Ratio", "P_r (Quality Ratio)"),
               ("Size Reduction", "P_g (Size Reduction)")]
    colors  = ["tab:blue", "tab:orange", "tab:green"]

    for ax, (col, ylabel) in zip(axes, metrics):
        for n_val, color in zip(N_VALUES, colors):
            sub = df[df["n"] == n_val].sort_values(x_col)
            ax.plot(sub[x_col], sub[col], marker="o", linewidth=2,
                    label=f"n={n_val:,}", color=color)
        ax.set_xlabel(x_label)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs {x_label}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Effect of {title} on Pruning Performance (Maximum Coverage)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"{filename}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_data = load_from_pickle(f"{MODEL_DIR}/best_model_data.pkl")
    n_features = len(model_data["codes"])
    model = GCN(input_channels=n_features, hidden_channels=16, out_channels=2).to(device)
    model.load_state_dict(torch.load(f"{MODEL_DIR}/best_model.pth", map_location=device))
    model.eval()
    print(f"Model loaded  features: {list(model_data['codes'].keys())}\n")

    budgets  = [10, 25, 50, 75, 100, 150, 200]
    m_values = [2, 4, 6, 8, 10]

    all_budget  = []
    all_density = []

    for N in N_VALUES:
        print(f"\n{'='*60}")
        print(f"  Graph size n = {N:,}")
        print(f"{'='*60}")

        # ------------------------------------------------------------------
        # Experiment 1: Budget (constraint tightness)
        # ------------------------------------------------------------------
        print(f"\n--- Experiment 1: Budget (n={N:,}) ---")
        graph_hk = make_hk_graph(n=N, m=3)
        save_as_pickle(graph_hk, os.path.join(DATA_TEST_DIR, f"synthetic_HK_n{N}_m3"))
        print(f"  HK graph: {graph_hk.number_of_nodes():,} nodes, {graph_hk.number_of_edges():,} edges")

        for k in budgets:
            print(f"  budget={k:3d} ...", end=" ", flush=True)
            res = evaluate(graph_hk, budget=k, model_data=model_data, model=model, device=device)
            res.update(n=N, budget=k, budget_ratio=k/N)
            all_budget.append(res)
            print(f"C={res['C']:.4f}  P_r={res['Ratio']:.4f}  P_g={res['Size Reduction']:.4f}  inference={res['Inference Time']*1000:.2f}ms")

        # ------------------------------------------------------------------
        # Experiment 2: Graph density (HK parameter m)
        # ------------------------------------------------------------------
        print(f"\n--- Experiment 2: Density (n={N:,}) ---")
        for m in m_values:
            G = make_hk_graph(n=N, m=m)
            save_as_pickle(G, os.path.join(DATA_TEST_DIR, f"synthetic_HK_n{N}_m{m}"))
            avg_deg = 2 * G.number_of_edges() / G.number_of_nodes()
            print(f"  m={m}  avg_deg={avg_deg:.1f} ...", end=" ", flush=True)
            res = evaluate(G, budget=BUDGET_FIX, model_data=model_data, model=model, device=device)
            res.update(n=N, m=m, avg_degree=avg_deg)
            all_density.append(res)
            print(f"C={res['C']:.4f}  P_r={res['Ratio']:.4f}  P_g={res['Size Reduction']:.4f}  inference={res['Inference Time']*1000:.2f}ms")

    # ------------------------------------------------------------------
    # Save & plot — one curve per n value
    # ------------------------------------------------------------------
    df_budget  = pd.DataFrame(all_budget)
    df_density = pd.DataFrame(all_density)

    for df, fname in [(df_budget, "budget_results"), (df_density, "density_results")]:
        save_as_pickle(df, os.path.join(RESULTS_DIR, f"{fname}.pkl"))
        df.to_csv(os.path.join(RESULTS_DIR, f"{fname}.csv"), index=False)

    plot_multi_n(df_budget,  "budget",     "Budget k",       "Constraint Tightness", "budget_experiment")
    plot_multi_n(df_density, "avg_degree", "Average Degree", "Graph Density",        "density_experiment")

    print("\nAll done. Results in", RESULTS_DIR)


if __name__ == "__main__":
    main()
