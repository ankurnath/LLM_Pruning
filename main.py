import argparse
import os
import pickle
import torch
from tqdm import tqdm
import numpy as np

from utils import *
from max_cover import *
from max_cut import *
from max_cut_weighted import *
from imm import *
from knapsack_im import knapsack_im_greedy
from heuristic_description import heuristic_description
# from maximum_cut_negative_greedy import assign_random_edge_weights, standard_greedy


# -----------------------------
# Node definition
# -----------------------------
class Node:
    def __init__(self, features=None, codes=None, parent=None,
                 score=0.0, ratio=0.0, size_reduction=0.0,
                 feedback=None, history=None, summary=None, model=None):
        self.features = features or []        
        self.codes = codes or {}              
        self.parent = parent                  
        self.children = []                    

        # Evaluation metrics
        self.score = score
        self.ratio = ratio
        self.size_reduction = size_reduction
        self.model = model

        # Feedback & context
        self.feedback = feedback              # feedback from this node
        self.history = history or []          # all feedbacks along the path
        self.summary = summary                # compressed version of history

    def add_child(self, child_node):
        self.children.append(child_node)

    def __repr__(self):
        return f"Node(features={self.features}, score={self.score:.4f})"


# -----------------------------
# Beam Search Feature Exploration
# -----------------------------
def beam_search_feature_generation(problem, budget, dataset, depth, beam_size, expansion_factor):

    heuristic_map = {
        "Maximum Coverage": greedy_max_cover,
        "Maximum Coverage Weighted": knapsack_greedy_max_cover,
        "Influence Maximization": imm,
        "Influence Maximization Weighted": knapsack_im_greedy,
        "Maximum Cut": maxcut_greedy,
        "Maximum Cut Weighted": DLA,
        # "Maximum Cut Negative": standard_greedy
    }

    if problem not in heuristic_map:
        raise ValueError(f"Unknown problem: {problem}")
    heuristic = heuristic_map[problem]

    # -----------------------------
    # Load graphs
    # -----------------------------
    train_graph = load_from_pickle(f'data/train/{dataset}')
    val_graph   = load_from_pickle(f'data/val/{dataset}')
    test_graph  = load_from_pickle(f'data/test/{dataset}')

    if problem in ["Maximum Coverage Weighted", "Influence Maximization Weighted", "Maximum Cut Weighted"]:
        train_graph = assign_normalized_degree_weights(train_graph)
        val_graph   = assign_normalized_degree_weights(val_graph)
        test_graph  = assign_normalized_degree_weights(test_graph)

    # if problem == 'Maximum Cut Negative':
    #     train_graph = assign_random_edge_weights(train_graph)
    #     val_graph   = assign_random_edge_weights(val_graph)
    #     test_graph  = assign_random_edge_weights(test_graph)

    # -----------------------------
    # Beam Search Setup
    # -----------------------------
    save_folder = f"{problem}/{dataset}"
    os.makedirs(save_folder, exist_ok=True)

    model_save_path = os.path.join(save_folder, "best_model.pth")
    best_model_data_path = os.path.join(save_folder, "best_model_data.pkl")
    history_path = os.path.join(save_folder, "history.pkl")

    root = Node(features=[], codes={}, score=0.0, history=[], summary=None)
    beam = [root]
    best_node = root
    best_score = float('-inf')
    history_records = []

    # -----------------------------
    # Iterative Beam Search
    # -----------------------------
    for depth in tqdm(range(depth)):
        print(f"\nBeam Search Depth {depth+1}")

        candidates = []

        for parent in beam:

            # 1. Extend history
            cumulative_history = parent.history + [parent.feedback] if parent.feedback else parent.history

            # 2. Summarize history for LLM
            if cumulative_history:
                summary_prompt = generate_summary_prompt(cumulative_feedback="\n".join(cumulative_history))
                summary = get_response(client, summary_prompt)
            else:
                summary = None

            # 3. Generate candidate features with LLM
            node_feature_prompt = generate_llm_prompt(
                problem=problem,
                problem_definition=problem_definitions[problem],
                explainer_feedback=summary
            )
            for _ in range(expansion_factor):   # 🔑 expand multiple children per node
                
                proposed_features = get_response(client, node_feature_prompt)

                try:
                    features, definitions, reasons = parse_llm_features(proposed_features)
                    print(f"Proposed features: {features}")
                except Exception as e:
                    print(f"Parse error: {e}")
                    continue

                # 4. Generate train features
                try:
                    train_features, codes, falied_features = generate_train_features(
                        problem=problem,
                        features=features,
                        definitions=definitions,
                        train_graph=train_graph,
                        test_graph=test_graph,
                        budget=budget,
                        timeout=5
                    )
                except Exception as e:
                    print(f"Feature gen error: {e}")
                    continue

                if not codes:
                    continue

                # 5. Evaluate with GNN
                model, ratio, size_reduction, feedback = train_test_evaluate_gnn(
                    problem= problem,
                    dataset= dataset,
                    train_features=train_features,
                    train_graph=train_graph,
                    val_graph=val_graph,
                    codes=codes,
                    heuristic=heuristic,
                    budget=budget
                )

                feedback += falied_features

                score = ratio * size_reduction

                # 6. Create child node
                child_node = Node(
                    features=list(codes.keys()),
                    codes=codes,
                    parent=parent,
                    score=score,
                    ratio=ratio,
                    size_reduction=size_reduction,
                    feedback=feedback,
                    history=cumulative_history,
                    summary=summary,
                    model=model
                )
                parent.add_child(child_node)
                candidates.append(child_node)

        # 🔑 Keep top-β candidates
        candidates.sort(key=lambda n: n.score, reverse=True)
        beam = candidates[:beam_size]

        # Track best
        if beam and beam[0].score > best_score:
            best_score = beam[0].score
            best_node = beam[0]

            best_model_data = {
                "features": best_node.features,
                "codes": best_node.codes,
                "ratio": best_node.ratio,
                "size_reduction": best_node.size_reduction,
                "score": best_node.score,
                "iteration": depth+1
            }
            with open(best_model_data_path, "wb") as f:
                pickle.dump(best_model_data, f)
            torch.save(best_node.model.state_dict(), model_save_path)

        # Record history
        for cand in beam:
            history_records.append({
                "iteration": depth+1,
                "features": cand.features,
                "ratio": cand.ratio,
                "size_reduction": cand.size_reduction,
                "score": cand.score,
                "feedback": cand.feedback,
                "summary": cand.summary
            })

    # Save history
    with open(history_path, "wb") as f:
        pickle.dump(history_records, f)

    print(f"\n✅ Beam search finished. Best score = {best_score:.4f}")
    print(f"Best model, data, and history saved in '{save_folder}'.")


# -----------------------------
# Entry Point
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, default="Influence Maximization Weighted", help="Problem name")
    parser.add_argument("--budget", type=int, default=100, help="Budget for the problem")
    parser.add_argument("--dataset", type=str, default='HK', help="Dataset to use")
    parser.add_argument("--depth", type=int, default=3, help="Number of feature search depth")
    parser.add_argument("--beam_size", type=int, default=3, help="Beam size for pruning")
    parser.add_argument("--expansion_factor", type=int, default=2, help="Number of children expanded per node")

    args = parser.parse_args()

    beam_search_feature_generation(
        problem=args.problem,
        budget=args.budget,
        dataset=args.dataset,
        depth=args.depth,
        beam_size=args.beam_size,
        expansion_factor=args.expansion_factor
    )


if __name__ == "__main__":
    main()






