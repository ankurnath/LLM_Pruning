from utils import load_from_pickle,save_as_pickle
from max_cover import *
from max_cut import *
from max_cut_weighted import *
from imm import *
# from knapsack_imm import knapsack_greedy
from knapsack_im import knapsack_im_greedy
# from heuristic_description import heuristic_description


import torch.nn.functional as F


import torch
import torch.nn.functional as F




def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--problem", type=str, default="Maximum Cut", help="Problem name")
    parser.add_argument("--budget", type=int, default=100, help="Budget for the problem")
    parser.add_argument('--base_dataset',type=str, default= 'HK', help="train Dataset to use")
    parser.add_argument("--train_dataset", type=str, default= 'HK', help="train Dataset to use")
    parser.add_argument("--test_dataset", type=str, default= 'HK', help="test Dataset to use")
    parser.add_argument("--k", type=int, default=1000, help="Size of pruned set to use")
    

    args = parser.parse_args()

    problem = args.problem
    budget = args.budget
    train_dataset = args.train_dataset
    test_dataset = args.test_dataset
    k = args.k

    heuristics = {
        "Maximum Coverage": greedy_max_cover,
        "Maximum Coverage Weighted": knapsack_greedy_max_cover,
        "Influence Maximization": imm,
        "Influence Maximization Weighted": knapsack_im_greedy,
        "Maximum Cut": maxcut_greedy,
        "Maximum Cut Weighted": DLA,
    }

    try:
        heuristic = heuristics[problem]
    except KeyError:
        raise ValueError(f"Unknown problem: {problem}")
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    results = []

    test_graph = load_from_pickle(f'../snap_dataset/test/{test_dataset}')

    print(f'Loaded test graph with {test_graph.number_of_nodes()} nodes and {test_graph.number_of_edges()} edges.')

    test_graph,_,_ = relabel_graph(test_graph)




    

    if problem in ["Maximum Coverage Weighted", "Influence Maximization Weighted", "Maximum Cut Weighted"]:
        test_graph = assign_normalized_degree_weights(test_graph)
        weight = np.array([test_graph.nodes[v].get("weight", 1.0) for v in test_graph.nodes], dtype=np.float32)
        mask = torch.tensor(weight <= budget, dtype=torch.bool, device=device)



    best_model_data_path = os.path.join(f"simple_feedback/{problem}/{args.base_dataset}", "best_model_data.pkl")

    start = time.time()
    test_X = []
    best_model_data = load_from_pickle(best_model_data_path)
    # print(f"Best model data: {best_model_data}")

    print(f"Codes for feature extraction: {best_model_data['codes'].keys()}")
    for feature in best_model_data['codes']:
        namespace = {}
        exec(best_model_data['codes'][feature], namespace)
        feature_values = namespace["extract_feature"](G=test_graph)
        test_X.append(feature_values)
    time_taken_to_calculate = time.time() - start

    test_X = torch.tensor(np.array(test_X).T, dtype=torch.float)

    test_data = from_networkx(test_graph)

    
    test_data.x = test_X
    test_data = test_data.to(device)
    model     = GCN(input_channels= test_data.x.shape[1] ,hidden_channels = 16, out_channels = 2).to(device)

    model_save_path = os.path.join(f"simple_feedback/{problem}/{train_dataset}", "best_model.pth")
    model.load_state_dict(torch.load(model_save_path))

    
    model = model.to(device)
    model.eval()

    # y_pred = torch.argmax(model(test_data.x, test_data.edge_index), axis=1).cpu().numpy()
    # indices = np.where(y_pred == 1)[0]

    start = time.time()

    with torch.no_grad():
        logits = model(test_data.x, test_data.edge_index)

        
        probs = torch.softmax(logits, dim=1)[:, 1]  # P(class=1)

    time_taken_to_select = time.time() - start

    # print(f"Predicted probabilities for {test_graph.number_of_nodes()} nodes.",probs)

    # pick top-k nodes

    if problem.endswith("Weighted"):

        probs *= mask
        pass

    best_k = 500
    topk_vals, topk_idx = torch.topk(probs, min(5000,test_graph.number_of_nodes()))  # precompute once
    best_idx = topk_idx[:best_k].cpu().tolist()

    best_obj_val, _, best_solution = heuristic(test_graph, budget=budget, ground_set=best_idx)
    print(f"Initial k={best_k}, obj_val={best_obj_val:.4f}")

    

    for k in [1000, 2000, min(5000,test_graph.number_of_nodes())]:
        idx = topk_idx[:k].cpu().tolist()
        obj_val, _, sol = heuristic(test_graph, budget=budget, ground_set=idx)

        print(f"k={k}, obj_val={obj_val:.4f}, best_obj_val={best_obj_val:.4f}")

        if (obj_val - best_obj_val) / best_obj_val > 0.01:
            best_obj_val, best_solution, best_k, best_idx = obj_val, sol, k, idx
        else:
            break

    print(f"Final k={best_k}, obj_val={best_obj_val:.4f}")

    # topk_vals, topk_idx = torch.topk(probs, k)
    # indices = topk_idx.cpu().tolist()

    # print(f"Selected {len(indices)} nodes out of {test_graph.number_of_nodes()} nodes.")

    # time_taken_to_select = time.time() - start


    

    # print(solution)

    start = time.time()
    obj_val_pruned, number_of_queries_pruned, solution_pruned = heuristic(
        test_graph, budget=budget, ground_set=best_idx
    )

    # print(solution_pruned)
    time_taken_pruned = time.time() - start


    start = time.time()

    # save_folder = f"{problem}/{test_dataset}/results"
    obj_val, number_of_queries, solution = heuristic(test_graph, budget=budget)
    time_taken = time.time() - start

    ratio = obj_val_pruned / obj_val if obj_val != 0 else 0
    size_reduction = 1 - len(best_idx) / test_graph.number_of_nodes()
    time_ratio = time_taken / time_taken_pruned if time_taken_pruned != 0 else 0

    print(f"Dataset: {test_dataset}")
    print(f"ratio: {ratio} Size Reduction: {size_reduction} Time Ratio: {time_ratio}")

    results.append({
        "Dataset": test_dataset,
        "Obj Value(Unpruned)": obj_val,
        "Obj Value(Pruned)": obj_val_pruned,
        # "Queries": number_of_queries,
        # "Queries Pruned": number_of_queries_pruned,
        "Ratio": ratio,
        "Size Reduction": size_reduction,
        "CombinedMetric": ratio * size_reduction,
        "TimeRatio": time_ratio,
        "Feature Extraction Time": time_taken_to_calculate,
        'TimeToPrune': time_taken_to_select,
        "Time(Unpruned)": time_taken,
        "Time(Pruned)": time_taken_pruned,
        "Num Selected Nodes": len(best_idx)
    })

    # ------------------------------
    # Save results after all datasets
    # ------------------------------

    results = pd.DataFrame(results)

    print(results)
    results_folder = f"simple_feedback/{problem}/{train_dataset}/results"
    os.makedirs(results_folder, exist_ok=True)

    results_path_pkl = os.path.join(results_folder, f"{test_dataset}.pkl")
    save_as_pickle(results, results_path_pkl)


if __name__ == "__main__":
    main()










