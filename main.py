from utils import *
from max_cover import *



def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--problem", type=str, default="Maximum Coverage Weighted", help="Problem name")
    parser.add_argument("--budget", type=int, default=100, help="Budget for the problem")
    parser.add_argument("--dataset", type=str, default= 'Facebook', help="Dataset to use")
    parser.add_argument("--iterations", type=int, default=10, help="Number of feature search iterations")

    args = parser.parse_args()

    problem = args.problem
    budget = args.budget
    dataset = args.dataset
    iterations = args.iterations

    if problem == "Maximum Coverage":
        heuristic = greedy_max_cover
    elif problem == "Maximum Coverage Weighted":
        heuristic = knapsack_greedy_max_cover
    else:
        raise ValueError(f"Unknown problem: {problem}")


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    results = []


    train_graph = load_from_pickle(f'../snap_dataset/train/{dataset}')

    train_graph,val_graph = train_test_split(
        graph=train_graph, 
        ratio=0.8, 
        edge_level_split=True, 
        seed=42)
    
    train_graph, train_forward_mapping, train_reverse_mapping = relabel_graph(train_graph)
    train_graph = assign_normalized_degree_weights(train_graph)

    val_graph, val_forward_mapping, val_reverse_mapping = relabel_graph(val_graph)
    val_graph = assign_normalized_degree_weights(val_graph)

    test_graph = load_from_pickle(f'../snap_dataset/test/{dataset}')
    test_graph, _, _ = relabel_graph(test_graph)
    test_graph = assign_normalized_degree_weights(test_graph)

    explainer_feedback_list = []
    history = []
    best_score = float('-inf')

    save_folder = f"{problem}/{dataset}"
    os.makedirs(save_folder, exist_ok=True)

    model_save_path = os.path.join(save_folder, "best_model.pth")
    best_model_data_path = os.path.join(save_folder, "best_model_data.pkl")
    history_path = os.path.join(save_folder, "history.pkl")

    # best_model_data = {
    #     "iteration": None,
    #     "codes": None,
    #     "ratio": None,
    #     "size_reduction": None,
    #     "multiplication": None,
    # }

    for iter in tqdm(range(iterations)):
        print(f"Feature Space Search Iteration {iter+1} for problem: {problem}")

        cumulative_feedback = "\n".join(explainer_feedback_list) if explainer_feedback_list else None

        if cumulative_feedback:

            summary_prompt = generate_summary_prompt(cumulative_feedback= cumulative_feedback)
            summary = get_response(client, summary_prompt)
        else:
            summary = None       
        node_feature_prompt = generate_llm_prompt(
            problem = problem,
            problem_definition = problem_definitions[problem],
            explainer_feedback = summary
        )
        
        proposed_features = get_response(client,node_feature_prompt)

        try:
            features, definitions, reasons = parse_llm_features(proposed_features)
        except Exception as e:
            print(f"Error parsing LLM features: {e}")
            continue

        try:
            train_features, codes = generate_train_features(features, definitions, train_graph, test_graph, timeout= 5)
        except Exception as e:
            print(f"Error generating train features: {e}")
            continue

        if len(codes) == 0:
            print("No valid features generated. Skipping iteration.")
            continue
        
        model, ratio, size_reduction, explainer_feedback_new = train_test_evaluate_gnn(
            train_features = train_features, 
            train_graph = train_graph,
            test_graph = val_graph,
            codes = codes,
            heuristic = heuristic,
            budget = budget
        )

        multiplication = ratio * size_reduction

        if multiplication > best_score:
            best_score = multiplication
            best_model_data = {
                "iteration": iter + 1,
                "codes": codes,
                "ratio": ratio,
                "size_reduction": size_reduction,
                "multiplication": multiplication,
            }
            with open(best_model_data_path, "wb") as f:
                pickle.dump(best_model_data, f)
            torch.save(model.state_dict(), model_save_path)

        history.append({
            "iteration": iter + 1,
            "features": list(codes.keys()),
            "ratio": ratio,
            "size_reduction": size_reduction,
            "explainer_feedback": explainer_feedback_new
        })
        
        explainer_feedback_list.append(
            f"Iteration {iter+1} | Ratio: {ratio}, Size reduction: {size_reduction}, "
            f"Feature importance: {explainer_feedback_new}"
        )

    

    with open(history_path, "wb") as f:
        pickle.dump(history, f)

    print(f"Best model, data, and history saved in '{save_folder}'.")

    

    

    start = time.time()
    test_X = []
    best_model_data = load_from_pickle(best_model_data_path)

    print(f"Best model data: {best_model_data}")
    for feature in best_model_data['codes']:
        namespace = {}
        exec(best_model_data['codes'][feature], namespace)
        feature_values = namespace["extract_feature"](G=test_graph, budget=budget)
        test_X.append(feature_values)
    time_taken_to_calculate = time.time() - start

    test_X = torch.tensor(np.array(test_X).T, dtype=torch.float)

    test_data = from_networkx(test_graph)
    test_data.x = test_X
    test_data = test_data.to(device)
    model     = GCN(input_channels= test_data.x.shape[1] ,hidden_channels = 16, out_channels = 2).to(device)
    model.load_state_dict(torch.load(model_save_path))

    
    model = model.to(device)
    model.eval()

    y_pred = torch.argmax(model(test_data.x, test_data.edge_index), axis=1).cpu().numpy()
    indices = np.where(y_pred == 1)[0]

    start = time.time()
    obj_val, number_of_queries, solution = heuristic(test_graph, budget=budget)
    time_taken = time.time() - start

    start = time.time()
    obj_val_pruned, number_of_queries_pruned, solution_pruned = heuristic(
        test_graph, budget=budget, ground_set=indices
    )
    time_taken_pruned = time.time() - start

    ratio = obj_val_pruned / obj_val if obj_val != 0 else 0
    size_reduction = 1 - len(indices) / test_graph.number_of_nodes()
    time_ratio = time_taken / time_taken_pruned if time_taken_pruned != 0 else 0

    results.append({
        "Dataset": dataset,
        "Obj Value": obj_val,
        "Obj Value Pruned": obj_val_pruned,
        "Queries": number_of_queries,
        "Queries Pruned": number_of_queries_pruned,
        "Ratio": ratio,
        "Size Reduction": size_reduction,
        "Time Ratio": time_ratio,
        "Feature Extraction Time": time_taken_to_calculate,
        "Time Baseline": time_taken,
        "Time Pruned": time_taken_pruned,
        "Num Selected Nodes": len(indices)
    })

    # ------------------------------
    # Save results after all datasets
    # ------------------------------

    results = pd.DataFrame(results)
    results_folder = f"{problem}/{dataset}"
    os.makedirs(results_folder, exist_ok=True)

    results_path_pkl = os.path.join(results_folder, "results.pkl")


    with open(results_path_pkl, "wb") as f:
        pickle.dump(results, f)



    print(f"\n✅ All results saved in:\n- {results_path_pkl}")

    print(results)


if __name__ == "__main__":
    main()






