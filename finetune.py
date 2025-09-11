from utils import load_from_pickle,save_as_pickle,train_test_split
from max_cover import *
from max_cut import *
from max_cut_weighted import *
from imm import *
from knapsack_imm import knapsack_greedy
# from heuristic_description import heuristic_description


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--problem", type=str, default="Maximum Coverage", help="Problem name")
    parser.add_argument("--budget", type=int, default=100, help="Budget for the problem")
    parser.add_argument("--base_dataset",type=str, default= 'HK', help="Finetune the model which is intially trained for this")
    parser.add_argument("--dataset", type=str, default= 'Facebook', help="Finetune on this Dataset to use")
    
    

    args = parser.parse_args()

    problem = args.problem
    budget = args.budget
    base_dataset  = args.base_dataset
    dataset = args.dataset

    heuristics = {
    "Maximum Coverage": greedy_max_cover,
    "Maximum Coverage Weighted": knapsack_greedy_max_cover,
    "Influence Maximization": imm,
    "Influence Maximization Weighted": knapsack_greedy,
    "Maximum Cut": maxcut_greedy,
    "Maximum Cut Weighted": DLA,
    }

    try:
        heuristic = heuristics[problem]
    except KeyError:
        raise ValueError(f"Unknown problem: {problem}")
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    

    train_graph = load_from_pickle(f'../snap_dataset/train/{dataset}')

    print(f'Loaded test graph with {train_graph.number_of_nodes()} nodes and {train_graph.number_of_edges()} edges.')

    # train_graph,_,_ = relabel_graph(train_graph)


    train_graph,val_graph = train_test_split(
        graph=train_graph, 
        ratio=0.8, 
        edge_level_split=True, 
        seed=42)
    print(f'Train graph has {train_graph.number_of_nodes()} nodes and {train_graph.number_of_edges()} edges.')
    print(f'Validation graph has {val_graph.number_of_nodes()} nodes and {val_graph.number_of_edges()} edges.')
    
    
    train_graph,_,_ = relabel_graph(train_graph)
    val_graph,_,_ = relabel_graph(val_graph)


    if problem in ["Maximum Coverage Weighted", "Influence Maximization Weighted", "Maximum Cut Weighted"]:
        train_graph = assign_normalized_degree_weights(train_graph)
        val_graph = assign_normalized_degree_weights(val_graph)
        

    best_model_data_path = os.path.join(f"{problem}/{base_dataset}", "best_model_data.pkl")

    start = time.time()
    train_X = []
    val_X = []
    
    best_model_data = load_from_pickle(best_model_data_path)
    # print(f"Best model data: {best_model_data}")

    print('Calculating features for training graph:',best_model_data["codes"].keys())
    for feature in best_model_data['codes']:
        namespace = {}
        exec(best_model_data['codes'][feature], namespace)
        feature_values = namespace["extract_feature"](G=train_graph)
        train_X.append(feature_values)

        feature_values = namespace["extract_feature"](G=val_graph)
        val_X.append(feature_values)
    time_taken_to_calculate = time.time() - start

    train_X = torch.tensor(np.array(train_X).T, dtype=torch.float)
    val_X = torch.tensor(np.array(val_X).T, dtype=torch.float)

    train_data = from_networkx(train_graph)
    val_data   = from_networkx(val_graph)
    
    train_data.x = train_X 
    val_data.x   = val_X
    
    model     = GCN(input_channels= train_data.x.shape[1] ,hidden_channels = 16, out_channels = 2).to(device)

    # model_save_path = os.path.join(f"{problem}/{base_dataset}", "best_model.pth")
    # model.load_state_dict(torch.load(model_save_path))

    
    

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    

    print('Training GNN')
    model.train()

    obj_val,number_of_queries,solution = heuristic(train_graph, budget=budget, ground_set=None)

    mapping = dict(zip(train_graph.nodes(), range(train_graph.number_of_nodes())))
    train_mask = torch.tensor([mapping[node] for node in solution], dtype=torch.long)
    y= torch.zeros(train_graph.number_of_nodes(),dtype=torch.long)

    for node in solution:
        y[mapping[node]]=1

    train_data.y = y

    model = model.to(device)
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    # for epoch in tqdm(range(1, 100000)):

    best_C = float('-inf')
    os.makedirs(f"{problem}/{dataset}",exist_ok=True)
    finetune_model_save_path = os.path.join(f"{problem}/{dataset}", "best_model.pth")


    


    # Uncomment below to train the model




    

    
    # for epoch in range(1, 10000):
    #     optimizer.zero_grad()
    #     mask = torch.cat([train_mask, torch.randint(0, train_mask.size(0), (train_mask.size(0),))], dim=0)
    #     out  = model(train_data.x, train_data.edge_index)        # [N, num_classes]
    #     loss = criterion(out[mask], train_data.y[mask]) 
    #     # loss = criterion(out, train_data.y)                # full-graph loss
    #     loss.backward()
    #     optimizer.step()

    #     if epoch % 100 == 0:
    #         # print(f'Epoch: {epoch:03d})')
                  
    #         obj_val,number_of_queries,solution = heuristic(val_graph, budget=budget, ground_set=None)

    #         y_pred = torch.argmax(model(val_data.x, val_data.edge_index), axis=1).cpu().numpy()
    #         indices = np.where(y_pred == 1)[0]
    #         if indices.size == 0:
    #             continue

    #         obj_val_pruned, number_of_queries_pruned, solution_pruned = heuristic(
    #             val_graph, budget=budget, ground_set=indices
    #         )
    #         time_taken_pruned = time.time() - start

    #         ratio = obj_val_pruned / obj_val if obj_val != 0 else 0
    #         size_reduction = 1 - len(indices) / train_graph.number_of_nodes()

    #         C = ratio * size_reduction

    #         if C > best_C:
    #             best_C = C
    #             torch.save(model.state_dict(), finetune_model_save_path)
                # print(f'New best model saved with C={best_C}, ratio={ratio}, size_reduction={size_reduction}')


    

    


   
    

if __name__ == "__main__":
    main()










