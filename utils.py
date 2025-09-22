import pickle
import json
import numpy as np 
import re
import networkx as nx
import signal
from tqdm import tqdm  # progress bar
import time
import torch
from torch_geometric.utils.convert import from_networkx
from torch_geometric.nn import GCNConv
from torch_geometric.explain import Explainer, GNNExplainer

from argparse import ArgumentParser
from utils import *
import networkx as nx
import numpy as np
import random
import os
import logging

import os
import pickle
import torch
import networkx as nx
import numpy as np
import time
import pandas as pd

import random
from openai import OpenAI
import argparse

from key import *

import time

import requests
import json

def save_as_pickle(obj, filename):
    """
    Save a Python object to a pickle file.

    Args:
        obj: The Python object to save.
        filename (str): Path where the pickle file will be stored.
    """
    with open(filename, "wb") as f:
        pickle.dump(obj, f)
    print(f"Data has been saved to {filename}")





# client = None
# def get_response(client, prompt):

#     url = 'http://127.0.0.1:11015/completions'

#     data = {
#         'prompt': prompt,
#         'repeat_prompt': 20,
#         'system_prompt': '',
#         'stream': False,
#         'params': {
#             'temperature': None,
#             'top_k': None,
#             'top_p': None,
#             'add_special_tokens': False,
#             'skip_special_tokens': True,
#         }
#     }

#     headers = {'Content-Type': 'application/json'}

#     response = requests.post(url, data=json.dumps(data), headers=headers)

#     print(f'time: {time.time()}s')

#     if response.status_code == 200:
        
#         # print(f'Response: {response.json()}')
#         content = response.json()["content"]

        
#     else:
#         print('Failed to make the POST request.')

    

#     # print(f'Query time: {durations}')

#     return content

# if __name__ == "__main__":
#     response = get_response(client, prompt="What is the capital of France?")
    # print(f"Response: {response}")

# response = get_response(client, prompt = "What is the capital of France?")



problem_definitions = {
    "Maximum Coverage": (
        "The problem is defined over a graph \( G = (V, E) \).\n"
        "Given a budget \( k \), the objective is to select a subset of nodes \( S \subseteq V \) such that:\n"
        "f(S) = |{v ∈ V | v ∈ S or ∃(u, v) ∈ E, u ∈ S}|, subject to |S| ≤ k.\n"
    ),

    "Maximum Coverage Weighted": (
        
        "The problem is defined over a graph is defined over an undirected graph "
        "G = (V, E), where each node v in V has an associated weight w(v) > 0. "
        "The weight of each node is w(v) = β/|V| * (|N(v)| - α), where N(v) is the set of neighbors of v, α = 1/20, "
        "and β is a normalizing factor ensuring c(v) ≥ 1 . "
        "Given a budget k, the goal is to select a subset S ⊆ V such that the sum of weights is at most k and "
        "the coverage function f(S) = |{v ∈ V | v ∈ S or ∃(u, v) ∈ E, u ∈ S}| is maximized."
    
    ),
    "Maximum Cut": (
        "The problem is defined over an undirected graph \( G = (V, E) \).\n"
        "The goal is to partition the vertex set \( V \) into two disjoint subsets \( S \) and \( V \\setminus S \)\n"
        "such that the number of edges crossing the cut, i.e., with one endpoint in \( S \) and the other in \( V \\setminus S \),\n"
        "is maximized.\n"
        "Formally, maximize the cut value:\n"
        "f(S) = |{(u, v) ∈ E : u ∈ S, v ∈ V \\setminus S}|.\n"
    ),

    "Maximum Cut Negative": (
    "The problem is defined over an undirected weighted graph \( G = (V, E, w) \), "
    "where each edge \( (u, v) \in E \) has an associated weight \( w_{uv} \in \{ -1, +1 \}. \n"
    "The goal is to partition the vertex set \( V \) into two disjoint subsets \( S \) and \( V \\setminus S \)\n"
    "such that the total weight of edges crossing the cut, i.e., edges with one endpoint in \( S \) and the other in \( V \\setminus S \),\n"
    "is maximized.\n"
    "Formally, maximize the cut value:\n"
    "f(S) = \\sum_{(u, v) \in E} w_{uv} \\, \\mathbf{1}[u \in S, v \in V \\setminus S].\n"
    ),


    "Maximum Cut Weighted": (
            "The problem is defined over an undirected graph G = (V, E), where each node v ∈ V has an "
            "associated weight w(v) > 0. The weight of each node is defined as "
            "w(v) = (β / |V|) * (|N(v)| - α), where N(v) is the set of neighbors of v, "
            "α = 1/20, and β is a normalizing factor ensuring w(v) ≥ 1. "
            "Given a budget k, the task is to select a subset S ⊆ V such that "
            "the total weight ∑_{v ∈ S} w(v) ≤ k, while maximizing the cut value, i.e., "
            "the number of edges crossing between S and V \\ S. "
            "Formally, the objective is:\n"
            "f(S) = |{ (u, v) ∈ E : u ∈ S, v ∈ V \\ S }|."
    ),

    "Influence Maximization": (
    "The problem is defined over a directed graph \( G = (V, E) \).\n"
    "Given a budget \( k \), the objective is to select a seed set of nodes \( S \subseteq V \), with \( |S| \leq k \),\n"
    "such that the expected number of influenced nodes is maximized under the Independent Cascade (IC) model.\n"
    "In the IC model, influence propagates in discrete steps: when a node becomes active, it has a single chance to activate each of its inactive neighbors\n"
    "with a given probability \( p_{uv} \) on edge \( (u, v) \).\n"
    "The process continues until no more activations are possible.\n"
    "Formally, maximize:\n"
    "f(S) = \mathbb{E}[|\text{Influence}(S)|],\n"
    "where \\( \\text{Influence}(S) \\) is the random set of nodes activated starting from the seed set \\( S \\).\n"
    ),

    "Influence Maximization Weighted": (
        "The problem is defined over a directed graph G = (V, E), where each node v ∈ V has an associated weight w(v) > 0. "
        "The weight of each node is defined as w(v) = (β / |V|) * (|N(v)| - α), where N(v) is the set of out-neighbors of v, "
        "α = 1/20, and β is a normalizing factor ensuring w(v) ≥ 1. "
        "Given a budget k, the goal is to select a seed set S ⊆ V such that the total weight ∑_{v ∈ S} w(v) ≤ k, "
        "while maximizing the expected influence spread under the Independent Cascade (IC) model. "
        "In the IC model, when a node becomes active, it has a single chance to activate each inactive neighbor v through edge (u, v) "
        "with probability p_uv. The process continues until no further activations occur. "
        "Formally, the objective is:\n"
        "f(S) = E[|Influence(S)|], "
        "where Influence(S) is the random set of nodes activated starting from the weighted seed set S."
    )


}

def load_from_pickle(file_path):
    """
    Load data from a pickle file.

    Parameters:
    - file_path: The path to the pickle file.

    Returns:
    - loaded_data: The loaded data.
    """
    with open(file_path, 'rb') as file:
        loaded_data = pickle.load(file)
    # print(f'Data has been loaded from {file_path}')
    return loaded_data

client = OpenAI(api_key = load_from_pickle('../key.pkl'))

def get_response(client,prompt):

    response = client.responses.create(
        # model="gpt-4.1",
        model = "gpt-5-nano",
        input= prompt
    )

    return response.output_text



def relabel_graph(graph: nx.Graph):
    """
    Relabel the nodes of the input graph to have consecutive integer labels starting from 0.

    Parameters:
    graph (nx.Graph): The input graph to be relabeled.

    Returns:
    tuple: A tuple containing the relabeled graph, a forward mapping dictionary, 
           and a reverse mapping dictionary.
           - relabeled_graph (nx.Graph): The graph with nodes relabeled to consecutive integers.
           - forward_mapping (dict): A dictionary mapping original node labels to new integer labels.
           - reverse_mapping (dict): A dictionary mapping new integer labels back to the original node labels.
    """
    forward_mapping = dict(zip(graph.nodes(), range(graph.number_of_nodes())))
    reverse_mapping = dict(zip(range(graph.number_of_nodes()), graph.nodes()))
    graph = nx.relabel_nodes(graph, forward_mapping)

    return graph, forward_mapping, reverse_mapping



def generate_summary_prompt(cumulative_feedback):
    """
    Generate a prompt for summarizing the cumulative feedback from previous iterations.

    Parameters:
    cumulative_feedback (str): The cumulative feedback string from previous iterations.

    Returns:
    str: A formatted prompt for summarizing the feedback.
    """
    return (
        "You are an expert in graph neural networks and combinatorial optimization.\n\n"
        "Summarize the following feedback from previous iterations:\n"
        f"{cumulative_feedback}\n\n"
        "Provide a concise summary of the key points and insights."
    )


def generate_llm_prompt(
        problem,
        problem_definition,
        # heuristic_description,
        explainer_feedback=None,
    ):

    if problem.endswith('Weighted'):
        few_shot_examples = "Degree to weight ratio must be included." 
    else:
        few_shot_examples = "Degree must be included."

    base_prompt = (
        f"You are an expert in graph neural networks and combinatorial optimization.\n\n"
        f"For the {problem} problem ({problem_definition}), propose up to ten node-level features. {few_shot_examples}. "
        f"for a GNN binary classifier that predicts nodes likely to be in the optimal solution.\n\n"
        f"The heuristic can only select nodes from the reduced candidate set provided by the GNN. "
        f"The goal is to shrink the candidate set while ensuring the heuristic still achieves the same objective value.\n\n"
        # f"Heuristic:\n{heuristic_description}\n\n"
    )

    if explainer_feedback:
        base_prompt += (
            f"Feedback from previous iterations:\n{explainer_feedback}\n\n"
            "Refinement rules:\n"
            "1) Keep high-importance features.\n"
            "2) Adjust or normalize medium-importance features.\n"
            "3) Remove or replace low-importance features.\n"
            "4) Add new features inspired by important patterns.\n"
            "5) Ensure all features are distinct and non-redundant.\n\n"
        )

    base_prompt += (
        "Return the output as a JSON array of objects with keys:\n"
        '- "feature": feature name in snake_case\n'
        '- "definition": one-line definition of how to compute it\n'
        '- "reason": 1–2 sentences on why it is useful\n\n'
        "Do not include any text outside the JSON array."
    )

    return base_prompt



# def generate_llm_prompt(problem, problem_definition, explainer_feedback=None):
#     base_prompt = (
#         f"You are an expert in graph neural networks and combinatorial optimization.\n\n"
#         f"For the {problem} problem ({problem_definition}), list node-level features "
#         f"for a GNN binary classifier predicting nodes likely in the optimal solution to reduce the candidate set.\n\n"
#     )
#     if explainer_feedback:
#         base_prompt += f"Use this feedback from GNNExplainer to refine the feature proposal:\n{explainer_feedback}\n\n"

#     base_prompt += (
#         "Return output as a JSON array where each element is an object with keys:\n"
#         "- `feature`: name of the feature (snake_case)\n"
#         "- `definition`: a one-line formal definition of the feature\n"
#         "- `reason`: a concise explanation (1–2 sentences) why this feature is important\n\n"
#         "Do not add any extra text outside the JSON.\n\n"
#     )

#     return base_prompt

def clean_llm_json(raw_response):
    # Remove ```json ... ``` wrappers if present
    cleaned = re.sub(r"```json\s*|\s*```", "", raw_response.strip())
    
    # Remove any text before or after JSON array
    json_match = re.search(r"(\[.*\])", cleaned, re.DOTALL)
    if json_match:
        cleaned = json_match.group(1)
    
    return cleaned.strip()

def parse_llm_features(llm_response):
    try:
        cleaned_response = clean_llm_json(llm_response)
        data = json.loads(cleaned_response)
        
        features = [item["feature"] for item in data]
        definitions = {item["feature"]: item["definition"] for item in data}
        reasons = {item["feature"]: item["reason"] for item in data}
        
        return features, definitions, reasons
    
    except json.JSONDecodeError as e:
        print("Error parsing cleaned response:", e)
        return [], {}, {}
    
import networkx as nx
import numpy as np

# def assign_random_edge_weights(graph):
#     """
#     Assigns +1 or -1 randomly to each edge in the graph.
#     """
#     edge_weights = {edge: np.random.choice([-1, 1]) for edge in graph.edges()}
#     nx.set_edge_attributes(graph, edge_weights, name="weight")
#     return graph


def assign_normalized_degree_weights(graph, alpha= 1/20):
    out_degrees = {node: (graph.degree(node) - alpha) / graph.number_of_nodes() for node in graph.nodes()}
    out_degree_min = np.min(list(out_degrees.values()))
    node_weights = {node: out_degrees[node] / out_degree_min for node in out_degrees}

    nx.set_node_attributes(graph, node_weights, name="weight")
    
    return graph

def clean_code_block(response):
    code_match = re.search(r"```python(.*?)```", response, re.DOTALL)
    return code_match.group(1).strip() if code_match else response


def generate_train_features(problem,features,definitions,train_graph,test_graph, budget = 100, timeout=5):
    class TimeoutException(Exception):
        pass

    def handler(signum, frame):
        raise TimeoutException
    
    signal.signal(signal.SIGALRM, handler)

    train_X = []
    codes = {}
    

    if problem.endswith('Weighted'):
        graph_description = (
            f"The input is a weighted NetworkX graph `G` where each node has an attribute `'weight'`, "
            f"and an integer variable `budget` is provided.\n"
        )
        additional_description = "If the feature involves weight, use the existing `'weight'` attribute directly without recomputing it from other functions"
    
    elif problem == "Maximum Cut Negative":

        graph_description = (
            f"The input is a weighted NetworkX graph `G` where each edge has an attribute `'weight'`, "
            f"and an integer variable `budget` is provided.\n"
        )
        additional_description = "If the feature involves weight, use the existing `'weight'` attribute directly without recomputing it from other functions"
    
    else:
        graph_description = (
            f"The input is a NetworkX graph `G` where each node has no attribute `'weight'`, "
            f"and an integer variable `budget` is provided.\n"
        )
        additional_description = ''

    # Add tqdm to loop
    for idx, feature in enumerate(tqdm(features, desc="Extracting features", unit="feature")):
        prompt_code = (
            f"{graph_description}"
            f"Feature name: '{feature}'\n"
            f"Feature definition: '{definitions[feature]}'\n"
            # f"Write Python code for a function `extract_feature(G, budget)` that computes this feature for all nodes in `G`. "
            f"Write Python code for a function `extract_feature(G)` that computes this feature for all nodes in `G`. "
            f"{additional_description}"
            # f"If the budget is relevant to the computation, incorporate it. "
            f"The function should return a NumPy array with the computed feature values, ordered to align with the order of `G.nodes()`.\n"
            f"Ensure the code is efficient and avoids expensive computations.\n"
            f"DO NOT INCLUDE ANY EXPLANATIONS OR COMMENTS.\n"
        )



        start = time.time()
        code_response = get_response(client,prompt_code)
        code = clean_code_block(code_response)

        # print(f"Code for feature '{feature}':\n{code}\n")

        
        end = time.time()
        # print(f"Code for feature '{feature}' generated in {end - start:.2f} seconds")

        try:
            # print(f"Extracting feature '{feature}'")
            signal.alarm(timeout)  # Set timeout

            namespace = {}
            exec(code, namespace)  # Execute code in namespace

            namespace["extract_feature"](G=test_graph)
            feature_values = namespace["extract_feature"](G=train_graph)



            if  isinstance(feature_values, np.ndarray) and feature_values.shape[0] == train_graph.number_of_nodes():
                # train_X[feature] = feature_values
                train_X.append(feature_values)

                codes[feature] = code
                # codes.append(code)
        except (TimeoutException, Exception) as e:

            
            print(f"⚠️ Skipping feature '{feature}' due to error: {e}")

            print('*'*30)
            print(code)
            print('*'*30)
        finally:
            signal.alarm(0)  # Reset alarm

    return torch.tensor(np.array(train_X).T, dtype=torch.float),codes

class GCN(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, out_channels):
        super().__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(input_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

def train_test_evaluate_gnn(
                            problem,
                            dataset,
                            train_features, 
                            train_graph,
                            test_graph,
                            codes,
                            heuristic,
                            budget=100,


                           ):


    train_data = from_networkx(train_graph)
    train_data.x = train_features

    # import os

    save_folder = f'Presolved/{problem}/{dataset}'
    os.makedirs(save_folder, exist_ok=True)

    try:
        obj_val, number_of_queries, solution = load_from_pickle(
            f'{save_folder}/train'
        )
    except:
        obj_val, number_of_queries, solution = heuristic(
            train_graph, budget=budget, ground_set=None
        )
        save_as_pickle((obj_val, number_of_queries, solution), f'{save_folder}/train')


    mapping = dict(zip(train_graph.nodes(), range(train_graph.number_of_nodes())))
    train_mask = torch.tensor([mapping[node] for node in solution], dtype=torch.long)
    y= torch.zeros(train_graph.number_of_nodes(),dtype=torch.long)

    for node in solution:
        y[mapping[node]]=1

    train_data.y = y 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data = train_data.to(device)

    # counts = torch.bincount(train_data.y)
    # counts = counts + 1e-6               # avoid zero
    # inv_freq = 1.0 / counts.float()
    # num_classes = counts.size(0)
    # class_weights = inv_freq * (num_classes / inv_freq.sum())

    

    model     = GCN(input_channels=train_data.x.shape[1] ,hidden_channels=16, out_channels=2).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    # criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))


    print('Training GNN')
    model.train()
    for epoch in tqdm(range(1, 1000)):
        optimizer.zero_grad()
        mask = torch.cat([train_mask, torch.randint(0, train_mask.size(0), (train_mask.size(0),))], dim=0)
        out  = model(train_data.x, train_data.edge_index)        # [N, num_classes]
        loss = criterion(out[mask], train_data.y[mask]) 
        # loss = criterion(out, train_data.y)                # full-graph loss
        loss.backward()
        optimizer.step()


    test_data = from_networkx(test_graph)


    

    test_X = []
    for feature in codes:
        namespace = {}
        exec(codes[feature], namespace)  # Execute code in namespace
        feature_values = namespace["extract_feature"](G=test_graph)
        test_X.append(feature_values)  # Assuming budget is relevant
    test_data.x = torch.tensor(np.array(test_X).T, dtype=torch.float).to(device)
    test_data = test_data.to(device)

    test_data.y = torch.zeros(test_graph.number_of_nodes(), dtype=torch.long).to(device)


    y_pred = torch.argmax(model(test_data.x, test_data.edge_index),axis=1).cpu().numpy()

    indices = np.where(y_pred == 1)[0]

    # obj_val,number_of_queries,solution= heuristic(test_graph, budget=budget)
    save_folder = f'Presolved/{problem}/{dataset}'
    os.makedirs(save_folder, exist_ok=True)

    try:
        obj_val, number_of_queries, solution = load_from_pickle(
            f'{save_folder}/test'
        )
    except:
        obj_val, number_of_queries, solution = heuristic(
            test_graph, budget=budget, ground_set=None
        )
        save_as_pickle((obj_val, number_of_queries, solution), f'{save_folder}/test')

    test_data.y[solution] = 1

    print('Objective value:', obj_val)
    obj_val_pruned, number_of_queries_pruned, solution_pruned = heuristic(test_graph, budget=budget, ground_set=indices)
    print('Objective value pruned:', obj_val_pruned)



    print('Ratio',obj_val_pruned/obj_val)
    print('queries ratio',number_of_queries_pruned/number_of_queries)

    ratio = obj_val_pruned / obj_val
    size_reduction = 1 - len(indices) / test_graph.number_of_nodes()

    print('Explaining GNN predictions')

    explainer = Explainer(
        model=model,
        algorithm = GNNExplainer(epochs=300),
        explanation_type='phenomenon',
        node_mask_type='attributes',
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='raw',
        ),
    )

    # Randomly select 100 nodes from indices
    sampled_indices = random.sample(indices.tolist(), min(100, len(indices)))


    feature_importances = []

    for node_index in sampled_indices:
        explanation = explainer(
            test_data.x, 
            test_data.edge_index, 
            target=test_data.y,
            index=int(node_index)
        )
        feature_importances.append(
            explanation.node_mask.sum(dim=0).cpu().detach()
        )

    # Aggregate by mean
    feature_importances = torch.stack(feature_importances)
    mean_importance = feature_importances.mean(dim=0)
                                               
    # Normalize mean_importance
    mean_importance = mean_importance / mean_importance.sum()

    # Create same-line feedback
    explainer_feedback = ", ".join(
        f"{feature}: {importance:.4f}" 
        for feature, importance in zip(codes, mean_importance.tolist())
    )

    # print(explainer_feedback)


    return model,ratio,size_reduction, explainer_feedback




def train_test_split (graph:nx.Graph, ratio:float, edge_level_split:bool, seed:int):

    np.random.seed(seed)
    random.seed(seed)
    

    if edge_level_split:
        
        edges = np.array(graph.edges())
        indices = [i for i in range(len(edges))]
        random.shuffle(indices)
        train_ids = edges[indices[:int(ratio * graph.number_of_edges())]]
        test_ids = edges[indices[int(ratio * graph.number_of_edges()):]]
        logging.info(f" Train edges: {len(train_ids)}, Test edges: {len(test_ids)}")
        train_graph = nx.Graph()
        train_graph.add_edges_from(train_ids)
        for node in list(train_graph.nodes()):
            if train_graph.degree(node) == 0:
                train_graph.remove_node(node)
        

        test_graph = nx.Graph()
        test_graph.add_edges_from(test_ids)
        for node in list(test_graph.nodes()):
            if test_graph.degree(node) == 0:
                test_graph.remove_node(node)
        
        return train_graph, test_graph

    else:
        raise NotImplementedError('Node level splitting not implemented')


