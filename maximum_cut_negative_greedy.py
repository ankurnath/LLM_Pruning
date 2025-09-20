import numpy as np
import networkx as nx


import gurobipy as gp
from gurobipy import GRB
from utils import load_from_pickle,relabel_graph


# def assign_random_edge_weights(graph):
#     """
#     Assigns +1 or -1 randomly to each edge in the graph.
#     """
#     edge_weights = {edge: np.random.choice([-1, 1]) for edge in graph.edges()}
#     nx.set_edge_attributes(graph, edge_weights, name="weight")
#     return graph

def assign_random_edge_weights(graph: nx.Graph, seed: int = 42) -> nx.Graph:
    rng = np.random.default_rng(seed)
    edge_weights = {edge: rng.choice([-1, 1]) for edge in graph.edges()}
    nx.set_edge_attributes(graph, edge_weights, name="weight")
    return graph

import networkx as nx

def get_gains(graph: nx.Graph, ground_set=None):
    """
    Compute initial delta_local_cuts and gains.
    """
    delta_local_cuts = {node: 0.0 for node in graph.nodes()}

    # Initial score and deltas
    for u, v, w in graph.edges(data="weight", default=1.0):
        # delta_local_cuts[u] += w * (2 * spins[u] - 1) * (2 * spins[v] - 1)
        # delta_local_cuts[v] += w * (2 * spins[v] - 1) * (2 * spins[u] - 1)
        delta_local_cuts[u] += w
        delta_local_cuts[v] += w

    if ground_set is None:
        gains = delta_local_cuts.copy()
    else:
        gains = {node: delta_local_cuts[node] for node in ground_set}

    return delta_local_cuts, gains


def gain_adjustment(graph: nx.Graph, v, spins, delta_local_cuts, gains, curr_score):
    """
    Adjust score and update neighbors after selecting node v.
    """
    curr_score += delta_local_cuts[v]
    delta_local_cuts[v] = -delta_local_cuts[v]
    if v in gains:
        gains[v] = -gains[v]

    # Update neighbors
    for u, w in graph[v].items():
        weight = w.get("weight", 1.0)
        delta_local_cuts[u] += weight * (2 * spins[u] - 1) * (2 - 4 * spins[v])
        if u in gains:
            gains[u] += weight * (2 * spins[u] - 1) * (2 - 4 * spins[v])

    spins[v] = 1 - spins[v]
    return curr_score, spins, delta_local_cuts, gains


def standard_greedy(graph: nx.Graph, budget, ground_set=None):
    """
    Standard greedy algorithm for Max-Cut on a NetworkX weighted graph.
    """
    nodes = list(graph.nodes())
    # spins = {node: 1 for node in nodes}  # start with all 1
    curr_score = 0.0
    solution = set()

    delta_local_cuts, gains = get_gains(graph, ground_set)
    spins = {node: 1 for node in nodes}

    while len(solution) < budget:
        # Select node with maximum delta
        v = max(gains, key=delta_local_cuts.get)

        if delta_local_cuts[v] <= 0:
            break

        curr_score, spins, delta_local_cuts, gains = gain_adjustment(
            graph, v, spins, delta_local_cuts, gains, curr_score
        )

        if v in solution:
            solution.remove(v)
        else:
            solution.add(v)

    return curr_score, 1, list(solution)


import networkx as nx

def calculate_obj(graph: nx.Graph, solution: set) -> float:
    """
    Calculate the Max-Cut objective value for a given solution.
    
    Parameters
    ----------
    graph : nx.Graph
        A weighted NetworkX graph. Edge weights should be stored as "weight".
    solution : set
        A set of nodes representing one side of the cut.
    
    Returns
    -------
    float
        The total cut value (sum of weights of edges crossing the cut).
    """
    obj_value = 0.0
    for u, v, w in graph.edges(data="weight", default=1.0):
        if (u in solution and v not in solution) or (v in solution and u not in solution):
            obj_value += w
    return obj_value




# def standard_greedy(graph: nx.Graph, budget, ground_set = None):
#     """
#     Standard greedy algorithm for Max-Cut on a NetworkX weighted graph.
#     Assumes graph has edge weights stored as "weight".
#     """
#     nodes = list(graph.nodes())
#     n = len(nodes)
    

#     spins = {node: 1 for node in nodes}  # start with all 1
#     delta_local_cuts = {node: 0.0 for node in nodes}
#     curr_score = 0.0

#     # Initial score and deltas
#     for u, v, w in graph.edges(data="weight", default=1.0):
#         delta_local_cuts[u] += w * (2 * spins[u] - 1) * (2 * spins[v] - 1)
#         delta_local_cuts[v] += w * (2 * spins[v] - 1) * (2 * spins[u] - 1)
#         # curr_score += w * (spins[u] + spins[v] - 2 * spins[u] * spins[v])

#     objective_value = 0

#     if ground_set is None:
#         gains = delta_local_cuts.copy()

#     else:
#         gains = {node:delta_local_cuts[node] for  node in ground_set}

    
#     solution = set([])

#     # for _ in range(budget):
#     while len(solution) < budget:
#         # Select node with maximum delta
#         # v = max(delta_local_cuts, key=delta_local_cuts.get)
#         v = max(gains,key=delta_local_cuts.get)

#         if delta_local_cuts[v] <= 0:
#             break

#         curr_score += delta_local_cuts[v]
#         delta_local_cuts[v] = -delta_local_cuts[v]
#         gains[v] = -gains[v]

#         # Update neighbors
#         for u, w in graph[v].items():
#             weight = w.get("weight", 1.0)
#             delta_local_cuts[u] += weight * (2 * spins[u] - 1) * (2 - 4 * spins[v])
#             if u in gains:
#                 gains[u] += weight * (2 * spins[u] - 1) * (2 - 4 * spins[v])

#         spins[v] = 1 - spins[v]

#         if v in solution:
#             solution.pop(v)
#         else:
#             solution.add(v)



#     return curr_score,1,list(solution)





def gurobi_solver(graph, max_time=None, max_threads=15, budget=None):
    model = gp.Model()
    model.setParam("OutputFlag", 0)

    if max_time:
        model.setParam('TimeLimit', max_time)

    if max_threads:
        model.setParam('Threads', max_threads)

    vdict = model.addVars(graph.number_of_nodes(), vtype=GRB.BINARY, name="Build")

    cut = [
        data['weight'] * (vdict[i] + vdict[j] - 2 * vdict[i] * vdict[j])
        for i, j, data in graph.edges(data=True)
    ]

    model.setObjective(sum(cut), GRB.MAXIMIZE)

    # Budget constraint: at most `budget` nodes can be chosen
    if budget is not None:
        model.addConstr(
            gp.quicksum(vdict[i] for i in range(graph.number_of_nodes())) <= budget,
            "BudgetConstraint"
        )

    model.optimize()

    return model.ObjVal,1, [key for key in vdict.keys() if abs(vdict[key].x) > 1e-6]






if __name__ == "__main__":

    # graph = nx.barabasi_albert_graph(n=10000,m=4,seed=0)

    # graph = assign_random_edge_weights(graph=graph)

    graph = load_from_pickle(f'../snap_dataset/test/{'Facebook'}')

    graph,_,_ = relabel_graph(graph)

    graph = assign_random_edge_weights(graph=graph)

    objective_value,_,solution = standard_greedy(graph=graph,budget=100,ground_set=None)

    print('Objective value', objective_value)
    # print('Solution', solution)


    objective_value,_,solution = gurobi_solver(graph=graph,budget=100)

    print('Objective value', objective_value)
    # print('Solution', solution)


    #############




