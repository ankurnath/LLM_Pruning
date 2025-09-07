from collections import defaultdict
import networkx as nx

# Compute initial coverage gains for each node
def compute_max_cover_initial_gains(graph, ground_set):
    if ground_set is None:
        gains = {node: graph.degree(node) + 1 for node in graph.nodes()}
    else:
        # print('A candidate node set has been provided.')
        gains = {node: graph.degree(node) + 1 for node in ground_set}
        # print('Size of the candidate set =', len(gains))
    return gains


# Update gains after selecting a node
def update_max_cover_gains_after_selection(graph, gains, selected_node, uncovered_nodes):
    if uncovered_nodes[selected_node]:
        gains[selected_node] -= 1
        uncovered_nodes[selected_node] = False
        for neighbor in graph.neighbors(selected_node):
            if neighbor in gains and gains[neighbor] > 0:
                gains[neighbor] -= 1

    for neighbor in graph.neighbors(selected_node):
        if uncovered_nodes[neighbor]:
            uncovered_nodes[neighbor] = False
            if neighbor in gains:
                gains[neighbor] -= 1
            for neighbor_of_neighbor in graph.neighbors(neighbor):
                if neighbor_of_neighbor in gains:
                    gains[neighbor_of_neighbor] -= 1

    assert gains[selected_node] == 0, f"Gains of selected node = {gains[selected_node]}"


# Greedy Maximum Coverage without weights
def greedy_max_cover(graph, budget, ground_set=None):
    number_of_queries = 0
    gains = compute_max_cover_initial_gains(graph, ground_set)
    solution = []
    uncovered_nodes = defaultdict(lambda: True)
    objective_value = 0

    for i in range(budget):
        number_of_queries += (len(gains) - i)
        selected_node = max(gains, key=gains.get)

        if gains[selected_node] == 0:
            print('All elements are already covered.')
            break

        solution.append(selected_node)
        objective_value += gains[selected_node]
        update_max_cover_gains_after_selection(graph, gains, selected_node, uncovered_nodes)

    # print('Objective value =', objective_value)
    # print('Number of queries =', number_of_queries)
    return objective_value, number_of_queries, solution


# Compute the coverage of a solution
def compute_max_cover_objective(graph, solution):
    covered_nodes = set()
    for node in solution:
        covered_nodes.add(node)
        for neighbor in graph.neighbors(node):
            covered_nodes.add(neighbor)
    return len(covered_nodes)


# Greedy Maximum Coverage with Knapsack constraint
def knapsack_greedy_max_cover(graph, budget, ground_set=None):
    number_of_queries = 0
    node_weights = nx.get_node_attributes(graph, "weight")
    gains = compute_max_cover_initial_gains(graph, ground_set)
    number_of_queries += len(gains)

    # Find best singleton
    best_singleton = None
    best_singleton_gain = 0
    for node in gains:
        if node_weights[node] <= budget and gains[node] > best_singleton_gain:
            best_singleton = node
            best_singleton_gain = gains[node]

    solution = []
    uncovered_nodes = defaultdict(lambda: True)
    total_weight = 0

    while gains:
        number_of_queries += len(gains)
        max_gain_ratio = 0
        selected_node = None

        for node in gains:
            if node_weights[node] + total_weight <= budget:
                ratio = gains[node] / node_weights[node]
                if ratio > max_gain_ratio:
                    max_gain_ratio = ratio
                    selected_node = node

        if selected_node is None:
            break

        solution.append(selected_node)
        update_max_cover_gains_after_selection(graph, gains, selected_node, uncovered_nodes)
        total_weight += node_weights[selected_node]
        gains.pop(selected_node)

        if total_weight >= budget:
            break

    if compute_max_cover_objective(graph, solution) < best_singleton_gain:
        solution = [best_singleton]

    objective_value = compute_max_cover_objective(graph, solution)
    return objective_value, number_of_queries, solution
