import numpy as np
from argparse import ArgumentParser
import networkx as nx



    

def maxcut_get_gains(graph, ground_set):
    if ground_set is None:
        gains = {node: graph.degree(node) for node in graph.nodes()}
    else:
        print('A ground set has been given')
        gains = {node: graph.degree(node) for node in ground_set}
        print('Size of the ground set = ', len(gains))
    
    return gains


def maxcut_gain_adjustment(graph, gains, selected_element, spins):
    gains[selected_element] = -gains[selected_element]

    for neighbor in graph.neighbors(selected_element):
        if neighbor in gains:
            gains[neighbor] += (2 * spins[neighbor] - 1) * (2 - 4 * spins[selected_element])

    spins[selected_element] = 1 - spins[selected_element]
     




def maxcut_greedy(graph, budget, ground_set=None):
    number_of_queries = 0
    gains = maxcut_get_gains(graph, ground_set)
    
    solution = []
    spins = {node: 1 for node in graph.nodes()}
    obj_val = 0

    for i in range(budget):
        number_of_queries += (len(gains) - i)

        selected_element = max(gains, key=gains.get)

        if gains[selected_element] == 0:
            print('All elements are already covered')
            break
        solution.append(selected_element)
        obj_val += gains[selected_element]
        
        maxcut_gain_adjustment(graph, gains, selected_element, spins)
    print('Objective value =', obj_val)
    print('Number of queries =', number_of_queries)

    return obj_val, number_of_queries, solution


if __name__ == "__main__":

    graph = nx.barabasi_albert_graph(100, 5)
    budget = 10
    obj_val, number_of_queries, solution = maxcut_greedy(graph, budget)
    print(f"Objective Value: {obj_val}, Number of Queries: {number_of_queries}, Solution: {solution}")
    
  

