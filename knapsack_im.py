from collections import deque
from imm import *


def get_gains(graph,num_rr):

    graph_ = get_graph(graph)


    # tracks what nodes cover each reversible sets
    RR = []

    worker = []
    worker_num =NUM_PROCESSORS
    create_worker(num =worker_num, worker = worker,  model = 'IC', graph_=graph_)

    for ii in range(worker_num):
            worker[ii].inQ.put(num_rr / worker_num)
    for w in worker:
        R_list = w.outQ.get()
        RR += R_list

    finish_worker(worker)

    

    gains = defaultdict(int) 
    # Keep tracks for what node covers which RR sets
    node_rr_set = defaultdict(list)

    for j,rr in enumerate(RR):
        for rr_node in rr:
            gains[rr_node]+=1
            node_rr_set[rr_node].append(j)

    
    return gains,node_rr_set,RR



def gain_adjustment(gains,node_rr_set,RR,selected_element,covered_rr_set):
     
    for index in node_rr_set[selected_element]:
        if index not in covered_rr_set:
            covered_rr_set.add(index)
            for rr_node in RR[index]:
                if rr_node in gains:
                    gains[rr_node]-=1

    
    assert gains[selected_element] == 0, 'gains adjustment error'





# def knapsack_im_greedy(graph,ground_set , budget ,node_weights,
#                     gains=None,node_rr_set=None,RR=None,num_rr=None):


def knapsack_im_greedy(graph, budget, ground_set=None, num_rr=100000):

    N = graph.number_of_nodes()
    node_weights = np.array([graph.nodes[i]['weight'] for i in range(N)])


    
    gains,node_rr_set,RR = get_gains(graph=graph,num_rr=num_rr)

    if ground_set is not None:
        gains= {node:gains[node] for node in ground_set if node in gains}

    

    max_singleton = None

    max_singleton_gain = 0

    number_of_queries = len(gains)

    # sprint(len(gains))

    # only taking element in ground set
    for element in gains:
        # try:
        if node_weights[element]<= budget and gains[element] > max_singleton_gain:
            max_singleton = element
            max_singleton_gain = gains [element]
        # except:
        #     sprint(element)
        #     sprint(node_weights[element])
        #     sprint(gains[element])
        #     raise ValueError


    constraint = 0
    N = len (gains)

    covered_rr_set = set ()
    solution = []

    objective_value = 0
    
    while gains:
        number_of_queries+= len(gains)
        max_gain_ratio = 0

        selected_element = None

        for element in gains:
            if gains[element]/node_weights[element]> max_gain_ratio:
                max_gain_ratio = gains[element]/node_weights[element]
                selected_element = element

        if max_gain_ratio == 0 :

            break

        if node_weights[selected_element]+constraint <= budget:

            solution.append(selected_element)
            objective_value += gains[selected_element]
            gain_adjustment(gains=gains,node_rr_set=node_rr_set,RR=RR,
                            selected_element=selected_element,covered_rr_set=covered_rr_set)
            constraint += node_weights[selected_element]


        if constraint == budget:
            break

        gains.pop(selected_element)

    # print('Number of queries:',number_of_queries)
        
    if calculate_spread(graph,solution)< calculate_spread(graph,solution=[max_singleton]) :
        solution = [ max_singleton ]

    return objective_value, number_of_queries, solution
