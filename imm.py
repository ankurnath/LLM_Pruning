import pickle
import networkx as nx
import os
import random
import multiprocessing as mp
import time
import math
import networkx as nx
from collections import defaultdict




from utils import *

# NUM_PROCESSORS = mp.cpu_count()

NUM_PROCESSORS = 40

class Worker(mp.Process):
    def __init__(self, inQ, outQ, model, graph_):
        super(Worker, self).__init__(target=self.start)
        self.inQ = inQ
        self.outQ = outQ
        self.R = []
        self.count = 0
        # self.node_num = node_num
        self.model = model
        self.graph = graph_
        self.nodes= list(graph_.network.keys())

    def run(self):

        while True:
            theta = self.inQ.get()
            while self.count < theta:
                # v = random.randint(1, self.node_num-1)
                v=random.choice(self.nodes)

                rr = generate_rr(v, self.model, self.graph)
                self.R.append(rr)
                self.count += 1
            self.count = 0
            self.outQ.put(self.R)
            self.R = []

def create_worker(num, worker,  model, graph_):
    for i in range(num):
        worker.append(Worker(mp.Queue(), mp.Queue(), model, graph_))
        worker[i].start()


def finish_worker(worker):
    for w in worker:
        w.terminate()

def sampling(epsoid, l, node_num, seed_size, worker, graph_, model):
    R = []
    LB = 1
    n = node_num
    k = seed_size
    epsoid_p = epsoid * math.sqrt(2)
    worker_num = NUM_PROCESSORS
    # create_worker(worker_num, worker, node_num, model, graph_)
    create_worker(worker_num, worker, model, graph_)
    for i in range(1, int(math.log2(n - 1)) + 1):
        # s = time.time()
        x = n / (math.pow(2, i))
        lambda_p = ((2 + 2 * epsoid_p / 3) * (logcnk(n, k) + l * math.log(n) + math.log(math.log2(n))) * n) / pow(
            epsoid_p, 2)
        theta = lambda_p / x

        # print(f'Creating new {theta-len(R)} RR sets')
        for ii in range(worker_num):
            worker[ii].inQ.put((theta - len(R)) / worker_num)
        for w in worker:
            R_list = w.outQ.get()
            R += R_list
        # finish_worker()
        # worker = []
        # end = time.time()
        # print('time to find rr', end - s)
        start = time.time()
        # _, f, my_variable,_= node_selection(R, k, node_num)

        solution ,f = node_selection(R, k)
        end = time.time()
        # print('node selection time', time.time() - start)
        # f = F(R,Si)
        if n * f >= (1 + epsoid_p) * x:
            LB = n * f / (1 + epsoid_p)
            break
    # finish_worker()
    alpha = math.sqrt(l * math.log(n) + math.log(2))
    beta = math.sqrt((1 - 1 / math.e) * (logcnk(n, k) + l * math.log(n) + math.log(2)))
    lambda_aster = 2 * n * pow(((1 - 1 / math.e) * alpha + beta), 2) * pow(epsoid, -2)
    theta = lambda_aster / LB
    length_r = len(R)
    diff = theta - length_r
    _start = time.time()
    if diff > 0:
        for ii in range(worker_num):
            worker[ii].inQ.put(diff / worker_num)
        for w in worker:
            R_list = w.outQ.get()
            R += R_list
    '''

    while length_r <= theta:
        v = random.randint(1, n)
        rr = generate_rr(v)
        R.append(rr)
        length_r += 1
    '''
    _end = time.time()
    finish_worker(worker)
    # print('Number of RR sets:',len(R))
    return R



def generate_rr(v, model, graph):
    if model == 'IC':
        return generate_rr_ic(v, graph)
    elif model == 'LT':
        return generate_rr_lt(v, graph)
    

# def node_selection(R,k,node_num=None):
def node_selection(R,k):
    # Sk=set() 
    # list1=[] 
    solution = []
    rr_degree=defaultdict(int)
    node_rr_set = defaultdict(list)
    matched_count = 0

    covered_rr_set=set()
    for j,rr in enumerate(R):
        for rr_node in rr:
            rr_degree[rr_node]+=1
            node_rr_set[rr_node].append(j)
    
    merginal_gains=[]
    for i in range(k):
        max_point=max(rr_degree,key=rr_degree.get)
        # Sk.add(max_point) 
        # list1.append(max_point)
        solution.append(max_point)
        matched_count +=rr_degree[max_point]
        merginal_gains.append(rr_degree[max_point])

        for index in node_rr_set[max_point]:
            if index not in covered_rr_set:
                covered_rr_set.add(index)
                for rr_node in R[index]:
                    rr_degree[rr_node]-=1

    # return Sk, matched_count / len(R), list1,merginal_gains
    return solution,  matched_count / len(R)


def generate_rr_ic(node, graph):
    activity_set = list()
    activity_set.append(node)
    # activity_nodes = list()
    activity_nodes = set()
    # activity_nodes.append(node)
    activity_nodes.add(node)
    while activity_set:
        new_activity_set = list()
        for seed in activity_set:
            for node, weight in graph.get_neighbors(seed):
                if node not in activity_nodes:
                    if random.random() < weight:
                        # activity_nodes.append(node)
                        activity_nodes.add(node)
                        new_activity_set.append(node)
        activity_set = new_activity_set
    return activity_nodes


def generate_rr_lt(node, graph):
    # calculate reverse reachable set using LT model
    # activity_set = list()
    activity_nodes = list()
    # activity_set.append(node)
    activity_nodes.append(node)
    activity_set = node

    while activity_set != -1:
        new_activity_set = -1

        neighbors = graph.get_neighbors(activity_set)
        if len(neighbors) == 0:
            break
        candidate = random.sample(neighbors, 1)[0][0]
        if candidate not in activity_nodes:
            activity_nodes.append(candidate)
            # new_activity_set.append(candidate)
            new_activity_set = candidate
        activity_set = new_activity_set
    return activity_nodes

def logcnk(n, k):
    res = 0
    for i in range(n - k + 1, n + 1):
        res += math.log(i)
    for i in range(1, k + 1):
        res -= math.log(i)
    return res


def get_graph(network):
    """
    Takes  a networkx object
    """
    
    weight = 0.01    
    print(f'Creating with graph default weight = {weight}')
    
    graph_ = Graph()
    # node_num = network.number_of_nodes()

    for u, v in tqdm(network.edges()):

        
        # weight = network[u][v]['weight']
        # if type(weight) == dict:
        #     weight = weight['weight']
        graph_.add_edge(int(u), int(v), weight)
        if type(network) == nx.Graph:
            # graph_.add_edge(int(v), int(u), network[v][u]['weight'])
            graph_.add_edge(int(v), int(u), weight)

    return graph_
    
class Graph:
    """
        graph data structure to store the network
    :return:
    """
    def __init__(self):
        self.network = dict()

    def add_node(self, node):
        if node not in self.network:
            self.network[node] = dict()

    def add_edge(self, s, e, w):
        """
        :param s: start node
        :param e: end node
        :param w: weight
        """
        Graph.add_node(self, s)
        Graph.add_node(self, e)
        # add inverse edge
        self.network[e][s] = w

    def get_out_degree(self, source):
        return len(self.network[source])

    def get_neighbors(self, source):
        if source in self.network:
            return self.network[source].items()
        else:
            return []

    def get_neighbors_keys(self, source):
        if source in self.network:
            return self.network[source].keys()
        else:
            return []

def make_subgraph(graph, nodes):
    assert type(graph) == nx.Graph
    subgraph = nx.Graph()
    edges_to_add = []
    for node in nodes:
        edges_to_add += [(u, v) for u, v in list(graph.edges(node))]
    subgraph.add_edges_from(edges_to_add)
    return subgraph


from collections import deque

def calculate_spread(graph,solution,mc=10000):
    # sprint('Calculating spread')
    # print(f'Default weight has been loaded:{0.01}')
    weight = 0.01    
    print(f'Default weight = {weight}')

    spread = 0

    for _ in tqdm(range(mc)):

        activated_nodes = set(solution)
        queue = deque(solution)

        while queue:
            node = queue.popleft()

            for neighbor in graph.neighbors(node):
                if neighbor not in activated_nodes and random.random() < weight:
                    activated_nodes.add(neighbor)
                    queue.append(neighbor)
        
        spread += len(activated_nodes)

    return spread/mc



def imm(graph, seed_size, ground_set, model="IC", epsoid=0.5, l=1,seed=0):
    """
    graph must be a file path to a .txt file of edge lists where the first line has the number of nodes in the first
    column, or it must be a networkx graph object with edge weights under the key 'weight'.
    """
    
    if ground_set is not None:
        subgraph = make_subgraph(graph,ground_set)
        graph_ = get_graph(subgraph)
        node_num = subgraph.number_of_nodes()

    else: 
        graph_ = get_graph(graph)
        node_num = graph.number_of_nodes()


    # np.random.seed(args.seed)
    # random.seed(args.seed)
    worker = []
    n = node_num
    k = seed_size
    l = l * (1 + math.log(2) / math.log(n))
    R = sampling(epsoid, l, node_num, seed_size, worker, graph_, model)

    # R = sampling(epsoid, l, seed_size, worker, graph_, model)
    # Sk, z, x,merginal_gains = node_selection(R, k, node_num)
    solution, coverage = node_selection(R, k)

    objective_value = calculate_spread(graph, solution)

    print(objective_value)
    # print(R[:10])
    return objective_value,len(R),solution



 

        


if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument( "--dataset", type=str, default='Facebook', help="Name of the dataset to be used (default: 'Facebook')" )
    # parser.add_argument( "--model", type=str, default='CONST', help="Name of the dataset to be used (default: 'Facebook')" )
    parser.add_argument( "--budgets", type=int, default=100 , help="Budgets" )
    parser.add_argument( "--seed", type=int, default=0, help="Random seed for reproducibility (default: 0)" )

    args = parser.parse_args()


    graph = load_from_pickle(file_path=f'../snap_dataset/train/{args.dataset}')

    
    _,_,solution =imm (graph=graph,seed_size=args.budgets,ground_set= None,seed=args.seed)
    print(solution)
    print([graph.degree(node) for node in solution])


    





    







