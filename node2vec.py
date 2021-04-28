import networkx as nx
import numpy as np
import os
import time
from gensim.models import Word2Vec
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from itertools import chain
import random


def generate_false_edges(true_edges, num_false_edges=5):
    """
    generate false edges given true edges
    """
    nodes = list(set(chain.from_iterable(true_edges)))
    true_edges = set(true_edges)
    false_edges = set()
    
    while len(false_edges) < num_false_edges:
        # randomly sample two different nodes and check whether the pair exisit or not
        head, tail = np.random.choice(nodes, 2)
        if head != tail and ((head, tail) not in true_edges and (tail, head) not in true_edges) and ((head, tail) not in false_edges and (tail, head) not in false_edges):
            false_edges.add((head, tail))    
    false_edges = sorted(false_edges)
    
    return false_edges

# Random Walk Generator
def __alias_setup(probs):
    """
    compute utility lists for non-uniform sampling from discrete distributions.
    details: https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    """
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = list()
    larger = list()
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q

def get_alias_node(graph, node):
    """
    get the alias node setup lists for a given node.
    """
    # get the unnormalized probabilities with the first-order information
    unnormalized_probs = list()
    for nbr in graph.neighbors(node):
        if 'weight' in graph[node][nbr]:
            unnormalized_probs.append(graph[node][nbr]['weight'])
        else:
            unnormalized_probs.append(1)
    unnormalized_probs = np.array(unnormalized_probs)
    if len(unnormalized_probs) > 0:
        normalized_probs = unnormalized_probs / unnormalized_probs.sum()
    else:
        normalized_probs = unnormalized_probs
        
    return __alias_setup(normalized_probs)
    
def get_alias_edge(graph, src, dst, p=1, q=1):
    """
    get the alias edge setup lists for a given edge.
    """
    # get the unnormalized probabilities with the second-order information
    unnormalized_probs = list()
    for dst_nbr in graph.neighbors(dst):
        if dst_nbr == src: # distance is 0
            if 'weight' in graph[dst][dst_nbr]:
                unnormalized_probs.append(graph[dst][dst_nbr]['weight'] / p)
            else:
                unnormalized_probs.append(1 / p)
        elif graph.has_edge(dst_nbr, src): # distance is 1
            if 'weight' in graph[dst][dst_nbr]:
                unnormalized_probs.append(graph[dst][dst_nbr]['weight'])
            else:
                unnormalized_probs.append(1)
        else: # distance is 2
            if 'weight' in graph[dst][dst_nbr]:
                unnormalized_probs.append(graph[dst][dst_nbr]['weight'] / q)
            else:
                unnormalized_probs.append(1 / q)
    unnormalized_probs = np.array(unnormalized_probs)
    if len(unnormalized_probs) > 0:
        normalized_probs = unnormalized_probs / unnormalized_probs.sum()
    else:
        normalized_probs = unnormalized_probs

    return __alias_setup(normalized_probs)


# DFS (better)
def __alias_draw(J, q):
    """
    draw sample from a non-uniform discrete distribution using alias sampling.
    """
    K = len(J)

    kk = int(np.floor(np.random.rand() * K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]
        
def generate_dfs_first_order_random_walk(graph, alias_nodes, walk_length=10, start_node=None):
    """
    simulate a random walk starting from start node and considering the first order information.
    """
    if start_node == None:
        start_node = np.random.choice(graph.nodes())
    walk = [start_node]
    cur = start_node
    while len(walk) < walk_length:
        cur_nbrs = list(graph.neighbors(cur))
        if len(cur_nbrs) > 0:
            # sample the next node based on alias_nodes
            cur = cur_nbrs[__alias_draw(*alias_nodes[cur])]
            walk.append(cur)
        else:
            break

    return walk
    
def generate_dfs_second_order_random_walk(graph, alias_nodes, alias_edges, walk_length=10, start_node=None):
    """
    simulate a random walk starting from start node and considering the second order information.
    """
    if start_node == None:
        start_node = np.random.choice(graph.nodes())
    walk = [start_node]
    
    prev = None
    cur = start_node
    while len(walk) < walk_length:
        cur_nbrs = list(graph.neighbors(cur))
        if len(cur_nbrs) > 0:
            if prev is None:
                # sample the next node based on alias_nodes
                prev, cur = cur, cur_nbrs[__alias_draw(*alias_nodes[cur])]
            else:
                # sample the next node based on alias_edges
                prev, cur = cur, cur_nbrs[__alias_draw(*alias_edges[(prev, cur)])]
            walk.append(cur)
        else:
            break

    return walk


# BFS
def __alias_draw_sequence(J, q, cur_nbrs):
    """
    draw sample from a non-uniform discrete distribution using alias sampling.
    """

    K = len(J)
    sequence = np.random.choice(K, K, False)
    to_return = list()

    for kk in sequence:
        if np.random.rand() < q[kk]:
            to_return.append(cur_nbrs[kk])
        else:
            to_return.append(cur_nbrs[J[kk]])

    return to_return

def generate_bfs_first_order_random_walk(graph, alias_nodes, walk_length=10, start_node=None):
    """
    simulate a random walk starting from start node and considering the first order information.
    """
    if start_node == None:
        start_node = np.random.choice(graph.nodes())
    walk = [start_node]
    cur_idx = 0
    while len(walk) < walk_length:
        cur_nbrs = list(graph.neighbors(walk[cur_idx]))
        if len(cur_nbrs) > 0:
            # sample the next node based on alias_nodes
            walk += __alias_draw_sequence(*alias_nodes[walk[cur_idx]], cur_nbrs)
            cur_idx += 1
        else:
            break
    return walk[:walk_length]
    
def generate_bfs_second_order_random_walk(graph, alias_nodes, alias_edges, walk_length=10, start_node=None):
    """
    simulate a random walk starting from start node and considering the second order information.
    """
    if start_node == None:
        start_node = np.random.choice(graph.nodes())
    walk = [start_node]
    
    prev = None
    cur_idx = 0
    prev = [0]
    while len(walk) < walk_length:
        cur_nbrs = list(graph.neighbors(walk[cur_idx]))
        if len(cur_nbrs) > 0:
            walk_sequence = None
            if len(prev) == 1:
                # sample the next node based on alias_nodes
                walk_sequence = __alias_draw_sequence(*alias_nodes[walk[cur_idx]], cur_nbrs)
            else:
                # sample the next node based on alias_edges
                walk_sequence = __alias_draw_sequence(*alias_edges[(prev[cur_idx], walk[cur_idx])], cur_nbrs)
            walk += walk_sequence
            for i in range(len(walk_sequence)):
                prev.append(walk[cur_idx])
            cur_idx += 1
        else:
            break

    return walk[:walk_length]


# Get Similarity between 2 Nodes
def get_cosine_sim(u_emb, v_emb):
    if type(u_emb) != np.ndarray or type(v_emb) != np.ndarray:
        return 0
    else:
        return np.dot(u_emb, v_emb) / (np.linalg.norm(u_emb) * np.linalg.norm(v_emb))


# Build Models
class EmbedModel():

    def __init__(self, graph,
                    node_dim=10,
                    num_walks=10,
                    walk_length=10,
                    walk_method="dfs"):
        self.graph = graph.copy()
        self.valid_edges = random.sample(self.graph.edges, int(0.25 * self.graph.number_of_edges()))
        self.train_graph = graph.copy()
        self.train_graph.remove_edges_from(self.valid_edges)
        self.false_edges = generate_false_edges(self.graph.edges, num_false_edges=int(0.25 * self.graph.number_of_edges()))

        self.node_dim = node_dim
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.walk_method = walk_method
        self.model = None
        
    def get_embedding(self, node=None):
        if node is None:
            return dict(zip(self.train_graph.nodes, [self.model.wv.vectors[self.model.wv.index2word.index(node)] for node in self.train_graph.nodes]))
        try:
            return self.model.wv.vectors[self.model.wv.index2word.index(node)]
        except:
            return None
        
    def evaluate(self):
        y_true = [1] * len(self.valid_edges) + [0] * len(self.false_edges)
        
        y_score = list()
        for e in self.valid_edges:
            y_score.append(get_cosine_sim(self.get_embedding(e[0]), self.get_embedding(e[1])))
        for e in self.false_edges:
            y_score.append(get_cosine_sim(self.get_embedding(e[0]), self.get_embedding(e[1])))

        return roc_auc_score(y_true, y_score)

class Deepwalk(EmbedModel):

    def __init__(self, graph,
                    node_dim=10,
                    num_walks=10,
                    walk_length=10,
                    walk_method="dfs"):
        if walk_method == "bfs":
            walk_method = generate_bfs_first_order_random_walk
        elif walk_method == "dfs":
            walk_method = generate_dfs_first_order_random_walk
        super().__init__(graph, node_dim, num_walks, walk_length, walk_method)

    def __call__(self):
        print("building a DeepWalk model...", end='\t')
        st = time.time()
        np.random.seed(0)
        nodes = list(self.train_graph.nodes())
        walks = list()
        # generate alias nodes
        alias_nodes = dict()
        for node in self.train_graph.nodes():
            alias_nodes[node] = get_alias_node(self.train_graph, node)
        # generate random walks
        for walk_iter in range(self.num_walks):
            np.random.shuffle(nodes)
            for node in nodes:
                walks.append(self.walk_method(
                    self.train_graph, alias_nodes, walk_length=self.walk_length, start_node=node
                ))

        walk_lens = [len(w) for w in walks]
        if len(walk_lens) > 0:
            avg_walk_len = sum(walk_lens) / len(walk_lens)
        else:
            avg_walk_len = 0.0
        
        print("number of walks: %d\taverage walk length: %.4f" % (len(walks), avg_walk_len), end="\t")
    
        # train a skip-gram model for these walks
        self.model = Word2Vec(walks, size=self.node_dim, window=3, min_count=0, sg=1, workers=os.cpu_count(), iter=10)
        print("trainig time: %.4f" % (time.time()-st))
        
        return self.model

class node2vec(EmbedModel):

    def __init__(self, graph,
                    p=1, q=1,
                    node_dim=10,
                    num_walks=10,
                    walk_length=10,
                    walk_method="dfs"):
        if walk_method == "bfs":
            walk_method = generate_bfs_second_order_random_walk
        elif walk_method == "dfs":
            walk_method = generate_dfs_second_order_random_walk
        super().__init__(graph, node_dim, num_walks, walk_length, walk_method)
        self.p = p
        self.q = q

    def __call__(self):
        print("building a node2vec model...", end='\t')
        st = time.time()
        np.random.seed(0)
        nodes = list(self.train_graph.nodes())
        walks = list()
        # generate alias nodes
        alias_nodes = dict()
        for node in self.train_graph.nodes():
            alias_nodes[node] = get_alias_node(self.train_graph, node)
        alias_edges = dict()
        for edge in self.train_graph.edges():
            alias_edges[edge] = get_alias_edge(self.train_graph, edge[0], edge[1], p=self.p, q=self.q)
            alias_edges[(edge[1], edge[0])] = get_alias_edge(self.train_graph, edge[1], edge[0], p=self.p, q=self.q)
        # generate random walks
        for walk_iter in range(self.num_walks):
            np.random.shuffle(nodes)
            for node in nodes:
                walks.append(self.walk_method(
                    self.train_graph, alias_nodes, alias_edges, walk_length=self.walk_length, start_node=node
                ))

        walk_lens = [len(w) for w in walks]
        if len(walk_lens) > 0:
            avg_walk_len = sum(walk_lens) / len(walk_lens)
        else:
            avg_walk_len = 0.0
        
        print("number of walks: %d\taverage walk length: %.4f" % (len(walks), avg_walk_len), end="\t")
    
        # train a skip-gram model for these walks
        self.model = Word2Vec(walks, size=self.node_dim, window=3, min_count=0, sg=1, workers=os.cpu_count(), iter=10)
        print("trainig time: %.4f" % (time.time()-st))
        
        return self.model
