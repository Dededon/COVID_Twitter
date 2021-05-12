import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN, MeanShift
import umap
from sklearn.metrics import roc_auc_score
from node2vec import get_cosine_sim



class Embedding():

    def __init__(self, embedding):
        ##### uncomment for node2vec #######
        # self.__nodes = list(embedding.keys()) 
        # self.__embedding = np.array(list(embedding.values()))
        self.__embedding = np.array(embedding)
        self.__nodes_pos = None
        self.__nodes_cluster = None

    @property
    def embedding(self):
        return dict(zip(self.__nodes, self.__embedding))

    @property
    def nodes_pos(self):
        return dict(zip(self.__nodes, self.__nodes_pos))

    def get_nodes_pos(self):
        return self.__nodes_pos

    def get_nodes_cluster(self):
        return self.__nodes_cluster

    @property
    def nodes_cluster(self):
        return dict(zip(self.__nodes, self.__nodes_cluster))

    def dimReduce(self, method=None, **kwargs):
        if kwargs == None:
            kwargs = dict()
        if method is None and self.__embedding.shape[1] == 2:
            self.__nodes_pos = np.array(self.__embeding)
        elif method is None or method == 'UMAP':
            reducer = umap.UMAP(**kwargs)
            self.__nodes_pos = reducer.fit_transform(self.__embedding)
        elif method == 't-SNE':
            reducer = TSNE(verbose = 1, **kwargs)
            self.__nodes_pos = reducer.fit_transform(self.__embedding)
        elif method == 'FD':
            pos = self.embedding
            self.__nodes_pos = np.array(list(nx.fruchterman_reingold_layout(**kwargs).values()))
        else:
            return

    def cluster(self, method=None, **kwargs):
        if kwargs == None:
            kwargs = dict()
        clusterer = None
        if method is None or method == 'MeanShift':
            clusterer = MeanShift(**kwargs)
        elif method == 'DBSCAN':
            clusterer = DBSCAN(**kwargs)
        else:
            return
        self.__nodes_cluster = clusterer.fit_predict(self.__nodes_pos)

    def plot(self, graph):
        plt.figure(figsize = (16, 12))
        nx.draw(graph, dict(zip(self.__nodes, self.__nodes_pos)), node_size=100, node_color=self.__nodes_cluster)
        plt.draw()

    def evaluate_dimReduce(self, valid_edges, false_edges):
        y_true = [1] * len(valid_edges) + [0] * len(false_edges)

        y_score = list()
        for e in valid_edges:
            y_score.append(get_cosine_sim(self.nodes_pos[e[0]], self.nodes_pos[e[1]]))
        for e in false_edges:
            y_score.append(get_cosine_sim(self.nodes_pos[e[0]], self.nodes_pos[e[1]]))

        return roc_auc_score(y_true, y_score)
