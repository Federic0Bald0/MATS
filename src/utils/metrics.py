from typing import List, Union

import numpy as np
import networkx as nx

from matplotlib import pyplot as plt


class Metrics:

    def __init__(self, name):
        self.name = name

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray):
        pass

class CausalOrder(Metrics):
    """ Causal Order DAG """
    def __init__(self):
        super().__init__('Causal Order')
    
    @staticmethod
    def _add_undirected_edges_independent(adj_matrix: np.ndarray) -> np.ndarray:
        n = len(adj_matrix[0])
        for i, _ in enumerate(range(n)):
            for j, _ in enumerate(range(n)):
                if i != j and adj_matrix[i, j] == 0 and adj_matrix[j, i] == 0:
                    adj_matrix[i, j] = np.nan 
                    adj_matrix[j, i] = np.nan

        return adj_matrix

    def __call__(
        self,
        y_true: nx.DiGraph,
        y_pred: nx.DiGraph,
    ):
        y_true = nx.to_numpy_array(y_true)
        y_pred = nx.to_numpy_array(y_pred)
        n = len(y_true)
        y_true = self._add_undirected_edges_independent(y_true)
        y_pred = np.where(y_pred > 0, 1, 0)

        diff = np.abs(y_true - y_pred)
        diff = np.nan_to_num(diff, nan=0)

        error = np.sum(diff.ravel())
        norm_error = np.sum(diff.ravel()) / ((n * (n - 1)/2))
        return error, norm_error
    

# class SHD(Metrics):
#     """ Structural Hamming Distance """
#     def __init__(self):
#         super().__init__('Structural Hamming Distance')

#     def __call__(
#         self,
#         true_graph: nx.DiGraph,
#         estimated_graph: nx.DiGraph,
#     ) -> float:
#         """ Evaluate the structural hamming distance """
#         true_graph = nx.to_numpy_array(true_graph)
#         true_graph = np.where(true_graph > 0, 1, 0)
#         estimated_graph = nx.to_numpy_array(estimated_graph)
#         estimated_graph = np.where(estimated_graph > 0, 1, 0)
        
#         shd = np.sum(np.abs(true_graph - estimated_graph))
#         return shd


class TopologicalOrderError(Metrics):

    def __init__(self):
        super().__init__('Topological Order Error')

    def __call__(
        self,
        true_graph: nx.DiGraph,
        estimated_graph: nx.DiGraph,
    ) -> float:
        """ Evaluate the topological order error """
        # print(true_graph, estimated_graph)
        error = 0
        for node in true_graph.nodes:
            children = set(true_graph.successors(node))
            estimated_descendants = set(nx.descendants(estimated_graph, node))
            for child in children:
                if child not in estimated_descendants:
                    error += 1
        norm_error = error / len(true_graph.edges)

        return error, norm_error