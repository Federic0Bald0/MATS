import networkx as nx
import numpy as np
from src.utils.metrics import TopologicalOrderError
from causallearn.utils.PCUtils import SkeletonDiscovery
from causallearn.utils.cit import *
import matplotlib.pyplot as plt
from cdt.metrics import SHD
from itertools import combinations

def compute_subset(A, res, subset, index):
    res.append(subset[:])
    for i in range(index, len(A)):
        subset.append(A[i])
        compute_subset(A, res, subset, i + 1)
        subset.pop()


def match_var(var, var_names, var_description):
    for i, name in enumerate(var_names):
        if var.lower() == name.lower():
            return i
    for i, desc in enumerate(var_description):
        if var.lower() in desc.lower():
            return i


def subsets(A):
    subset = []
    res = []
    index = 0
    compute_subset(A, res, subset, index)
    return res


def partitions_of_size_n(lst, n):
    return list(combinations(lst, n))


def eval_causal_order(true_graph: nx.DiGraph, estimated_graphs: nx.DiGraph):

    topo_metric = TopologicalOrderError()
    shd_metric = SHD
    topo_errors, shd = [], []
    for estimated_graph in estimated_graphs:
        topo_error, _ = topo_metric(true_graph, estimated_graph)
        shd_error = shd_metric(true_graph, estimated_graph)
        topo_errors.append(topo_error)
        shd.append(shd_error)

    return topo_errors, shd


def discover_with_orders(dataset, causal_orders, indep_test='fisherz'):
    skeleton = SkeletonDiscovery.skeleton_discovery(
            data=dataset.data.values,
            alpha=0.05,
            indep_test=CIT(dataset.data.values, method='fisherz'),
            node_names=list(dataset.var_description),
            stable=True
            )
    skeleton.to_nx_graph()
    skeleton = skeleton.nx_graph

    # asign labels to the skeleton
    mapping = {i: v for i, v in enumerate(dataset.graph.nodes)}
    skeleton = nx.relabel_nodes(skeleton, mapping)
    
    
    DAGs = []
    for order in causal_orders:
        to_remove = []
        for edge in order.edges:
            if not skeleton.has_edge(edge[0], edge[1]):
                to_remove.append(edge)
        order.remove_edges_from(to_remove)
        
        duplicate = False
        for graph in DAGs:
            if graph.nodes == order.nodes and graph.edges == order.edges:
                duplicate = True
        DAGs.append(order) if not duplicate else None
    return DAGs
