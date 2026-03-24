from typing import Tuple, List

from collections import Counter

import networkx as nx
import numpy as np
from sklearn.linear_model import LinearRegression

from src.utils.graph import MixedGraph


def total_effect_orders(orders: List[nx.DiGraph], dataset, treatment, target):
    # get precessors of X 
    adjustment_sets = []
    non_identifiable = []
    for order in orders:
        ancestors = list(nx.ancestors(order, treatment))
        if target in ancestors:
            non_identifiable.append(order)
        else:
            adjustment_sets.append(ancestors)

    # count repetitions adjustment sets
    adjustment_sets = dict(Counter(tuple(sorted(set(adj_set))) for adj_set in adjustment_sets))
    adjustment_sets['non_identifiable'] = len(non_identifiable)
    total_effects = {}
    for key, value in adjustment_sets.items():
        print(f'Adjustment set: {key}, size: {value}')
        if key != 'non_identifiable':
            if list(key):
                # merge X, Z 
                model = LinearRegression().fit(
                    dataset.data[[treatment] + list(key)].values,
                    dataset.data[target].values.reshape(-1, 1)                     
                )
                total_effect = model.coef_[0][0]
            else:
                total_effect = LinearRegression().fit(
                    dataset.data[[treatment]],
                    dataset.data[target].values.reshape(-1, 1) 
                ).coef_[0][0]

            total_effects[str(key)] = (total_effect, value)

    return total_effects, adjustment_sets, non_identifiable


def total_effect_mpdag(G: MixedGraph, dataset, treatment, target) -> float:
    """ if not identifiable, return None """
    # identifiable 
    for uedge in G.undirected_edges:
        uedge = uedge[0]
        # get name 
        uedge[0] = dataset.var_description[uedge[0]]
        uedge[1] = dataset.var_description[uedge[1]]
        if uedge[0] == treatment or uedge[1] == treatment:
            return None
    
    # get parents of X
    parents = list(G.parents(treatment))
    parents = [dataset.var_description[i] for i in parents]

    X = dataset.data[treatment].values.reshape(-1, 1)
    Y = dataset.data[target].values.reshape(-1, 1)
    Z = dataset.data[parents].values if parents else None

    if parents:
        # merge X, Z 
        Z = Z.reshape(-1, Z.shape[1])
        X = np.hstack((X, Z))
        model = LinearRegression().fit(X, Y)
        total_effect = model.coef_[0][0]
    else:
        total_effect = LinearRegression().fit(X, Y).coef_[0][0]

    return total_effect