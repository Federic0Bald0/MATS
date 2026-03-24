from typing import List, Set

import copy
import igraph as ig
import networkx as nx
from collections import defaultdict

import numpy as np


class MixedGraph:
    def __init__(self, reverse_weights=True):
        super().__init__()
        self.undirected_edges = []
        self.directed_G = ig.Graph(directed=True)
        self.reverse_weights = [] if reverse_weights is not None else None

    def subgraph(self, nodes):
        G_sub = self.copy()
        G_sub.directed_G = self.directed_G.induced_subgraph(nodes)
        return G_sub
    
    def parents(self, node):
        return self.directed_G.predecessors(node)

    def is_dag(self):
        return self.directed_G.is_dag()
    
    def strongly_connected_components(self):
        return self.directed_G.connected_components(mode='strong')

    def nodes(self):
        return self.directed_G.vs
    
    def edges(self):
        return self.directed_G.es
    
    def degree(self, node):
        return self.directed_G.degree(node)

    def add_node(self, name):
        self.directed_G.add_vertex(name)

    def add_nodes(self, nodes):
        self.directed_G.add_vertices(nodes)

    def get_eid(self, i, j):
        return self.directed_G.get_eid(i, j)
    
    def get_adjacency(self):
        return np.array(self.directed_G.get_adjacency())

    def add_directed_edge(self, i, j, weight=None, reverse_weight=None):
        if not self.directed_G.are_adjacent(i, j) and not self.directed_G.are_adjacent(j, i):
            self.directed_G.add_edge(i, j, weight=weight)
            if self.reverse_weights is not None:
                self.reverse_weights.append(reverse_weight)

    def add_directed_edges(self, edges):    
        for edge in edges:
            self.add_directed_edge(*edge)

    def has_directed_edge(self, i, j):
        return self.directed_G.are_adjacent(i, j)

    def remove_directed_edge(self, eid):
        self.directed_G.delete_edges(eid)

    def add_undirected_edge(self, i, j, weight=None):
        if not self.has_undirected_edge(i, j):
            self.undirected_edges.append((sorted([i, j]), weight))

    def has_undirected_edge(self, i, j):
        edge = sorted([i, j])
        for uedge in self.undirected_edges:
            if uedge[0] == edge:
                return True
        return False

    def remove_directed_edges(self, edges):
        self.directed_G.delete_edges(edges)
    
    def remove_undirected_edge(self, edge):
        edge = sorted(edge)
        for i in range(len(self.undirected_edges)):
            if list(self.undirected_edges[i][0]) == edge:
                self.undirected_edges.pop(i)
                break

    def get_weight(self, eid, reverse=False):
        if reverse:
            return self.reverse_weights[eid]
        return self.directed_G.es()[eid]['weight']
    
    def set_weight(self, eid, weight, reverse=False):  
        if reverse:
            self.reverse_weights[eid] = weight
        else:
            self.directed_G.es()[eid]['weight'] = weight 

    def feedback_arc_set(self, reverse_score=False):
        weights = None
        if reverse_score:
            weights = []
            for edge in self.directed_G.es:
                weights.append(
                    1 - self.copy().reverse_edges([edge.index]).get_score()
                    )
        return self.directed_G.feedback_arc_set(weights=weights, method='ip')

    def reverse_edges(self, eids=List[int]):
        self.directed_G.reverse_edges(eids)
        for eid in eids:
            self.set_weight(eid, self.reverse_weights[eid])

        return self

    def get_score(self):
        score = 0
        for edge in self.directed_G.es:
            score += edge['weight']   
        for uedge in self.undirected_edges:
            score += uedge[1] if uedge else 0
        num_edges = len(self.directed_G.es) + (len(self.undirected_edges) / 2)         
        return score / num_edges if num_edges > 0 else 0

    def remove_vertex(self, i):
        # removing directed node and edges
        self.directed_G.delete_vertices(i)
        updated_uedges = []
        for uedge in self.undirected_edges:
            edge = list(uedge[0])
            weight = uedge[1]
            # if the edge involves the node we remove it 
            if not i in edge:
               
                if edge[0] > i:
                    uedge = ((edge[0] - 1, edge[1]), weight)
                if edge[1] > i:
                    uedge = ((edge[0], edge[1] - 1), weight)
                updated_uedges.append(uedge)
        
        self.undirected_edges = updated_uedges
    
    def plot(self, filename):
        ig.plot(
            self.directed_G,
            target=filename,
            bbox=(0, 0, 1000, 1000),
            vertex_label=[v['name'] for v in self.directed_G.vs]
        )

    def to_nx(self):
        adj = self.get_adjacency()
        g = nx.DiGraph(adj)
        mapping = {i: v['name'] for i, v in enumerate(self.directed_G.vs)}
        g = nx.relabel_nodes(g, mapping)
        return g
    
    def copy(self):
        graph_copy = MixedGraph()
        graph_copy.directed_G = self.directed_G.copy()
        graph_copy.undirected_edges = self.undirected_edges.copy()
        if self.reverse_weights is not None:
            graph_copy.reverse_weights = self.reverse_weights.copy()
        return graph_copy
    
    def __eq__(self, other):    
        return (self.directed_G == other.directed_G and 
                self.undirected_edges == other.undirected_edges)

    def is_ciclic(self):
        return not self.directed_G.is_acyclic()
    
    def to_mpdag(self):
        # make copy of the graph
        G_mpdag = self.copy()
        # find cycles 
        sccs = G_mpdag.strongly_connected_components()
        for scc in sccs:
            # remove every direct edge in the cycle and add it as undirected edge
            if len(scc) > 1:
                for i in range(len(scc)):
                    for j in range(i + 1, len(scc)):
                        if G_mpdag.has_directed_edge(scc[i], scc[j]):
                            G_mpdag.remove_directed_edge(G_mpdag.get_eid(scc[i], scc[j]))
                        if G_mpdag.has_directed_edge(scc[j], scc[i]):
                            G_mpdag.remove_directed_edge(G_mpdag.get_eid(scc[j], scc[i]))
                        G_mpdag.add_undirected_edge(scc[i], scc[j])

        # G_mpdag.plot('mpdag_initial.png')

        for uedge in G_mpdag.undirected_edges:
            uedge = uedge[0]
            for node in G_mpdag.nodes():
                if node.index != uedge[0] and node.index != uedge[1]:
                    if (G_mpdag.has_directed_edge(uedge[0], node.index) and 
                            G_mpdag.has_directed_edge(node.index, uedge[1])):
                        G_mpdag.remove_undirected_edge(uedge)
                        G_mpdag.add_directed_edge(uedge[0], uedge[1])
                    
                    if (G_mpdag.has_directed_edge(node.index, uedge[0]) and 
                            G_mpdag.has_directed_edge(uedge[1], node.index)):
                        G_mpdag.remove_undirected_edge(uedge)
                        G_mpdag.add_directed_edge(uedge[1], uedge[0])

        # G_mpdag.plot('mpdag_final.png')
        return G_mpdag



                       