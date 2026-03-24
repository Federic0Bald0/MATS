
from typing import List, Union, Tuple

import asyncio
import itertools    
import numpy as np

from src.order_search.searcher import Searcher
from src.utils.graph import MixedGraph
from src.dataset.dataset import Dataset
from src.knowledge_base.knowledge_base import InconsistentKnowledgeBase
from src.utils import utils


INF_WEIGHT = -1e8


class MATS(Searcher):
    def __init__(
        self, 
        model: str,
        dataset: Dataset,
        temperature: float,
        triplets: bool = False,
        languages: List[str] = ['english'],
        seed: int = None,
        verbose: int = 1    
    ):
        super().__init__(model, dataset, temperature, triplets, languages, seed, verbose)
        self.nodes_name = [node_name for node_name in self.dataset.var_description]
        self.nodes_idx = [i for i in range(len(self.nodes_name))]

    def build_maximal_weighted_graph(self) -> MixedGraph:
        # graph that has maximum weight for each edge (can be cyclic)
        G = MixedGraph()
        G.add_nodes(self.nodes_name)
        n = len(self.dataset.var_description)
        for i, node_i in enumerate(self.nodes_name):
            for j, node_j in enumerate(self.nodes_name):
                if node_i == node_j:
                    continue
                if self.consistency_matrix[i, j] == self.consistency_matrix[j, i]:
                    G.add_undirected_edge(i, j, weight=self.consistency_matrix[i, j])
                elif self.consistency_matrix[i, j] > self.consistency_matrix[j, i]:
                    G.add_directed_edge(
                        node_i,
                        node_j, 
                        weight=self.consistency_matrix[i, j],
                        reverse_weight=self.consistency_matrix[j, i])
                else:
                    G.add_directed_edge(
                        node_j,
                        node_i,
                        weight=self.consistency_matrix[j, i],
                        reverse_weight=self.consistency_matrix[i, j])
        return G
    
    @staticmethod
    def find_bidirected_nodes(G: MixedGraph) -> List[str]:
        # nodes for which consistency is always simetric 
        # (all edges for and form this node are bidirectional)
        bidirected_nodes = []    
        for i in G.nodes():
            if G.degree(i) == 0:
                bidirected_nodes.append(i)
        return bidirected_nodes
    
    def connect_singletons(self, G: MixedGraph, sccs: List[List[str]]):
        # components with one element (singletons), can be connected to 
        # to any other component considering the single edge connecting 
        # to the other scc
        singeltons = [scc for scc in sccs if len(scc) == 1]
        for (singleton, scc) in itertools.product(singeltons, sccs):
            if singleton == scc:
                continue
            i = singleton[0]
            j, edges = 0, []
            while j < len(scc):
                if G.has_directed_edge(i, j):    
                    edges = [
                        (i, j, self.consistency_matrix[i, j], self.consistency_matrix[j, i])
                        for j in scc]
                elif G.has_directed_edge(j, i): 
                    edges = [
                        (j, i, self.consistency_matrix[j, i], self.consistency_matrix[i, j])
                        for j in scc]
                    break
                j += 1
            G.add_directed_edges(edges)

    def direct_graphs(self, G: MixedGraph) -> List[MixedGraph]:
        # directed al the undirected edges outputing a class of compatible graphs 
        # with the same score
        # G.plot('S.png')
        queue = [G]
        cyclic_graphs = []
        while queue:
            G = queue.pop()
            #
            if G.undirected_edges == []:
                cyclic_graphs.append(G)
                continue
            edge = G.undirected_edges[0][0]
            G.remove_undirected_edge(edge)

            edge = list(edge)
            
            if not G.has_directed_edge(*edge):
                edge_attr = (
                    edge[0],
                    edge[1],
                    self.consistency_matrix[edge[0], edge[1]],
                    self.consistency_matrix[edge[1], edge[0]]
                )
                new_G = G.copy()
                new_G.add_directed_edge(*edge_attr)
                if new_G.is_dag():
                    queue.append(new_G)
                    
            if not G.has_directed_edge(*edge[::-1]):
                edge_attr = (
                    edge[1],
                    edge[0],
                    self.consistency_matrix[edge[1], edge[0]],
                    self.consistency_matrix[edge[0], edge[1]]
                )
                new_G = G.copy()
                new_G.add_directed_edge(*edge_attr)
                if new_G.is_dag():
                    queue.append(new_G)

        directed_graphs = []
        for g in cyclic_graphs:
            if g.is_dag():
                directed_graphs.append(g)

        return directed_graphs    
    
    # def find_best_minimal_feedback_arc_set(self, G: MixedGraph, scc: List[int]) -> List[int]:

    #     # find the best feedback arc sets for a strongly connected component
    #     G_scc = G.subgraph(scc)
    #     print(f'Finding best minimal feedback arc set for scc: {scc}')
    #     # find best and maximal fas (reference for the search of equivalent solutions)
    #     best_minimal_fas = [G_scc.feedback_arc_set(reverse_score=True)]
    #     best_score = G_scc.copy().reverse_edges(best_minimal_fas[0]).get_score()
    #     subsets_fas = utils.subsets(best_minimal_fas[0])
    #     # for each each subsest of the best fas, we check if there are equivalent solution 
    #     # if the edges are excluded by the search of the fas
    #     while subsets_fas:
    #         subset_fas = subsets_fas.pop()
    #         G_scc_copy = G_scc.copy()
    #         for edge in subset_fas:
    #             # set to infinte weight edges that are excluded from the search
    #             G_scc_copy.set_weight(edge, INF_WEIGHT, reverse=True)
    #         temptative_fas = G_scc_copy.feedback_arc_set(reverse_score=True)
    #         score = G_scc_copy.reverse_edges(temptative_fas).get_score()
    #         # if it's an equivalent solution and it leads to the best minimal fas,
    #         # then add the partition set of the new fas to the subset to evaluate
    #         if score == best_score and temptative_fas not in best_minimal_fas:
    #             best_minimal_fas.append(temptative_fas)
    #             subset_tempatative_fas = utils.subsets(temptative_fas)
    #             for subset_tempatative_fas in subset_tempatative_fas:
    #                 if subset_tempatative_fas not in subsets_fas:
    #                     subsets_fas.append(subset_tempatative_fas.union(subset_fas))
    #         else:
    #             # remove all subsets that include the current subset
    #             subsets_fas = [subset for subset in subsets_fas if not set(subset_fas).issubset(set(subset))]
    #     # map subgraph edges to graph edges (this is needed since we are working with sccs)
    #     reindexed_best_minimal_fas = []
    #     for fas in best_minimal_fas:
    #         reindexed_fas = []    
    #         for i, edge in enumerate(fas):
    #             edge = G_scc.directed_G.es[edge]
    #             edge = (edge.source, edge.target) 
    #             source_name = G_scc.directed_G.vs[edge[0]]['name']
    #             target_name = G_scc.directed_G.vs[edge[1]]['name']
    #             # source_idx_G = self.nodes_name.index(source_name)
    #             # target_idx_G = self.nodes_name.index(target_name)
    #             edge_idx_G = G.get_eid(source_name, target_name)
    #             reindexed_fas.append(edge_idx_G)
    #         reindexed_best_minimal_fas.append(reindexed_fas)

    #     return reindexed_best_minimal_fas

    def find_best_minimal_feedback_arc_set(self, G: MixedGraph, scc: List[int]) -> List[int]:

        # find the best feedback arc sets for a strongly connected component
        G_scc = G.subgraph(scc)
        # find best and maximal fas (reference for the search of equivalent solutions)
        best_minimal_fas = [G_scc.feedback_arc_set(reverse_score=True)]
        best_score = G_scc.copy().reverse_edges(best_minimal_fas[0]).get_score()
        i = 1
        subsets_fas = utils.partitions_of_size_n(best_minimal_fas[0], i)
        suboptimal_fas = []
        # for each each subsest of the best fas, we check if there are equivalent solution 
        # if the edges are excluded by the search of the fas
        
        while subsets_fas:
            subset_fas = subsets_fas.pop()
            G_scc_copy = G_scc.copy()
            for edge in subset_fas:
                # set to infinte weight edges that are excluded from the search
                G_scc_copy.set_weight(edge, INF_WEIGHT, reverse=True)
            temptative_fas = G_scc_copy.feedback_arc_set(reverse_score=True)
            score = G_scc_copy.reverse_edges(temptative_fas).get_score()
            # if it's an equivalent solution and it leads to the best minimal fas,
            # then add the partition set of the new fas to the subset to evaluate
            if score == best_score and temptative_fas not in best_minimal_fas:
                best_minimal_fas.append(temptative_fas)
                subset_tempatative_fas = utils.subsets(temptative_fas)
                for subset_tempatative_fas in subset_tempatative_fas:
                    if subset_tempatative_fas not in subsets_fas:
                        subsets_fas.append(subset_tempatative_fas.union(subset_fas))
            else:
                # remove all subsets that include the current subset
                subsets_fas = [subset for subset in subsets_fas if not set(subset_fas).issubset(set(subset))]
                suboptimal_fas.append(set(subset_fas))

            if np.any([len(subset) > i for subset in subsets_fas]):
                i += 1
                for fas in best_minimal_fas:
                    subsets = (utils.partitions_of_size_n(best_minimal_fas[0], i))
                    for subset in subsets:
                        if subset not in subsets_fas and set(subset) not in suboptimal_fas:
                            subsets_fas.append(subset.union(fas))
        # map subgraph edges to graph edges (this is needed since we are working with sccs)
        reindexed_best_minimal_fas = []
        for fas in best_minimal_fas:
            reindexed_fas = []    
            for i, edge in enumerate(fas):
                edge = G_scc.directed_G.es[edge]
                edge = (edge.source, edge.target) 
                source_name = G_scc.directed_G.vs[edge[0]]['name']
                target_name = G_scc.directed_G.vs[edge[1]]['name']
                # source_idx_G = self.nodes_name.index(source_name)
                # target_idx_G = self.nodes_name.index(target_name)
                edge_idx_G = G.get_eid(source_name, target_name)
                reindexed_fas.append(edge_idx_G)
            reindexed_best_minimal_fas.append(reindexed_fas)

        return reindexed_best_minimal_fas
    
    async def search(self, mpdag: bool = False) -> Tuple[Union[List[MixedGraph], MixedGraph], List[str]]:
    
        # finds the set of graphs with maximum consistency that are valid causal orders
        # build Maximal Graph 
        await self._build_consistency_matrix()
        print(self.consistency_matrix)

        G = self.build_maximal_weighted_graph()
        semi_complte_graph = G.copy()        
        # find and remove edges that are connected just by undirected edges
        bidirected_nodes_names = []
        # bidirected_nodes = list(reversed(self.find_bidirected_nodes(G)))
        # bidirected_nodes_names = [self.nodes_name[node.index] for node in bidirected_nodes]

        if mpdag:
            return G, bidirected_nodes_names
        
        # find acyclic sccs that maximaze the consistency
        sccs = list(G.strongly_connected_components())
        non_singletons = [scc for scc in sccs if len(scc) > 1]
        best_minimal_fas_sccs = []
        for scc in non_singletons:
            best_minimal_fas = self.find_best_minimal_feedback_arc_set(G, scc)
            print('here')
            best_minimal_fas_sccs.append(best_minimal_fas)

        # find fas
        reverse_arc_sets = []
        if len(non_singletons) > 1:
            for best_minimal_fas in best_minimal_fas_sccs:
                reverse_arc_sets.append(list(itertools.product(*best_minimal_fas)))
            reverse_arc_sets = [list(reverse_arc_set) for reverse_arc_set in itertools.product(*reverse_arc_sets)]
            reverse_arc_sets = [[[edge[0] for edge in reverse_arc_set] for reverse_arc_set in reverse_arc_sets]]
        else:
            reverse_arc_sets = best_minimal_fas_sccs

        # reverse edges
        acyclic_graphs = [] if not G.is_dag() else [G]
        for reverse_arc_set in reverse_arc_sets:
            for edges in reverse_arc_set:
                G_copy = G.copy()
                G_copy.reverse_edges(edges)                    
                acyclic_graphs.append(G_copy)
        
        # direct undirected edges
        dags = []
        for G in acyclic_graphs:
            dags += self.direct_graphs(G)

        return dags, bidirected_nodes_names, semi_complte_graph
        
    
    




        