from typing import List

import yaml
import pickle
import pandas as pd
from networkx.drawing.nx_agraph import write_dot, read_dot
import networkx as nx

from src.utils.settings import TEXT_DATA_PATH, TABULAR_DATA_PATH, DATA_PATH, GRAPH_PATH


class Dataset:
    """ Abstract class for dataset """
    def __init__(self, name: str, filename: str = None, linear: bool = True, normal: bool = False) -> None:
        self.name = name
        self.filename = filename if filename is not None else name + '.csv'
        self.df_text = pd.read_csv(TEXT_DATA_PATH + self.filename)
        self.treatment = self.df_text[self.df_text['treatment'] == 1].var_description_english.values[0]
        self.target = self.df_text[self.df_text['target'] == 1].var_description_english.values[0]
        
        # print(self.treatment, self.target)
        linear = 'linear' if linear else 'non_linear'
        normal = 'normal' if normal else 'uniform'
        self.data = pd.read_csv(f'{TABULAR_DATA_PATH}{self.name}_{linear}_{normal}.csv')
        self.graph = nx.read_weighted_edgelist(f'{GRAPH_PATH}{self.name}.weighted.edgelist', delimiter=',', create_using=nx.DiGraph)

    def __getitem__(self, key):
        return self.df_text[key].values
    
    def __len__(self):
        return len(self.df_text)
    
    def var_description_lang(self, index: int, language: str = 'english') -> str:
        return self.df_text[f'var_description_{language}'][index]
    
    def to_dot(self):
        write_dot(self.graph, f'{GRAPH_PATH}{self.name}.dot')

    def compute_total_effect_from_scm(self):
        """
        Compute the total causal effect of X on Y in a linear SCM.

        Parameters:
        - G: A directed graph (DiGraph) where edges have weights representing direct effects.
        - X: The source node.
        - Y: The target node.
        - coefficients

        Returns:
        - total_effect: The sum of all causal path effects from X to Y.
        """
        total_effect = 0

        # Find all directed paths from X to Y
        all_paths = list(nx.all_simple_paths(self.graph, source=self.treatment, target=self.target))
       
        # Compute the effect along each path
        for path in all_paths:
            path_effect = 1  # Start with multiplicative identity
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                path_effect *= self.graph[u][v]['weight']
                # Multiply the edge weights along the path
            total_effect += path_effect  # Sum up contributions from all paths
        return total_effect

    def generate_triplets(self, enum: bool = False) -> List[List[str]]:
        # generate all triplets from variables description 
        triplets = []
        for i, var_i in enumerate(self.var_name):
            for j, var_j in enumerate(self.var_name):
                for k, var_k in enumerate(self.var_name):
                    if (var_i != var_j and var_i != var_k and var_j != var_k
                        and sorted([var_i, var_j, var_k]) not in triplets):
                        if enum:
                            triplets.append([i, j, k])
                        else:
                            triplets.append(sorted([var_i, var_j, var_k]))

        return triplets

    @property
    def var_description(self) -> List[str]:
        return self.df_text.var_description_english.values
    
    @property
    def var_name(self) -> List[str]:
        return self.df_text.var_name
