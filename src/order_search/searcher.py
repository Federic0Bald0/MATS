from typing import List, Tuple

import os, asyncio
import networkx as nx
import numpy as np
from tqdm import tqdm
from tqdm.contrib import itertools
from itertools import permutations

import src.utils.settings as settings
from src.dataset.dataset import Dataset
from src.knowledge_base.knowledge_base import InconsistentKnowledgeBase
import src.knowledge_base.prompts as prompts
from src.utils import utils


class Searcher:
    def __init__(
        self, 
        model: str,
        dataset: Dataset,
        temperature: float,
        triplets: bool = False,
        languages: List[str] = ['english'],
        seed: int = None,
        verbose: int = 1
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.dataset = dataset
        self.languages = languages
        self.verbose = verbose
        self.triplets = triplets
        self.seed = seed
        context = prompts.context.format(field=prompts.fields[self.dataset.name])

        self.uncertain_expert = InconsistentKnowledgeBase(
            model=model,
            temperature=temperature,
            context=context,
            seed=seed
        )
        self.consistency_matrix = np.zeros(
            (len(self.dataset.var_description),
             len(self.dataset.var_description)))

    async def _pairwise_consistency(self) -> float:
        n = len(self.dataset.var_description)
        iterator = itertools.product(range(n), range(n))
        iterator = tqdm(iterator) if self.verbose else iterator
        for i, j in tqdm(itertools.product(range(n), range(n))):
            if i != j:
                var_i = self.dataset.var_description[i]
                var_j = self.dataset.var_description[j]
                self.consistency_matrix[i, j] += await self.uncertain_expert.pairwise_rephrased(var_i, var_j)
                # print(self.consistency_matrix[i, j])
                # print(var_i, var_j)

    async def _triplets_consistency(self, triplets: List[Tuple[str, str, str]]) -> np.ndarray:    
        var_0 = self.dataset.var_name[0]
        n = len([1 for triplet in triplets if var_0 in triplet]) * 6 # *6 because of permutations
        descriptions = list(self.dataset.var_description)
        var_name = list(self.dataset.var_name)
        for triplet in tqdm(triplets):
            triplet_index = [var_name.index(var) for var in triplet]
            for triplet_permutation in permutations(triplet_index):
                triplet_permutation_desc = [descriptions[i] for i in triplet_permutation]
                reply = await self.uncertain_expert.tripletwise(*triplet_permutation_desc)
                if reply == 1:
                    i, j, k = triplet_permutation
                    self.consistency_matrix[i, j] += 1
                    self.consistency_matrix[j, k] += 1
                    self.consistency_matrix[j, k] += 1
        self.consistency_matrix = self.consistency_matrix / n

    async def _build_consistency_matrix(self):
        # name filename the consistency table
        query_type = 'triplets' if self.triplets else 'pairwise'
        filename = f'{settings.CONSISTENCY_MATRIX_PATH}{self.dataset.name}'
        filename += f'_{self.model}'
        filename += f'_{query_type}'
        filename += f'_{self.temperature}'
        if self.seed is not None:
            print(self.seed)
            filename += f'_{self.seed}'
        filename += '.npy'
        # check if already computed
        if os.path.exists(filename):
            self.consistency_matrix = np.load(filename)
        else:
            print(f'Building consistency matrix')
            if self.triplets:
                triplets = self.dataset.generate_triplets()
                await self._triplets_consistency(triplets)

            else:
                await self._pairwise_consistency()
            np.save(filename, self.consistency_matrix)

    def search(self):
            pass   