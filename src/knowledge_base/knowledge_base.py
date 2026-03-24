from src.knowledge_base.prompts import *

from typing import Tuple, List

import httpx
from tqdm import tqdm
import asyncio
import numpy as np
import networkx as nx

import re

from src.knowledge_base.backend import (
    OllamaBackend,
    OpenAIBackend
)
from src.utils import utils


class InconsistentKnowledgeBase:
    def __init__(
            self,
            model: str, # 
            context: str = '',
            temperature: float = 0.0,
            verbose: int = 0,
            seed: int = None
        ) -> None:
        super().__init__()
        print(f'Using model: {model}')
        if ('gpt' in model or 'o1' in model):
            self.backend = OpenAIBackend(model=model, seed=seed)
        else:
            self.backend = OllamaBackend(model=model, seed=seed)
        self.model = model
        self.context = context,
        self.verbose = verbose
        self.temperature = temperature  

    async def async_query(self, func, *args, **kwargs):
        for attempt in range(3):
            try:
                response = await func(*args, **kwargs)
            except (httpx.RemoteProtocolError, httpx.ReadTimeout) as e:
                print(f"Attempt {attempt + 1}: RemoteProtocolError - {e}")
                asyncio.sleep(2 ** attempt)
        return response
    
    def _query(self, func, *args, **kwargs):
        return func(*args, **kwargs)
    
    async def find_rephrasing(
        self,
        var_i: str,
        var_j: str,
        n_rephrase: int = 20
    ) -> List[str]:
        sentence = f"Does {var_i} causes {var_j}?"
        prompt = rephrasing_prompt.format(sentence=sentence, n_rephrase=n_rephrase)
        kwargs = {'temperature': self.temperature, 'max_tokens': 1000}
        if isinstance(self.backend, OpenAIBackend):
            response = await self.async_query(self.backend, prompt, **kwargs)
        else:
            response = self._query(self.backend, prompt, **kwargs)

        lines = response.strip().split('\n')
        sentences = [line.split('. ', 1)[1] for line in lines if '. ' in line]

        return sentences
    
    async def find_full_graph(
        self,
        var_names: List[str],
        descriptions: List[str],
        order: bool = False
    ):
        var_name_desc = ''
        for i in range(len(var_names)):
            var_name_desc += f'{var_names[i]}: {descriptions[i]}\n'
        if order:
            prompt = prompt_full_order.format(var_names=var_names, var_name_desc=var_name_desc)
        else:
            prompt = prompt_full_order.format(var_names=var_names, var_name_desc=var_name_desc)        
        kwargs = {'temperature': self.temperature, 'max_tokens': 10000}
        if isinstance(self.backend, OpenAIBackend):
            response = await self.async_query(self.backend, prompt, **kwargs)
        else:
            response = self._query(self.backend, prompt, **kwargs)
        response = re.search('<Answer>.*?</Answer>', response.replace('\n', ''))
        response = response.group(0)
        response = response.replace('<Answer>', '')
        response = response.replace('</Answer>', '')
        response = response.replace('}', '')
        response = response.replace('{', '')
        response = response.replace('\n', '')
        response = response.replace('\t', '')
        return eval(response)
    
    async def tripletwise(self, var_i: str, var_j: str, var_k: str) -> Tuple[Tuple[str], float]:
        prompt = prompt_triplets.format(var_i=var_i, var_j=var_j, var_k=var_k)
        kwargs = {'temperature': self.temperature, 'max_tokens': 1000, 'stopping_criteria': ['(A)', '(B)']}
        if isinstance(self.backend, OpenAIBackend):
            response = await self.async_query(self.backend, prompt, **kwargs)
        else:
            response = self._query(self.backend, prompt, **kwargs)
        if response.find('(A)') != -1:
            return 1
        else:
            return 0
        
    async def pairwise_rephrased(
        self,
        var_i: str,
        var_j: str,
        n_rephrase: int = 20
    ) -> Tuple[Tuple[str], float]:
        sentences = await self.find_rephrasing(var_i, var_j, n_rephrase=n_rephrase)
        # print(sentences)
        reply_counter = np.zeros(3)
        for sentence in sentences:
            prompt = yes_no_prompt.format(sentence=sentence)
            
            # prompt = self.context + prompt if self.context else prompt
            kwargs = {'temperature': self.temperature, 'max_tokens': 10, 'stopping_criteria': ['Yes', 'No']}
            if isinstance(self.backend, OpenAIBackend):
                response = await self.async_query(self.backend, prompt, **kwargs)
            else:
                response = self._query(self.backend, prompt, **kwargs)
            if response.find('Yes') != -1:
                reply_counter[0] += 1
            elif response.find('No') != -1:
                reply_counter[1] += 1
            else:
                reply_counter[2] += 1

        return (reply_counter[0] / len(sentences))
    
    async def pairwise_random(
        self,
        var_i: str,
        var_j: str,
        rep: int = 15,
    ) -> Tuple[Tuple[str], float]:
        sentence = f"{var_i} causes {var_j}"
        prompt = true_false_prompt.format(sentence=sentence)
        prompt = self.context + prompt if self.context else prompt
        reply_counter = np.zeros(3)      
        for _ in range(rep):
            kwargs = {'temperature': self.temperature, 'max_tokens': 10, 'stopping_criteria': ['True', 'False']}
            if isinstance(self.backend, OpenAIBackend):
                response = await self.async_query(self.backend, prompt, **kwargs)
            else:
                response = self._query(self.backend, prompt, **kwargs)
            if response.find('True') != -1:
                reply_counter[0] += 1
            elif response.find('False)') != -1:
                reply_counter[1] += 1
            else:
                reply_counter[2] += 1

        return (reply_counter[0] / rep)
    
    async def pairwise(self, var_i: str, var_j: str) -> Tuple[Tuple[str], float]:
        reply_counter = np.zeros(3)
        for verb in causal_verbs:
            prompt = prompt_pairwise.format(var_i=var_i, verb=verb, var_j=var_j)
            prompt = self.context + prompt if self.context else prompt
            kwargs = {'temperature': self.temperature, 'max_tokens': 10, 'stopping_criteria': ['True', 'False']}
            if isinstance(self.backend, OpenAIBackend):
                response = await self.async_query(self.backend, prompt, **kwargs)
            else:
                response = self._query(self.backend, prompt, **kwargs)
            if response.find('True') != -1:
                reply_counter[0] += 1
            elif response.find('False') != -1:
                reply_counter[1] += 1
            else:
                reply_counter[2] += 1
            if reply_counter[2] > 0:
                print(f'No answer found for {var_i} and {var_j} with verb {verb}. Response: {response}')

        print(f'{var_i} and {var_j} - {reply_counter[0] / len(causal_verbs)}')
            
        return (reply_counter[0] / len(causal_verbs))
    
    async def pairwise_yes_no(
        self,
        var_i: str,
        var_j: str,
        verb_k: str = 'causes'
    ) -> Tuple[Tuple[str], float]:
        reply_counter = np.zeros(3)
        reply_counter = np.zeros(3)
        for verb in causal_verbs:
            prompt = yes_no_pairwise_prompt.format(var_i=var_i, verb_k=verb_k, var_j=var_j)
            prompt = self.context + prompt if self.context else prompt
            kwargs = {'temperature': self.temperature, 'max_tokens': 10, 'stopping_criteria': ['Yes', 'No']}
            if isinstance(self.backend, OpenAIBackend):
                response = await self.async_query(self.backend, prompt, **kwargs)
            else:
                response = self._query(self.backend, prompt, **kwargs)
            if response.find('Yes') != -1:
                reply_counter[0] += 1
            elif response.find('No') != -1:
                reply_counter[1] += 1
            else:
                reply_counter[2] += 1

        return (reply_counter[0] / len(causal_verbs))
    
    async def independence_test(self, i: int, j: int, k: List[int], vars: List[str]) -> float:
        var_i, var_j = vars[i], vars[j]
        vars_k = [vars[idx] for idx in k] if k else None
        vars_k = ', '.join(vars_k) if vars_k else 'nothing'
        prompt = independence_test.format(var_i=var_i, var_j=var_j, vars_k=vars_k)
        kwargs = {'temperature': self.temperature, 'max_tokens': 1000, 'stopping_criteria': ['(A)', '(B)']}
        if isinstance(self.backend, OpenAIBackend):
            response = await self.async_query(self.backend, prompt, **kwargs)
        else:
            response = self._query(self.backend, prompt, **kwargs)
        if response.find('(A)') != -1:
            return 1
        else:
            return 0
        
    async def disambiguation(self, var_i: str, var_j: str) -> Tuple[str, str]:
        prompt = disambiguation.format(var_i=var_i, var_j=var_j)
        kwargs = {'temperature': self.temperature, 'max_tokens': 1000, 'stopping_criteria': ['(A)', '(B)']}
        if isinstance(self.backend, OpenAIBackend):
            response = await self.async_query(self.backend, prompt, **kwargs)
        else:
            response = self._query(self.backend, prompt, **kwargs)
     
        if response.find('(A)') != -1:
            return 1
        elif response.find('(B)') != -1:
            return 0
        else:
            ValueError('No answer found')
           
    async def triplet_orientation(
            self,
            var_i: str,
            var_j: str,
            var_k: str,
            var_names: List[str],
            descriptions: List[str]
    ) -> List[Tuple]:
        var_names = list(var_names)
        descriptions = list(descriptions)
        i, j, k = var_names.index(var_i), var_names.index(var_j), var_names.index(var_k)
        description_i, description_j, description_k = descriptions[i], descriptions[j], descriptions[k]
        prompt = triplet_orientation_CoT.format(
            context=descriptions,
            var_i=var_i, 
            var_j=var_j,
            var_k=var_k,
            description_i=description_i,
            description_j=description_j,
            description_k=description_k
        )
        kwargs = {'temperature': self.temperature, 'max_tokens': 10000, 'stopping_criteria': ['(A)', '(B)']}
        if isinstance(self.backend, OpenAIBackend):
            response = await self.async_query(self.backend, prompt, **kwargs)
        else:
            response = self._query(self.backend, prompt, **kwargs)
    
        result = re.search('<Answer>.*?</Answer>', response.replace('\n', ''))

        result = result.group(0)
        result = result.replace('<Answer>', '')
        result = result.replace('</Answer>', '')
        result = result.replace('}', '')
        result = result.replace('{', '')
        result = result.replace('\n', '')
        result = result.replace('\t', '')
        i = -1
        while result[i] == ' ':
            i -= 1
        if result[i] != ']':
            result = result + ']'
        if result[0] != '[':
            result = '[' + result
        # eval to cast
        return eval(result)
    

if __name__ == '__main__':
    from src.dataset.dataset import Dataset
    expert = InconsistentKnowledgeBase(model='gpt-4.1-mini', temperature=0.0)
    result = asyncio.run(expert.find_rephrasing('smoking', 'cancer', n_rephrase=10))
    print(result)