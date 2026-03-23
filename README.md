# Retrieving Classes of Causal Orders with Inconsistent Knowledge Bases
Traditional causal discovery methods often depend on strong, untestable assumptions, making them unreliable in real-world applications.
In this context, Large Language Models (LLMs) have emerged as a promising alternative for extracting causal knowledge from text-based metadata, effectively consolidating domain expertise.
However, LLMs are prone to hallucinations, necessitating strategies that account for these limitations.
One effective approach is to use a consistency measure as a proxy of reliability. 
Moreover, LLMs do not clearly distinguish direct from indirect causal relationships, complicating the discovery of causal Directed Acyclic Graphs (DAGs), which are often sparse.
This ambiguity is evident in the way informal sentences are formulated in various domains. 
For this reason, focusing on causal orders provides a more practical and direct task for LLMs.
We propose a new method for deriving abstractions of causal orders that maximizes a consistency score obtained from an LLM.
Our approach begins by computing pairwise consistency scores between variables, from which we construct a semi-complete partially directed graph that consolidates these scores into an abstraction. 
Using this structure, we identify both a maximally oriented partially directed acyclic graph and an optimal set of acyclic tournaments that maximize consistency across all configurations.
We further demonstrate how both the abstraction and the class of causal orders can be used to estimate causal effects.
We evaluate our method on a wide set of causal DAGs extracted from scientific literature in epidemiology and public health.
Our results show that the proposed approach can effectively recover the correct causal order, providing a reliable and practical LLM-assisted causal framework.
