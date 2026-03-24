languages = [
    'english',
    'italian',
    'french',
    'chinese'
]

# prompts_pairwise = {
#     "english":  """
#                 {var_i} {verb} {var_j}:
#                 (A) True
#                 (B) False
#                 The answer is:""",
#     "italian":  """{var_i} {verb} {var_j}:
#                 (A) Vero
#                 (B) Falso
#                 La risposta è:""",
#     "french":  """{var_i} {verb} {var_j}:
#                 (A) Vrai
#                 (B) Faux
#                 La réponse est:""",
#     "chinese":  """{var_i} {verb} {var_j}:
#                 (A) 真
#                 (B) 假
#                 答案是:"""
# }

prompt_pairwise = """
                {var_i} {verb} {var_j}:
                (A) True
                (B) False
                The answer is:
                """

prompt_triplets = """
                {var_i} causes {var_j} that causes {var_k}
                (A) True
                (B) False
                The answer is:
                """

prompt_full_graph = """
    Given the following variables: 
    {var_names}
    Where:
    {var_name_desc}
    Privide the causal graph only in the form of a list of tuples, where each tuple is a directed edge.
    The full answer should be between the tags <Answer></Answer>.
""" 

prompt_full_order = """
    Given the following variables:
    {var_names}
    Where:
    {var_name_desc}
    Provide the causal order only in the form of a list of tuples, where each tuple is a directed edge.
    The full answer should be between the tags <Answer></Answer>.
"""


prompt_cause_effect = """
    Identify the causal relationships between the given variables and create a directed acyclic graph. 
    Make sure to give a reasoning for your answer and then output the directed graph
    in the form of a list of tuples, where each tuple is a directed edge. The desired output should
    be in the following form: ('A','B') where first tuple represents a directed edge from
    Node 'A' to Node 'B', second tuple represents a directed edge from Node 'B' to Node 'C'and
    so on.
    Use the description about the node provided with the nodes in brackets to form a better decision
    about the causal direction orientation between the nodes.
    It is very important that you output the final Causal graph within the tags <Answer></Answer>otherwise your answer will not be processed.
    Do not absolutely change the description of the nodes provided in the brackets. This might cause errors in the final output.
    Example: 
    Input: Nodes: 'A', 'B';
    Output: <Answer>('B','A')</Answer>
    Question: 
    Input: Nodes: {x} {y} 
    Output:
"""

causal_verbs = [
    'causes',
    'results in',
    'provokes',
    'leads to',
    'results in',
    'sparks',
    'is responsible for',
    'triggers',
    'brings about',
    'yields',
    'induces',
    'produces',
    'generates',
    # 'prompts',
    # 'instigates',
    # 'intiates',
    # 'generates',
    # 'catalsyzes',
    # 'engenders',
    # 'incites'
]

# causal_verbs = {
#     "english": [
#         'causes',
#         'provokes',
#         'leads to',
#         'results in',
#         'induces',
#         'triggers',
#         'yields',
#         'brings about',
#         'produces',
#         'generates',
#         'sparks',
#         'prompts',
#         'instigates',
#         'intiates',
#         'generates',
#         'catalsyzes',
#         'engenders',
#         'incites'
#     ],
#     "italian": [
#         'causa',
#         'provoca',
#         'porta a',
#         'risulta in',
#         'induce',
#         'scatena',
#         'rende',
#         'produce',
#         'genera',
#         'instiga',
#         'inizia',
#         'genera',
#         'catalizza',
#         'incita'
#     ],
#     "french": [
#         'cause',
#         'provoque',
#         'mène à',
#         'résulte en',
#         'induit',
#         'déclenche',
#         'rend',
#         'produit',
#         'génère',
#         'instigue',
#         'initie',
#         'génère',
#         'catalyse',
#         'incite'
#     ],
#     "chinese": [
#         '导致',
#         '引起',
#         '导致',
#         '导致',
#         '诱导',
#         '触发',
#         '产生',
#         '导致',
#         '产生',
#         '产生',
#         '引发',
#         '促使',
#         '引发',
#         '引发'
#     ]
# }

############################################################################################################
###                     PROMPT FOR INDEPENDENCE TESTING                                                  ###
############################################################################################################

independence_test = """
    You will be asked to provide your estimate
    on statistical independence between two variables
    (eventually conditioned on a set of variables).\n
    Your answer should not be based on data or observations, 
    but only on the available knowledge.\n
    Even when unsure or uncertain, provide a valid answer.\n
    Answer only in the required format.\n
    {var_i} is independent {var_j} conditional on {vars_k}:
    (A) True
    (B) False
    The answer is:
"""

############################################################################################################
###                     PROMPT FOR TRIPLETS ORIENTATION                                                  ###
############################################################################################################

triplet_orientation_CoT = """  
    Identify the causal relationships between the given variables and create a directed acyclic graph
    to {context}. Make sure to give a reasoning for your answer and then output the directed graph
    in the form of a list of tuples, where each tuple is a directed edge. The desired output should
    be in the following form: [('A','B'), ('B','C')] where first tuple represents a directed edge from
    Node 'A' to Node 'B', second tuple represents a directed edge from Node 'B' to Node 'C'and
    so on.
    If a node should not form any causal relationship with other nodes, then you can add it as an
    isolated node of the graph by adding it seperately. For example, if 'C' should be an isolated
    node in a graph with nodes 'A', 'B', 'C', then the final DAG representation should be like
    [('A','B'), ('C')].
    Use the description about the node provided with the nodes in brackets to form a better decision
    about the causal direction orientation between the nodes.
    It is very important that you output the final Causal graph within the tags <Answer></Answer>otherwise your answer will not be processed.
    Example:
    Input: Nodes: ['A', 'B', 'C', 'D'];
    Description of Nodes: [(description of Node A), (description of Node B), (description of Node
    C), (description of Node D)]
    Output: <Answer>[('A','B'),('C','D')]</Answer>
    Question:
    Input: Nodes: {var_i}, {var_j}, {var_k};
    Description of Nodes: [{description_i}, {description_j}, {description_k}]
    Output:
"""

triplet_orientation = """  
    Identify the causal relationships between the given variables and create a directed acyclic graph
    to {context}. The output provided should be in the form of a list of tuples, where each tuple 
    is a directed edge.
    For example [('A','B'), ('B','C')] where first tuple represents a directed edge from
    Node 'A' to Node 'B', second tuple represents a directed edge from Node 'B' to Node 'C'and
    so on.
    If a node should not form any causal relationship with other nodes, then you can add it as an
    isolated node of the graph by adding it seperately. For example, if 'C' should be an isolated
    node in a graph with nodes 'A', 'B', 'C', then the final DAG representation should be like
    [('A','B'), ('C')].
    Use the description about the node provided with the nodes in brackets to form a better decision
    about the causal direction orientation between the nodes.
    It is very important that you output the final Causal graph within the tags <Answer></Answer>otherwise your answer will not be processed.
    Example:
    Input: Nodes: ['A', 'B', 'C', 'D'];
    Description of Nodes: [(description of Node A), (description of Node B), (description of Node
    C), (description of Node D)]
    Output: <Answer>[('A','B'),('C','D')]</Answer>
    Question:
    Input: Nodes: {var_i}, {var_j}, {var_k};
    Description of Nodes: [{description_i}, {description_j}, {description_k}]
    Output:
"""

disambiguation = """
    Given the following variables, select the variable that is more likely to cause the other.
    (A) {var_i} causes {var_j}
    (B) {var_j} causes {var_i}
    The answer is:
"""

rephrasing_prompt = """
    Provide me with {n_rephrase} rephrased versions of the following sentence: {sentence}
    The rephrased sentences should preserve the semantic meaning, even if absurd, of the original one.
    The answers should be in the following format:
    1. <rephrased sentence 1>
    2. <rephrased sentence 2>
    3. <rephrased sentence 3>
    """

true_false_prompt = """
    {sentence}
    (A) True
    (B) False
    The answer is:
    """

yes_no_prompt = """
    {sentence}
    (A) Yes
    (B) No          
    The answer is:
    """

yes_no_pairwise_prompt = """
    Does {var_i} {verb_k} {var_j}?
    (A) Yes
    (B) No
    The answer is:
    """

# CONTEXT 
context = """
    You are an expert in the field of {field}.
    The task is to provide causal relationships between the variables.
    Keep your answers concise and to the point.
    """

fields = {
    'asia': "tubercolosis and disease spreading",
    'cancer': "cancer",
    'climate': "epidemiology",
    'child': "child development and health",
    'covid_1': "public health and epidemiology",
    'covid_2': "public health and epidemiology",
    'covid_3': "public health and epidemiology",
    'covid_4': "public health and epidemiology",
    'genetic': "genetics",
    'msu': "public health",
    'neighborhood': "public health",
    'opioids': "public health",
    'sachs': "biology",
    'supermarket': "public health"
}

