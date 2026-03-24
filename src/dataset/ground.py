import pickle
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

from src.utils.settings import TEXT_DATA_PATH, GRAPH_PATH

class GroundGraph:
    def __init__(self, name: str):
        self.name = name
        self.G = None

    def graph(self) -> nx.DiGraph:
        return self.G
    
    def save(self):
        nx.write_weighted_edgelist(self.G, f'{GRAPH_PATH}{self.name}.weighted.edgelist', delimiter=',')

        # with open(f'{GRAPH_PATH}{self.name}.pkl', 'wb') as f:
        #     pickle.dump(self.G, f, pickle.HIGHEST_PROTOCOL)
    
class Covid1(GroundGraph):
    def __init__(self, name='covid_1'):
        super().__init__(name)
        self.G = nx.DiGraph()
        self.df = pd.read_csv(TEXT_DATA_PATH + f'{name}.csv')
        nodes = self.df.var_description_english.values
        self.G.add_nodes_from(nodes)
        self.G.add_edge('age bracket of a subject', 'voluntarily using data collection of epidemiological data via mobile phone apps')
        self.G.add_edge('higher risk of being infected with covid-19', 'voluntarily using data collection of epidemiological data via mobile phone apps')
    

class Covid2(GroundGraph):
    def __init__(self, name='covid_2'):
        super().__init__(name)
        self.G = nx.DiGraph()
        self.df = pd.read_csv(TEXT_DATA_PATH + f'{name}.csv')
        nodes = self.df.var_description_english.values
        self.G.add_nodes_from(nodes)
        self.G.add_edge('being a healthcare worker', 'smoking')
        self.G.add_edge('being a healthcare worker', 'being tested for covid-19')
        self.G.add_edge('severity of covid-19 infection', 'being tested for covid-19')


class Covid3(GroundGraph):
    def __init__(self, name='covid_3'):
        super().__init__(name)
        self.G = nx.DiGraph()
        self.df = pd.read_csv(TEXT_DATA_PATH + f'{name}.csv')
        nodes = self.df.var_description_english.values
        self.G.add_nodes_from(nodes)
        self.G.add_edge('higher susceptibility to develop severe covid-19 infection', 'hospitalization')
        self.G.add_edge('higher susceptibility to develop severe covid-19 infection', 'death')
        self.G.add_edge('using ace inhibitors', 'hospitalization')
        self.G.add_edge('using ace inhibitors', 'death')
        self.G.add_edge('hospitalization', 'death')


class Covid4(GroundGraph):
    def __init__(self, name='covid_4'):
        super().__init__(name)
        self.G = nx.DiGraph()
        self.df = pd.read_csv(TEXT_DATA_PATH + f'{name}.csv')
        nodes = self.df.var_description_english.values
        self.G.add_nodes_from(nodes)
        self.G.add_edge('prevalence of diabetes', 'infection fatality rate or infected hospitalisation rate')
        self.G.add_edge('proportion of population over 60', 'prevalence of diabetes')
        self.G.add_edge('proportion of population over 60', 'covid-19 incidence')
        self.G.add_edge('proportion of population over 60', 'infection fatality rate or infected hospitalisation rate')
        self.G.add_edge('covid-19 incidence', 'infection fatality rate or infected hospitalisation rate')
        self.G.add_edge('number of intensive care beds per inhabitants', 'infection fatality rate or infected hospitalisation rate')


class Genetic(GroundGraph):
    def __init__(self, name='genetic'):
        super().__init__(name)
        self.G = nx.DiGraph()
        self.df = pd.read_csv(TEXT_DATA_PATH + f'{name}.csv')
        nodes = self.df.var_description_english.values
        self.G.add_nodes_from(nodes)
        self.G.add_edge('gene FTO', 'fat mass')
        self.G.add_edge('gene MC4R', 'fat mass')
        self.G.add_edge('gene TMEM18', 'fat mass')
        self.G.add_edge('gene GNPDA2', 'fat mass')
        self.G.add_edge('fat mass', 'bone mineral density')

    
class MobileStrokeUnit(GroundGraph):
    def __init__(self, name='msu'):
        super().__init__(name)
        self.G = nx.DiGraph()
        self.df = pd.read_csv(TEXT_DATA_PATH + f'{name}.csv')
        nodes = self.df.var_description_english.values
        self.G.add_nodes_from(nodes)
        self.G.add_edge('dispatched mobile stroke unit', 'time to thrombolysis')
        self.G.add_edge('time to thrombolysis', 'level of functional independence with reference to pre-stroke activities')
        self.G.add_edge('stroke severity', 'time to thrombolysis')
        self.G.add_edge('stroke severity', 'level of functional independence with reference to pre-stroke activities')
        self.G.add_edge('systolic blood pressure', 'time to thrombolysis')
        self.G.add_edge('systolic blood pressure', 'level of functional independence with reference to pre-stroke activities')

    
class Neighborhood(GroundGraph):
    def __init__(self, name='neighborhood'):
        super().__init__(name)
        self.G = nx.DiGraph()
        self.df = pd.read_csv(TEXT_DATA_PATH + f'{name}.csv')
        nodes = self.df.var_description_english.values
        self.G.add_nodes_from(nodes)
        self.G.add_edge('low accessibility to green spaces', 'low socioeconomic position neighborhood')
        self.G.add_edge('low accessibility to green spaces', 'reduced walkability of the neighborhood')
        self.G.add_edge('low socioeconomic position neighborhood', 'exposure to high criminality levels')
        self.G.add_edge('low socioeconomic position neighborhood', 'lack of services')
        self.G.add_edge('proximity to pollution industry', 'reduced walkability of the neighborhood')
        self.G.add_edge('proximity to pollution industry', 'low socioeconomic position neighborhood')
        self.G.add_edge('lack of services', 'reduced walkability of the neighborhood')
        self.G.add_edge('exposure to high criminality levels', 'reduced walkability of the neighborhood')


class Opioids(GroundGraph):
    def __init__(self, name='opioids'):
        super().__init__(name)
        self.G = nx.DiGraph()
        self.df = pd.read_csv(TEXT_DATA_PATH + f'{name}.csv')
        nodes = self.df.var_description_english.values
        self.G.add_nodes_from(nodes)
        self.G.add_edge('age', 'chronic pain')
        self.G.add_edge('marital status', 'chronic pain')
        self.G.add_edge('smoking', 'chronic pain')
        self.G.add_edge('alcohol intake', 'chronic pain')
        self.G.add_edge('anti-depressant medication prescription', 'chronic pain')
        self.G.add_edge('sex', 'chronic pain')
        self.G.add_edge('education level', 'chronic pain')
        self.G.add_edge('poverty income ratio', 'chronic pain')
        self.G.add_edge('health insurance coverage', 'chronic pain')
        self.G.add_edge('age', 'mortality')
        self.G.add_edge('marital status', 'mortality')
        self.G.add_edge('smoking', 'mortality')
        self.G.add_edge('alcohol intake', 'mortality')
        self.G.add_edge('anti-depressant medication prescription', 'mortality')
        self.G.add_edge('sex', 'mortality')
        self.G.add_edge('education level', 'mortality')
        self.G.add_edge('poverty income ratio', 'mortality')
        self.G.add_edge('health insurance coverage', 'mortality')
        self.G.add_edge('chronic pain', 'mortality')    
        self.G.add_edge('chronic pain', 'propoxyphene use')
        self.G.add_edge('chronic pain', 'codeine use')
        self.G.add_edge('chronic pain', 'oxycodone use')
        self.G.add_edge('chronic pain', 'hydrocodone use')
        self.G.add_edge('chronic pain', 'tramadol use')
        self.G.add_edge('chronic pain', 'morphine use')
        self.G.add_edge('chronic pain', 'fentanyl use')
        self.G.add_edge('chronic pain', 'pentazocine use')
        self.G.add_edge('chronic pain', 'dihydrocodeine use')
        self.G.add_edge('chronic pain', 'hydromorphone use')
        self.G.add_edge('chronic pain', 'peperidine use')
        self.G.add_edge('chronic pain', 'methadone use')
        self.G.add_edge('chronic pain', 'pentazocine use')
        self.G.add_edge('chronic pain', 'oxymorphone use')
        self.G.add_edge('chronic pain', 'tapentadol use')
        self.G.add_edge('codeine use', 'mortality')
        self.G.add_edge('oxycodone use', 'mortality')
        self.G.add_edge('hydrocodone use', 'mortality')
        self.G.add_edge('tramadol use', 'mortality')
        self.G.add_edge('morphine use', 'mortality')
        self.G.add_edge('fentanyl use', 'mortality')
        self.G.add_edge('pentazocine use', 'mortality')
        self.G.add_edge('dihydrocodeine use', 'mortality')
        self.G.add_edge('hydromorphone use', 'mortality')
        self.G.add_edge('peperidine use', 'mortality')
        self.G.add_edge('methadone use', 'mortality')
        self.G.add_edge('pentazocine use', 'mortality')
        self.G.add_edge('oxymorphone use', 'mortality')
        self.G.add_edge('tapentadol use', 'mortality')
        self.G.add_edge('age', 'propoxyphene use')
        self.G.add_edge('age', 'codeine use')
        self.G.add_edge('age', 'oxycodone use')
        self.G.add_edge('age', 'hydrocodone use')
        self.G.add_edge('age', 'tramadol use')
        self.G.add_edge('age', 'morphine use')
        self.G.add_edge('age', 'fentanyl use')
        self.G.add_edge('age', 'pentazocine use')
        self.G.add_edge('age', 'dihydrocodeine use')
        self.G.add_edge('age', 'hydromorphone use')
        self.G.add_edge('age', 'peperidine use')
        self.G.add_edge('age', 'methadone use')
        self.G.add_edge('age', 'pentazocine use')
        self.G.add_edge('age', 'oxymorphone use')
        self.G.add_edge('age', 'tapentadol use')
        self.G.add_edge('marital status', 'propoxyphene use')
        self.G.add_edge('marital status', 'codeine use')
        self.G.add_edge('marital status', 'oxycodone use')
        self.G.add_edge('marital status', 'hydrocodone use')
        self.G.add_edge('marital status', 'tramadol use')
        self.G.add_edge('marital status', 'morphine use')
        self.G.add_edge('marital status', 'fentanyl use')
        self.G.add_edge('marital status', 'pentazocine use')
        self.G.add_edge('marital status', 'dihydrocodeine use')
        self.G.add_edge('marital status', 'hydromorphone use')
        self.G.add_edge('marital status', 'peperidine use')
        self.G.add_edge('marital status', 'methadone use')
        self.G.add_edge('marital status', 'pentazocine use')
        self.G.add_edge('marital status', 'oxymorphone use')
        self.G.add_edge('marital status', 'tapentadol use')
        self.G.add_edge('smoking', 'propoxyphene use')
        self.G.add_edge('smoking', 'codeine use')
        self.G.add_edge('smoking', 'oxycodone use')
        self.G.add_edge('smoking', 'hydrocodone use')
        self.G.add_edge('smoking', 'tramadol use')
        self.G.add_edge('smoking', 'morphine use')
        self.G.add_edge('smoking', 'fentanyl use')
        self.G.add_edge('smoking', 'pentazocine use')
        self.G.add_edge('smoking', 'dihydrocodeine use')
        self.G.add_edge('smoking', 'hydromorphone use')
        self.G.add_edge('smoking', 'peperidine use')
        self.G.add_edge('smoking', 'methadone use')
        self.G.add_edge('smoking', 'pentazocine use')
        self.G.add_edge('smoking', 'oxymorphone use')
        self.G.add_edge('smoking', 'tapentadol use')
        self.G.add_edge('alcohol intake', 'propoxyphene use')
        self.G.add_edge('alcohol intake', 'codeine use')
        self.G.add_edge('alcohol intake', 'oxycodone use')
        self.G.add_edge('alcohol intake', 'hydrocodone use')
        self.G.add_edge('alcohol intake', 'tramadol use')
        self.G.add_edge('alcohol intake', 'morphine use')
        self.G.add_edge('alcohol intake', 'fentanyl use')
        self.G.add_edge('alcohol intake', 'pentazocine use')
        self.G.add_edge('alcohol intake', 'dihydrocodeine use')
        self.G.add_edge('alcohol intake', 'hydromorphone use')
        self.G.add_edge('alcohol intake', 'peperidine use')
        self.G.add_edge('alcohol intake', 'methadone use')
        self.G.add_edge('alcohol intake', 'pentazocine use')
        self.G.add_edge('alcohol intake', 'oxymorphone use')
        self.G.add_edge('alcohol intake', 'tapentadol use')
        self.G.add_edge('anti-depressant medication prescription', 'propoxyphene use')
        self.G.add_edge('anti-depressant medication prescription', 'codeine use')
        self.G.add_edge('anti-depressant medication prescription', 'oxycodone use')
        self.G.add_edge('anti-depressant medication prescription', 'hydrocodone use')
        self.G.add_edge('anti-depressant medication prescription', 'tramadol use')
        self.G.add_edge('anti-depressant medication prescription', 'morphine use')
        self.G.add_edge('anti-depressant medication prescription', 'fentanyl use')
        self.G.add_edge('anti-depressant medication prescription', 'pentazocine use')
        self.G.add_edge('anti-depressant medication prescription', 'dihydrocodeine use')
        self.G.add_edge('anti-depressant medication prescription', 'hydromorphone use')
        self.G.add_edge('anti-depressant medication prescription', 'peperidine use')
        self.G.add_edge('anti-depressant medication prescription', 'methadone use')
        self.G.add_edge('anti-depressant medication prescription', 'pentazocine use')
        self.G.add_edge('anti-depressant medication prescription', 'oxymorphone use')
        self.G.add_edge('anti-depressant medication prescription', 'tapentadol use')
        self.G.add_edge('sex', 'propoxyphene use')
        self.G.add_edge('sex', 'codeine use')
        self.G.add_edge('sex', 'oxycodone use')
        self.G.add_edge('sex', 'hydrocodone use')
        self.G.add_edge('sex', 'tramadol use')
        self.G.add_edge('sex', 'morphine use')
        self.G.add_edge('sex', 'fentanyl use')
        self.G.add_edge('sex', 'pentazocine use')
        self.G.add_edge('sex', 'dihydrocodeine use')
        self.G.add_edge('sex', 'hydromorphone use')
        self.G.add_edge('sex', 'peperidine use')
        self.G.add_edge('sex', 'methadone use')
        self.G.add_edge('sex', 'pentazocine use')
        self.G.add_edge('sex', 'oxymorphone use')
        self.G.add_edge('sex', 'tapentadol use')     
        self.G.add_edge('education level', 'propoxyphene use')
        self.G.add_edge('education level', 'codeine use')
        self.G.add_edge('education level', 'oxycodone use')
        self.G.add_edge('education level', 'hydrocodone use')
        self.G.add_edge('education level', 'tramadol use')
        self.G.add_edge('education level', 'morphine use')
        self.G.add_edge('education level', 'fentanyl use')
        self.G.add_edge('education level', 'pentazocine use')
        self.G.add_edge('education level', 'dihydrocodeine use')
        self.G.add_edge('education level', 'hydromorphone use')
        self.G.add_edge('education level', 'peperidine use')
        self.G.add_edge('education level', 'methadone use')
        self.G.add_edge('education level', 'pentazocine use')
        self.G.add_edge('education level', 'oxymorphone use')
        self.G.add_edge('education level', 'tapentadol use')
        self.G.add_edge('poverty income ratio', 'propoxyphene use')
        self.G.add_edge('poverty income ratio', 'codeine use')
        self.G.add_edge('poverty income ratio', 'oxycodone use')
        self.G.add_edge('poverty income ratio', 'hydrocodone use')
        self.G.add_edge('poverty income ratio', 'tramadol use')
        self.G.add_edge('poverty income ratio', 'morphine use')
        self.G.add_edge('poverty income ratio', 'fentanyl use')
        self.G.add_edge('poverty income ratio', 'pentazocine use')
        self.G.add_edge('poverty income ratio', 'dihydrocodeine use')
        self.G.add_edge('poverty income ratio', 'hydromorphone use')
        self.G.add_edge('poverty income ratio', 'peperidine use')
        self.G.add_edge('poverty income ratio', 'methadone use')
        self.G.add_edge('poverty income ratio', 'pentazocine use')
        self.G.add_edge('poverty income ratio', 'oxymorphone use')
        self.G.add_edge('poverty income ratio', 'tapentadol use')
        self.G.add_edge('health insurance coverage', 'propoxyphene use')
        self.G.add_edge('health insurance coverage', 'codeine use')
        self.G.add_edge('health insurance coverage', 'oxycodone use')
        self.G.add_edge('health insurance coverage', 'hydrocodone use')
        self.G.add_edge('health insurance coverage', 'tramadol use')
        self.G.add_edge('health insurance coverage', 'morphine use')
        self.G.add_edge('health insurance coverage', 'fentanyl use')
        self.G.add_edge('health insurance coverage', 'pentazocine use')
        self.G.add_edge('health insurance coverage', 'dihydrocodeine use')
        self.G.add_edge('health insurance coverage', 'hydromorphone use')
        self.G.add_edge('health insurance coverage', 'peperidine use')
        self.G.add_edge('health insurance coverage', 'methadone use')
        self.G.add_edge('health insurance coverage', 'pentazocine use')
        self.G.add_edge('health insurance coverage', 'oxymorphone use')
        self.G.add_edge('health insurance coverage', 'tapentadol use')


class Supermarket(GroundGraph):
    def __init__(self, name='supermarket'):
        super().__init__(name)
        self.G = nx.DiGraph()
        self.df = pd.read_csv(TEXT_DATA_PATH + f'{name}.csv')
        nodes = self.df.var_description_english.values
        self.G.add_nodes_from(nodes)
        self.G.add_edge('level of education and income in the a neighborhood', 'supermarket characteristics')
        self.G.add_edge('level of education and income in the a neighborhood', 'weight status and adiposity')
        self.G.add_edge('household income', 'level of education and income in the a neighborhood')
        self.G.add_edge('level of education', 'level of education and income in the a neighborhood')
        self.G.add_edge('household income', 'supermarket characteristics')
        self.G.add_edge('household income', 'purchased food items')
        self.G.add_edge('level of education', 'household income')
        self.G.add_edge('level of education', 'food preferences')
        self.G.add_edge('food preferences', 'purchased food items')
        self.G.add_edge('purchased food items', 'weight status and adiposity')
        self.G.add_edge('supermarket characteristics', 'purchased food items')
        self.G.add_edge('food preferences', 'supermarket characteristics')


class Cancer(GroundGraph):
    def __init__(self, name='cancer'):
        super().__init__(name)
        self.G = nx.DiGraph()
        self.df = pd.read_csv(TEXT_DATA_PATH + f'{name}.csv')
        nodes = self.df.var_description_english.values
        self.G.add_nodes_from(nodes)
        self.G.add_edge('level of pollution', 'lung cancer')
        self.G.add_edge('smoking', 'lung cancer')
        self.G.add_edge('lung cancer', 'x-ray')
        self.G.add_edge('lung cancer', 'laboured breathing')

class Asia(GroundGraph):
    def __init__(self, name='asia'):
        super().__init__(name)
        self.G = nx.DiGraph()
        self.df = pd.read_csv(TEXT_DATA_PATH + f'{name}.csv')
        nodes = self.df.var_description_english.values
        self.G.add_nodes_from(nodes)
        self.G.add_edge('visiting asia', 'tuberculosis')
        self.G.add_edge('tuberculosis', 'individual has either tuberculosis or lung cancer')
        self.G.add_edge('smoking cigarettes', 'lung cancer')
        self.G.add_edge('smoking cigarettes', 'bronchitis')
        self.G.add_edge('lung cancer', 'individual has either tuberculosis or lung cancer')
        self.G.add_edge('bronchitis', "dyspnoea or laboured breathing ")
        self.G.add_edge('individual has either tuberculosis or lung cancer', "dyspnoea or laboured breathing ")
        self.G.add_edge('individual has either tuberculosis or lung cancer', 'positive x-ray')


class Climate(GroundGraph):
    def __init__(self, name='climate'): 
        super().__init__(name)
        self.G = nx.DiGraph()
        self.df = pd.read_csv(TEXT_DATA_PATH + f'{name}.csv')
        nodes = self.df.var_description_english.values
        self.G.add_nodes_from(nodes)
        self.G.add_edge('temperature', 'relative humidity')
        self.G.add_edge('temperature', 'transmission rate of a disease')
        self.G.add_edge('dew point temperature', 'relative humidity')
        self.G.add_edge('relative humidity', 'transmission rate of a disease')
        self.G.add_edge('transmission rate of a disease', 'rate of incidence of a disease in the population')
        self.G.add_edge('rate of incidence of a disease in the population', 'observed incidence rate in the population')
        self.G.add_edge('rate of infection of a disease in the population', 'rate of incidence of a disease in the population')
        self.G.add_edge('size of the susceptible population', 'rate of incidence of a disease in the population')


class Sachs(GroundGraph):
    def __init__(self, name='sachs'):
        super().__init__(name)
        self.G = nx.DiGraph()
        self.df = pd.read_csv(TEXT_DATA_PATH + f'{name}.csv')
        nodes = self.df.var_description_english.values
        self.G.add_nodes_from(nodes)
        self.G.add_edge('protein kinase C', 'phosphorylation of protein kinase A')
        self.G.add_edge('protein kinase C', 'RAF kinase')
        self.G.add_edge('protein kinase C', 'C-Jun N-terminal kinases (Jnk)')
        self.G.add_edge('protein kinase C', 'p38 mitogen-activated protein kinases')
        self.G.add_edge('protein kinase C', 'mitogen-activated protein kinase kinase (Mek)')
        self.G.add_edge('phosphorylation of protein kinase A', 'RAF kinase')
        self.G.add_edge('phosphorylation of protein kinase A', 'C-Jun N-terminal kinases (Jnk)')
        self.G.add_edge('phosphorylation of protein kinase A', 'p38 mitogen-activated protein kinases')
        self.G.add_edge('phosphorylation of protein kinase A', 'mitogen-activated protein kinase kinase (Mek)')
        self.G.add_edge('phosphorylation of protein kinase A', 'extracellular signal-regulated kinases (Erk)')
        self.G.add_edge('phosphorylation of protein kinase A', 'protein kinase B (Akt)')
        self.G.add_edge('RAF kinase', 'mitogen-activated protein kinase kinase (Mek)')
        self.G.add_edge('mitogen-activated protein kinase kinase (Mek)', 'extracellular signal-regulated kinases (Erk)')
        self.G.add_edge('extracellular signal-regulated kinases (Erk)', 'protein kinase B (Akt)')
        self.G.add_edge('phospholipase C', "phosphatidylinositol 4;5-bisphosphate")
        self.G.add_edge('phospholipase C', "phosphatidylinositol 3;4;5-triphosphate")
        self.G.add_edge("phosphatidylinositol 3;4;5-triphosphate", "phosphatidylinositol 4;5-bisphosphate")


class Child(GroundGraph):
    def __init__(self, name='child'):
        self.name = name
        self.G = nx.DiGraph()
        self.df = pd.read_csv(TEXT_DATA_PATH + f'{name}.csv')
        nodes = self.df.var_description_english.values
        self.G.add_nodes_from(nodes)
        self.G.add_edge("lack of oxygen to the blood during the infant's birth", "infant methemoglobinemia")
        self.G.add_edge("infant methemoglobinemia", "presence of an illness")
        self.G.add_edge("infant methemoglobinemia", "blood flow across the ductus arteriosus")
        self.G.add_edge("infant methemoglobinemia", "mixing of oxygenated and deoxygenated blood")
        self.G.add_edge("infant methemoglobinemia", "the state of the blood vessels in the lungs")
        self.G.add_edge("infant methemoglobinemia", "low blood flow in the lungs")
        self.G.add_edge("infant methemoglobinemia", "thickening of the left ventricle")
        self.G.add_edge("infant methemoglobinemia", "age of infant at disease presentation")
        self.G.add_edge("presence of an illness", "grunting in infants")
        self.G.add_edge("presence of an illness", "low oxygen areas equally distributed around the body")
        self.G.add_edge("blood flow across the ductus arteriosus", "low oxygen areas equally distributed around the body")
        self.G.add_edge("mixing of oxygenated and deoxygenated blood", "low oxygen areas equally distributed around the body")
        self.G.add_edge("mixing of oxygenated and deoxygenated blood", "hypoxia when breathing oxygen")
        self.G.add_edge("the state of the blood vessels in the lungs", "grunting in infants")
        self.G.add_edge("the state of the blood vessels in the lungs", "hypoxia when breathing oxygen")
        self.G.add_edge("the state of the blood vessels in the lungs", "level of CO2 in the body")
        self.G.add_edge("the state of the blood vessels in the lungs", "having a chest x-ray")
        self.G.add_edge("thickening of the left ventricle", "report of having LVH")
        self.G.add_edge("grunting in infants", "report of infant grunting")
        self.G.add_edge("low oxygen areas equally distributed around the body", "level of oxygen in the lower body")
        self.G.add_edge("hypoxia when breathing oxygen", "level of oxygen in the lower body")
        self.G.add_edge("hypoxia when breathing oxygen", "level of oxygen in the right up quadricep muscule")
        self.G.add_edge("level of CO2 in the body", "a document reporting high level of CO2 levels in blood")
        self.G.add_edge("having a chest x-ray", "lung excessively filled with blood")
        

if __name__ == '__main__':
    Asia().save()
    Cancer().save()
    Child().save()
    Climate().save()
    Covid1().save()
    Covid2().save()
    Covid3().save()
    Covid4().save()
    Genetic().save()
    MobileStrokeUnit().save()
    Neighborhood().save()
    Opioids().save()
    Sachs().save()
    Supermarket().save()
