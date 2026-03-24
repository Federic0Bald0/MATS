from typing import List

import csv, asyncio
import sys, time
import argparse

import numpy as np 

import networkx as nx


from src.order_search import MATS
from src.dataset.dataset import Dataset
from src.dataset.ground import *
from src.utils import utils
from src.reasoning.total_effect import *
from cdt.metrics import SHD
import matplotlib.pyplot as plt


def input_parser(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        type=str,
        default='cancer'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4.1'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7
    )   
    parser.add_argument(
        '--triplets',
        type=bool,
        default=False
    )
    parser.add_argument(
        '--discovery',
        type=bool, 
        default=False,
    )
    parser.add_argument(
        '--seed',
        type=int, 
        default=None,
    )
    
    return parser.parse_args()

async def main(args):
    dataset = Dataset(args.dataset, linear=True, normal=False)
    true_graph = dataset.graph
    print('Model:', args.model)
    # searcher
    searcher = MATS(
        model=args.model,
        dataset=dataset,
        temperature=args.temperature,
        triplets=args.triplets,
        seed=args.seed,
    )

    # search
    t0 = time.time()
    causal_orders, bidirected_nodes, semi_complete_graph = await searcher.search()
    t1 = time.time()
    causal_orders = [order.to_nx() for order in causal_orders]  

    # evaluation
    print(args.temperature)
    results = {
        'method': 'mats',
        'dataset': args.dataset,
        'model': args.model,
        'topo_error': None,
        'shd': None,
        'time': t1 - t0,
        'temperature': args.temperature,
        'ciclic': np.any([not nx.is_directed_acyclic_graph(order) for order in causal_orders]),
        'triplets': args.triplets,
        'seed': args.seed,
    }
    
    # causal orders
    topo_error, shd = utils.eval_causal_order(true_graph, causal_orders)
    results['topo_error'] = topo_error
    results['shd'] = shd

    # causal graphs
    DAGs = utils.discover_with_orders(
        dataset=dataset,
        causal_orders=causal_orders,
        indep_test='fisherz',
    )

    _, shd = utils.eval_causal_order(true_graph, DAGs)
    results['shd'] = shd

    with open(f'results/causal_orders_results.csv', 'a') as f:
        csv.writer(f).writerow(results.values())

    # total effect estimation 
    true_total_effect = dataset.compute_total_effect_from_scm()
    print('True total effect:', true_total_effect)

    # mpdag 
    mpdag = semi_complete_graph.to_mpdag()
    # mpdag.plot('mpdag.png')
    
    estimate_effect_mpdag = total_effect_mpdag(mpdag, dataset, dataset.treatment, dataset.target)
    print('Estimated total effect from MPDAG:', estimate_effect_mpdag)
    estimate_effect_orders, adjustment_set, non_identifiable = total_effect_orders(causal_orders, dataset, dataset.treatment, dataset.target)
    print('Estimated total effect from causal orders:', estimate_effect_orders)
    print('Adjustment sets:', adjustment_set)
    print('Non-identifiable orders:', non_identifiable)
    # absolute error
    if estimate_effect_mpdag is not None:
        abs_error_mpdag = abs(true_total_effect - estimate_effect_mpdag)
        print('Absolute error from MPDAG:', abs_error_mpdag)
    else:
        abs_error_mpdag = None
        print('MPDAG is not identifiable')

    for key, value in estimate_effect_orders.items():
        abs_error = abs(true_total_effect - value[0])
        print(f'Absolute error from causal order {key}:', abs_error)


if __name__ == '__main__':

    args = input_parser(sys.argv[1:])
    asyncio.run(main(args))


