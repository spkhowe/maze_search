import pickle
import numpy as np
import pprint
import matplotlib.pyplot as plt
import mst_prototype
from mst_prototype import map2tree, node_values, calculate_random_utility, calculate_combined_utility, calculate_expected_utility, calculate_discounted_utility, calculate_pwu, calculate_novel_utility
import models

pp = pprint.PrettyPrinter(compact=False, width=90)


def combine_all_trees(all_maps):
    """
    all_maps: dict with map name as keys and map as values 
    """
    all_trees = {}
    for name, map_ in all_maps.items():
        all_trees[name] = map2tree(map_)
    return all_trees


def combine_path_node_values(all_maps, model_name, params):
    """
    For the given model, combines node values for all maps 
    refactors for an output that is readily accessible by loglike()

    input: maps, params, model
    return: node values = {'map': {'pid': {'nid': value}}}
    """

    models = {
        'Random_utility': calculate_random_utility,
        'Expected_Utility': calculate_expected_utility,
        'Discounted_Utility': calculate_discounted_utility,
        'Probability_Weighted_Utility': calculate_pwu,
        "Hyperbolic_Utility": calculate_novel_utility,
        "Combined_Utility": calculate_combined_utility
    }

    nodevalue_function = models[model_name]

    refactored_node_values = {}
    for name, map_ in all_maps.items():
        values = node_values(map_, params, nodevalue_function)
        refactored_node_values[name] = values
    return refactored_node_values


def loglike(params, model_name, decisions_list, all_trees, all_maps):
    """
    decisions_list = [(world, nid), ...]
    return average loglike for sid for all decisions is world is None
    if world is specified, return average loglike for decisions made in that world

    * note from steph: i think ignore what they're saying about specifying world. i think its just:
    return average loglike for sid for all decisions, which calculates the probability of a model for a person
    """
    # Decisions {'P1': {'Map2': {'path': [(5, 4), (5, 2), (3, 4)], 'nodes': [0, 1, 4]}}}

    # with open(f'__experiment_1/node_values/{model_name}/node_values_{params}.pickle', 'rb') as handle:
    #     # {world: {pid: {nid: node value}}}
    #     node_values = pickle.load(handle)

    node_values = combine_path_node_values(all_maps, model_name, params)

    cum_loglike = 0

    for world, nid in decisions_list:
        pid = all_trees[world][nid]['pid']
        # root node has no value
        if pid == 'NA' or pid is None:
            continue

        # parent node is not a decision node
        if len(all_trees[world][pid]['children']) <= 1:
            continue

        cum_loglike += np.log(node_values[world][pid][nid])
        # node values = {'map': {'pid': {'nid': value}}}

    return cum_loglike / len(decisions_list)


def mle(parameters, model_name, decisions_list, all_trees, all_maps):
    """ maximum likelihood estimation """

    max_loglike = float('-inf')

    for params in parameters:

        avg_loglike = loglike(params, model_name, decisions_list, all_trees, all_maps)

        if avg_loglike > max_loglike:
            max_loglike = avg_loglike
            mle_params = params

    return max_loglike, mle_params


def model_fitting(parameters, model_name, DECISIONS):
    # {sid: list of decision nodes}
    sid2decisions = {}

    # {sid: (max_loglike, mle_params)}
    sid2mle = {}

    for sid in DECISIONS:
        for world in DECISIONS[sid]:
            sid2decisions.setdefault(sid, []).extend([(world, nid) for nid in DECISIONS[sid][world]['nodes']])

        sid2mle[sid] = mle(parameters, model_name, sid2decisions[sid])

    _, axs = plt.subplots(1, 3)
    axs = axs.flat

    axs[0].hist([max_ll for max_ll, _ in sid2mle.values()], edgecolor='white')
    axs[1].hist([params[0] for _, params in sid2mle.values()], edgecolor='white')
    axs[2].hist([params[2] for _, params in sid2mle.values()], edgecolor='white')

    plt.show()


if __name__ == "__main__":
    sid = 'S99991343'
    parameters = [(tau, 1, 1) for tau in models.TAUS]
    model_name = 'Expected_Utility'

    parameters = [(round(tau, 3), round(gamma, 3), 1) for tau in models.TAUS for gamma in models.GAMMAS]
    model_name = 'Discounted_Utility'

    parameters = [(round(tau, 3), 1, round(gamma, 3)) for tau in models.TAUS for gamma in models.BETAS]
    model_name = 'Probability_Weighted_Utility'

    # loglike(sid, parameters[0], model_name)
    # mle(sid, parameters, model_name)
    # model_fitting(parameters, model_name)