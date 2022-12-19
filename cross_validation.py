import pickle
import statistics
import numpy as np
import pprint
import matplotlib.pyplot as plt
import random
import loglikes
import models
import mst_prototype

from mst_prototype import map2tree

pp = pprint.PrettyPrinter(compact=False, width=90)

# with open(f'__experiment_1/parsed_data/tree.pickle', 'rb') as handle:
#     TREE = pickle.load(handle)

# with open(f'__experiment_1/parsed_data/subject_decisions.pickle', 'rb') as handle:
#     # {sid: {world: {'nodes': [], 'path': []}}}
#     DECISIONS = pickle.load(handle)
          

def split_train_test_kfold(decisions_list, k=4):
    """ k = 4 """

    n = len(decisions_list) // 4  # length of each split
    splits = [decisions_list[i * n: (i + 1) * n] for i in range(4)]

    for i in range(k):
        train = sum([splits[j] for j in range(4) if j != i], [])
        test = splits[i]

        yield train, test


def split_train_test_rand(decisions_list, k=4):
    """ 75% train, 25% test """

    for _ in range(k):
        test = random.sample(decisions_list, k=len(decisions_list) // k)
        train = [d for d in decisions_list if d not in test]

        yield train, test


def model_preference(DECISIONS, all_trees, all_maps):
    """
    DECISIONS: { sid: {world: {'nodes': [], 'path': [] }}}
    TREE: { map: tree }

    Returns: model preference summary {model_name: number of subjects that prefer this model}
    """
    #sweep across different model parameters to find the best fit for a particular model = params to search 
    #models that are being considered

    # params: tau, gamma, beta, k

    model2parameters = {
        # 'Random_utility': [(tau, 1, 1, 1) for tau in models.TAUS], 
        'Expected_Utility': [(tau, 1, 1, 1) for tau in mst_prototype.TAUS],
        'Discounted_Utility': [(round(tau, 3), round(gamma, 3), 1, 1) for tau in mst_prototype.TAUS for gamma in mst_prototype.GAMMAS],
        'Probability_Weighted_Utility': [(round(tau, 3), 1, round(beta, 3), 1) for tau in mst_prototype.TAUS for beta in mst_prototype.BETAS],
        "Hyperbolic_Utility" : [(round(tau, 3), 1, 1, round(kappa, 3)) for tau in mst_prototype.TAUS for kappa in mst_prototype.KAPPAS],
        "Combined_Utility" : [ (round(tau, 3), round(gamma, 3), round(beta, 3), round(kappa, 3)) for tau in mst_prototype.TAUS for gamma in mst_prototype.GAMMAS for beta in mst_prototype.BETAS for kappa in mst_prototype.KAPPAS]
    }

    model_preference = {}  # {model_name: number of subjects that prefer this model}
    all_fit_parameters = {} # {model_name: list of all parameters that were fit from all subjects and all k tests}

    #DECISIONS = {subject1: {world1: [path through world 1 (using nid?)], world2:[path through world 2], ...}, subject2: {}, ... }
    for sid in DECISIONS:
        decisions_list = [] # decision list for single subject 
        max_avg_loglike = float('-inf')

        # collect all decisions made by subject sid
        for world in DECISIONS[sid]:
            decisions_list.extend((world, nid) for nid in DECISIONS[sid][world]['nodes']) 

        for model_name, parameters in model2parameters.items():
            print(model_name)
            avg_test_loglike, k = 0, 4 # what 
            for train, test in split_train_test_rand(decisions_list, k):  # splits into train and test data 
                max_loglike, mle_params = loglikes.mle(parameters, model_name, train, all_trees, all_maps) # calculate max path, and mle params for these parameters 
                avg_test_loglike += loglikes.loglike(mle_params, model_name, test, all_trees, all_maps) / k # test log likelihood
                if model_name not in all_fit_parameters:
                    all_fit_parameters[model_name] = []
                all_fit_parameters[model_name].append(mle_params)
            if avg_test_loglike > max_avg_loglike: 
                max_avg_loglike = avg_test_loglike
                best_model = model_name
            

        model_preference[best_model] = model_preference.get(best_model, 0) + 1

    median_params = process_parameters(all_fit_parameters)

    return model_preference, median_params


def process_parameters(all_fit_parameters):
    """
    returns the median parameters for each utility function 

    expected utility:  tau
    discounted utility: tau, gamma
    probability weighted: tau, beta
    hyperbolic: tau, kappa
    """
    # median_params = {
    #     "Expected_Utility": {'tau': 0},
    #     "Discounted_Utility": {'tau': 0, 'gamma': 0},
    #     "Probability_Weighted_Utility": {'tau': 0, 'beta': 0},
    #     "Hyperbolic_Utility": {'tau': 0, 'kappa': 0},
    #     # "Combined_Utility": {'tau': 0, 'gamma': 0, 'beta': 0, 'kappa': 0}
    # }

    median_params_dict = {
        "Expected_Utility": {},
        "Discounted_Utility": {},
        "Probability_Weighted_Utility": {},
        "Hyperbolic_Utility": {},
        "Combined_Utility": {'tau': 0, 'gamma': 0, 'beta': 0, 'kappa': 0}
    }

    for model, param_list in all_fit_parameters.items():
        # median_params = [statistics.median(list(j)) for j in zip(*param_list)]
        # tau, gamma, beta, kappa = median_params
        taus = [params[0] for params in param_list]
        gammas = [params[1] for params in param_list]
        betas = [params[2] for params in param_list]
        kappas = [params[3] for params in param_list]
        if model == "Expected_Utility":
            median_params_dict['Expected_Utility']['tau'] = statistics.median(taus)
        elif model == "Discounted_Utility":
            median_params_dict["Discounted_Utility"]['tau'] = statistics.median(taus)
            median_params_dict["Discounted_Utility"]['gamma'] = statistics.median(gammas)
        elif model == "Probability_Weighted_Utility":
            median_params_dict["Probability_Weighted_Utility"]['tau'] = statistics.median(taus)
            median_params_dict["Probability_Weighted_Utility"]['beta'] = statistics.median(betas)
        elif model == "Hyperbolic_Utility":
            median_params_dict["Hyperbolic_Utility"]['tau'] = statistics.median(taus)
            median_params_dict["Hyperbolic_Utility"]['kappa'] = statistics.median(kappas)
        elif model == "Combined_Utility":
            median_params_dict["Combined_Utility"]['tau'] = statistics.median(taus)
            median_params_dict["Combined_Utility"]['gamma'] = statistics.median(gammas)
            median_params_dict["Combined_Utility"]['beta'] = statistics.median(betas)
            median_params_dict["Combined_Utility"]['kappa'] = statistics.median(kappas)

    return median_params_dict


if __name__ == "__main__":
    model_preference()

