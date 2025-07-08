#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import numpy as np
import os
import pandas as pd
import physbo


exp_results_path = './exp_results.csv'
candidates_path  = './candidates.csv'
bayesopt_results_dir = './bayesopt_results'
search_score = 'EI'
random_search_num = 3
bayes_search_num = 17

# generate dir
if not os.path.isdir(bayesopt_results_dir):
    os.makedirs(bayesopt_results_dir)


def get_data_from_result_list(index):
    with open(exp_results_path) as f:
        reader = csv.reader(f)
        l = [row for row in reader]

    return float(l[index+1][0])


def generate_save_dir(i):
    each_exp_save_dir = os.path.join(bayesopt_results_dir, 'exp_' + str(i))
    if not os.path.isdir(each_exp_save_dir):
        os.mkdir(each_exp_save_dir)

    return each_exp_save_dir


def save_candidates_and_actions(data, policy, each_exp_save_dir):
    candidates_each_exp_path = os.path.join(each_exp_save_dir, 'candidates.csv')
    chosen_actions_each_exp_path = os.path.join(each_exp_save_dir, 'chosen_actions.csv')

    ## chosen_actions
    chosen_actions = policy.history.chosen_actions[:policy.history.total_num_search]
    df_chosen_actions = pd.Series(chosen_actions)
    df_chosen_actions.to_csv(chosen_actions_each_exp_path, index=False, header=False)
    ## candidates
    for i,a in enumerate(chosen_actions):
        data["y"].iloc[a] = policy.training.t[i]
        data.to_csv(candidates_each_exp_path, index=False)


def main():
    chosen_actions = []

    # load_dataset
    data = pd.read_csv(candidates_path)
    X = data.iloc[:,:-1].to_numpy()
    actions = list(data[data["y"] < np.inf].index)

    # generate policy
    policy = physbo.search.discrete.policy(X)

    # search
    for i in range(random_search_num):
        next_action = policy.random_search(max_num_probes=1)
        exp_result = get_data_from_result_list(next_action[0])
        policy.write(next_action, exp_result)
        physbo.search.utility.show_search_results(policy.history, 10)
        # save
        each_exp_save_dir = generate_save_dir(i)
        save_candidates_and_actions(data, policy, each_exp_save_dir)

    for i in range(bayes_search_num):
        next_action = policy.bayes_search(max_num_probes=1, score=search_score)
        exp_result = get_data_from_result_list(next_action[0])
        policy.write(next_action, exp_result)
        physbo.search.utility.show_search_results(policy.history, 10)
        # save
        each_exp_save_dir = generate_save_dir(i+random_search_num)
        save_candidates_and_actions(data, policy, each_exp_save_dir)


if __name__ == '__main__':
    main()
