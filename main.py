from collections import OrderedDict
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
import datetime
import traceback
from itertools import product
import pandas as pd
import seaborn as sns
from tqdm import tqdm


class Main:
    def __init__(self, exp_per_state=10):
        self.exp_per_state = exp_per_state

    @staticmethod
    def random_c_matrix(source_level_size, target_level_size, capacity):
        C = np.random.rand(target_level_size, source_level_size)
        C = capacity / (np.sum(C)) * C
        return C

    @staticmethod
    def level_up(source_level, C):
        target_level = np.zeros((C.shape[0], source_level.shape[1]))
        for i in range(C.shape[0]):
            for j in range(C.shape[1]):
                # print(sum(C[:, j]))
                normalization = C[i, j] / max(sum(C[:, j]), sum(source_level[j]))  # sum(C[:, j]) * sum(source_level[j])
                target_level[i] += normalization * source_level[j]
        return target_level

    @staticmethod
    def base_initialization(base_layer_size, capacity, method='constant'):
        base = np.eye(base_layer_size) * (capacity / base_layer_size)
        # if method == 'constant':
        #     pass
        return base

    def simulate_flow(self, layer_sizes, capacity, initialization_method='constant'):
        current_level = self.base_initialization(base_layer_size=layer_sizes[-1], capacity=capacity,
                                                 method=initialization_method)
        for layer_size in layer_sizes[:0:-1]:
            C = main.random_c_matrix(source_level_size=current_level.shape[0],
                                     target_level_size=layer_size,
                                     capacity=capacity)
            current_level = main.level_up(source_level=current_level, C=C)
        total_flow_ratio = current_level.sum() / capacity
        min_commodity_ratio = current_level.sum(axis=0).min() * (layer_sizes[-1] / capacity)
        return total_flow_ratio, min_commodity_ratio

    def run_experiments(self, size_combination, capacity, initialization_method='constant'):
        total_flow_ratio_list = []
        min_ratio_list = []
        for _ in range(self.exp_per_state):
            total_flow, min_ratio = self.simulate_flow(layer_sizes=size_combination,
                                                       capacity=capacity,
                                                       initialization_method=initialization_method)
            total_flow_ratio_list.append(total_flow)
            min_ratio_list.append(min_ratio)
        total_flow_list = np.array(total_flow_ratio_list)
        min_ratio_list = np.array(min_ratio_list)
        return total_flow_list.mean(), total_flow_list.std(), min_ratio_list.mean(), min_ratio_list.std()

    def run(self, run_name, max_sizes, initialization_method='constant'):
        results = []
        size_ranges = [range(1, max_sizes[0] + 1)] + [range(2, max_size + 1) for max_size in max_sizes[1:]]
        for size_combination in product(*size_ranges):
            print("sizes: ", size_combination)
            exp_results = self.run_experiments(size_combination=size_combination,
                                               capacity=size_combination[-1],
                                               initialization_method=initialization_method)
            results.append(size_combination + exp_results)

        columns = [f'size_{i}' for i in range(1, len(max_sizes) + 1)] + \
                  ['flow_ratio_avg', 'flow_ratio_var', 'min_ratio_avg', 'min_ratio_var']
        results_df = pd.DataFrame(results, columns=columns)
        results_df.to_csv(f'{run_name}.csv', index=False)
        print(results_df)


if __name__ == '__main__':
    main = Main(exp_per_state=100)

    run_name = 'tryout'
    max_sizes = [1, 20, 20]
    initialization_method = 'constant'
    main.run(run_name=run_name, max_sizes=max_sizes, initialization_method=initialization_method)

    # t1 = np.array([[0, 1], [1, 0]]) * 0.3
    # C1 = np.array([[0.4, 0.4], [0.1, 0.3], [0.6, 0.2]])
    # C1 = main.random_c_matrix(source_level_size=t1.shape[0], target_level_size=3, capacity=2)
    # t2 = main.level_up(source_level=t1, C=C1)
    # print(t2)
    # C2 = np.array([[0.4, 0.4, 0.2], [0.2, 0.6, 0.2]])
    # C2 = main.random_c_matrix(source_level_size=t2.shape[0], target_level_size=2, capacity=2)
    # t3 = main.level_up(source_level=t2, C=C2)
    # print(t3)
    # C3 = np.array([[0.4, 0.2]])
    # C3 = main.random_c_matrix(source_level_size=t3.shape[0], target_level_size=1, capacity=2)
    # t4 = main.level_up(source_level=t3, C=C3)
    # print(t4)
    # print(t4.sum())
    # print(t4.sum(axis=0))
