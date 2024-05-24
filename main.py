from collections import OrderedDict
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
import numpy as np
import datetime
import traceback
from itertools import product
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm


def level_up(source_level, C):
    target_level = np.zeros((C.shape[0], source_level.shape[1]))
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            # print(sum(C[:, j]))
            normalization = C[i, j] / max(sum(C[:, j]), sum(source_level[j]))  # sum(C[:, j]) * sum(source_level[j])
            target_level[i] += normalization * source_level[j]
    return target_level


def random_c_matrix(source_level_size, target_level_size, capacity):
    C = np.random.rand(target_level_size, source_level_size)
    C = capacity / (np.sum(C)) * C
    return C


t1 = np.array([[0, 1], [1, 0]])
# C1 = np.array([[0.4, 0.4], [0.1, 0.3], [0.6, 0.2]])
C1 = random_c_matrix(source_level_size=t1.shape[0], target_level_size=3, capacity=2)
t2 = level_up(source_level=t1, C=C1)
print(t2)
C2 = random_c_matrix(source_level_size=t2.shape[0], target_level_size=2, capacity=2)
t3 = level_up(source_level=t2, C=C2)
print(t3)
C3 = random_c_matrix(source_level_size=t3.shape[0], target_level_size=1, capacity=2)
t4 = level_up(source_level=t3, C=C3)
print(t4)


# print(np.sum(t4, axis=0))

class Main:
    def __init__(self, exp_per_state=10):
        self.exp_per_state = exp_per_state

    def calculate_max_flow(self, size_intervals, max_capacity_per_level,
                           initialization_method='efficient', initialization_factor='normal'):
        # initialization_list = []
        max_flow_list = []
        flow_ratio_list = []
        for _ in range(self.exp_per_state):
            dag = DAG(level_size_intervals=size_intervals,
                      max_capacity_per_level=max_capacity_per_level)
            _, t_f, f_r = dag.calculate_max_flow(level_index=-1,
                                                 initialization_method=initialization_method,
                                                 initialization_factor=initialization_factor)
            # initialization_list.append(l_in)
            max_flow_list.append(t_f)
            flow_ratio_list.append(f_r)
            # max_flow_list.append(dag.calculate_max_flow())

        max_flow_list = np.array(max_flow_list)
        flow_ratio_list = np.array(flow_ratio_list)
        return max_flow_list.mean(), max_flow_list.std(), flow_ratio_list.mean(), flow_ratio_list.std()

    def run(self, run_name, max_sizes, static_capacities, initialization_method='efficient',
            initialization_factor='normal'):
        results = []
        size_ranges = [range(1, max_sizes[0] + 1)] + [range(2, max_size + 1) for max_size in max_sizes[1:]]
        for size_combination in product(*size_ranges):
            print("sizes: ", size_combination)
            exp_results = self.calculate_max_flow(size_intervals=size_combination,
                                                  max_capacity_per_level=static_capacities,
                                                  initialization_method=initialization_method,
                                                  initialization_factor=initialization_factor)
            results.append(size_combination + exp_results)

        columns = [f'size_{i}' for i in range(1, len(max_sizes) + 1)] + \
                  ['flow_avg', 'flow_var', 'flow_ratio_avg', 'flow_ratio_var']
        results_df = pd.DataFrame(results, columns=columns)
        results_df.to_csv(f'{run_name}.csv', index=False)


if __name__ == '__main__':
    main = Main(exp_per_state=100)

    run_name = 'ms_2020_sc_1010_im_ncn_if_n'  # ms_2020_sc_1010_im_e_if_n, ms_2020_sc_1010_im_ncn_if_n
    max_sizes = [1, 20, 20]
    static_capacities = [10, 10]
    initialization_method = 'node_constant_normalized'  # efficient, node_constant, node_constant_normalized
    initialization_factor = 'normal'  # normal, binary, uniform
