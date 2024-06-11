import datetime
import traceback
import numpy as np
from itertools import product
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from dag import DAG


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

    main.run(run_name=run_name, max_sizes=max_sizes, static_capacities=static_capacities,
             initialization_method=initialization_method, initialization_factor=initialization_factor)
