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

    def calculate_max_flow(self, size_intervals, max_capacity_per_level):
        max_flow_list = []
        for _ in range(self.exp_per_state):
            dag = DAG(level_size_intervals=size_intervals,
                      max_capacity_per_level=max_capacity_per_level)
            max_flow_list.append(dag.calculate_max_flow())
        max_flow_list = np.array(max_flow_list)
        return max_flow_list.mean(), max_flow_list.std()

    def run(self, max_sizes, run_name):
        results = []
        static_capacities = (len(max_sizes) - 1) * [10]
        size_ranges = [range(1, max_sizes[0] + 1)] + [range(2, max_size + 1) for max_size in max_sizes[1:]]
        for size_combination in product(*size_ranges):
            print("sizes: ", size_combination)
            exp_results = self.calculate_max_flow(size_intervals=size_combination,
                                                  max_capacity_per_level=static_capacities)
            results.append(size_combination + exp_results)

        columns = [f'size_{i}' for i in range(1, len(max_sizes) + 1)] + ['flow_avg', 'flow_var']
        results_df = pd.DataFrame(results, columns=columns)
        results_df.to_csv(f'{run_name}.csv', index=False)


if __name__ == '__main__':
    main = Main(exp_per_state=100)
    main.run(max_sizes=[1, 20, 20], run_name='tryout')
