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
    def __init__(self, layer_num=3, exp_per_state=10):
        self.layer_num = layer_num
        self.exp_per_state = exp_per_state

    def calculate_max_flow(self, size_intervals, max_capacity_per_level):
        max_flow_list = []
        for _ in range(self.exp_per_state):
            dag = DAG(level_size_intervals=size_intervals,
                      max_capacity_per_level=max_capacity_per_level)
            max_flow_list.append(dag.calculate_max_flow())
        max_flow_list = np.array(max_flow_list)
        return max_flow_list.mean(), max_flow_list.std()

    def calculate_interval(self):
        # calculate_max_flow(size_intervals, max_edges_per_level, max_capacity_per_level)
        pass

    def save_experiment_data(self):
        pass

    def visualize_experiment(self, df, var1, var2, save=True):
        # Pivot the DataFrame to prepare for heatmap
        heatmap_data = df.pivot_table(values='flow_avg', index=var1, columns=var2, aggfunc=np.mean)

        # Generate the heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_data, annot=False, cmap='coolwarm')  # 'annot' annotates the values
        plt.title(f'Heatmap of Mean Results by {var1} and {var2}')
        plt.show()

    def run(self, max_sizes):
        results = []
        static_capacities = (len(max_sizes) - 1) * [10]
        size_ranges = [range(1, max_size + 1) for max_size in max_sizes]
        for size_combination in product(*size_ranges):
            # static_capacities = (len(size_combination) - 1) * [size_combination[-1]]
            max_edges = [size_combination[i] * size_combination[i + 1] for i in range(len(size_combination) - 1)]
            min_edges = [max(size_combination[i], size_combination[i + 1]) for i in range(len(size_combination) - 1)]
            edge_ranges = [range(min_edges[i], max_edges[i] + 1) for i in range(len(size_combination) - 1)]
            for edge_combination in product(*edge_ranges):
                print("sizes: ", size_combination)
                print("edges: ", edge_combination)
                # Simulate experiment
                exp_results = self.calculate_max_flow(size_intervals=size_combination,
                                                      max_edges_per_level=edge_combination,
                                                      max_capacity_per_level=static_capacities)
                results.append(size_combination + edge_combination + exp_results)
        columns = [f'size_{i}' for i in range(1, len(max_sizes) + 1)] + \
                  [f'max_edges_{i}{i+1}' for i in range(1, len(max_sizes))] + ['flow_avg', 'flow_var']
        results_df = pd.DataFrame(results, columns=columns)
        results_df.to_csv('experiment_results.csv', index=False)
        self.visualize_experiment(results_df, 'size_3', 'max_edges_23')


if __name__ == '__main__':
    main = Main(layer_num=3, exp_per_state=100)
    main.run(max_sizes=[1, 20, 20])
