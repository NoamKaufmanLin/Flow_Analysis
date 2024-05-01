import datetime
import traceback
import json
import numpy as np
from itertools import product

from dag import DAG


class Main:
    def __init__(self, layer_num=3, exp_per_state=10):
        self.layer_num = layer_num
        self.exp_per_state = exp_per_state

    def calculate_max_flow(self, size_intervals, max_edges_per_level, max_capacity_per_level):
        max_flow_list = []
        for _ in range(self.exp_per_state):
            dag = DAG(level_size_intervals=size_intervals,
                      max_edges_per_level=max_edges_per_level,
                      max_capacity_per_level=max_capacity_per_level)
            max_flow_list.append(dag.calculate_max_flow())
        max_flow_list = np.array(max_flow_list)
        return max_flow_list.mean(), max_flow_list.std()

    def calculate_interval(self):
        # calculate_max_flow(size_intervals, max_edges_per_level, max_capacity_per_level)
        pass

    def save_experiment_data(self):
        pass

    def visualize_experiment(self, save=True):
        pass

    def run(self, max_sizes):
        static_capacities = (len(max_sizes) - 1) * [10]
        size_ranges = [range(1, max_size + 1) for max_size in max_sizes]
        for size_combination in product(*size_ranges):
            max_edges = [size_combination[i] * size_combination[i + 1] for i in range(len(size_combination) - 1)]
            min_edges = [max(size_combination[i], size_combination[i + 1]) for i in range(len(size_combination) - 1)]
            edge_ranges = [range(min_edges[i], max_edges[i] + 1) for i in range(len(size_combination) - 1)]
            for edge_combination in product(*edge_ranges):
                print("sizes: ", size_combination)
                print("edges: ", edge_combination)
                print(self.calculate_max_flow(size_intervals=size_combination,
                                              max_edges_per_level=edge_combination,
                                              max_capacity_per_level=static_capacities))


if __name__ == '__main__':
    main = Main(layer_num=3, exp_per_state=10000)
    main.run(max_sizes=[1, 2, 2])
