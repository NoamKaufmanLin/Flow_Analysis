import datetime
import traceback
import json
import numpy as np

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

    def run(self):
        # for batch in self.conf_manager.get_experiment_batches():
        pass


if __name__ == '__main__':
    main = Main(layer_num=3)
    main.run()
