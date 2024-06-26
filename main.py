from itertools import product
import pandas as pd
from experiment_manager import Experiment
from graph_elements import DAG


class Main:
    def __init__(self, exp_per_state=10):
        self.exp_per_state = exp_per_state

    def run(self, run_name, max_sizes, sride=1):
        results = []
        size_ranges = [range(1, max_sizes[0] + 1)] + [range(2, max_size + 1, sride) for max_size in max_sizes[1:]]
        for size_combination in product(*size_ranges):
            print("sizes: ", size_combination)
            dag = DAG(level_sizes=size_combination)
            exp = Experiment(dag=dag, exp_num=self.exp_per_state)
            exp_results = exp.run_exp()
            results.append(size_combination + exp_results)

        columns = [f'size_{i}' for i in range(1, len(max_sizes) + 1)] + \
                  ['flow_ratio_avg', 'flow_ratio_var', 'min_ratio_avg', 'min_ratio_var']
        results_df = pd.DataFrame(results, columns=columns)
        results_df.to_csv(f'{run_name}.csv', index=False)
        print(results_df)


if __name__ == '__main__':
    main = Main(exp_per_state=20)
    run_name = '20_20_20_constant20_stride2'
    max_sizes = [1, 10, 10, 10]
    sride = 2
    # initialization_method = 'binary'
    main.run(run_name=run_name, max_sizes=max_sizes, sride=sride)
