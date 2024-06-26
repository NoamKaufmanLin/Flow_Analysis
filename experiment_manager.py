import numpy as np
from graph_elements import DAG
from mc_flow import Commodity, MultiCommodityLP


class Experiment:
    def __init__(self, dag, exp_num=10):
        self.dag = dag
        self.exp_num = exp_num
        self.commodities = self.init_commodities()

    def init_commodities(self):
        commodities = []
        for node in self.dag.node_levels[-1].values():
            commodity = Commodity(source_node=node.position_index, target_node=0, demand=1,
                                  id=node.position_index % self.dag.level_sizes[-1])
            commodities.append(commodity)
        return commodities

    def solve_mclp(self, objective_method):
        mclp = MultiCommodityLP(graph=self.dag.graph, commodities=self.commodities, objective_method=objective_method)
        return mclp.get_optimized_objective()

    def run_exp(self):
        lambda_total_list = []
        lambda_min_list = []
        for _ in range(self.exp_num):
            self.dag.init_capacities()
            lambda_total = self.solve_mclp(objective_method="TotalThroughput")/self.dag.level_sizes[-1]
            lambda_min = self.solve_mclp(objective_method="MinThroughput")
            lambda_total_list.append(lambda_total)
            lambda_min_list.append(lambda_min)

        lambda_total_list = np.array(lambda_total_list)
        lambda_min_list = np.array(lambda_min_list)
        return lambda_total_list.mean(), lambda_total_list.std(), lambda_min_list.mean(), lambda_min_list.std()


if __name__ == '__main__':
    dag = DAG(level_sizes=[1, 2, 2])
    exp = Experiment(dag=dag, exp_num=10)
    print(exp.run_exp())
