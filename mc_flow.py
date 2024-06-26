from uuid import uuid4
import pulp
from graph_elements import DAG

pulp.LpSolverDefault.msg = 0


class Commodity:
    def __init__(self, id, source_node, target_node, demand):
        self.id = id
        self.source_node = source_node
        self.target_node = target_node
        self.demand = demand


class MultiCommodityLP:
    def __init__(self, graph, commodities, objective_method="TotalThroughput"):
        self.prob = pulp.LpProblem("MaxMultiCommodityFlow", pulp.LpMaximize)
        self.graph = graph
        self.commodities = commodities
        self.f = self.create_flow_variables()  # Flow fraction
        self.create_constraints()
        self.create_objective(method=objective_method)

    def node_flow(self, node, commodity_id):
        incoming_flow = pulp.lpSum(self.f[(commodity_id, v, node)] for v in self.graph.predecessors(node))
        outgoing_flow = pulp.lpSum(self.f[(commodity_id, node, v)] for v in self.graph.successors(node))
        return outgoing_flow - incoming_flow

    def create_flow_variables(self):
        f = {}
        for edge in self.graph.edges():
            for commodity in self.commodities:
                f[(commodity.id, *edge)] = pulp.LpVariable(f"f_{commodity.id}_{edge[0]}_{edge[1]}",
                                                           lowBound=0, upBound=1)
        return f

    def create_constraints(self):
        # Link capacity constraints
        for edge in self.graph.edges():
            self.prob += (pulp.lpSum(
                self.f[(commodity.id, *edge)] * commodity.demand for commodity in self.commodities) <=
                          self.graph.get_edge_data(*edge)['capacity'], f"LinkCapacity_{edge}")

        # Flow conservation on transit nodes
        for commodity in self.commodities:
            for node in self.graph.nodes():
                if node != commodity.source_node and node != commodity.target_node:
                    self.prob += (self.node_flow(node=node, commodity_id=commodity.id) == 0,
                                  f"FlowConservation_{commodity.id}_{node}")

        # Source and target nodes
        for commodity in self.commodities:
            source_flow = self.node_flow(node=commodity.source_node, commodity_id=commodity.id)
            target_flow = self.node_flow(node=commodity.target_node, commodity_id=commodity.id)

            # Total flow conservation constraint
            self.prob += (source_flow == - target_flow, f"FlowConservation_{commodity.id}")
            # Source flow constraint
            self.prob += (source_flow <= 1, f"SourceFlow_{commodity.id}")

    def create_objective(self, method="TotalThroughput"):
        if method == "TotalThroughput":
            self.prob += pulp.lpSum(self.node_flow(node=commodity.source_node, commodity_id=commodity.id)
                                    for commodity in self.commodities), "TotalThroughput"

        elif method == "MinThroughput":
            min_throughput = pulp.LpVariable("min_throughput", lowBound=0)
            for commodity in self.commodities:
                source_flow = self.node_flow(node=commodity.source_node, commodity_id=commodity.id)
                self.prob += (min_throughput <= source_flow, f"MinThroughputConstraint_{commodity.id}")
            self.prob += min_throughput, "MinThroughput"

    def get_optimized_objective(self):
        self.prob.solve()
        return pulp.value(self.prob.objective)

    def extract_results(self, do_print=False):
        results = {}
        for commodity in self.commodities:
            results[commodity.id] = {}
            for edge in self.graph.edges():
                variable = self.f[(commodity.id, *edge)]
                results[commodity.id][edge] = variable.varValue if variable.varValue is not None else 0.0

        if do_print:
            for k, v in results.items():
                print(f"Commodity {k}:")
                for edge, flow in v.items():
                    print(f"  Edge {edge}: Flow {flow}")

        return results


if __name__ == '__main__':
    dag = DAG(level_sizes=[1, 2, 2])
    dag.init_capacities()
    # dag.plot_graph()

    c1 = Commodity(source_node=3, target_node=0, demand=1, id=1)  # [(4, 0, 1), (5, 0, 1)]
    c2 = Commodity(source_node=4, target_node=0, demand=1, id=2)
    commodities = [c1, c2]
    mclp = MultiCommodityLP(graph=dag.graph, commodities=commodities, objective_method="TotalThroughput")
    print()
