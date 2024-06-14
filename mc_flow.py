from uuid import uuid4
import pulp
from graph_elements import DAG


class Commodity:
    def __init__(self, source_node, target_node, demand):
        self.uuid = str(uuid4())
        self.source_node = source_node
        self.target_node = target_node
        self.demand = demand


class MultiCommodityLP:
    def __init__(self, graph, commodities):
        self.prob = pulp.LpProblem("MaxMultiCommodityFlow", pulp.LpMaximize)
        self.graph = graph
        self.commodities = commodities
        self.f = self.create_flow_variables()  # Flow fraction
        self.create_constraints()

    def node_flow(self, node, commodity_uuid):
        incoming_flow = pulp.lpSum(self.f[(commodity_uuid, self.graph.get_edge_data(v, node)['uuid'])]
                                   for v in self.graph.predecessors(node))
        outgoing_flow = pulp.lpSum(self.f[(commodity_uuid, self.graph.get_edge_data(node, v)['uuid'])]
                                   for v in self.graph.successors(node))
        return outgoing_flow - incoming_flow

    def create_flow_variables(self):
        f = {}
        for edge in self.graph.edges():
            for commodity in range(len(self.commodities)):
                f[(commodity.uuid, edge.uuid)] = pulp.LpVariable(f"f_{commodity.uuid}_{edge.uuid}",
                                                                 lowBound=0, upBound=1)
        return f

    def create_constraints(self):
        # Link capacity constraints
        for edge in self.graph.edges():
            self.prob += (pulp.lpSum(self.f[(commodity.uuid, edge.uuid)] * commodity.demand for commodity in
                                     self.commodities) <= edge.capacity,
                          f"LinkCapacity_{edge.uuid}")

        # Flow conservation on transit nodes
        for commodity in self.commodities:
            for node in self.graph.nodes():
                if node != commodity.source_node and node != commodity.target_node:
                    self.prob += (self.node_flow(node=node, commodity_uuid=commodity.uuid) == 0,
                                  f"FlowConservation_{commodity.uuid}_{node.uuid}")

        # Source and target nodes
        for commodity in self.commodities:
            source_flow = self.node_flow(node=commodity.source_node, commodity_uuid=commodity.uuid)
            target_flow = self.node_flow(node=commodity.target_node, commodity_uuid=commodity.uuid)

            # Total flow conservation constraint
            self.prob += (source_flow == - target_flow, f"FlowConservation_{commodity.uuid}")
            # Source flow constraint
            self.prob += (source_flow <= 1, f"SourceFlow_{commodity.uuid}")

    def create_objective(self, method="TotalThroughput"):
        if method == "TotalThroughput":
            self.prob += pulp.lpSum(self.node_flow(node=commodity.source_node, commodity_uuid=commodity.uuid)
                                    for commodity in self.commodities), "TotalThroughput"

        elif method == "MaxMinThroughput":
            min_throughput = pulp.LpVariable("min_throughput", lowBound=0)
            for commodity in self.commodities:
                source_flow = self.node_flow(node=commodity.source_node, commodity_uuid=commodity.uuid)
                self.prob += (min_throughput <= source_flow, f"MinThroughputConstraint_{commodity.uuid}")
            self.prob += min_throughput, "MaxMinThroughput"

    def solve_problem(self):
        self.prob.solve()

    def extract_results(self):
        # result = {}
        # for (i, (_, _, _)) in enumerate(commodities):
        #     result[i] = {}
        #     for (u, v, _) in edges:
        #         result[i][(u, v)] = f[(i, u, v)].varValue
        #
        # return result
        pass


def max_multi_commodity_flow(num_nodes, edges, commodities):
    # Create the linear programming problem
    prob = pulp.LpProblem("MaxMultiCommodityFlow", pulp.LpMaximize)

    # Create variables
    f = {}
    for edge in edges:
        for i in range(len(commodities)):
            # TODO: edge/commodity_id?
            f[(i, edge.source_node, edge.target_node)] = pulp.LpVariable(f"f_{i}_{edge.source_node}_{edge.target_node}",
                                                                         0, 1)

    # Link capacity constraints
    for edge in edges:
        prob += (pulp.lpSum(f[(i, edge.source_node, edge.target_node)] * d_i for (i, (_, _, d_i)) in
                            enumerate(commodities)) <= edge.capacity,
                 f"LinkCapacity_{edge.source_node}_{edge.target_node}")

    # Flow conservation on transit nodes
    for (i, (s_i, t_i, d_i)) in enumerate(commodities):
        for u in range(num_nodes):
            if u != s_i and u != t_i:
                prob += (pulp.lpSum(f[(i, u, w)] for (_, w, _) in edges if (i, u, w) in f) -
                         pulp.lpSum(f[(i, w, u)] for (w, _, _) in edges if (i, w, u) in f) == 0,
                         f"FlowConservation_{i}_{u}")

    # # Flow conservation at the source
    # for (i, (s_i, t_i, d_i)) in enumerate(commodities):
    #     prob += (pulp.lpSum(f[(i, s_i, w)] for (s_i, w, _) in edges if (i, s_i, w) in f) -
    #              pulp.lpSum(f[(i, w, s_i)] for (w, s_i, _) in edges if (i, w, s_i) in f) == 1, f"SourceFlow_{i}")
    #
    # # Flow conservation at the destination
    # for (i, (s_i, t_i, d_i)) in enumerate(commodities):
    #     prob += (pulp.lpSum(f[(i, w, t_i)] for (w, t_i, _) in edges if (i, w, t_i) in f) -
    #              pulp.lpSum(f[(i, t_i, w)] for (t_i, w, _) in edges if (i, t_i, w) in f) == 1, f"SinkFlow_{i}")
    print()

    # Flow conservation (total)
    for (i, (s_i, t_i, d_i)) in enumerate(commodities):
        prob += (pulp.lpSum(f[(i, s_i, w)] for (_, w, _) in edges if (i, s_i, w) in f) -
                 pulp.lpSum(f[(i, w, s_i)] for (w, _, _) in edges if (i, w, s_i) in f) ==
                 pulp.lpSum(f[(i, w, t_i)] for (w, _, _) in edges if (i, w, t_i) in f) -
                 pulp.lpSum(f[(i, t_i, w)] for (_, w, _) in edges if (i, t_i, w) in f), f"FlowConservation_{i}")

        prob += (pulp.lpSum(f[(i, s_i, w)] for (_, w, _) in edges if (i, s_i, w) in f) -
                 pulp.lpSum(f[(i, w, s_i)] for (w, _, _) in edges if (i, w, s_i) in f) <= 1, f"SourceFlow_{i}")

    # Objective function: maximize total throughput
    prob += pulp.lpSum(pulp.lpSum(f[(i, s_i, w)] for (_, w, _) in edges if (i, s_i, w) in f) -
                       pulp.lpSum(f[(i, w, s_i)] for (w, _, _) in edges if (i, w, s_i) in f)
                       for (i, (s_i, t_i, d_i)) in enumerate(commodities)), "TotalThroughput"

    # Objective function: maximize total throughput
    prob += pulp.lpSum(pulp.lpSum(f[(i, s_i, w)] for (_, w, _) in edges if (i, s_i, w) in f) -
                       pulp.lpSum(f[(i, w, s_i)] for (w, _, _) in edges if (i, w, s_i) in f)
                       for (i, (s_i, t_i, d_i)) in enumerate(commodities)), "TotalThroughput"
    lambda_vars = [pulp.LpVariable(f"lambda_{i}", 0, None) for (i, (_, _, _)) in enumerate(commodities)]
    for (i, (s_i, t_i, d_i)) in enumerate(commodities):
        prob += (pulp.lpSum(f[(i, s_i, w)] for (_, w, _) in edges if (i, s_i, w) in f) -
                 pulp.lpSum(f[(i, w, s_i)] for (w, _, _) in edges if (i, w, s_i) in f) == lambda_vars[i], f"lambda_{i}")

        prob += (pulp.lpSum(f[(i, s_i, w)] for (_, w, _) in edges if (i, s_i, w) in f) -
                 pulp.lpSum(f[(i, w, s_i)] for (w, _, _) in edges if (i, w, s_i) in f) <= 1, f"SourceFlow_{i}")

    # Solve the problem
    prob.solve()

    # Extract the results
    result = {}
    for (i, (_, _, _)) in enumerate(commodities):
        result[i] = {}
        for (u, v, _) in edges:
            result[i][(u, v)] = f[(i, u, v)].varValue

    return result


if __name__ == '__main__':
    dag = DAG(level_sizes=[1, 2, 2])
    dag.init_capacities()
    # dag.plot_graph()
    # c1 = Commodity(source_node=com[0], target_node=com[1], demand=com[2])  # [(4, 0, 1), (5, 0, 1)]
    # commodities = [c1, c2]
    #
    # mclp = MultiCommodityLP(graph=dag.graph, commodities=commodities)
    print()
