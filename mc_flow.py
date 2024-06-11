import networkx as nx
import numpy as np
import pulp
import datetime
import traceback
from itertools import product
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from graph_elements import Node, DirectedEdge, Commodity


class DAG:
    def __init__(self, level_sizes):
        self.level_sizes = level_sizes
        self.node_levels = []
        self.edge_levels = []
        self.node_count = 0

        self.graph = nx.DiGraph()
        self.add_levels()
        self.add_edges()

    def add_edges(self):
        for index in range(1, len(self.level_sizes)):
            level_edges = {}
            source_level = self.node_levels[index].values()
            target_level = self.node_levels[index - 1].values()

            for source_node in source_level:
                for target_node in target_level:
                    edge = DirectedEdge(source_node=source_node.position_index, target_node=target_node.position_index)
                    self.graph.add_edge(source_node.position_index, target_node.position_index, uuid=edge.uuid)
                    level_edges[edge.uuid] = edge
            self.edge_levels.append(level_edges)

    def add_level(self, size, level_index):
        level_nodes = {}
        for i in range(size):
            node = Node(level_index=level_index, node_index=i, position_index=i + self.node_count)
            self.graph.add_node(node.position_index, level_index=node.level_index,
                                node_index=node.node_index, uuid=node.uuid)
            level_nodes[node.uuid] = node
        self.node_count += size
        self.node_levels.append(level_nodes)

    def add_levels(self):
        for i, level_size in enumerate(self.level_sizes):
            self.add_level(size=level_size, level_index=i)

    def init_capacities(self):
        for level in self.edge_levels:
            level_edges = list(level.values())
            C = np.random.rand(len(level_edges))
            C = 1 / (np.sum(C)) * C
            for edge, capacity in zip(level_edges, C):
                edge.capacity = capacity

        for edge in self.graph.edges:
            edge_uuid = self.graph.get_edge_data(*edge)['uuid']
            match = list(filter(lambda l: edge_uuid in l, self.edge_levels))[0][edge_uuid]
            edge_cap = match.capacity
            self.graph[edge[0]][edge[1]]['capacity'] = edge_cap

    def plot_graph(self):
        plt.figure(figsize=(4, 4))
        # pos = nx.spring_layout(self.graph)  # positions for all nodes

        level_indexes = nx.get_node_attributes(self.graph, 'level_index')
        node_indexes = nx.get_node_attributes(self.graph, 'node_index')
        node_labels = {}
        for i in range(self.node_count):
            node_labels[i] = (level_indexes[i], node_indexes[i])
        pos = {node: node_labels[i] for i, node in enumerate(self.graph.nodes())}

        # nx.draw(self.graph, pos=pos, with_labels=True, node_size=600)
        nx.draw_networkx_nodes(self.graph, pos, node_size=600)
        nx.draw_networkx_edges(self.graph, pos, edgelist=self.graph.edges(), arrowstyle='->', arrowsize=20)
        nx.draw_networkx_labels(self.graph, pos, labels=node_labels, font_size=10, font_family="sans-serif")

        edge_labels = nx.get_edge_attributes(self.graph, 'capacity')
        edge_labels = {edge: f"{cap:.2f}" for edge, cap in edge_labels.items()}
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, label_pos=0.2, font_size=8)

        plt.axis('off')
        plt.show()


class MultiCommodityLP:
    def __init__(self, graph):
        self.prob = pulp.LpProblem("MaxMultiCommodityFlow", pulp.LpMaximize)
        self.graph = graph
        self.c = {}  # capacity
        self.f = {}  # Flow fraction

    def create_variables(self):
        pass

    def create_constraints(self):
        # Link capacity constraints
        # Flow conservation on transit nodes
        # Flow conservation (total)
        pass

    def create_objective(self):
        # Objective function: maximize total throughput
        pass

    def solve_problem(self):
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


# Example usage
# num_nodes = 4
# edges_variables = [(1, 0, 1), (2, 0, 1), (3, 1, 1), (3, 2, 1)]
# edges = [DirectedEdge(source_node=edge[0], target_node=edge[1], capacity=edge[2]) for edge in edges_variables]
# commodities_variables = [(3, 0, 1), (3, 0, 1), (3, 0, 1)]  #
# commodities = [Commodity(source_node=com[0], target_node=com[1], demand=com[2]) for com in commodities_variables]
#
# result = max_multi_commodity_flow(num_nodes, edges, commodities)
# for k, v in result.items():
#     print(f"Commodity {k}:")
#     for edge, flow in v.items():
#         print(f"  Edge {edge}: Flow {flow}")

if __name__ == '__main__':
    dag = DAG(level_sizes=[1, 2, 2])
    dag.init_capacities()
    dag.plot_graph()
