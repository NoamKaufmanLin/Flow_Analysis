from collections import OrderedDict
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random


class Node:
    def __init__(self, level_index, node_index):
        self.level_index = level_index
        self.node_index = node_index  # index in level
        self.position_index = None  # global index in DiGraph
        self.in_degree = 0  # Non-zero capacity edges
        self.out_degree = 0  # Non-zero capacity edges

    def __str__(self):
        return f'{self.level_index},{self.node_index}'


class Level:
    def __init__(self, level_index, size=2):
        self.size = self.set_size(size=size)
        self.nodes = [Node(level_index=level_index, node_index=i) for i in range(self.size)]

    def __iter__(self):
        return iter(self.nodes)

    @staticmethod
    def set_size(size):
        if isinstance(size, tuple):
            return random.randint(size[0], size[1])
        elif isinstance(size, int):
            return size


class CapacityManager:
    def __init__(self, graph, prev_level, next_level, max_edges_count, max_capacity_sum):
        # Edges point from next_level to prev_level
        self.graph = graph  # TODO: Isn't is heavy to ssave the graph again and again?
        self.prev_level = prev_level
        self.next_level = next_level
        self.max_edges_count = max_edges_count
        self._edges_count = 0
        self.max_capacity_sum = max_capacity_sum
        self._capacity_sum = 0
        self.initial_capacity = 1  # TODO: ?

    def assign_capacities(self):
        self._init_capacities()
        while self._edges_count < self.max_edges_count and self._capacity_sum < self.max_capacity_sum:
            edge = self._get_random_edge()
            capacity = self._get_random_capacity()
            self._increase_edge_capacity(edge, capacity)

    def _init_capacities(self):
        # TODO: Optimize edge randomization to prevent collisions (regarding max_edges_count)
        for target_node in self.prev_level:
            edge = (random.choice(self.next_level), target_node)
            self._increase_edge_capacity(edge, self.initial_capacity)

        for source_node in self.next_level:
            edge = (source_node, random.choice(self.prev_level))
            self._increase_edge_capacity(edge, self.initial_capacity)

    def _get_random_edge(self):
        return random.choice(self.next_level), random.choice(self.prev_level)

    def _get_random_capacity(self):
        return 1

    def _increase_edge_capacity(self, edge, capacity):
        current_capacity = self.graph.get_edge_data(edge[0].position_index, edge[1].position_index)['capacity']
        if current_capacity == 0:
            self._edges_count += 1
        nx.set_edge_attributes(self.graph, {
            (edge[0].position_index, edge[1].position_index): {'capacity': current_capacity + capacity}})
        self._capacity_sum += capacity


class DAG:
    def __init__(self, level_size_intervals, max_edges_per_level, max_capacity_per_level):
        self.level_size_intervals = level_size_intervals
        self.level_sizes = []
        self.levels = []
        self.node_count = 0
        self.max_edges_per_level = max_edges_per_level
        self.max_capacity_per_level = max_capacity_per_level

        self.graph = nx.DiGraph()
        self.add_levels()
        self.add_edges()

    def add_edges(self):
        for index in range(1, len(self.levels)):
            next_level = self.levels[index].nodes
            prev_level = self.levels[index - 1].nodes
            for x in next_level:
                for y in prev_level:
                    self.graph.add_edge(x.position_index, y.position_index, capacity=0)

            capacity_manager = CapacityManager(graph=self.graph, prev_level=prev_level, next_level=next_level,
                                               max_edges_count=self.max_edges_per_level[index - 1],
                                               max_capacity_sum=self.max_capacity_per_level[index - 1])
            capacity_manager.assign_capacities()

    def add_level(self, size):
        next_level = Level(level_index=len(self.levels), size=size)
        self.levels.append(next_level)
        for i, node in enumerate(next_level.nodes):
            node.position_index = i + self.node_count
            self.graph.add_node(node.position_index, level_index=node.level_index, node_index=node.node_index)
        self.node_count += next_level.size
        self.level_sizes.append(next_level.size)

    def add_levels(self):
        for level_size_interval in self.level_size_intervals:
            self.add_level(size=level_size_interval)

    def calculate_max_flow(self, level_index=-1, flow_func=None):
        # Set sources and sinks
        flow_graph = self.graph.copy()
        for node in self.levels[level_index].nodes:
            source_capacity = sum([self.graph.get_edge_data(node.position_index, successor)['capacity']
                                   for successor in self.graph.successors(node.position_index)])
            flow_graph.add_edge(-1, node.position_index, capacity=source_capacity)
        return nx.maximum_flow_value(flow_graph, -1, 0, capacity='capacity', flow_func=flow_func)

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
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, label_pos=0.15, font_size=8)

        plt.axis('off')
        plt.show()


size_intervals = [1, 7, 4]  # (6, 8)
max_edges_per_level = [10, 20, 20]
max_capacity_per_level = [10, 20, 20]
dag = DAG(level_size_intervals=size_intervals,
          max_edges_per_level=max_edges_per_level,
          max_capacity_per_level=max_capacity_per_level)
dag.plot_graph()
max_flow = dag.calculate_max_flow()
print(max_flow)

# d_{i->} > 0, d_{->j} > 0, Sum d_{i->} < Theta_{d,i}
# Sum c_{i->} = Sum c_{->i}
# Run experiments
