from uuid import uuid4
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt


class Node:
    def __init__(self, level_index, node_index, position_index):
        self.uuid = str(uuid4())
        self.level_index = level_index
        self.node_index = node_index  # index in level
        self.position_index = position_index  # global index in DiGraph
        self.in_degree = 0  # Non-zero capacity edges
        self.out_degree = 0  # Non-zero capacity edges

    def __str__(self):
        return f'{self.level_index},{self.node_index}'


class DirectedEdge:
    def __init__(self, source_node, target_node, capacity=None):
        self.uuid = str(uuid4())
        self.source_node = source_node
        self.target_node = target_node
        self.capacity = None


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
        # TODO: Expand methods
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
