from uuid import uuid4


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


class Commodity:
    def __init__(self, source_node, target_node, demand):
        self.uuid = str(uuid4())
        self.source_node = source_node
        self.target_node = target_node
        self.demand = demand
