from typing import List, Tuple

from graph_rl import Node


class Graph:
    def __init__(self, nodes: List[Node], connections: List[Tuple[int, int]]) -> None:
        self.nodes = nodes
        self.parse_graph()
