from abc import ABC, abstractmethod
from typing import List

import numpy as np

from graph_rl.graph import Graph


class Controller(ABC):
    """Interface for controller classes"""

    @abstractmethod
    def pick_procs(self, graph: Graph, n_procs: int) -> List[int]:
        """Generate node assignments for the input graph

        Args:
            graph ([Graph]): Graph containing nodes to assign
            n_procs (int): Number of processors available

        Returns:
            List[int]: Processor assignments, ordered according to the input graph's node list.
        """


class DummyController(Controller):
    """Dummy controller class for non-master processors"""

    def pick_procs(self, graph: Graph, n_procs: int) -> None:
        """Fake node assignments function - just returns None

        Args:
            graph ([Graph]): Graph containing nodes to assign
            n_procs (int): Number of processors available

        Returns:
            List[int]: Processor assignments, ordered according to the input graph's node list.
        """
        return None


class RandomController(Controller):
    """Controller for random search of processor assignments"""

    def pick_procs(self, graph: Graph, n_procs: int) -> List[int]:
        """Generate random node assignments for the input graph

        Args:
            graph ([Graph]): Graph containing nodes to assign
            n_procs (int): Number of processors available

        Returns:
            List[int]: Processor assignments, ordered according to the input graph's node list.
        """
        return np.random.randint(n_procs, size=len(graph.nodes))
