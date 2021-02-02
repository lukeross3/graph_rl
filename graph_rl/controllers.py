from abc import ABC, abstractmethod

import numpy as np
import torch.nn as nn

from graph_rl.graph import Graph


class Controller(ABC):
    """Interface for controller classes"""

    @abstractmethod
    def pick_procs(self, graph: Graph, n_procs: int) -> np.ndarray:
        """Generate node assignments for the input graph

        Args:
            graph ([Graph]): Graph containing nodes to assign
            n_procs (int): Number of processors available

        Returns:
            np.ndarray: Processor assignments, ordered according to the input graph's node list.
        """


class DummyController(Controller):
    """Dummy controller class for non-master processors"""

    def pick_procs(self, graph: Graph, n_procs: int) -> None:
        """Fake node assignments function - just returns None

        Args:
            graph ([Graph]): Graph containing nodes to assign
            n_procs (int): Number of processors available

        Returns:
            np.ndarray: Processor assignments, ordered according to the input graph's node list.
        """
        return None


class RandomController(Controller):
    """Controller for random search of processor assignments"""

    def pick_procs(self, graph: Graph, n_procs: int) -> np.ndarray:
        """Generate random node assignments for the input graph

        Args:
            graph ([Graph]): Graph containing nodes to assign
            n_procs (int): Number of processors available

        Returns:
            np.ndarray: Processor assignments, ordered according to the input graph's node list.
        """
        return np.random.randint(n_procs, size=len(graph.nodes))


class TransformerController(Controller):
    """Controller for RL-based search of processor assignments"""

    def __init__(self, model: nn.Module) -> None:
        """Initialize the Controller

        Args:
            model (nn.Module): Pytorch model
        """
        self.model = model

    def pick_procs(self, graph: Graph, n_procs: int) -> np.ndarray:
        """Generate node assignments using controller model for the input graph

        Args:
            graph ([Graph]): Graph containing nodes to assign
            n_procs (int): Number of processors available

        Returns:
            np.ndarray: Processor assignments, ordered according to the input graph's node list.
        """
        return [0]
