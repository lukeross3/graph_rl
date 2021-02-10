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

    def __init__(self, model: nn.Module, update_size: int = 8) -> None:
        """Initialize the Controller

        Args:
            model (nn.Module): Pytorch model
            update_size (int, optional): Number of forward passes to run and store before running
                backward pass.
        """
        self.model = model
        self.update_size = update_size
        self.n_forward_passes = 0
        self.probs_tensor = None
        self.rewards = None

    def pick_procs(self, graph: Graph, n_procs: int) -> np.ndarray:
        """Generate node assignments using controller model for the input graph

        Args:
            graph ([Graph]): Graph containing nodes to assign
            n_procs (int): Number of processors available

        Returns:
            np.ndarray: Processor assignments, ordered according to the input graph's node list.
        """
        
        # Run model if previous batch has run out or it's the first pass
        if (self.n_forward_passes == self.update_size - 1) or (self.probs_tensor is None):
            self.n_forward_passes = 0
            self.probs_tensor = self.model(graph, batch_size=self.update_size)

        # Sample from generated probability distribution
        procs = [0]

        # Increment the number of forward passes
        self.n_forward_passes += 1

        return procs
        
