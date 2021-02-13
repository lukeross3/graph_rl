from abc import ABC, abstractmethod

import numpy as np
import torch.nn as nn
from torch.distributions.categorical import Categorical

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
        self.n_forward_passes = self.update_size - 1
        self.proc_batch = None
        self.log_probs = None
        self.rewards = None

    def _pick_procs_batch(self, graph: Graph):
        """Generate node assignments for with batch size self.update_size, and sets member
        variables for public API `pick_procs`.

        Args:
            graph ([Graph]): Graph containing nodes to assign
        """
        probs_tensor = self.model(graph, batch_size=self.update_size)
        dist = Categorical(probs_tensor)
        self.proc_batch = dist.sample()
        self.log_probs = dist.log_prob(self.proc_batch)

    def pick_procs(self, graph: Graph, n_procs: int) -> np.ndarray:
        """Generate node assignments using controller model for the input graph

        Args:
            graph ([Graph]): Graph containing nodes to assign
            n_procs (int): Number of processors available

        Returns:
            np.ndarray: Processor assignments, ordered according to the input graph's node list.
        """

        # Run model if previous batch has run out or it's the first pass
        if self.n_forward_passes == self.update_size - 1:

            # Update model if it's not the first pass
            if self.proc_batch is not None:
                pass  # TODO

            self.n_forward_passes = 0
            self._pick_procs_batch(graph)

        # Pick assignment from batch
        procs = self.proc_batch[:, self.n_forward_passes]

        # Increment the number of forward passes
        self.n_forward_passes += 1

        return procs
        

from graph_rl.models import TFEncoderWithEmbs, count_parameters

model = TFEncoderWithEmbs(n_procs=100, n_layers=1)
n_params = count_parameters(model)
print(f"n_params: {n_params}")

# Import MPI
from mpi4py import MPI
from graph_rl.nodes import AddN

# Init the graph
nodes = [AddN(1) for _ in range(5)]
connections = [
    (-1, 0),
    (0, 1),
    (1, 2),
    (2, 3),
]
comm = MPI.COMM_WORLD
g = Graph(nodes, connections, comm)

controller = TransformerController(model)
procs = controller.pick_procs(g, None)
print(f"procs: {procs}")
