from typing import List
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
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

    @abstractmethod
    def register_times(self, t: List[float]) -> None:
        """Register the time taken by the last processor assignment

        Args:
            t (List[float]): Time taken by the last processor assignment
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

    def register_times(self, t: List[float]) -> None:
        """Fake time registration function - just returns None

        Args:
            t (List[float]): Time taken by the last processor assignment
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

    def register_times(self, t: List[float]) -> None:
        """Fake time registration function - just returns None since the random controller does
        not involve a feedback loop.

        Args:
            t (List[float]): Time taken by the last processor assignment
        """
        return None


class TransformerController(Controller):
    """Controller for RL-based search of processor assignments"""

    def __init__(
        self, model: nn.Module, update_size: int = 8, use_baseline: bool = True, lr: float = 1e-3
    ) -> None:
        """Initialize the Controller

        Args:
            model (nn.Module): Pytorch model
            update_size (int, optional): Number of forward passes to run and store before running
                backward pass.
            use_baseline (bool, optional): Whether or not to use an average baseline in the
                reinforce loss calculation. Defaults to True.
            lr (float, optional): Learning rate for Adam optimizer. Defaults to 1e-3.
        """
        # Modeling vars
        self.model = model
        self.update_size = update_size
        self.use_baseline = use_baseline
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Bookkeeping vars
        self.n_forward_passes = self.update_size
        self.proc_batch = None
        self.log_probs = None
        self.times = []

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

    def _model_step(self) -> None:
        """Run a single reinforce step using registered times"""

        # Preconditions checking
        assert len(self.times) == self.log_probs.shape[1], (
            f"Number of registered times ({len(self.times)}) != number of log_probs "
            f"({self.log_probs.shape[1]})"
        )

        # Compute loss and run backward pass
        self.optimizer.zero_grad()
        rewards = torch.Tensor(self.times)
        if self.use_baseline:
            rewards = rewards - torch.mean(rewards)
        loss = (-self.log_probs * rewards).sum()
        loss.backward()
        self.optimizer.step()

    def pick_procs(self, graph: Graph, n_procs: int) -> np.ndarray:
        """Generate node assignments using controller model for the input graph

        Args:
            graph ([Graph]): Graph containing nodes to assign
            n_procs (int): Number of processors available

        Returns:
            np.ndarray: Processor assignments, ordered according to the input graph's node list.
        """

        # Run model if previous batch has run out or it's the first pass
        if self.n_forward_passes == self.update_size:

            # Update model if it's not the first pass
            if self.proc_batch is not None:
                self._model_step()

            # Reset internal vars
            self.n_forward_passes = 0
            self.proc_batch = None
            self.log_probs = None
            self.times = []

            # Run model inference to get next `update_size` processor assignments
            self._pick_procs_batch(graph)

        # Pick assignment from batch
        procs = self.proc_batch[:, self.n_forward_passes]

        # Increment the number of forward passes
        self.n_forward_passes += 1

        return procs

    def register_times(self, t: List[float]) -> None:
        """Record runtimes of associated processor assignments

        Args:
            t (List[float]): Time taken by the last processor assignments
        """
        self.times.extend(t)
