import time
from typing import Any, Tuple

import numpy as np
from mpi4py import MPI

from graph_rl.graph import Graph
from graph_rl.controllers import Controller, DummyController


class Session:
    def __init__(self, comm: MPI.Comm, controller: Controller) -> None:
        # Assign instance variables
        self.comm = comm
        self.controller = controller

        # Check controller
        if self._is_master():
            if isinstance(controller, DummyController):
                raise ValueError(
                    "Controller on master process (rank == 0) cannot be instance of DummyController"
                )
        else:
            if not isinstance(controller, DummyController):
                raise ValueError(
                    "Controller on non-master process (rank != 0) "
                    "must be instance of DummyController"
                )

    def _is_master(self) -> bool:
        """Return True if the current processor rank is 0

        Returns:
            bool: Whether or not the processor rank is 0
        """
        return self.comm.rank == 0

    def time_graph(self, x: Any, graph: Graph, n: int = 1) -> float:
        """Time the parallel execution of the graph on input x, taking the median over n runs.

        Args:
            x (Any): Graph input
            graph (Graph): Graph to time (processors must already be initialized)
            n (int, optional): Number of runs to take median over. Defaults to 1.

        Returns:
            float: [description]
        """
        times = np.empty(n)
        for i in range(n):
            self.comm.Barrier()
            t0 = MPI.Wtime()
            _ = graph.forward(x)
            self.comm.Barrier()
            t1 = MPI.Wtime()
            times[i] = t1 - t0
            graph.reset()
        return np.median(times)

    def learn_assignments(
        self, x: Any, graph: Graph, n_iter: int = 1000, timing_runs: int = 1, init_runs: int = 1,
    ) -> np.array:

        # Initialize local vars on the master process
        if self._is_master():
            start_time = time.time()
            iter_times = np.empty(n_iter)
            times = np.empty(n_iter)
            best_times = np.empty(n_iter)
        else:
            iter_times = None
            times = None
            best_times = None
        best_proc_assignment = None

        # Loop over init_runs to "skip" the slower runs that include initialization
        for i in range(init_runs):
            if self.comm.rank == 0:
                proc_list = np.random.randint(self.comm.size, size=len(graph.nodes))
            else:
                proc_list = None
            proc_list = self.comm.bcast(proc_list, root=0)
            graph.proc_init(proc_list)
            _ = self.time_graph(x, graph, timing_runs)

        # Loop over iterations
        for i in range(n_iter):

            # Initialize processor assignments
            proc_list = self.controller.pick_procs(graph, self.comm.size)
            proc_list = self.comm.bcast(proc_list, root=0)
            graph.proc_init(proc_list)

            # Time the forward pass and update local vars
            t = self.time_graph(x, graph, timing_runs)
            if self._is_master():
                iter_times[i] = time.time() - start_time
                times[i] = t
                if i == 0:
                    best_time = np.inf
                else:
                    best_time = best_times[i - 1]
                if t < best_time:
                    best_times[i] = t
                    best_proc_assignment = proc_list
                else:
                    best_times[i] = best_time

            # TODO: Extra timing if best time?

        # Broadcast the results
        iter_times = self.comm.bcast(iter_times, root=0)
        times = self.comm.bcast(times, root=0)
        best_times = self.comm.bcast(best_times, root=0)
        best_proc_assignment = self.comm.bcast(best_proc_assignment, root=0)
        return iter_times, times, best_times, best_proc_assignment
