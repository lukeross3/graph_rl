from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt

from graph_rl.graph import ParallelGraph
from graph_rl.controllers import RandomController, DummyController
from graph_rl.session import Session


# Initialize graph
comm = MPI.COMM_WORLD
graph = ParallelGraph(10, 6, comm)
x = np.ones(1)

# Initialize controllers
if comm.rank == 0:
    controller = RandomController()
else:
    controller = DummyController()

# Initialize Session
sess = Session(comm, controller)
iter_times, times, best_times, best_proc_assignment = sess.learn_assignments(
    x, graph, n_iter=100, timing_runs=1
)
if comm.rank == 0:
    # Print best proc assignment
    print(f"best_proc_assignment: {best_proc_assignment}")

    # Plot results over iterations
    plt.scatter(np.arange(len(times)), times, label="times")
    plt.plot(best_times, "-r", label="best_times")
    plt.legend(loc="best")
    plt.xlabel("Simulation Iteration")
    plt.ylabel("Graph Execution Time (s)")
    plt.savefig("results_over_iters.png", dpi=200)

    # Plot results over time
    plt.cla()
    plt.scatter(iter_times, times, label="times")
    plt.plot(iter_times, best_times, "-r", label="best_times")
    plt.legend(loc="best")
    plt.xlabel("Simulation Time (s)")
    plt.ylabel("Graph Execution Time (s)")
    plt.savefig("results_over_time.png", dpi=200)
