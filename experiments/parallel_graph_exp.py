import os
import pickle
import argparse

from mpi4py import MPI
import numpy as np

import graph_rl
from graph_rl.graph import ParallelGraph
from graph_rl.controllers import DummyController
from graph_rl.session import Session


# Parse command line args
parser = argparse.ArgumentParser(description="Run an experiment with a Parallel Permutation graph")
parser.add_argument(
    "-c",
    "--controller",
    metavar="c",
    type=str,
    default="RandomController",
    help="Name of controller",
)
parser.add_argument(
    "-l", "--layers", metavar="l", type=int, default=10, help="Number of layers in the Graph"
)
parser.add_argument(
    "-w", "--width", metavar="w", type=int, default=6, help="Width of each layer in the Graph"
)
parser.add_argument(
    "-t",
    "--node_time",
    metavar="t",
    type=float,
    default=0.000001,
    help="Sleep time of each node in the Graph",
)
parser.add_argument(
    "-i",
    "--input_length",
    metavar="i",
    type=int,
    default=4000,
    help="Length of the input/output vectors sent between Graph nodes",
)
parser.add_argument(
    "-n",
    "--n_iterations",
    metavar="n",
    type=int,
    default=1000,
    help="Number of timing iterations for search",
)
parser.add_argument(
    "-r",
    "--timing_runs",
    metavar="r",
    type=int,
    default=1,
    help="Graph execution time is calculated as the median of r runs",
)
parser.add_argument(
    "-d",
    "--experiment_dir",
    metavar="d",
    type=str,
    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "pickle"),
    help="Directory in which to save results",
)
parser.add_argument(
    "-e",
    "--experiment_name",
    metavar="e",
    type=str,
    required=True,
    help="Name of save file",
)
args = parser.parse_args()

# Check save path
assert os.path.isdir(args.experiment_dir)
save_path = os.path.join(args.experiment_dir, args.experiment_name)
assert not os.path.isfile(save_path)

# Initialize graph
comm = MPI.COMM_WORLD
graph = ParallelGraph(args.layers, args.width, comm, node_kwargs={"t": args.node_time})
x = np.ones(args.input_length)

# Initialize controllers
if comm.rank == 0:
    controller_module = getattr(graph_rl, args.controller)
    controller = controller_module()
else:
    controller = DummyController()

# Initialize Session
sess = Session(comm, controller)
iter_times, times, best_times, best_proc_assignment = sess.learn_assignments(
    x, graph, n_iter=args.n_iterations, timing_runs=args.timing_runs
)
if comm.rank == 0:

    # Save results to disk
    pkl = {
        "parameters": vars(args),
        "results": {
            "best_proc_assignment": best_proc_assignment,
            "iter_times": iter_times,
            "times": times,
            "best_times": best_times,
        },
    }
    with open(save_path, "wb") as f:
        pickle.dump(pkl, f)
