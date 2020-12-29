# Run with command: 'mpiexec -np 2 python examples/test_mpi.py'

from mpi4py import MPI
import numpy as np


def main():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # automatic MPI datatype discovery
    data = None
    if rank == 0:
        data = {"Hello": "World", "Data": np.arange(10)}
        comm.send(data, dest=1, tag=13)
    elif rank == 1:
        data = comm.recv(source=0, tag=13)

    print(str(rank + 1) + "/" + str(size) + ": " + str(data))


if __name__ == "__main__":
    main()
