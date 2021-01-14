import pytest
import numpy as np

from graph_rl import Session, DummyController, RandomController, AddN, Graph


def _is_monotonic_increasing(x: np.array) -> bool:
    return np.all(np.diff(x) >= 0)


def _is_monotonic_decreasing(x: np.array) -> bool:
    return np.all(np.diff(x) <= 0)


@pytest.mark.mpi_skip()
def test_init(mocker):

    # Input vars
    mock_comm = mocker.Mock()
    mock_comm.rank = 0
    random_controller = RandomController()
    dummy_controller = DummyController()

    # Master proc with non-dummy controller
    sess = Session(mock_comm, random_controller)
    assert isinstance(sess.controller, RandomController)
    assert sess.comm.rank == 0

    # Master proc with dummy controller
    with pytest.raises(ValueError) as e:
        sess = Session(mock_comm, dummy_controller)
    assert "Controller on master process (rank == 0) cannot be instance of DummyController" == str(
        e.value
    )

    # Worker proc with non-dummy controller
    mock_comm.rank = 1
    with pytest.raises(ValueError) as e:
        sess = Session(mock_comm, random_controller)
    assert (
        "Controller on non-master process (rank != 0) must be instance of DummyController"
        == str(e.value)
    )

    # Worker proc with dummy controller
    sess = Session(mock_comm, dummy_controller)
    assert isinstance(sess.controller, DummyController)
    assert sess.comm.rank == 1


@pytest.mark.mpi(min_size=2)
def test_time_graph():

    # Import MPI
    from mpi4py import MPI

    # Init the graph
    nodes = [AddN(1) for _ in range(4)]
    connections = [
        (-1, 0),
        (0, 1),
        (1, 2),
        (2, 3),
    ]
    comm = MPI.COMM_WORLD
    g = Graph(nodes, connections, comm)

    # Assign nodes to processors
    proc_list = [0, 1, 0, 1]
    g.proc_init(proc_list)

    # Init the session object
    if comm.rank == 0:
        controller = RandomController()
    else:
        controller = DummyController()
    sess = Session(comm, controller)

    # Time once
    t = sess.time_graph(42, g, n=1)
    assert isinstance(t, float)

    # Time as median of 3 runs
    t = sess.time_graph(42, g, n=3)
    assert isinstance(t, float)


@pytest.mark.mpi(min_size=2)
def test_learn_assignments():

    # Import MPI
    from mpi4py import MPI

    # Init the graph
    nodes = [AddN(1) for _ in range(4)]
    connections = [
        (-1, 0),
        (0, 1),
        (1, 2),
        (2, 3),
    ]
    comm = MPI.COMM_WORLD
    g = Graph(nodes, connections, comm)

    # Init the session object
    if comm.rank == 0:
        controller = RandomController()
    else:
        controller = DummyController()
    sess = Session(comm, controller)

    # Learn assignments with default values
    n_iter = 10
    iter_times, times, best_times, best_proc_assignment = sess.learn_assignments(
        42, g, n_iter=n_iter
    )
    assert len(iter_times) == len(times) == len(best_times) == n_iter
    assert len(best_proc_assignment) == len(nodes)
    assert _is_monotonic_increasing(iter_times)
    assert _is_monotonic_decreasing(best_times)
    assert not (_is_monotonic_increasing(times) or _is_monotonic_decreasing(times))
