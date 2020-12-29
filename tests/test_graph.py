import pytest

from graph_rl import Graph, AddN


@pytest.mark.mpi_skip()
def test_parse_graph():

    # Real test case
    nodes = [AddN(1) for _ in range(4)]
    connections = [
        (-1, 0),
        (0, 1),
        (1, 2),
        (2, 3),
    ]
    comm = None
    g = Graph(nodes, connections, comm)
    assert len(g.nodes) == 4
    assert g.nodes[0].input_dependencies == [-1]
    assert g.nodes[0].output_dependencies == [1]
    assert g.nodes[1].input_dependencies == [0]
    assert g.nodes[1].output_dependencies == [2]
    assert g.nodes[2].input_dependencies == [1]
    assert g.nodes[2].output_dependencies == [3]
    assert g.nodes[3].input_dependencies == [2]
    assert g.nodes[3].output_dependencies == []

    # Check that the right exceptions are raised
    connections = [(3, 1)]
    with pytest.raises(ValueError) as e:
        g = Graph(nodes, connections, comm)
    assert "Input index must be greater than output index" in str(e.value)

    connections = [(1, 1)]
    with pytest.raises(ValueError) as e:
        g = Graph(nodes, connections, comm)
    assert "no self-loops allowed in a directed acyclic graph (tree)" in str(e.value)


@pytest.mark.mpi(min_size=2)
def test_proc_init():

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
    assert g.my_node_indices is None
    assert g.proc_list is None

    # Assign nodes to processors
    proc_list = [0, 1, 0, 1]
    g.proc_init(proc_list)
    assert g.proc_list == proc_list
    if g.comm.rank == 0:
        assert g.my_node_indices == [0, 2]
    elif g.comm.rank == 1:
        assert g.my_node_indices == [1, 3]
    else:
        assert g.my_node_indices == []

    # Re-assign nodes
    proc_list = [0, 1, 1, 1]
    g.proc_init(proc_list)
    assert g.proc_list == proc_list
    if g.comm.rank == 0:
        assert g.my_node_indices == [0]
    elif g.comm.rank == 1:
        assert g.my_node_indices == [1, 2, 3]
    else:
        assert g.my_node_indices == []

    # Check for exceptions
    with pytest.raises(ValueError) as e:
        g.proc_init([0, 1, 0])
    assert "Got 3 processor assignments for 4 nodes" == str(e.value)


@pytest.mark.mpi(min_size=2)
def test_forward():

    # Import MPI
    from mpi4py import MPI

    # Init the graph
    nodes = [AddN(i) for i in range(5)]
    connections = [(-1, 0), (0, 1), (0, 2), (2, 3), (3, 4)]  # Dead end node!
    comm = MPI.COMM_WORLD
    g = Graph(nodes, connections, comm)

    # Assign nodes to processors (Both MPI reqs and same node copies)
    proc_list = [0, 0, 1, 1, 0]
    g.proc_init(proc_list)

    # Check the forward pass
    assert g.forward(0) == 9

    # Check the nodes' outputs
    for i, (node) in enumerate(g.nodes):
        if i in g.my_node_indices:  # Check only the nodes on current processor
            if i == 1:  # Dead end node
                assert node.output is None
            else:  # Any computed node
                assert node.output is not None

    # Reset the graph's output values
    g.reset()
    for node in g.nodes:
        assert node.output is None

    # Assign nodes new procs and run again on new input
    proc_list = [1, 0, 1, 1, 1]
    g.proc_init(proc_list)
    assert g.forward(10) == 19

    # Check the nodes' outputs
    for i, (node) in enumerate(g.nodes):
        if i in g.my_node_indices:  # Check only the nodes on current processor
            if i == 1:  # Dead end node
                assert node.output is None
            else:  # Any computed node
                assert node.output is not None
