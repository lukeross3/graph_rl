import pytest
from mpi4py import MPI

from graph_rl import Graph, AddN


@pytest.mark.mpi(min_size=2)
def test_parse_graph():
    nodes = [AddN(1) for _ in range(4)]
    connections = [
        (-1, 0),
        (0, 1),
        (1, 2),
        (2, 3),
    ]
    comm = MPI.COMM_WORLD
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
