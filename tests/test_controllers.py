import pytest

from graph_rl.controllers import DummyController, RandomController


@pytest.mark.mpi_skip()
def test_dummy_controller():
    controller = DummyController()
    assert controller.pick_procs(None, None) is None
    assert controller.pick_procs("", 42) is None
    assert controller.pick_procs(1.8, None) is None


@pytest.mark.mpi_skip()
def test_random_controller(mocker):
    mock_graph = mocker.Mock()
    mock_graph.nodes = [0, 1, 2]
    n_procs = 6
    controller = RandomController()
    for i in range(10):
        proc_list = controller.pick_procs(mock_graph, n_procs)
        assert min(proc_list) >= 0
        assert max(proc_list) < n_procs
        assert len(proc_list) == len(mock_graph.nodes) == 3
