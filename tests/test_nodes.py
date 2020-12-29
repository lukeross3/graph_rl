import pytest

from graph_rl import Timing, AddN


@pytest.mark.mpi_skip()
def test_addN_node():

    # Initialize the node
    n = 1.5
    node = AddN(n=n)
    assert node.output is None
    assert node.n == n

    # Run forward pass
    x = 42
    result = node.forward(x)
    assert result == x + n
    assert node.output == x + n

    # Reset the node
    node.reset()
    assert node.output is None


@pytest.mark.mpi_skip()
def test_timing_node(mocker):

    # Initialize the node
    sleep_time = 100
    node = Timing(t=sleep_time)
    assert node.output is None
    assert node.t == sleep_time

    # Patch the time.sleep call
    mock_sleep = mocker.patch("graph_rl.nodes.time.sleep", autospec=True)

    # Run forward pass
    x = 42
    result = node.forward(x)
    assert result == x
    assert node.output == x
    mock_sleep.assert_called_with(sleep_time)

    # Reset the node
    node.reset()
    assert node.output is None
