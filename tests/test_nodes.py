from graph_rl import Node, Timing


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
