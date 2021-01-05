import pytest

from graph_rl import Session, DummyController, RandomController


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


@pytest.mark.mpi_skip()
def test_time_graph():
    pass


@pytest.mark.mpi_skip()
def test_learn_assignments():
    pass
