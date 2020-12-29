import pytest

from graph_rl import __version__


@pytest.mark.mpi_skip()
def test_version():
    assert __version__ == "0.1.0"
