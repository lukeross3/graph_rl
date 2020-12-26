import os
import toml

pyproj = toml.load(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.pardir, "pyproject.toml")
)
__version__ = pyproj["tool"]["poetry"]["version"]

from .nodes import Node, AddOne, Timing
from .graph import Graph
