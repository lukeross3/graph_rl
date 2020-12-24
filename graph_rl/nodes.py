from abc import ABC, abstractmethod
from typing import Any, List

class Node(ABC):
    """Abstract base class for the basic unit of computation in a computation graph.
    """

    def __init__(self):
        self._input_dependencies = None
        self._output_dependencies = None
        self._output = None

    def set_input_dependencies(self, input_dependencies: List[int]) -> None:
        self._input_dependencies = input_dependencies

    @abstractmethod
    def forward(self, x: Any) -> Any:
        """Run the computation on input x.

        Args:
            x (Any): Input to computation

        Returns:
            Any: Output of computation
        """
