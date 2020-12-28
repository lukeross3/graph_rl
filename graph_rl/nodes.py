import time
from abc import ABC, abstractmethod
from typing import Any, List


class Node(ABC):
    """Abstract base class for the basic unit of computation in a computation graph."""

    def __init__(self):
        self.input_dependencies = []
        self.output_dependencies = []
        self.output = None

    @abstractmethod
    def forward(self, x: Any) -> Any:
        """Run the computation on input x. Note the default implementation sets the output member
        and returns the input, so overriding classes should compute the output y, then return
        super().forward(y) in order to set the relevant members.

        Args:
            x (Any): Input to computation

        Returns:
            Any: Output of computation
        """
        self.output = x
        return x

    def reset(self):
        """Reset the Node's output value to be None"""
        self.output = None


class AddOne(Node):
    """Testing Node which adds 1 to the scalar input value"""

    def forward(self, x: float) -> float:
        """Add one to the scalar input

        Args:
            x (float): Input scalar

        Returns:
            float: Output scalar
        """
        y = x + 1
        return super().forward(y)


class Timing(Node):
    """Testing node which only returns its original input after sleeping for some set time period"""

    def __init__(self, t: float = 0.01) -> None:
        """Initialize the node

        Args:
            t (float, optional): Seconds to sleep before returning the forward call. Defaults to
                0.01.
        """
        super().__init__()
        self.t = t

    def forward(self, x: Any) -> Any:
        """Sleep for t seconds before returning the original input x.

        Args:
            x (Any): Input to the node

        Returns:
            Any: Output from the node
        """
        time.sleep(self.t)
        return super().forward(x)
