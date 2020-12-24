import time
from abc import ABC, abstractmethod
from typing import Any, List


class Node(ABC):
    """Abstract base class for the basic unit of computation in a computation graph."""

    def __init__(self):
        self.input_dependencies = None
        self.output_dependencies = None
        self.output = None

    @abstractmethod
    def forward(self, x: Any) -> Any:
        """Run the computation on input x. Note the default implementation sets the output member
        and returns the input, so overriding classes should compute the output y, then return
        super().fordward(y) in order to set the relevant members.

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
