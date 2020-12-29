from typing import List, Tuple, Any

from mpi4py import MPI

from graph_rl import Node


class Graph:
    """Implements the computation graph with MPI. Requires a list of nodes, a list of their
    connections, and the MPI Comm object for parallel computing and message passing.

    NOTE: The input dependencies for each node will be set in the order they are provided. For
        this reason, it is recommended that Nodes not use multiple inputs when the Node operation
        is not commutative, because the order of connections when initializing the graph could
        change the order of input dependencies, changing the execution of the graph. Named key
        value pairs in python dicts provide a better programming interface for nodes in the graph.
        The output node simply names its output(s) and returns it/them as a python dict. Subsequent
        Nodes can then extract the value(s) based on expected names instead of argument order.
    """

    def __init__(
        self, nodes: List[Node], connections: List[Tuple[int, int]], comm: MPI.Comm
    ) -> None:
        self.nodes = nodes
        self.comm = comm
        self._parse_graph(connections)
        self.my_node_indices = None
        self.proc_list = None

    def _parse_graph(self, connections: List[Tuple[int, int]]) -> None:
        """Assigns the input and output dependencies of each node in the graph, given a list of
        individual ordered connections.

        Args:
            connections (List[Tuple[int, int]]): A list of dependencies. Each element is a tuple,
                where the first element is the source node, and the second element is the target
                (i.e. the second node depends on the first node's output).
        """
        # Loop over connections
        for (input_idx, output_idx) in connections:

            # Check that the connection is valid
            if input_idx > output_idx:
                raise ValueError(
                    f"Invalid connection ({input_idx}, {output_idx}): Input index must be greater "
                    "than output index to ensure a directed acyclic graph (tree)."
                )
            if input_idx == output_idx:
                raise ValueError(
                    f"Invalid connection ({input_idx}, {output_idx}): no self-loops allowed in a "
                    "directed acyclic graph (tree)."
                )

            # Special case for nodes requiring the first input
            if input_idx == -1:
                self.nodes[output_idx].input_dependencies.append(input_idx)

            # Register connections with source and target nodes
            else:
                self.nodes[input_idx].output_dependencies.append(output_idx)
                self.nodes[output_idx].input_dependencies.append(input_idx)

    def proc_init(self, proc_list: List[int]) -> None:
        """Assign each node to its respective processor

        Args:
            proc_list (List[int]): List of processor indexes such that node i gets assigned to the
                processor with rank proc_list[i]
        """

        # Validate input
        if len(proc_list) != len(self.nodes):
            raise ValueError(
                f"Got {len(proc_list)} processor assignments for {len(self.nodes)} nodes"
            )

        # Store list of processor assignments
        self.proc_list = proc_list

        # Pick subset of nodes to be run on the current processor
        self.my_node_indices = [idx for idx, proc in enumerate(proc_list) if proc == self.comm.rank]

    def forward(self, x: Any) -> Any:
        """Run the computation graph on input x.

        Args:
            x (Any): Input to run through graph

        Returns:
            Any: Output from graph
        """
        # Loop through nodes on the current processor
        for node_idx in self.my_node_indices:

            # If no one needs output, skip the node
            output_dependencies = self.nodes[node_idx].output_dependencies
            if len(output_dependencies) == 0 and node_idx != len(self.nodes) - 1:
                continue

            # Get all required inputs for the current node before running the node
            input_dependencies = self.nodes[node_idx].input_dependencies
            reqs = []
            req_ids = []
            inputs = [None for _ in range(len(input_dependencies))]

            # Loop over each required input
            for i in range(len(input_dependencies)):
                input_node = input_dependencies[i]

                # Input from same processor, no need for MPI
                if input_node in self.my_node_indices:
                    inputs[i] = self.nodes[input_node].output

                # Input layer, just grab from function input
                elif input_node == -1:
                    inputs[i] = x

                # Need input from another processor: post a receive for data (non-blocking)
                else:
                    reqs.append(self.comm.irecv(source=self.proc_list[input_node]))
                    req_ids.append(i)

            # Wait until all inputs available then gather them
            responses = MPI.Request.waitall(reqs)
            for resp, req_id in zip(responses, req_ids):
                inputs[req_id] = resp

            # Run forward computation
            output = self.nodes[node_idx].forward(*inputs)

            # Send output to each node requiring it
            for next_node in output_dependencies:

                # Only send if to a different node
                if next_node not in self.my_node_indices:
                    self.comm.isend(output, dest=self.proc_list[next_node])

        # Broadcast the last value to the other nodes
        root_rank = self.proc_list[-1]
        if self.comm.rank == root_rank:
            output = self.nodes[-1].output
        else:
            output = None
        output = self.comm.bcast(output, root=root_rank)
        return output

    def reset(self):
        """Reset each node in the graph, deleting any stored output_data"""
        for i in range(len(self.nodes)):
            self.nodes[i].reset()
