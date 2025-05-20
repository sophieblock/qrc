"""

This file contains class definitions for a directed acyclic graph
(DAG) data structure for workflow resource estimation. Individual
tasks and processes are contained in their respective Nodes which
are connected with DirectedEdges. Nodes contain the Process necessary
for executing and updating the subtask. DirectedEdges contain the 
relevant input/output data that is passed along the Network.
"""

from typing import List, Tuple, Dict
from collections import OrderedDict
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import copy
from numpy import inf as INFINITY
from .process import Process, ClassicalProcess, QuantumProcess, InitError
from .resources.resources import Allocation
from .resources.classical_resources import ClassicalAllocation, ClassicalResource
from .resources.quantum_resources import QuantumAllocation, QuantumResource
from .broker import Broker
from .data import Data, Result
from .utilities import (
    get_next_id,
    all_dict1_vals_in_dict2_vals,
    any_dict1_vals_in_dict2_vals,
    publish_gantt,
)


import datetime
from ...util.log import get_logger
logger = get_logger(__name__)


class DirectedEdge:
    """The "directed edge" connecting Nodes in a DAG. A DirectedEdge
    may have up to one source node and any number of destination nodes
    but must be connected to atleast one node at any given instance.
    The DirectedEdge class contains and propagates the necessary
    Data for Node execution and additional information regarding
    its connectivity and status. DirectedEdges with multiple destination
    nodes propagate the same Data to all destination nodes.
    """

    # Specify self.status
    INCOMPLETE = "INCOMPLETE"
    COMPLETED = "COMPLETED"

    # Specify self.edge_type
    INPUT = "INPUT"
    CONNECTED = "CONNECTED"
    OUTPUT = "OUTPUT"

    def __init__(
        self,
        data=None,
        edge_type=CONNECTED,
        source_node=None,
        dest_nodes=None,
    ):

        if data != None:
            assert isinstance(
                data, tuple
            ), f"DirectedEdge may only accept data as tuples"
        self.data: Tuple[Data] = data
        self.edge_type: str = edge_type
        self.source_node: Node = source_node
        self.dest_nodes: list[Node] = dest_nodes or []

        self.verify_connectivity()

    def update_data(self, data):
        self.data = data if isinstance(data, tuple) else (data,)

    def update_source_node(self, new_source):
        self.source_node = new_source
        if self.edge_type == DirectedEdge.INPUT:
            self.edge_type = DirectedEdge.CONNECTED
        self.verify_connectivity()

    def insert_destination_node(self, new_dest_node):
        if new_dest_node not in self.dest_nodes:
            self.dest_nodes.append(new_dest_node)

        if self.edge_type == DirectedEdge.OUTPUT:
            self.edge_type = DirectedEdge.CONNECTED
        self.verify_connectivity()

    def remove_destination_node(self, dest_node):
        self.dest_nodes.remove(dest_node)

        if self.dest_nodes == [] and self.edge_type == DirectedEdge.CONNECTED:
            self.edge_type = DirectedEdge.OUTPUT
        self.verify_connectivity()

    def verify_connectivity(self):
        if self.edge_type == DirectedEdge.CONNECTED:
            assert isinstance(
                self.source_node, Node
            ), f"Edge {self} of edge type {self.edge_type} is missing a source node"
            assert self.dest_nodes != [] and all(
                isinstance(dest_node, Node) for dest_node in self.dest_nodes
            ), f"Edge {self} of edge type {self.edge_type} is missing a destination node"
        elif self.edge_type == DirectedEdge.INPUT:
            assert (
                self.source_node is None
            ), f"Edge {self} of edge type {self.edge_type} should not have a source node"
            assert self.dest_nodes != [] and all(
                isinstance(dest_node, Node) for dest_node in self.dest_nodes
            ), f"Edge {self} of edge type {self.edge_type} is missing a destination node"
        elif self.edge_type == DirectedEdge.OUTPUT:
            assert isinstance(
                self.source_node, Node
            ), f"Edge {self} of edge type {self.edge_type} is missing a source node"
            assert (
                self.dest_nodes == []
            ), f"Edge {self} of edge type {self.edge_type} should not have a destination node"
        else:
            raise ValueError(
                f"""Edge type {self.edge_type} is invalid, only {DirectedEdge.INPUT}, 
                             {DirectedEdge.CONNECTED} or {DirectedEdge.OUTPUT} are supported."""
            )

    def __add__(self, other):
        assert (
            self.source_node == other.source_node
        ), f"Directed edges {self} and {other} do not have the same source node"
        assert (
            self.dest_nodes == other.dest_nodes
        ), f"Directed edges {self} and {other} do not have the same destination node"
        # assert (
        #     self.status == other.status
        # ), f"Directed edges {self} and {other} do not have the same status"
        assert (
            self.edge_type == other.edge_type
        ), f"Directed edges {self} and {other} do not have the same edge type"

        return DirectedEdge(
            data=self.data + other.data,
            source_node=self.source_node,
            dest_nodes=self.dest_nodes,
            edge_type=self.edge_type,
        )
    def __repr__(self) -> str:
        if self.edge_type == "OUTPUT":
            usage_list = [d.properties.get('Usage','?') for d in self.data] if self.data else []
            return f'{repr(self.source_node)} (usage={usage_list}) -> OUT'
        elif self.edge_type == "INPUT":
            usage_list = [d.properties.get('Usage','?') for d in self.data] if self.data else []
            return f'in data: {usage_list} -> {self.dest_nodes}'
        return f'{repr(self.source_node)} -> {self.dest_nodes}'


class ExpectedNodeError(Exception):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class EdgeAssignmentError(Exception):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Node:
    """Represents, stores and manages information on the most basic process/task
    of the larger network. Wraps and runs the entire process of verifying, simulating and
    storing data for the given process. Can be aggregated along with Edges to form a larger
    Network specifying a larger, more complex and fixed algorithm class.
    """

    # Specifies self.network_type
    INPUT = "INPUT"
    NETWORK = "NETWORK"
    OUTPUT = "OUTPUT"

    def __init__(
        self,
        id=None,
        process_model=None,
        network_type=None,
        input_edges=None,
        output_edges=None,
        output_nodes=None,
        extend_dynamic=False,
        **kwargs,
    ):
        """
        Args:
            id (str, optional): id of the Node, constructed using get_next_id() if not specified. Defaults to None.
            process_model (Process, optional): The process model this Node simulates. Defaults to None.
            network_type (str, optional): Type of Node inside a larger Network, can be INPUT, NETWORK or OUTPUT. Defaults to None.
            input_edges (List[DirectedEdge], optional): List of incoming DirectedEdges to the Node. Defaults to [].
            output_edges (List[DirectedEdge], optional): List of outgoing DirectedEdges from the Node. Defaults to [].
            output_nodes (List[Nodes], optional): List of Nodes to send the process results to. Defaults to [].
        """

        self.id: str = id or get_next_id()
        self.model: Process = process_model
        self.process: Process = process_model(**kwargs) if self.model != None else None
        self.network_type: str = network_type
        self.input_edges: List[DirectedEdge] = input_edges or []
        self.output_edges: List[DirectedEdge] = output_edges or []
        self.output_nodes: List[Node] = output_nodes or []
        self.allocation: Allocation = None
        self.extend_dynamic = extend_dynamic
        self.kwargs = kwargs

        if isinstance(self.process, Process):
            self.expected_input_properties = self.process.expected_input_properties
    def is_process_initialized(self) -> bool:
        """
        Returns True if self.process is fully constructed with a signature 
        and ephemeral rename pass is (likely) done. This is a heuristic 
        that checks several common fields:
        1) self.process is not None
        2) self.process.signature is built
        3) ephemeral rename pass is done 
            (e.g. if final usage-based name is in self.process.normalized_map)
        """
        if not self.process:
            return False

        # If the user’s ephemeral rename pass sets self._signature 
        # and modifies self.normalized_map, check they exist
        # if not self.process.signature:
        #     return False
        if not hasattr(self.process, "initialized"):
            return False
        rights = list(self.process.signature.rights())
        logger.debug(f"{self} with right registers: {rights}")
        return True
    def is_input(self):
        return self.network_type == "INPUT"

    def is_network(self):
        return self.network_type == "NETWORK"

    def is_output(self):
        return self.network_type == "OUTPUT"

    def is_none(self):
        return self.network_type is None

    def append_input_edge(self, edge: DirectedEdge):
        assert (
            not edge in self.input_edges
        ), f"Edge {edge} already exists as an input edge to Node {self.id}"
        assert (
            self in edge.dest_nodes
        ), f"Cannot append Edge to Node {self} that is assigned to {edge.dest_nodes}"
        self.input_edges.append(edge)

    def insert_output_node(self, output_node):
        assert isinstance(output_node, Node), f"{output_node} must be a Node object"
        self.__validate_connection(output_node)

        if output_node in self.output_nodes:
            return
        if self.network_type == Node.OUTPUT:
            assert (
                self.output_nodes == []
            ), f"Node {self.id} is classified as an OUTPUT Node but has non-empty list of output nodes: {self.output_nodes}"
            self.network_type = Node.NETWORK
            output_node.network_type = Node.OUTPUT

        self.output_nodes.append(output_node)
        if output_node.network_type is None:
            output_node.network_type = (
                Node.OUTPUT if len(output_node.output_nodes) == 0 else Node.NETWORK
            )
        # if output_node.network_type != Node.OUTPUT:
        #     output_node.network_type = Node.NETWORK

    def __validate_connection(self, dest):
        assert isinstance(dest, Node), f"{dest} must be a Node object"
        if self.is_output() and not dest.is_output():
            raise ValueError(
                f"Attempting to connect an OUTPUT Node to a {dest.network_type} Node"
            )
        # elif self.is_network() and dest.is_input():
        #     raise ValueError(f"Attempting to connect a NETWORK Node to an INPUT Node")

    def check_if_previously_completed(self) -> bool:
        if self.output_edges != [] and all(
            output.status == DirectedEdge.COMPLETED for output in self.output_edges
        ):
            return True
        return False

    def start(self):
        """Prepares the Node for process execution. This method
        validates the incoming data, initializes the specified Process
        and maps empty, INCOMPLETE DirectedEdges to its output_nodes.

        Should only start when the desired list of output_nodes are
        fully specified, otherwise outgoing DirectedEdges will not
        map correctly.
        """

        self.__initialize_process_with_inputs()

        assert (
            self.process.validate_data()
        ), f"Invalid input data for process {str(self.process)}"

        self.start_time = datetime.datetime.now()
        self.total_duration = 0.0
        # logger.debug(f"Starting node {repr(self)} with inputs: {[d for d in self.process.inputs]}")
        # logger.debug(f"Starting node {self.id} at t={self.start_time}")
    def check_if_ready(self):
        """
        checks if this node is ready to start or not

        This method should evaluate the node inputs (list of Data instances) 
        against the process.signature (i.e. the RegisterSpec's). If correct,
        then the node should be marked for starting. 
        Returns True if this Node’s incoming Data satisfy its Process’s
        expected_input_properties (via validate_data_properties), False otherwise.
        Does not raise; simply peeks at validity.
        """
        # 1) gather all Data from incoming edges
        inputs = [d for edge in self.input_edges for d in edge.data]

        # 2) temporarily assign into the existing process and test
        orig_inputs = self.process.inputs
        self.process.inputs = inputs
        try:
            ok = self.process.validate_data_properties(show_debug_log=False)
        except InitError:
            ok = False
        # 3) restore
        self.process.inputs = orig_inputs
        return ok

    def ensure_ready(self):
        """
        Validates inputs by actually instantiating the Process.
        On success, it replaces self.process with the new instance,
        sets its status to Process.READY, and returns None.
        On failure, it propagates the InitError with full mismatch detail.
        """
        # 1) gather all Data
        inputs = [d for edge in self.input_edges for d in edge.data]

        # 2) inject into our existing process and run the verbose check
        self.process.inputs = inputs
        # this will raise InitError with your pretty log if anything fails
        self.process.validate_data_properties(show_debug_log=True)

        # 3) mark ready
        self.process.status = Process.READY
    def ready_to_start(self):
        # TODO: update to just return self.check_if_ready()
        self.inputs: List[Data] = []
        for input_edge in self.input_edges:
            for data in input_edge.data:
                self.inputs.append(data)

        is_ready = all(
            self.__property_in_input_edge(expected_input_property)
            for expected_input_property in self.process.expected_input_properties
        )
        del self.inputs
        return is_ready
    

    def __property_in_input_edge(self, property: dict):
        for data in self.inputs:
            if all(
                all_dict1_vals_in_dict2_vals(data.properties.get(key, None), val)
                for key, val in property.items()
            ):
                self.inputs.remove(data)
                return True
        return False

    def __initialize_process_with_inputs(self):
        """Aggregates data from incoming DirectedEdges and initializes the Process object"""
        if self.process.status == "ACTIVE":
            raise AssertionError(
                f"ERROR: Attempting to initialize node {str(self)} but is already being executed in the network"
            )
        elif self.process.status == "COMPLETED":
            raise AssertionError(
                f"ERROR: Attempting to initialize node {str(self)} but has already been executed in the network"
            )

        input_data = []
        for input_edge in self.input_edges:
            for data in input_edge.data:
                input_data.append(data)

        self.process = self.model(inputs=input_data, **self.kwargs)
        # del self.model

    def execute_process(self, timestep: float):
        """Simulates the wrapped process entirely. The execute assuming:

        1. Given inputs are valid for the process
        2. Resources are allocated by the broker
        """
        assert self.verify_allocation(
            self.allocation
        ), f"Node {self.id} cannot execute it's process with given allocation {self.allocation}"
        assert (
            self.process.status == Process.ACTIVE
        ), f"Node {self.id} cannot execute it's process with status {self.process.status}"

        self.process.time_till_completion = self.process.time_till_completion - timestep
        assert (
            self.process.time_till_completion >= 0.0
        ), f"{self} has negative time till completion of process execution"

        if self.process.time_till_completion == 0.0:
            # self.process.update()
            self.process.status = Process.COMPLETED

    def verify_allocation(self, allocation: Allocation) -> bool:
        """Return True when *either* no resources are required *or*
        the allocation satisfies the declared requirement.
        """
        required_resource = self.process.required_resources
        # ∅ requirement  →  always satisfied
        if not required_resource:          # None, [], (), {} all count as “no resource”
            return True

        if isinstance(required_resource, ClassicalResource):
            assert isinstance(
                allocation, ClassicalAllocation
            ), (
                "Expected ClassicalAllocation for ClassicalResource "
                f"requirement but got {type(allocation)}"
            )
            conditions = (
                required_resource.memory     <= allocation.allocated_memory,
                required_resource.num_cores  <= allocation.allocated_cores,
            )

        elif isinstance(required_resource, QuantumResource):
            assert isinstance(
                allocation, QuantumAllocation
            ), (
                "Expected QuantumAllocation for QuantumResource "
                f"requirement but got {type(allocation)}"
            )
            conditions = (
                required_resource.circuit.qubit_count
                == len(allocation.allocated_qubit_idxs),
            )
            logger.debug(f'Required number of qubit == {required_resource.circuit.qubit_count}, number of qubits allocated from synthesizer: {len(allocation.allocated_qubit_idxs)}')
        else:
            raise TypeError(
                "required_resources must be a Resource, a sequence of Resources, "
                "or a falsy placeholder signalling ‘no requirement’"
            )

        return all(conditions)

    def complete(self):
        assert (
            self.allocation is None
        ), f"Node {self.id} has not deallocated resources after completing"
        assert (
            self.process.status == Process.COMPLETED
        ), f"Node {self.id} has not completed, cannot mark as complete"

        # self.completion_time = datetime.datetime.now()
        self.generate_output_edges()
        self.map_edges_to_output_nodes()
        # logger.debug(f"Node {self.id} has completed at t={self.completion_time}")

    def generate_output_edges(self):
        """Generates the output edges of a Node after it's Process has completed execution"""

        for result in self.process.generate_output():
            usage_str = result.properties.get("Usage", None)
            # logger.debug(f"     [EdgeGen]  Handling output result: {result} => usage={usage_str} hint={result.metadata.hint}")
            
            self.output_edges.append(
                DirectedEdge(
                    data=(result,),
                    source_node=self,
                    dest_nodes=None,
                    edge_type=DirectedEdge.OUTPUT,
                )
            )
        # logger.debug(f" [EdgeGen]  final output_edges => {self.output_edges}")
    def map_edges_to_output_nodes(self):
        """Maps outgoing DirectedEdges to the next appropriate Nodes in the
        network as specified by self.output_nodes.

        Edges are mapped to output nodes when the data properties contained
        in the DirectedEdge match expected_input_properties of a Node.

        Unassigned Edges are left as 'hanging' edges of the Network.
        """
        if self.output_nodes == []:
            return

        for edge in self.output_edges:
            if edge.edge_type == DirectedEdge.OUTPUT:
                try:
                    expected_nodes = self.get_expected_nodes(edge, self.output_nodes)
                except ExpectedNodeError:
                    continue
                self.__map_edge_to_output_nodes(edge, expected_nodes)

    def get_expected_nodes(self, edge: DirectedEdge, output_nodes):
        """Finds the expected output Node that outgoing DirectedEdge edge
        should connect to based off matching data properties.

        Args:
            edge (DirectedEdge): Outgoing DirectedEdge object to connect to output node
            output_nodes (List[Node]): List of output nodes to search

        Raises:
            ExpectedNodeError: Cannot find a connecting Node with matching data properties

        Returns:
            Node: The expected Node that DirectedEdge edge connects to
        """
        # logger.debug(f" [Hooking] Checking data usage for {edge} => data usage(s): "
        #             f"{[d.properties.get('Usage') for d in edge.data]}")
        assert all(
            isinstance(output_node, Node) for output_node in output_nodes
        ), f"{output_nodes} must be a list of Nodes"
        assert (
            edge.source_node == self
        ), f"Node {self} does not have permission to assign edge of Node {edge.source_node}"
        usage_list = [d.properties.get('Usage','?') for d in edge.data] if edge.data else []
        # logger.debug(f"[Hooking] Checking which nodes require edge {edge}")

        dest_nodes = []
        for node in output_nodes:
            # We'll debug the node's expected_input_properties
            # logger.debug(f" [Hooking] Node {repr(node)}, initialied? {node.is_process_initialized()}:")
            # Then the logic to see if it matches
            if all(self.__edge_properties_in_output_node(edge_data.properties, node)
                for edge_data in edge.data):
                dest_nodes.append(node)
                # logger.debug(f"     => matched node {repr(node)}!")
        if not dest_nodes:
            logger.debug(f" [Hooking] Node {repr(node)}, initialied? {node.is_process_initialized()}:      => NO match among {output_nodes}*")
            logger.debug(f"     => NO match among {output_nodes}*")
            raise ExpectedNodeError(
                f"A mapping between Edge {edge} and Nodes {self.output_nodes} cannot be found"
            )
        return dest_nodes

    def __edge_properties_in_output_node(
        self, data_properties: dict, output_node
    ) -> bool:
        """Checks whether data_properties of a DirectedEdge are contained in
        the expected_input_properties of output_node.

        Args:
            data_properties (dict): Data properties of a DirectedEdge object
            output_node (Node): Output Node that may or may not contain the expected data_properties
        """
        assert isinstance(output_node, Node), f"{output_node} must be a Node object"

        for input_properties in output_node.expected_input_properties:
            if all(
                all_dict1_vals_in_dict2_vals(data_properties.get(key, None), val)
                for key, val in input_properties.items()
            ):
                return True
        return False

    def __map_edge_to_output_nodes(self, edge: DirectedEdge, output_nodes):
        """Maps edge to the specified output_node. The edge must be of type
        OUTPUT and output_node must be in the list of output_nodes. Can be used
        to manually override automatic assignments of edges to nodes.
        """
        assert all(
            isinstance(output_node, Node) for output_node in output_nodes
        ), f"{output_nodes} must contain Node objects"
        assert all(
            output_node in self.output_nodes for output_node in output_nodes
        ), f"Attempting to map edge to Node outside list of accepted output_nodes"

        for output_node in output_nodes:
            if output_node not in edge.dest_nodes:
                edge.insert_destination_node(output_node)
                output_node.append_input_edge(edge)

    @property
    def describe(self):
        input_ids = [edge.source_node for edge in self.input_edges]
        output_ids = [node for node in self.output_nodes]
        description = (
            f"Node ID: {self.id}\n"
            f" - Process Model: {self.process if self.process else None}\n"
            f" - Network Type: {self.network_type}\n"
            f" - Input Edges: {[str(edge) for edge in self.input_edges]}\n"
            f" - Output Edges: {[str(edge) for edge in self.output_edges]}\n"
            f" - Output Nodes: {output_ids}\n"
            # f" - Process.signature: {self.process.signature}\n"
            # f" - Process.signature.left: {self.process.signature._lefts}\n"
            # f" - Process.signature.right: {self.process.signature._rights}\n"
            f" - Allocation: {self.allocation}\n"
        )
        logger.debug(description)
    def __str__(self):
        return self.id + ": " + self.process.__class__.__name__

    def __repr__(self):
        return self.process.__class__.__name__ + "." + self.id
def network_run_helper(network, starting_nodes,starting_inputs):
    assert all(
        node.network_type == Node.INPUT for node in starting_nodes
    ), f"Given Nodes {starting_nodes} must all be INPUT nodes of the Network"

    if isinstance(starting_inputs, list):
        assert len(starting_nodes) == len(
            starting_inputs
        ), f"Each Node in starting_nodes must have a corresponding Data object(s) in starting_inputs"

        for idx in range(len(starting_nodes)):
            inputs = starting_inputs[idx]
            node = starting_nodes[idx]
            input_edge = DirectedEdge(
                data=tuple(inputs),
                edge_type="INPUT",
                source_node=None,
                dest_nodes=[node],
            )
            node.append_input_edge(input_edge)
    else:
        assert all(
            len(node.input_edges) > 0 for node in starting_nodes
        ), f"One or more Nodes in starting_nodes is not assigned an input DirectedEdge"

    network.reset_network()
    return network, starting_nodes

class AllocationError(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Network:
    """Collection of connected Nodes and Edges to represent
    larger, more complex but fixed algorithms. Smaller Networks
    can be summed together to form larger and more complex Networks
    assuming the outputs of one Network are valid inputs of the other.
    """
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"

    def __init__(
        self,
        name="default",
        nodes=None,
        input_nodes=None,
        output_nodes=None,
        broker: Broker = None,
        status = "INACTIVE",
    ):
        """
        Args:
            name (str): Name of the Network
            nodes (List[Node]): All the nodes contained in the Network. Defaults to None.
            input_nodes (List[Node]): List of input Nodes for the Network intended to be the starting Node for a workflow. Defaults to None.
            output_nodes (List[Node]): List of output Nodes for the Network intended to be the final Node for a workflow. Defaults to None.
            broker (Broker): Broker used for resource allocation. Defaults to None.
        """
        self.name: str = name
        self.nodes: List[Node] = nodes or []
        self.input_nodes: List[Node] = input_nodes or []
        self.output_nodes: List[Node] = output_nodes or []
        self.broker: Broker = broker or Broker()
        self.execution_order: List[Node] = []
        self.status = status

        # assert all(
        #     input_node.network_type == Node.INPUT for input_node in self.input_nodes
        # ), f"One or more input_nodes do not have network_type INPUT"
        # assert all(
        #     output_node.network_type == Node.OUTPUT for output_node in self.output_nodes
        # ), f"One or more output_nodes do not have network_type OUTPUT"

    def add_connection(self, source: Process | Node, dest: Process | Node):
        """Adds a connection between source and destination in the network. Inputs
        can either be a Process class or a Node instance if additional kwargs are necessary.
        """

        source_node = self.__init_input_as_node(source)
        dest_node = self.__init_input_as_node(dest)

        if source_node in self.nodes and dest_node in self.nodes:
            # self.__validate_connection(source_node, dest_node)
            pass

        elif source_node in self.nodes:
            if source_node.is_output():
                self.output_nodes.remove(source_node)
                source_node.network_type = "NETWORK"
            dest_node.network_type = "OUTPUT"
            self.nodes.append(dest_node)
            self.output_nodes.append(dest_node)

        elif dest_node in self.nodes:
            if dest_node.is_input():
                self.input_nodes.remove(dest_node)
                if len(dest_node.output_nodes) == 0:
                    dest_node.network_type = "OUTPUT"
                    self.output_nodes.append(dest_node)
                else:
                    dest_node.network_type = "NETWORK"
            source_node.network_type = "INPUT"
            self.nodes.append(source_node)
            self.input_nodes.append(source_node)

        else:
            self.nodes.append(source_node)
            self.nodes.append(dest_node)

            if source_node.is_none() or dest_node.is_none():
                raise ValueError(
                    "Both source and dest nodes must have a specified network_type"
                )

                # source_node.network_type = "INPUT"
                # self.input_nodes.append(source_node)

                # dest_node.network_type = "OUTPUT"
                # self.output_nodes.append(dest_node)

        source_node.insert_output_node(dest_node)

    def __init_input_as_node(self, input: Process | Node):
        if isinstance(input, Process):
            return Node(process_model=input)
        elif isinstance(input, Node):
            return input
        else:
            raise ValueError(
                f"Expected either a Process class or Node instance, got {type(input)} instead"
            )

    def extend_input_nodes(self, node: Node):
        if node not in self.input_nodes:
            node.network_type = Node.INPUT
            self.input_nodes.append(node)
            self.nodes.append(node)

    def extend_output_nodes(self, node: Node):
        if node not in self.output_nodes:
            node.network_type = Node.OUTPUT
            self.output_nodes.append(node)
            self.nodes.append(node)

    def extend_node_destination(self, nodes: List[Node], output_node: Node):
        """For all nodes in nodes, insert output_node into the list of output
        nodes. That is, all nodes in nodes have output_node as an output_node.
        Automatically updates network_types if necessary.
        """
        nodes_cpy = copy.copy(nodes)
        for node in nodes_cpy:
            if node.network_type == Node.OUTPUT:
                assert (
                    node in self.output_nodes
                ), f"Node {node} of network type OUTPUT is not in the list of output_nodes for {self.name}"
                node.network_type == Node.NETWORK
                self.output_nodes.remove(node)
                self.extend_output_nodes(output_node)

            node.insert_output_node(output_node)

    def find_node_with_id(self, id):
        for node in self.nodes:
            if node.id == id:
                return node

    def update_broker(self, broker):
        self.broker = broker
    def network_prep(self, starting_nodes, starting_inputs):
        assert all(
            node.network_type == Node.INPUT for node in starting_nodes
        ), f"Given Nodes {starting_nodes} must all be INPUT nodes of the Network"

        if isinstance(starting_inputs, list):
            assert len(starting_nodes) == len(
                starting_inputs
            ), f"Each Node in starting_nodes must have a corresponding Data object(s) in starting_inputs"

            for idx in range(len(starting_nodes)):
                inputs = starting_inputs[idx]
                node = starting_nodes[idx]
                input_edge = DirectedEdge(
                    data=tuple(inputs),
                    edge_type="INPUT",
                    source_node=None,
                    dest_nodes=[node],
                )
                node.append_input_edge(input_edge)
        else:
            assert all(
                len(node.input_edges) > 0 for node in starting_nodes
            ), f"One or more Nodes in starting_nodes is not assigned an input DirectedEdge"

        self.reset_network()
        return starting_nodes

    def run(
        self,
        starting_nodes: List[Node],
        starting_inputs: List[List[Data]] = None,
        simulate=True,
    ):
        """
        Execute the workflow beginning at *starting_nodes*.

        Handles three situations explicitly:

        A. Normal progress – at least one node is ACTIVE, so time advances.
        B. Pre-flight consistency – if ready_to_start() is True,
           ensure_ready() must leave the process in status READY.
        C. Dead-lock – no node can be initialised or progressed while
           unfinished nodes still exist → raise RuntimeError.
        """
        # ------------------------------------------------------------------
        # 0.  Seed INPUT nodes with starting data and clear old state
        # ------------------------------------------------------------------
        starting_nodes = self.network_prep(starting_nodes, starting_inputs)
        remaining_nodes: list[Node] = starting_nodes.copy()
        active_nodes:    list[Node] = []
        self.execution_order.clear()

        elapsed_time: float = 0.0
        execution_idx: int  = 0
        results: Dict[Node, Result] = {}

       
        # ------------------------------------------------------------------
        # 1.  Pre-flight check on *static* INPUT nodes
        # ------------------------------------------------------------------
        for node in starting_nodes:
            if getattr(node, "extend_dynamic", False):
                logger.debug(f"Skipping pre-flight for dynamic node {repr(node)}")
                continue

            # quick boolean vs. heavier structural check
            assert node.ready_to_start() == node.check_if_ready(), (
                f"ready_to_start()/check_if_ready() diverged on {repr(node)}"
            )

            # heavy validation (raises InitError with nice message on mismatch)
            node.ensure_ready()
            logger.debug(f"Pre-flight: {repr(node)} READY")
        # ------------------------------------------------------------------
        # 2.  Main simulation loop
        # ------------------------------------------------------------------
        while remaining_nodes or active_nodes:
            self.status = Network.ACTIVE

            # --------------------------------------------------------------
            # 2.1  try to initialise any nodes that are now ready
            # --------------------------------------------------------------
            idx = 0
            while idx < len(remaining_nodes):
                node = remaining_nodes[idx]
                try:
                    self.__initialize_node(node)
                except (AllocationError, InitError) as e:
                    logger.debug(f"Init-skip {repr(node)}: {e}")
                    idx += 1
                    continue

                active_nodes.append(node)
                remaining_nodes.remove(node)
                self.execution_order.append(node)
                node.process.update()

                # bookkeeping for results dataframe
                if isinstance(node.allocation, ClassicalAllocation):
                    mem = node.allocation.allocated_memory if simulate else node.process.memory
                    results[node] = Result(
                        network_idx=execution_idx,
                        start_time=elapsed_time,
                        end_time=None,
                        device_name=node.allocation.device_name,
                        memory_usage=mem,
                        qubits_used="N/A",
                        circuit_depth="N/A",
                        success_probability="N/A",
                    )
                elif isinstance(node.allocation, QuantumAllocation):
                    results[node] = Result(
                        network_idx=execution_idx,
                        start_time=elapsed_time,
                        end_time=None,
                        device_name=node.allocation.device_name,
                        memory_usage="N/A",
                        qubits_used=node.allocation.allocated_qubit_idxs,
                        circuit_depth=node.allocation.transpiled_circuit.depth(),
                        success_probability=node.process._compute_sucess_probability(
                            node.allocation
                        ),
                    )
                execution_idx += 1  # only advance when a node really started

            # --------------------------------------------------------------
            # 2.2  dead-lock / termination check BEFORE advancing time
            # --------------------------------------------------------------
            if not active_nodes:
                if remaining_nodes:                       # → Case C
                    raise RuntimeError("Network deadlock detected")
                break                                      # all work finished

            # --------------------------------------------------------------
            # 2.3  advance time by minimum remaining execution
            # --------------------------------------------------------------
            dt = self.__compute_min_timestep(active_nodes)
            assert dt != INFINITY
            elapsed_time += dt

            # --------------------------------------------------------------
            # 2.4  update ACTIVE nodes and harvest newly ready successors
            # --------------------------------------------------------------
            idx = 0
            while idx < len(active_nodes):
                node = active_nodes[idx]
                self.__update_node(node, dt)

                if node.process.status == Process.COMPLETED:
                    if node.process.dynamic:
                        self.__extend_dynamic_node(node)

                    self.__complete_node(node)
                    active_nodes.remove(node)
                    remaining_nodes.extend(self.get_prepared_output_nodes(node))
                    self.describe
                    results[node].end_time = elapsed_time
                    continue  # do NOT increment idx – list contracted
                idx += 1

        # ------------------------------------------------------------------
        # 3.  Finish
        # ------------------------------------------------------------------
        self.status = Network.INACTIVE
        self.results_df = self.generate_results_dataframe(results)
        return self.results_df
    

    def reset_network(self):
        for node in self.nodes:
            node.output_edges = []
            if node.network_type != Node.INPUT:
                node.input_edges = []
    def _initialize_node(self, node: Node):
        node.start()
        self.__allocate_resources(node)
        logger.debug(f"Starting node {repr(node)} with inputs: {[d for d in node.process.inputs]} ---> resources allocated: {node.allocation}")
        assert node.verify_allocation(
            node.allocation
        ), f"Node {node.id} cannot execute it's process with given allocation {node.allocation}"

        if isinstance(node.process, ClassicalProcess):
            time_till_completion = node.process._compute_classical_process_update_time(
                node.allocation
            )
        elif isinstance(node.process, QuantumProcess):
            time_till_completion = node.process._compute_quantum_process_update_time(
                node.allocation
            )
        else:
            # defaulting
            flops = node.process.flops
            time_till_completion = flops / node.allocation.clock_frequency


            # node.process._set_transpiled_circ()
            # node.process._
        node.process.time_till_completion = time_till_completion

    def __initialize_node(self, node: Node):
        node.start()
        self.__allocate_resources(node)
        logger.debug(
            f"Starting node {repr(node)} with inputs: {[d for d in node.process.inputs]} "
            # f"\ndict: {node.process.input_data} "
            f"---> resources allocated: {node.allocation}"
        )
        assert node.verify_allocation(
            node.allocation
        ), f"Node {node.id} cannot execute it's process with given allocation {node.allocation}"

        if isinstance(node.process, ClassicalProcess):
            time_till_completion = node.process._compute_classical_process_update_time(
                node.allocation
            )
        elif isinstance(node.process, QuantumProcess):
            time_till_completion = node.process._compute_quantum_process_update_time(
                node.allocation
            )
        else:
            time_till_completion
            # node.process._set_transpiled_circ()
            # node.process._
        node.process.time_till_completion = time_till_completion

    def __allocate_resources(self, node: Node):
        """Requests allocation of resources for a process state from its broker.

        This method shall initiate a request for resource allocation for
        a process state from its associated broker. This request does not
        guarantee resource allocation.

        Returns:
            `Allocation`: An allocation result

        """
        if not node.process.required_resources:
            node.process.status = Process.ACTIVE
            node.allocation = None                  # ← was Allocation() ✅
            return
          
        else:
            try:
                allocation = self.broker.request_allocation(
                    node.process.required_resources
                )
            except AssertionError as e:
                if str(e) == "No available devices":
                    raise AllocationError(
                        f"Node {node.id} resource allocation FAILED"
                    ) from None
                raise

        node.process.status = Process.ACTIVE
        node.allocation = allocation

    def __update_node(self, node: Node, timestep: float):
        node.execute_process(timestep)

    def __extend_dynamic_node(self, dynamic_node: Node):
        if dynamic_node.network_type == "OUTPUT":
            self.output_nodes.remove(dynamic_node)
            dynamic_node.network_type = "NETWORK"

        sub_network = dynamic_node.process.extend_network()
        # output_nodes = copy.copy(dynamic_node.output_nodes)
        output_nodes = copy.copy(
            [node for node in dynamic_node.output_nodes if node.extend_dynamic]
        )

        if isinstance(sub_network, Node):
            dynamic_node.output_nodes = [sub_network]
            self.nodes.append(sub_network)
            if len(output_nodes) == 0:
                sub_network.network_type = "OUTPUT"
                self.output_nodes.append(sub_network)
            else:
                sub_network.network_type = "NETWORK"
            sub_network.output_nodes = output_nodes

        elif isinstance(sub_network, list):
            assert all(
                isinstance(subnode, Node) for subnode in sub_network
            ), f"Process.extend_network must either be a Node, a list of Nodes, or a Network"

            dynamic_node.output_nodes = sub_network
            for subnode in sub_network:
                self.nodes.append(subnode)
                if len(output_nodes) == 0:
                    subnode.network_type = "OUTPUT"
                    self.output_nodes.append(subnode)
                else:
                    subnode.network_type = "NETWORK"
                subnode.output_nodes = output_nodes

        elif isinstance(sub_network, Network):
            # dynamic_node.output_nodes = [node for node in sub_network.input_nodes]
            for node in sub_network.input_nodes:
                dynamic_node.output_nodes.append(node)

            if sub_network.output_nodes == []:
                assert all(
                    node.network_type == "INPUT" for node in sub_network.nodes
                ), f"Network {sub_network.name} cannot have NETWORK nodes without OUTPUT nodes"

                for node in sub_network.input_nodes:
                    self.nodes.append(node)
                    if len(output_nodes) == 0:
                        node.network_type = "OUTPUT"
                        self.output_nodes.append(node)
                    else:
                        node.network_type = "NETWORK"
                    node.output_nodes = output_nodes

            else:
                for node in sub_network.nodes:
                    self.nodes.append(node)
                    if node.network_type == "OUTPUT":
                        if len(output_nodes) == 0:
                            node.network_type = "OUTPUT"
                            self.output_nodes.append(node)
                        else:
                            node.network_type = "NETWORK"
                        node.output_nodes = output_nodes
                    else:
                        node.network_type = "NETWORK"

        else:
            raise ValueError(
                "Process.extend_network must either be a Node, a list of Nodes, or a Network"
            )

    def __complete_node(self, node: Node):
        self.__deallocate_resources(node)
        node.complete()

    def __deallocate_resources(self, node: Node):
        """Requests deallocation of resources from a process state's broker.

        This method shall initiate a request for resource freeing and
        release for a process state from its broker. This request does
        not guarantee release of resources.

        """

        """Ask the broker for resources—unless the process requires none."""
        if not node.process.required_resources:
            node.allocation = None
            return

        try:
            self.broker.request_deallocation(node.allocation)
        except Exception:  # TODO: replace with concrete broker exception
            raise AllocationError(f"Node {node.id} resource deallocation FAILED")
        node.allocation = None

    def get_prepared_output_nodes(self, node: Node):
        """Returns a list of all nodes in node.output_nodes whose
        input DirectedEdges are all completed and thus ready to execute
        """
        prepared_nodes = []
        for output_node in node.output_nodes:
            if output_node.ready_to_start() and output_node.network_type != "INPUT":
                prepared_nodes.append(output_node)

        return prepared_nodes

    def __compute_min_timestep(self, active_nodes: List[Node]) -> float:
        """Compares the remaining process execution time of all active nodes and selects
        the smallest one.

        Returns:
            float: The minimum timestep
        """
        assert all(
            node.process.status == Process.ACTIVE for node in active_nodes
        ), f"One or more Nodes in {active_nodes} have not been allocated resources"

        min_timestep = INFINITY
        for node in active_nodes:
            if node.process.time_till_completion < min_timestep:
                min_timestep = node.process.time_till_completion

        return min_timestep

    def generate_gantt_plot(self,show=False):
        assert isinstance(
            self.results_df, pd.DataFrame
        ), f"Generate simulation results by calling network.run()!"

        publish_gantt(self.results_df,show=show)

    def generate_results_dataframe(self, results: Dict[Node, Result]) -> pd.DataFrame:
        results_dict = {
            "Network Idx": [],
            "Node Idx": [],
            "Process": [],
            "Start Time [s]": [],
            "End Time [s]": [],
            "Device Name": [],
            "Memory [B]": [],
            "Qubits Used": [],
            "Circuit Depth": [],
            "Success Probability": [],
        }

        for key, val in results.items():
            results_dict["Network Idx"].append(val.network_idx)
            results_dict["Node Idx"].append(str(key).split(":")[0])
            results_dict["Process"].append(str(key).split(" ")[1])
            results_dict["Start Time [s]"].append(val.start_time)
            results_dict["End Time [s]"].append(val.end_time)
            results_dict["Device Name"].append(val.device_name)
            results_dict["Memory [B]"].append(val.memory_usage)
            results_dict["Qubits Used"].append(val.qubits_used)
            results_dict["Circuit Depth"].append(val.circuit_depth)
            results_dict["Success Probability"].append(val.success_probability)

        dataframe = pd.DataFrame(results_dict)
        dataframe = dataframe.sort_values(by="Network Idx")
        dataframe.drop("Network Idx", axis=1, inplace=True)

        return dataframe

    def to_networkx(self):
        G = nx.DiGraph()
        edge_labels = {}
        mapping = {}

        G.add_node("Output", network_type=Node.OUTPUT)

        if self.execution_order == []:
            show_edge_labels = False
            for node in self.nodes:
                mapping[node] = str(node)

            for node in self.nodes:
                G.add_node(mapping[node], network_type=node.network_type)
                if node.network_type == Node.OUTPUT:
                    G.add_edge(mapping[node], "Output")
                else:
                    for output_node in node.output_nodes:
                        G.add_edge(
                            mapping[node],
                            mapping[output_node],
                        )
        else:
            idx = 0
            for node in self.execution_order:
                mapping[node] = str(idx) + ": " + node.process.__class__.__name__
                idx = idx + 1

            for node in self.nodes:
                G.add_node(mapping[node], network_type=node.network_type)
                for edge in node.output_edges:
                    if edge.dest_nodes == []:
                        G.add_edge(mapping[edge.source_node], "Output")
                    for dest_node in edge.dest_nodes:
                        G.add_edge(
                            mapping[edge.source_node],
                            mapping[dest_node],
                            dtype=edge.data[0].properties["Data Type"],
                        )
                        edge_labels[(mapping[edge.source_node], mapping[dest_node])] = (
                            edge.data[0].properties["Usage"]
                        )
        
        return G, mapping

    def visualize(self, show_edge_labels=False, show=False, ax=None,figsize=(12, 10),):
        G = nx.DiGraph()
        edge_labels = {}
        mapping = {}

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        if self.execution_order == []:
            show_edge_labels = False
            for node in self.nodes:
                mapping[node] = str(node)

            for node in self.nodes:
                G.add_node(mapping[node], network_type=node.network_type)
                if node.network_type == Node.OUTPUT:
                    G.add_edge(mapping[node], "Output")
                else:
                    for output_node in node.output_nodes:
                        G.add_edge(
                            mapping[node],
                            mapping[output_node],
                        )
        else:
            idx = 0
            for node in self.execution_order:
                mapping[node] = str(idx) + ": " + node.process.__class__.__name__
                idx = idx + 1

            for node in self.nodes:
                G.add_node(mapping[node], network_type=node.network_type)
                for edge in node.output_edges:
                    if edge.dest_nodes == []:
                        G.add_edge(mapping[edge.source_node], "Output")
                    for dest_node in edge.dest_nodes:
                        G.add_edge(
                            mapping[edge.source_node],
                            mapping[dest_node],
                            dtype=edge.data[0].properties["Data Type"],
                        )
                        edge_labels[(mapping[edge.source_node], mapping[dest_node])] = (
                            edge.data[0].properties["Usage"]
                        )

        pos = nx.drawing.layout.planar_layout(G)
        color_map = []
        for node, attributes in G.nodes(data=True):
            if node != "Output":
                color_map.append(
                    "red" if attributes["network_type"] == Node.INPUT else "#1f78b4"
                )
            else:
                color_map.append("blue")

        nx.draw(
            G, pos, with_labels=True, node_size=200, font_size=8, node_color=color_map, ax=ax,
        )
        if show_edge_labels:
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)
        
        if show:
            plt.show()
    

    def combine(self, other, network_name=None):
        """Combines two networks under a single larger network such that
        input and output nodes are concatenated
        """
        assert isinstance(other, Network), f"{other} must be a Network object"

        name = network_name or self.name
        nodes = self.nodes + other.nodes
        input_nodes = self.input_nodes + other.input_nodes
        output_nodes = self.output_nodes + other.output_nodes
        broker = self.broker + other.broker

        return Network(name, nodes, input_nodes, output_nodes, broker)

    def __add__(self, other, network_name=None):
        """Combines the networks such that the outputs of self are connected
        to the inputs of other. Outputs without a connected input will be
        treated as outputs of the combined network
        """
        assert isinstance(other, Network), f"{other} must be a valid Network object"

        self.unassigned_output_nodes = []
        for output_node in self.output_nodes:
            assert (
                output_node.output_nodes == []
            ), f"Output nodes of Network {self.name} should not have outgoing nodes"
            self.__assign_network_output_nodes(output_node, other.input_nodes)

        name = network_name or self.name
        nodes = self.nodes + other.nodes
        input_nodes = self.input_nodes
        output_nodes = other.output_nodes + self.unassigned_output_nodes
        broker = self.broker + other.broker

        return Network(name, nodes, input_nodes, output_nodes, broker)

    def __assign_network_output_nodes(self, node: Node, input_nodes: list[Node]):
        """Attempts to assign output node to the appropriate destination nodes in input_nodes"""

        output_nodes = set()
        for node_property in node.process.output_properties:
            for input_node in input_nodes:
                if self.__output_property_in_expected_input_properties(
                    node_property, input_node.expected_input_properties
                ):
                    input_node.network_type = Node.NETWORK
                    output_nodes.add(input_node)
                    break

        if len(output_nodes) == 0:
            self.unassigned_output_nodes.append(node)
        else:
            node.network_type = Node.NETWORK
            node.output_nodes = list(output_nodes)

    def __output_property_in_expected_input_properties(
        self, output_property: dict, expected_input_properties: List[dict]
    ):
        for expected_input_property in expected_input_properties:
            if all(
                all_dict1_vals_in_dict2_vals(output_property.get(key, None), val)
                for key, val in expected_input_property.items()
            ):
                return True
        return False

    def separate(self, separation_edge):
        del separation_edge
        pass
    @property
    def describe(self):
        node_ids = [node for node in self.nodes]
       # node_edges = [node.id for node in self.nodes]
        input_ids = [node for node in self.input_nodes]
        node_edges = [f"{node}, {node.network_type}" for node in self.nodes]
        output_ids = [node for node in self.output_nodes]
        if self.status == Network.INACTIVE:
            description = (
                f"Network: {self.name}, status {self.status}\n"
                # f"Nodes: {node_ids}\n"
                f"Input Nodes: {input_ids}\n"
                # f"Node .network_types: {node_edges}\n"
                f"Output Nodes: {output_ids}"
            
            
            )
        else:
            active = [node for node in self.nodes if node.process.status == Process.ACTIVE]
            inactive = [node for node in self.nodes if node.process.status == Process.UNSTARTED]
            completed = [node for node in self.nodes if node.process.status == Process.COMPLETED]
            description = (
                f"Network: {self.name}, status {self.status}\n"
                # f"Nodes: {node_ids}\n"
                f" - Active Nodes: {active}\n"
                # f"Node .network_types: {node_edges}\n"
                f" - Unstarted Nodes: {inactive}\n"
                f" - Completed Nodes: {completed}"
            
            
            )

        logger.debug(description)
        # print(description)
    def __str__(self):
        return self.name
    
    def run_backup(
        self,
        starting_nodes: List[Node],
        starting_inputs: List[List[Data]] = None,
        simulate=True,
    ):
        """
        Execute the workflow beginning at *starting_nodes*.

        Handles three situations explicitly:

        A. Normal progress – at least one node is ACTIVE, so time advances.
        B. Pre-flight consistency – if ready_to_start() is True,
           ensure_ready() must leave the process in status READY.
        C. Dead-lock – no node can be initialised or progressed while
           unfinished nodes still exist → raise RuntimeError.
        """
        # ------------------------------------------------------------------
        # 0.  Seed INPUT nodes with starting data and clear old state
        # ------------------------------------------------------------------
        starting_nodes = self.network_prep(starting_nodes, starting_inputs)
        remaining_nodes: list[Node] = starting_nodes.copy()
        active_nodes:    list[Node] = []
        self.execution_order.clear()

        elapsed_time: float = 0.0
        execution_idx: int  = 0
        results: Dict[Node, Result] = {}

        # --- PRE-FLIGHT VALIDATION (only on *static* INPUT nodes) ---
        # for node in starting_nodes:
        #     # skip any node that is marked dynamic → will be fed later by extend_network
        #     if getattr(node, "extend_dynamic", False):
        #         logger.debug(f"Skipping pre-flight for dynamic node {repr(node)}")
        #         continue

        #     # quick vs. full consistency check
        #     original_ok = node.ready_to_start()
        #     new_ok      = node.check_if_ready()
        #     assert original_ok == new_ok, (
        #         f"Boolean ready_to_start() and check_if_ready() diverged on {repr(node)}"
        #     )

        #     # now do the *verbose* validation (will raise with pretty InitError if bad)
        #     try:
        #         node.ensure_ready()
        #         logger.debug(f"Pre-flight: Node {repr(node)} READY for execution.")
        #     except InitError as e:
        #         logger.error(f"Pre-flight full check failed for {repr(node)}:\n{e}")
        #         # abort early if any static input is invalid
        #         raise
        # ------------------------------------------------------------------
        # 1.  Pre-flight check on *static* INPUT nodes
        # ------------------------------------------------------------------
        for node in starting_nodes:
            if getattr(node, "extend_dynamic", False):
                logger.debug(f"Skipping pre-flight for dynamic node {repr(node)}")
                continue

            # quick boolean vs. heavier structural check
            assert node.ready_to_start() == node.check_if_ready(), (
                f"ready_to_start()/check_if_ready() diverged on {repr(node)}"
            )

            # heavy validation (raises InitError with nice message on mismatch)
            node.ensure_ready()
            logger.debug(f"Pre-flight: {repr(node)} READY")
        # Main simulation loop
        # We know we’ll only spin when:
        #  - Case A: active_nodes non-empty → we can advance time and complete work
        #  - Case B: active_nodes empty but some remaining are ready → we can init new work
        #  - Case C: neither → deadlock, so we bail out immediately
        while len(remaining_nodes) != 0 or len(active_nodes) != 0:
            self.status = Network.ACTIVE
            did_initialize = False
            did_complete   = False
            node_idx = 0
            while node_idx < len(remaining_nodes):
                node = remaining_nodes[node_idx]
                try:
                    self.__initialize_node(node)
                    logger.debug(f"Attempting to init {node} with input edges {node.input_edges}")
                    active_nodes.append(node)
                    remaining_nodes.remove(node)
                    did_initialize = True
                    self.execution_order.append(node)
                    node.process.update()

                    if isinstance(node.allocation, ClassicalAllocation):
                        if simulate:
                            memory_usage = node.allocation.allocated_memory
                        else:
                            node.process.time_till_completion = node.process.time
                            memory_usage = node.process.memory

                        results[node] = Result(
                            network_idx=execution_idx,
                            start_time=elapsed_time,
                            end_time=None,
                            device_name=node.allocation.device_name,
                            memory_usage=memory_usage,
                            qubits_used="N/A",
                            circuit_depth="N/A",
                            success_probability="N/A",
                        )

                    elif isinstance(node.allocation, QuantumAllocation):
                        assert isinstance(
                            node.process, QuantumProcess
                        ), f"Expected QuantumProcess for QuantumAllocation {node.allocation}"
                        results[node] = Result(
                            network_idx=execution_idx,
                            start_time=elapsed_time,
                            end_time=None,
                            device_name=node.allocation.device_name,
                            memory_usage="N/A",
                            qubits_used=node.allocation.allocated_qubit_idxs,
                            circuit_depth=node.allocation.transpiled_circuit.depth(),
                            success_probability=node.process._compute_sucess_probability(
                                node.allocation
                            ),
                        )

                except (AllocationError, InitError) as e:
                    logger.debug(f"Could not init {node}: {e}")
                    self.describe
                    node_idx = node_idx + 1
                    continue

                execution_idx = execution_idx + 1

            node_idx = 0
            update_timestep = self.__compute_min_timestep(active_nodes)
            assert update_timestep != INFINITY
            elapsed_time = elapsed_time + update_timestep
            prev_active = active_nodes.copy()
            prev_remaining  = remaining_nodes.copy()
            while node_idx < len(active_nodes):
                active_node = active_nodes[node_idx]
                self.__update_node(active_node, update_timestep)
                if active_node.process.status == Process.COMPLETED:
                    if active_node.process.dynamic:
                        self.__extend_dynamic_node(active_node)
                    self.__complete_node(active_node)
                    active_nodes.remove(active_node)
                    remaining_nodes.extend(self.get_prepared_output_nodes(active_node))
                    logger.debug(f'Removing {repr(active_node)} from active nodes and preparing new remaining nodes:\n  - active nodes: {prev_active} -> {active_nodes}\n  - remaining nodes: {prev_remaining} -> {remaining_nodes}')
                    results[active_node].end_time = elapsed_time
                    continue

                node_idx = node_idx + 1
        self.status = Network.INACTIVE
        self.results_df: pd.DataFrame = self.generate_results_dataframe(results)
        return self.results_df