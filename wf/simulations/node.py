from typing import (
    Callable,
    cast,
    Dict,
    FrozenSet,
    Hashable,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    overload,
    Sequence,
    Set,
    Tuple,
    TYPE_CHECKING,
    TypeVar,
    Union,
)
from functools import cached_property
import attrs
import numpy as np
from typing import List, Tuple,Union,TypeVar, Sequence
from collections import OrderedDict
import matplotlib.pyplot as plt
import networkx as nx
import copy
from numpy import inf as INFINITY
from .process import Process
from .resources import Allocation
from .broker import Broker
from .data import Data, Flow,Signature
from .utilities import (
    get_next_id,
    all_dict1_vals_in_dict2_vals,
    any_dict1_vals_in_dict2_vals,
)
import datetime
from ...util.log import logging

logger = logging.getLogger(__name__)


from .resources.classical_resources import ClassicalAllocation, ClassicalResource
from .resources.quantum_resources import QuantumAllocation, QuantumResource

class ExpectedNodeError(Exception):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class EdgeAssignmentError(Exception):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


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

        # logger.debug(f"Connectivity verified for edge {self}")
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
        return f'{self.source_node} -> {self.dest_nodes} (type: {self.edge_type})'





class Node:
    """
    Represents, stores and manages information on the most basic process/task 
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
        **kwargs,
    ):
        """
        Args:
            id (str, optional): id of the Node, constructed using get_next_id() if not specified. Defaults to None.
            process_model (Process, optional): The process model this Node simulates. Defaults to None.
            network_type (str, optional): Type of Node inside a larger Network, can be INPUT, NETWORK or OUTPUT. Defaults to None.
            input_edges (List[DirectedEdge], optional): List of incoming DirectedEdges to the Node. Defaults to [].
            output_edges (List[DirectedEdge], optional): List of outgoing DirectedEdges from the Node. Defaults to [].
            output_nodes (List[Node], optional): List of Nodes to send the process results to. Defaults to [].
        """

        self.id: str = id or get_next_id()
        self.model: Process = process_model
        # logger.debug(f"{self.id} for model: {self.model}")
        if hasattr(self.model, 'inputs'):
            # logger.debug(f"Leaving model un-instantiated")
            self.process = self.model
        else:
            self.process: Process = process_model(**kwargs) if self.model else None

        self.network_type: str = network_type
        self.input_edges: List[DirectedEdge] = input_edges or []
        self.output_edges: List[DirectedEdge] = output_edges or []
        self.output_nodes: List[Node] = output_nodes or []
        self.allocation: Allocation = None
        self.kwargs = kwargs

        # logger.debug(f"Signature for process {process_model}: {self.process.signature}")
        if isinstance(self.process, Process):

            
            self.expected_input_properties = self.process.expected_input_properties
            # logger.debug(f"Generated input props: {self.expected_input_properties}")
        # logger.debug(f"Process added as to Node")
    def is_input(self):
        return self.network_type == "INPUT"

    def is_network(self):
        return self.network_type == "NETWORK"

    def is_output(self):
        return self.network_type == "OUTPUT"

    def is_none(self):
        return self.network_type is None
    
    def as_composite(self):
        from .builder import ProcessBuilder
        builder, init_ports = ProcessBuilder.from_signature(self.process.signature)
        logger.debug(f"As composite operation. Initial ports: {init_ports}")
        return builder.finalize(**builder.add_d(self.process,**init_ports))
    
    def as_composite_block(self, builder, init_ports):
        
        new_dict = builder.add_node(self,**init_ports)
        
        return builder.finalize(**new_dict)
    
    def append_input_edge(self, edge: DirectedEdge):
        assert (
            not edge in self.input_edges
        ), f"Edge {edge} already exists as an input edge"
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
    def __validate_connection(self, dest):
        assert isinstance(dest, Node), f"{dest} must be a Node object"
        if self.is_output() and not dest.is_output():
            raise ValueError(
                f"Attempting to connect an OUTPUT Node to a {dest.network_type} Node"
            )
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
        logger.debug(f"Starting node {self} at t={self.start_time}")

    def ready_to_start(self):
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
        required_resource = self.process.required_resources

        if isinstance(required_resource, ClassicalResource):
            assert isinstance(
                allocation, ClassicalAllocation
            ), f"Expected ClassicalAllocation for ClassicalResource requirement but got {type(allocation)} instead"
            conditions = (
                required_resource.memory <= allocation.allocated_memory,
                required_resource.num_cores <= allocation.allocated_cores,
            )
        elif isinstance(required_resource, QuantumResource):
            assert isinstance(
                allocation, QuantumAllocation
            ), f"Expected QuantumAllocation for QuantumResource requirement but got {type(allocation)} instead"
            conditions = (
                required_resource.circuit.qubit_count
                == len(allocation.allocated_qubit_idxs),
            )
        logger.debug(f"Allocation verified for {self}")
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
        for result in self.process.generate_output():
            self.output_edges.append(
                DirectedEdge(
                    data = (result,),
                    source_node=self,
                    dest_nodes=None,
               
                    edge_type=DirectedEdge.OUTPUT,
                )
            )
    def map_edges_to_output_nodes(self):
        """Maps the empty and INCOMPLETE outgoing DirectedEdges
        to the next appropriate Nodes in the network as specified by
        self.output_nodes.

        Edges are mapped to output nodes when the data properties contained
        in the DirectedEdge match expected_input_properties of a Node.

        Unassigned Edges are left as 'hanging' edges of the Network.
        """

        # assert (
        #     self.output_nodes != []
        # ), f"Completed Node {self.id} has no output Nodes to map edges to"
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
        assert all(
            isinstance(output_node, Node) for output_node in output_nodes
        ), f"{output_nodes} must be a list of Nodes"
        assert (
            edge.source_node == self
        ), f"Node {self} does not have permission to assign edge of Node {edge.source_node}"

        dest_nodes = []
        for node in output_nodes:
            if all(
                self.__edge_properties_in_output_node(edge_data.properties, node)
                for edge_data in edge.data
            ):
                dest_nodes.append(node)

        if dest_nodes == []:
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

    # def valid_process_result_in_dest_node(self, result: Data, dest_node):
    #     """Verifies whether the generated process result contains the
    #     correct data properties as expected by one of expected_input_properties in dest_node.
    #     """
    #     assert isinstance(dest_node, Node), f"{dest_node} must be a Node object"
    #     return any(
    #         self.result_property_in_expected_input_property(result, input_property)
    #         for input_property in dest_node.expected_input_properties
    #     )

    # def result_property_in_expected_input_property(
    #     self, result: Data, input_property: dict
    # ) -> bool:
    #     """Verifies whether all items in input_property are also present
    #     in result data properties.
    #     """
    #     if all(
    #         all_dict1_vals_in_dict2_vals(result.properties.get(key, None), val)
    #         for key, val in input_property.items()
    #     ):
    #         return True
    #     return False
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
        # print(description)
    def __str__(self):
        return self.id + ": " + self.process.__class__.__name__
    def __repr__(self):
        return self.process.__class__.__name__ + "." + self.id
    
    def connect_to(self, next_node):
        """Use inferred outputs to connect this node to the next node."""
        output_edges = self.infer_output_edges()
        next_node_inputs = next_node.process.expected_input_properties

    def infer_output_edges(self):
        """Infer output edges based on the process outputs without execution."""
        inferred_outputs = self.process.infer_outputs()
        for output in inferred_outputs:
            self.output_edges.append(DirectedEdge(data=(output,), source_node=self))

   
    
    def gen_expected_output_edges(self):
        """Generates outgoing DirectedEdges containing the expected output properties of the process model.
        
        If the process has a signature, it uses the right-side registers of the signature to generate the edges.
        If no signature is set, it falls back to using the output properties.
        """
        if self.process.inputs and self.process.signature:
            print(f"Generating output edges using signature for {self}")
            logger.debug(f"Generating output edges using signature for {self}")
            for right_register in self.process.signature.rights():
                edge = DirectedEdge(
                    data=(right_register,),
                    source_node=self,
                    dest_nodes=None,
                    status=DirectedEdge.INCOMPLETE,
                    edge_type=DirectedEdge.OUTPUT,
                )
                if edge not in self.output_edges:
                    self.output_edges.append(edge)
        elif self.process.inputs:
            print(f"Generating output edges using inputs for {self}")
            logger.debug(f"Generating output edges using inputs for {self}")
            for input_data in self.process.inputs:
                edge = DirectedEdge(
                    data=(input_data,),
                    source_node=self,
                    dest_nodes=None,
                    status=DirectedEdge.INCOMPLETE,
                    edge_type=DirectedEdge.OUTPUT,
                )
                if edge not in self.output_edges:
                    self.output_edges.append(edge)

        else:
            logger.debug(f"Generating output edges using output properties for {self}")
            for property in self.process.output_properties:
                edge = DirectedEdge(
                    data=(Data(data=None, properties=property),),
                    source_node=self,
                    dest_nodes=None,
                    status=DirectedEdge.INCOMPLETE,
                    edge_type=DirectedEdge.OUTPUT,
                )
                logger.debug(f" - new edge: {edge}")
                if edge not in self.output_edges:
                    self.output_edges.append(edge)

    def get_node_id(self) -> str:
        """Return the unique ID of the Node."""
        return self.id
    def get_process_name(self) -> str:
        """Return the name of the process model associated with the Node."""
        return self.process.__class__.__name__ if self.process else "UnassignedProcess"