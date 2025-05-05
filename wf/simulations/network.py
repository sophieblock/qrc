
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
from .node import Node,DirectedEdge
from .utilities import (
    all_dict1_vals_in_dict2_vals,
    publish_gantt,
)

import datetime
from ...util.log import logging


logger = logging.getLogger(__name__)





class AllocationError(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Network:
    """Collection of connected Nodes and Edges to represent
    larger, more complex but fixed algorithms. Smaller Networks
    can be summed together to form larger and more complex Networks
    assuming the outputs of one Network are valid inputs of the other.
    """

    def __init__(
        self,
        name="default",
        nodes=None,
        input_nodes=None,
        output_nodes=None,
        broker: Broker = None,
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

    def run(
        self,
        starting_nodes: List[Node],
        starting_inputs: List[List[Data]] = None,
        simulate=True,
    ):
        """Runs the simulation of the Network beginning at starting_nodes.

        Args:
            starting_nodes (List[Node]): List of nodes to start the simulation
            starting_inputs (List[List[Data]], optional): Input Data to begin Simulation. Defaults to None.
        """
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

        remaining_nodes = copy.copy(starting_nodes)
        active_nodes: List[Node] = []
        elapsed_time = 0.0
        execution_idx = 0
        results: Dict[Node, Result] = {}

        while len(remaining_nodes) != 0 or len(active_nodes) != 0:
            node_idx = 0
            while node_idx < len(remaining_nodes):
                node = remaining_nodes[node_idx]
                try:
                    self.__initialize_node(node)
                    active_nodes.append(node)
                    remaining_nodes.remove(node)
                    self.execution_order.append(node)
                    logger.debug(f'Execution order: {self.execution_order}')
                    node.process.update()
                    if not simulate:
                        node.process.time_till_completion = node.process.time
                        memory_usage = node.process.memory
                    else:
                        memory_usage = node.allocation.allocated_memory

                    if isinstance(node.allocation, ClassicalAllocation):
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

                except (AllocationError, InitError):
                    node_idx = node_idx + 1
                    continue

                execution_idx = execution_idx + 1

            node_idx = 0
            update_timestep = self.__compute_min_timestep(active_nodes)
            elapsed_time = elapsed_time + update_timestep
            while node_idx < len(active_nodes):
                active_node = active_nodes[node_idx]
                self.__update_node(active_node, update_timestep)
                if active_node.process.status == Process.COMPLETED:
                    if active_node.process.dynamic:
                        self.__extend_dynamic_node(active_node)
                    self.__complete_node(active_node)
                    active_nodes.remove(active_node)
                    remaining_nodes.extend(self.get_prepared_output_nodes(active_node))
                    results[active_node].end_time = elapsed_time
                    continue

                node_idx = node_idx + 1

        self.results_df: pd.DataFrame = self.generate_results_dataframe(results)
        return self.results_df

    def reset_network(self):
        for node in self.nodes:
            node.output_edges = []
            if node.network_type != Node.INPUT:
                node.input_edges = []

    def __initialize_node(self, node: Node):
        node.start()
        self.__allocate_resources(node)
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
        if node.process.required_resources is None:
            allocation = Allocation()

        else:
            logger.debug(f"Node {node.id} is attempting allocation...")
            try:
                allocation = self.broker.request_allocation(
                    node.process.required_resources
                )
                
            except AssertionError as e:
                if str(e) == "No available devices":
                    raise AllocationError(f"Node {node.id} resource allocation FAILED")
                else:
                    raise e
        node.process.status = Process.ACTIVE
        node.allocation = allocation
        logger.debug(f"Node {node.id} resource allocation successful -> Resources Allocation: {allocation}")

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

        if node.process.required_resources is None:
            node.allocation = None
        else:
            try:
                self.broker.request_deallocation(node.allocation)
            except:  ## TODO specify expected exception error
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

    def generate_gantt_plot(self):
        assert isinstance(
            self.results_df, pd.DataFrame
        ), f"Generate simulation results by calling network.run()!"

        publish_gantt(self.results_df)

    def generate_results_dataframe(self, results: Dict[Node, Result]) -> pd.DataFrame:
        results_dict = {
            "Network Idx": [],
            "Node Idx": [],
            "Process": [],
            "Start Time [s]": [],
            "End Time [s]": [],
            "Device Name": [],
            "Memory Cost [B]": [],
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
            results_dict["Memory Cost [B]"].append(val.memory_usage)
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

    def visualize(self, show_edge_labels=False, show=True, ax=None):
        G = nx.DiGraph()
        edge_labels = {}
        mapping = {}

        if ax is None:
            fig, ax = plt.subplots()

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
        description = (
            f"Network Name: {self.name}\n"
            f"Nodes: {node_ids}\n"
            f"Input Nodes: {input_ids}\n"
            f"Node .network_types: {node_edges}\n"
            f"Output Nodes: {output_ids}\n"
           
           
        )
        logger.debug(description)
        # print(description)
