from typing import List
from workflow.simulation.refactor.data_types import MatrixType
from workflow.simulation.refactor.data import Data
from workflow.simulation.refactor.process import Process, ClassicalProcess
from workflow.simulation.refactor.graph import Node, DirectedEdge, Network
from workflow.simulation.refactor.resources.classical_resources import (
    ClassicalResource,
    ClassicalDevice,
)
from workflow.simulation.refactor.broker import Broker
from numpy import inf as INFINITY
import pytest


class AddOne(ClassicalProcess):

    def __init__(
        self,
        inputs: list[Data] = None,
        expected_input_properties=None,
        required_resources=None,
        output_properties=None,
        remaining_itrs=0,
    ):

        super().__init__(
            inputs,
            expected_input_properties,
            required_resources,
            output_properties,
            dynamic=False if remaining_itrs == 0 else True,
        )
        self.remaining_itrs: int = remaining_itrs

    def set_expected_input_properties(self):
        """Sets the expected input properties of the Process to execute/update successfully.
        This method should update self.expected_input_properties to be a list of dictionaries, with each
        dictionary specifying the expected properties of an input Data instance. A Process that requires
        N inputs will require N such dictionaries specifying its properties. Will be unique to each Process.
        """

        self.expected_input_properties = [{"Data Type": int, "Usage": "Integer"}]

    def set_required_resources(self):
        """Sets the appropriate resources required to run the process. This method should update
        self.required_resources to be a list of Resources required for the Process to successfully run.
        Will be unique for each process.

        """

        self.required_resources = ClassicalResource(memory=1, num_cores=1)

    def set_output_properties(self):
        """Sets the expected properties of the generated outputs of the Process.
        This method should update self.output_properties to be a list of dictionaries,
        with each dictionary specifying the expected properties of each output Data
        """

        self.output_properties = [{"Data Type": int, "Usage": "Integer"}]

    def compute_flop_count(self):
        """Computes the number of required flops for this process to complete given
        valid inputs from self.inputs. Will be unique to each Process.
        """

        return 1

    def validate_data(self) -> bool:
        """Ensures input data can correctly run for the process. Will
        be unique for each process

        """

        conditions = (
            len(self.inputs) == 1,
            isinstance(self.inputs[0].data, int),
        )

        return all(condition for condition in conditions)

    def update(self):
        """Updates the given process state.

        This method shall apply this model's methodology
        to the given process_state to adjust the process
        on the current timestep. The workflow state is
        also provided to expose time parameters, such as
        the current simulation time and timestep. This process
        is abstract, and must be implemented per the
        individual models that subclass this class.

        Args:
            workflow_state (`WorkflowState`): The workflow state.
            process_state (`ProcessState`): The process state.

        """
        self.integer = self.inputs[0].data
        self.integer += 1

    def generate_output(self) -> list[Data]:
        """After completing updates, the process will wrap its results
        and return a (unordered) set of Data states to be processed by Node.
        The expected output will be unique to the process.
        """

        return (Data(data=self.integer, properties={"Usage": "Integer"}),)

    def extend_network(self) -> Node:
        return Node(process_model=AddOne, remaining_itrs=self.remaining_itrs - 1)


def test_node_init():
    addone = Node(process_model=AddOne)


def test_node_update():
    addone = Node(process_model=AddOne)
    input_edge = DirectedEdge(
        data=(Data(data=5, properties={"Usage": "Integer"}),),
        edge_type="INPUT",
        source_node=None,
        dest_nodes=[addone],
    )
    addone.append_input_edge(input_edge)
    addone.start()
    addone.process.update()
    assert addone.process.integer == 6


def test_extend_dynamic_node():
    addone = Node(process_model=AddOne, remaining_itrs=1)
    input_edge = DirectedEdge(
        data=(Data(data=5, properties={"Usage": "Integer"}),),
        edge_type="INPUT",
        source_node=None,
        dest_nodes=[addone],
    )
    addone.append_input_edge(input_edge)
    addone.start()
    addone.process.update()
    assert addone.process.integer == 6

    subnode = addone.process.extend_network()
    input_edge = DirectedEdge(
        data=(Data(data=6, properties={"Usage": "Integer"}),),
        edge_type="CONNECTED",
        source_node=addone,
        dest_nodes=[subnode],
    )
    subnode.append_input_edge(input_edge)
    subnode.start()
    assert subnode.process.dynamic == False
    assert subnode.process.remaining_itrs == 0


def generate_broker():
    supercomputer = ClassicalDevice(
        device_name="Supercomputer",
        processor_type="CPU",
        RAM=100 * 10**9,
        properties={"Cores": 20, "Clock Speed": 3 * 10**9},
    )

    broker = Broker(classical_devices=[supercomputer])
    return broker


def test_dynamic_network():
    addone = Node(process_model=AddOne, network_type="INPUT", remaining_itrs=10)
    network = Network(
        name="dynamic test",
        nodes=[addone],
        input_nodes=[addone],
        output_nodes=[],
        broker=generate_broker(),
    )

    network.run(
        starting_nodes=[addone],
        starting_inputs=[(Data(data=5, properties={"Usage": "Integer"}),)],
    )
    assert len(network.input_nodes) == 1
    assert len(network.nodes) == 11
    assert len(network.output_nodes) == 1

    output_node = network.output_nodes[0]
    output_edge = output_node.output_edges[0]
    result = output_edge.data[0].data
    assert result == 16
