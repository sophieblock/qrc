from workflow.simulation.refactor.data_types import MatrixType
from workflow.simulation.refactor.data import Data
from workflow.simulation.refactor.process import ClassicalProcess, Process, InitError
from workflow.simulation.refactor.graph import Node, DirectedEdge
from workflow.simulation.refactor.resources.classical_resources import (
    ClassicalAllocation,
    ClassicalResource,
)
from workflow.simulation.refactor.utilities import compute_resources
import pytest
import re
from numpy import inf as INFINITY


class MatrixMult(ClassicalProcess):
    """A sample Process that generates a basic description of a person given their name,
    age and company. Returns a string description of the person

    """

    def __init__(
        self,
        inputs: list[Data] = None,
        expected_input_properties=None,
        required_resources=None,
        output_properties=None,
    ):
        super().__init__(
            inputs, expected_input_properties, required_resources, output_properties
        )

    def set_expected_input_properties(self):
        """Sets the expected input properties of the Process to execute/update successfully.
        This method should update self.expected_input_properties to be a list of dictionaries, with each
        dictionary specifying the expected properties of an input Data instance. A Process that requires
        N inputs will require N such dictionaries specifying its properties. Will be unique to each Process.
        """

        self.expected_input_properties = [
            {"Data Type": type(MatrixType()), "Usage": "Matrix"}
        ] * 2

    def set_required_resources(self):
        """Sets the appropriate resources required to run the process. This method should update
        self.required_resources to be a list of Resources required for the Process to successfully run.
        Will be unique for each process.

        """

        self.required_resources = ClassicalResource(memory=5 * 10**6, num_cores=1)

    def set_output_properties(self):
        """Sets the expected properties of the generated outputs of the Process.
        This method should update self.output_properties to be a list of dictionaries,
        with each dictionary specifying the expected properties of each output Data
        """

        self.output_properties = [{"Data Type": type(MatrixType()), "Usage": "Matrix"}]

    def compute_flop_count(self):
        """Computes the number of required flops for this process to complete given
        valid inputs from self.inputs. Will be unique to each Process.
        """

        pass

    def validate_data(self) -> bool:
        """Ensures input data can correctly run for the process. Will
        be unique for each process

        """
        accepted_element_dtypes = [float, int]
        conditions = (
            len(self.inputs) == 2,
            all(
                input.data.element_dtype in accepted_element_dtypes
                for input in self.inputs
            ),
            self.valid_dimensions(),
        )

        return all(condition for condition in conditions)

    def valid_dimensions(self) -> bool:
        input_matrices = [input.data for input in self.inputs]
        # return all(input_matrices[i].cols == input_matrices[i+1].rows for i in range(len(input_matrices)-1))
        return (
            input_matrices[0].rows == input_matrices[1].cols
            or input_matrices[1].rows == input_matrices[0].cols
        )
    @compute_resources
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
        self.compute_required_resources()
        element_dtype = (
            int
            if all(input.data.element_dtype == int for input in self.inputs)
            else float
        )
        if self.inputs[0].data.cols == self.inputs[1].data.rows:
            self.result = MatrixType(
                self.inputs[0].data.rows,
                self.inputs[1].data.cols,
                element_dtype=element_dtype,
            )
        elif self.inputs[1].data.cols == self.inputs[0].data.rows:
            self.result = MatrixType(
                self.inputs[1].data.rows,
                self.inputs[0].data.cols,
                element_dtype=element_dtype,
            )

    def compute_required_resources(self):
        pass

    def generate_output(self) -> list[Data]:
        """After completing updates, the process will wrap its results
        and return a (unordered) set of Data states to be processed by Node.
        The expected output will be unique to the process.
        """

        return (Data(data=self.result, properties={"Usage": "Matrix"}),)


def test_matmult_update():
    mat1 = Data(MatrixType(2, 4, element_dtype=float), {"Usage": "Matrix"})
    mat2 = Data(MatrixType(4, 5, element_dtype=float), {"Usage": "Matrix"})
    matmul = MatrixMult([mat1, mat2])
    matmul.update()
    assert matmul.result.cols == 5
    assert matmul.result.rows == 2


def test_empty_gen_expected_output_edges():
    empty_node = Node(id=None, process_model=None)
    with pytest.raises(
        AttributeError, match=f"object has no attribute 'expected_input_properties'"
    ):
        empty_node.expected_input_properties
    assert empty_node.output_edges == []


def test_init_expected_input_properties():
    node = Node(id=None, process_model=MatrixMult)
    assert (
        node.expected_input_properties
        == [{"Data Type": type(MatrixType()), "Usage": "Matrix"}] * 2
    )


def test_append_existing_edge():
    node = Node(id=None, process_model=MatrixMult)
    input_edge = DirectedEdge(
        data=None, edge_type=DirectedEdge.INPUT, source_node=None, dest_nodes=[node]
    )
    node.input_edges = [input_edge]

    with pytest.raises(
        AssertionError, match=re.escape(f"Edge {input_edge} already exists as an input edge")
    ):
        node.append_input_edge(input_edge)


def test_append_edge_invalid_dest_node():
    node = Node(id=None, process_model=MatrixMult)
    other_node = Node()
    input_edge = DirectedEdge(
        data=None,
        edge_type=DirectedEdge.INPUT,
        source_node=None,
        dest_nodes=[other_node],
    )

    with pytest.raises(
        AssertionError,
        match=re.escape(
            f"Cannot append Edge to Node {node} that is assigned to {input_edge.dest_nodes}"
        ),
    ):
        node.append_input_edge(input_edge)


def test_append_input_edge():
    node = Node(id=None, process_model=MatrixMult)
    input_edge = DirectedEdge(
        data=None, edge_type=DirectedEdge.INPUT, source_node=None, dest_nodes=[node]
    )
    node.append_input_edge(input_edge)
    assert node.input_edges == [input_edge]


def test_insert_output_node_type_OUTPUT():
    node = Node(id=None, process_model=MatrixMult, network_type=Node.OUTPUT)
    output_node = Node(id=None, process_model=MatrixMult, network_type=Node.OUTPUT)
    node.insert_output_node(output_node)
    assert node.network_type == Node.NETWORK
    assert output_node.network_type == Node.OUTPUT
    assert node.output_nodes == [output_node]


def test_insert_output_node_type_INPUT():
    node = Node(id=None, process_model=MatrixMult, network_type=Node.INPUT)
    output_node = Node(id=None, process_model=MatrixMult)
    node.insert_output_node(output_node)
    assert node.network_type == Node.INPUT
    assert output_node.network_type == Node.OUTPUT
    assert node.output_nodes == [output_node]


def test_insert_output_node_type_NETWORK():
    node = Node(id=None, process_model=MatrixMult, network_type=Node.NETWORK)
    output_node = Node(id=None, process_model=MatrixMult)
    node.insert_output_node(output_node)
    assert node.network_type == Node.NETWORK
    assert output_node.network_type == Node.OUTPUT
    assert node.output_nodes == [output_node]


# def test_map_edges_to_output_nodes():
#     node = Node(id=None, process_model=MatrixMult)
#     destination_node = Node(id=None, process_model=MatrixMult)

#     node.insert_output_node(destination_node)

#     assert node.output_edges[0].dest_nodes == []
#     node.map_edges_to_output_nodes()
#     assert node.output_edges[0].dest_nodes == [destination_node]


def test_start_valid_data():
    node = Node(id=None, process_model=MatrixMult)
    input_edge1 = DirectedEdge(
        data=(Data(MatrixType(2, 4, element_dtype=float), {"Usage": "Matrix"}),),
        edge_type=DirectedEdge.INPUT,
        source_node=None,
        dest_nodes=[node],
    )
    input_edge2 = DirectedEdge(
        data=(Data(MatrixType(4, 5, element_dtype=float), {"Usage": "Matrix"}),),
        edge_type=DirectedEdge.INPUT,
        source_node=None,
        dest_nodes=[node],
    )

    node.append_input_edge(input_edge1)
    node.append_input_edge(input_edge2)
    assert node.input_edges == [input_edge1, input_edge2]

    node.start()
    assert node.process.inputs == [input_edge1.data[0], input_edge2.data[0]]

    node.process.update()
    result = node.process.generate_output()[0]
    assert result.data.rows == 2
    assert result.data.cols == 5


def test_process_update_valid_data():
    node = Node(id=None, process_model=MatrixMult)
    input_edge1 = DirectedEdge(
        data=(Data(MatrixType(2, 4, element_dtype=float), {"Usage": "Matrix"}),),
        edge_type=DirectedEdge.INPUT,
        source_node=None,
        dest_nodes=[node],
    )
    input_edge2 = DirectedEdge(
        data=(Data(MatrixType(4, 5, element_dtype=float), {"Usage": "Matrix"}),),
        edge_type=DirectedEdge.INPUT,
        source_node=None,
        dest_nodes=[node],
    )

    node.append_input_edge(input_edge1)
    node.append_input_edge(input_edge2)
    node.start()

    node.process.update()
    result = node.process.generate_output()[0]
    assert result.data.rows == 2
    assert result.data.cols == 5


def test_start_invalid_matrix_dimensions():
    node = Node(id=None, process_model=MatrixMult)
    input_edge1 = DirectedEdge(
        data=(Data(MatrixType(2, 4, element_dtype=float), {"Usage": "Matrix"}),),
        edge_type=DirectedEdge.INPUT,
        source_node=None,
        dest_nodes=[node],
    )
    input_edge2 = DirectedEdge(
        data=(Data(MatrixType(5, 5, element_dtype=float), {"Usage": "Matrix"}),),
        edge_type=DirectedEdge.INPUT,
        source_node=None,
        dest_nodes=[node],
    )

    node.append_input_edge(input_edge1)
    node.append_input_edge(input_edge2)
    assert node.input_edges == [input_edge1, input_edge2]

    with pytest.raises(
        AssertionError, match=f"Invalid input data for process {str(node.process)}"
    ):
        node.start()


def test_start_invalid_data_element_dtype():
    node = Node(id=None, process_model=MatrixMult)
    input_edge1 = DirectedEdge(
        data=(Data(MatrixType(2, 4, element_dtype=str), {"Usage": "Matrix"}),),
        edge_type=DirectedEdge.INPUT,
        source_node=None,
        dest_nodes=[node],
    )
    input_edge2 = DirectedEdge(
        data=(Data(MatrixType(4, 5, element_dtype=float), {"Usage": "Matrix"}),),
        edge_type=DirectedEdge.INPUT,
        source_node=None,
        dest_nodes=[node],
    )

    node.append_input_edge(input_edge1)
    node.append_input_edge(input_edge2)
    assert node.input_edges == [input_edge1, input_edge2]

    with pytest.raises(AssertionError, match="Invalid input data for process"):
        node.start()


def test_start_invalid_input_properties():
    node = Node(id=None, process_model=MatrixMult)
    input_edge1 = DirectedEdge(
        data=(Data(MatrixType(2, 4, element_dtype=str), {"Usage": "Invalid Usage"}),),
        edge_type=DirectedEdge.INPUT,
        source_node=None,
        dest_nodes=[node],
    )
    input_edge2 = DirectedEdge(
        data=(Data(MatrixType(4, 5, element_dtype=float), {"Usage": "Invalid Usage"}),),
        edge_type=DirectedEdge.INPUT,
        source_node=None,
        dest_nodes=[node],
    )

    node.append_input_edge(input_edge1)
    node.append_input_edge(input_edge2)
    assert node.input_edges == [input_edge1, input_edge2]

    with pytest.raises(
        InitError, match=re.escape(f"does not satisfy expected properties:")
    ):
        node.start()


def test_execution_exact_timestep():
    node = Node(id=None, process_model=MatrixMult)
    input_edge1 = DirectedEdge(
        data=(Data(MatrixType(2, 4, element_dtype=float), {"Usage": "Matrix"}),),
        edge_type=DirectedEdge.INPUT,
        source_node=None,
        dest_nodes=[node],
    )
    input_edge2 = DirectedEdge(
        data=(Data(MatrixType(4, 5, element_dtype=float), {"Usage": "Matrix"}),),
        edge_type=DirectedEdge.INPUT,
        source_node=None,
        dest_nodes=[node],
    )

    node.append_input_edge(input_edge1)
    node.append_input_edge(input_edge2)
    node.start()

    # Dummy allocation to satisfy allocation
    node.allocation = ClassicalAllocation(device_name="name", cores=1, memory=10**9)
    node.process.status = Process.ACTIVE
    node.process.time_till_completion = 5
    node.execute_process(timestep=5)
    assert node.process.time_till_completion == 0
    assert node.process.status == Process.COMPLETED


def test_execution_smaller_timestep():
    node = Node(id=None, process_model=MatrixMult)
    input_edge1 = DirectedEdge(
        data=(Data(MatrixType(2, 4, element_dtype=float), {"Usage": "Matrix"}),),
        edge_type=DirectedEdge.INPUT,
        source_node=None,
        dest_nodes=[node],
    )
    input_edge2 = DirectedEdge(
        data=(Data(MatrixType(4, 5, element_dtype=float), {"Usage": "Matrix"}),),
        edge_type=DirectedEdge.INPUT,
        source_node=None,
        dest_nodes=[node],
    )

    node.append_input_edge(input_edge1)
    node.append_input_edge(input_edge2)
    node.start()

    # Dummy allocation to satisfy allocation
    node.allocation = ClassicalAllocation(device_name="name", cores=1, memory=10**9)
    node.process.status = Process.ACTIVE
    node.process.time_till_completion = 5
    node.execute_process(timestep=4)
    assert node.process.time_till_completion == 1
    assert node.process.status == Process.ACTIVE


def test_execution_larger_timestep():
    node = Node(id=None, process_model=MatrixMult)
    input_edge1 = DirectedEdge(
        data=(Data(MatrixType(2, 4, element_dtype=float), {"Usage": "Matrix"}),),
        edge_type=DirectedEdge.INPUT,
        source_node=None,
        dest_nodes=[node],
    )
    input_edge2 = DirectedEdge(
        data=(Data(MatrixType(4, 5, element_dtype=float), {"Usage": "Matrix"}),),
        edge_type=DirectedEdge.INPUT,
        source_node=None,
        dest_nodes=[node],
    )

    node.append_input_edge(input_edge1)
    node.append_input_edge(input_edge2)
    node.start()

    # Dummy allocation to satisfy allocation
    node.allocation = ClassicalAllocation(device_name="name", cores=1, memory=10**9)
    node.process.status = Process.ACTIVE
    node.process.time_till_completion = 5

    with pytest.raises(
        AssertionError,
        match=f"{node} has negative time till completion of process execution",
    ):
        node.execute_process(timestep=6)


def test_execution_invalid_allocation():
    node = Node(id=None, process_model=MatrixMult)
    input_edge1 = DirectedEdge(
        data=(Data(MatrixType(2, 4, element_dtype=float), {"Usage": "Matrix"}),),
        edge_type=DirectedEdge.INPUT,
        source_node=None,
        dest_nodes=[node],
    )
    input_edge2 = DirectedEdge(
        data=(Data(MatrixType(4, 5, element_dtype=float), {"Usage": "Matrix"}),),
        edge_type=DirectedEdge.INPUT,
        source_node=None,
        dest_nodes=[node],
    )

    node.append_input_edge(input_edge1)
    node.append_input_edge(input_edge2)
    node.start()

    # Dummy allocation to satisfy allocation
    node.allocation = ClassicalAllocation(device_name="name", cores=1, memory=10**6)
    node.process.status = Process.ACTIVE
    node.process.time_till_completion = 5
    with pytest.raises(
        AssertionError,
        match=re.escape(
            f"Node {node.id} cannot execute it's process with given allocation {node.allocation}"
        ),
    ):
        node.execute_process(timestep=5)


def test_map_results_to_edge():
    node = Node(id=None, process_model=MatrixMult)
    input_edge1 = DirectedEdge(
        data=(Data(MatrixType(2, 4, element_dtype=float), {"Usage": "Matrix"}),),
        edge_type=DirectedEdge.INPUT,
        source_node=None,
        dest_nodes=[node],
    )
    input_edge2 = DirectedEdge(
        data=(Data(MatrixType(4, 5, element_dtype=float), {"Usage": "Matrix"}),),
        edge_type=DirectedEdge.INPUT,
        source_node=None,
        dest_nodes=[node],
    )

    node.append_input_edge(input_edge1)
    node.append_input_edge(input_edge2)
    node.start()

    # Dummy allocation to satisfy allocation
    node.allocation = ClassicalAllocation(device_name="name", cores=1, memory=10**9)
    node.process.status = Process.ACTIVE
    node.process.time_till_completion = 5
    node.execute_process(timestep=5)

    # assert node.output_edges[0].data[0].data == None
    # node.map_results_to_edges()

    # Dummy deallocation to satisfy deallocation
    node.allocation = None
    node.process.update()
    node.complete()
    assert isinstance(node.output_edges[0].data[0].data, MatrixType)
    assert node.output_edges[0].data[0].data.rows == 2
    assert node.output_edges[0].data[0].data.cols == 5
    assert node.output_edges[0].data[0].data.element_dtype == float
