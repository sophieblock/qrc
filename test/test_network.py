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
from workflow.simulation.refactor.Process_Library.matrix_ops import MatrixMult, ConcatMatrix
import pytest


class MatrixMult(ClassicalProcess):
    """A simple Process that takes two inputs of MatrixType() and returns
    the dimensions of the product of the matrices
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
        if self.inputs[0].data.cols == self.inputs[1].data.rows:
            return (
                self.inputs[0].data.rows
                * self.inputs[1].data.cols
                * (2 * self.inputs[0].data.cols - 1)
            )
        elif self.inputs[1].data.cols == self.inputs[0].data.rows:
            return (
                self.inputs[1].data.rows
                * self.inputs[0].data.cols
                * (2 * self.inputs[1].data.cols - 1)
            )

    def validate_data(self) -> bool:
        """Ensures input data can correctly run for the process. Will
        be unique for each process.
        """
        accepted_element_dtypes = [float, int]
        conditions = (
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


class GenVector(ClassicalProcess):
    """A simple Process that generates a MatrixType() object
    depending on the specified dimensions, Vector Orientation and element type

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

        if inputs != None:
            self.map_inputs_to_data()

    def map_inputs_to_data(self):
        self.data = {}
        for input in self.inputs:
            usage = input.properties["Usage"]
            self.data[usage] = input.data

        if (
            self.data["Vector Orientation"] == "Row Vector"
            and self.data["Vector Rows"] == 1
        ):
            self.data["Data Type"] = "Vector"
        elif (
            self.data["Vector Orientation"] == "Column Vector"
            and self.data["Vector Columns"] == 1
        ):
            self.data["Data Type"] = "Vector"
        else:
            self.data["Data Type"] = "Matrix"

    def set_expected_input_properties(self):
        """Sets the expected input properties of the Process to execute/update successfully.
        This method should update self.expected_input_properties to be a list of dictionaries, with each
        dictionary specifying the expected properties of an input Data instance. A Process that requires
        N inputs will require N such dictionaries specifying its properties. Will be unique to each Process.
        """

        self.expected_input_properties = [
            {"Data Type": int, "Usage": "Vector Rows"},
            {"Data Type": int, "Usage": "Vector Columns"},
            {"Data Type": str, "Usage": "Vector Orientation"},
            {"Data Type": type(float), "Usage": "Element Type"},
        ]

    def set_required_resources(self):
        """Sets the appropriate resources required to run the process. This method should update
        self.required_resources to be a list of Resources required for the Process to successfully run.
        Will be unique for each process.

        """

        self.required_resources = ClassicalResource(memory=500 * 10**3, num_cores=1)

    def set_output_properties(self):
        """Sets the expected properties of the generated outputs of the Process.
        This method should update self.output_properties to be a list of dictionaries,
        with each dictionary specifying the expected properties of each output Data
        """

        self.output_properties = [
            {
                "Data Type": type(MatrixType()),
                "Usage": ("Matrix", "Vector"),
                "Vector Orientation": ("Row Vector", "Column Vector"),
            }
        ]

    def compute_flop_count(self):
        """Computes the number of required flops for this process to complete given
        valid inputs from self.inputs. Will be unique to each Process.
        """
        dims = []
        for input in self.inputs:
            if isinstance(input.data, int):
                dims.append(input.data)

        return dims[0] * dims[1]

    def validate_data(self) -> bool:
        """Ensures input data can correctly run for the process. Will
        be unique for each process

        """
        accepted_element_dtypes = [float, int]
        accepted_orientations = ["Row Vector", "Column Vector"]
        conditions = (
            self.data["Vector Rows"] > 0,
            self.data["Vector Columns"] > 0,
            self.data["Vector Orientation"] in accepted_orientations,
            self.data["Element Type"] in accepted_element_dtypes,
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
        self.compute_required_resources()
        self.result = MatrixType(
            self.data["Vector Rows"],
            self.data["Vector Columns"],
            self.data["Element Type"],
        )

    def compute_required_resources(self):
        pass

    def generate_output(self) -> list[Data]:
        """After completing updates, the process will wrap its results
        and return a (unordered) set of Data states to be processed by Node.
        The expected output will be unique to the process.
        """

        return (
            Data(
                data=self.result,
                properties={
                    "Usage": self.data["Data Type"],
                    "Vector Orientation": self.data["Vector Orientation"],
                },
            ),
        )


class ConcatMatrix(ClassicalProcess):
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
            {"Data Type": type(MatrixType()), "Usage": ("Matrix", "Vector")}
        ] * 2

    def set_required_resources(self):
        """Sets the appropriate resources required to run the process. This method should update
        self.required_resources to be a list of Resources required for the Process to successfully run.
        Will be unique for each process.

        """

        self.required_resources = ClassicalResource(memory=2 * 10**6, num_cores=1)

    def set_output_properties(self):
        """Sets the expected properties of the generated outputs of the Process.
        This method should update self.output_properties to be a list of dictionaries,
        with each dictionary specifying the expected properties of each output Data
        """

        self.output_properties = [
            {
                "Data Type": type(MatrixType()),
                "Usage": "Matrix",
                "Vector Orientation": ("Row Vector", "Column Vector"),
            }
        ]

    def compute_flop_count(self):
        """Computes the number of required flops for this process to complete given
        valid inputs from self.inputs. Will be unique to each Process.
        """
        return (
            self.inputs[0].data.cols * self.inputs[0].data.rows
            + self.inputs[1].data.cols * self.inputs[1].data.rows
        )

    def validate_data(self) -> bool:
        """Ensures input data can correctly run for the process. Will
        be unique for each process

        """
        accepted_element_dtypes = [float, int]
        accepted_orientations = ["Row Vector", "Column Vector"]
        conditions = (
            all(
                input.data.element_dtype in accepted_element_dtypes
                for input in self.inputs
            ),
            all(
                input.properties["Vector Orientation"] in accepted_orientations
                for input in self.inputs
            ),
            self.same_orientations(),
            self.valid_dimensions(),
        )

        return all(condition for condition in conditions)

    def same_orientations(self):
        return (
            self.inputs[0].properties["Vector Orientation"]
            == self.inputs[1].properties["Vector Orientation"]
        )

    def valid_dimensions(self) -> bool:
        if self.inputs[0].properties["Vector Orientation"] == "Row Vector":
            return self.inputs[0].data.cols == self.inputs[1].data.cols
        elif self.inputs[0].properties["Vector Orientation"] == "Column Vector":
            return self.inputs[0].data.rows == self.inputs[1].data.rows
        else:
            return False

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
        if self.inputs[0].properties["Vector Orientation"] == "Row Vector":
            self.result = MatrixType(
                sum(input.data.rows for input in self.inputs),
                self.inputs[0].data.cols,
                element_dtype=element_dtype,
            )
        elif self.inputs[0].properties["Vector Orientation"] == "Column Vector":
            self.result = MatrixType(
                self.inputs[0].data.rows,
                sum(input.data.cols for input in self.inputs),
                element_dtype=element_dtype,
            )

    def compute_required_resources(self):
        pass

    def generate_output(self) -> list[Data]:
        """After completing updates, the process will wrap its results
        and return a (unordered) set of Data states to be processed by Node.
        The expected output will be unique to the process.
        """

        return (
            Data(
                data=self.result,
                properties={
                    "Usage": "Matrix",
                    "Vector Orientation": self.inputs[0].properties[
                        "Vector Orientation"
                    ],
                },
            ),
        )


def generate_vector(rows, cols, vec_orientation, dtype):
    vector_rows = Data(rows, {"Usage": "Vector Rows"})
    vector_cols = Data(cols, {"Usage": "Vector Columns"})
    orientation = Data(vec_orientation, {"Usage": "Vector Orientation"})
    element_dtype = Data(dtype, {"Usage": "Element Type"})

    return GenVector(inputs=[vector_rows, vector_cols, orientation, element_dtype])


def generate_vector_alt_order(rows, cols, vec_orientation, dtype):
    vector_rows = Data(rows, {"Usage": "Vector Rows"})
    vector_cols = Data(cols, {"Usage": "Vector Columns"})
    orientation = Data(vec_orientation, {"Usage": "Vector Orientation"})
    element_dtype = Data(dtype, {"Usage": "Element Type"})

    return GenVector(inputs=[orientation, vector_rows, element_dtype, vector_cols])


def test_generate_vector():
    vec = generate_vector(1, 5, "Row Vector", float)


def test_generate_vector_alt_order():
    vec = generate_vector(1, 5, "Row Vector", float)
    alt_vec = generate_vector_alt_order(1, 5, "Row Vector", float)

    assert vec.validate_data()
    assert alt_vec.validate_data()

    vec.update()
    alt_vec.update()

    assert vec.result.rows == alt_vec.result.rows
    assert vec.result.cols == alt_vec.result.cols
    assert vec.result.element_dtype == alt_vec.result.element_dtype


def test_concat_matrix_row():
    vec1 = generate_vector(1, 5, "Row Vector", float)
    vec2 = generate_vector(2, 5, "Row Vector", float)

    vec1.update()
    mat1 = vec1.generate_output()[0]

    vec2.update()
    mat2 = vec2.generate_output()[0]
    concat_mat = ConcatMatrix(inputs=[mat1, mat2])

    assert concat_mat.validate_data()
    concat_mat.update()
    result = concat_mat.generate_output()

    assert result[0].data.rows == 3
    assert result[0].data.cols == 5
    assert result[0].properties["Vector Orientation"] == "Row Vector"
    assert result[0].data.element_dtype == float


def test_concat_matrix_col():
    vec1 = generate_vector(5, 1, "Column Vector", float)
    vec2 = generate_vector(5, 2, "Column Vector", float)

    vec1.update()
    mat1 = vec1.generate_output()[0]

    vec2.update()
    mat2 = vec2.generate_output()[0]
    concat_mat = ConcatMatrix(inputs=[mat1, mat2])

    assert concat_mat.validate_data()
    concat_mat.update()
    result = concat_mat.generate_output()

    assert result[0].data.rows == 5
    assert result[0].data.cols == 3
    assert result[0].properties["Vector Orientation"] == "Column Vector"
    assert result[0].data.element_dtype == float


def prepare_genvec_input_edges(node: Node, rows, cols, vec_orientation, dtype):
    input_edge = DirectedEdge(
        data=(
            Data(rows, {"Usage": "Vector Rows"}),
            Data(cols, {"Usage": "Vector Columns"}),
            Data(vec_orientation, {"Usage": "Vector Orientation"}),
            Data(dtype, {"Usage": "Element Type"}),
        ),
        edge_type=DirectedEdge.INPUT,
        source_node=None,
        dest_nodes=[node],
    )

    node.append_input_edge(input_edge)


def generate_broker():
    supercomputer = ClassicalDevice(
        device_name="Supercomputer",
        processor_type="CPU",
        RAM=100 * 10**9,
        properties={"Cores": 20, "Clock Speed": 3 * 10**9},
    )

    broker = Broker(classical_devices=[supercomputer])
    return broker


def validate_node_types(network: Network):
    assert all(input_node in network.nodes for input_node in network.input_nodes)
    assert all(output_node in network.nodes for output_node in network.output_nodes)

    def node_type(node):
        if node.network_type == Node.INPUT:
            return node in network.input_nodes
        elif node.network_type == Node.OUTPUT:
            return node in network.output_nodes
        else:
            return node.network_type == Node.NETWORK

    return all(node_type(node) for node in network.nodes)


def test_concat_matrix_network():
    genvec_node1 = Node(id=None, process_model=GenVector, network_type=Node.INPUT)
    genvec_node2 = Node(id=None, process_model=GenVector, network_type=Node.INPUT)
    concat_node = Node(id=None, process_model=ConcatMatrix, network_type=Node.OUTPUT)

    genvec_node1.insert_output_node(concat_node)
    genvec_node2.insert_output_node(concat_node)

    starting_inputs = [
        [
            Data(1, {"Usage": "Vector Rows"}),
            Data(5, {"Usage": "Vector Columns"}),
            Data("Row Vector", {"Usage": "Vector Orientation"}),
            Data(float, {"Usage": "Element Type"}),
        ],
        [
            Data(2, {"Usage": "Vector Rows"}),
            Data(5, {"Usage": "Vector Columns"}),
            Data("Row Vector", {"Usage": "Vector Orientation"}),
            Data(float, {"Usage": "Element Type"}),
        ],
    ]

    network = Network(
        "Concatenate Matrices",
        nodes=[genvec_node1, genvec_node2, concat_node],
        input_nodes=[genvec_node1, genvec_node2],
        output_nodes=[concat_node],
        broker=generate_broker(),
    )
    assert validate_node_types(network)
    df = network.run(network.input_nodes, starting_inputs)

    assert len(concat_node.output_edges) == 1
    output_edge = concat_node.output_edges[0]
    result = output_edge.data[0]
    assert result.data.rows == 3
    assert result.data.cols == 5
    assert result.properties["Vector Orientation"] == "Row Vector"
    assert result.data.element_dtype == float

    assert df.loc[0]["Start Time [s]"] == 0.0
    assert (
        df.loc[0]["End Time [s]"]
        == genvec_node1.process.compute_flop_count()
        / network.broker.classical_devices[0].clock_frequency
    )
    assert (
        df.loc[0]["Memory Cost [B]"] == genvec_node1.process.required_resources.memory
    )

    assert df.loc[1]["Start Time [s]"] == 0.0
    assert (
        df.loc[1]["End Time [s]"]
        == genvec_node2.process.compute_flop_count()
        / network.broker.classical_devices[0].clock_frequency
    )
    assert (
        df.loc[1]["Memory Cost [B]"] == genvec_node2.process.required_resources.memory
    )

    assert df.loc[2]["Start Time [s]"] == max(
        df.loc[0]["End Time [s]"], df.loc[1]["End Time [s]"]
    )
    assert (
        df.loc[2]["End Time [s]"]
        == df.loc[2]["Start Time [s]"]
        + concat_node.process.compute_flop_count()
        / network.broker.classical_devices[0].clock_frequency
    )
    assert df.loc[2]["Memory Cost [B]"] == concat_node.process.required_resources.memory


def generate_concat_matrix_network(row1, col1, row2, col2, orientation):
    genvec_node1 = Node(id=None, process_model=GenVector, network_type=Node.INPUT)
    genvec_node2 = Node(id=None, process_model=GenVector, network_type=Node.INPUT)
    concat_node = Node(id=None, process_model=ConcatMatrix, network_type=Node.OUTPUT)

    genvec_node1.insert_output_node(concat_node)
    genvec_node2.insert_output_node(concat_node)

    prepare_genvec_input_edges(genvec_node1, row1, col1, orientation, float)
    prepare_genvec_input_edges(genvec_node2, row2, col2, orientation, float)

    return Network(
        "Concatenate Matrices",
        nodes=[genvec_node1, genvec_node2, concat_node],
        input_nodes=[genvec_node1, genvec_node2],
        output_nodes=[concat_node],
        broker=generate_broker(),
    )


def test_combine_network():
    network1 = generate_concat_matrix_network(1, 5, 2, 5, "Row Vector")
    network2 = generate_concat_matrix_network(3, 5, 4, 5, "Row Vector")
    network = network1.combine(network2)

    assert validate_node_types(network)
    assert network.input_nodes == network1.input_nodes + network2.input_nodes
    assert network.output_nodes == network1.output_nodes + network2.output_nodes
    assert len(network.broker.classical_devices) == 1
    assert network.broker.classical_devices == network1.broker.classical_devices
    assert network.broker.classical_devices == network2.broker.classical_devices


def test_extend_network_output_node():
    network1 = generate_concat_matrix_network(1, 5, 2, 5, "Row Vector")
    network2 = generate_concat_matrix_network(3, 5, 4, 5, "Row Vector")
    network = network1.combine(network2)

    concat_node = Node(id=None, process_model=ConcatMatrix, network_type=Node.OUTPUT)
    network.extend_node_destination(
        network1.output_nodes + network2.output_nodes, concat_node
    )

    assert validate_node_types(network)
    assert network.output_nodes[0] == concat_node
    assert network1.output_nodes[0].output_nodes == [concat_node]
    assert network2.output_nodes[0].output_nodes == [concat_node]
    assert network1.output_nodes[0].network_type == Node.NETWORK
    assert network2.output_nodes[0].network_type == Node.NETWORK


def test_combine_concat_matrix_networks():
    network1 = generate_concat_matrix_network(1, 5, 2, 5, "Row Vector")
    network2 = generate_concat_matrix_network(3, 5, 4, 5, "Row Vector")
    network = network1.combine(network2)

    concat_node = Node(id=None, process_model=ConcatMatrix, network_type=Node.OUTPUT)
    network.extend_node_destination(
        network1.output_nodes + network2.output_nodes, concat_node
    )
    assert validate_node_types(network)
    df = network.run(network.input_nodes)

    assert len(concat_node.output_edges) == 1
    output_edge = concat_node.output_edges[0]
    result = output_edge.data[0]
    assert result.data.rows == 10
    assert result.data.cols == 5
    assert result.properties["Vector Orientation"] == "Row Vector"
    assert result.data.element_dtype == float

    def get_dataframe_node(dataframe_idx) -> Node:
        return network.find_node_with_id(df.loc[dataframe_idx]["Node Idx"])

    print(df.to_string())
    ## verify start time, end time and memory costs for genvector (first 4 rows in df)
    for df_idx in range(4):
        node: Node = get_dataframe_node(df_idx)
        assert df.loc[df_idx]["Start Time [s]"] == 0.0
        assert (
            df.loc[df_idx]["End Time [s]"]
            == node.process.compute_flop_count()
            / network.broker.classical_devices[0].clock_frequency
        )
        assert (
            df.loc[df_idx]["Memory Cost [B]"] == node.process.required_resources.memory
        )

    # ConcatMatrix begins when its GenVectors complete:
    node: Node = get_dataframe_node(4)
    assert df.loc[4]["Start Time [s]"] == max(
        df.loc[0]["End Time [s]"], df.loc[1]["End Time [s]"]
    )
    assert (
        df.loc[4]["End Time [s]"]
        == df.loc[4]["Start Time [s]"]
        + node.process.compute_flop_count()
        / network.broker.classical_devices[0].clock_frequency
    )
    assert df.loc[4]["Memory Cost [B]"] == node.process.required_resources.memory

    node: Node = get_dataframe_node(5)
    assert df.loc[5]["Start Time [s]"] == max(
        df.loc[2]["End Time [s]"], df.loc[3]["End Time [s]"]
    )
    assert (
        df.loc[5]["End Time [s]"]
        == df.loc[5]["Start Time [s]"]
        + node.process.compute_flop_count()
        / network.broker.classical_devices[0].clock_frequency
    )
    assert df.loc[5]["Memory Cost [B]"] == node.process.required_resources.memory

    # MatrixMult begins when both ConcatMatrix completes:
    node: Node = get_dataframe_node(6)
    assert df.loc[6]["Start Time [s]"] == max(
        df.loc[4]["End Time [s]"], df.loc[5]["End Time [s]"]
    )
    assert (
        df.loc[6]["End Time [s]"]
        == df.loc[6]["Start Time [s]"]
        + node.process.compute_flop_count()
        / network.broker.classical_devices[0].clock_frequency
    )
    assert df.loc[6]["Memory Cost [B]"] == node.process.required_resources.memory


def test_combine_matmul_matrix_networks():
    base_network1 = generate_concat_matrix_network(5, 1, 5, 2, "Column Vector")
    base_network2 = generate_concat_matrix_network(5, 3, 5, 2, "Column Vector")
    concat_network1 = base_network1.combine(base_network2)

    concat_node = Node(id=None, process_model=ConcatMatrix, network_type=Node.OUTPUT)
    concat_network1.extend_node_destination(
        base_network1.output_nodes + base_network2.output_nodes, concat_node
    )

    base_network1 = generate_concat_matrix_network(1, 5, 2, 5, "Row Vector")
    base_network2 = generate_concat_matrix_network(3, 5, 4, 5, "Row Vector")
    concat_network2 = base_network1.combine(base_network2)

    concat_node = Node(id=None, process_model=ConcatMatrix, network_type=Node.OUTPUT)
    concat_network2.extend_node_destination(
        base_network1.output_nodes + base_network2.output_nodes, concat_node
    )

    matmul_network = concat_network1.combine(concat_network2)
    matmul_node = Node(id=None, process_model=MatrixMult, network_type=Node.OUTPUT)
    matmul_network.extend_node_destination(
        concat_network1.output_nodes + concat_network2.output_nodes, matmul_node
    )
    assert validate_node_types(matmul_network)
    matmul_network.run(matmul_network.input_nodes)

    assert len(matmul_node.output_edges) == 1
    output_edge = matmul_node.output_edges[0]
    result = output_edge.data[0]
    assert result.data.rows == 10
    assert result.data.cols == 8
    assert result.data.element_dtype == float


def test_add_matmul_matrix_networks():
    base_network1 = generate_concat_matrix_network(5, 1, 5, 2, "Column Vector")
    base_network2 = generate_concat_matrix_network(5, 3, 5, 2, "Column Vector")
    concat_network1 = base_network1.combine(base_network2)

    for output_node in concat_network1.output_nodes:
        output_node.process.output_properties[0]["Vector Orientation"] = "Column Vector"

    base_network1 = generate_concat_matrix_network(1, 5, 2, 5, "Row Vector")
    base_network2 = generate_concat_matrix_network(3, 5, 4, 5, "Row Vector")
    concat_network2 = base_network1.combine(base_network2)

    # for output_node in concat_network2.output_nodes:
    #     output_node.output_edges[0].data[0].properties[
    #         "Vector Orientation"
    #     ] = "Row Vector"

    for output_node in concat_network2.output_nodes:
        output_node.process.output_properties[0]["Vector Orientation"] = "Row Vector"

    input_network = concat_network1.combine(concat_network2)

    concat_node1 = Node(id=None, process_model=ConcatMatrix, network_type=Node.INPUT)
    concat_node2 = Node(id=None, process_model=ConcatMatrix, network_type=Node.INPUT)
    matmul_node = Node(id=None, process_model=MatrixMult, network_type=Node.OUTPUT)

    concat_node1.insert_output_node(matmul_node)
    concat_node2.insert_output_node(matmul_node)
    output_network = Network(
        name="Concat and Matrix Mult",
        nodes=[concat_node1, concat_node2, matmul_node],
        input_nodes=[concat_node1, concat_node2],
        output_nodes=[matmul_node],
        broker=generate_broker(),
    )

    # We want to add the outputs of input_network (2 column and 2 row matrices) as the inputs
    # of output_network which consists of 2 matrix concatenate nodes and a final matrix mult node.
    # Since the edge-to-node assignments are ambiguous at this point, we will add additional
    # expected_input_properties to include "Vertex Orientation"

    for input_property in output_network.input_nodes[0].expected_input_properties:
        input_property["Vector Orientation"] = "Row Vector"
    for input_property in output_network.input_nodes[1].expected_input_properties:
        input_property["Vector Orientation"] = "Column Vector"

    network = input_network + output_network

    assert len(network.input_nodes) == 8
    assert len(network.output_nodes) == 1
    assert len(network.broker.classical_devices) == 1
    assert network.broker.classical_devices == input_network.broker.classical_devices
    assert network.broker.classical_devices == output_network.broker.classical_devices
    assert validate_node_types(network)
    df = network.run(network.input_nodes)

    assert len(network.output_nodes[0].output_edges) == 1
    output_edge = network.output_nodes[0].output_edges[0]
    result = output_edge.data[0]
    assert result.data.rows == 10
    assert result.data.cols == 8
    assert result.data.element_dtype == float
