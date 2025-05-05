from qrew.simulation.refactor.data import Data
from qrew.simulation.refactor.data_types import *
from qrew.simulation.refactor.process import Process, ClassicalProcess
from qrew.simulation.refactor.graph import Node, Network
from qrew.simulation.refactor.resources import Resource
from qrew.simulation.refactor.resources.classical_resources import ClassicalResource
from qrew.simulation.refactor.data_types import MatrixType, TensorType
import sys, copy
from qrew.simulation.refactor.utilities import compute_resources
# from numpy import infty as INFINITY
import numpy as np
from numpy.typing import NDArray

class GE_FindPivot(ClassicalProcess):
    def __init__(
        self,
        inputs: list[Data] = None,
        expected_input_properties=None,
        required_resources=None,
        output_properties=None,
    ):
        if inputs != None:
            self.input_data = {}
            for data in inputs:
                self.input_data[data.properties["Usage"]] = data.data

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
            {"Data Type": type(np.array([])), "Usage": "Matrix"},
            {"Data Type": int, "Usage": "Column Idx"},
        ]

    def set_required_resources(self):
        """Sets the appropriate resources required to run the process. This method should update
        self.required_resources to be a list of Resources required for the Process to successfully run.
        Will be unique for each process.
        """

        self.required_resources = ClassicalResource(
            memory=sys.getsizeof(self.input_data["Matrix"]),
            num_cores=1,
        )

    def set_output_properties(self):
        """Sets the expected properties of the generated outputs of the Process.
        This method should update self.output_properties to be a list of dictionaries,
        with each dictionary specifying the expected properties of each output Data
        """

        self.output_properties = [
            {"Data Type": TensorType, "Usage": "Matrix"},
            {"Data Type": int, "Usage": "Pivot Idx"},
            {"Data Type": int, "Usage": "Column Idx"},
        ]

    def compute_flop_count(self):
        """Computes the number of required flops for this process to complete given
        valid inputs from self.inputs. Will be unique to each Process.
        """
        return 10

    def validate_data(self) -> bool:
        """Ensures input data can correctly run for the process. Will
        be unique for each process.
        """

        conditions = (self.valid_dimensions(), self.valid_column_idx())

        return all(condition for condition in conditions)

    def valid_dimensions(self) -> bool:
        dims = self.input_data["Matrix"].shape
        # return len(dims) == 2 and dims[0] == dims[1] - 1
        return len(dims) == 2

    def valid_column_idx(self) -> bool:
        dims = self.input_data["Matrix"].shape
        return self.input_data["Column Idx"] <= dims[0]
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

        matrix = self.input_data["Matrix"]
        matrix_size = matrix.shape[0]
        column_idx = self.input_data["Column Idx"]

        idx_max = column_idx
        value_max = matrix[idx_max][column_idx]

        for idx in range(column_idx + 1, matrix_size):
            if abs(matrix[idx][column_idx]) > value_max:
                value_max = matrix[idx][column_idx]
                idx_max = idx

        self.result = idx_max

    def compute_required_resources(self):
        pass

    def generate_output(self) -> list[Data]:
        """After completing updates, the process will wrap its results
        and return a (unordered) set of Data states to be processed by Node.
        The expected output will be unique to the process.
        """
        
        return (
            Data(data=self.input_data["Matrix"], properties={"Usage": "Matrix"}),
            Data(data=self.result, properties={"Usage": "Pivot Idx"}),
            Data(
                data=self.input_data["Column Idx"], properties={"Usage": "Column Idx"}
            ),
        )


class GE_SwapRows(ClassicalProcess):
    def __init__(
        self,
        inputs: list[Data] = None,
        expected_input_properties=None,
        required_resources=None,
        output_properties=None,
    ):
        if inputs != None:
            self.input_data = {}
            for data in inputs:
                self.input_data[data.properties["Usage"]] = data.data

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
            {"Data Type": type(np.array([])), "Usage": "Matrix"},
            {"Data Type": int, "Usage": "Pivot Idx"},
            {"Data Type": int, "Usage": "Column Idx"},
        ]

    def set_required_resources(self):
        """Sets the appropriate resources required to run the process. This method should update
        self.required_resources to be a list of Resources required for the Process to successfully run.
        Will be unique for each process.
        """

        self.required_resources = ClassicalResource(
            memory=sys.getsizeof(self.input_data["Matrix"])
            + sys.getsizeof(self.input_data["Matrix"][self.input_data["Pivot Idx"]]),
            num_cores=1,
        )

    def set_output_properties(self):
        """Sets the expected properties of the generated outputs of the Process.
        This method should update self.output_properties to be a list of dictionaries,
        with each dictionary specifying the expected properties of each output Data
        """

        self.output_properties = [
            {"Data Type": type(np.array([])), "Usage": "Matrix"},
            {"Data Type": int, "Usage": "Column Idx"},
        ]

    def compute_flop_count(self):
        """Computes the number of required flops for this process to complete given
        valid inputs from self.inputs. Will be unique to each Process.
        """
        return 10

    def validate_data(self) -> bool:
        """Ensures input data can correctly run for the process. Will
        be unique for each process.
        """

        conditions = (
            self.valid_dimensions(),
            self.valid_column_idx(),
            self.valid_pivot_idx(),
        )

        return all(condition for condition in conditions)

    def valid_dimensions(self) -> bool:
        dims = self.input_data["Matrix"].shape
        # return len(dims) == 2 and dims[0] == dims[1] - 1
        return len(dims) == 2

    def valid_column_idx(self) -> bool:
        dims = self.input_data["Matrix"].shape
        return self.input_data["Column Idx"] <= dims[0]

    def valid_pivot_idx(self) -> bool:
        dims = self.input_data["Matrix"].shape
        return self.input_data["Pivot Idx"] <= dims[0]
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

        matrix = self.input_data["Matrix"]
        pivot_idx = self.input_data["Pivot Idx"]
        current_idx = self.input_data["Column Idx"]
        logger.debug(f' - Matrix in: \n{matrix},\n - piv idx: {pivot_idx}\n - current idx: {current_idx}')
        temp = copy.copy(matrix[pivot_idx])
        matrix[pivot_idx] = matrix[current_idx]
        matrix[current_idx] = temp
        logger.debug(f'Updated matrix: \n{matrix}')
        self.result = matrix

        self.compute_required_resources()

    def compute_required_resources(self):
        pass

    def generate_output(self) -> list[Data]:
        """After completing updates, the process will wrap its results
        and return a (unordered) set of Data states to be processed by Node.
        The expected output will be unique to the process.
        """

        return (
            Data(data=self.result, properties={"Usage": "Matrix"}),
            Data(
                data=self.input_data["Column Idx"], properties={"Usage": "Column Idx"}
            ),
        )

class GE_RowReduction(ClassicalProcess):
    def __init__(
        self,
        inputs: list[Data] = None,
        row_idx: int = None,
        expected_input_properties=None,
        required_resources=None,
        output_properties=None,
    ):
        if inputs != None:
            self.input_data = {}
            for data in inputs:
                self.input_data[data.properties["Usage"]] = data.data
        self.row_idx = row_idx
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
            {"Data Type": type(np.array([])), "Usage": "Principle Row"},
            {
                "Data Type": type(np.array([])),
                "Usage": "Reduction Row",
                "Row Idx": self.row_idx,
            },
            {"Data Type": int, "Usage": "Column Idx"},
        ]

    def set_required_resources(self):
        """Sets the appropriate resources required to run the process. This method should update
        self.required_resources to be a list of Resources required for the Process to successfully run.
        Will be unique for each process.
        """

        self.required_resources = ClassicalResource(
            memory=sys.getsizeof(self.input_data["Principle Row"])
            + sys.getsizeof(self.input_data["Reduction Row"]),
            num_cores=1,
        )

    def set_output_properties(self):
        """Sets the expected properties of the generated outputs of the Process.
        This method should update self.output_properties to be a list of dictionaries,
        with each dictionary specifying the expected properties of each output Data
        """

        self.output_properties = [
            {"Data Type": type(np.array([])), "Usage": "Reduced Row"}
        ]

    def compute_flop_count(self):
        """Computes the number of required flops for this process to complete given
        valid inputs from self.inputs. Will be unique to each Process.
        """
        row_length = len(self.input_data["Reduction Row"])

        return 2 * row_length + 1

    def validate_data(self) -> bool:
        """Ensures input data can correctly run for the process. Will
        be unique for each process.
        """
        conditions = (self.valid_dimensions(),)

        return all(condition for condition in conditions)

    def valid_dimensions(self) -> bool:
        principle_dims = self.input_data["Principle Row"].shape
        reduction_dims = self.input_data["Reduction Row"].shape
        return principle_dims == reduction_dims
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
        principle_row = self.input_data["Principle Row"]
        reduction_row = self.input_data["Reduction Row"]
        column_idx = self.input_data["Column Idx"]

        reduction_factor = reduction_row[column_idx] / principle_row[column_idx]

        for idx in range(column_idx, len(principle_row)):
            reduction_row[idx] = (
                reduction_row[idx] - principle_row[idx] * reduction_factor
            )

        self.result = reduction_row

        self.compute_required_resources()

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
                properties={"Usage": "Reduced Row", "Row Idx": self.row_idx},
            ),
        )


class GE_RowDeconstruction(ClassicalProcess):
    def __init__(
        self,
        inputs: list[Data] = None,
        expected_input_properties=None,
        required_resources=None,
        output_properties=None,
        continue_itr=True,
    ):
        if inputs != None:
            self.input_data = {}
            for data in inputs:
                self.input_data[data.properties["Usage"]] = data.data

        super().__init__(
            inputs,
            expected_input_properties,
            required_resources,
            output_properties,
            dynamic=True,
        )
        self.continue_itr = continue_itr

    def set_expected_input_properties(self):
        """Sets the expected input properties of the Process to execute/update successfully.
        This method should update self.expected_input_properties to be a list of dictionaries, with each
        dictionary specifying the expected properties of an input Data instance. A Process that requires
        N inputs will require N such dictionaries specifying its properties. Will be unique to each Process.
        """

        self.expected_input_properties = [
            {"Data Type": type(np.array([])), "Usage": "Matrix"},
            {"Data Type": int, "Usage": "Column Idx"},
        ]

    def set_required_resources(self):
        """Sets the appropriate resources required to run the process. This method should update
        self.required_resources to be a list of Resources required for the Process to successfully run.
        Will be unique for each process.
        """

        self.required_resources = ClassicalResource(
            memory=sys.getsizeof(self.input_data["Matrix"]),
            num_cores=1,
        )

    def set_output_properties(self):
        """Sets the expected properties of the generated outputs of the Process.
        This method should update self.output_properties to be a list of dictionaries,
        with each dictionary specifying the expected properties of each output Data
        """

        self.output_properties = [
            {"Data Type": type(np.array([])), "Usage": "Matrix"},
            {"Data Type": int, "Usage": "Column Idx"},
            {"Data Type": type(np.array([])), "Usage": "Principle Row"},
            {"Data Type": type(np.array([])), "Usage": "Reduction Row"},
        ]

    def compute_flop_count(self):
        """Computes the number of required flops for this process to complete given
        valid inputs from self.inputs. Will be unique to each Process.
        """

        return 10

    def validate_data(self) -> bool:
        """Ensures input data can correctly run for the process. Will
        be unique for each process.
        """

        conditions = (self.valid_dimensions(), self.valid_column_idx())

        return all(condition for condition in conditions)

    def valid_dimensions(self) -> bool:
        dims = self.input_data["Matrix"].shape
        # return len(dims) == 2 and dims[0] == dims[1] - 1
        return len(dims) == 2

    def valid_column_idx(self) -> bool:
        dims = self.input_data["Matrix"].shape
        return self.input_data["Column Idx"] <= dims[0]
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

    def compute_required_resources(self):
        pass

    def generate_output(self) -> list[Data]:
        """After completing updates, the process will wrap its results
        and return a (unordered) set of Data states to be processed by Node.
        The expected output will be unique to the process.
        """

        matrix = self.input_data["Matrix"]
        column_idx = self.input_data["Column Idx"]

        output = [
            Data(data=matrix, properties={"Usage": "Matrix"}),
            Data(data=column_idx, properties={"Usage": "Column Idx"}),
            Data(data=matrix[column_idx], properties={"Usage": "Principle Row"}),
        ]

        for row_idx in range(column_idx + 1, matrix.shape[0]):
            output.append(
                Data(
                    data=matrix[row_idx],
                    properties={"Usage": "Reduction Row", "Row Idx": row_idx},
                )
            )

        return tuple(output)

    def extend_network(self) -> Network:
        matrix_size = self.input_data["Matrix"].shape[0]
        column_idx = self.input_data["Column Idx"]

        input_nodes = [
            Node(process_model=GE_RowReduction, network_type="INPUT", row_idx=row_idx)
            for row_idx in range(column_idx + 1, matrix_size)
        ]
        output_nodes = [
            Node(
                process_model=GE_RowReconstruction,
                network_type="OUTPUT",
                column_idx=column_idx,
                matrix_size=matrix_size,
                continue_itr=self.continue_itr,
            )
        ]

        for node in input_nodes:
            node.output_nodes = output_nodes

        network = Network(
            name="Row Reduction",
            nodes=input_nodes + output_nodes,
            input_nodes=input_nodes + output_nodes,
            output_nodes=output_nodes,
            broker=None,
        )

        return network


class GE_RowReconstruction(ClassicalProcess):
    def __init__(
        self,
        column_idx,
        matrix_size,
        inputs: list[Data] = None,
        expected_input_properties=None,
        required_resources=None,
        output_properties=None,
        continue_itr: bool = True,
    ):
        if inputs != None:
            self.input_data = {}
            for data in inputs:
                if "Row Idx" in data.properties.keys():
                    self.input_data[
                        "Reduced Row " + str(data.properties["Row Idx"])
                    ] = data.data
                else:
                    self.input_data[data.properties["Usage"]] = data.data

        self.column_idx = column_idx
        self.matrix_size = matrix_size

        if continue_itr:
            is_dynamic = True if column_idx < matrix_size - 1 else False
        else:
            is_dynamic = False

        super().__init__(
            inputs,
            expected_input_properties,
            required_resources,
            output_properties,
            dynamic=is_dynamic,
        )

    def set_expected_input_properties(self):
        """Sets the expected input properties of the Process to execute/update successfully.
        This method should update self.expected_input_properties to be a list of dictionaries, with each
        dictionary specifying the expected properties of an input Data instance. A Process that requires
        N inputs will require N such dictionaries specifying its properties. Will be unique to each Process.
        """

        expected_input_properties = [
            {"Data Type": type(np.array([])), "Usage": "Matrix"},
            {"Data Type": int, "Usage": "Column Idx"},
        ]

        for row_idx in range(self.column_idx + 1, self.matrix_size):
            expected_input_properties.append(
                {
                    "Data Type": [type(np.array([])), list],
                    "Usage": "Reduced Row",
                    "Row Idx": row_idx,
                }
            )

        self.expected_input_properties = expected_input_properties

    def set_required_resources(self):
        """Sets the appropriate resources required to run the process. This method should update
        self.required_resources to be a list of Resources required for the Process to successfully run.
        Will be unique for each process.
        """

        self.required_resources = ClassicalResource(
            memory=sys.getsizeof(self.input_data["Matrix"])
            + sys.getsizeof(
                self.input_data["Matrix"][self.column_idx + 1 : self.matrix_size]
            ),
            num_cores=1,
        )

    def set_output_properties(self):
        """Sets the expected properties of the generated outputs of the Process.
        This method should update self.output_properties to be a list of dictionaries,
        with each dictionary specifying the expected properties of each output Data
        """

        self.output_properties = [{"Data Type": type(np.array([])), "Usage": "Matrix"}]

    def compute_flop_count(self):
        """Computes the number of required flops for this process to complete given
        valid inputs from self.inputs. Will be unique to each Process.
        """

        return 10

    def validate_data(self) -> bool:
        """Ensures input data can correctly run for the process. Will
        be unique for each process.
        """

        conditions = (
            self.valid_dimensions(),
            self.valid_column_idx(),
            self.valid_matrix_size(),
        )

        return all(condition for condition in conditions)

    def valid_dimensions(self) -> bool:
        dims = self.input_data["Matrix"].shape
        # return len(dims) == 2 and dims[0] == dims[1] - 1
        return len(dims) == 2

    def valid_column_idx(self) -> bool:
        dims = self.input_data["Matrix"].shape
        return (
            self.input_data["Column Idx"] <= dims[0]
            and self.column_idx == self.input_data["Column Idx"]
        )

    def valid_matrix_size(self) -> bool:
        return self.input_data["Matrix"].shape[0] == self.matrix_size
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
        matrix = self.input_data["Matrix"]

        for row_idx in range(self.column_idx + 1, self.matrix_size):
            matrix[row_idx] = self.input_data["Reduced Row " + str(row_idx)]

        self.result = matrix
        self.compute_required_resources()

    def compute_required_resources(self):
        pass

    def generate_output(self) -> list[Data]:
        """After completing updates, the process will wrap its results
        and return a (unordered) set of Data states to be processed by Node.
        The expected output will be unique to the process.
        """

        return (
            Data(data=self.result, properties={"Usage": "Matrix"}),
            Data(data=self.column_idx + 1, properties={"Usage": "Column Idx"}),
        )

    def extend_network(self) -> Network:
        find_pivot = Node(process_model=GE_FindPivot, network_type=Node.INPUT)
        swap_rows = Node(process_model=GE_SwapRows, network_type=Node.NETWORK)
        row_reduction = Node(
            process_model=GE_RowDeconstruction, network_type=Node.OUTPUT
        )

        find_pivot.insert_output_node(swap_rows)
        swap_rows.insert_output_node(row_reduction)

        network = Network(
            name="Next Reduction Itr",
            nodes=[find_pivot, swap_rows, row_reduction],
            input_nodes=[find_pivot],
            output_nodes=[row_reduction],
            broker=None,
        )

        return network


def generate_GE_network(broker=None):
    find_pivot = Node(process_model=GE_FindPivot, network_type=Node.INPUT)
    swap_rows = Node(process_model=GE_SwapRows, network_type=Node.NETWORK)
    row_reduction = Node(process_model=GE_RowDeconstruction, network_type=Node.OUTPUT)

    find_pivot.insert_output_node(swap_rows)
    swap_rows.insert_output_node(row_reduction)

    network = Network(
        name="Next Reduction Itr",
        nodes=[find_pivot, swap_rows, row_reduction],
        input_nodes=[find_pivot],
        output_nodes=[row_reduction],
        broker=broker,
    )

    return network
