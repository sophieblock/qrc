
from ..data_types import *
from ..data import Data, DataSpec
from ..schema import RegisterSpec, Signature, Flow
from ..utilities import compute_resources
from ..resources import Resource,ClassicalResource
from ..process import Process,ClassicalProcess
from ..graph import Node, DirectedEdge,Network
from ....util.log import logging

logger = logging.getLogger(__name__)

from functools import cached_property



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
        accepted_element_types = [float, int]
        accepted_orientations = ["Row Vector", "Column Vector"]
        conditions = (
            self.data["Vector Rows"] > 0,
            self.data["Vector Columns"] > 0,
            self.data["Vector Orientation"] in accepted_orientations,
            self.data["Element Type"] in accepted_element_types,
        )

        return all(condition for condition in conditions)

    def update(self):
       
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
        # if not all(
        #     inp.data.element_type in (int, float)
        #     or isinstance(inp.data.element_type, CType)
        #     for inp in self.inputs
        # ):
        #     return False
        if not all(isinstance(inp.data.element_type, NumericType) for inp in self.inputs):
            return False
        return self.valid_dimensions()

    def valid_dimensions(self) -> bool:
        input_matrices = [input.data for input in self.inputs]
        # return all(input_matrices[i].cols == input_matrices[i 1].rows for i in range(len(input_matrices)-1))
        return (
            input_matrices[0].rows == input_matrices[1].cols
            or input_matrices[1].rows == input_matrices[0].cols
        )

    def update(self):
       
        self.compute_required_resources()
        element_type = (
            int
            if all(input.data.element_type == int for input in self.inputs)
            else float
        )
        if self.inputs[0].data.cols == self.inputs[1].data.rows:
            self.result = MatrixType(
                self.inputs[0].data.rows,
                self.inputs[1].data.cols,
                element_type=element_type,
            )
        elif self.inputs[1].data.cols == self.inputs[0].data.rows:
            self.result = MatrixType(
                self.inputs[1].data.rows,
                self.inputs[0].data.cols,
                element_type=element_type,
            )

    def compute_required_resources(self):
        pass

    def generate_output(self) -> list[Data]:
        """After completing updates, the process will wrap its results
        and return a (unordered) set of Data states to be processed by Node.
        The expected output will be unique to the process.
        """

        return (Data(data=self.result, properties={"Usage": "Matrix"}),)



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
        return (self.inputs[0].data.cols * self.inputs[0].data.rows +
              self.inputs[1].data.cols * self.inputs[1].data.rows
        )

    def validate_data(self) -> bool:
        """
        Ensures:
          1. Each input has a valid 'Vector Orientation' property,
          2. Both inputs share the same orientation,
          3. The dimensions line up for a row- or column-wise concat.
        """
        accepted_orientations = ("Row Vector", "Column Vector")

        # 1) orientation tags must be correct
        if not all(inp.properties.get("Vector Orientation") in accepted_orientations
                for inp in self.inputs):
            return False
 
        # 2) both must share the same orientation
        if not self.same_orientations():
            return False
 
        # 3) dimensions must match up for a concat
        if not self.valid_dimensions():
            return False

        return True

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
       
        self.compute_required_resources()

        element_type = (
            int
            if all(input.data.element_type == int for input in self.inputs)
            else float
        )
        if self.inputs[0].properties["Vector Orientation"] == "Row Vector":
            self.result = MatrixType(
                sum(input.data.rows for input in self.inputs),
                self.inputs[0].data.cols,
                element_type=element_type,
            )
        elif self.inputs[0].properties["Vector Orientation"] == "Column Vector":
            self.result = MatrixType(
                self.inputs[0].data.rows,
                sum(input.data.cols for input in self.inputs),
                element_type=element_type,
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
