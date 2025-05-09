from workflow.simulation.refactor.graph import Network, Node, DirectedEdge
from workflow.simulation.refactor.process import QuantumProcess
from workflow.simulation.refactor.quantum import (
    QuantumCircuit,
    qiskit_to_quantum_circuit,
)
from workflow.simulation.refactor.broker import Broker
from workflow.simulation.refactor.resources.quantum_resources import QuantumResource
from workflow.simulation.refactor.data import Data
from workflow.simulation.refactor.devices.quantum_devices import *
from workflow.simulation.refactor.quantum import *
from workflow.simulation.refactor.quantum_gates import *
from typing import List
import pytest


def generate_quantum_adder():
    circuit = QuantumCircuit(qubit_count=4)
    circuit.add_instruction(QuantumInstruction(gate=X(), qubit_indices=(0,)))
    circuit.add_instruction(QuantumInstruction(gate=X(), qubit_indices=(1,)))
    circuit.add_instruction(QuantumInstruction(gate=H(), qubit_indices=(3,)))
    circuit.add_instruction(QuantumInstruction(gate=CX(), qubit_indices=(2, 3)))
    circuit.add_instruction(QuantumInstruction(gate=T(), qubit_indices=(0,)))
    circuit.add_instruction(QuantumInstruction(gate=T(), qubit_indices=(1,)))
    circuit.add_instruction(QuantumInstruction(gate=T(), qubit_indices=(2,)))
    circuit.add_instruction(QuantumInstruction(gate=Tdg(), qubit_indices=(3,)))
    circuit.add_instruction(QuantumInstruction(gate=CX(), qubit_indices=(0, 1)))
    circuit.add_instruction(QuantumInstruction(gate=CX(), qubit_indices=(2, 3)))
    circuit.add_instruction(QuantumInstruction(gate=CX(), qubit_indices=(3, 0)))
    circuit.add_instruction(QuantumInstruction(gate=CX(), qubit_indices=(1, 2)))
    circuit.add_instruction(QuantumInstruction(gate=CX(), qubit_indices=(0, 1)))
    circuit.add_instruction(QuantumInstruction(gate=CX(), qubit_indices=(2, 3)))
    circuit.add_instruction(QuantumInstruction(gate=Tdg(), qubit_indices=(0,)))
    circuit.add_instruction(QuantumInstruction(gate=Tdg(), qubit_indices=(1,)))
    circuit.add_instruction(QuantumInstruction(gate=Tdg(), qubit_indices=(2,)))
    circuit.add_instruction(QuantumInstruction(gate=T(), qubit_indices=(3,)))
    circuit.add_instruction(QuantumInstruction(gate=CX(), qubit_indices=(0, 1)))
    circuit.add_instruction(QuantumInstruction(gate=CX(), qubit_indices=(2, 3)))
    circuit.add_instruction(QuantumInstruction(gate=S(), qubit_indices=(3,)))
    circuit.add_instruction(QuantumInstruction(gate=CX(), qubit_indices=(3, 0)))
    circuit.add_instruction(QuantumInstruction(gate=H(), qubit_indices=(3,)))

    return circuit


class QuantumAdder(QuantumProcess):
    def __init__(
        self,
        inputs=None,
        expected_input_properties=None,
        required_resources=None,
        output_properties=None,
    ):
        super().__init__(
            inputs=inputs,
            expected_input_properties=expected_input_properties,
            required_resources=required_resources,
            output_properties=output_properties,
        )

    def compute_circuit(self) -> QuantumCircuit:
        return generate_quantum_adder()

    def set_required_resources(self):
        """Sets the appropriate resources required to run the process. This method should update
        self.required_resources to be a list of Resources required for the Process to successfully run.
        Will be unique for each process.
        """

        self.required_resources = QuantumResource(
            quantum_circuit=self.compute_circuit()
        )

    def set_expected_input_properties(self):
        """Sets the expected input properties of the Process to execute/update successfully.
        This method should update self.expected_input_properties to be a list of dictionaries, with each
        dictionary specifying the expected properties of an input Data instance. A Process that requires
        N inputs will require N such dictionaries specifying its properties. Will be unique to each Process.
        """

        self.expected_input_properties = [{"Data Type": int, "Usage": "Num Qubits"}]

    def set_output_properties(self):
        """Sets the expected properties of the generated outputs of the Process.
        This method should update self.output_properties to be a list of dictionaries,
        with each dictionary specifying the expected properties of each output Data
        """

        return None

    def validate_data(self) -> bool:
        """Process specific verification that ensures input data has
        correct specifications. Will be unique to each process
        """

        return True

    def update(self):
        self.fidelity = self.compute_fidelity()

    def compute_fidelity(self):
        return 1.0

    def generate_output(self) -> List[Data]:
        """After completing updates, the process will wrap its results
        and return a (unordered) set of Data states to be processed by Node.
        The expected output will be unique to the process.
        """

        return [Data(data=self.fidelity, properties={"Usage": "Fidelity"})]


def generate_quantum_broker():
    broker = Broker(
        quantum_devices=[
            IBM_Brisbane,
            IBM_Brussels,
            IBM_Fez,
            IBM_Kyiv,
            IBM_Nazca,
            IBM_Sherbrooke,
        ]
    )
    return broker



def test_run_quantum_network_Brisbane():
    broker = Broker(
        quantum_devices=[
            IBM_Brisbane
        ]
    )

    node = Node(process_model=QuantumAdder, network_type="INPUT")
    network = Network(
        name="Quantum Adder",
        nodes=[node],
        input_nodes=[node],
        output_nodes=[],
        broker=broker,
    )
    df = network.run(
        network.input_nodes,
        starting_inputs=[(Data(data=5, properties={"Usage": "Num Qubits"}),)],
    )

def test_run_quantum_network_Brussels():
    broker = Broker(
        quantum_devices=[
            IBM_Brussels
        ]
    )

    node = Node(process_model=QuantumAdder, network_type="INPUT")
    network = Network(
        name="Quantum Adder",
        nodes=[node],
        input_nodes=[node],
        output_nodes=[],
        broker=broker,
    )
    df = network.run(
        network.input_nodes,
        starting_inputs=[(Data(data=5, properties={"Usage": "Num Qubits"}),)],
    )

def test_run_quantum_network_Fez():
    broker = Broker(
        quantum_devices=[
            IBM_Fez
        ]
    )

    node = Node(process_model=QuantumAdder, network_type="INPUT")
    network = Network(
        name="Quantum Adder",
        nodes=[node],
        input_nodes=[node],
        output_nodes=[],
        broker=broker,
    )
    df = network.run(
        network.input_nodes,
        starting_inputs=[(Data(data=5, properties={"Usage": "Num Qubits"}),)],
    )

def test_run_quantum_network_Kyiv():
    broker = Broker(
        quantum_devices=[
            IBM_Kyiv
        ]
    )

    node = Node(process_model=QuantumAdder, network_type="INPUT")
    network = Network(
        name="Quantum Adder",
        nodes=[node],
        input_nodes=[node],
        output_nodes=[],
        broker=broker,
    )
    df = network.run(
        network.input_nodes,
        starting_inputs=[(Data(data=5, properties={"Usage": "Num Qubits"}),)],
    )

def test_run_quantum_network_Nazca():
    broker = Broker(
        quantum_devices=[
            IBM_Nazca
        ]
    )

    node = Node(process_model=QuantumAdder, network_type="INPUT")
    network = Network(
        name="Quantum Adder",
        nodes=[node],
        input_nodes=[node],
        output_nodes=[],
        broker=broker,
    )
    df = network.run(
        network.input_nodes,
        starting_inputs=[(Data(data=5, properties={"Usage": "Num Qubits"}),)],
    )

def test_run_quantum_network_Sherbrooke():
    broker = Broker(
        quantum_devices=[
            IBM_Sherbrooke
        ]
    )

    node = Node(process_model=QuantumAdder, network_type="INPUT")
    network = Network(
        name="Quantum Adder",
        nodes=[node],
        input_nodes=[node],
        output_nodes=[],
        broker=broker,
    )
    df = network.run(
        network.input_nodes,
        starting_inputs=[(Data(data=5, properties={"Usage": "Num Qubits"}),)],
    )