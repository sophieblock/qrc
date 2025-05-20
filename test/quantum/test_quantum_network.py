import pytest
from qrew.simulation.refactor.process import QuantumProcess
from qrew.simulation.refactor.quantum import *
from qrew.simulation.refactor.graph import Node,Network
from qrew.simulation.refactor.broker import Broker
from qrew.simulation.refactor.resources.quantum_resources import QuantumResource
from qrew.simulation.refactor.data import Data
from qrew.simulation.refactor.devices.quantum_devices import *
from qrew.simulation.refactor.quantum import *
from qrew.simulation.refactor.quantum_gates import *
from typing import List

# from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit import QuantumCircuit as QiskitCircuit
from qiskit import transpile


def _make_tiny_swap_resource():
    """
    A 2-qubit CX is enough – it forces the allocator to reserve both qubits
    on a 2-node device.
    """
    qc = QuantumCircuit(qubit_count=2)
    qc.add_instruction(QuantumInstruction(X(),  (0,)))
    qc.add_instruction(QuantumInstruction(CX(), (0, 1)))
    return QuantumResource(quantum_circuit=qc)      # default LS-parameters


def _make_singleton_toy_device():
    g = nx.Graph([(0, 1)])
    dev = QuantumDevice("toy-2q-singleton", g, gate_set=("X", "CX"))
    dev._is_singleton = True          # triggers deepcopy inside Broker
    return dev


def test_broker_makes_fresh_copy_and_resets():
    """
    • Broker₁ should mutate **its own** copy of the device.
    • Broker₂ gets a brand-new, fully-reset copy – so it can allocate again.
    • The original singleton object stays pristine throughout.
    """
    device_singleton = _make_singleton_toy_device()
    resource         = _make_tiny_swap_resource()

    # ---- first broker -----------------------------------------------------
    broker1 = Broker(quantum_devices=[device_singleton])
    alloc1  = broker1.request_allocation(resource)
    # inside broker1 the device has no free qubits left
    assert broker1.quantum_devices[0].available_qubits == []

    # original singleton object was *deep-copied* → still fully available
    assert device_singleton.available_qubits == [0, 1]

    # ---- second broker ----------------------------------------------------
    broker2 = Broker(quantum_devices=[device_singleton])
    # starts from a clean slate
    assert broker2.quantum_devices[0].available_qubits == [0, 1]

    alloc2  = broker2.request_allocation(resource)
    # again, broker2’s private copy is now depleted
    assert broker2.quantum_devices[0].available_qubits == []
    # original singleton still untouched
    assert device_singleton.available_qubits == [0, 1]


def test_broker_reset_even_without_singleton_flag():
    """
    If the device is *not* marked as a singleton (no deepcopy), the Broker
    still calls .reset() so every run starts from a clean state.
    """
    # create a normal (non-singleton) device and exhaust it via Broker₁
    g = nx.Graph([(0, 1)])
    dev = QuantumDevice("toy-2q", g, gate_set=("X", "CX"))
    res = _make_tiny_swap_resource()

    broker1 = Broker(quantum_devices=[dev])
    broker1.request_allocation(res)
    assert broker1.quantum_devices[0].available_qubits == []

    # Broker₂ receives *the very same* object but reset() is called → available
    broker2 = Broker(quantum_devices=[dev])
    assert broker2.quantum_devices[0].available_qubits == [0, 1]

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



def generate_quantum_adder_qiskit():
    qc = QiskitCircuit(4, 0)
    qc.x(0)
    qc.x(1)
    qc.h(3)
    qc.cx(2, 3)
    qc.t(0)
    qc.t(1)
    qc.t(2)
    qc.tdg(3)
    qc.cx(0, 1)
    qc.cx(2, 3)
    qc.cx(3, 0)
    qc.cx(1, 2)
    qc.cx(0, 1)
    qc.cx(2, 3)
    qc.tdg(0)
    qc.tdg(1)
    qc.tdg(2)
    qc.t(3)
    qc.cx(0, 1)
    qc.cx(2, 3)
    qc.s(3)
    qc.cx(3, 0)
    qc.h(3)

    return qc


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

    def compute_circuit(self):
        from qrew.simulation.refactor.q_interop.qiskit_interop import qiskit_to_quantum_circuit
        qc = generate_quantum_adder_qiskit()
        qc_basis = transpile(qc, basis_gates=["rz", "cz", "sx", "id", "x"])
        circuit = qiskit_to_quantum_circuit(qc_basis)
        return circuit

    def set_required_resources(self):
        """Sets the appropriate resources required to run the process. This method should update
        self.required_resources to be a list of Resources required for the Process to successfully run.
        Will be unique for each process.
        """

        self.required_resources = QuantumResource(
            quantum_circuit=self.compute_circuit(),
            LS_parameters={"transition based": True, "epsilon": 0.3, "objective": "depth", "hard_island": True},
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
            IBM_Fez,
            IBM_Kyiv,
            IBM_Nazca,
            IBM_Sherbrooke,
        ]
    )
    return broker

def test_ibm_brisbane_device():
    transpiled_swap_qc = IBM_Brisbane.transpiled_swap_circuit

    

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


@pytest.mark.xfail(reason="get_SWAP_results currently drops SWAPs that involve "
                          "qubits outside the initial mapping",
                   strict=True)
def test_swap_results_include_all_swaps_kyiv():
    """
    get_SWAP_results() should report *every* SWAP the solver schedules,
    even those that involve qubits outside the initial mapping.  The current
    implementation filters them out, so the set difference is empty and the
    assertion fails.
    """
    circuit = generate_quantum_adder()
    ls = LayoutSynthesizer(
        quantum_circuit=circuit,
        device=IBM_Kyiv,
        transition_based=True,  # fast to solve, guarantees SWAPs
        objective="depth"
    )
    result_circuit, initial_qubit_map, final_qubit_map, objective_result, results_dict = ls.find_optimal_layout()
    
    print(dir(result_circuit),result_circuit.instructions)
    # --- initial physical qubits (time 0 mapping) --------------------------
    model = ls.solver.model()
    initial_map = {
        model[ls.variables["pi"][q][0]].as_long()
        for q in range(ls.circuit.qubit_count)
    }
    assert initial_qubit_map == initial_map

    # --- qubits that appear in reported SWAPs ------------------------------
    swap_qubits = {
        q for edge, _ in results_dict["SWAPs"] for q in edge
    }

    # At least one SWAP should touch a qubit *not* in the initial mapping
    assert swap_qubits - initial_map, (
        "get_SWAP_results is missing SWAPs that involve qubits outside the "
        "initial mapping – they are being silently dropped."
    )
