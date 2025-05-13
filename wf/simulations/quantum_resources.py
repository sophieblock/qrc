from typing import List, Dict, Tuple
import networkx as nx

from ..quantum import (
    QuantumCircuit,
    QuantumGate,
    LayoutSynthesizer,
    quantum_circuit_to_qiskit,
    qiskit_to_quantum_circuit,
)
from .resources import Allocation, Device, Resource
from qiskit import QuantumCircuit as QiskitQuantumCircuit
from qiskit import transpile

import qiskit
from ....util.log import logging
logger = logging.getLogger(__name__)

class QuantumAllocation(Allocation):
    def __init__(
        self,
        device_name,
        allocated_qubit_idxs=None,
        transpiled_circuit=None,
        qubit_connectivity=None,
    ):
        super().__init__(device_name=device_name, device_type="QUANTUM")
        self.allocated_qubit_idxs: Tuple[int] = allocated_qubit_idxs
        self.transpiled_circuit: QuantumCircuit = transpiled_circuit
        self.device_connectivity: nx.Graph = qubit_connectivity
    def __repr__(self):
        return f"QAllocation(device_name={self.device_name}, qubits={self.allocated_qubit_idxs})"

class QuantumResource(Resource):
    def __init__(
        self,
        quantum_circuit: QuantumCircuit = None,
        LS_parameters={"transition based": True, "epsilon": 0.3, "objective": "depth"},
    ):
        super().__init__(resource_type="QUANTUM")
        self.circuit = quantum_circuit
        self.LS_parameters = LS_parameters


class QuantumDevice(Device):
    def __init__(
        self,
        device_name,
        connectivity,
        gate_set=None,
    ):
        self.name: str = device_name
        self.connectivity: nx.Graph = connectivity
        self.gate_set: Tuple[QuantumGate] = gate_set
        self.transpiled_swap_circuit: QiskitQuantumCircuit = self.get_transpiled_swap()
        self.swap_duration: int = self.transpiled_swap_circuit.depth()
        self.available_qubits: List[int] = [node for node in self.connectivity.nodes]

        for node in self.connectivity.nodes:
            self.connectivity.nodes[node]["Available"] = True

        self.set_max_connections(self.connectivity)

        if isinstance(connectivity, nx.Graph):
            assert all(
                qubit1 != qubit2 for qubit1, qubit2 in self.connectivity.edges
            ), f"Connectivity graph for device {self.name} contains self edges"

    def get_transpiled_swap(self):
        qc = QiskitQuantumCircuit(2, 0)
        qc.swap(0, 1)
        # qc = transpile(qc, basis_gates=[gate.name.lower() for gate in self.gate_set])
        logger.info(f"qiskit version: {qiskit.__version__}")
        logger.debug(f"Attempting to get {self}'s transpiled swap for gates: {self.gate_set}")
        qc = transpile(
            qc, 
            basis_gates=[gate_name.lower() for gate_name in self.gate_set],
            
        )

        return qc

    def check_if_available(self, resource: QuantumResource):
        required_connections = resource.circuit.qubit_count
        if self.max_available_connections >= required_connections:
            return True
        return False
    # def allocate(self, resource: QuantumResource):
    #     quantum_circuit = resource.circuit
    #     transition_based = resource.LS_parameters["transition based"]
    #     epsilon = resource.LS_parameters["epsilon"]
    #     objective = resource.LS_parameters["objective"]

    #     # Validate gate support
    #     missing_gates = [g for g in quantum_circuit.gate_set if g not in self.gate_set]
    #     if missing_gates:
    #         msg = f"Device {self.name} does not support one or more gates: {missing_gates}. Supported: {self.gate_set}"
    #         logger.error(msg)
    #         raise AssertionError(msg)

    #     # Check device capacity
    #     if not self.check_if_available(resource):
    #         msg = f"Device {self.name} does not have sufficient available qubits to run {quantum_circuit}"
    #         logger.error(msg)
    #         raise AssertionError(msg)

    #     logger.info("Running layout synthesis on device=%s, objective=%s", self.name, objective)
    #     optimized_circuit, qubit_mapping, _, _ = self.layout_synthesis(
    #         quantum_circuit, transition_based, epsilon, objective
    #     )

    #     # Mark mapped qubits as unavailable
    #     for qubit_idx in qubit_mapping:
    #         self.connectivity.nodes[qubit_idx]["Available"] = False
    #     self.update_available_qubits()

    #     allocation = QuantumAllocation(
    #         device_name=self.name,
    #         allocated_qubit_idxs=qubit_mapping,
    #         transpiled_circuit=optimized_circuit,
    #         qubit_connectivity=self.connectivity,
    #     )
    #     logger.debug("Allocation completed: %s", allocation)
    #     return allocation
    def allocate(self, resource: QuantumResource):
        """Transpile, map, and allocate a quantum circuit on this device.

        Returns
        -------
        QuantumAllocation
            A record containing *every* physical qubit touched by the compiled
            circuit (not just the initial mapping) plus the compiled circuit
            itself.
        """
        # 1) Transpile the logical circuit to this device’s basis
        quantum_circuit = self.__transpile(resource.circuit)

        # 2) Find an optimal layout / (re)ordering
        transition_based = resource.LS_parameters["transition based"]
        epsilon          = resource.LS_parameters["epsilon"]
        objective        = resource.LS_parameters["objective"]

        assert all(
            gate in self.gate_set for gate in quantum_circuit.gate_set
        ), (
            f"Device {self.name} does not support one or more gates in "
            f"{quantum_circuit}"
        )

        logger.debug(
            "Starting layout synthesis (objective=%s, transition_based=%s, epsilon=%s)",
            objective, transition_based, epsilon
        )
        optimized_circuit, qubit_mapping, _, _ = self.layout_synthesis(
            quantum_circuit,
            transition_based=transition_based,
            epsilon=epsilon,
            objective=objective,
        )

        # 3) Figure out *which* physical qubits the compiled circuit touches
        touched_qubits = {
            idx
            for instr in optimized_circuit.instructions
            for idx in instr.gate_indices
        }
        sorted_touched = tuple(sorted(touched_qubits))

        logger.debug("Touched qubits by optimized circuit: %s", sorted_touched)
        logger.debug("Available qubits before allocation: %s", self.available_qubits)

        # 4) Mark them unavailable on the device
        logger.debug("Marking qubits %s as unavailable on device '%s'", sorted_touched, self.name)
        for q in sorted_touched:
            self.connectivity.nodes[q]["Available"] = False
        self.update_available_qubits()

        logger.debug("Available qubits after allocation: %s", self.available_qubits)

        allocation = QuantumAllocation(
            device_name=self.name,
            allocated_qubit_idxs=qubit_mapping,
            transpiled_circuit=optimized_circuit,
            qubit_connectivity=self.connectivity,
        )
        logger.debug("Allocation completed: %s", allocation)

        return allocation

    def __transpile(self, quantum_circuit: QuantumCircuit) -> QuantumCircuit:
        circuit = quantum_circuit_to_qiskit(quantum_circuit)
        circuit = transpile(
            # circuit, basis_gates=[gate.name.lower() for gate in self.gate_set]
            circuit,
            basis_gates=[gate_name.lower() for gate_name in self.gate_set],
        )
        circuit = qiskit_to_quantum_circuit(circuit)
        return circuit

    def deallocate(self, allocation: QuantumAllocation):
        assert (
            allocation.device_name == self.name
        ), f"Allocated device name {allocation.device_name} does not match {self.name}"
        assert (
            allocation.device_type == "QUANTUM"
        ), f"Allocated device type {allocation.device_type} does not match QUANTUM"

        allocated_qubit_idxs: List[int] = allocation.allocated_qubit_idxs
        for qubit_idx in allocated_qubit_idxs:
            self.connectivity.nodes[qubit_idx]["Available"] = True
        self.update_available_qubits()

    def update_available_qubits(self):
        self.available_qubits = [
            node
            for node in self.connectivity.nodes
            if self.connectivity.nodes[node]["Available"]
        ]
        self.set_max_connections()

    def set_max_connections(self, available_connectivity=None) -> int:
        """Finds the largest number of connected available qubits"""
        if available_connectivity == None:
            connectivity = self.get_available_connectivity()
        else:
            connectivity = available_connectivity

        max_connections = 0
        for component in nx.connected_components(connectivity):
            if len(component) > max_connections:
                max_connections = len(component)
        self.max_available_connections = max_connections

    def get_available_connectivity(self):
        available_connectivity = nx.Graph()
        self.add_available_nodes(available_connectivity)
        self.add_available_edges(available_connectivity)

        return available_connectivity

    def add_available_nodes(self, available_connectivity: nx.Graph):
        for node in self.connectivity.nodes:
            if self.connectivity.nodes[node]["Available"]:
                available_connectivity.add_node(node)

    def add_available_edges(self, available_connectivity: nx.Graph):
        for edge in self.connectivity.edges:
            if (
                self.connectivity.nodes[edge[0]]["Available"]
                and self.connectivity.nodes[edge[1]]["Available"]
            ):
                available_connectivity.add_edge(edge[0], edge[1])

    def layout_synthesis(
        self,
        quantum_circuit: QuantumCircuit,
        transition_based=True,
        epsilon=0.3,
        objective="depth",
    ):
        assert (
            self.max_available_connections >= quantum_circuit.qubit_count
        ), f"Device {self.name} does not have sufficient available qubits to run {quantum_circuit}"

        layout_synthesizer = LayoutSynthesizer(
            quantum_circuit=quantum_circuit,
            device=self,
            transition_based=transition_based,
            epsilon=epsilon,
            objective=objective,
        )

        (
            optimized_circuit,
            initial_qubit_mapping,
            final_qubit_mapping,
            objective_result,
        ) = layout_synthesizer.find_optimal_layout()
        # 2) If transition-based, override with the actual circuit depth
        # if transition_based and objective == "depth":
        #     true_depth = optimized_circuit.depth()
        #     logger.debug(
        #         "Overriding transition‐based reported depth %d with true depth %d",
        #         objective_result,
        #         true_depth,
        #     )
        #     objective_result = true_depth

        # logger.info(
        #     "Layout synthesis complete on device=%s (transition_based=%s). "
        #     "Objective result=%d",
        #     self.name,
        #     transition_based,
        #     objective_result,
        # )
        return (
            optimized_circuit,
            initial_qubit_mapping,
            final_qubit_mapping,
            objective_result,
        )