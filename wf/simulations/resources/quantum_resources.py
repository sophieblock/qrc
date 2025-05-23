from typing import List, Dict, Tuple
import networkx as nx
import cirq
from ..quantum import (
    QuantumCircuit,
    QuantumGate,
    QuantumInstruction,
    LayoutSynthesizer,
    
)
from ..q_interop.transpilers import QuantumCompiler, QiskitCompiler, CirqCompiler
from .resources import Allocation, Device, Resource
from qiskit import QuantumCircuit as QiskitQuantumCircuit

from packaging.version import parse as v
import qiskit

# Qiskit ≥ 0.24 has Target / CouplingMap / InstructionProperties
if v(qiskit.__version__) >= v("0.24.0"):
    from qiskit.transpiler import Target, CouplingMap, InstructionProperties
    from qiskit.circuit.library import standard_gates
    from qiskit import transpile
else:
    from qiskit import transpile
    from qiskit.transpiler import Target, CouplingMap, InstructionProperties
    from qiskit.circuit.library.standard_gates import standard_gates
    

from ....util.log import get_logger,logging
logger = get_logger(__name__)

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
        LS_parameters={"transition based": True, "epsilon": 0.3, "objective": "depth", "hard_island": False},
    ):
        super().__init__(resource_type="QUANTUM")
        self.circuit = quantum_circuit
        self.LS_parameters = LS_parameters
  
from rich.pretty import pretty_repr
class QuantumDevice(Device):
    def __init__(
        self,
        device_name,
        connectivity,
        compiler: "QuantumCompiler" = None,
        gate_set: tuple[str, ...] | None = None,
    ):
        self.name: str = device_name
        self.connectivity: nx.Graph = connectivity
        if gate_set is None:
            if compiler is None:
                raise ValueError("Either gate_set or compiler must be provided.")
            gate_set = tuple(g.upper() for g in compiler.basis_gates)

        self.gate_set = gate_set

        self.compiler = compiler or QiskitCompiler(basis_gates=self.gate_set)

        
      
        
        for node in self.connectivity.nodes:
            self.connectivity.nodes[node]["Available"] = True

        self.available_qubits: List[int] = [node for node in self.connectivity.nodes]
        self.set_max_connections(self.connectivity)
        # ask the compiler for a 2-qubit SWAP in native basis to measure depth
        # swap_proto = self.compiler._transpiled_swap_native
        swap_proto = self.compiler._transpiled_swap
        self.transpiled_swap_circuit = swap_proto
        self.swap_duration: int = self.compiler.swap_duration
        

        if isinstance(connectivity, nx.Graph):
            assert all(
                qubit1 != qubit2 for qubit1, qubit2 in self.connectivity.edges
            ), f"Connectivity graph for device {self.name} contains self edges"
        # self.describe(log=True)
        logger.debug("%s initialised: |V|=%d, |E|=%d, swap_D=%d",
                     self, self.connectivity.number_of_nodes(),
                     self.connectivity.number_of_edges(),
                     self.swap_duration)
    
    def get_coupling_graph(self):
        """
        Return the coupling as a grpah or adjacency matrix. 
        """
        pass

    def is_fully_connected(self) -> bool:
        """
        Convenience method that checks for all-to-all connectivity. Could be used by QuantumCompiler
        to decide whether to even bother with routing. For example, trapped-ion systems are all-to-all and thus it is likely that no swaps are needed, although they could still be useful for reasons other than connectivtiy like moving qubits to mitigate crosstalk.

        """
        n = self.connectivity.number_of_nodes()
        return self.connectivity.number_of_edges() == n * (n - 1) // 2
    
    def swap_decomposition(self, physical_idxs: tuple[int, int]) -> list[QuantumInstruction]:
        return self.compiler.swap_decomposition(physical_idxs)
    
    def reset(self) -> None:
    
        for q in self.connectivity.nodes:
            self.connectivity.nodes[q]["Available"] = True
        self.update_available_qubits()
        
    def update_available_qubits(self):
        self.available_qubits = [
            node
            for node in self.connectivity.nodes
            if self.connectivity.nodes[node]["Available"]
        ]
        self.set_max_connections()

    def set_qubits_as_available(self, indices):
        pass


    def check_if_available(self, resource: QuantumResource):
        required_connections = resource.circuit.qubit_count
        if self.max_available_connections >= required_connections:
            return True
        return False
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

    def __transpile(self, circuit: QuantumCircuit):
        return self.compiler.transpile(circuit, self)

    
    

    
    def layout_synthesis(
        self,
        quantum_circuit: QuantumCircuit,
        transition_based=True,
        hard_island=False,
        epsilon=0.3,
        objective="depth",
    ):
        assert (
            self.max_available_connections >= quantum_circuit.qubit_count
        ), f"Device {self.name} with max connection {self.max_available_connections} does not have sufficient available qubits to run {quantum_circuit}"

        layout_synthesizer = LayoutSynthesizer(
            quantum_circuit=quantum_circuit,
            device=self,
            transition_based=transition_based,
            hard_island=hard_island,
            epsilon=epsilon,
            objective=objective,
        )

        (
            optimized_circuit,
            initial_qubit_mapping,
            final_qubit_mapping,
            objective_result,
            results_dict
        ) = layout_synthesizer.find_optimal_layout()
        
        return (
            optimized_circuit,
            initial_qubit_mapping,
            final_qubit_mapping,
            objective_result,
            results_dict
        )
    def allocate(self, resource: QuantumResource):
        """Transpile, map, and allocate a quantum circuit on this device.

        Returns
        -------
        QuantumAllocation
            A record containing *every* physical qubit touched by the compiled
            circuit (not just the initial mapping) plus the compiled circuit
            itself.
        TODO: Address comment on logical-to-physical: "QuantumDevice.allocate must see the device coupling map but stay routing-free so that `LayoutSynthesizer` controls movement. Using the
        same optimization level here ensures the logical body of the circuit
        is transformed consistently with how SWAP depth was measured."
        """
        # 1) Transpile the logical circuit to this device’s basis
        quantum_circuit = self.__transpile(resource.circuit)
        # logger.debug(f"resource.LS_parameters: {resource.LS_parameters}")
        # 2) Find an optimal layout / (re)ordering
        transition_based = resource.LS_parameters["transition based"]
        hard_island      = resource.LS_parameters.get("hard_island", False)
        epsilon          = resource.LS_parameters["epsilon"]
        objective        = resource.LS_parameters["objective"]
        assert all(
            gate in self.gate_set for gate in quantum_circuit.gate_set
        ), (
            f"Device {self.name} does not support one or more gates in "
            f"{quantum_circuit}"
        )

        logger.debug(
            "Starting layout synthesis (objective=%s, transition_based=%s, epsilon=%s, hard_island=%s)",
            objective, transition_based, epsilon,hard_island
        )
        optimized_circuit, init_qubit_map, _, _,results_dict = self.layout_synthesis(
            quantum_circuit,
            transition_based=transition_based,
            hard_island=hard_island,
            epsilon=epsilon,
            objective=objective,
        )
        # pprint.pp(results_dict,compact=True,width=100)
        logger.debug(f'init mapping logical->physical (t=0): {nice_mapping(init_qubit_map)}')
        logger.info(f"Inputs: " + pretty_repr(results_dict, max_length=10))
        # logger.debug("Available qubits before allocation: %s", self.available_qubits)
        # 3) Figure out *which* physical qubits the compiled circuit touches
        all_qubits = {idx
                      for instr in optimized_circuit.instructions
                      for idx in instr.gate_indices}
        if hard_island:
            # • relay-paths forbidden ⇒ “extra” qubits are a bug
            assert all_qubits.issubset(set(init_qubit_map)), (
                f"Compiled circuit touches extra qubits {all_qubits - set(init_qubit_map)} "
                f"which are outside the initial island {init_qubit_map}"
            )
        else:
            # • relay-paths allowed ⇒ only warn (helps during debugging)
            extra = all_qubits - set(init_qubit_map)
            if extra:
                logger.debug(
                    "Relay qubits introduced outside the initial island: %s", sorted(extra)
                )
        
        
        # ---------------------------------------------------------
        # ❷  Reserve every qubit that actually appears in the circuit
        #     (whether island or relay) so that the global scheduler
        #     will not hand them to another job.
        # ---------------------------------------------------------
        logger.debug("Marking qubits %s as unavailable on device '%s'",
                     sorted(all_qubits), self.name)
        self.inspect_device()
        for q in all_qubits:
            self.connectivity.nodes[q]["Available"] = False
        self.update_available_qubits()

        self.inspect_device()

        # logger.debug("Available qubits after allocation: %s", self.available_qubits)

        allocation = QuantumAllocation(
            device_name=self.name,
            allocated_qubit_idxs=list(all_qubits),
            transpiled_circuit=optimized_circuit,
            qubit_connectivity=self.connectivity,
        )
        logger.debug("Allocation completed: %s", allocation)
        # self.describe(log=True)

        return allocation
    
    def deallocate(self, allocation: QuantumAllocation):
        """
        Release the physical qubits recorded in allocation back to the pool by marking them `Available = True` in the connectivity graph.

        TODO: *Address potential ancilla leakage*
        This is an issue we will have to address for any device that uses qiskit's transpile method. It will silentlly add ancilla qubits to the circuit if a synthesis pass thinks it beneficial. Those ancillas will then appear in the transpiled circuit ***even when they are not marked in allocated_qubit_idxs ***.

        If in a given workflow, our global scheduler could hand the same hysical qubit to another process thinking it is available.

        Proposed long-term fix: Build a CouplingMap limited to the currently-available nodes and hand it to qiskit so that any attempt to allocate outside that set will fail inside their transpiler (might need more plumbing with version 2.0.0)

        Current fix: If hard_island=True and we detect an out-of-island qubit at deallocation time, we throw an error.

        """
        assert (
            allocation.device_name == self.name
        ), f"Allocated device name {allocation.device_name} does not match {self.name}"
        assert (
            allocation.device_type == "QUANTUM"
        ), f"Allocated device type {allocation.device_type} does not match QUANTUM"

        allocated_qubit_idxs: List[int] = allocation.allocated_qubit_idxs
        island = set(allocated_qubit_idxs)          # t = 0 island
        touched = {
            q for instr in allocation.transpiled_circuit.instructions
            for q in instr.gate_indices
        }
        assert touched.issubset(island), (
            "Deallocation detected qubits outside the initial island:\n"
            f"  island : {sorted(island)}\n"
            f"  extra  : {sorted(touched - island)}\n"
            "⇢ Mark test as xfail until coupling-map containment is implemented."
        )

        for qubit_idx in allocated_qubit_idxs:
            self.connectivity.nodes[qubit_idx]["Available"] = True
        self.update_available_qubits()

    # @property
    def describe(self,log=False):
        description = (
            f"Quantum Device: {self.name}\n"
            # f" - Native gates: {self.gate_set}\n"
            f" - Available Qubits: {self.available_qubits}\n"
            f" - connectivity: {self.connectivity}\n"
            # f" - Output Edges: {[str(edge) for edge in self.output_edges]}\n"
            # f" - Allocation: {self.allocation}\n"
        )

        if log:
            logger.debug(description)
    def inspect_device(self):
        inspect_device(self)
    def __repr__(self):
        return f'QDevice({self.name},num qubits={len(self.available_qubits)})'
    
import pprint
def inspect_device(dev: QuantumDevice, *, show_cmap=False) -> None:
    """
    Pretty-print the state of a `QuantumDevice`.

    Parameters
    ----------
    dev : QuantumDevice
        An *instantiated* QuantumDevice (after `generate_quantum_device`).

    show_cmap : bool
        If True, also emit a Qiskit CouplingMap constructed from the
        networkx graph so you can drop it straight into `qiskit.transpile`.
    """
    g = dev.connectivity          # networkx.Graph
    logger.debug(f"\n📟  Device: {dev.name}")
    logger.debug(f"• # physical qubits: {g.number_of_nodes()}")
    # logger.debug(f"• connectivity      : {sorted(g.edges())}")
    logger.debug(f"• max connected sub-graph size (available) : "
          f"{dev.max_available_connections}")

    # show per-node availability
    logger.debug("• availability map:")
    avail = {v: g.nodes[v]["Available"] for v in sorted(g.nodes)}
    pprint.pp(avail, compact=True, width=50)

def nice_mapping(mapping: tuple[int, ...]) -> str:
    """
    Convert `(4, 2, 0, 1)` to 'q0→4  q1→2  q2→0  q3→1'.
    """
    return "  ".join(f"q{q}→{p}" for q, p in enumerate(mapping))