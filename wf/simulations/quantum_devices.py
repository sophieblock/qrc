
from __future__ import annotations
from abc import ABC, abstractmethod
from qrew.simulation.refactor.devices.ibm_device_parser import LoadQuantumDeviceData
from qrew.simulation.refactor.resources.quantum_resources import QuantumDevice
from qrew.simulation.refactor.quantum_gates import *
from qrew.simulation.refactor.quantum import quantum_circuit_to_qiskit, qiskit_to_quantum_circuit,QuantumCircuit
from ....util.log import logging
logger = logging.getLogger(__name__)
# from qrew.visualization_tools import (
#     translate_c_to_cirq,

# )

class Compiler(ABC):
    """Minimal façade: translate a logical QuantumCircuit into
    device-native instructions without performing *routing* (that’s
    handled later by LayoutSynthesizer)."""

    @abstractmethod
    def transpile(
        self,
        circuit: QuantumCircuit,
        device: "QuantumDevice",
    ) -> QuantumCircuit:
        ...
      
# ──────────────────────────  Q I S K I T  ──────────────────────────────────
class QiskitCompiler(Compiler):
    """
    Wraps qiskit.transpile() but pins every parameter so the behaviour is
    deterministic and routing-free.
    """

    def __init__(self, *, basis_gates: tuple[str, ...]):
        self.basis_gates = [g.lower() for g in basis_gates]

    def transpile(self, circuit, device: "QuantumDevice"):
        from qiskit import transpile
        from qiskit.transpiler import CouplingMap

        qiskit_circ = quantum_circuit_to_qiskit(circuit)

        # Explicit coupling map (so Sabre can’t decide we’re all-to-all)
        cmap = CouplingMap.from_graph(device.connectivity)

        qc_native = transpile(
            qiskit_circ,
            basis_gates=self.basis_gates,
            optimization_level=0,       # 0 = just basis + layout, no swaps
            coupling_map=cmap,
            layout_method="trivial",    # logical 0→phys 0, etc.
            routing_method=None,        # skip swap insertion
            translation_method="translator",
        )
        return qiskit_to_quantum_circuit(qc_native)

# ──────────────────────────  C I R Q  (RouteCQC) ──────────────────────────
class CirqCompiler(Compiler):
    """
    Uses Cirq’s RouteCQC transformer to produce a circuit whose qubits are
    exactly the physical qubits of `device.connectivity`.
    """

    def __init__(self, *, lookahead_radius: int = 8):
        self.lookahead = lookahead_radius

    def transpile(self, circuit, device: "QuantumDevice"):
        import cirq
        from qrew.visualization_tools import translate_c_to_cirq, translate_cirq_to_c

        cirq_circ = translate_c_to_cirq(circuit)

        router = cirq.transformers.RouteCQC(
            device.connectivity,              # networkx graph → cirq graph
            lookahead_radius=self.lookahead,
        )
        routed_cirq = router(cirq_circ)

        return translate_cirq_to_c(routed_cirq)
IBM_Brisbane = QuantumDevice(
    device_name="IBM Brisbane",
    connectivity=LoadQuantumDeviceData("ibm_brisbane.csv").generate_connectivity(),
   
    gate_set=("ECR", "I", "RZ", "SX", "X"),
)


IBM_Kyiv = QuantumDevice(
    device_name="IBM Kyiv",
    connectivity=LoadQuantumDeviceData("ibm_kyiv.csv").generate_connectivity(),
    # gate_set=(ECR, I, RZ, SX, X),
    gate_set=("ECR", "I", "RZ", "SX", "X"),
)



IBM_Sherbrooke = QuantumDevice(
    device_name="IBM Sherbrooke",
    connectivity=LoadQuantumDeviceData("ibm_sherbrooke.csv").generate_connectivity(),
    gate_set=("ECR", "I", "RZ", "SX", "X"),
)

IBM_Fez = QuantumDevice(
    device_name="IBM Fez",
    connectivity=LoadQuantumDeviceData("ibm_fez.csv").generate_connectivity(),
    gate_set=("CZ", "I", "RZ", "SX", "X"),
)

IBM_Nazca = QuantumDevice(
    device_name="IBM Nazca",
    connectivity=LoadQuantumDeviceData("ibm_nazca.csv").generate_connectivity(),
    gate_set=("ECR", "I", "RZ", "SX", "X"),
)


def view_connectivity(device: QuantumDevice):
    connectivity = device.connectivity
    # print(connectivity[0][1]["CZ duration"])
    for node in connectivity.nodes:
        print(node, connectivity.nodes[node])
    for edge in connectivity.edges:
        print(edge, connectivity[edge[0]][edge[1]])

