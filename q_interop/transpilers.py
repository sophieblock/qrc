from __future__ import annotations
from typing import List, Tuple,Sequence
from abc import ABC, abstractmethod

from ..quantum import QuantumCircuit, QuantumInstruction, QuantumGate
from ..quantum_gates import SWAP
from .qiskit_interop import quantum_circuit_to_qiskit,qiskit_to_quantum_circuit
from .cirq_interop import translate_qc_to_cirq
# from ..resources import QuantumDevice

class QuantumCompiler(ABC):
    """Translate a logical QuantumCircuit into
    device-native instructions without performing *routing* (that’s
    handled later by LayoutSynthesizer)."""
    swap_duration: int = 1
    @abstractmethod
    def transpile(
        self,
        circuit: QuantumCircuit,
        device: "QuantumDevice",
    ) -> QuantumCircuit:
        ...
    @property
    @abstractmethod
    def swap_duration(self) -> int:          # logical timesteps
        ...
    @abstractmethod
    def swap_decomposition(
        self, physical_idxs: Tuple[int, int]
    ) -> List[QuantumInstruction]:
        """Return the native-gate replacement of a SWAP on *physical_idxs*."""
        ...
   
# ──────────────────────────  Q I S K I T  ──────────────────────────────────
from qiskit import QuantumCircuit as QiskitCircuit
from qiskit import transpile as qiskit_transpile
from qiskit.transpiler import CouplingMap
def _to_cmap(graph: nx.Graph | None) -> CouplingMap | None:
    """networkx → Qiskit CouplingMap (or ``None`` if no hardware given)."""
    if graph is None:
        return None
    return CouplingMap(couplinglist=list(graph.edges))


def _is_single_swap(qc: QuantumCircuit) -> bool:
    """True if `qc` is exactly one SWAP on two qubits (calibration probe)."""
    return (
        len(qc.instructions) == 1
        and isinstance(qc.instructions[0].gate, SWAP)
        and len(qc.instructions[0].gate_indices) == 2
    )
class QiskitCompiler(QuantumCompiler):
    """
    Deterministic, routing-free wrapper around :func:`qiskit.transpile`.

    * ``basis_gates`` are **immutable** after construction.
    * ``swap_duration`` **learns** the depth of one SWAP in that basis.
      First a *guess* is computed during ``__init__`` (so the value is
      never ``None``).  When the compiler later sees a real-world SWAP on a
      concrete device, it **updates** the value if the measured depth is
      larger — exactly what `test_depth_matches_property` expects.
    """

    def __init__(
        self,
        *,
        basis_gates: Sequence[str],
        optimization_level: int = 0,
        routing_method: str | None = None,
        layout_method: str = "trivial",
        translation_method: str = 'translator',
        **extra,
    ):
        self.basis_gates = tuple(basis_gates)
        self.optimization_level = optimization_level
        self.routing_method = routing_method
        self.layout_method = layout_method
        self.translation_method = translation_method
        self._extra = extra # for future 

        # # for IBM-heavy devices a SWAP is 3 CXs ⇒ takes one logical step
        # self.swap_duration = 1
        self._swap_duration: int | None = None
        self._transpiled_swap: QuantumCircuit | None = None

        # first *rough* calibration (no coupling map → depth is a lower bound)
        self._calibrate_swap(coupling_map=None)
        # self._set_transpiled_swap()

    def transpile(
        self, circuit: QuantumCircuit, device: "QuantumDevice"
    ) -> QuantumCircuit:
        # 0) convert *to* Qiskit
        qiskit_in = quantum_circuit_to_qiskit(circuit)

        # 1) run deterministic, routing-free transpile
        qiskit_out = self._qiskit_transpile(
            qiskit_in,
            coupling_map=_to_cmap(device.connectivity),
        )

        # 2) convert back to our in-house IR
        native = qiskit_to_quantum_circuit(qiskit_out)

        # 3) 1st-time-on-real-device swap-depth upgrade
        if _is_single_swap(circuit) and native.depth() > self._swap_duration:
            self._calibrate_swap(
                coupling_map=_to_cmap(device.connectivity)
            )

        return native
    
    def _calibrate_swap(self, *, coupling_map: CouplingMap | None):
        """Run Qiskit on a 2-qubit SWAP and cache the result."""
        qc = QiskitCircuit(2)
        qc.swap(0, 1)

        qiskit_qc = self._qiskit_transpile(qc, coupling_map=coupling_map)
        self._transpiled_swap  = qiskit_to_quantum_circuit(qiskit_qc) # converted transpiled swap
        self._transpiled_swap_native = qiskit_qc # transpiled swap as qiskit QuantumCircuit
        self._swap_duration = qiskit_qc.depth()


    @property
    def swap_duration(self) -> int:
        assert self._swap_duration is not None, f'{self} swap duration not allowed: {self._swap_duration}'
        return self._swap_duration
    @property
    def transpiled_swap_circuit(self) -> QuantumCircuit:
        assert self._transpiled_swap is not None
        return self._transpiled_swap
    def swap_decomposition(
        self, physical_idxs: Tuple[int, int]
    ) -> List[QuantumInstruction]:
        """Return *our* swap translated to the requested indices."""
        if self._transpiled_swap is None:          # paranoia
            self._calibrate_swap(coupling_map=None)

        return [
            QuantumInstruction(
                gate=inst.gate,
                qubit_indices=tuple(physical_idxs[i] for i in inst.gate_indices),
            )
            for inst in self._transpiled_swap.instructions
        ]
    
    def _qiskit_transpile(
        self, circ: QiskitCircuit, *, coupling_map: CouplingMap | None
    ) -> QiskitCircuit:
        """All transpile calls funneled through one place → identical opts."""

        
        return qiskit_transpile(
            circ,
            basis_gates=[g.lower() for g in self.basis_gates],
            optimization_level=self.optimization_level,
            coupling_map=coupling_map,
            layout_method=self.layout_method,
            routing_method=self.routing_method,
            translation_method=self.translation_method,
            **self._extra,
        )
import cirq
# ──────────────────────────  C I R Q  (RouteCQC) ──────────────────────────
class CirqCompiler(QuantumCompiler):
    """Thin wrapper around Cirq’s *RouteCQC* device mapper."""

    def __init__(self, *, lookahead_radius: int = 8):
        self.lookahead_radius = lookahead_radius
        # one-time SWAP calibration (Cirq decomposes to three CZPow gates)
        self._calibrate_swap()

    # public API -------------------------------------------------------------
    def transpile(
        self, circuit: QuantumCircuit, device: "QuantumDevice"
    ) -> QuantumCircuit:
        cirq_in = translate_qc_to_cirq(circuit)

        if device is not None:
            router = cirq.transformers.RouteCQC(
                device.connectivity, lookahead_radius=self.lookahead_radius
            )
            cirq_out = router(cirq_in)
        else:                       # no routing context (e.g. SWAP calibration)
            cirq_out = cirq_in

        return cirq_to_quantum_circuit(cirq_out)

    # SWAP bookkeeping -------------------------------------------------------
    def _calibrate_swap(self):
        qc = QuantumCircuit(2)
        qc.add_instruction(QuantumInstruction(SWAP(), (0, 1)))
        native = self.transpile(qc, device=None)

        self._transpiled_swap = native
        self._swap_duration = native.depth()

    @property
    def swap_duration(self) -> int:
        return self._swap_duration

    def swap_decomposition(
        self, physical_idxs: Tuple[int, int]
    ) -> List[QuantumInstruction]:
        rewired: List[QuantumInstruction] = []
        for instr in self._transpiled_swap.instructions:
            new_qargs = tuple(physical_idxs[i] for i in instr.gate_indices)
            rewired.append(QuantumInstruction(instr.gate, new_qargs))
        return rewired


# class CirqCompiler(QuantumCompiler):
#     """
#     Uses Cirq’s RouteCQC transformer to produce a circuit whose qubits are
#     exactly the physical qubits of `device.connectivity`.
#     """

#     def __init__(self, *, lookahead_radius: int = 8):
#         self.lookahead = lookahead_radius
#         self._set_transpiled_swap()
#     def transpile(self, circuit, device: "QuantumDevice"):
        
#         cirq_circ = translate_qc_to_cirq(circuit)

#         router = cirq.transformers.RouteCQC(
#             device.connectivity,              # networkx graph → cirq graph
#             lookahead_radius=self.lookahead,
#         )
#         routed_cirq = router(cirq_circ)
#         if device is None:            # standalone translation (e.g. swap-probe)
#             routed_cirq = cirq_circ
#         else:
#             router = cirq.transformers.RouteCQC(
#                 device.connectivity, lookahead_radius=self.lookahead
#             )
#             routed_cirq = router(cirq_circ)


#         return cirq_to_builtin_qc(routed_cirq)
#     # -------------  public API  -------------------------------------------------

#     # def transpile(self, circuit: QuantumCircuit) -> QuantumCircuit:
#     #     cirq_circ = translate_qc_to_cirq(circuit)

#     #     # naïve routing / optimisation
#     #     for _ in range(self.optimisation_passes):
#     #         cirq.ExpandComposite().optimize_circuit(cirq_circ)
#     #         cirq.MergeSingleQubitGates().optimize_circuit(cirq_circ)

#     #     return cirq_to_quantum_circuit(cirq_circ)
#     def _set_transpiled_swap(self):
#         qc = QuantumCircuit(2)
#         qc.add_instruction(QuantumInstruction(SWAP(), (0, 1)))
#         cirq_qc = translate_qc_to_cirq(qc)
#         native = self.transpile(cirq_qc, device=None)   # device not needed for 2-q test
#         self._transpiled_swap = native
#         self._swap_duration = native.depth()
#     @property
#     def swap_duration(self) -> int:
        
#         return self._swap_duration
    
#     def swap_decomposition(
#         self, physical_idxs: Tuple[int, int]
#     ) -> List[QuantumInstruction]:
#         q0, q1 = cirq.LineQubit.range(2)
#         tmp = cirq.Circuit(cirq.SWAP(q0, q1))
#         for _ in range(self.optimisation_passes):
#             cirq.ExpandComposite().optimize_circuit(tmp)

#         logical = cirq_to_builtin_qc(tmp)

#         rewired: List[QuantumInstruction] = []
#         for instr in logical.instructions:
#             new_qargs = tuple(physical_idxs[i] for i in instr.gate_indices)
#             rewired.append(QuantumInstruction(instr.gate, new_qargs))
#         return rewired
