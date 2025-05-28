import math
import pytest

from qiskit import QuantumCircuit as QiskitCircuit
import qiskit.circuit.library as qiskit_library

import qrew.simulation.refactor.quantum_gates as g
from qrew.simulation.refactor.quantum import QuantumCircuit, QuantumInstruction
from qrew.simulation.refactor.q_interop.qiskit_interop import (
    qiskit_to_quantum_circuit,
    quantum_circuit_to_qiskit,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Mapping   in-house-gate-class  →  (callable that produces qiskit operation, #qubits)
_GATE_CASES = [
    (g.X,      lambda: qiskit_library.XGate(),          1),
    (g.Y,      lambda: qiskit_library.YGate(),          1),
    (g.Z,      lambda: qiskit_library.ZGate(),          1),
    (g.H,      lambda: qiskit_library.HGate(),          1),
    (g.S,      lambda: qiskit_library.SGate(),          1),
    (g.T,      lambda: qiskit_library.TGate(),          1),
    (g.Tdg,    lambda: qiskit_library.TdgGate(),        1),
    (g.RX,     lambda: qiskit_library.RXGate(math.pi/3),1),
    (g.RY,     lambda: qiskit_library.RYGate(math.pi/7),1),
    (g.RZ,     lambda: qiskit_library.RZGate(math.pi/5),1),
    (g.CX,     lambda: qiskit_library.CXGate(),         2),
    (g.CZ,     lambda: qiskit_library.CZGate(),         2),
    # (g.CZPow,  lambda: qiskit_library.CZGate().power(0.3), 2),
    (g.SWAP,   lambda: qiskit_library.SwapGate(),       2),  
]

def _fingerprint_qiskit(qc: QiskitCircuit):
    """[(name, qidx tuple, param-tuple)]"""
    fp = []
    for inst, qargs, _c in qc.data:
        params = tuple(round(float(p), 8) for p in inst.params)
        idx = lambda q: q.index if hasattr(q, "index") else q._index
        fp.append((inst.name.upper(), tuple(idx(q) for q in qargs), params))
    return fp

def _fingerprint_inhouse(qc: QuantumCircuit):
    fp = []
    for ins in qc.instructions:
        params = ()
        if hasattr(ins.gate, "param"):
            params = (round(float(ins.gate.param), 8),)
        elif hasattr(ins.gate, "params") and ins.gate.params:
            params = tuple(round(float(p), 8) for p in ins.gate.params)
        fp.append((ins.gate.name.upper(), tuple(ins.gate_indices), params))
    return fp
def qiskit_to_quantum_circuit(qc: QiskitCircuit):
    from numpy import angle, pi
    circ = QuantumCircuit(qubit_count=qc.num_qubits, instructions=[])

    for inst, qargs, _c in qc.data:
        name = inst.name.upper()

        # -------------- tidy special-cases -----------------
        if name == "TDG":
            name = "Tdg"

        elif name == "UNITARY" and len(qargs) == 2:          # CZ^α (Qiskit ≥0.45)
            mat   = inst.to_matrix()
            alpha = float(angle(mat[3, 3]) / pi)
            GateCls = g.CZPow
            circ.add_instruction(
                QuantumInstruction(GateCls(alpha),
                                     tuple(q.index if hasattr(q,"index") else q._index
                                           for q in qargs))
            )
            continue                                          # done with this op


        # -------------- generic path -----------------------
        GateCls = getattr(g, name)
        qidx = tuple(q.index if hasattr(q,"index") else q._index for q in qargs)

        if name in ("RX", "RY", "RZ"):
            circ.add_instruction(QuantumInstruction(GateCls(inst.params[0]), qidx))
        elif name == "CZPow":
            circ.add_instruction(QuantumInstruction(GateCls(inst.params[0]), qidx))
        else:
            circ.add_instruction(QuantumInstruction(GateCls(), qidx))

    return circ



def quantum_circuit_to_qiskit(qc: QuantumCircuit):
    circuit = QiskitCircuit(qc.qubit_count, 0)
    for instruction in qc.instructions:
        gate_name = instruction.gate.name

        if gate_name == "CZPow":
            circuit.append(
                qiskit_library.CZGate().power(instruction.gate.param),
                list(instruction.gate_indices),
            )
        elif gate_name == "SWAP":
            circuit.append(
                qiskit_library.SwapGate(), 
                list(instruction.gate_indices),
                )
        else:
            qiskit_gate = getattr(qiskit_library, gate_name + "Gate")
            if hasattr(instruction.gate, "param"):
                circuit.append(
                    qiskit_gate(instruction.gate.param), list(instruction.gate_indices)
                )
            else:
                circuit.append(qiskit_gate(), list(instruction.gate_indices))

    return circuit

# ---------------------------------------------------------------------------
# Parametrised single-gate tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("gate_cls, make_qiskit_gate, n_qubits", _GATE_CASES,
                         ids=[cls.__name__ for cls, _, _ in _GATE_CASES])
def test_single_gate_roundtrip_qiskit_to_inhouse(gate_cls, make_qiskit_gate, n_qubits):
    """Qiskit  →  in-house  →  Qiskit"""
    # make a tiny qiskit circuit on indices (0) or (0,1)
    qc_qiskit = QiskitCircuit(n_qubits)
    qarg = (0,) if n_qubits == 1 else (0, 1)
    qc_qiskit.append(make_qiskit_gate(), qarg)

    # convert forth & back
    qc_inhouse = qiskit_to_quantum_circuit(qc_qiskit)
    qc_back    = quantum_circuit_to_qiskit(qc_inhouse)

    # fingerprints must agree (name & indices & params)
    assert _fingerprint_qiskit(qc_qiskit) == _fingerprint_qiskit(qc_back)
    assert _fingerprint_inhouse(qc_inhouse) == _fingerprint_qiskit(qc_qiskit)


@pytest.mark.parametrize("gate_cls, make_qiskit_gate, n_qubits", _GATE_CASES,
                         ids=[cls.__name__ for cls, _, _ in _GATE_CASES])
def test_single_gate_roundtrip_inhouse_to_qiskit(gate_cls, make_qiskit_gate, n_qubits):
    """in-house  →  Qiskit  →  in-house"""
    qc_in = QuantumCircuit(qubit_count=n_qubits)
    qarg = (0,) if n_qubits == 1 else (0, 1)

    # create in-house instruction
    if gate_cls in (g.RX, g.RY, g.RZ):              # 1-param gates
        theta = make_qiskit_gate().params[0]
        qc_in.add_instruction(QuantumInstruction(gate_cls(theta), qarg))
    elif gate_cls is g.CZPow:                       # pow gate
        exponent = make_qiskit_gate().params[0]
        qc_in.add_instruction(QuantumInstruction(g.CZPow(exponent), qarg))
    else:
        qc_in.add_instruction(QuantumInstruction(gate_cls(), qarg))

    qc_qiskit = quantum_circuit_to_qiskit(qc_in)
    qc_back   = qiskit_to_quantum_circuit(qc_qiskit)

    assert _fingerprint_qiskit(qc_qiskit) == _fingerprint_inhouse(qc_in)
    assert _fingerprint_inhouse(qc_back)  == _fingerprint_inhouse(qc_in)


# ---------------------------------------------------------------------------
#  Multi-gate / ordering test (quick sanity check)
# ---------------------------------------------------------------------------
def test_full_circuit_conversion():
    """Use several gates together to ensure order & indices survive."""
    qc_q = QiskitCircuit(3)
    qc_q.h(0)
    qc_q.cx(0, 1)
    qc_q.swap(1, 2)          # <-- SWAP again
    qc_q.rx(0.314, 2)

    # round-trip
    qc_in  = qiskit_to_quantum_circuit(qc_q)
    qc_out = quantum_circuit_to_qiskit(qc_in)

    assert _fingerprint_qiskit(qc_q) == _fingerprint_qiskit(qc_out)