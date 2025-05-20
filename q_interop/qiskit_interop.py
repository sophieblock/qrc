import qiskit
import qrew.simulation.refactor.quantum_gates as quantum_gateset
from ..quantum_gates import *
from ..quantum import QuantumCircuit,QuantumInstruction
from qiskit import QuantumCircuit as QiskitCircuit
import qiskit.circuit.library as qiskit_library
def get_qiskit_gates():
    """Map gate name of the Tangelo format to the equivalent add_gate method of
    Qiskit's QuantumCircuit class API and supported gates:
    https://qiskit.org/documentation/stubs/qiskit.circuit.QuantumCircuit.html
    """

    import qiskit

    GATE_QISKIT = dict()
    GATE_QISKIT["H"] = qiskit.QuantumCircuit.h
    GATE_QISKIT["X"] = qiskit.QuantumCircuit.x
    GATE_QISKIT["Y"] = qiskit.QuantumCircuit.y
    GATE_QISKIT["Z"] = qiskit.QuantumCircuit.z
    GATE_QISKIT["CH"] = qiskit.QuantumCircuit.ch
    GATE_QISKIT["CX"] = qiskit.QuantumCircuit.cx
    GATE_QISKIT["CY"] = qiskit.QuantumCircuit.cy
    GATE_QISKIT["CZ"] = qiskit.QuantumCircuit.cz
    GATE_QISKIT["S"] = qiskit.QuantumCircuit.s
    GATE_QISKIT["T"] = qiskit.QuantumCircuit.t
    GATE_QISKIT["RX"] = qiskit.QuantumCircuit.rx
    GATE_QISKIT["RY"] = qiskit.QuantumCircuit.ry
    GATE_QISKIT["RZ"] = qiskit.QuantumCircuit.rz
    GATE_QISKIT["CRX"] = qiskit.QuantumCircuit.crx
    GATE_QISKIT["CRY"] = qiskit.QuantumCircuit.cry
    GATE_QISKIT["CRZ"] = qiskit.QuantumCircuit.crz
    GATE_QISKIT["MCRX"] = qiskit.QuantumCircuit.mcrx
    GATE_QISKIT["MCRY"] = qiskit.QuantumCircuit.mcry
    GATE_QISKIT["MCRZ"] = qiskit.QuantumCircuit.mcrz
    GATE_QISKIT["MCPHASE"] = qiskit.QuantumCircuit.mcp
    GATE_QISKIT["CNOT"] = qiskit.QuantumCircuit.cx
    GATE_QISKIT["SWAP"] = qiskit.QuantumCircuit.swap
    GATE_QISKIT["XX"] = qiskit.QuantumCircuit.rxx
    GATE_QISKIT["CSWAP"] = qiskit.QuantumCircuit.cswap
    GATE_QISKIT["PHASE"] = qiskit.QuantumCircuit.p
    GATE_QISKIT["CPHASE"] = qiskit.QuantumCircuit.cp
    GATE_QISKIT["MEASURE"] = qiskit.QuantumCircuit.measure
    return GATE_QISKIT

def qiskit_to_quantum_circuit(qc: QiskitCircuit):
    from numpy import angle, pi
    circ = QuantumCircuit(qubit_count=qc.num_qubits, instructions=[])

    for inst, qargs, _c in qc.data:
        name = inst.name.upper()

        # -------------- tidy special-cases -----------------
        if name == "TDG":
            name = "Tdg"


        # -------------- generic path -----------------------
        GateCls = getattr(quantum_gateset, name)
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