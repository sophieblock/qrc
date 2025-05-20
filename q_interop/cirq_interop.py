import cirq
from ..quantum_gates import *
from ..quantum import QuantumCircuit
_BASE_GATES = {
  
    "H":   cirq.H,
    "X":   cirq.X,
    "Y":   cirq.Y,
    "Z":   cirq.Z,
    "S":   cirq.S,
    "T":   cirq.T,
    "Tdg": cirq.T**-1,        
    "RX":  cirq.rx,           
    "RY":  cirq.ry,
    "RZ":  cirq.rz,
  
    "CX":  cirq.CNOT,
    "CZ":  cirq.CZ,
    "SWAP": cirq.SWAP,
    "CZPow": cirq.CZPowGate,   
}


def _cirq_gate_from_instruction(instr):
    """
    Return the Cirq gate/operation corresponding to *instr* (QuantumInstruction).

    For 1-qubit parameterised gates we call the factory with the angle.
    For CZPow we instantiate with exponent=instr.gate.param.
    """
    name = instr.gate.name

    if name in {"RX", "RY", "RZ"}:
        theta = getattr(instr.gate, "param")
        return _BASE_GATES[name](theta)

    if name == "CZPow":
        exponent = getattr(instr.gate, "param")
        return _BASE_GATES[name](exponent=exponent)

    if name not in _BASE_GATES:
        raise ValueError(f"Gate '{name}' not (yet) supported by translate_qc_to_cirq")

    return _BASE_GATES[name]


# ────────────────────────────────────────────────────────────────────────────
# 2.  Main translation routine
# ────────────────────────────────────────────────────────────────────────────
def translate_qc_to_cirq(
    qc,
    *,
    noise_model=None,
    save_measurements=False,
    include_idle_identity=True,
):

    n_qubits = qc.qubit_count
    qubits   = cirq.LineQubit.range(n_qubits)

    ops = []

    if include_idle_identity and n_qubits:
        ops.append(cirq.Moment(cirq.I.on_each(*qubits)))

    meas_count = 0

    for instr in qc.instructions:
        gate_name = instr.gate.name

        # 0.  Resolve the Cirq gate or factory
        if gate_name == "MEASURE":
            key = str(meas_count) if save_measurements else None
            ops.append(cirq.measure(qubits[instr.gate_indices[0]], key=key))
            meas_count += 1
            continue

        cirq_gate = _cirq_gate_from_instruction(instr)

        # 1-qubit vs 2-qubit dispatch
        if len(instr.gate_indices) == 1:
            q0 = qubits[instr.gate_indices[0]]
            ops.append(cirq_gate(q0) if callable(cirq_gate) else cirq_gate.on(q0))

        elif len(instr.gate_indices) == 2:
            q0, q1 = (qubits[i] for i in instr.gate_indices)
            # Cirq convention: for CX the first is *control*, second *target*
            if callable(cirq_gate):
                ops.append(cirq_gate(q0, q1))
            else:
                ops.append(cirq_gate.on(q0, q1))
        else:
            raise NotImplementedError(
                f"Gates on >2 qubits not supported (instruction={instr})"
            )

        # (Optional) insert noise here in future — parity with Tangelo stub.
        if noise_model and gate_name in noise_model.noisy_gates:
            raise NotImplementedError("Noise model insertion not implemented yet")

    return cirq.Circuit(ops)


def quantum_circuit_to_cirq(qc: QuantumCircuit):
    """
    TODO: check if this even works (its old)
    Converts a custom QuantumCircuit into a Cirq circuit.

    Args:
        qc (QuantumCircuit): The custom QuantumCircuit object to convert.

    Returns:
        cirq.Circuit: The equivalent Cirq circuit.
    """
    # Define Cirq qubits
    qubits = [cirq.LineQubit(i) for i in range(qc.qubit_count)]
    cirq_circuit = cirq.Circuit()

    # Map gates from the custom gateset to Cirq
    for instruction in qc.instructions:
        gate_name = instruction.gate.name
        qubit_indices = instruction.gate_indices

        # Single-qubit gates
        if gate_name == "X":
            cirq_circuit.append(X(qubits[qubit_indices[0]]))
        elif gate_name == "Y":
            cirq_circuit.append(Y(qubits[qubit_indices[0]]))
        elif gate_name == "Z":
            cirq_circuit.append(Z(qubits[qubit_indices[0]]))
        elif gate_name == "H":
            cirq_circuit.append(H(qubits[qubit_indices[0]]))
        elif gate_name == "T":
            cirq_circuit.append(T(qubits[qubit_indices[0]]))
        elif gate_name == "S":
            cirq_circuit.append(S(qubits[qubit_indices[0]]))

        # Parametrized single-qubit rotations
        elif gate_name == "RX":
            cirq_circuit.append(rx(instruction.gate.param)(qubits[qubit_indices[0]]))
        elif gate_name == "RY":
            cirq_circuit.append(ry(instruction.gate.param)(qubits[qubit_indices[0]]))
        elif gate_name == "RZ":
            cirq_circuit.append(rz(instruction.gate.param)(qubits[qubit_indices[0]]))

        # Two-qubit gates
        elif gate_name == "CNOT":
            cirq_circuit.append(CNOT(qubits[qubit_indices[0]], qubits[qubit_indices[1]]))
        elif gate_name == "CZ":
            cirq_circuit.append(CZ(qubits[qubit_indices[0]], qubits[qubit_indices[1]]))
        elif gate_name == "SWAP":
            cirq_circuit.append(SWAP(qubits[qubit_indices[0]], qubits[qubit_indices[1]]))

        # Parametrized CZPow gate
        elif gate_name == "CZPow":
            cirq_circuit.append(
                cirq.CZPowGate(exponent=instruction.gate.param)(
                    qubits[qubit_indices[0]], qubits[qubit_indices[1]]
                )
            )

        # Measurement gate
        elif gate_name == "MEASURE":
            cirq_circuit.append(measure(qubits[qubit_indices[0]], key=f"q{qubit_indices[0]}"))

        else:
            raise ValueError(f"Gate {gate_name} is not supported in Cirq conversion.")

    return cirq_circuit
