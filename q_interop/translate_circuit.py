from qrew.simulation.refactor.q_interop.qiskit_interop import quantum_circuit_to_qiskit, qiskit_to_quantum_circuit
from qrew.simulation.refactor.q_interop.cirq_interop import translate_qc_to_cirq


FROM_TANGELO = {
    "braket": translate_c_to_braket,
    "cirq": translate_c_to_cirq,
    "ionq": translate_c_to_json_ionq,
    "openqasm": translate_c_to_openqasm,
    "projectq": translate_c_to_projectq,
    "qdk": translate_c_to_qsharp,
    "qiskit": translate_c_to_qiskit,
    "qulacs": translate_c_to_qulacs,
    "pennylane": translate_c_to_pennylane,
    "stim": translate_c_to_stim,
    "sympy": translate_c_to_sympy
}

TO_TANGELO = {
    "braket": translate_c_from_braket,
    "ionq": translate_c_from_json_ionq,
    "openqasm": translate_c_from_openqasm,
    "projectq": translate_c_from_projectq,
    "qiskit": translate_c_from_qiskit
}


def translate_circuit(circuit, target, source="tangelo", output_options=None):
    """Function to convert a quantum circuit defined within the "source" format
    to a "target" format.

    Args:
        circuit (source format): Self-explanatory.
        target (string): Identifier for the target format.
        source (string): Identifier for the source format.
        output_options (dict): Backend specific options (e.g. a noise model,
            number of qubits, etc.).

    Returns:
        (circuit in target format): Translated quantum circuit.
    """

    source = source.lower()
    target = target.lower()

    if output_options is None:
        output_options = dict()

    if source == target:
        return circuit

    # Convert to Tangelo format if necessary.
    if source != "tangelo":
        if source not in TO_TANGELO:
            raise NotImplementedError(f"Circuit conversion from {source} to {target} is not supported.")
        circuit = TO_TANGELO[source](circuit)

    # Convert to another target format if necessary.
    if target != "tangelo":
        if target not in FROM_TANGELO:
            raise NotImplementedError(f"Circuit conversion from {source} to {target} is not supported.")
        circuit = FROM_TANGELO[target](circuit, **output_options)

    return circuit
