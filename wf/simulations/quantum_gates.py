
import numpy as np
from typing import Callable

class QuantumGate:
    def __init__(self, name, num_qubits):
        self.name: str = name
        self.num_qubits: int = num_qubits


class X(QuantumGate):
    def __init__(self):
        self.gate = np.array([[0.0, 1.0], [1.0, 0.0]])
        super().__init__(name="X", num_qubits=1)


class Y(QuantumGate):
    def __init__(self):
        self.gate = np.array([[0.0, -1.0j], [1.0j, 0.0]])
        super().__init__(name="Y", num_qubits=1)


class Z(QuantumGate):
    def __init__(self):
        self.gate = np.array([[1.0, 0.0], [0.0, -1.0]])
        super().__init__(name="Z", num_qubits=1)


class H(QuantumGate):
    def __init__(self):
        self.gate = 1 / np.sqrt(2) * np.array([[1.0, 1.0], [1.0, -1.0]])
        super().__init__(name="H", num_qubits=1)


class T(QuantumGate):
    def __init__(self):
        self.gate = np.array([[1.0, 0.0], [0.0, np.exp(1j * np.pi / 4)]])
        super().__init__(name="T", num_qubits=1)


class Tdg(QuantumGate):
    def __init__(self):
        self.gate = T().gate.conj().T
        super().__init__(name="Tdg", num_qubits=1)


class S(QuantumGate):
    def __init__(self):
        self.gate = np.array([[1.0, 0.0], [0.0, 1.0j]])
        super().__init__(name="S", num_qubits=1)


class I(QuantumGate):
    def __init__(self):
        self.gate = np.eye(2, dtype=float)
        super().__init__(name="I", num_qubits=1)


class RX(QuantumGate):
    def __init__(self, theta: float):
        self.param = theta
        self.gate = np.exp(-1j * theta / 2 * X().gate)
        super().__init__(name="RX", num_qubits=1)


class RY(QuantumGate):
    def __init__(self, theta: float):
        self.param = theta
        self.gate = np.exp(-1j * theta / 2 * Y().gate)
        super().__init__(name="RY", num_qubits=1)


class RZ(QuantumGate):
    def __init__(self, theta: float):
        self.param = theta
        self.gate = np.exp(-1j * theta / 2 * Z().gate)
        super().__init__(name="RZ", num_qubits=1)


class SX(QuantumGate):
    def __init__(self):
        self.gate = (
            1 / 2 * np.array([[1.0 + 1.0j, 1.0 - 1.0j], [1.0 - 1.0j, 1.0 + 1.0j]])
        )
        super().__init__(name="SX", num_qubits=1)


class RESET(QuantumGate):
    def __init__(self):
        self.gate = None
        super().__init__(name="RESET", num_qubits=1)


class MEASURE(QuantumGate):
    def __init__(self):
        self.gate = None
        super().__init__(name="MEASURE", num_qubits=1)


class SWAP(QuantumGate):
    def __init__(self):
        self.gate = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        super().__init__(name="SWAP", num_qubits=2)


class CX(QuantumGate):
    def __init__(self):
        self.gate = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )
        super().__init__(name="CX", num_qubits=2)


class CZ(QuantumGate):
    def __init__(self):
        self.gate = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, -1.0],
            ]
        )
        super().__init__(name="CZ", num_qubits=2)


class CZPow(QuantumGate):
    """
    A CZ gate with arbitrary exponent α  (diag = [1,1,1,e^{iπ α}]).

    The constructor now accepts
        • a scalar exponent (float / int)               – old behaviour
        • a qiskit Gate / ControlledGate / UnitaryGate
        • a 4×4 numpy matrix
    All of them are converted to the scalar   self.param  ∈ ℝ.
    """
    def __init__(self, exponent=1.0):
        import numpy as np

        def _extract_alpha(mat: np.ndarray) -> float:
            """angle/π of the |11⟩ phase – robust to global phase"""
            phase = np.angle(mat[3, 3])
            return float(phase / np.pi)

        # ------------------------------------------------------------
        # 1) work out the numeric exponent  α
        # ------------------------------------------------------------
        if isinstance(exponent, (int, float)):                    # usual case
            alpha = float(exponent)

        else:                                                     # Gate / matrix
            try:                                                  # qiskit Gate
                mat = exponent.to_matrix()
            except AttributeError:
                mat = np.asarray(exponent)

            if mat.shape != (4, 4):
                raise ValueError("CZPow expects a 4×4 unitary or a scalar α")
            alpha = _extract_alpha(mat)

        # ------------------------------------------------------------
        # 2) build the internal matrix & store α
        # ------------------------------------------------------------
        self.param = alpha
        self.gate = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, np.exp(1j * np.pi * alpha)],
            ],
            dtype=complex,
        )
        super().__init__(name="CZPow", num_qubits=2)

class I(QuantumGate):
    def __init__(self):
        self.gate = np.eye(2, dtype=float)
        super().__init__(name="I", num_qubits=1)
class ECR(QuantumGate):
    def __init__(self):
        self.gate = (
            1
            / np.sqrt(2)
            * np.array(
                [
                    [0.0, 1.0, 0.0, 1.0j],
                    [1.0, 0.0, -1.0j, 0.0],
                    [0.0, 1.0j, 0.0, 1.0],
                    [-1.0j, 0.0, 1.0, 0.0],
                ]
            )
        )
        super().__init__(name="ECR", num_qubits=2)

class U(QuantumGate):
    """
    Arbitrary single-qubit rotation  U(θ, φ, λ)  following the
    OpenQASM / Qiskit convention.

        U(θ, φ, λ) = RZ(φ) · RY(θ) · RZ(λ)

    Parameters
    ----------
    theta, phi, lam : float
        Rotation angles in radians.
    """
    def __init__(self, theta: float, phi: float, lam: float):
        self.params: Tuple[float, float, float] = (theta, phi, lam)

        c, s = np.cos(theta / 2), np.sin(theta / 2)
        mat = np.array(
            [[c, -np.exp(1j * lam) * s],
             [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]],
            dtype=complex,
        )
        self.gate = mat
        super().__init__(name="U", num_qubits=1)


class U1(U):
    """Phase gate  U1(λ) = U(0, 0, λ)."""
    def __init__(self, lam: float):
        super().__init__(0.0, 0.0, lam)


class U2(U):
    """√X-like gate  U2(φ, λ) = U(π/2, φ, λ)."""
    def __init__(self, phi: float, lam: float):
        super().__init__(np.pi / 2, phi, lam)


class U3(U):
    """Alias kept for Qiskit/Cirq compatibility."""
    def __init__(self, theta: float, phi: float, lam: float):
        super().__init__(theta, phi, lam)