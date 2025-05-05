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
    def __init__(self, exponent: float):
        self.param = exponent
        self.gate = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, np.exp(1j * np.pi * exponent)],
            ]
        )
        super().__init__(name="CZPow", num_qubits=2)


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
