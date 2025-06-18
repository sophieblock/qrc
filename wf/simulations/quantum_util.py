from typing import List, Dict, Tuple,NamedTuple
from .schema import RegisterSpec
class QuantumOperand(NamedTuple):
    register: RegisterSpec
    offset:   int        # e.g. 0..reg.num_qubits-1

def idx(reg: RegisterSpec, k: int = 0) -> QuantumOperand:
    assert 0 <= k < reg.num_qubits, f"offset {k} ≥ num_qubits={reg.num_qubits}"
    return QuantumOperand(register=reg, offset=k)

