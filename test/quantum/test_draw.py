import networkx as nx
from olsq.solve import collision_extracting
import pytest


from qrew.simulation.refactor.quantum import QuantumCircuit,QuantumInstruction, LayoutSynthesizer
from qrew.simulation.refactor.quantum_gates import *


# def line4_device():
#     g = nx.Graph([(0, 1), (1, 2), (2,3)])
#     return QuantumDevice(
#         "line3", g, gate_set=("H", "CX", "SWAP"),
#     )
def leaf_cnot():
    qc = QuantumCircuit(qubit_count=2)   # logical q0, q1
    qc.add_instruction(QuantumInstruction(H(),  (0,)))
    qc.add_instruction(QuantumInstruction(H(),  (1,)))
    qc.add_instruction(QuantumInstruction(CX(), (0, 1)))   # needs 0-2 edge
    return qc


def test_draw_line4():
    circ = leaf_cnot()
    circ.draw()
