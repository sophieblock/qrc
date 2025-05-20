

from qrew.simulation.refactor.devices.ibm_device_parser import LoadQuantumDeviceData
from qrew.simulation.refactor.resources.quantum_resources import QuantumDevice
from qrew.simulation.refactor.quantum_gates import *
# from qrew.simulation.refactor.q_interop.qiskit_interop import quantum_circuit_to_qiskit, qiskit_to_quantum_circuit
# from qrew.simulation.refactor.q_interop.cirq_interop import qc_to_cirq,translate_qc_to_cirq
from qrew.simulation.refactor.q_interop.transpilers import QiskitCompiler, CirqCompiler

from ....util.log import get_logger
logger = get_logger(__name__)

IBM_Brisbane = QuantumDevice(
    device_name="IBM Brisbane",
    connectivity=LoadQuantumDeviceData("ibm_brisbane.csv").generate_connectivity(),
   
    compiler=QiskitCompiler(basis_gates=("ECR", "I", "RZ", "SX", "X")),
)


IBM_Kyiv = QuantumDevice(
    device_name="IBM Kyiv",
    connectivity=LoadQuantumDeviceData("ibm_kyiv.csv").generate_connectivity(),
    compiler=QiskitCompiler(basis_gates=("ECR", "I", "RZ", "SX", "X")),

)



IBM_Sherbrooke = QuantumDevice(
    device_name="IBM Sherbrooke",
    connectivity=LoadQuantumDeviceData("ibm_sherbrooke.csv").generate_connectivity(),
    compiler=QiskitCompiler(basis_gates=("ECR", "I", "RZ", "SX", "X")),

)

IBM_Fez = QuantumDevice(
    device_name="IBM Fez",
    connectivity=LoadQuantumDeviceData("ibm_fez.csv").generate_connectivity(),
    compiler=QiskitCompiler(basis_gates=("CZ", "I", "RZ", "SX", "X")),

)

IBM_Nazca = QuantumDevice(
    device_name="IBM Nazca",
    connectivity=LoadQuantumDeviceData("ibm_nazca.csv").generate_connectivity(),
    compiler=QiskitCompiler(basis_gates=("ECR", "I", "RZ", "SX", "X")),
)
# google_Sycamore = QuantumDevice(
#     "Google Sycamore",
#     connectivity=LoadQuantumDeviceData("sycamore23.csv").generate_connectivity(),
#     gate_set=("CZ","X","Y","PhX"),
#     compiler=CirqCompiler(lookahead_radius=8),
# )

def view_connectivity(device: QuantumDevice):
    connectivity = device.connectivity
    # print(connectivity[0][1]["CZ duration"])
    for node in connectivity.nodes:
        print(node, connectivity.nodes[node])
    for edge in connectivity.edges:
        print(edge, connectivity[edge[0]][edge[1]])

