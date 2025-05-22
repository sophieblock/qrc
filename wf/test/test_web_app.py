import pytest

from qrew.simulation.refactor.devices.quantum_devices import *
from qrew.simulation.refactor.quantum import *
from qrew.simulation.refactor.quantum_gates import *
from qrew.simulation.refactor.process import QuantumProcess
from qrew.simulation.refactor.quantum import *
from qrew.simulation.refactor.graph import Node,Network
from qrew.simulation.refactor.broker import Broker
from qrew.simulation.refactor.resources import QuantumResource, ClassicalResource, ClassicalDevice
from qrew.simulation.refactor.data import Data

from qrew.simulation.refactor.Process_Library.chemistry_vqe import *
from qrew.results import visualize_graph_from_nx

from qrew.simulation.refactor.quantum_gates import X, H, CZPow, CX, RX, RZ
from qrew.chemistry_ingestion.molecule import (
    atom_basis_data,
    BasisFunction,
    ANGSTROM_TO_BOHR,
)

def gen_broker():
    supercomputer = ClassicalDevice(
        device_name="Supercomputer",
        processor_type="CPU",
        RAM=100 * 10**9,
        properties={"Cores": 20, "Clock Speed": 3 * 10**9},
    )

    broker = Broker(
        classical_devices=[supercomputer],
        quantum_devices=[
            IBM_Brisbane,
            IBM_Kyiv,
            IBM_Fez,
            IBM_Nazca,
            IBM_Sherbrooke,
        ],
    )
    return broker

def run_VQE_ansatz_network(
    symbols: list[str],
    coordinates: list[list[int]],
    basis: str = "sto-3g",
    charge: int = 0,
    geometry_model="RHF",
    qubit_mapping="JW",
    ansatz="UCCSD",
    ansatz_params="random",
    broker=None,
    simulate=False,
):
    """
    Args:
        symbols (list[str]): Atomic symbols of the molecule, e.g. H4 -> ["H","H","H","H"]
        coordinates (list[list[int]]): Spatial coordinates of each corresponding atom in symbols in angstrom 
                                    e.g. -> [[0.7071, 0.0, 0.0],[0.0, 0.7071, 0.0],[-1.0071, 0.0, 0.0],[0.0, -1.0071, 0.0]]
        basis (str, optional): Basis type. Defaults to "sto-3g".
        charge (int, optional): Total charge of molecule. Defaults to 0.
        geometry_model (str, optional): Molecular geometry model from ["UHF","RHF"]. Defaults to "RHF".
        qubit_mapping (str, optional): Fermion to qubit mapping from ["JW","BK","SCBK"]. Defaults to "JW".
        ansatz (str, optional): Ansatz model. Defaults to "UCCSD".
        ansatz_params (str, optional): Initial ansatz parameters from ["random", "ones"] or a np.ndarray. Defaults to "random".
        broker (_type_, optional): Broker for the network. Defaults to None.

    Returns:
        df, network (pd.dataframe, Network): Returns the pandas dataframe object storing resource estimation results for each of the
                                            processed Nodes and the Network instance.
    """

    bohr_coords = [
        [elmt * ANGSTROM_TO_BOHR for elmt in coordinates[idx]]
        for idx in range(len(coordinates))
    ]

    init_mol_inputs = [
        [
            Data(data=symbols, properties={"Usage": "Atomic Symbols"}),
            Data(data=bohr_coords, properties={"Usage": "Coordinates"}),
            Data(data=basis, properties={"Usage": "Basis"}),
            Data(data=charge, properties={"Usage": "Total Charge"}),
            Data(data=geometry_model, properties={"Usage": "Geometry Model"}),
        ]
    ]
    init_ansatz_inputs = [
        [
            Data(data=qubit_mapping, properties={"Usage": "Qubit Mapping"}),
            Data(data=ansatz, properties={"Usage": "Ansatz"}),
            Data(data=ansatz_params, properties={"Usage": "Ansatz Params"}),
        ]
    ]

    network = gen_VQE_ansatz_network(broker=broker)
    
    df = network.run(
        starting_nodes=network.input_nodes,
        starting_inputs=init_mol_inputs + init_ansatz_inputs,
    )

    return df, network

def gen_VQE_ansatz_network(broker=None):
    init_mol = Node(process_model=GSE_InitMolecule, network_type="INPUT")
    nuc_repl = Node(process_model=GSE_ComputeNuclearRepulsion, network_type="NETWORK")
    init_basis_set = Node(
        process_model=GSE_InitBasisSet, network_type="NETWORK", extend_dynamic=True
    )
    core = Node(process_model=GSE_ComputeCore, network_type="NETWORK")
    repulsion = Node(process_model=GSE_ComputeRepulsion, network_type="NETWORK")
    overlap = Node(process_model=GSE_ComputeOverlap, network_type="NETWORK")
    overlap_eigs = Node(
        process_model=GSE_ComputeEigs, network_type="NETWORK", matrix_name="Overlap"
    )
    trnsfrm_mat = Node(process_model=GSE_ComputeTrnsfrmMat, network_type="NETWORK")
    scf_itr = Node(
        process_model=GSE_SCFItr, network_type="NETWORK", extend_dynamic=True
    )
    init_ansatz = Node(
        process_model=GSE_InitAnsatz, network_type="INPUT", extend_dynamic=True
    )
    ref_circuit = Node(process_model=GSE_GenRefCircuit, network_type="NETWORK")
    set_ansatz = Node(process_model=GSE_SetAnsatzParams, network_type="NETWORK")
    compute_ferm_op = Node(process_model=GSE_ComputeFermionOp, network_type="NETWORK")
    compute_qubit_op = Node(process_model=GSE_ComputeQubitOp, network_type="NETWORK")
    gen_ansatz_circ = Node(
        process_model=GSE_GenAnsatzCircuit, network_type="OUTPUT", extend_dynamic=True
    )

    init_mol.insert_output_node(init_basis_set)
    init_mol.insert_output_node(core)
    init_mol.insert_output_node(nuc_repl)
    init_basis_set.insert_output_node(core)
    init_basis_set.insert_output_node(overlap)
    init_basis_set.insert_output_node(repulsion)
    overlap.insert_output_node(overlap_eigs)
    overlap_eigs.insert_output_node(trnsfrm_mat)

    init_mol.insert_output_node(scf_itr)
    nuc_repl.insert_output_node(scf_itr)
    core.insert_output_node(scf_itr)
    repulsion.insert_output_node(scf_itr)
    trnsfrm_mat.insert_output_node(scf_itr)

    init_mol.insert_output_node(init_ansatz)
    scf_itr.insert_output_node(init_ansatz)

    init_ansatz.insert_output_node(ref_circuit)
    init_ansatz.insert_output_node(set_ansatz)
    init_ansatz.insert_output_node(compute_ferm_op)
    set_ansatz.insert_output_node(compute_ferm_op)

    compute_ferm_op.insert_output_node(compute_qubit_op)
    init_ansatz.insert_output_node(compute_qubit_op)

    ref_circuit.insert_output_node(gen_ansatz_circ)
    compute_qubit_op.insert_output_node(gen_ansatz_circ)

    network = Network(
        nodes=[
            init_mol,
            init_basis_set,
            core,
            repulsion,
            overlap,
            overlap_eigs,
            trnsfrm_mat,
            scf_itr,
            init_ansatz,
            set_ansatz,
            ref_circuit,
            compute_ferm_op,
            compute_qubit_op,
            gen_ansatz_circ,
        ],
        input_nodes=[init_mol, init_ansatz],
        output_nodes=[gen_ansatz_circ],
        broker=broker,
    )

    return network
def test_run_vqe_ansatz_network():
    symbols = ["H","H"]
    coords = [[0, 0, 0], [0.7414, 0, 0]]
    charge = 0
    df, network = run_VQE_ansatz_network(
        symbols=symbols, coordinates=coords, broker=gen_broker(), simulate = True
    )
    nvg,mpg = network.to_networkx()

    pg = visualize_graph_from_nx(nvg,
                            default_data_node_shape='triangle',
                            proc_node_color_fn=None,
                            proc_node_shape_fn=None,
                            )
    

