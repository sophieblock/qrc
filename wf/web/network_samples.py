from ..simulation.refactor.Process_Library.Gaussian_Elim import *
from ..simulation.refactor.resources import ClassicalDevice
from ..simulation.refactor.broker import Broker

# from qrew.simulation.refactor.Process_Library.Gaussian_Elim import *
# from qrew.simulation.refactor.resources import ClassicalDevice
# from qrew.simulation.refactor.broker import Broker


from scipy.linalg import lu


def generate_broker():
    supercomputer = ClassicalDevice(
        device_name="Supercomputer",
        processor_type="CPU",
        RAM=100 * 10**9,
        properties={"Cores": 20, "Clock Speed": 3 * 10**9},
    )

    broker = Broker(classical_devices=[supercomputer])
    return broker


def generate_GE_network_random_matrix(dim1: int, dim2: int):
    matrix = np.random.rand(dim1, dim2)
    network = generate_GE_network(broker=generate_broker())

    return matrix, network



def generate_Gaussian_elimination_network_random_demo(nr,nc,simulate=True):
    matrix,network = generate_GE_network_random_matrix(nr,nc)
    df = network.run(
        network.input_nodes,
        starting_inputs=[
            (Data(data=matrix,properties={"Usage":"Matrix"}),
            (Data(data=0, properties={"Usage":"Column Idx"})),
            )
        ],
        simulate=simulate
    )

    # network.generate_gantt_plot()
    return network,df


from qrew.simulation.refactor.Process_Library.chemistry_vqe import *
from qrew.chemistry_ingestion.molecule import (
    atom_basis_data,
    mol_basis_data,
    QuantizedMolecule,
    ANGSTROM_TO_BOHR,
)
from qrew.simulation.refactor.devices.quantum_devices import *

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
# -------------------------------------------------------------------
# Example function to generate an electronic energy network
# -------------------------------------------------------------------
def generate_electronic_energy_network(symbols, coords, charge, broker, simulate=True):

    H4 = [
        ("H", [0.7071, 0.0, 0.0]),
        ("H", [0.0, 0.7071, 0.0]),
        ("H", [-1.0071, 0.0, 0.0]),
        ("H", [0.0, -1.0071, 0.0]),
    ]

    H2 = [
        ("H", [0., 0., 0.]),
        ("H", [0.74, 0., 0.]),
    ]

    O2 = [
        ("O", [0., 0, 0.]),
        ("O", [1.21, 0., 0.]),
    ]

    sc = [(symbols[i], coords[i]) for i in range(len(symbols))]

    #mol_H4_sto3g = QuantizedMolecule(H4, 0, 0, basis="sto-3g", units="angstrom")
    #mol_H2_sto3g = QuantizedMolecule(H2, 0, 0, basis="sto-3g", units="angstrom")
    #mol_O2_sto3g = QuantizedMolecule(O2, 0, 0, basis="sto-3g", units="angstrom")
    mol_sto3g = QuantizedMolecule(sc, 0, 0, basis='sto-3g', units='angstrom')

    basis = "sto-3g"
    #symbols = ["H", "H", "H", "H"]
    #symbols = ["H", "H"]
    #symbols = ["O", "O"]
    coordinates = coords
   
    coordinates = [
        [elmt * ANGSTROM_TO_BOHR for elmt in coordinates[idx]]
        for idx in range(len(coordinates))
    ]
    charge = 0

    geommodel = 'RHF'

    starting_inputs = [
        [
            Data(data=basis, properties={"Usage": "Basis"}),
            Data(data=symbols, properties={"Usage": "Atomic Symbols"}),
            Data(data=coordinates, properties={"Usage": "Coordinates"}),
            Data(data=charge, properties={"Usage": "Total Charge"}),
            Data(data=geommodel, properties={"Usage": "Geometry Model"}),
        ]
    ]

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
    coulomb_term = Node(
        process_model=GSE_ComputeCoulombTerm,
        network_type="INPUT",
        extend_dynamic=True,
    )
    exchange_term = Node(
        process_model=GSE_ComputeExchangeTerm,
        network_type="INPUT",
        extend_dynamic=True,
    )
    compute_fock = Node(process_model=GSE_ComputeFockMat, network_type="NETWORK")
    trnsfrm_fock = Node(process_model=GSE_TransformFock, network_type="NETWORK")
    ortho_fock_eigs = Node(
        process_model=GSE_ComputeEigs,
        network_type="NETWORK",
        matrix_name="Orthogonal Fock",
    )
    trnsfrm_eigvecs = Node(process_model=GSE_TransformEigvecs, network_type="NETWORK")
    update_density = Node(process_model=GSE_UpdateDensityMat, network_type="NETWORK")
    compute_elec_energy = Node(
        process_model=GSE_ComputeElecEnergy, network_type="OUTPUT"
    )

    init_mol.insert_output_node(init_basis_set)
    init_mol.insert_output_node(coulomb_term)
    init_mol.insert_output_node(exchange_term)
    init_mol.insert_output_node(core)
    init_mol.insert_output_node(update_density)
    init_mol.insert_output_node(nuc_repl)
    init_basis_set.insert_output_node(core)
    init_basis_set.insert_output_node(overlap)
    init_basis_set.insert_output_node(repulsion)
    core.insert_output_node(compute_fock)
    repulsion.insert_output_node(coulomb_term)
    repulsion.insert_output_node(exchange_term)
    overlap.insert_output_node(overlap_eigs)
    overlap_eigs.insert_output_node(trnsfrm_mat)
    trnsfrm_mat.insert_output_node(trnsfrm_fock)
    trnsfrm_mat.insert_output_node(trnsfrm_eigvecs)
    coulomb_term.insert_output_node(compute_fock)
    exchange_term.insert_output_node(compute_fock)
    compute_fock.insert_output_node(trnsfrm_fock)
    trnsfrm_fock.insert_output_node(ortho_fock_eigs)
    ortho_fock_eigs.insert_output_node(trnsfrm_eigvecs)
    trnsfrm_eigvecs.insert_output_node(update_density)
    update_density.insert_output_node(compute_elec_energy)
    core.insert_output_node(compute_elec_energy)
    compute_fock.insert_output_node(compute_elec_energy)
    nuc_repl.insert_output_node(compute_elec_energy)

    network = Network(
        nodes=[
            init_mol,
            nuc_repl,
            init_basis_set,
            core,
            repulsion,
            overlap,
            overlap_eigs,
            trnsfrm_mat,
            coulomb_term,
            exchange_term,
            compute_fock,
            trnsfrm_fock,
            ortho_fock_eigs,
            trnsfrm_eigvecs,
            update_density,
            compute_elec_energy,
        ],
        input_nodes=[init_mol, coulomb_term, exchange_term],
        output_nodes=[compute_elec_energy],
        broker=broker, #gen_broker(),
    )

    nn = len(symbols)
    init_density = torch.zeros((nn, nn), dtype=torch.float64)
    add_inputs = [[Data(data=init_density, properties={"Usage": "Density Matrix"})]] * 2
    print("starting to run network...")
    df = network.run(
        starting_nodes=[init_mol, coulomb_term, exchange_term],
        starting_inputs=starting_inputs + add_inputs,
        simulate=simulate,
    )
    print("finished running network...")

    output_edges = network.output_nodes[0].output_edges
   

    return network, df

def generate_electronic_energy_network2(symbols, coordinates, basis, charge, geometry_model, qubit_mapping, ansatz, ansatz_params, broker=None, simulate=True):
   
    df, network = run_VQE_ansatz_network(
        symbols=symbols, 
        coordinates=coordinates, 
        basis=basis,
        charge=charge,
        geometry_model=geometry_model,
        qubit_mapping=qubit_mapping,
        ansatz=ansatz,
        ansatz_params=ansatz_params,
        broker=broker or gen_broker(),
        simulate=simulate,
    )

    output_edges = network.output_nodes[0].output_edges
   
    return network, df