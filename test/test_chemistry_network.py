from typing import List
from openfermion.ops import FermionOperator, QubitOperator

from workflow.simulation.refactor.data_types import MatrixType
from workflow.simulation.refactor.data import Data
from workflow.simulation.refactor.process import Process, ClassicalProcess
from workflow.simulation.refactor.graph import Node, DirectedEdge, Network
from workflow.simulation.refactor.resources.classical_resources import (
    ClassicalResource,
    ClassicalDevice,
)
from workflow.simulation.refactor.Process_Library.chemistry_vqe import *
from workflow.simulation.refactor.broker import Broker
from workflow.simulation.refactor.quantum import QuantumCircuit
from workflow.simulation.refactor.devices.quantum_devices import *
from numpy import inf as INFINITY
import pytest

from workflow.chemistry_ingestion.molecule import (
    atom_basis_data,
    mol_basis_data,
    QuantizedMolecule,
    ANGSTROM_TO_BOHR,
)

H4 = [
    ("H", [0.7071, 0.0, 0.0]),
    ("H", [0.0, 0.7071, 0.0]),
    ("H", [-1.0071, 0.0, 0.0]),
    ("H", [0.0, -1.0071, 0.0]),
]

mol_H4_sto3g = QuantizedMolecule(H4, 0, 0, basis="sto-3g", units="angstrom")

basis = "sto-3g"
symbols = ["H", "H", "H", "H"]
coordinates = [
    [0.7071, 0.0, 0.0],
    [0.0, 0.7071, 0.0],
    [-1.0071, 0.0, 0.0],
    [0.0, -1.0071, 0.0],
]
coordinates = [
    [elmt * ANGSTROM_TO_BOHR for elmt in coordinates[idx]]
    for idx in range(len(coordinates))
]
charge = 0
geometry_model = "RHF"

starting_inputs = [
    [
        Data(data=basis, properties={"Usage": "Basis"}),
        Data(data=symbols, properties={"Usage": "Atomic Symbols"}),
        Data(data=coordinates, properties={"Usage": "Coordinates"}),
        Data(data=charge, properties={"Usage": "Total Charge"}),
        Data(data=geometry_model, properties={"Usage": "Geometry Model"}),
    ]
]


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
            IBM_Brussels,
            IBM_Fez,
            IBM_Kyiv,
            IBM_Nazca,
            IBM_Sherbrooke,
        ],
    )
    return broker


def test_init_mol():
    coordinates = [
        [0.7071, 0.0, 0.0],
        [0.0, 0.7071, 0.0],
        [-1.0071, 0.0, 0.0],
        [0.0, -1.0071, 0.0],
    ]
    init_mol = GSE_InitMolecule(
        inputs=[
            Data(data=basis, properties={"Usage": "Basis"}),
            Data(data=symbols, properties={"Usage": "Atomic Symbols"}),
            Data(data=coordinates, properties={"Usage": "Coordinates"}),
            Data(data=charge, properties={"Usage": "Total Charge"}),
            Data(data=geometry_model, properties={"Usage": "Geometry Model"}),
        ]
    )

    init_mol.update()
    assert init_mol.results["Atomic Data"] == [("sto-3g", "H")] * 4

    output_data = init_mol.generate_output()
    assert [output.properties["Index"] for output in output_data[:4]] == list(range(4))


def test_extract_basis():
    extract_basis = GSE_ExtractBasis(
        inputs=[
            Data(data=("sto-3g", "H"), properties={"Usage": "Atomic Data", "Index": 0})
        ],
        index=0,
    )

    extract_basis.update()
    result = extract_basis.results
    expected_result = atom_basis_data("sto-3g", "H")
    assert result[0] == expected_result[0]

    output_data = extract_basis.generate_output()
    assert output_data[0].data[0] == expected_result[0]


def test_gen_basis():
    input_data = atom_basis_data("sto-3g", "H")
    gen_basis = GSE_GenBasisData(
        inputs=[
            Data(
                data=input_data, properties={"Usage": "Atomic Basis Data", "Index": idx}
            )
            for idx in range(4)
        ],
        num_atoms=4,
    )

    gen_basis.update()
    assert gen_basis.results[0] == [len(input_data)] * 4
    assert gen_basis.results[1] == [input_data[0]] * 4


def test_init_mol_network():
    init_mol = Node(process_model=GSE_InitMolecule, network_type="INPUT")

    network = Network(
        nodes=[init_mol], input_nodes=[init_mol], output_nodes=[], broker=gen_broker()
    )

    network.run(starting_nodes=[init_mol], starting_inputs=starting_inputs)

    assert len(network.input_nodes) == 1
    assert len(network.nodes) == 6
    assert len(network.output_nodes) == 1

    expected_data = atom_basis_data("sto-3g", "H")

    output_edges = network.output_nodes[0].output_edges
    for edge in output_edges:
        if edge.data[0].properties["Usage"] == "Num Basis Per Atom":
            assert edge.data[0].data == [len(expected_data)] * 4
        elif edge.data[0].properties["Usage"] == "Basis Data":
            assert edge.data[0].data == [expected_data[0]] * 4
        else:
            raise ValueError()


def test_init_basis_set_network():
    init_mol = Node(process_model=GSE_InitMolecule, network_type="INPUT")
    init_basis_set = Node(
        process_model=GSE_InitBasisSet, network_type="OUTPUT", extend_dynamic=True
    )

    init_mol.insert_output_node(init_basis_set)

    network = Network(
        nodes=[init_mol, init_basis_set],
        input_nodes=[init_mol],
        output_nodes=[init_basis_set],
        broker=gen_broker(),
    )

    network.run(starting_nodes=[init_mol], starting_inputs=starting_inputs)

    output_edges = network.output_nodes[0].output_edges
    assert len(output_edges) == 1
    assert output_edges[0].data[0].properties["Usage"] == "Basis Set"

    basis_set = output_edges[0].data[0].data
    for idx in range(len(basis_set)):
        assert basis_set[idx].l == mol_H4_sto3g.basis_set[idx].l
        assert torch.allclose(basis_set[idx].alpha, mol_H4_sto3g.basis_set[idx].alpha)
        assert torch.allclose(basis_set[idx].coeff, mol_H4_sto3g.basis_set[idx].coeff)
        assert torch.allclose(basis_set[idx].r, mol_H4_sto3g.basis_set[idx].r)


def test_compute_S():
    init_mol = Node(process_model=GSE_InitMolecule, network_type="INPUT")
    init_basis_set = Node(
        process_model=GSE_InitBasisSet, network_type="NETWORK", extend_dynamic=True
    )
    overlap = Node(process_model=GSE_ComputeOverlap, network_type="OUTPUT")

    init_mol.insert_output_node(init_basis_set)
    init_basis_set.insert_output_node(overlap)

    network = Network(
        nodes=[init_mol, init_basis_set, overlap],
        input_nodes=[init_mol],
        output_nodes=[overlap],
        broker=gen_broker(),
    )

    network.run(starting_nodes=[init_mol], starting_inputs=starting_inputs)

    output_edges = network.output_nodes[0].output_edges
    assert len(output_edges) == 1
    assert output_edges[0].data[0].properties["Usage"] == "Overlap Matrix"

    overlap_matrix = output_edges[0].data[0].data
    assert torch.allclose(mol_H4_sto3g._overlap_integrals, overlap_matrix)


def test_compute_repulsion_tensor():
    init_mol = Node(process_model=GSE_InitMolecule, network_type="INPUT")
    init_basis_set = Node(
        process_model=GSE_InitBasisSet, network_type="NETWORK", extend_dynamic=True
    )
    repulsion = Node(process_model=GSE_ComputeRepulsion, network_type="OUTPUT")

    init_mol.insert_output_node(init_basis_set)
    init_basis_set.insert_output_node(repulsion)

    network = Network(
        nodes=[init_mol, init_basis_set, repulsion],
        input_nodes=[init_mol],
        output_nodes=[repulsion],
        broker=gen_broker(),
    )

    network.run(starting_nodes=[init_mol], starting_inputs=starting_inputs)

    output_edges = network.output_nodes[0].output_edges
    assert len(output_edges) == 1
    assert output_edges[0].data[0].properties["Usage"] == "Repulsion Tensor"

    repulsion_tensor = output_edges[0].data[0].data
    assert torch.allclose(repulsion_tensor, mol_H4_sto3g._repulsion_tensor)


def test_compute_core():
    init_mol = Node(process_model=GSE_InitMolecule, network_type="INPUT")
    init_basis_set = Node(
        process_model=GSE_InitBasisSet, network_type="NETWORK", extend_dynamic=True
    )
    core = Node(process_model=GSE_ComputeCore, network_type="OUTPUT")

    init_mol.insert_output_node(init_basis_set)
    init_mol.insert_output_node(core)
    init_basis_set.insert_output_node(core)

    network = Network(
        nodes=[init_mol, init_basis_set, core],
        input_nodes=[init_mol],
        output_nodes=[core],
        broker=gen_broker(),
    )

    network.run(starting_nodes=[init_mol], starting_inputs=starting_inputs)

    output_edges = network.output_nodes[0].output_edges
    assert len(output_edges) == 1
    assert output_edges[0].data[0].properties["Usage"] == "Core Matrix"

    core_matrix = output_edges[0].data[0].data
    assert torch.allclose(core_matrix, mol_H4_sto3g._h_core)


def test_compute_nuclear_repulsion():
    init_mol = Node(process_model=GSE_InitMolecule, network_type="INPUT")
    nuc_repl = Node(process_model=GSE_ComputeNuclearRepulsion, network_type="OUTPUT")

    init_mol.insert_output_node(nuc_repl)

    network = Network(
        nodes=[init_mol, nuc_repl],
        input_nodes=[init_mol],
        output_nodes=[nuc_repl],
        broker=gen_broker(),
    )

    network.run(starting_nodes=[init_mol], starting_inputs=starting_inputs)

    output_edges = network.output_nodes[0].output_edges
    assert len(output_edges) == 1
    assert output_edges[0].data[0].properties["Data Type"] == float
    assert output_edges[0].data[0].properties["Usage"] == "Nuclear Repulsion"

    nuclear_repulsion = output_edges[0].data[0].data
    assert nuclear_repulsion == mol_H4_sto3g.nuclear_repulsion_energy


def test_compute_overlap_eigs():
    init_mol = Node(process_model=GSE_InitMolecule, network_type="INPUT")
    init_basis_set = Node(
        process_model=GSE_InitBasisSet, network_type="NETWORK", extend_dynamic=True
    )
    overlap = Node(process_model=GSE_ComputeOverlap, network_type="NETWORK")
    overlap_eigs = Node(
        process_model=GSE_ComputeEigs, network_type="OUTPUT", matrix_name="Overlap"
    )

    init_mol.insert_output_node(init_basis_set)
    init_basis_set.insert_output_node(overlap)
    overlap.insert_output_node(overlap_eigs)

    network = Network(
        nodes=[init_mol, init_basis_set, overlap, overlap_eigs],
        input_nodes=[init_mol],
        output_nodes=[overlap_eigs],
        broker=gen_broker(),
    )

    network.run(starting_nodes=[init_mol], starting_inputs=starting_inputs)

    output_edges = network.output_nodes[0].output_edges
    assert len(output_edges) == 2
    assert output_edges[0].data[0].properties["Usage"] == "Overlap Eigvals"
    assert output_edges[1].data[0].properties["Usage"] == "Overlap Eigvecs"


def test_compute_transform_mat():
    init_mol = Node(process_model=GSE_InitMolecule, network_type="INPUT")
    init_basis_set = Node(
        process_model=GSE_InitBasisSet, network_type="NETWORK", extend_dynamic=True
    )
    overlap = Node(process_model=GSE_ComputeOverlap, network_type="NETWORK")
    overlap_eigs = Node(
        process_model=GSE_ComputeEigs, network_type="NETWORK", matrix_name="Overlap"
    )
    trnsfrm_mat = Node(process_model=GSE_ComputeTrnsfrmMat, network_type="OUTPUT")

    init_mol.insert_output_node(init_basis_set)
    init_basis_set.insert_output_node(overlap)
    overlap.insert_output_node(overlap_eigs)
    overlap_eigs.insert_output_node(trnsfrm_mat)

    network = Network(
        nodes=[init_mol, init_basis_set, overlap, overlap_eigs, trnsfrm_mat],
        input_nodes=[init_mol],
        output_nodes=[trnsfrm_mat],
        broker=gen_broker(),
    )

    network.run(starting_nodes=[init_mol], starting_inputs=starting_inputs)

    output_edges = network.output_nodes[0].output_edges
    assert len(output_edges) == 1
    assert output_edges[0].data[0].properties["Usage"] == "Transform Matrix"


def test_compute_coulomb_term():
    init_mol = Node(process_model=GSE_InitMolecule, network_type="INPUT")
    init_basis_set = Node(
        process_model=GSE_InitBasisSet, network_type="NETWORK", extend_dynamic=True
    )
    repulsion = Node(process_model=GSE_ComputeRepulsion, network_type="NETWORK")

    init_mol.insert_output_node(init_basis_set)
    init_basis_set.insert_output_node(repulsion)

    network = Network(
        nodes=[init_mol, init_basis_set, repulsion],
        input_nodes=[init_mol],
        output_nodes=[repulsion],
        broker=gen_broker(),
    )

    network.run(starting_nodes=[init_mol], starting_inputs=starting_inputs)

    output_edges = network.output_nodes[0].output_edges
    assert len(output_edges) == 1
    assert output_edges[0].data[0].properties["Data Type"] == torch.Tensor
    assert output_edges[0].data[0].properties["Usage"] == "Repulsion Tensor"

    init_density = torch.zeros((4, 4), dtype=torch.float64)
    coulomb_term = GSE_ComputeCoulombTerm(
        inputs=[
            output_edges[0].data[0],
            Data(data=init_density, properties={"Usage": "Density Matrix"}),
        ]
    )

    coulomb_term.update()
    output = coulomb_term.generate_output()

    assert output[0].properties["Data Type"] == torch.Tensor
    assert output[0].properties["Usage"] == "Coulomb Term"


def test_compute_exchange_term():
    init_mol = Node(process_model=GSE_InitMolecule, network_type="INPUT")
    init_basis_set = Node(
        process_model=GSE_InitBasisSet, network_type="NETWORK", extend_dynamic=True
    )
    repulsion = Node(process_model=GSE_ComputeRepulsion, network_type="NETWORK")
    exchange_term = Node(
        process_model=GSE_ComputeExchangeTerm,
        network_type="INPUT",
        extend_dynamic=True,
    )

    init_mol.insert_output_node(init_basis_set)
    init_mol.insert_output_node(exchange_term)
    init_basis_set.insert_output_node(repulsion)
    repulsion.insert_output_node(exchange_term)

    network = Network(
        nodes=[init_mol, init_basis_set, repulsion, exchange_term],
        input_nodes=[init_mol, exchange_term],
        output_nodes=[exchange_term],
        broker=gen_broker(),
    )

    init_density = torch.zeros((4, 4), dtype=torch.float64)
    add_inputs = [[Data(data=init_density, properties={"Usage": "Density Matrix"})]]
    network.run(
        starting_nodes=[init_mol, exchange_term],
        starting_inputs=starting_inputs + add_inputs,
    )

    output_edges = network.output_nodes[0].output_edges
    assert len(output_edges) == 1
    assert output_edges[0].data[0].properties["Data Type"] == torch.Tensor
    assert output_edges[0].data[0].properties["Usage"] == "Exchange Term"


def test_transform_fock():
    init_mol = Node(process_model=GSE_InitMolecule, network_type="INPUT")
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
    trnsfrm_fock = Node(process_model=GSE_TransformFock, network_type="OUTPUT")

    init_mol.insert_output_node(init_basis_set)
    init_mol.insert_output_node(coulomb_term)
    init_mol.insert_output_node(exchange_term)
    init_mol.insert_output_node(core)
    init_basis_set.insert_output_node(core)
    init_basis_set.insert_output_node(overlap)
    init_basis_set.insert_output_node(repulsion)
    core.insert_output_node(compute_fock)
    repulsion.insert_output_node(coulomb_term)
    repulsion.insert_output_node(exchange_term)
    overlap.insert_output_node(overlap_eigs)
    overlap_eigs.insert_output_node(trnsfrm_mat)
    trnsfrm_mat.insert_output_node(trnsfrm_fock)
    coulomb_term.insert_output_node(compute_fock)
    exchange_term.insert_output_node(compute_fock)
    compute_fock.insert_output_node(trnsfrm_fock)

    network = Network(
        nodes=[
            init_mol,
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
        ],
        input_nodes=[init_mol, coulomb_term, exchange_term],
        output_nodes=[trnsfrm_fock],
        broker=gen_broker(),
    )

    init_density = torch.zeros((4, 4), dtype=torch.float64)
    add_inputs = [[Data(data=init_density, properties={"Usage": "Density Matrix"})]] * 2
    network.run(
        starting_nodes=[init_mol, coulomb_term, exchange_term],
        starting_inputs=starting_inputs + add_inputs,
    )

    output_edges = network.output_nodes[0].output_edges
    assert len(output_edges) == 1
    assert output_edges[0].data[0].properties["Data Type"] == torch.Tensor
    assert output_edges[0].data[0].properties["Usage"] == "Orthogonal Fock Matrix"


def test_orth_fock_eigs():
    init_mol = Node(process_model=GSE_InitMolecule, network_type="INPUT")
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
        network_type="OUTPUT",
        matrix_name="Orthogonal Fock",
    )

    init_mol.insert_output_node(init_basis_set)
    init_mol.insert_output_node(coulomb_term)
    init_mol.insert_output_node(exchange_term)
    init_mol.insert_output_node(core)
    init_basis_set.insert_output_node(core)
    init_basis_set.insert_output_node(overlap)
    init_basis_set.insert_output_node(repulsion)
    core.insert_output_node(compute_fock)
    repulsion.insert_output_node(coulomb_term)
    repulsion.insert_output_node(exchange_term)
    overlap.insert_output_node(overlap_eigs)
    overlap_eigs.insert_output_node(trnsfrm_mat)
    trnsfrm_mat.insert_output_node(trnsfrm_fock)
    coulomb_term.insert_output_node(compute_fock)
    exchange_term.insert_output_node(compute_fock)
    compute_fock.insert_output_node(trnsfrm_fock)
    trnsfrm_fock.insert_output_node(ortho_fock_eigs)

    network = Network(
        nodes=[
            init_mol,
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
        ],
        input_nodes=[init_mol, coulomb_term, exchange_term],
        output_nodes=[ortho_fock_eigs],
        broker=gen_broker(),
    )

    init_density = torch.zeros((4, 4), dtype=torch.float64)
    add_inputs = [[Data(data=init_density, properties={"Usage": "Density Matrix"})]] * 2
    network.run(
        starting_nodes=[init_mol, coulomb_term, exchange_term],
        starting_inputs=starting_inputs + add_inputs,
    )

    output_edges = network.output_nodes[0].output_edges
    assert len(output_edges) == 2
    assert output_edges[0].data[0].properties["Data Type"] == torch.Tensor
    assert output_edges[0].data[0].properties["Usage"] == "Orthogonal Fock Eigvals"
    assert output_edges[1].data[0].properties["Data Type"] == torch.Tensor
    assert output_edges[1].data[0].properties["Usage"] == "Orthogonal Fock Eigvecs"


def test_update_density_matrix():
    init_mol = Node(process_model=GSE_InitMolecule, network_type="INPUT")
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
    transf_eigvecs = Node(process_model=GSE_TransformEigvecs, network_type="NETWORK")
    update_density = Node(process_model=GSE_UpdateDensityMat, network_type="OUTPUT")

    init_mol.insert_output_node(init_basis_set)
    init_mol.insert_output_node(coulomb_term)
    init_mol.insert_output_node(exchange_term)
    init_mol.insert_output_node(core)
    init_mol.insert_output_node(update_density)
    init_basis_set.insert_output_node(core)
    init_basis_set.insert_output_node(overlap)
    init_basis_set.insert_output_node(repulsion)
    core.insert_output_node(compute_fock)
    repulsion.insert_output_node(coulomb_term)
    repulsion.insert_output_node(exchange_term)
    overlap.insert_output_node(overlap_eigs)
    overlap_eigs.insert_output_node(trnsfrm_mat)
    trnsfrm_mat.insert_output_node(trnsfrm_fock)
    trnsfrm_mat.insert_output_node(transf_eigvecs)
    coulomb_term.insert_output_node(compute_fock)
    exchange_term.insert_output_node(compute_fock)
    compute_fock.insert_output_node(trnsfrm_fock)
    trnsfrm_fock.insert_output_node(ortho_fock_eigs)
    ortho_fock_eigs.insert_output_node(transf_eigvecs)
    transf_eigvecs.insert_output_node(update_density)

    network = Network(
        nodes=[
            init_mol,
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
            transf_eigvecs,
            update_density,
        ],
        input_nodes=[init_mol, coulomb_term, exchange_term],
        output_nodes=[update_density],
        broker=gen_broker(),
    )

    init_density = torch.zeros((4, 4), dtype=torch.float64)
    add_inputs = [[Data(data=init_density, properties={"Usage": "Density Matrix"})]] * 2
    network.run(
        starting_nodes=[init_mol, coulomb_term, exchange_term],
        starting_inputs=starting_inputs + add_inputs,
    )

    output_edges = network.output_nodes[0].output_edges
    assert len(output_edges) == 1
    assert output_edges[0].data[0].properties["Data Type"] == torch.Tensor
    assert output_edges[0].data[0].properties["Usage"] == "Updated Density Matrix"


def test_compute_electronic_energy():
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
        broker=gen_broker(),
    )

    init_density = torch.zeros((4, 4), dtype=torch.float64)
    add_inputs = [[Data(data=init_density, properties={"Usage": "Density Matrix"})]] * 2
    network.run(
        starting_nodes=[init_mol, coulomb_term, exchange_term],
        starting_inputs=starting_inputs + add_inputs,
    )

    output_edges = network.output_nodes[0].output_edges
    assert len(output_edges) == 1
    assert output_edges[0].data[0].properties["Data Type"] == float
    assert output_edges[0].data[0].properties["Usage"] == "Electronic Energy"


def test_scf_network():
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
    scf_itr = Node(process_model=GSE_SCFItr, network_type="OUTPUT", extend_dynamic=True)

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
        ],
        input_nodes=[init_mol],
        output_nodes=[scf_itr],
        broker=gen_broker(),
    )

    df = network.run(starting_nodes=[init_mol], starting_inputs=starting_inputs)

    output_edges = network.output_nodes[0].output_edges
    assert len(output_edges) == 5

    assert output_edges[0].data[0].properties["Data Type"] == float
    assert output_edges[0].data[0].properties["Usage"] == "Mean Field Energy"
    assert output_edges[1].data[0].properties["Data Type"] == list
    assert output_edges[1].data[0].properties["Usage"] == "Molecular Orbital Energies"
    assert output_edges[2].data[0].properties["Data Type"] == torch.Tensor
    assert output_edges[2].data[0].properties["Usage"] == "Molecular Orbital Coeff"
    assert output_edges[3].data[0].properties["Data Type"] == torch.Tensor
    assert output_edges[3].data[0].properties["Usage"] == "Fock Matrix"
    assert output_edges[4].data[0].properties["Data Type"] == list
    assert (
        output_edges[4].data[0].properties["Usage"] == "Molecular Orbital Occupancies"
    )


def test_init_ansatz_network():
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
        ],
        input_nodes=[init_mol],
        output_nodes=[init_ansatz],
        broker=gen_broker(),
    )

    add_inputs = [
        [
            Data(data="JW", properties={"Usage": "Qubit Mapping"}),
            Data(data="UCCSD", properties={"Usage": "Ansatz"}),
            Data(data="random", properties={"Usage": "Ansatz Params"}),
        ]
    ]
    df = network.run(
        starting_nodes=[init_mol, init_ansatz],
        starting_inputs=starting_inputs + add_inputs,
    )

    output_edges = network.output_nodes[0].output_edges
    assert len(output_edges) == 5

    assert output_edges[0].data[0].properties["Data Type"] == int
    assert output_edges[0].data[0].properties["Usage"] == "Num Parameters"
    assert output_edges[1].data[0].properties["Data Type"] == str
    assert output_edges[1].data[0].properties["Usage"] == "Ansatz Params"
    assert output_edges[2].data[0].properties["Data Type"] == int
    assert output_edges[2].data[0].properties["Usage"] == "Num Active Spin Orbitals"
    assert output_edges[3].data[0].properties["Data Type"] == int
    assert output_edges[3].data[0].properties["Usage"] == "Num Active Electrons"
    assert output_edges[4].data[0].properties["Data Type"] == str
    assert output_edges[4].data[0].properties["Usage"] == "Qubit Mapping"


def test_set_ansatz_params():
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
    set_ansatz = Node(process_model=GSE_SetAnsatzParams, network_type="OUTPUT")

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

    init_ansatz.insert_output_node(set_ansatz)

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
        ],
        input_nodes=[init_mol],
        output_nodes=[set_ansatz],
        broker=gen_broker(),
    )

    add_inputs = [
        [
            Data(data="JW", properties={"Usage": "Qubit Mapping"}),
            Data(data="UCCSD", properties={"Usage": "Ansatz"}),
            Data(data="random", properties={"Usage": "Ansatz Params"}),
        ]
    ]
    df = network.run(
        starting_nodes=[init_mol, init_ansatz],
        starting_inputs=starting_inputs + add_inputs,
    )

    output_edges = network.output_nodes[0].output_edges
    assert len(output_edges) == 1

    assert output_edges[0].data[0].properties["Data Type"] == np.ndarray
    assert output_edges[0].data[0].properties["Usage"] == "Ansatz Params"
    assert output_edges[0].data[0].properties["Status"] == "Active"


def test_construct_ref_circuit_network_JW():
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
    ref_circuit = Node(process_model=GSE_GenRefCircuit, network_type="OUTPUT")

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
            ref_circuit,
        ],
        input_nodes=[init_mol],
        output_nodes=[ref_circuit],
        broker=gen_broker(),
    )

    add_inputs = [
        [
            Data(data="JW", properties={"Usage": "Qubit Mapping"}),
            Data(data="UCCSD", properties={"Usage": "Ansatz"}),
            Data(data="random", properties={"Usage": "Ansatz Params"}),
        ]
    ]
    df = network.run(
        starting_nodes=[init_mol, init_ansatz],
        starting_inputs=starting_inputs + add_inputs,
    )

    output_edges = network.output_nodes[0].output_edges
    assert len(output_edges) == 1

    assert output_edges[0].data[0].properties["Data Type"] == QuantumCircuit
    assert output_edges[0].data[0].properties["Usage"] == "Reference Circuit"

    ref_circuit = output_edges[0].data[0].data
    assert len(ref_circuit.gate_set) == 1


@pytest.mark.slow
def test_construct_ref_circuit_network_BK():
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
    ref_circuit = Node(process_model=GSE_GenRefCircuit, network_type="OUTPUT")

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
            ref_circuit,
        ],
        input_nodes=[init_mol],
        output_nodes=[ref_circuit],
        broker=gen_broker(),
    )

    add_inputs = [
        [
            Data(data="BK", properties={"Usage": "Qubit Mapping"}),
            Data(data="UCCSD", properties={"Usage": "Ansatz"}),
            Data(data="random", properties={"Usage": "Ansatz Params"}),
        ]
    ]
    df = network.run(
        starting_nodes=[init_mol, init_ansatz],
        starting_inputs=starting_inputs + add_inputs,
    )

    output_edges = network.output_nodes[0].output_edges
    assert len(output_edges) == 1

    assert output_edges[0].data[0].properties["Data Type"] == QuantumCircuit
    assert output_edges[0].data[0].properties["Usage"] == "Reference Circuit"


@pytest.mark.slow
def test_construct_ref_circuit_network_SCBK():
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
    ref_circuit = Node(process_model=GSE_GenRefCircuit, network_type="OUTPUT")

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
            ref_circuit,
        ],
        input_nodes=[init_mol],
        output_nodes=[ref_circuit],
        broker=gen_broker(),
    )

    add_inputs = [
        [
            Data(data="SCBK", properties={"Usage": "Qubit Mapping"}),
            Data(data="UCCSD", properties={"Usage": "Ansatz"}),
            Data(data="random", properties={"Usage": "Ansatz Params"}),
        ]
    ]
    df = network.run(
        starting_nodes=[init_mol, init_ansatz],
        starting_inputs=starting_inputs + add_inputs,
    )

    output_edges = network.output_nodes[0].output_edges
    assert len(output_edges) == 1

    assert output_edges[0].data[0].properties["Data Type"] == QuantumCircuit
    assert output_edges[0].data[0].properties["Usage"] == "Reference Circuit"


def test_compute_ferm_op():
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
    set_ansatz = Node(process_model=GSE_SetAnsatzParams, network_type="NETWORK")
    compute_ferm_op = Node(process_model=GSE_ComputeFermionOp, network_type="OUTPUT")

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

    init_ansatz.insert_output_node(set_ansatz)
    init_ansatz.insert_output_node(compute_ferm_op)
    set_ansatz.insert_output_node(compute_ferm_op)

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
            compute_ferm_op,
        ],
        input_nodes=[init_mol],
        output_nodes=[compute_ferm_op],
        broker=gen_broker(),
    )

    add_inputs = [
        [
            Data(data="JW", properties={"Usage": "Qubit Mapping"}),
            Data(data="UCCSD", properties={"Usage": "Ansatz"}),
            Data(data="random", properties={"Usage": "Ansatz Params"}),
        ]
    ]
    df = network.run(
        starting_nodes=[init_mol, init_ansatz],
        starting_inputs=starting_inputs + add_inputs,
    )

    output_edges = network.output_nodes[0].output_edges
    assert len(output_edges) == 1

    assert output_edges[0].data[0].properties["Data Type"] == FermionOperator
    assert output_edges[0].data[0].properties["Usage"] == "Fermion Operator"


def test_compute_qubit_op():
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
    set_ansatz = Node(process_model=GSE_SetAnsatzParams, network_type="NETWORK")
    compute_ferm_op = Node(process_model=GSE_ComputeFermionOp, network_type="NETWORK")
    compute_qubit_op = Node(process_model=GSE_ComputeQubitOp, network_type="OUTPUT")

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

    init_ansatz.insert_output_node(set_ansatz)
    init_ansatz.insert_output_node(compute_ferm_op)
    set_ansatz.insert_output_node(compute_ferm_op)

    compute_ferm_op.insert_output_node(compute_qubit_op)
    init_ansatz.insert_output_node(compute_qubit_op)

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
            compute_ferm_op,
            compute_qubit_op,
        ],
        input_nodes=[init_mol],
        output_nodes=[compute_qubit_op],
        broker=gen_broker(),
    )

    add_inputs = [
        [
            Data(data="JW", properties={"Usage": "Qubit Mapping"}),
            Data(data="UCCSD", properties={"Usage": "Ansatz"}),
            Data(data="random", properties={"Usage": "Ansatz Params"}),
        ]
    ]
    df = network.run(
        starting_nodes=[init_mol, init_ansatz],
        starting_inputs=starting_inputs + add_inputs,
    )

    output_edges = network.output_nodes[0].output_edges
    assert len(output_edges) == 1

    assert output_edges[0].data[0].properties["Data Type"] == QubitOperator
    assert output_edges[0].data[0].properties["Usage"] == "Qubit Operator"
    assert output_edges[0].data[0].properties["Status"] == "Mapped"


def test_gen_ansatz_circuit():
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
        input_nodes=[init_mol],
        output_nodes=[gen_ansatz_circ],
        broker=gen_broker(),
    )

    add_inputs = [
        [
            Data(data="JW", properties={"Usage": "Qubit Mapping"}),
            Data(data="UCCSD", properties={"Usage": "Ansatz"}),
            Data(data="random", properties={"Usage": "Ansatz Params"}),
        ]
    ]
    df = network.run(
        starting_nodes=[init_mol, init_ansatz],
        starting_inputs=starting_inputs + add_inputs,
    )

    output_edges = network.output_nodes[0].output_edges
    assert len(output_edges) == 2

    assert output_edges[0].data[0].properties["Data Type"] == list
    assert output_edges[0].data[0].properties["Usage"] == "Ansatz Params"
    assert output_edges[1].data[0].properties["Data Type"] == QuantumCircuit
    assert output_edges[1].data[0].properties["Usage"] == "Ansatz Circuit"

    # network.generate_gantt_plot()

    # def test_run_quantum_circuit():
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
        process_model=GSE_GenAnsatzCircuit, network_type="NETWORK", extend_dynamic=True
    )
    # run_circuit = Node(process_model=GSE_RunCircuit, network_type="OUTPUT")

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
    # gen_ansatz_circ.insert_output_node(run_circuit)

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
        input_nodes=[init_mol],
        output_nodes=[gen_ansatz_circ],
        broker=gen_broker(),
    )

    add_inputs = [
        [
            Data(data="JW", properties={"Usage": "Qubit Mapping"}),
            Data(data="UCCSD", properties={"Usage": "Ansatz"}),
            Data(data="random", properties={"Usage": "Ansatz Params"}),
        ]
    ]

    df = network.run(
        starting_nodes=[init_mol, init_ansatz],
        starting_inputs=starting_inputs + add_inputs,
    )

    output_edges = network.output_nodes[0].output_edges
    assert len(output_edges) == 2

    assert output_edges[0].data[0].properties["Data Type"] == list
    assert output_edges[0].data[0].properties["Usage"] == "Ansatz Params"
    assert output_edges[1].data[0].properties["Data Type"] == QuantumCircuit
    assert output_edges[1].data[0].properties["Usage"] == "Ansatz Circuit"

    # network.generate_gantt_plot()


def test_run_ansatz_network_H4():
    symbols = ["H", "H", "H", "H"]
    coordinates = [
        [0.7071, 0.0, 0.0],
        [0.0, 0.7071, 0.0],
        [-1.0071, 0.0, 0.0],
        [0.0, -1.0071, 0.0],
    ]
    df, network = run_VQE_ansatz_network(
        symbols=symbols, coordinates=coordinates, broker=gen_broker()
    )

    output_edges = network.output_nodes[0].output_edges
    assert len(output_edges) == 2

    assert output_edges[0].data[0].properties["Data Type"] == list
    assert output_edges[0].data[0].properties["Usage"] == "Ansatz Params"
    assert output_edges[1].data[0].properties["Data Type"] == QuantumCircuit
    assert output_edges[1].data[0].properties["Usage"] == "Ansatz Circuit"

    # print dataframe
    # print(df.to_string())

    # generate gantt plot
    # network.generate_gantt_plot()


def test_run_ansatz_network_H2():
    symbols = ["H", "H"]
    coordinates = [
        [0.0, 0.0, 0.0],
        [0.7414, 0.0, 0.0],
    ]
    df, network = run_VQE_ansatz_network(
        symbols=symbols, coordinates=coordinates, broker=gen_broker(), simulate=False
    )

    output_edges = network.output_nodes[0].output_edges
    assert len(output_edges) == 2

    assert output_edges[0].data[0].properties["Data Type"] == list
    assert output_edges[0].data[0].properties["Usage"] == "Ansatz Params"
    assert output_edges[1].data[0].properties["Data Type"] == QuantumCircuit
    assert output_edges[1].data[0].properties["Usage"] == "Ansatz Circuit"

    # print dataframe
    # print(df.to_string())
    # assert 1 == 0

    # generate gantt plot
    # network.generate_gantt_plot()


# Added stuff for demo:

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
    # coordinates = [
    #     [0.7071, 0.0, 0.0],
    #     [0.0, 0.7071, 0.0],
    #     [-1.0071, 0.0, 0.0],
    #     [0.0, -1.0071, 0.0],
    # ]
    # coordinates = [
    #               [0.0, 0.0, 0.0],
    #               [0.74, 0.0, 0.0],
    # ]
    # coordinates = [
    #               [0.0, 0.0, 0.0],
    #               [1.21, 0.0, 0.0],
    # ]
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
    #assert len(output_edges) == 1
    #assert output_edges[0].data[0].properties["Data Type"] == float
    #assert output_edges[0].data[0].properties["Usage"] == "Electronic Energy"

    #network.generate_gantt_plot()

    return network, df

def generate_electronic_energy_network2(symbols, coordinates, basis, charge, geometry_model, qubit_mapping, ansatz, ansatz_params, broker=None, simulate=True):
    # symbols = ["H", "H", "H", "H"]
    # coordinates = [
    #     [0.7071, 0.0, 0.0],
    #     [0.0, 0.7071, 0.0],
    #     [-1.0071, 0.0, 0.0],
    #     [0.0, -1.0071, 0.0],
    # ]
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
    #assert len(output_edges) == 2

    #assert output_edges[0].data[0].properties["Data Type"] == list
    #assert output_edges[0].data[0].properties["Usage"] == "Ansatz Params"
    #assert output_edges[1].data[0].properties["Data Type"] == QuantumCircuit
    #assert output_edges[1].data[0].properties["Usage"] == "Ansatz Circuit"

    # print dataframe
    print(df.to_string())

    return network, df