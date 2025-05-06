import pytest
from workflow.chemistry_ingestion.molecule import QuantizedMolecule,Molecule, BasisFunction, mol_basis_data, atom_basis_data

from workflow.chemistry_ingestion.interfaces import (
    IntegralSolver, 
    compute_core_matrix, 
    compute_attraction_integral_matrix, 
    compute_kinetic_integral_matrix,
    compute_repulsion_tensor,
    compute_overlap_matrix,
    electron_repulsion,
    gaussian_overlap
)

import pennylane.numpy as np
import pennylane as qml

import torch

import os
# Suppress OpenMP warnings
os.environ["OMP_NUM_THREADS"] = "1"  # Or a higher number suitable for your system
os.environ["KMP_WARNINGS"] = "0"


def test_overlap_matrix():
    H2 = [
        ("H", (0., 0., 0.)),
        ("H", (0., 0., 1.))
    ]
    mol = Molecule(xyz=H2, q=0,spin=0)
    basis_functions = mol.basis_set
    basis_exponents = [bf.params[0] for bf in basis_functions]
    basis_coefficients = [bf.params[1] for bf in basis_functions]
    atomic_coordinates = [bf.params[2] for bf in basis_functions]

    overlap_mat = compute_overlap_matrix(
        basis_functions, basis_exponents, basis_coefficients, atomic_coordinates
    )
    
    expected_overlap_matrix = np.array([[1.0, 0.7965883009074122], [0.7965883009074122, 1.0]])
    assert np.allclose(overlap_mat, expected_overlap_matrix)

def test_repulsion_tensor():
    H2 = [
        ("H", (0., 0., 0.)),
        ("H", (0., 0., 1.))
    ]
    mol = Molecule(xyz=H2, q=0,spin=0)
    
    rep_tensor = compute_repulsion_tensor(mol.basis_set)
    
    
    expected_rep_tensor = np.array(
                    [
                        [
                            [[0.77460594, 0.56886157], [0.56886157, 0.65017755]],
                            [[0.56886157, 0.45590169], [0.45590169, 0.56886157]],
                        ],
                        [
                            [[0.56886157, 0.45590169], [0.45590169, 0.56886157]],
                            [[0.65017755, 0.56886157], [0.56886157, 0.77460594]],
                        ],
                    ]
                )
    assert np.allclose(rep_tensor,expected_rep_tensor)

def test_electron_repulsion():
    la, lb, lc, ld = (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)
    ra, rb, rc, rd = (
        torch.tensor([0.0, 0.0, 0.0]),
        torch.tensor([0.0, 0.0, 0.0]),
        torch.tensor([0.0, 0.0, 0.0]),
        torch.tensor([0.0, 0.0, 0.0]),
    )
    alpha, beta, gamma, delta = (
        torch.tensor(1.0),
        torch.tensor(1.0),
        torch.tensor(1.0),
        torch.tensor(1.0),
    )

    result = electron_repulsion(la, lb, lc, ld, ra, rb, rc, rd, alpha, beta, gamma, delta)
    
    qml_output = qml.qchem.electron_repulsion(la, lb, lc, ld, ra, rb, rc, rd, alpha, beta, gamma, delta)
   
    assert np.allclose(result,qml_output)

def test_gaussian_overlap():
    la, lb, ra, rb, alpha, beta = (
        (0, 0, 0),
        (0, 0, 0),
        torch.tensor([0.0, 0.0, 0.0]),
        torch.tensor([0.0, 0.0, 0.0]),
        torch.tensor([np.pi / 2]),
        torch.tensor([np.pi / 2]),
    )
    go =gaussian_overlap(la, lb, ra, rb, alpha,beta)
    assert np.allclose(go,[1.0])

    la, lb, ra, rb, alpha, beta =  (
                (1, 0, 0),
                (0, 0, 1),
                torch.tensor([0.0, 0.0, 0.0]),
                torch.tensor([0.0, 0.0, 0.0]),
                torch.tensor([6.46480325]),
                torch.tensor([6.46480325]),
                
            )
    
    go =gaussian_overlap(la, lb, ra, rb, alpha,beta)
    assert np.allclose(go,[0.0])
    
def test_H_core_matrices():
    H2 = [
        ("H", (0., 0., 0.)),
        ("H", (0., 0., 1.))
    ]
    mol = Molecule(xyz=H2, q=0,spin=0)
    
    v = compute_attraction_integral_matrix(mol.basis_set, mol.nuclear_charges, mol.coordinates)
    v_ref = np.array(
                    [
                        [-2.03852075, -1.6024171],
                        [-1.6024171, -2.03852075],
                    ]
                )
    assert np.allclose(v, v_ref)

    t = compute_kinetic_integral_matrix(mol.basis_set)
    t_ref = np.array(
                    [
                        [0.7600318862777408, 0.38325367405372557],
                        [0.38325367405372557, 0.7600318862777408],
                    ]
                )
    assert np.allclose(t, t_ref)
    
    core = compute_core_matrix(mol.basis_set, mol.nuclear_charges, mol.coordinates)

   
    core_ref = np.array(
        [
            [-1.27848869, -1.21916299],
            [-1.21916299, -1.27848869],
        ]
    )

    assert np.allclose(core, core_ref)

if __name__ == "__main__":
    test_H_core_matrices()
    test_electron_repulsion()
    test_gaussian_overlap()
    test_overlap_matrix()
    test_repulsion_tensor()
    