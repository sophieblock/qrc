import numpy as np
from openfermion.transforms import (
    jordan_wigner, bravyi_kitaev, bravyi_kitaev_tree, reorder
)
from openfermion.transforms import bravyi_kitaev as openfermion_bravyi_kitaev

from openfermion.utils import count_qubits, up_then_down as up_then_down_order
from openfermion.ops import FermionOperator, QubitOperator
import copy
import warnings
# Available mappings
available_mappings = {"JW", "BK","SCBK"}
def occupation_vector_to_fermion_operator(occupation_vector):
    """
    Convert an occupation vector into a FermionOperator.
    Args:
        occupation_vector (np.ndarray): Binary occupation vector (e.g., [1, 0, 1, ...])
    Returns:
        FermionOperator: Corresponding FermionOperator representing the state.
    """
    fermion_op = FermionOperator()
    for idx, occ in enumerate(occupation_vector):
        if occ == 1:
            fermion_op += FermionOperator(f"{idx}^ {idx}", 1.0)  # Creation followed by annihilation
    return fermion_op
def bravyi_kitaev(fermion_operator, n_qubits):
    """Transform FermionOperator to QubitOperator using Bravyi-Kitaev mapping."""
    if not isinstance(n_qubits, int):
        raise TypeError("Number of qubits (n_qubits) must be integer type.")
    if n_qubits < count_qubits(fermion_operator):
        raise ValueError("Invalid (too few) number of qubits (n_qubits).")
    qubit_operator = openfermion_bravyi_kitaev(fermion_operator, n_qubits=n_qubits)

    return qubit_operator

def fermion_to_qubit_mapping(fermion_operator, mapping, n_spinorbitals=None, n_electrons=None, up_then_down=False, spin=0):
    """Map a fermionic operator to a qubit operator using a specific mapping."""
    if not isinstance(fermion_operator, FermionOperator):
        raise TypeError("Input must be a FermionOperator.")
    if mapping.upper() not in available_mappings:
        raise ValueError(f"Invalid mapping: {mapping}. Supported mappings: {available_mappings}")
    if mapping.upper() in {"BK", "SCBK"} and n_spinorbitals is None:
        raise ValueError(f"{mapping.upper()} requires n_spinorbitals.")

    if mapping.upper() == "JW":
        return jordan_wigner(fermion_operator)
    elif mapping.upper() == "BK":
        return bravyi_kitaev(fermion_operator, n_qubits=n_spinorbitals)
    elif mapping.upper() == "SCBK":
        if n_electrons is None:
            raise ValueError("SCBK requires n_electrons.")
        return symmetry_conserving_bravyi_kitaev(
            fermion_operator, n_spinorbitals, n_electrons, up_then_down, spin
        )

from openfermion import FermionOperator as ofFermionOperator
def symmetry_conserving_bravyi_kitaev(fermion_operator, n_spinorbitals, n_electrons, up_then_down=False, spin=0):
    """Apply symmetry-conserving BK transformation to a fermionic operator."""
    # Function logic remains the same
    if not isinstance(fermion_operator, ofFermionOperator):
        raise ValueError("Supplied operator should be an instance "
                         "of openfermion FermionOperator class.")
    if type(n_spinorbitals) is not int:
        raise ValueError("Number of spin-orbitals should be an integer.")
    if type(n_electrons) is not int:
        raise ValueError("Number of electrons should be an integer.")
    if n_spinorbitals < count_qubits(fermion_operator):
        raise ValueError("Number of spin-orbitals is too small for FermionOperator input.")
    # Check that the input operator is suitable for application of scBK
    check_operator(fermion_operator, num_orbitals=(n_spinorbitals//2), up_then_down=up_then_down)

    # If necessary, arrange spins up then down, then BK map to qubit Hamiltonian.
    if not up_then_down:
        fermion_operator = reorder(fermion_operator, up_then_down_order, num_modes=n_spinorbitals)
    qubit_operator = bravyi_kitaev_tree(fermion_operator, n_qubits=n_spinorbitals)
    qubit_operator.compress()

    n_alpha = n_electrons//2 + spin//2 + (n_electrons % 2)

    # Allocates the parity factors for the orbitals as in arXiv:1704.08213.
    parity_final_orb = (-1)**n_electrons
    parity_middle_orb = (-1)**n_alpha

    # Removes the final qubit, then the middle qubit.
    qubit_operator = edit_operator_for_spin(qubit_operator,
                                            n_spinorbitals,
                                            parity_final_orb)
    qubit_operator = edit_operator_for_spin(qubit_operator,
                                            n_spinorbitals/2,
                                            parity_middle_orb)

    # We remove the N/2-th and N-th qubit from the register.
    to_prune = (n_spinorbitals//2 - 1, n_spinorbitals - 1)
    qubit_operator = prune_unused_indices(qubit_operator, prune_indices=to_prune, n_qubits=n_spinorbitals)

    return qubit_operator


def edit_operator_for_spin(qubit_operator, spin_orbital, orbital_parity):
    """Removes the Z terms acting on the orbital from the operator. For qubits
    to be tapered out, the action of Z-operators in operator terms are reduced
    to the associated eigenvalues. This simply corresponds to multiplying term
    coefficients by the related eigenvalue +/-1.

    Args:
        qubit_operator (QubitOperator): input operator.
        spin_orbital (int): index of qubit encoding (spin/occupation) parity.
        orbital_parity (int): plus/minus one, parity of eigenvalue.

    Returns:
        QubitOperator: updated operator, with relevant coefficients multiplied
            by +/-1.
    """
    new_qubit_dict = {}
    for term, coefficient in qubit_operator.terms.items():
        # If Z acts on the specified orbital, precompute its effect and
        # remove it from the Hamiltonian.
        if (spin_orbital - 1, "Z") in term:
            new_coefficient = coefficient*orbital_parity
            new_term = tuple(i for i in term if i != (spin_orbital - 1, "Z"))
            # Make sure to combine terms comprised of the same operators.
            if new_qubit_dict.get(new_term) is None:
                new_qubit_dict[new_term] = new_coefficient
            else:
                old_coefficient = new_qubit_dict.get(new_term)
                new_qubit_dict[new_term] = new_coefficient + old_coefficient
        else:
            # Make sure to combine terms comprised of the same operators.
            if new_qubit_dict.get(term) is None:
                new_qubit_dict[term] = coefficient
            else:
                old_coefficient = new_qubit_dict.get(term)
                new_qubit_dict[term] = coefficient + old_coefficient

    qubit_operator.terms = new_qubit_dict
    qubit_operator.compress()

    return qubit_operator


def prune_unused_indices(qubit_operator, prune_indices, n_qubits):
    """Rewritten from openfermion implementation. This uses the number of qubits,
    rather than the operator itself to specify the number of qubits relevant to
    the problem. This is especially important for, e.g. terms in the ansatz
    which may not individually pertain to all qubits in the problem.

    Remove indices that do not appear in any terms.

    Indices will be renumbered such that if an index i does not appear in any
    terms, then the next largest index that appears in at least one term will be
    renumbered to i.

    Args:
        qubit_operator (QubitOperator): input operator.
        prune_indices (tuple of int): indices to be removed from qubit register.
        n_qubits (int): number of qubits in register.

    Returns:
        QubitOperator: output operator, with designated qubit indices excised.
    """

    indices = np.linspace(0, n_qubits - 1, n_qubits, dtype=int)
    indices = np.delete(indices, prune_indices)

    # Construct a dict that maps the old indices to new ones
    index_map = {}
    for index in enumerate(indices):
        index_map[index[1]] = index[0]

    new_operator = copy.deepcopy(qubit_operator)
    new_operator.terms.clear()

    # Replace the indices in the terms with the new indices
    for term in qubit_operator.terms:
        new_term = [(index_map[op[0]], op[1]) for op in term]
        new_operator.terms[tuple(new_term)] = qubit_operator.terms[term]

    return new_operator

def check_operator(fermion_operator, num_orbitals=None, up_then_down=False):
    """Check if the fermionic operator conserves spin and occupation parity."""
    if up_then_down and (num_orbitals is None):
        raise ValueError("Up then down spin ordering requires number of modes specified.")
    for term in fermion_operator.terms:
        number_change = 0
        spin_change = 0
        for index, action in term:
            number_change += 2*action - 1
            if up_then_down:
                spin_change += (2*action - 1)*(-2*(index // num_orbitals) + 1)*0.5

            else:
                spin_change += (2*action - 1)*(-2*(index % 2) + 1)*0.5
        if number_change % 2 != 0:
            raise ValueError("Invalid operator: input fermion operator does not conserve occupation parity.")
        if spin_change % 2 != 0:
            raise ValueError("Invalid operator: input fermion operator does not conserve spin parity.")


"""
Statevector mapping helpers
"""

from openfermion.transforms import bravyi_kitaev_code
import cirq

def do_bk_transform(vector):
    """Apply Bravyi-Kitaev transformation to fermion occupation vector.
    Currently, simple wrapper on openfermion tools.

    Args:
        vector (numpy array of int): fermion occupation vector.

    Returns:
        numpy array of int: qubit-encoded occupation vector.
    """
    mat = bravyi_kitaev_code(len(vector)).encoder.toarray()
    vector_bk = np.mod(np.dot(mat, vector), 2)
    return vector_bk

from openfermion.transforms.opconversions.bravyi_kitaev_tree import _transform_ladder_operator, FenwickTree
def do_scbk_transform(vector, n_spinorbitals):
    """Instantiate qubit vector for symmetry-conserving Bravyi-Kitaev
    Generate Majorana mode for each occupied spin-orbital and apply X gate
    to each non-Z operator in the Pauli word.

    Args:
        vector (numpy array of int): fermion occupation vector.
        n_spinorbitals (int): number of qubits in register.

    Returns:
        numpy array of int: qubit-encoded occupation vector.
    """

    fenwick_tree = FenwickTree(n_spinorbitals)
    # Generate QubitOperator that represents excitation through Majorana mode (a_i^+ - a_) for each occupied orbital
    qu_op = QubitOperator((), 1)
    for ind, oc in enumerate(vector):
        if oc == 1:
            qu_op *= (_transform_ladder_operator((ind, 1), fenwick_tree) - _transform_ladder_operator((ind, 0), fenwick_tree))

    # Include all qubits that have Pauli operator X or Y acting on them in new occupation vector.
    vector_bk = np.zeros(n_spinorbitals)
    active_qus = [i for i, j in next(iter(qu_op.terms)) if j != "Z"]
    for q in active_qus:
        vector_bk[q] = 1

    # Delete n_spinorbital and last qubit as is done for the scBK transform.
    vector_bk = np.delete(vector_bk, n_spinorbitals-1)
    vector_scbk = np.delete(vector_bk, n_spinorbitals//2-1)
    return vector_scbk
def vector_to_circuit(vector):
    """Translate occupation vector into a Cirq circuit."""
    n_qubits = len(vector)
    qubits = [cirq.LineQubit(i) for i in range(n_qubits)]

    # Initialize a circuit
    circuit = cirq.Circuit()

    # Apply X gate for each qubit based on occupation vector
    for i, occupation in enumerate(vector):
        if occupation:
            circuit.append(cirq.X(qubits[i]))

    return circuit

def get_mapped_vector(vector, mapping, up_then_down=False):
    if up_then_down:
        vector = np.concatenate((vector[::2], vector[1::2]))
    
    if mapping.upper() == "JW":
        return vector  # Jordan-Wigner encoding
    elif mapping.upper() == "BK":
        return do_bk_transform(vector)
    elif mapping.upper() == "SCBK":
        if not up_then_down:
            # warnings.warn("Symmetry-conserving Bravyi-Kitaev enforces all spin-up followed by all spin-down ordering.", RuntimeWarning)
            vector = np.concatenate((vector[::2], vector[1::2]))
        return do_scbk_transform(vector, len(vector))
    else:
        raise ValueError(f"Unsupported mapping: {mapping}. Available mappings: {available_mappings}.")
 
def get_vector(n_spinorbitals, n_electrons, mapping, up_then_down=False, spin=None):
    if mapping.upper() not in available_mappings:
        raise ValueError(f"Invalid mapping selection, available mappings: {available_mappings}. Got {mapping}")

    vector = np.zeros(n_spinorbitals, dtype=int)
    if spin:
        n_alpha = n_electrons // 2 + spin // 2 + (n_electrons % 2)
        n_beta = n_electrons // 2 - spin // 2
        vector[0:2 * n_alpha:2] = 1
        vector[1:2 * n_beta + 1:2] = 1
    else:
        vector[:n_electrons] = 1
    return get_mapped_vector(vector, mapping, up_then_down)

def get_reference_circuit(n_spinorbitals, n_electrons, mapping, up_then_down=False, spin=None):
    """Build the HF state preparation circuit."""
    vector = get_vector(n_spinorbitals, n_electrons, mapping, up_then_down=up_then_down, spin=spin)
    return vector_to_circuit(vector)
