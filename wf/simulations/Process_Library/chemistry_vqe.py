from typing import List
import torch
import itertools
import numpy as np
import random
from pennylane.qchem.integrals import primitive_norm
from pennylane.qchem.basis_data import atomic_numbers
import pennylane.numpy as pnp
from openfermion.transforms import (
    jordan_wigner,
    bravyi_kitaev,
    symmetry_conserving_bravyi_kitaev,
)
from openfermion.ops import FermionOperator, QubitOperator
from openfermion.circuits import uccsd_singlet_generator

from qrew.simulation.refactor.data import Data
from qrew.simulation.refactor.process import ClassicalProcess, QuantumProcess
from qrew.simulation.refactor.graph import Node, Network

# from qrew.simulation.refactor.
from qrew.simulation.refactor.resources.classical_resources import ClassicalResource
from qrew.simulation.refactor.resources.quantum_resources import QuantumResource
from qrew.simulation.refactor.quantum import QuantumCircuit

# from qrew.simulation.refactor.quantum_gates import X
from qrew.simulation.refactor.quantum_gates import X, H, CZPow, CX, RX, RZ
from qrew.chemistry_ingestion.molecule import (
    atom_basis_data,
    BasisFunction,
    ANGSTROM_TO_BOHR,
)
from qrew.chemistry_ingestion.fermion_to_qubit_utils import (jordan_wigner,
                                                                 bravyi_kitaev,
                                                                 fermion_to_qubit_mapping,
                                                                 available_mappings,
                                                                 do_bk_transform,
                                                                 occupation_vector_to_fermion_operator)
from qrew.chemistry_ingestion.interfaces import (
    overlap_matrix,
    compute_repulsion_tensor,
    compute_core_matrix,
    nuclear_energy,
)

def expansion_flop(la, lb, ra, rb, alpha, beta, t, out_shape=None):
    if la == lb == t == 0:
        return 1  # Single exponential computation
    if t < 0 or t > (la + lb):
        return 0  # Invalid case, no FLOPs
    # Recursive case
    if lb == 0:
        
        assert la > 0, "Invalid recursive call for `expansion`: `la` must be greater than 0."
        return (
            expansion_flop(la - 1, lb, ra, rb, alpha, beta, t - 1)
            + expansion_flop(la - 1, lb, ra, rb, alpha, beta, t)
            + expansion_flop(la - 1, lb, ra, rb, alpha, beta, t + 1)
            + 3  # FLOPs for multiplication/division in the recursive formula
        )
    else:
       
        assert lb > 0, "Invalid recursive call for `expansion`: `lb` must be greater than 0."
        return (
            expansion_flop(la, lb - 1, ra, rb, alpha, beta, t - 1)
            + expansion_flop(la, lb - 1, ra, rb, alpha, beta, t)
            + expansion_flop(la, lb - 1, ra, rb, alpha, beta, t + 1)
            + 3  # FLOPs for multiplication/division in the recursive formula
        )
def gaussian_overlap_flop(la, lb, ra, rb, alpha, beta, out_val=None):

    flops = 0
    assert la.shape == lb.shape == ra.shape == rb.shape, (
        "Inputs `la`, `lb`, `ra`, and `rb` must have the same shape."
    )
    assert alpha.shape == beta.shape, "Inputs `alpha` and `beta` must have the same shape."

    for i in range(3):  # 3 Cartesian dimensions
        # FLOPs for sqrt, division, and expansion (theoretical estimation)
        expansion_flops = expansion_flop(
            la[i].item(), lb[i].item(), ra[i].item(), rb[i].item(), alpha, beta, 0
        )
        # expansion_flops = torch.ops.molecule_ops.expansion(
        #     la[i].item(), lb[i].item(), ra[i].item(), rb[i].item(), alpha, beta, 0
        # )
        flops += 1 + 1 + expansion_flops  # 1 for sqrt, 1 for division, plus expansion FLOPs
    return flops

class GSE_InitMolecule(ClassicalProcess):
    def __init__(
        self,
        inputs: list[Data] = None,
        expected_input_properties=None,
        required_resources=None,
        output_properties=None,
    ):
        if inputs != None:
            self.input_data = {}
            for data in inputs:
                self.input_data[data.properties["Usage"]] = data.data

        super().__init__(
            inputs,
            expected_input_properties,
            required_resources,
            output_properties,
            dynamic=True,
        )

    def set_expected_input_properties(self):
        self.expected_input_properties = [
            {"Data Type": str, "Usage": "Basis"},
            {"Data Type": list, "Usage": "Atomic Symbols"},
            {"Data Type": list, "Usage": "Coordinates"},
            {"Data Type": int, "Usage": "Total Charge"},
            {"Data Type": str, "Usage": "Geometry Model"},
        ]

    def set_required_resources(self):
        self.required_resources = ClassicalResource(
            memory=42,
            num_cores=1,
        )

    def set_output_properties(self):
        pass

    def compute_flop_count(self):
        return 10

    def validate_data(self) -> bool:
        conditions = (self.__valid_geometry_model(),)
        return all(condition for condition in conditions)

    def __valid_geometry_model(self):
        supported_models = ["UHF", "RHF"]
        return self.input_data["Geometry Model"] in supported_models

    def update(self):
        basis = self.input_data["Basis"]
        symbols = self.input_data["Atomic Symbols"]
        coordinates = torch.from_numpy(
            pnp.array(self.input_data["Coordinates"], requires_grad=False)
        )
        nuclear_charges = [atomic_numbers[symbol] for symbol in symbols]
        n_electrons = sum(nuclear_charges) - self.input_data["Total Charge"]

        self.results = {
            "Atomic Data": [(basis, symbol) for symbol in symbols],
            "Coordinates": coordinates,
            "Nuclear Charges": nuclear_charges,
            "Num Electrons": n_electrons,
            "Geometry Model": self.input_data["Geometry Model"],
        }

    def generate_output(self) -> list[Data]:
        output = []
        for key, val in self.results.items():
            if key == "Atomic Data":
                for idx in range(len(self.results["Atomic Data"])):
                    output.append(
                        Data(
                            data=self.results["Atomic Data"][idx],
                            properties={"Usage": "Atomic Data", "Index": idx},
                        )
                    )
            else:
                output.append(Data(data=val, properties={"Usage": key}))

        return tuple(output)

    def extend_network(self) -> Network:
        num_atoms = len(self.input_data["Atomic Symbols"])
        extract_basis_nodes = [
            Node(process_model=GSE_ExtractBasis, network_type="INPUT", index=idx)
            for idx in range(num_atoms)
        ]
        gen_basis_data = Node(
            process_model=GSE_GenBasisData, network_type="OUTPUT", num_atoms=num_atoms
        )

        for node in extract_basis_nodes:
            node.insert_output_node(gen_basis_data)

        return Network(
            nodes=extract_basis_nodes + [gen_basis_data],
            input_nodes=extract_basis_nodes,
            output_nodes=[gen_basis_data],
        )


class GSE_ExtractBasis(ClassicalProcess):
    def __init__(
        self,
        inputs: list[Data] = None,
        expected_input_properties=None,
        required_resources=None,
        output_properties=None,
        index=None,
    ):
        if inputs != None:
            self.input_data = {}
            for data in inputs:
                self.input_data[data.properties["Usage"]] = data.data
        self.index = index
        super().__init__(
            inputs,
            expected_input_properties,
            required_resources,
            output_properties,
        )

    def set_expected_input_properties(self):
        self.expected_input_properties = [
            {"Data Type": tuple, "Usage": "Atomic Data", "Index": self.index},
        ]

    def set_required_resources(self):
        self.required_resources = ClassicalResource(
            memory=42,
            num_cores=1,
        )

    def set_output_properties(self):
        pass

    def compute_flop_count(self):
        return 10

    def validate_data(self) -> bool:
        data = self.input_data["Atomic Data"]
        conditions = (self.__valid_basis(data[0]), self.__valid_symbol(data[1]))

        return all(condition for condition in conditions)

    def __valid_basis(self, basis):
        valid_basis = ["sto-3g"]

        return True if basis in valid_basis else False

    def __valid_symbol(self, symbol):
        valid_atomic_symbols = ["H", "C", "O"]

        return True if symbol in valid_atomic_symbols else False

    def update(self):
        data = self.input_data["Atomic Data"]
        self.results = atom_basis_data(data[0], data[1])

    def generate_output(self) -> list[Data]:

        return (
            Data(
                data=self.results,
                properties={"Usage": "Atomic Basis Data", "Index": self.index},
            ),
        )


class GSE_GenBasisData(ClassicalProcess):
    def __init__(
        self,
        inputs: list[Data] = None,
        expected_input_properties=None,
        required_resources=None,
        output_properties=None,
        num_atoms=None,
    ):
        if inputs != None:
            self.input_data = {}
            for data in inputs:
                self.input_data[data.properties["Index"]] = data.data
        self.num_atoms = num_atoms

        super().__init__(
            inputs,
            expected_input_properties,
            required_resources,
            output_properties,
        )

    def set_expected_input_properties(self):
        self.expected_input_properties = [
            {"Data Type": list, "Usage": "Atomic Basis Data", "Index": idx}
            for idx in range(self.num_atoms)
        ]

    def set_required_resources(self):
        self.required_resources = ClassicalResource(
            memory=42,
            num_cores=1,
        )

    def set_output_properties(self):
        pass

    def compute_flop_count(self):
        return 10

    def validate_data(self) -> bool:

        return True

    def update(self):
        basis_data = [self.input_data[idx] for idx in range(self.num_atoms)]
        num_basis = [len(basis) for basis in basis_data]
        self.results = (num_basis, [data[0] for data in basis_data])

    def generate_output(self) -> list[Data]:

        return (
            Data(
                data=self.results[0],
                properties={"Usage": "Num Basis Per Atom"},
            ),
            Data(data=self.results[1], properties={"Usage": "Basis Data"}),
        )


## TODO: Change BasisFunction to groups of Angular Momen, Alpha, Coeff, etc..
class GSE_InitBasisSet(ClassicalProcess):
    def __init__(
        self,
        inputs: list[Data] = None,
        expected_input_properties=None,
        required_resources=None,
        output_properties=None,
    ):
        if inputs != None:
            self.input_data = {}
            for data in inputs:
                self.input_data[data.properties["Usage"]] = data.data

        super().__init__(
            inputs,
            expected_input_properties,
            required_resources,
            output_properties,
        )

    def set_expected_input_properties(self):
        self.expected_input_properties = [
            {"Data Type": list, "Usage": "Num Basis Per Atom"},
            {"Data Type": list, "Usage": "Basis Data"},
            {"Data Type": torch.Tensor, "Usage": "Coordinates"},
        ]

    def set_required_resources(self):
        self.required_resources = ClassicalResource(
            memory=42,
            num_cores=1,
        )

    def set_output_properties(self):
        pass

    def compute_flop_count(self):
        return 10

    def validate_data(self) -> bool:

        return True

    def update(self):
        n_basis_per_atom = self.input_data["Num Basis Per Atom"]
        basis_data = self.input_data["Basis Data"]
        coordinates = self.input_data["Coordinates"]

        angular_momentum = [i[0] for i in basis_data]
        alpha = list(map(torch.tensor, [i[1] for i in basis_data]))
        coeff = list(map(torch.tensor, [i[2] for i in basis_data]))

        # normalize
        coeff = [
            (c * primitive_norm(angular_momentum[i], alpha[i]))
            for i, c in enumerate(coeff)
        ]
        coordinates = list(
            itertools.chain(
                *[
                    [coordinates[i]] * n_basis_per_atom[i]
                    for i in range(len(n_basis_per_atom))
                ]
            )
        )

        self.result = [
            BasisFunction(angular_momentum[i], alpha[i], coeff[i], coordinates[i])
            for i in range(len(angular_momentum))
        ]

    def generate_output(self) -> list[Data]:

        return (
            Data(
                data=self.result,
                properties={"Usage": "Basis Set"},
            ),
        )


class GSE_ComputeOverlap(ClassicalProcess):
    def __init__(
        self,
        inputs: list[Data] = None,
        expected_input_properties=None,
        required_resources=None,
        output_properties=None,
    ):
        if inputs != None:
            self.input_data = {}
            for data in inputs:
                self.input_data[data.properties["Usage"]] = data.data

        super().__init__(
            inputs,
            expected_input_properties,
            required_resources,
            output_properties,
        )

    def set_expected_input_properties(self):
        self.expected_input_properties = [{"Data Type": list, "Usage": "Basis Set"}]

    def set_required_resources(self):
        self.required_resources = ClassicalResource(
            memory=len(self.input_data["Basis Set"])**2 * 8,
            num_cores=1,
        )

    def set_output_properties(self):
        pass

    def compute_flop_count(self):
 
        num_basis = len(self.input_data["Basis Set"])
        num_pairs = num_basis * (num_basis - 1) // 2  # Unique pairs
        flops_per_pair = 9  # Simplified estimate for overlap computation
        return num_pairs * flops_per_pair
    
    def validate_data(self) -> bool:

        return True

    def update(self):
        basis_exponents = [bf.params[0] for bf in self.input_data["Basis Set"]]
        basis_coefficients = [bf.params[1] for bf in self.input_data["Basis Set"]]
        atomic_coordinates = [bf.params[2] for bf in self.input_data["Basis Set"]]

        self.result = overlap_matrix(
            self.input_data["Basis Set"],
            basis_exponents,
            basis_coefficients,
            atomic_coordinates,
        )

    def generate_output(self) -> list[Data]:

        return (
            Data(
                data=self.result,
                properties={"Usage": "Overlap Matrix"},
            ),
        )


class GSE_ComputeRepulsion(ClassicalProcess):
    def __init__(
        self,
        inputs: list[Data] = None,
        expected_input_properties=None,
        required_resources=None,
        output_properties=None,
    ):
        if inputs != None:
            self.input_data = {}
            for data in inputs:
                self.input_data[data.properties["Usage"]] = data.data

        super().__init__(
            inputs,
            expected_input_properties,
            required_resources,
            output_properties,
        )

    def set_expected_input_properties(self):
        self.expected_input_properties = [{"Data Type": list, "Usage": "Basis Set"}]

    def set_required_resources(self):
        """
        Memory (bytes) = N^4 * 8, where:
            - N is the number of basis functions
            - 8 bytes per double-precision value (float64)
        
        Source: https://arxiv.org/pdf/1910.02987
        """
        basis_set = self.input_data["Basis Set"]
        num_basis = len(basis_set)

        memory_bytes = (num_basis**4) * 8
        self.required_resources = ClassicalResource(memory=memory_bytes, num_cores=1)

    def set_output_properties(self):
        pass

    def compute_flop_count(self):
        """
        N^4 * 50 * P, where:
            - N is the number of basis functions
            - 50 is an approximate constant for 4-center integrals (Obara-Saika recursion)
            - P is the average number of primitives per basis function

        """
        basis_set = self.input_data["Basis Set"]
        num_basis = len(basis_set)
        num_primitives = sum(len(bf.params[0]) for bf in basis_set)

        return num_basis**4 * 50 * num_primitives

    def validate_data(self) -> bool:

        return True

    def update(self):
        self.result = compute_repulsion_tensor(self.input_data["Basis Set"])

    def generate_output(self) -> list[Data]:

        return (
            Data(
                data=self.result,
                properties={"Usage": "Repulsion Tensor"},
            ),
        )


class GSE_ComputeCore(ClassicalProcess):
    def __init__(
        self,
        inputs: list[Data] = None,
        expected_input_properties=None,
        required_resources=None,
        output_properties=None,
    ):
        if inputs != None:
            self.input_data = {}
            for data in inputs:
                self.input_data[data.properties["Usage"]] = data.data

        super().__init__(
            inputs,
            expected_input_properties,
            required_resources,
            output_properties,
        )

    def set_expected_input_properties(self):
        self.expected_input_properties = [
            {"Data Type": list, "Usage": "Basis Set"},
            {"Data Type": torch.Tensor, "Usage": "Coordinates"},
            {"Data Type": list, "Usage": "Nuclear Charges"},
        ]

    def set_required_resources(self):
        """
    
        Memory (bytes) = N^2 * 8, where:
        - N is the number of basis functions
        - 8 bytes per double-precision value (float64)
        
        Source: https://arxiv.org/pdf/1910.02987
        """
        basis_set = self.input_data["Basis Set"]
        num_basis = len(basis_set)

        memory_bytes = (num_basis**2) * 8
        self.required_resources = ClassicalResource(memory=memory_bytes, num_cores=1)

    def set_output_properties(self):
        pass

    def compute_flop_count(self):
        """

        FLOPs = N^2 * 50 * P, where:
        - N is the number of basis functions
        - 50 is the approximate FLOP count for kinetic/nuclear attraction integrals
        - P is the number of primitives per basis function
    
        """
        basis_set = self.input_data["Basis Set"]
        num_basis = len(basis_set)
        num_primitives = sum(len(bf.params[0]) for bf in basis_set)

        return num_basis**2 * 50 * num_primitives

    def validate_data(self) -> bool:

        return True

    def update(self):
        basis_functions = self.input_data["Basis Set"]
        charges = self.input_data["Nuclear Charges"]
        r = self.input_data["Coordinates"]
        self.result = compute_core_matrix(basis_functions, charges, r)

    def generate_output(self) -> list[Data]:

        return (
            Data(
                data=self.result,
                properties={"Usage": "Core Matrix"},
            ),
        )


class GSE_ComputeNuclearRepulsion(ClassicalProcess):
    def __init__(
        self,
        inputs: list[Data] = None,
        expected_input_properties=None,
        required_resources=None,
        output_properties=None,
    ):
        if inputs != None:
            self.input_data = {}
            for data in inputs:
                self.input_data[data.properties["Usage"]] = data.data

        super().__init__(
            inputs,
            expected_input_properties,
            required_resources,
            output_properties,
        )

    def set_expected_input_properties(self):
        self.expected_input_properties = [
            {"Data Type": torch.Tensor, "Usage": "Coordinates"},
            {"Data Type": list, "Usage": "Nuclear Charges"},
        ]

    def set_required_resources(self):
        
        self.required_resources = ClassicalResource(memory=8, num_cores=1)


    def set_output_properties(self):
        pass

    def compute_flop_count(self):
        num_nuclei = len(self.input_data["Nuclear Charges"])
        num_pairs = num_nuclei * (num_nuclei - 1) // 2
        return num_pairs * 3


    def validate_data(self) -> bool:

        return True

    def update(self):
        charges = self.input_data["Nuclear Charges"]
        coordinates = self.input_data["Coordinates"]
        self.result = nuclear_energy(charges, coordinates)(*coordinates).item()

    def generate_output(self) -> list[Data]:

        return (
            Data(
                data=self.result,
                properties={"Usage": "Nuclear Repulsion"},
            ),
        )


class GSE_ComputeEigs(ClassicalProcess):
    def __init__(
        self,
        inputs: list[Data] = None,
        expected_input_properties=None,
        required_resources=None,
        output_properties=None,
        matrix_name=None,
    ):
        if inputs != None:
            self.input_data = {}
            for data in inputs:
                self.input_data[data.properties["Usage"]] = data.data

        self.matrix_name = matrix_name
        super().__init__(
            inputs,
            expected_input_properties,
            required_resources,
            output_properties,
        )

    def set_expected_input_properties(self):
        self.expected_input_properties = [
            {"Data Type": torch.Tensor, "Usage": self.matrix_name + " Matrix"}
        ]

    def set_required_resources(self):
        self.required_resources = ClassicalResource(
            memory=42,
            num_cores=1,
        )

    def set_output_properties(self):
        pass

    def compute_flop_count(self):
        return 10

    def validate_data(self) -> bool:

        return True

    def update(self):
        eigvals, eigvecs = torch.linalg.eigh(
            self.input_data[self.matrix_name + " Matrix"]
        )
        self.results = {
            self.matrix_name + " Eigvals": eigvals,
            self.matrix_name + " Eigvecs": eigvecs,
        }

    def generate_output(self) -> list[Data]:

        return tuple(
            Data(
                data=val,
                properties={"Usage": key},
            )
            for key, val in self.results.items()
        )


class GSE_ComputeTrnsfrmMat(ClassicalProcess):
    def __init__(
        self,
        inputs: list[Data] = None,
        expected_input_properties=None,
        required_resources=None,
        output_properties=None,
    ):
        if inputs != None:
            self.input_data = {}
            for data in inputs:
                self.input_data[data.properties["Usage"]] = data.data

        super().__init__(
            inputs,
            expected_input_properties,
            required_resources,
            output_properties,
        )

    def set_expected_input_properties(self):
        self.expected_input_properties = [
            {"Data Type": torch.Tensor, "Usage": "Overlap Eigvals"},
            {"Data Type": torch.Tensor, "Usage": "Overlap Eigvecs"},
        ]

    def set_required_resources(self):
        self.required_resources = ClassicalResource(
            memory=42,
            num_cores=1,
        )

    def set_output_properties(self):
        pass

    def compute_flop_count(self):
        return 10

    def validate_data(self) -> bool:

        return True

    def update(self):
        eigvals = self.input_data["Overlap Eigvals"]
        eigvecs = self.input_data["Overlap Eigvecs"]
        self.result = eigvecs @ torch.diag(1.0 / torch.sqrt(eigvals)) @ eigvecs.T

    def generate_output(self) -> list[Data]:

        return (
            Data(
                data=self.result,
                properties={"Usage": "Transform Matrix"},
            ),
        )


class GSE_ComputeCoulombTerm(ClassicalProcess):
    def __init__(
        self,
        inputs: list[Data] = None,
        expected_input_properties=None,
        required_resources=None,
        output_properties=None,
    ):
        if inputs != None:
            self.input_data = {}
            for data in inputs:
                self.input_data[data.properties["Usage"]] = data.data

        super().__init__(
            inputs,
            expected_input_properties,
            required_resources,
            output_properties,
        )

    def set_expected_input_properties(self):
        self.expected_input_properties = [
            {"Data Type": torch.Tensor, "Usage": "Density Matrix"},
            {"Data Type": torch.Tensor, "Usage": "Repulsion Tensor"},
        ]

    def set_required_resources(self):
        """
        Memory (bytes) = 8 * (N^4 + 2 * N^2), where:
            - N^4 elements for the Repulsion Tensor.
            - N^2 elements for the Density Matrix.
            - N^2 elements for the Coulomb Term Matrix.
        
        """
        N = self.input_data["Density Matrix"].shape[0]
        memory_bytes = 8 * (N**4 + 2 * N**2)
        self.required_resources = ClassicalResource(memory=memory_bytes, num_cores=1)

    def compute_flop_count(self):
        """
        FLOPs = 2 * N^4, where:
            - N is the number of basis functions.
            - Each element in the Coulomb matrix involves N^2 multiplications and additions.

        
        """
        N = self.input_data["Density Matrix"].shape[0]
        return 2 * N**4

    def set_output_properties(self):
        pass

    def compute_flop_count(self):
        return 10

    def validate_data(self) -> bool:

        return True

    def update(self):
        density_mat = self.input_data["Density Matrix"]
        self.result = torch.einsum(
            "pqrs,rs->pq", self.input_data["Repulsion Tensor"], density_mat
        )

    def generate_output(self) -> list[Data]:

        return (
            Data(
                data=self.result,
                properties={"Usage": "Coulomb Term"},
            ),
        )


class GSE_ComputeExchangeTerm(ClassicalProcess):
    def __init__(
        self,
        inputs: list[Data] = None,
        expected_input_properties=None,
        required_resources=None,
        output_properties=None,
    ):
        if inputs != None:
            self.input_data = {}
            for data in inputs:
                self.input_data[data.properties["Usage"]] = data.data

        super().__init__(
            inputs,
            expected_input_properties,
            required_resources,
            output_properties,
        )

    def set_expected_input_properties(self):
        self.expected_input_properties = [
            {"Data Type": torch.Tensor, "Usage": "Density Matrix"},
            {"Data Type": torch.Tensor, "Usage": "Repulsion Tensor"},
        ]

    def set_required_resources(self):
        N = self.input_data["Density Matrix"].shape[0]
        memory_bytes = 8 * (N**4 + 2 * N**2)
        self.required_resources = ClassicalResource(memory=memory_bytes, num_cores=1)
    def set_output_properties(self):
        pass

    def compute_flop_count(self):
        N = self.input_data["Density Matrix"].shape[0]
        return 2 * N**4

    def validate_data(self) -> bool:

        return True

    def update(self):
        density_mat = self.input_data["Density Matrix"]
        self.result = torch.einsum(
            "prqs,rs->pq", self.input_data["Repulsion Tensor"], density_mat
        )

    def generate_output(self) -> list[Data]:

        return (
            Data(
                data=self.result,
                properties={"Usage": "Exchange Term"},
            ),
        )


class GSE_ComputeFockMat(ClassicalProcess):
    def __init__(
        self,
        inputs: list[Data] = None,
        expected_input_properties=None,
        required_resources=None,
        output_properties=None,
    ):
        if inputs != None:
            self.input_data = {}
            for data in inputs:
                self.input_data[data.properties["Usage"]] = data.data

        super().__init__(
            inputs,
            expected_input_properties,
            required_resources,
            output_properties,
        )

    def set_expected_input_properties(self):
        self.expected_input_properties = [
            {"Data Type": torch.Tensor, "Usage": "Core Matrix"},
            {"Data Type": torch.Tensor, "Usage": "Exchange Term"},
            {"Data Type": torch.Tensor, "Usage": "Coulomb Term"},
        ]

    def set_required_resources(self):
        self.required_resources = ClassicalResource(
            memory=42,
            num_cores=1,
        )

    def set_output_properties(self):
        pass

    def compute_flop_count(self):
        return 10

    def validate_data(self) -> bool:

        return True

    def update(self):
        fock_matrix = (
            self.input_data["Core Matrix"]
            + 2 * self.input_data["Coulomb Term"]
            - self.input_data["Exchange Term"]
        )
        self.result = fock_matrix

    def generate_output(self) -> list[Data]:

        return (
            Data(
                data=self.result,
                properties={"Usage": "Fock Matrix"},
            ),
        )


class GSE_TransformFock(ClassicalProcess):
    def __init__(
        self,
        inputs: list[Data] = None,
        expected_input_properties=None,
        required_resources=None,
        output_properties=None,
    ):
        if inputs != None:
            self.input_data = {}
            for data in inputs:
                self.input_data[data.properties["Usage"]] = data.data

        super().__init__(
            inputs,
            expected_input_properties,
            required_resources,
            output_properties,
        )

    def set_expected_input_properties(self):
        self.expected_input_properties = [
            {"Data Type": torch.Tensor, "Usage": "Fock Matrix"},
            {"Data Type": torch.Tensor, "Usage": "Transform Matrix"},
        ]

    def set_required_resources(self):
        self.required_resources = ClassicalResource(
            memory=42,
            num_cores=1,
        )

    def set_output_properties(self):
        pass

    def compute_flop_count(self):
        return 10

    def validate_data(self) -> bool:

        return True

    def update(self):
        transform_matrix = self.input_data["Transform Matrix"]
        fock_matrix = self.input_data["Fock Matrix"]

        ortho_fock_matrix = transform_matrix.T @ fock_matrix @ transform_matrix
        self.result = ortho_fock_matrix

    def generate_output(self) -> list[Data]:

        return (
            Data(
                data=self.result,
                properties={"Usage": "Orthogonal Fock Matrix"},
            ),
        )


class GSE_TransformEigvecs(ClassicalProcess):
    def __init__(
        self,
        inputs: list[Data] = None,
        expected_input_properties=None,
        required_resources=None,
        output_properties=None,
    ):
        if inputs != None:
            self.input_data = {}
            for data in inputs:
                self.input_data[data.properties["Usage"]] = data.data

        super().__init__(
            inputs,
            expected_input_properties,
            required_resources,
            output_properties,
        )

    def set_expected_input_properties(self):
        self.expected_input_properties = [
            {"Data Type": torch.Tensor, "Usage": "Orthogonal Fock Eigvecs"},
            {"Data Type": torch.Tensor, "Usage": "Transform Matrix"},
        ]

    def set_required_resources(self):
        self.required_resources = ClassicalResource(
            memory=42,
            num_cores=1,
        )

    def set_output_properties(self):
        pass

    def compute_flop_count(self):
        return 10

    def validate_data(self) -> bool:

        return True

    def update(self):
        self.result = (
            self.input_data["Transform Matrix"]
            @ self.input_data["Orthogonal Fock Eigvecs"]
        )

    def generate_output(self) -> list[Data]:

        return (
            Data(
                data=self.result,
                properties={"Usage": "Transformed Eigvecs"},
            ),
        )


class GSE_UpdateDensityMat(ClassicalProcess):
    def __init__(
        self,
        inputs: list[Data] = None,
        expected_input_properties=None,
        required_resources=None,
        output_properties=None,
    ):
        if inputs != None:
            self.input_data = {}
            for data in inputs:
                self.input_data[data.properties["Usage"]] = data.data

        super().__init__(
            inputs,
            expected_input_properties,
            required_resources,
            output_properties,
        )

    def set_expected_input_properties(self):
        self.expected_input_properties = [
            {"Data Type": torch.Tensor, "Usage": "Transformed Eigvecs"},
            {"Data Type": int, "Usage": "Num Electrons"},
        ]

    def set_required_resources(self):
        self.required_resources = ClassicalResource(
            memory=42,
            num_cores=1,
        )

    def set_output_properties(self):
        pass

    def compute_flop_count(self):
        return 10

    def validate_data(self) -> bool:

        return True

    def update(self):
        C = self.input_data["Transformed Eigvecs"]
        C_occ = C[:, : self.input_data["Num Electrons"] // 2]  # Occupied orbitals
        self.result = C_occ @ C_occ.T

    def generate_output(self) -> list[Data]:

        return (
            Data(
                data=self.result,
                properties={"Usage": "Updated Density Matrix"},
            ),
        )


class GSE_ComputeElecEnergy(ClassicalProcess):
    def __init__(
        self,
        inputs: list[Data] = None,
        expected_input_properties=None,
        required_resources=None,
        output_properties=None,
    ):
        if inputs != None:
            self.input_data = {}
            for data in inputs:
                self.input_data[data.properties["Usage"]] = data.data

        super().__init__(
            inputs,
            expected_input_properties,
            required_resources,
            output_properties,
        )

    def set_expected_input_properties(self):
        self.expected_input_properties = [
            {"Data Type": torch.Tensor, "Usage": "Updated Density Matrix"},
            {"Data Type": torch.Tensor, "Usage": "Core Matrix"},
            {"Data Type": torch.Tensor, "Usage": "Fock Matrix"},
            {"Data Type": float, "Usage": "Nuclear Repulsion"},
        ]

    def set_required_resources(self):
        self.required_resources = ClassicalResource(
            memory=42,
            num_cores=1,
        )

    def set_output_properties(self):
        pass

    def compute_flop_count(self):
        return 10

    def validate_data(self) -> bool:

        return True

    def update(self):
        density_matrix = self.input_data["Updated Density Matrix"]
        core_matrix = self.input_data["Core Matrix"]
        fock_matrix = self.input_data["Fock Matrix"]
        nuclear_repulsion_energy = self.input_data["Nuclear Repulsion"]

        E_elec = torch.sum(density_matrix * (core_matrix + fock_matrix))
        self.result = E_elec.item() + nuclear_repulsion_energy

    def generate_output(self) -> list[Data]:

        return (
            Data(
                data=self.result,
                properties={"Usage": "Electronic Energy"},
            ),
        )


class GSE_SCFItr(ClassicalProcess):
    def __init__(
        self,
        inputs: list[Data] = None,
        expected_input_properties=None,
        required_resources=None,
        output_properties=None,
        init=True,
        tol=1e-8,
        remaining_itrs=50,
    ):
        if inputs != None:
            self.input_data = {}
            for data in inputs:
                self.input_data[data.properties["Usage"]] = data.data
        self.init = init
        self.tol = tol
        self.itrs = remaining_itrs
        super().__init__(
            inputs,
            expected_input_properties,
            required_resources,
            output_properties,
            dynamic=True if remaining_itrs > 0 or init else False,
        )

    def set_expected_input_properties(self):
        self.expected_input_properties = [
            {"Data Type": torch.Tensor, "Usage": "Core Matrix"},
            {"Data Type": torch.Tensor, "Usage": "Repulsion Tensor"},
            {"Data Type": torch.Tensor, "Usage": "Transform Matrix"},
            {"Data Type": int, "Usage": "Num Electrons"},
            {"Data Type": float, "Usage": "Nuclear Repulsion"},
            {"Data Type": list, "Usage": "Num Basis Per Atom"},
        ]

        if not self.init:
            input_properties = [
                {"Data Type": torch.Tensor, "Usage": "Updated Density Matrix"},
                {"Data Type": torch.Tensor, "Usage": "Fock Matrix"},
                {"Data Type": torch.Tensor, "Usage": "Orthogonal Fock Eigvals"},
                {"Data Type": torch.Tensor, "Usage": "Transformed Eigvecs"},
                {"Data Type": float, "Usage": "Previous Electronic Energy"},
                {"Data Type": float, "Usage": "Electronic Energy"},
            ]

            self.expected_input_properties = (
                self.expected_input_properties + input_properties
            )

    def set_required_resources(self):
        self.required_resources = ClassicalResource(
            memory=42,
            num_cores=1,
        )

    def set_output_properties(self):
        pass

    def compute_flop_count(self):
        return 10

    def validate_data(self) -> bool:

        return True

    def update(self):
        if self.init:
            self.converge = False
            n_basis = sum(self.input_data["Num Basis Per Atom"])
            self.input_data["Density Matrix"] = torch.zeros(
                (n_basis, n_basis), dtype=torch.float64
            )
            self.input_data["Electronic Energy"] = 0.0

        else:
            electronic_energy = self.input_data["Electronic Energy"]
            electronic_energy_prev = self.input_data["Previous Electronic Energy"]
            self.converge = abs(electronic_energy - electronic_energy_prev) < self.tol
            self.dynamic = not self.converge

        if self.converge:
            num_electrons = self.input_data["Num Electrons"]
            num_basis = sum(self.input_data["Num Basis Per Atom"])
            mo_occ = [2 if i < num_electrons // 2 else 0 for i in range(num_basis)]

            self.results = {
                "Mean Field Energy": self.input_data["Electronic Energy"],
                "Molecular Orbital Energies": self.input_data[
                    "Orthogonal Fock Eigvals"
                ].tolist(),
                "Molecular Orbital Coeff": self.input_data["Transformed Eigvecs"]
                .clone()
                .detach(),
                "Fock Matrix": self.input_data["Fock Matrix"],
                "Molecular Orbital Occupancies": mo_occ,
            }

    def generate_output(self) -> list[Data]:

        if self.converge:
            return tuple(
                Data(data=val, properties={"Usage": key})
                for key, val in self.results.items()
            )

        else:
            output = []
            for key, val in self.input_data.items():
                if key in [
                    "Previous Electronic Energy",
                    "Orthogonal Fock Eigvals",
                    "Fock Matrix",
                    "Transformed Eigvecs",
                ]:
                    continue
                elif key == "Electronic Energy":
                    output.append(
                        Data(
                            data=val, properties={"Usage": "Previous Electronic Energy"}
                        )
                    )
                elif key == "Updated Density Matrix":
                    output.append(
                        Data(data=val, properties={"Usage": "Density Matrix"})
                    )
                else:
                    output.append(Data(data=val, properties={"Usage": key}))

            return tuple(output)

    def extend_network(self) -> Network:
        coulomb_term = Node(process_model=GSE_ComputeCoulombTerm, network_type="INPUT")
        exchange_term = Node(
            process_model=GSE_ComputeExchangeTerm, network_type="INPUT"
        )
        compute_fock = Node(
            process_model=GSE_ComputeFockMat, network_type="NETWORK"
        )  # "INPUT"
        transform_fock = Node(
            process_model=GSE_TransformFock, network_type="NETWORK"
        )  # INPUT
        compute_eigs = Node(
            process_model=GSE_ComputeEigs,
            network_type="NETWORK",
            matrix_name="Orthogonal Fock",
        )
        trnsfrm_eigvecs = Node(
            process_model=GSE_TransformEigvecs, network_type="NETWORK"
        )
        update_density = Node(
            process_model=GSE_UpdateDensityMat, network_type="NETWORK"
        )
        compute_elec_energy = Node(
            process_model=GSE_ComputeElecEnergy, network_type="NETWORK"
        )
        scf_itr = Node(
            process_model=GSE_SCFItr,
            network_type="OUTPUT",
            init=False,
            tol=self.tol,
            remaining_itrs=self.itrs - 1,
        )

        coulomb_term.insert_output_node(compute_fock)
        exchange_term.insert_output_node(compute_fock)
        compute_fock.insert_output_node(transform_fock)
        transform_fock.insert_output_node(compute_eigs)
        compute_eigs.insert_output_node(trnsfrm_eigvecs)
        trnsfrm_eigvecs.insert_output_node(update_density)
        compute_fock.insert_output_node(compute_elec_energy)
        update_density.insert_output_node(compute_elec_energy)

        update_density.insert_output_node(scf_itr)
        compute_elec_energy.insert_output_node(scf_itr)
        compute_fock.insert_output_node(scf_itr)
        compute_eigs.insert_output_node(scf_itr)
        trnsfrm_eigvecs.insert_output_node(scf_itr)

        nodes = [
            coulomb_term,
            exchange_term,
            compute_fock,
            transform_fock,
            compute_eigs,
            trnsfrm_eigvecs,
            update_density,
            compute_elec_energy,
            scf_itr,
        ]
        input_nodes = [
            coulomb_term,
            exchange_term,
            compute_fock,
            trnsfrm_eigvecs,
            transform_fock,
            update_density,
            compute_elec_energy,
            scf_itr,
        ]
        output_nodes = [scf_itr]

        network = Network(
            nodes=nodes, input_nodes=input_nodes, output_nodes=output_nodes
        )

        return network


class GSE_InitAnsatz(ClassicalProcess):
    def __init__(
        self,
        inputs: list[Data] = None,
        expected_input_properties=None,
        required_resources=None,
        output_properties=None,
    ):
        if inputs != None:
            self.input_data = {}
            for data in inputs:
                self.input_data[data.properties["Usage"]] = data.data

        super().__init__(
            inputs,
            expected_input_properties,
            required_resources,
            output_properties,
        )

    def set_expected_input_properties(self):
        self.expected_input_properties = [
            {"Data Type": str, "Usage": "Qubit Mapping"},
            {"Data Type": str, "Usage": "Ansatz"},
            {"Data Type": str, "Usage": "Geometry Model"},
            {"Data Type": list, "Usage": "Molecular Orbital Occupancies"},
            {"Data Type": [str, np.ndarray], "Usage": "Ansatz Params"},
        ]

    def set_required_resources(self):
        self.required_resources = ClassicalResource(
            memory=42,
            num_cores=1,
        )

    def set_output_properties(self):
        pass

    def compute_flop_count(self):
        return 10

    def validate_data(self) -> bool:
        conditions = (
            self.__valid_ansatz_name(),
            self.__valid_qubit_mapping(),
            self.__valid_ansatz_params(),
        )
        return all(condition for condition in conditions)

    def __valid_ansatz_name(self):
        supported_ansatzes = ["UCCSD"]
        return self.input_data["Ansatz"] in supported_ansatzes

    def __valid_qubit_mapping(self):
        supported_mappings = ["JW", "BK", "SCBK"]
        return self.input_data["Qubit Mapping"] in supported_mappings

    def __valid_ansatz_params(self):
        supported_ansatz_params = ["random", "ones"]
        ansatz_params = self.input_data["Ansatz Params"]

        if isinstance(ansatz_params, str):
            return ansatz_params in supported_ansatz_params
        elif isinstance(ansatz_params, np.ndarray):
            return True
        else:
            return False

    def update(self):
        mol_orbital_occupancies = self.input_data["Molecular Orbital Occupancies"]
        n_molecular_orbitals = len(mol_orbital_occupancies)
        geometry_model = self.input_data["Geometry Model"]

        if geometry_model == "RHF":
            n_active_spin_orbitals = 2 * n_molecular_orbitals
        elif geometry_model == "UHF":
            n_active_spin_orbitals = max(
                2 * len(mol_orbital_occupancies[0]), 2 * len(mol_orbital_occupancies[1])
            )
        else:
            raise ValueError(f"{geometry_model} is invalid or not implemented")

        n_active_electrons = sum(mol_orbital_occupancies)

        # self.n_spinorbitals = molecule.n_active_sos
        # self.n_electrons = molecule.n_active_electrons

        n_spatial_orbitals = n_active_spin_orbitals // 2
        n_occupied = int(np.ceil(n_active_electrons / 2))

        n_virtual = n_spatial_orbitals - n_occupied
        n_singles = n_occupied * n_virtual
        n_doubles = n_singles * (n_singles + 1) // 2
        n_params = n_singles + n_doubles

        self.results = {
            "Num Parameters": n_params,
            "Ansatz Params": self.input_data["Ansatz Params"],
            "Num Active Spin Orbitals": n_active_spin_orbitals,
            "Num Active Electrons": n_active_electrons,
            "Qubit Mapping": self.input_data["Qubit Mapping"],
        }

    def generate_output(self) -> list[Data]:
        outputs = []

        for key, val in self.results.items():
            if key == "Ansatz Params":
                outputs.append(
                    Data(data=val, properties={"Usage": key, "Status": "Inactive"})
                )
            else:
                outputs.append(Data(data=val, properties={"Usage": key}))

        return tuple(outputs)


class GSE_SetAnsatzParams(ClassicalProcess):
    def __init__(
        self,
        inputs: list[Data] = None,
        expected_input_properties=None,
        required_resources=None,
        output_properties=None,
    ):
        if inputs != None:
            self.input_data = {}
            for data in inputs:
                self.input_data[data.properties["Usage"]] = data.data

        super().__init__(
            inputs,
            expected_input_properties,
            required_resources,
            output_properties,
        )

    def set_expected_input_properties(self):
        self.expected_input_properties = [
            {
                "Data Type": [str, np.ndarray],
                "Usage": "Ansatz Params",
                "Status": "Inactive",
            },
            {"Data Type": int, "Usage": "Num Parameters"},
        ]

    def set_required_resources(self):
        self.required_resources = ClassicalResource(
            memory=42,
            num_cores=1,
        )

    def set_output_properties(self):
        pass

    def compute_flop_count(self):
        return 10

    def validate_data(self) -> bool:
        conditions = (self.__valid_params(),)

        return all(condition for condition in conditions)

    def __valid_params(self):
        supported_ansatz_params = ["random", "ones"]
        ansatz_params = self.input_data["Ansatz Params"]

        if isinstance(ansatz_params, str):
            return ansatz_params in supported_ansatz_params
        elif isinstance(ansatz_params, np.ndarray):
            return ansatz_params.size == self.input_data["Num Parameters"]
        else:
            return False

    def update(self):
        ansatz_params = self.input_data["Ansatz Params"]
        num_params = self.input_data["Num Parameters"]

        if isinstance(ansatz_params, np.ndarray):
            self.result = ansatz_params
        elif isinstance(ansatz_params, str):
            if ansatz_params == "ones":
                self.result = np.ones((num_params,), dtype=float)
            elif ansatz_params == "random":
                self.result = 2.0e-1 * (np.random.random((num_params,)) - 0.5)
            else:
                raise ValueError(
                    f'{ansatz_params} is either invalid or not supported. Choose from ["ones","random"]'
                )
        else:
            raise ValueError(
                f"{ansatz_params} should either be a string or numpy array"
            )

    def generate_output(self) -> list[Data]:

        return (
            Data(
                data=self.result,
                properties={"Usage": "Ansatz Params", "Status": "Active"},
            ),
        )


class GSE_GenRefCircuit(ClassicalProcess):
    def __init__(
        self,
        inputs: list[Data] = None,
        expected_input_properties=None,
        required_resources=None,
        output_properties=None,
    ):
        if inputs != None:
            self.input_data = {}
            for data in inputs:
                self.input_data[data.properties["Usage"]] = data.data

        super().__init__(
            inputs,
            expected_input_properties,
            required_resources,
            output_properties,
            dynamic=True,
        )

    def set_expected_input_properties(self):
        self.expected_input_properties = [
            {"Data Type": [str, np.ndarray], "Usage": "Ansatz Params"},
            {"Data Type": int, "Usage": "Num Parameters"},
            {"Data Type": int, "Usage": "Num Active Spin Orbitals"},
            {"Data Type": int, "Usage": "Num Active Electrons"},
            {"Data Type": str, "Usage": "Qubit Mapping"},
        ]

    def set_required_resources(self):
        self.required_resources = ClassicalResource(
            memory=42,
            num_cores=1,
        )

    def set_output_properties(self):
        pass

    def compute_flop_count(self):
        return 10

    def validate_data(self) -> bool:
        return True

    def update(self):
        occupation_vector = np.zeros(
            self.input_data["Num Active Spin Orbitals"], dtype=int
        )
        occupation_vector[: self.input_data["Num Active Electrons"]] = 1
        self.result = occupation_vector

    def generate_output(self) -> list[Data]:
        mapping = self.input_data["Qubit Mapping"]
        outputs = [
            Data(
                data=self.result,
                properties={"Usage": "Occupation Vector"},
            ),
            Data(
                data=self.input_data["Num Active Spin Orbitals"],
                properties={"Usage": "Num Active Spin Orbitals"},
            ),
        ]

        if mapping == "SCBK":
            outputs.append(
                Data(
                    data=self.input_data["Num Active Electrons"],
                    properties={"Usage": "Num Active Electrons"},
                ),
            )

        return tuple(outputs)

    def extend_network(self) -> Node | List[Node] | Network:
        mapping = self.input_data["Qubit Mapping"]
        
        
        if mapping == "JW":
            vec_mapping = Node(process_model=GSE_ApplyJWMapping, network_type="INPUT")
        elif mapping == "BK":
            vec_mapping = Node(process_model=GSE_ApplyBKMapping, network_type="INPUT")
        elif mapping == "SCBK":
            vec_mapping = Node(process_model=GSE_ApplySCBKMapping, network_type="INPUT")
        else:
            raise ValueError(f"{mapping} is invalid or not implemented")

        vec_to_circ = Node(process_model=GSE_OccVectorToCircuit, network_type="OUTPUT")
        vec_mapping.insert_output_node(vec_to_circ)

        network = Network(
            nodes=[vec_mapping, vec_to_circ],
            input_nodes=[vec_mapping, vec_to_circ],
            output_nodes=[vec_to_circ],
        )

        return network


### TODO: FIX output Qubit Operator vs Occupation Vector
class GSE_ApplyJWMapping(ClassicalProcess):
    def __init__(
        self,
        inputs: list[Data] = None,
        expected_input_properties=None,
        required_resources=None,
        output_properties=None,
    ):
        if inputs != None:
            self.input_data = {}
            for data in inputs:
                self.input_data[data.properties["Usage"]] = data.data
                if data.properties["Usage"] in [
                    "Fermion Operator",
                    "Occupation Vector",
                ]:
                    self.input_data["Op Type"] = data.properties["Usage"]
                    self.input_data["Op Data Type"] = data.properties["Data Type"]

        super().__init__(
            inputs,
            expected_input_properties,
            required_resources,
            output_properties,
        )

    def set_expected_input_properties(self):
        self.expected_input_properties = [
            {
                "Data Type": [np.ndarray, FermionOperator],
                "Usage": ["Fermion Operator", "Occupation Vector"],
            }
        ]

    def set_required_resources(self):
        self.required_resources = ClassicalResource(
            memory=42,
            num_cores=1,
        )

    def set_output_properties(self):
        pass

    def compute_flop_count(self):
        return 10

    def validate_data(self) -> bool:
        conditions = (self.__validate_properties(),)
        return all(condition for condition in conditions)

    def __validate_properties(self):
        if self.input_data["Op Type"] == "Occupation Vector":
            return self.input_data["Op Data Type"] == np.ndarray
        elif self.input_data["Op Type"] == "Fermion Operator":
            return self.input_data["Op Data Type"] == FermionOperator
        else:
            return False

    def update(self):
        """Apply JW Mapping to the FermionOperator or Occupation Vector."""
        op_type = self.input_data["Op Type"]
        if op_type == "Occupation Vector":
            # JW Mapping for occupation vector: no changes (identity mapping)
            self.result = self.input_data["Occupation Vector"]
        elif op_type == "Fermion Operator":
            # JW Mapping for FermionOperator
            self.result = jordan_wigner(self.input_data["Fermion Operator"])
        else:
            raise ValueError(f"Unsupported input type: {op_type}")

    def generate_output(self) -> list[Data]:
        if self.input_data["Op Type"] == "Occupation Vector":
            return (
                Data(
                    data=self.result,
                    properties={
                        "Usage": self.input_data["Op Type"],
                        "Status": "Mapped",
                    },
                ),
            )
        elif self.input_data["Op Type"] == "Fermion Operator":
            return (
                Data(
                    data=self.result,
                    properties={"Usage": "Qubit Operator", "Status": "Mapped"},
                ),
            )


class GSE_ApplyBKMapping(ClassicalProcess):
    def __init__(
        self,
        inputs: list[Data] = None,
        expected_input_properties=None,
        required_resources=None,
        output_properties=None,
    ):
        if inputs != None:
            self.input_data = {}
            for data in inputs:
                self.input_data[data.properties["Usage"]] = data.data
                if data.properties["Usage"] in [
                    "Fermion Operator",
                    "Occupation Vector",
                ]:
                    self.input_data["Op Type"] = data.properties["Usage"]
                    self.input_data["Op Data Type"] = data.properties["Data Type"]

        super().__init__(
            inputs,
            expected_input_properties,
            required_resources,
            output_properties,
        )

    def set_expected_input_properties(self):
        self.expected_input_properties = [
            {
                "Data Type": np.ndarray,
                "Usage": ["Fermion Operator", "Occupation Vector"],
            },
            {"Data Type": int, "Usage": "Num Active Spin Orbitals"},
        ]

    def set_required_resources(self):
        self.required_resources = ClassicalResource(
            memory=42,
            num_cores=1,
        )

    def set_output_properties(self):
        pass

    def compute_flop_count(self):
        return 10

    def validate_data(self) -> bool:
        conditions = (self.__validate_properties(),)
        return all(condition for condition in conditions)

    def __validate_properties(self):
        if self.input_data["Op Type"] == "Occupation Vector":
            return self.input_data["Op Data Type"] == np.ndarray
        elif self.input_data["Op Type"] == "Fermion Operator":
            return self.input_data["Op Data Type"] == FermionOperator
        else:
            return False

    def update(self):
        """Apply BK Mapping to the FermionOperator or Occupation Vector."""
        op_type = self.input_data["Op Type"]
        if op_type == "Occupation Vector":
            # Apply BK mapping and retain consistent usage tag
            self.result = do_bk_transform(self.input_data["Occupation Vector"])
            self.result_usage = "Occupation Vector"  # Retain input's usage
        elif op_type == "Fermion Operator":
            self.result = bravyi_kitaev(
                self.input_data["Fermion Operator"],
                n_qubits=self.input_data["Num Active Spin Orbitals"],
            )
            self.result_usage = "Qubit Operator"
        else:
            raise ValueError(f"Unsupported input type: {op_type}")

    def generate_output(self) -> list[Data]:
        """Ensure output is tagged properly and compatible downstream."""
        return (
            Data(
                data=self.result,
                properties={"Usage": self.result_usage, "Status": "Mapped"},
            ),
            Data(
                data=self.input_data["Num Active Spin Orbitals"],
                properties={"Usage": "Num Active Spin Orbitals"},
            ),
        )
    # def generate_output(self) -> list[Data]:
        # if self.input_data["Op Type"] == "Occupation Vector":
        #     return (
        #         Data(
        #             data=self.result,
        #             properties={
        #                 "Usage": "Occupation Vector",  # Ensure consistent property naming
        #                 "Status": "Mapped",
        #             },
        #         ),
        #         Data(
        #             data=self.input_data["Num Active Spin Orbitals"],
        #             properties={"Usage": "Num Active Spin Orbitals"},
        #         ),
        #     )
        # elif self.input_data["Op Type"] == "Fermion Operator":
        #     return (
        #         Data(
        #             data=self.result,
        #             properties={"Usage": "Qubit Operator", "Status": "Mapped"},
        #         ),
        #     )


class GSE_ApplySCBKMapping(ClassicalProcess):
    def __init__(
        self,
        inputs: list[Data] = None,
        expected_input_properties=None,
        required_resources=None,
        output_properties=None,
    ):
        if inputs != None:
            self.input_data = {}
            for data in inputs:
                self.input_data[data.properties["Usage"]] = data.data
                if data.properties["Usage"] in [
                    "Fermion Operator",
                    "Occupation Vector",
                ]:
                    self.input_data["Op Type"] = data.properties["Usage"]
                    self.input_data["Op Data Type"] = data.properties["Data Type"]

        super().__init__(
            inputs,
            expected_input_properties,
            required_resources,
            output_properties,
        )

    def set_expected_input_properties(self):
        self.expected_input_properties = [
            {
                "Data Type": np.ndarray,
                "Usage": ["Fermion Operator", "Occupation Vector"],
            },
            {"Data Type": int, "Usage": "Num Active Spin Orbitals"},
            {"Data Type": int, "Usage": "Num Active Electrons"},
        ]

    def set_required_resources(self):
        self.required_resources = ClassicalResource(
            memory=42,
            num_cores=1,
        )

    def set_output_properties(self):
        pass

    def compute_flop_count(self):
        return 10

    def validate_data(self) -> bool:
        conditions = (self.__validate_properties(),)
        return all(condition for condition in conditions)

    def __validate_properties(self):
        if self.input_data["Op Type"] == "Occupation Vector":
            return self.input_data["Op Data Type"] == np.ndarray
        elif self.input_data["Op Type"] == "Fermion Operator":
            return self.input_data["Op Data Type"] == FermionOperator
        else:
            return False

    def update(self):
        n_active_so = self.input_data["Num Active Spin Orbitals"]
        n_active_elecs = self.input_data["Num Active Electrons"]

        if self.input_data["Op Type"] == "Occupation Vector":
            occupation_vector = self.input_data["Occupation Vector"]
            fermion_op = FermionOperator(occupation_vector)
        elif self.input_data["Op Type"] == "Fermion Operator":
            fermion_op = self.input_data["Fermion Operator"]

        self.result = symmetry_conserving_bravyi_kitaev(
            fermion_op, n_active_so, n_active_elecs
        )

    def generate_output(self) -> list[Data]:

        return (
            Data(
                data=self.result,
                properties={"Usage": self.input_data["Op Type"], "Status": "Mapped"},
            ),
        )


class GSE_OccVectorToCircuit(ClassicalProcess):
    def __init__(
        self,
        inputs: list[Data] = None,
        expected_input_properties=None,
        required_resources=None,
        output_properties=None,
    ):
        if inputs != None:
            self.input_data = {}
            for data in inputs:
                self.input_data[data.properties["Usage"]] = data.data

        super().__init__(
            inputs,
            expected_input_properties,
            required_resources,
            output_properties,
        )

    def set_expected_input_properties(self):
        self.expected_input_properties = [
            {"Data Type": np.ndarray, "Usage": "Occupation Vector", "Status": "Mapped"},
            {"Data Type": int, "Usage": "Num Active Spin Orbitals"},
        ]

    def set_required_resources(self):
        self.required_resources = ClassicalResource(
            memory=42,
            num_cores=1,
        )

    def set_output_properties(self):
        pass

    def compute_flop_count(self):
        return 10

    def validate_data(self) -> bool:
        return True

    def update(self):
        num_qubits = self.input_data["Num Active Spin Orbitals"]
        circuit = QuantumCircuit(qubit_count=num_qubits, gate_set=["X"])

        for i, occ in enumerate(self.input_data["Occupation Vector"]):
            if occ == 1:
                circuit.add_instruction(gate=X(), indices=(i,))

        self.result = circuit

    def generate_output(self) -> list[Data]:

        return (
            Data(
                data=self.result,
                properties={"Usage": "Reference Circuit"},
            ),
        )


class GSE_ComputeFermionOp(ClassicalProcess):
    def __init__(
        self,
        inputs: list[Data] = None,
        expected_input_properties=None,
        required_resources=None,
        output_properties=None,
    ):
        if inputs != None:
            self.input_data = {}
            for data in inputs:
                self.input_data[data.properties["Usage"]] = data.data

        super().__init__(
            inputs,
            expected_input_properties,
            required_resources,
            output_properties,
        )

    def set_expected_input_properties(self):
        self.expected_input_properties = [
            {
                "Data Type": [str, np.ndarray],
                "Usage": "Ansatz Params",
                "Status": "Active",
            },
            {"Data Type": int, "Usage": "Num Active Spin Orbitals"},
            {"Data Type": int, "Usage": "Num Active Electrons"},
        ]

    def set_required_resources(self):
        self.required_resources = ClassicalResource(
            memory=42,
            num_cores=1,
        )

    def set_output_properties(self):
        pass

    def compute_flop_count(self):
        return 10

    def validate_data(self) -> bool:
        return True

    def update(self):
        ansatz_params = self.input_data["Ansatz Params"]
        n_spin_orbs = self.input_data["Num Active Spin Orbitals"]
        n_electrons = self.input_data["Num Active Electrons"]

        self.result = uccsd_singlet_generator(ansatz_params, n_spin_orbs, n_electrons)

    def generate_output(self) -> list[Data]:

        return (
            Data(
                data=self.result,
                properties={"Usage": "Fermion Operator"},
            ),
        )


class GSE_ComputeQubitOp(ClassicalProcess):
    def __init__(
        self,
        inputs: list[Data] = None,
        expected_input_properties=None,
        required_resources=None,
        output_properties=None,
    ):
        if inputs != None:
            self.input_data = {}
            for data in inputs:
                self.input_data[data.properties["Usage"]] = data.data

        super().__init__(
            inputs,
            expected_input_properties,
            required_resources,
            output_properties,
            dynamic=True,
        )

    def set_expected_input_properties(self):
        self.expected_input_properties = [
            {"Data Type": FermionOperator, "Usage": "Fermion Operator"},
            {"Data Type": str, "Usage": "Qubit Mapping"},
        ]

    def set_required_resources(self):
        self.required_resources = ClassicalResource(
            memory=42,
            num_cores=1,
        )

    def set_output_properties(self):
        pass

    def compute_flop_count(self):
        return 0

    def validate_data(self) -> bool:
        return True

    def update(self):
        pass

    def generate_output(self) -> list[Data]:
        return (
            Data(
                data=self.input_data["Fermion Operator"],
                properties={"Usage": "Fermion Operator"},
            ),
        )

    def extend_network(self) -> Node:
        mapping = self.input_data["Qubit Mapping"]

        # Select appropriate mapping node
        if mapping == "JW":
            return Node(process_model=GSE_ApplyJWMapping)
        elif mapping == "BK":
            return Node(process_model=GSE_ApplyBKMapping)
        else:
            raise ValueError(f"{mapping} is invalid or not implemented.")


class GSE_GenAnsatzCircuit(ClassicalProcess):
    def __init__(
        self,
        inputs: list[Data] = None,
        expected_input_properties=None,
        required_resources=None,
        output_properties=None,
        init=True,
    ):
        if inputs != None:
            self.input_data = {}
            for data in inputs:
                self.input_data[data.properties["Usage"]] = data.data

        self.init = init
        super().__init__(
            inputs,
            expected_input_properties,
            required_resources,
            output_properties,
        )

    def set_expected_input_properties(self):
        self.expected_input_properties = [
            {"Data Type": QubitOperator, "Usage": "Qubit Operator", "Status": "Mapped"},
            {"Data Type": QuantumCircuit, "Usage": "Reference Circuit"},
            # {"Data Type": int, "Usage": "Num Parameters"},
        ]

        if not self.init:
            self.expected_input_properties.append(
                {"Data Type": list, "Usage": "Ansatz Params"}
            )

    def set_required_resources(self):
        self.required_resources = ClassicalResource(
            memory=42,
            num_cores=1,
        )

    def set_output_properties(self):
        pass

    def compute_flop_count(self):
        return 10

    def validate_data(self) -> bool:
        return True

    def update(self):
        """Construct the Ansatz circuit."""
        qubit_operator: QubitOperator = self.input_data["Qubit Operator"]
        ref_circuit: QuantumCircuit = self.input_data["Reference Circuit"]

        # Sort Pauli words in QubitOperator
        pauli_words = sorted(
            qubit_operator.terms.items(), key=lambda x: len(x[0])
        )

        # Initialize the circuit
        circuit = QuantumCircuit(qubit_count=ref_circuit.qubit_count, gate_set=[])
        params = []

        # Include reference circuit instructions
        if len(ref_circuit.instructions) > 0:
            for instruction in ref_circuit.instructions:
                circuit.add_instruction(instruction)

        # Build the circuit for each Pauli word
        for i, (pauli_word, coef) in enumerate(pauli_words):
            indices = [idx for idx, _ in pauli_word]

            # Basis transformations
            for idx, op in pauli_word:
                if op == "X":
                    circuit.add_instruction(gate=H(), indices=(idx,))
                elif op == "Y":
                    circuit.add_instruction(gate=RX(theta=np.pi / 2), indices=(idx,))

            # CNOT Ladder
            for j in range(len(indices) - 1):
                circuit.add_instruction(gate=CX(), indices=(indices[j], indices[j + 1]))

            # Rotation gate (parameterized RZ or CRZ)
            param = random.uniform(0, 2 * np.pi)
            params.append(param)
            last_idx = indices[-1]

            if len(indices) > 1:
                circuit.add_instruction(
                    gate=CZPow(exponent=2 * param / np.pi),
                    indices=(indices[-2], last_idx),
                )
            else:
                circuit.add_instruction(gate=RZ(theta=2 * param), indices=(last_idx,))

            # Undo CNOT Ladder
            for j in range(len(indices) - 1, 0, -1):
                circuit.add_instruction(gate=CX(), indices=(indices[j - 1], indices[j]))

            # Undo basis transformations
            for idx, op in reversed(pauli_word):
                if op == "X":
                    circuit.add_instruction(gate=H(), indices=(idx,))
                elif op == "Y":
                    circuit.add_instruction(gate=RX(theta=-np.pi / 2), indices=(idx,))

        # Store results: parameters and final circuit
        self.result = {
            "Ansatz Params": params,
            "Ansatz Circuit": circuit,
        }

    def generate_output(self) -> list[Data]:
        """Generate outputs as Data objects."""
        return tuple(
            Data(
                data=val,
                properties={"Usage": key},
            )
            for key, val in self.result.items()
        )


# class GSE_RunCircuit(QuantumProcess):
# def __init__(
#     self,
#     inputs: list[Data] = None,
#     expected_input_properties=None,
#     required_resources=None,
#     output_properties=None,
# ):
#     if inputs != None:
#         self.input_data = {}
#         for data in inputs:
#             self.input_data[data.properties["Usage"]] = data.data

#     super().__init__(
#         inputs,
#         expected_input_properties,
#         required_resources,
#         output_properties,
#     )

# def set_expected_input_properties(self):
#     self.expected_input_properties = [
#         {"Data Type": QuantumCircuit, "Usage": "Ansatz Circuit"},
#         # {"Data Type": int, "Usage": "Num Parameters"},
#     ]

# def set_required_resources(self):
#     self.required_resources = QuantumResource(
#         quantum_circuit=self.compute_circuit()
#     )

# def compute_circuit(self):
#     """Computes the number of required flops for this process to complete given
#     valid inputs from self.inputs. Will be unique to each Process.
#     """

#     return self.input_data["Ansatz Circuit"]

# def set_output_properties(self):
#     pass

# def validate_data(self) -> bool:
#     return True

# def update(self):
#     assert 1 == 0

# def generate_output(self) -> list[Data]:

#     return (
#         Data(
#             data=val,
#             properties={"Usage": key},
#         )
#         for key, val in self.result.items()
#     )


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