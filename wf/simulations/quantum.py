from typing import List, Tuple, Dict, TYPE_CHECKING
from z3 import Optimize, Int, IntVector, And, Or, Bool, Implies, If, sat
import datetime
import copy
from .quantum_gates import QuantumGate, SWAP,X,Y,Z,T,Tdg,CX,H,S,CZ,CZPow
import networkx as nx
from .utilities import adjacent_edges_from_edge, adjacent_edge_idxs_from_edge
# import qiskit.circuit.quantumcircuit as QiskitCircuit
from qiskit import QuantumCircuit as QiskitCircuit
import qiskit.circuit.library as qiskit_library
import qrew.simulation.refactor.quantum_gates as quantum_gateset
from qrew.simulation.refactor.data_types import *

from ...util.log import logging
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from resources.quantum_resources import QuantumDevice


class QuantumInstruction:
    def __init__(self, gate, qubit_indices):
        self.gate: QuantumGate = gate
        self.gate_indices: Tuple[int] = qubit_indices

    def __str__(self):
        return self.gate.name + f" {self.gate_indices}"
    def __repr__(self) -> str:
        if hasattr(self.gate, "param"):
            param_str = f" param={self.gate.param}"
        elif hasattr(self.gate, "params") and self.gate.params:
            param_str = f" params={self.gate.params}"
        else:
            param_str = ""
        return (
            f"<QuantumInstruction name='{self.gate.name}' "
            f"qargs={self.gate_indices}{param_str}>"
        )

class QuantumCircuit:
    def __init__(self, qubit_count=None, gate_set=None, instructions=None):
        self.qubit_count = qubit_count
        self.gate_set: List[QuantumGate] = [] if gate_set is None else gate_set
        self.instructions: List[QuantumInstruction] = (
            [] if instructions is None else instructions
        )

        for gate in self.gate_set:
            try:
                getattr(quantum_gateset, gate)
            except:
                raise ValueError(f"Gate {gate} is not a valid gate within the gateset")

        for instruction in self.instructions:
            assert isinstance(
                instruction, QuantumInstruction
            ), f"{instruction} must be of type QuantumInstruction, got {type(instruction)} instead"
            assert self.valid_gate_indices(
                instruction
            ), f"Gate indices invalid or out of bounds for instruction {instruction}"
            if instruction.gate not in self.gate_set:
                self.gate_set.append(instruction.gate)

    def add_instruction(
        self,
        instruction: QuantumInstruction = None,
        gate: QuantumGate = None,
        indices: tuple[int] = None,
    ):
        if instruction is None:
            assert isinstance(
                gate, QuantumGate
            ), f"{gate} must be a QuantumGate instance"
            assert isinstance(
                indices, tuple
            ), f"{indices} must be a tuple of gate indices"
            circ_instrc = QuantumInstruction(gate=gate, qubit_indices=indices)
        else:
            assert isinstance(
                instruction, QuantumInstruction
            ), f"add_instruction requires a QuantumInstruction or a QuantumGate and indices as input"
            circ_instrc = instruction

        assert self.valid_gate_indices(
            circ_instrc
        ), f"Gate indices invalid or out of bounds for instruction {circ_instrc}"

        if circ_instrc.gate.name not in self.gate_set:
            self.gate_set.append(circ_instrc.gate.name)

        self.instructions.append(circ_instrc)

    def valid_gate_indices(self, instruction: QuantumInstruction):
        indices_in_range = all(
            qubit_index >= 0 and qubit_index <= self.qubit_count
            for qubit_index in instruction.gate_indices
        )
        indices_not_repeated = len(instruction.gate_indices) == len(
            list(set(instruction.gate_indices))
        )

        return indices_in_range and indices_not_repeated

    def depth(self) -> int:
        depth_at_idx = [0] * self.qubit_count
        for instruction in self.instructions:
            new_depth = max(
                depth_at_idx[gate_idx] for gate_idx in instruction.gate_indices
            )

            for gate_idx in instruction.gate_indices:
                depth_at_idx[gate_idx] = new_depth + 1

        return max(depth_at_idx)
    
    def draw(self, save_measurements=False):
        """Return a Cirq SVG diagram of the circuit, ready for Jupyter display."""
        from cirq.contrib.svg import SVGCircuit
        import cirq
        from qrew.visualization_tools import translate_c_to_cirq
        cirq_circ = translate_c_to_cirq(self, save_measurements=save_measurements)
        # Delete the leading identity moment Cirq inserted (looks cleaner)
        if cirq_circ and isinstance(cirq_circ[0], cirq.Moment):
            cirq_circ.__delitem__(0)
        return SVGCircuit(cirq_circ)

    def __repr__(self) -> str:
        """
        <QuantumCircuit 5 qubits, 13 instructions (CX:6, H:5, MEASURE:2), depth=7>
        echoes:
          • Qiskit  <QuantumCircuit 5 qubits, 13 classical bits, 17 elements>
          • Cirq    cirq.Circuit(…)
        """
        gate_hist = {}
        for instr in self.instructions:
            gate_hist[instr.gate.name] = gate_hist.get(instr.gate.name, 0) + 1
        gate_summary = ", ".join(f"{name}:{cnt}" for name, cnt in sorted(gate_hist.items()))

        return (
            f"<QuantumCircuit {self.qubit_count} qubits, "
            f"{len(self.instructions)} instructions ({gate_summary}), "
            f"depth={self.depth()}>"
        )
def qiskit_to_quantum_circuit(qc: QiskitCircuit):
    circuit = QuantumCircuit(qubit_count=qc.num_qubits, instructions=[])
    for qiskit_gate in qc.data:
        gate_name = qiskit_gate.operation.name.upper()
        params = qiskit_gate.params

        if gate_name == "TDG":
            gate_name = "Tdg"

        gate = getattr(quantum_gateset, gate_name)
        if len(params) > 0:
            circuit.add_instruction(
                gate=gate(params[0]),
                indices=tuple(qubit._index for qubit in qiskit_gate.qubits),
            )
        else:
            circuit.add_instruction(
                gate=gate(), indices=tuple(qubit._index for qubit in qiskit_gate.qubits)
            )

    return circuit


def quantum_circuit_to_qiskit(qc: QuantumCircuit):
    circuit = QiskitCircuit(qc.qubit_count, 0)
    for instruction in qc.instructions:
        gate_name = instruction.gate.name

        if gate_name == "CZPow":
            circuit.append(
                qiskit_library.CZGate().power(instruction.gate.param),
                list(instruction.gate_indices),
            )
        else:
            qiskit_gate = getattr(qiskit_library, gate_name + "Gate")
            if hasattr(instruction.gate, "param"):
                circuit.append(
                    qiskit_gate(instruction.gate.param), list(instruction.gate_indices)
                )
            else:
                circuit.append(qiskit_gate(), list(instruction.gate_indices))

    return circuit


class LayoutSynthesizer:
    def __init__(
        self,
        quantum_circuit: QuantumCircuit,
        device: "QuantumDevice",
        transition_based=True,
        hard_island = False,
        epsilon=0.3,
        objective="depth",
    ):
        """
        Args:
            quantum_circuit (QuantumCircuit): Quantum circuit of QuantumInstructions for layout synthesis
            device (QuantumDevice): The device for which to execute the circuit
            transition_based (bool, optional): _description_. Whether to use the transition based model. Defaults to False.
            epsilon (float, optional): _description_. Objective increment size at each optimization cycle. Defaults to 0.3.
            objective (str, optional): _description_. Optimization objective. Defaults to "depth".
        """
        self.circuit = quantum_circuit
        self.device = device
        self.transition_based = transition_based
        self.hard_island = hard_island
        self.epsilon = epsilon
        self.objective = objective
        self.swap_duration = 1 if transition_based else self.device.swap_duration

        assert objective in [
            "depth",
            "fidelity",
        ], "Objective must either be 'depth' or 'fidelity'"

        self.circuit_depth = self.compute_circuit_depth(quantum_circuit)
        self.circuit_num_gates = len(quantum_circuit.instructions)
        self.available_connectivity = device.get_available_connectivity()
        self.variables = dict()
        self.solver = Optimize()

    def compute_circuit_depth(self, quantum_circuit: QuantumCircuit):
        if self.transition_based:
            return 1

        return quantum_circuit.depth()
      

    def find_optimal_layout(self):
        """Takes the initialized quantum circuit and device and attempts to
        find an optimal solution based on the given objective. The resulting
        optimized circuit, the initial qubit mapping and final qubit mapping
        of the logical circuit qubits are returned
        """
        self.solve()
        result_circuit, initial_qubit_map, final_qubit_map, objective_result,_ = (
            self.post_process()
        )
        return result_circuit, initial_qubit_map, final_qubit_map, objective_result,_

    def solve(self, max_attempts=50, max_depth=10000):
        found_solution = False
        start_time = datetime.datetime.now()

        attempt_count = 0

        self.solver.set(timeout=10000)  # 10-second timeout
        logger.debug("Starting solver with max_attempts=%d, max_depth=%d", max_attempts, max_depth)

        while not found_solution:
            attempt_count += 1
            print(f"Attempting maximal depth {self.circuit_depth}...")
            if attempt_count > max_attempts or self.circuit_depth > max_depth:
                msg = (f"No solution found after {attempt_count} attempts "
                    f"and reaching depth {self.circuit_depth}. Aborting to prevent infinite loop.")
                logger.error(msg)
                raise RuntimeError(msg)
            
            self.initialize_variables()
            self.add_constraints()
            self.add_optimization_objective()

            satisfiable = self.solver.check()
       
            # logger.debug("Solver returned: %s", satisfiable)

            if satisfiable == sat:
                # logger.debug("Found solution at depth=%d on attempt #%d", self.circuit_depth, attempt_count)
                found_solution = True
                model_depth = self.solver.model()[self.variables["depth"]]
                logger.debug(f"Found solution at depth={self.circuit_depth}, model depth={model_depth.as_long()}, on attempt #{attempt_count}")
            else:
                logger.debug("No solution at depth=%d (attempt #%d). Adjusting depth...", self.circuit_depth, attempt_count)
                if self.transition_based:
                    self.circuit_depth = self.circuit_depth + 1
                else:
                    self.circuit_depth = int((1 + self.epsilon) * self.circuit_depth)
                self.solver = Optimize()
        total_time = datetime.datetime.now() - start_time
        logger.debug(f"Layout synthesis time completed in {total_time} after {attempt_count} attempts")


    def initialize_variables(self):
        """Initializes variables for solver"""
        self.variables["pi"] = self.initialize_pi()
        self.variables["time"] = self.initialize_time()
        self.variables["space"] = self.initialize_space()
        self.variables["sigma"] = self.initialize_sigma()
        self.variables["depth"] = self.initialize_depth()

    def initialize_pi(self) -> List[List[Int]]:
        """At cycle t, logical qubit q is mapped to pi[q][t]"""
        return [
            [
                Int("pi_{" + f"q{qubit_idx},t{timestep}" + "}")
                for timestep in range(self.circuit_depth)
            ]
            for qubit_idx in range(self.circuit.qubit_count)
        ]

    def initialize_time(self) -> IntVector:
        """Time coordinates for gate i is gate[i]"""
        return IntVector("time", self.circuit_num_gates)

    def initialize_space(self) -> IntVector:
        """Space coordinates for gate i is space[i]"""
        return IntVector("space", self.circuit_num_gates)

    def initialize_sigma(self) -> Dict:
        """If at time t, a SWAP gate completes on the edge between qubit indices q1 and q2, then sigma[q1][q2][t]=1"""
        sigma = {}
        for edge in self.available_connectivity.edges:
            if edge[0] not in sigma:
                sigma[edge[0]] = {}
            if edge[1] not in sigma[edge[0]]:
                sigma[edge[0]][edge[1]] = {}
            for timestep in range(self.circuit_depth):
                sigma[edge[0]][edge[1]][timestep] = Bool(
                    "sigma_{" + f"e({edge[0]},{edge[1]}),t{timestep}" + "}"
                )
        return sigma

    def initialize_depth(self) -> Int:
        """Variable depth of the compiled circuit"""
        return Int("depth")

    def add_constraints(self):
        """Add relevant constraints to layout synthesis optimization problem"""
        self.constraint_injective_mapping()
        self.constraint_avoid_collisions()
        self.constraint_mapping_and_space_consistency()
        self.constraint_no_SWAP_before_swap_duration()
        self.constraint_no_SWAP_overlap_same_edge()
        self.constraint_no_SWAP_overlap_adjacent_edge()
        if not self.transition_based:
            self.constraint_no_SWAP_overlap_gates()
        self.constraint_mapping_unchanged()
        self.constraint_mapping_transformed()
        if self.hard_island:
            logger.debug(f"Contraint initial island added: relay qubits forbidden")
            self.constraint_no_SWAP_outside_gates()

    def constraint_injective_mapping(self):
        """
        Different logical qubits should be mapped to different physical qubits at any specific time
        
        """
        pi = self.variables["pi"]
        for timestep in range(self.circuit_depth):
            for qubit_idx1 in range(self.circuit.qubit_count):
                valid_pi_vals = [
                    pi[qubit_idx1][timestep] == qubit_idx
                    for qubit_idx in self.device.available_qubits
                ]
                self.solver.add(Or(valid_pi_vals))
                for qubit_idx2 in range(qubit_idx1):
                    self.solver.add(
                        pi[qubit_idx1][timestep] != pi[qubit_idx2][timestep]
                    )

    def constraint_avoid_collisions(self):
        """Avoiding gate collisions and respecting gate dependencies"""
        time = self.variables["time"]
        for collision in self.collision_extraction():
            if self.transition_based:
                self.solver.add(time[collision[0]] <= time[collision[1]])
            else:
                self.solver.add(time[collision[0]] < time[collision[1]])

    ## TODO find better solution
    def collision_extraction(self):
        """Extract collision relations between gates:
        If gates g1 and g2 both acts on a qubit (at different times), we
        say that g1 and g2 collide on that qubit, which means (1,2)
        will be in the collision list
        """
        collisions = []
        circuit_instructions = self.circuit.instructions
        for current_idx in range(len(circuit_instructions)):
            for next_idx in range(current_idx + 1, len(circuit_instructions)):
                if any(
                    gate_idx in circuit_instructions[next_idx].gate_indices
                    for gate_idx in circuit_instructions[current_idx].gate_indices
                ):
                    collisions.append((current_idx, next_idx))
        return collisions

    def constraint_mapping_and_space_consistency(self):
        """for a gate g acting on its timestep t, it's space coordinate space[g] should match the mapping of the gates logical qubit(s) q: space[g] == pi[q][t]"""
        instructions = self.circuit.instructions
        time = self.variables["time"]
        space = self.variables["space"]
        pi = self.variables["pi"]
        for gate_idx in range(self.circuit_num_gates):
            self.solver.add(time[gate_idx] >= 0, time[gate_idx] < self.circuit_depth)
            gate_qubit_indices = instructions[gate_idx].gate_indices
            if len(gate_qubit_indices) == 1:
                valid_space_vals = [
                    space[gate_idx] == qubit_idx
                    for qubit_idx in self.device.available_qubits
                ]
                self.solver.add(Or(valid_space_vals))
                for timestep in range(self.circuit_depth):
                    self.solver.add(
                        Implies(
                            time[gate_idx] == timestep,
                            pi[gate_qubit_indices[0]][timestep] == space[gate_idx],
                        )
                    )
            else:
                self.solver.add(
                    space[gate_idx] >= 0,
                    space[gate_idx] < self.available_connectivity.number_of_edges(),
                )
                edge_idx = 0
                for edge in self.available_connectivity.edges:
                    for timestep in range(self.circuit_depth):
                        imply_condition = And(
                            time[gate_idx] == timestep, space[gate_idx] == edge_idx
                        )
                        or_operand1 = And(
                            edge[0] == pi[gate_qubit_indices[0]][timestep],
                            edge[1] == pi[gate_qubit_indices[1]][timestep],
                        )
                        or_operand2 = And(
                            edge[0] == pi[gate_qubit_indices[1]][timestep],
                            edge[1] == pi[gate_qubit_indices[0]][timestep],
                        )
                        imply_result = Or(or_operand1, or_operand2)
                        self.solver.add(Implies(imply_condition, imply_result))
                    edge_idx = edge_idx + 1

    def constraint_no_SWAP_before_swap_duration(self):
        """For all timesteps less than the swap duration of the device, there cannot be any SWAPs completed"""
        sigma = self.variables["sigma"]
        for timestep in range(min(self.swap_duration - 1, self.circuit_depth)):
            for edge in self.available_connectivity.edges:
                self.solver.add(sigma[edge[0]][edge[1]][timestep] == False)

    def constraint_no_SWAP_overlap_same_edge(self):
        """SWAP gates cannot be applied on the same duration on the same edge"""
        sigma = self.variables["sigma"]
        for timestep in range(self.swap_duration - 1, self.circuit_depth):
            for edge in self.available_connectivity.edges:
                for swap_timestep in range(timestep - self.swap_duration + 1, timestep):
                    constraint = Implies(
                        sigma[edge[0]][edge[1]][timestep] == True,
                        sigma[edge[0]][edge[1]][swap_timestep] == False,
                    )
                    self.solver.add(constraint)

    def constraint_no_SWAP_overlap_adjacent_edge(self):
        """Edges that overlap on the device cannot both have overlapping SWAP gates in time"""
        sigma = self.variables["sigma"]
        for timestep in range(self.swap_duration - 1, self.circuit_depth):
            for edge in self.available_connectivity.edges:
                for swap_timestep in range(
                    timestep - self.swap_duration + 1, timestep + 1
                ):
                    for adjacent_edge in adjacent_edges_from_edge(
                        self.available_connectivity, edge[0], edge[1]
                    ):
                        imply_operand1 = sigma[edge[0]][edge[1]][timestep] == True
                        try:
                            imply_operand2 = (
                                sigma[adjacent_edge[0]][adjacent_edge[1]][swap_timestep]
                                == False
                            )
                        except KeyError:
                            imply_operand2 = (
                                sigma[adjacent_edge[1]][adjacent_edge[0]][swap_timestep]
                                == False
                            )
                        constraint = Implies(imply_operand1, imply_operand2)
                        self.solver.add(constraint)

    def constraint_no_SWAP_overlap_gates(self):
        """SWAP gates should not overlap with any gates on the same qubit(s) at the same time"""
        instructions = self.circuit.instructions
        time = self.variables["time"]
        space = self.variables["space"]
        sigma = self.variables["sigma"]
        for timestep in range(self.swap_duration - 1, self.circuit_depth):
            edge_idx = 0
            for edge in self.available_connectivity.edges:
                for swap_timestep in range(
                    timestep - self.swap_duration + 1, timestep + 1
                ):
                    for instruction_idx in range(len(instructions)):
                        if len(instructions[instruction_idx].gate_indices) == 1:
                            imply_condition = And(
                                time[instruction_idx] == swap_timestep,
                                Or(
                                    space[instruction_idx] == edge[0],
                                    space[instruction_idx] == edge[1],
                                ),
                            )
                            self.solver.add(
                                Implies(
                                    imply_condition,
                                    sigma[edge[0]][edge[1]][timestep] == False,
                                )
                            )

                        else:
                            imply_condition = And(
                                time[instruction_idx] == swap_timestep,
                                space[instruction_idx] == edge_idx,
                            )
                            self.solver.add(
                                Implies(
                                    imply_condition,
                                    sigma[edge[0]][edge[1]][timestep] == False,
                                )
                            )
                            for adjacent_edge_idx in adjacent_edge_idxs_from_edge(
                                self.available_connectivity, edge[0], edge[1]
                            ):
                                imply_condition = And(
                                    time[instruction_idx] == swap_timestep,
                                    space[instruction_idx] == adjacent_edge_idx,
                                )
                                self.solver.add(
                                    Implies(
                                        imply_condition,
                                        sigma[edge[0]][edge[1]][timestep] == False,
                                    )
                                )
                edge_idx = edge_idx + 1

    def constraint_mapping_unchanged(self):
        """Mappings from physical to logical qubits remains unchanged if there no is a SWAP gate applied"""
        pi = self.variables["pi"]
        sigma = self.variables["sigma"]
        for timestep in range(self.circuit_depth - 1):
            for physical_qubit_idx in self.device.available_qubits:
                for logical_qubit_idx in range(self.circuit.qubit_count):
                    sum_list = []
                    for edge in self.available_connectivity.edges(physical_qubit_idx):
                        try:
                            if_condition = sigma[edge[0]][edge[1]][timestep]
                        except KeyError:
                            if_condition = sigma[edge[1]][edge[0]][timestep]
                        sum_list.append(If(if_condition, 1, 0))
                    implies_condition = And(
                        sum(sum_list) == 0,
                        pi[logical_qubit_idx][timestep] == physical_qubit_idx,
                    )
                    self.solver.add(
                        Implies(
                            implies_condition,
                            pi[logical_qubit_idx][timestep + 1] == physical_qubit_idx,
                        )
                    )

    def constraint_mapping_transformed(self):
        """If a SWAP acts on time t, mappings from physical to logical qubits is transformed at t + 1"""
        pi = self.variables["pi"]
        sigma = self.variables["sigma"]
        for timestep in range(self.circuit_depth - 1):
            for edge in self.available_connectivity.edges:
                for logical_qubit_idx in range(self.circuit.qubit_count):
                    implies_condition = And(
                        sigma[edge[0]][edge[1]][timestep] == True,
                        pi[logical_qubit_idx][timestep] == edge[0],
                    )
                    self.solver.add(
                        Implies(
                            implies_condition,
                            pi[logical_qubit_idx][timestep + 1] == edge[1],
                        )
                    )

                    implies_condition = And(
                        sigma[edge[0]][edge[1]][timestep] == True,
                        pi[logical_qubit_idx][timestep] == edge[1],
                    )
                    self.solver.add(
                        Implies(
                            implies_condition,
                            pi[logical_qubit_idx][timestep + 1] == edge[0],
                        )
                    )

    def constraint_no_SWAP_outside_gates(self):
        pi = self.variables["pi"]
        sigma = self.variables["sigma"]

        # physical_qubit_idxs = []
        # for logical_qubit_idx in range(self.circuit.qubit_count):
        #     physical_qubit_idxs.append(pi[logical_qubit_idx][0])

        # for timestep in range(self.circuit_depth - 1):
        #     for edge in self.available_connectivity.edges:
        #         implies_condition = Or(edge[0] not in physical_qubit_idxs,
        #                                 edge[1] not in physical_qubit_idxs)
        #         self.solver.add(
        #             Implies(
        #                 implies_condition,
        #                 sigma[edge[0]][edge[1]][timestep] == False
        #             )
        #         )      

        P0 = [pi[q][0] for q in range(self.circuit.qubit_count)]

        for timestep in range(self.circuit_depth - 1):
            for (i, j) in self.available_connectivity.edges:
                # symbolic “i ∉ P0  ∨  j ∉ P0”
                i_out = And([i != p for p in P0])
                j_out = And([j != p for p in P0])
                self.solver.add(
                    Implies( Or(i_out, j_out),  sigma[i][j][timestep] == False )
                )    

    def add_optimization_objective(self):
        """Adds the optimization objective to the solver. This
        can either be minimzing the depth or maximizing the fidelity
        """
        time = self.variables["time"]
        depth = self.variables["depth"]
        if self.objective == "depth":
            for gate_idx in range(self.circuit_num_gates):
                self.solver.add(depth >= time[gate_idx] + 1)
            self.solver.minimize(depth)
        elif self.objective == "fidelity":
            raise NotImplementedError

    def post_process(self):
        pi = self.variables["pi"]
        depth = self.variables["depth"]
        self.results = dict()

        model = self.solver.model()
        self.results["depth"] = model[depth].as_long()
        self.results["time"] = self.get_time_results()
        self.results["SWAPs"] = self.get_SWAP_results()
        # logger.debug(f"results: {self.results}")
        if self.transition_based:
            self.update_transition_results()

        result_circuit = self.get_circuit_results()
        initial_qubit_map = tuple(
            model[pi[qubit_idx][0]].as_long()
            for qubit_idx in range(self.circuit.qubit_count)
        )
        final_qubit_map = tuple(
            model[pi[qubit_idx][model[depth].as_long() - 1]].as_long()
            for qubit_idx in range(self.circuit.qubit_count)
        )
       
        objective_result = self.results["depth"]
        time = self.results["time"]
        swaps = self.results["SWAPs"]

        logger.debug(f"Initial qubit mapping: {initial_qubit_map}")
        logger.debug(f"Final qubit mapping: {final_qubit_map}")
        logger.debug(f"Objective result: {objective_result}, SWAP count: {swaps}, time: {time}")

        del self.results
        return result_circuit, initial_qubit_map, final_qubit_map, objective_result, {'depth':objective_result,'time':time,'swaps':swaps}

    def get_time_results(self):
        model = self.solver.model()
        time = self.variables["time"]
        result_time = []
        for gate_idx in range(self.circuit_num_gates):
            result_time.append(model[time[gate_idx]].as_long())
        return result_time

    def get_SWAP_results(self, post_filter=False):
        """
         post_filter=False now default, will remove the filtering code all-together once team agrees
        """

        model = self.solver.model()
        sigma = self.variables["sigma"]
        pi = self.variables["pi"]
        depth = self.variables["depth"]
        
        physical_qubit_idxs = tuple(
            model[pi[qubit_idx][0]].as_long()
            for qubit_idx in range(self.circuit.qubit_count)
        )
        result_SWAPs = []
        for edge in self.available_connectivity.edges:
            if hasattr(self,'results'):
                assert model[depth].as_long() == self.results['depth']
            for timestep in range(model[depth].as_long()):
                
                if model[sigma[edge[0]][edge[1]][timestep]]:
                    if post_filter:
                        # only keep SWAPs whose both qubits are in the initial mapping
                        if edge[0] in physical_qubit_idxs and edge[1] in physical_qubit_idxs:
                            result_SWAPs.append(([edge[0], edge[1]], timestep))
                    else:
                        result_SWAPs.append(([edge[0], edge[1]], timestep))
                    
        return result_SWAPs
    
    

    def update_transition_results(self):
        self.swap_duration = self.device.swap_duration
        transition_time = [0 for _ in range(self.circuit_num_gates)]
        transition_qubit_depth = {
            qubit_idx: -1 for qubit_idx in self.device.available_qubits
        }
        transition_SWAPs = []
        model = self.solver.model()
        pi = self.variables["pi"]

        for transition_block in range(self.results["depth"]):
            for gate_idx in range(self.circuit_num_gates):
                if transition_block == self.results["time"][gate_idx]:
                    logical_qubit_idxs = self.circuit.instructions[
                        gate_idx
                    ].gate_indices
                    physical_qubit_idxs = [
                        model[pi[qubit_idx][transition_block]].as_long()
                        for qubit_idx in logical_qubit_idxs
                    ]
                    gate_time = (
                        max(
                            transition_qubit_depth[qubit_idx]
                            for qubit_idx in physical_qubit_idxs
                        )
                        + 1
                    )
                    transition_time[gate_idx] = gate_time

                    for qubit_idx in physical_qubit_idxs:
                        transition_qubit_depth[qubit_idx] = gate_time

            if transition_block < self.results["depth"] - 1:
                for swap_edge, timestep in self.results["SWAPs"]:
                    if timestep < self.results["depth"] - 1:
                        swap_time = (
                            max(
                                transition_qubit_depth[swap_qubit_idx]
                                for swap_qubit_idx in swap_edge
                            )
                            + self.swap_duration
                        )
                        transition_qubit_depth[swap_edge[0]] = swap_time
                        transition_qubit_depth[swap_edge[1]] = swap_time
                        transition_SWAPs.append(
                            ((swap_edge[0], swap_edge[1]), swap_time)
                        )

        self.results["time"] = transition_time
        self.results["depth"] = max(transition_qubit_depth.values()) + 1
        self.results["SWAPs"] = transition_SWAPs

    def get_circuit_results(self):
        result_circuit_instructions: List[List[QuantumInstruction]] = (
            self.parse_instruction_results()
        )
        result_circuit: QuantumCircuit = self.build_circuit_result(
            result_circuit_instructions
        )

        return result_circuit

    def parse_instruction_results(self):
        result_circuit_instructions: List[List[QuantumInstruction]] = [
            [] for _ in range(self.results["depth"])
        ]

        self.parse_circuit_gates(result_circuit_instructions)
        self.parse_circuit_SWAPs(result_circuit_instructions)

        return result_circuit_instructions

    def parse_circuit_gates(
        self, result_circuit_instructions: List[List[QuantumInstruction]]
    ):
        pi = self.variables["pi"]
        time = self.variables["time"]
        model = self.solver.model()

        for gate_idx in range(self.circuit_num_gates):
            instruction = self.circuit.instructions[gate_idx]
            gate_logical_idxs = instruction.gate_indices

            if self.transition_based:
                timestep = model[time[gate_idx]].as_long()
            else:
                timestep = self.results["time"][gate_idx]

            gate_physical_idxs = [
                model[pi[qubit_idx][timestep]].as_long()
                for qubit_idx in gate_logical_idxs
            ]
            result_circuit_instructions[self.results["time"][gate_idx]].append(
                QuantumInstruction(
                    gate=instruction.gate, qubit_indices=tuple(gate_physical_idxs)
                )
            )

    def parse_circuit_SWAPs(
        self, result_circuit_instructions: List[List[QuantumInstruction]]
    ):
        for physical_qubit_indices, timestep in self.results["SWAPs"]:
            print("SWAP", physical_qubit_indices, timestep)
            result_circuit_instructions[timestep].append(
                QuantumInstruction(
                    gate=SWAP(), qubit_indices=tuple(physical_qubit_indices)
                )
            )

    def build_circuit_result(
        self, result_circuit_instructions: List[List[QuantumInstruction]]
    ):
        result_circuit = QuantumCircuit(
            qubit_count=self.device.connectivity.number_of_nodes(), instructions=[]
        )
        for timestep in range(self.results["depth"]):
            for instruction in result_circuit_instructions[timestep]:
                if (instruction.gate.name == "SWAP") and (
                    "SWAP" not in self.device.gate_set
                ):
                    gate_idxs = instruction.gate_indices
                    for transpiled_instruction in self.device.transpiled_swap_circuit:
                        transpiled_gate: QuantumGate = getattr(
                            quantum_gateset,
                            transpiled_instruction.operation.name.upper(),
                        )
                        transpiled_gate_idxs = tuple(
                            gate_idxs[qubit._index]
                            for qubit in transpiled_instruction.qubits
                        )
                        if len(transpiled_instruction.params) > 0:
                            params = transpiled_instruction.params[0]
                            result_circuit.add_instruction(
                                gate=transpiled_gate(params),
                                indices=transpiled_gate_idxs,
                            )
                        else:
                            result_circuit.add_instruction(
                                gate=transpiled_gate(), indices=transpiled_gate_idxs
                            )

                        # result_circuit.add_instruction(
                        #     QuantumInstruction(
                        #         gate=transpiled_gate, qubit_indices=transpiled_gate_idxs
                        #     )
                        # )
                else:
                    result_circuit.add_instruction(instruction)

        return result_circuit