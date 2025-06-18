from typing import List, Tuple, Dict, TYPE_CHECKING
from z3 import Optimize, Int, IntVector, And, Or,Not, Bool, Implies, If, sat,Sum
import datetime
import copy
from .quantum_gates import QuantumGate, SWAP,X,Y,Z,T,Tdg,CX,H,S,CZ,CZPow
import networkx as nx
from .utilities import adjacent_edges_from_edge, adjacent_edge_idxs_from_edge
# import qiskit.circuit.quantumcircuit as QiskitCircuit
from qiskit import QuantumCircuit as QiskitCircuit
import qiskit.circuit.library as qiskit_library
import qrew.simulation.quantum_gates as quantum_gateset
from qrew.simulation.data_types import *
from qrew.simulation.quantum_util import QuantumOperand

from ..util.log import get_logger,logging
logger = get_logger(__name__)




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
        self.register_map: Dict = {}

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
        # ─── resolve any QuantumOperand via register_map ─────────────────────────
        resolved = []
        for op in circ_instrc.gate_indices:
            if isinstance(op, QuantumOperand):
                # look up physical index
                phys_list = self.register_map.get(op.register)
                assert phys_list is not None, (
                    f"No mapping for register {op.register}. "
                    "Did you forget to attach register_map after allocate?"
                )
                resolved.append(phys_list[op.offset])
            else:
                resolved.append(op)
        circ_instrc = QuantumInstruction(gate=circ_instrc.gate,
                                         qubit_indices=tuple(resolved))

        assert self.valid_gate_indices(circ_instrc), (
            f"Gate indices invalid or out of bounds after resolution: {circ_instrc}"
        )

        if circ_instrc.gate.name not in self.gate_set:
            self.gate_set.append(circ_instrc.gate.name)

        self.instructions.append(circ_instrc)

    def valid_gate_indices(self, instruction: QuantumInstruction):
        indices_in_range = all(
            0 <= qubit_index <= self.qubit_count
            for qubit_index in instruction.gate_indices
        )
        # TODO: verify if it should be 0 <= qubit_index < self.qubit_count instead...
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
        from qrew.simulation.q_interop.cirq_interop import translate_qc_to_cirq
        cirq_circ = translate_qc_to_cirq(self, save_measurements=save_measurements)
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

from dataclasses import dataclass
from typing import Sequence, Tuple

@dataclass(frozen=True)
class LayoutTrace:
    """Snapshot of a solved layout‐synthesis instance.

    * island        – π[:,0]         (initial logical→physical map)
    * final_mapping – π[:,depth-1]
    * swaps         – list[tuple[Tuple[int,int], int]]  (edge, finish-time)
    """
    island:         Tuple[int, ...]
    final_mapping:  Tuple[int, ...]
    swaps:          Sequence[Tuple[Tuple[int, int], int]]

    def touches_only_island(self) -> bool:
        """Return True iff every SWAP edge stays inside *island*."""
        return all(u in self.island and v in self.island for (u, v), _ in self.swaps)

class LayoutSynthesizer:
    """
    Optimal / TB-optimal layout synthesiser (main source: Bochen Tan & Jason Cong,
        *Optimal Layout Synthesis for Quantum Computing*,
        ASPLOS 2021 (arXiv 2007.15671).)

    Args:
            quantum_circuit (QuantumCircuit): Quantum circuit of QuantumInstructions for layout synthesis
            device (QuantumDevice): The device for which to execute the circuit
            transition_based (bool, optional): _description_. Whether to use the transition based model. Defaults to False.
            epsilon (float, optional): _description_. Objective increment size at each optimization cycle. Defaults to 0.3.
            objective (str, optional): _description_. Optimization objective. Defaults to "depth".
    ----------------------------------------------------------------------------
    Variable mapping  (paper → this implementation)
    ----------------------------------------------------------------------------
      M(l,t)=p            ->  x[l,p,t]  … encoded as self.variables["pi"][l][t]
      σ(g)                ->  self.variables["space"][g]
      T(g)=t              ->  self.variables["time"][g]
      s_{e,t}             ->  self.variables["sigma"][p][q][t]
      depth D             ->  self.variables["depth"]
    All *π*, *time*, *space*, *σ* variables are native **z3.Int / Bool** objects.

    ----------------------------------------------------------------------------
    Constructor flags
    ----------------------------------------------------------------------------
    transition_based : bool   — use TB-OLSQ (single “transition block” then
                                reconstruct fine schedule) [§4, Fig. 7 in paper].
    hard_island      : bool   — forbid SWAPs that leave the initial island
                                (our extension; see Eq. (15) in OLSQ-GA 2022).
    epsilon          : float  — geometric back-off increment when the depth
                                guess is infeasible  (ε in Alg. 1, line 17).
    objective        : str    — "depth"  (Eq. 12 in paper) or
                                "fidelity" (Eq. 13; **not yet implemented**).

    References
    ----------------------------------------------------------------------------
      • Tan & Cong 2021 — *Transition-Based OLSQ*  (TB-OLSQ)
      • Tan et al. 2022 — *Gate Absorption for OLSQ*  (OLSQ-GA)
      • Chiang et al. 2023 — *Scalable SMT Encodings for Qubit Mapping*
      • Li et al. 2024 — *Noise-Aware Optimal Layout Synthesis*
    TODO: 
        (1) objective='fidelity' and min-swap
        (2) allow_parallel_swaps - let disjoint edges swap concurrently
    """
    def __init__(
        self,
        quantum_circuit: QuantumCircuit,
        device: "QuantumDevice",
        transition_based=True,
        hard_island = False,
        epsilon=0.3,
        objective="depth",
    ):
        self.circuit = quantum_circuit
        self.device = device
        self.transition_based = transition_based
        self.hard_island = hard_island
        self.epsilon = epsilon
        self.objective = objective
        self.swap_duration = 1 if transition_based else self.device.swap_duration

        assert objective in [
            "depth",
            # "fidelity",
            "swap",
        ], "Objective must either be 'depth' or 'fidelity' (fidelity not yet impl.)"

        self.circuit_depth_guess = self.compute_circuit_depth(quantum_circuit)
        self.circuit_num_gates = len(quantum_circuit.instructions)
        self.available_connectivity = device.get_available_connectivity()
        self.variables: Dict[str, object] = {}
        self.solver = Optimize()
        # unit-tests call *before* solve() runs
        self.depth_guess = self.compute_circuit_depth(quantum_circuit)
        self.allow_parallel = True

    def compute_circuit_depth(self, quantum_circuit: QuantumCircuit) -> int:
        """
        The initial depth bound fed into the solver.
        
        In TB-OLSQ mode (transistion_based = True, see end of section 4.7 in Tan & Cong 2020), the
        course-grain time upper boud T is initially set to 1. In other words, we collapse the entire circuit into a single "transition block" with an initial depth = 1.

        In normal mode we take the logical depth of the input circuit as a lower bound (cannot be beaten even with perfect layout).
        """
    
        return 1 if self.transition_based else quantum_circuit.depth()
      

    def find_optimal_layout(self):
        """Run the full synthesis loop

        Returns
        -------
        result_circuit: \
            Fully native, routed, and scheduled circuit (with SWAPs expanded into device basis if needed).
        initial_qubit_map:
            π[:,0] — mapping of *logical*→*physical* at t = 0.
        final_qubit_map: 
            π[:,depth-1] — mapping at the last timestep (useful for chaining)
        objective_result: int
            Circuit depth (if objective="depth") or placeholder
        """
        self.solve()
        result_circuit, initial_qubit_map, final_qubit_map, objective_result,extra = (
            self.post_process()
        )
        return result_circuit, initial_qubit_map, final_qubit_map, objective_result,extra
    
    def _build_model(self, depth_guess: int):
        self.depth_guess = depth_guess
        self._initialize_variables()
        self._add_constraints()
        self._add_optimization_objective()

    def solve(self, max_attempts=50, max_depth=10000):
        """
        Exponential-back-off search over depth guesses where each iteration builds a fresh SMT instance:
            (1) initializes variables
            (2) adds constraints
            (3) add objective
            (4) check if satisfiable
        
            If unsatisfyable (UNSAT), we enlarge depth (⌈(1+ε)·depth⌉ in normal mode or +1 in TB-mode) and retry
        """
        depth_guess = self.circuit_depth_guess
        found_solution = False
        start_time = datetime.datetime.now()

        attempt_count = 0

        # self.solver.set(timeout=10000)  # 10-second per iteration
        self.solver.set()
        logger.debug("Starting solver with max_attempts=%d, max_depth=%d", max_attempts, max_depth)

        while not found_solution:
            attempt_count += 1
            
            # print(f"Attempting maximal depth {self.depth_guess}...")
            if attempt_count > max_attempts or depth_guess > max_depth:
                raise RuntimeError(
                    f"Aborted after {attempt_count} attempts (depth={depth_guess})"
                )
            
            #  (re)build model
            self._build_model(depth_guess)
            logger.debug(f"Attempt #{attempt_count} —  depth guess = {self.depth_guess}")

            satisfiable = self.solver.check()
       
            # logger.debug("Solver returned: %s", satisfiable)

            if satisfiable == sat:
                # logger.debug("Found solution at depth=%d on attempt #%d", self.depth_guess, attempt_count)
                found_solution = True
                mdl   = self.solver.model()
                d_val = mdl.evaluate(self.variables["depth"], model_completion=True).as_long()
                logger.debug("Found solution at depth=%d, model depth=%d",self.depth_guess, d_val)
            else:
                logger.debug("No solution at depth=%d -> increasing search bound", self.depth_guess)
                if self.transition_based:
                    depth_guess = depth_guess + 1
                else:
                    depth_guess = int((1 + self.epsilon) * depth_guess)
                self.solver = Optimize() # fresh context
        total_time = datetime.datetime.now() - start_time
        logger.debug(f"Layout synthesis time completed in {total_time} after {attempt_count} attempts")


    def _initialize_variables(self):
        """Initializes variables for solver"""
        self.variables["pi"] = self._initialize_pi()
        self.variables["time"] = self._initialize_time()
        self.variables["space"] = self._initialize_space()
        self.variables["sigma"] = self._initialize_sigma()
        self.variables["depth"] = self._initialize_depth()

        self.variables["n_swaps"] = Int("total_swaps")
        swap_terms = [
            If(self.variables["sigma"][u][v][t], 1, 0)
            for (u, v) in self.available_connectivity.edges
            for t in range(self.depth_guess)
        ]
        self.solver.add(
            self.variables["n_swaps"] ==
            Sum(*swap_terms) if swap_terms else 0
        )
    def _initialize_pi(self) -> List[List[Int]]:
        """
        At cycle t, logical qubit q is mapped to pi[q][t]

        π[q][t] - physical index p such that  M(q,t)=p

        Range:  available physical qubit indices (device.available_qubits)
        """
        return [
            [
                Int("pi_{" + f"q{qubit_idx},t{timestep}" + "}")
                for timestep in range(self.depth_guess)
            ]
            for qubit_idx in range(self.circuit.qubit_count)
        ]

    def _initialize_time(self) -> IntVector:
        """Time coordinates for gate i is gate[i]
        
        T(g) - discrete *finish* time of gate g (Eq. 2 in paper). Domain will be constrained to [0, depth-1] later.

        """
        return IntVector("time", self.circuit_num_gates)

    def _initialize_space(self) -> IntVector:
        """Space coordinates for gate i is space[i] where space[i] is a function into P (for 1-q gate) 
        or E (for 2-q). We encode it as one Int per gate:
            - 1q gate  =>  physical ID p
            - 2q gate  =>  index into list(available_connectivity.edges)
        
        The consistency constraints (Eq. 3 & 4) will ensure σ(g) matches π at
        time T(g).
        """
        return IntVector("space", self.circuit_num_gates)

    def _initialize_sigma(self) -> Dict:
        """If at time t, a SWAP gate completes on the edge between qubit indices q1 and q2, then sigma[q1][q2][t]=1

        s_{e,t} - Bool that is *True* **iff** a SWAP finishes on edge e
        at time t (Eq. 5 etc.).

        Stored as σ[p][q][t] with (p,q) the ordered edge tuple present in
        self.available_connectivity.edges.
        """
        sigma: Dict[int, Dict[int, Dict[int, Bool]]] = {}
        for edge in self.available_connectivity.edges:
            if edge[0] not in sigma:
                sigma[edge[0]] = {}
            if edge[1] not in sigma[edge[0]]:
                sigma[edge[0]][edge[1]] = {}
            for timestep in range(self.depth_guess):
                sigma[edge[0]][edge[1]][timestep] = Bool(
                    "sigma_{" + f"e({edge[0]},{edge[1]}),t{timestep}" + "}"
                )
        return sigma

    def _initialize_depth(self) -> Int:
        """
        Global makespan variable D (Eq. 12 - objective)
        """
        return Int("depth")
    # -------------------------------------------------------------------------
    # Constraints generation
    # -------------------------------------------------------------------------
    def _add_constraints(self):
        """
        Emit constraints **Eq. (1) – (11)** (+ Eq. 15 if `hard_island`).

        Each helper below corresponds 1-to-1 with the paper so readers can jump
        straight from the PDF to the code.  See individual doc-strings.
        """
        self.constraint_injective_mapping()              # Eq. 1
        self.constraint_avoid_collisions()               # Eq. 2
        self.constraint_mapping_and_space_consistency()  # Eq. 3 & 4
        self.constraint_no_SWAP_before_swap_duration()   # Eq. 5
        self.constraint_no_SWAP_overlap_same_edge()      # Eq. 6
        self.constraint_no_SWAP_overlap_adjacent_edge()  # Eq. 7
        if not self.transition_based:                    # Eq. 8 & 9 only needed
            self.constraint_no_SWAP_overlap_gates()      #   in full model
        self.constraint_mapping_unchanged()              # Eq. 10
        self.constraint_mapping_transformed()            # Eq. 11
        if self.hard_island:
            self.constraint_no_SWAP_outside_gates()      # Eq. 15 (OLSQ-GA 2022)
    # -------------  Eq. (1) --------------------------------------------------
    def constraint_injective_mapping(self):
        """
        **Injective mapping** — no two logical qubits occupy the same physical
        qubit at the same time  (paper Eq. 1).

        We encode the column-wise “π[:,t] is a permutation” by

            ∑_q  [π[q,t] == p]  ≤  1      for every physical p, time t.
        """
        pi = self.variables["pi"]
        for t in range(self.depth_guess):
            for q1 in range(self.circuit.qubit_count):
                # each π[q1,t] ∈ available_qubits
                self.solver.add(
                    Or(*(pi[q1][t] == p for p in self.device.available_qubits))
                )
                # injective: q1 ≠ q2 ⇒ π[q1,t] ≠ π[q2,t]
                for q2 in range(q1):
                    self.solver.add(pi[q1][t] != pi[q2][t])

    # -------------  Eq. (2) --------------------------------------------------
    def constraint_avoid_collisions(self):
        """
        **No-collision & dependency** constraint (paper Eq. 2).

        For every pair of gates (g₁,g₂) that *share* a logical qubit we require

            T(g₁) < T(g₂)

        unless `transition_based` (TB-OLSQ) in which case ≤ is sufficient
        (multiple commuting gates may share one “transition block”).
        """
        time = self.variables["time"]
        for g1, g2 in self.collision_extraction():
            if self.transition_based:
                self.solver.add(time[g1] <= time[g2])
            else:
                self.solver.add(time[g1] < time[g2])

    def collision_extraction(self):
        """
        Returns a list [(g₁,g₂), …]  where gates g₁,g₂ share ≥1 logical qubit.
        Used exclusively by :py:meth:`constraint_avoid_collisions`.
        """
        collisions = []
        instr = self.circuit.instructions
        for i in range(len(instr)):
            for j in range(i + 1, len(instr)):
                if set(instr[i].gate_indices) & set(instr[j].gate_indices):
                    collisions.append((i, j))
        return collisions


    def constraint_mapping_and_space_consistency(self):
        """
        Consistency between mapping π and gate-location σ (paper Eq. 3 & 4).

            • 1-q gate  g(l)   ->  σ(g) == π[l, T(g)]
            • 2-q gate  g(l₁,l₂)  ->  (π[l₁,T], π[l₂,T])  is an *edge*.
        """
        instructions = self.circuit.instructions
        time = self.variables["time"]
        space = self.variables["space"]
        pi = self.variables["pi"]
        for gate_idx in range(self.circuit_num_gates):
            self.solver.add(time[gate_idx] >= 0, time[gate_idx] < self.depth_guess)
            gate_qubit_indices = instructions[gate_idx].gate_indices
            if len(gate_qubit_indices) == 1:
                valid_space_vals = [
                    space[gate_idx] == qubit_idx
                    for qubit_idx in self.device.available_qubits
                ]
                self.solver.add(Or(valid_space_vals))
                for timestep in range(self.depth_guess):
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
                    for timestep in range(self.depth_guess):
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
        """For all timesteps less than the swap duration of the device, there cannot be any SWAPs completed
        
        "Eq. 5 — no SWAP can **finish** before it has had D cycles to run.
        """
        sigma = self.variables["sigma"]
        for timestep in range(min(self.swap_duration - 1, self.depth_guess)):
            for edge in self.available_connectivity.edges:
                self.solver.add(sigma[edge[0]][edge[1]][timestep] == False)

    def constraint_no_SWAP_overlap_same_edge(self):
        """SWAP gates cannot be applied on the same duration on the same edge
        
        Eq. 6 — two SWAPs on the *same* edge may not overlap in time
        """
        sigma = self.variables["sigma"]
        for timestep in range(self.swap_duration - 1, self.depth_guess):
            for edge in self.available_connectivity.edges:
                for swap_timestep in range(timestep - self.swap_duration + 1, timestep):
                    constraint = Implies(
                        sigma[edge[0]][edge[1]][timestep] == True,
                        sigma[edge[0]][edge[1]][swap_timestep] == False,
                    )
                    self.solver.add(constraint)

    def constraint_no_SWAP_overlap_adjacent_edge(self):
        """Edges that overlap on the device cannot both have overlapping SWAP gates in time"""
        # … unchanged body …
        # ---- STRICT VERSION (uncomment if you want Eq. 11’s “one-swap” rule) -
        # for t in range(self.swap_duration - 1, self.depth_guess):
        #     self.solver.add(
        #         Sum(*(
        #             If(self.variables["sigma"][u][v][t], 1, 0)
        #             for (u,v) in self.available_connectivity.edges
        #         )) <= 1
        #     ) 
        sigma = self.variables["sigma"]
        # adjacent-edge exclusion
        for t in range(self.swap_duration - 1, self.depth_guess):
            for (u, v) in self.available_connectivity.edges:
                for (x, y) in adjacent_edges_from_edge(
                    self.available_connectivity, u, v
                ):
                    for τ in range(max(0, t - self.swap_duration + 1), t + 1):
                        self.solver.add(
                            Implies(
                                sigma[u][v][t],
                                Not(sigma[min(x,y)][max(x,y)][τ]),
                            )
                        )
        # optional global single-swap rule
        if not self.allow_parallel:
            for t in range(self.swap_duration - 1, self.depth_guess):
                self.solver.add(
                    Sum(*(If(sigma[u][v][t], 1, 0)
                          for (u, v) in self.available_connectivity.edges)) <= 1
                )
    # def constraint_no_SWAP_overlap_adjacent_edge(self):
    #     """Edges that overlap on the device cannot both have overlapping SWAP gates in time"""
    #     # … unchanged body …
    #     # ---- STRICT VERSION (uncomment if you want Eq. 11’s “one-swap” rule) -
    #     # for t in range(self.swap_duration - 1, self.depth_guess):
    #     #     self.solver.add(
    #     #         Sum(*(
    #     #             If(self.variables["sigma"][u][v][t], 1, 0)
    #     #             for (u,v) in self.available_connectivity.edges
    #     #         )) <= 1
    #     #     ) 
    #     sigma = self.variables["sigma"]
    #     for timestep in range(self.swap_duration - 1, self.depth_guess):
    #         for edge in self.available_connectivity.edges:
    #             for swap_timestep in range(
    #                 timestep - self.swap_duration + 1, timestep + 1
    #             ):
    #                 for adjacent_edge in adjacent_edges_from_edge(
    #                     self.available_connectivity, edge[0], edge[1]
    #                 ):
    #                     imply_operand1 = sigma[edge[0]][edge[1]][timestep] == True
    #                     try:
    #                         imply_operand2 = (
    #                             sigma[adjacent_edge[0]][adjacent_edge[1]][swap_timestep]
    #                             == False
    #                         )
    #                     except KeyError:
    #                         imply_operand2 = (
    #                             sigma[adjacent_edge[1]][adjacent_edge[0]][swap_timestep]
    #                             == False
    #                         )
    #                     constraint = Implies(imply_operand1, imply_operand2)
    #                     self.solver.add(constraint)

    def constraint_no_SWAP_overlap_gates(self):
        """Eq. 8 & 9 — SWAP cannot overlap any gate touching its qubits."""
        instructions = self.circuit.instructions
        time = self.variables["time"]
        space = self.variables["space"]
        sigma = self.variables["sigma"]
        for timestep in range(self.swap_duration - 1, self.depth_guess):
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
    # -------------  Eq. (10) – (11) -----------------------------------------
    def constraint_mapping_unchanged(self):
        """
        Eq. 10 — if **no SWAP finishes** at time t, the mapping is unchanged
        between t and t+1.
        """
        pi = self.variables["pi"]
        sigma = self.variables["sigma"]
        for timestep in range(self.depth_guess - 1):
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
        """
        Eq. 11 — if a SWAP on edge (p,q) finishes at t, then at t+1 the logical
        qubits at p and q are exchanged (all others unchanged).
        """
        pi = self.variables["pi"]
        sigma = self.variables["sigma"]
        for timestep in range(self.depth_guess - 1):
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
    # -------------  Eq. (15) [OLSQ-GA 2022] ---------------------------------
    def constraint_no_SWAP_outside_gates(self):
        """
        Hard-island constraint (OLSQ-GA 2022, Eq. 15).

        Forbids relay SWAPs that touch *any* qubit outside the initial island
        defined by π[:,0].  Useful when the user wants strict locality.
        """
        pi    = self.variables["pi"]
        sigma = self.variables["sigma"]
        L     = self.circuit.qubit_count
        T     = self.depth_guess
        island_elems = [pi[q][0] for q in range(L)]          # symbolic list

        # helper  bool “InIsland(x)”
        def in_island(x):
            return Or(*[x == p0 for p0 in island_elems])

        # (a) mapping never leaves the island
        for q in range(L):
            for t in range(1, T):
                self.solver.add(in_island(pi[q][t]))

        # (b) every SWAP edge completely inside the island
        for (u, v) in self.available_connectivity.edges:
            for t in range(T):
                self.solver.add(
                    Implies(sigma[u][v][t],
                            And(in_island(u), in_island(v)))
                )
       

    # -------------------------------------------------------------------------
    # Objective
    # -------------------------------------------------------------------------
    def _add_optimization_objective(self):
        """
        Adds the optimization objective to the solver. This
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
        
        elif self.objective == "swap":
            n_swaps = self.variables["n_swaps"]

            # depth is still linked to every gate finish time
            for g in range(self.circuit_num_gates):
                self.solver.add(depth >= time[g] + 1)

            # keep depth within the search bound
            self.solver.add(depth <= self.depth_guess)

            # primary goal: minimise swaps
            self.solver.minimize(n_swaps)

            # secondary tie-breaker: minimise depth too
            self.solver.minimize(depth)

    def post_process(self):
        pi = self.variables["pi"]
        depth = self.variables["depth"]
        self.results = dict()

        model = self.solver.model()
        self.results["depth"] = model[depth].as_long()
        self.results["time"] = self.get_time_results()
        self.results["SWAPs"] = self.get_SWAP_results()
        self.results["n_swaps"]= len(self.results["SWAPs"])
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
    

        # tests expect `compiled` in the meta-dict
        meta = {
            "compiled":result_circuit,
            "D": objective_result,
            "T": self.results["time"],
            "S": self.results["SWAPs"],
            "n_swaps":  self.results["n_swaps"],
            "P0":        initial_qubit_map,
            "Pf":        final_qubit_map,

        }
       

        
        # # reset self.depth_guess 
        # self.depth_guess = self.circuit_depth_guess
        result_circuit.meta = meta
        del self.results
        return (result_circuit, initial_qubit_map, final_qubit_map, objective_result, meta)

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
                    for sub in self.device.swap_decomposition(instruction.gate_indices):
                        result_circuit.add_instruction(sub)


  
                else:
                    result_circuit.add_instruction(instruction)

        return result_circuit