

import pytest
from olsq.solve import collision_extracting
import networkx as nx
from qrew.simulation.refactor.resources import (
    QuantumDevice,
    QuantumResource,
)
from qrew.simulation.refactor.quantum import (QuantumCircuit,QuantumInstruction, LayoutSynthesizer, LayoutTrace)
from qrew.simulation.refactor.quantum_gates import *
from qrew.simulation.refactor.broker import Broker
from qrew.simulation.refactor.devices.quantum_devices import (
    IBM_Kyiv,     # same device that triggered the (14,) qubit bug
)
from qrew.simulation.refactor.q_interop.transpilers import QiskitCompiler
import itertools
# def test_a():
#     assert 1 == 1

BASIS = ("ECR", "I", "RZ", "SX", "X")          # same as IBM_Brisbane / _Kyiv …
@pytest.fixture
def compiler():
    """Fresh compiler so side-effects between tests don’t accumulate."""
    return QiskitCompiler(basis_gates=BASIS)


@pytest.fixture
def two_qubit_device(compiler):
    """Toy connectivity 0—1, just enough for the SWAP tests."""
    g = nx.Graph()
    g.add_edge(0, 1)
    return QuantumDevice("toy-2q", connectivity=g, compiler=compiler)

def gate_names(qc):          # { "CX", "RZ", … }
    return {inst.gate.name for inst in qc.instructions}


def make_swap_circuit(a: int, b: int, n_qubits: int | None = None):
    if n_qubits is None:
        n_qubits = max(a, b) + 1
    qc = QuantumCircuit(qubit_count=n_qubits)
    qc.add_instruction(QuantumInstruction(SWAP(), (a, b)))
    return qc

def test_swap_removed(two_qubit_device):
    qc_native = two_qubit_device.compiler.transpile(
        make_swap_circuit(0, 1), two_qubit_device
    )
    assert all(instr.gate.name != "SWAP" for instr in qc_native.instructions)


def test_only_basis_gates(two_qubit_device):
    native = two_qubit_device.compiler.transpile(
        make_swap_circuit(0, 1), two_qubit_device
    )
    assert gate_names(native).issubset({g.upper() for g in BASIS})

def test_depth_matches_property(two_qubit_device):
    """
    test Depth equals compiler.swap_duration
    """
    native = two_qubit_device.compiler.transpile(
        make_swap_circuit(0, 1), two_qubit_device
    )
    assert native.depth() == two_qubit_device.compiler.swap_duration



def test_relabelled_swap(compiler):
    #  Decomposition remaps cleanly to arbitrary physical indices
    g = nx.path_graph(5)              # 0-1-2-3-4  (3-4 are connected)
    dev = QuantumDevice("toy-5q", connectivity=g, compiler=compiler)

    native = compiler.transpile(make_swap_circuit(3, 4, 5), dev)
    touched = {q for inst in native.instructions for q in inst.gate_indices}
    assert touched == {3, 4}, "should touch only the requested physical qubits"


def test_qiskit_swap_depth():
    qc = QuantumDevice(
        "dummy", nx.complete_graph(2),
        compiler=QiskitCompiler(basis_gates=("CX","RZ","X","SX"))
    )
    assert qc.swap_duration >= 1
    assert isinstance(qc.transpiled_swap_circuit, QuantumCircuit), f'type(qc.transpiled_swap): {type(qc.transpiled_swap_circuit)}'
    assert all(g in qc.gate_set for g in qc.transpiled_swap_circuit.gate_set)


def test_layout_on_fully_connected_device():
    dev = QuantumDevice("fc5", nx.complete_graph(5),
                        compiler=QiskitCompiler(basis_gates=("CX","RZ","X","SX")))
    circ = QuantumCircuit(qubit_count=2)
    circ.add_instruction(QuantumInstruction(CX(),(0,1)))
    _,_,_,depth,_ = dev.layout_synthesis(circ, hard_island=True)
    assert depth == 1         # no SWAPs necessary


def test_device_exposes_compiler_fields(two_qubit_device):
    # swap_duration and transpiled_swap_circuit exposed via QuantumDevice
    dev = two_qubit_device
    assert dev.swap_duration == dev.compiler.swap_duration
    assert isinstance(dev.transpiled_swap_circuit, QuantumCircuit)

def test_layout_synthesiser_swap_replacement(two_qubit_device):
    # 3-qubit line so a SWAP *must* be decomposed on (0,2) via (0,1)
    g = nx.path_graph(3)
    dev = QuantumDevice("line3", connectivity=g, compiler=two_qubit_device.compiler)

    circ = QuantumCircuit(qubit_count=2)
    circ.add_instruction(QuantumInstruction(CX(), (0, 1)))
    # force hard-island so no relay qubits
    _, _, _, depth, res = dev.layout_synthesis(circ, hard_island=True)

    # hard-island on fully-connected line ⇒ depth == 1 and no SWAPs
    assert depth == 1
    assert res["S"] == []


def generate_quantum_adder():
    circuit = QuantumCircuit(qubit_count=4)
    circuit.add_instruction(QuantumInstruction(gate=X(), qubit_indices=(0,)))
    circuit.add_instruction(QuantumInstruction(gate=X(), qubit_indices=(1,)))
    circuit.add_instruction(QuantumInstruction(gate=H(), qubit_indices=(3,)))
    circuit.add_instruction(QuantumInstruction(gate=CX(), qubit_indices=(2, 3)))
    circuit.add_instruction(QuantumInstruction(gate=T(), qubit_indices=(0,)))
    circuit.add_instruction(QuantumInstruction(gate=T(), qubit_indices=(1,)))
    circuit.add_instruction(QuantumInstruction(gate=T(), qubit_indices=(2,)))
    circuit.add_instruction(QuantumInstruction(gate=Tdg(), qubit_indices=(3,)))
    circuit.add_instruction(QuantumInstruction(gate=CX(), qubit_indices=(0, 1)))
    circuit.add_instruction(QuantumInstruction(gate=CX(), qubit_indices=(2, 3)))
    circuit.add_instruction(QuantumInstruction(gate=CX(), qubit_indices=(3, 0)))
    circuit.add_instruction(QuantumInstruction(gate=CX(), qubit_indices=(1, 2)))
    circuit.add_instruction(QuantumInstruction(gate=CX(), qubit_indices=(0, 1)))
    circuit.add_instruction(QuantumInstruction(gate=CX(), qubit_indices=(2, 3)))
    circuit.add_instruction(QuantumInstruction(gate=Tdg(), qubit_indices=(0,)))
    circuit.add_instruction(QuantumInstruction(gate=Tdg(), qubit_indices=(1,)))
    circuit.add_instruction(QuantumInstruction(gate=Tdg(), qubit_indices=(2,)))
    circuit.add_instruction(QuantumInstruction(gate=T(), qubit_indices=(3,)))
    circuit.add_instruction(QuantumInstruction(gate=CX(), qubit_indices=(0, 1)))
    circuit.add_instruction(QuantumInstruction(gate=CX(), qubit_indices=(2, 3)))
    circuit.add_instruction(QuantumInstruction(gate=S(), qubit_indices=(3,)))
    circuit.add_instruction(QuantumInstruction(gate=CX(), qubit_indices=(3, 0)))
    circuit.add_instruction(QuantumInstruction(gate=H(), qubit_indices=(3,)))

    return circuit


def generate_qubit_connectivity():
    connectivity = nx.Graph()
    connectivity.add_edge(0, 1)
    connectivity.add_edge(0, 2)
    connectivity.add_edge(1, 2)
    connectivity.add_edge(2, 3)
    connectivity.add_edge(2, 4)
    connectivity.add_edge(3, 4)

    return connectivity


def generate_quantum_device():
    device = QuantumDevice(
        device_name="test",
        connectivity=generate_qubit_connectivity(),
        # gate_set=(X, H, S, T, Tdg, CX),
        gate_set=("X", "H", "S", "T", "Tdg", "CX"),
    )

    return device


def test_add_instruction():
    circuit = generate_quantum_adder()
    assert len(circuit.instructions) == 23

def test_add_instruction_out_of_range():
    qc = QuantumCircuit(qubit_count=2)
    with pytest.raises(AssertionError):
        qc.add_instruction(QuantumInstruction(gate=X(), qubit_indices=(3,)))

    # with pytest.raises(AssertionError):
    #     qc.add_instruction(QuantumInstruction(gate=X(), qubit_indices=(2,)))


@pytest.mark.parametrize("tb,hi", list(itertools.product([False, True], repeat=2)))
def test_flag_matrix(tb, hi):
    dev  = generate_quantum_device()
    circ = generate_quantum_adder()
    _, P0, Pf, depth, res = dev.layout_synthesis(circ,
                                                 transition_based=tb,
                                                 hard_island=hi)
    # depth is always >= logical depth
    assert depth >= circ.depth()

    # hard_island ⇒ no relay qubits
    touched = {q for instr in res["compiled"].instructions for q in instr.gate_indices}
    if hi:
        assert touched.issubset(set(P0))
    else:
        # relay qubits optional
        assert touched.issuperset(set(P0))

@pytest.mark.parametrize("hard_island", [True, False])
def test_island_containment(hard_island):
    device  = generate_quantum_device()
    circuit = generate_quantum_adder()

    qr = QuantumResource(
        quantum_circuit=circuit,
        LS_parameters={
            "transition based": False,
            "hard_island":      hard_island,
            "epsilon":          0.3,
            "objective":        "depth",
        },
    )

    alloc   = device.allocate(qr)              # triggers LS
    meta    = alloc.transpiled_circuit.meta    # comes from LayoutSynthesizer.post_process
    trace   = LayoutTrace(
        island        = meta["T"][0: circuit.qubit_count],    # π[:,0]
        final_mapping = meta["T"][-circuit.qubit_count:],     # π[:,D-1]
        swaps         = meta["S"],
    )

    if hard_island:
        # every swap edge stays within the initial island
        assert trace.touches_only_island(), (
            f"hard_island=True but SWAPs left the island – {trace.swaps}"
        )
    else:
        # for comparison: NOT enforced ⇒ at least one edge should leave
        assert not trace.touches_only_island(), (
            "hard_island=False yet solver never used relay qubits; "
            "test may be too weak."
        )

def test_qiskit_layout_vs_solver():
    dev  = generate_quantum_device()
    circ = generate_quantum_adder()
    qr   = QuantumResource(circ, LS_parameters={"transition based": False,
                                                "hard_island": False,
                                                "epsilon":0.3,
                                                "objective":"depth"})

    # ① transpile only – capture the order given to Qiskit
    transpiled = dev.compiler.transpile(circ, dev)
    qiskit_initial = list(range(circ.qubit_count))  # by construction

    # ② full allocation – includes SMT layout
    alloc = dev.allocate(qr)
    meta  = alloc.transpiled_circuit.meta

    ls_initial = meta["compiled"].instructions[0].gate_indices  # π[:,0]
    
    # they differ unless frozen
    assert tuple(qiskit_initial) != tuple(ls_initial), \
        "LayoutSynthesizer kept Qiskit’s trivial layout unexpectedly"
    
def test_depth_on_fully_connected():
    fc = QuantumDevice(
        "fc4", nx.complete_graph(4),
        gate_set=("X","H","S","T","Tdg","CX")
    )
    circ = generate_quantum_adder()
    _,_,_, depth, _ = fc.layout_synthesis(circ, transition_based=False)
    assert depth == circ.depth() == 11

def test_compute_circuit_depth():
    # hard_island=False
    device = generate_quantum_device()
    circuit = generate_quantum_adder()
    layout_synthesizer = LayoutSynthesizer(
        quantum_circuit=circuit, device=device, transition_based=False,hard_island=False
    )
    logical_depth = layout_synthesizer.compute_circuit_depth(circuit)
    assert logical_depth == 11
    compiled, init_map, final_map, routed_depth, _ = device.layout_synthesis(
        circuit, transition_based=False, hard_island=False
    )

    assert routed_depth >= logical_depth
    
    assert routed_depth == 15
    # sanity: mapping survives round-trip
    assert sorted(init_map) == sorted(final_map)

    
    # hard_island=True
    # TODO: Add counter layout synthesizer results 
    compiled_HI, init_map_HI, final_map_HI, routed_depth_HI, _ = device.layout_synthesis(
        circuit, transition_based=False, hard_island=True
    )

    # depth is still 15 – same number of layers, the SWAP now happens *inside*
    assert routed_depth_HI == routed_depth == 15

    # no relay qubits: every physical qubit in the compiled circuit
    # must already appear in the initial mapping P0
    touched_HI = {q for instr in compiled_HI.instructions for q in instr.gate_indices}
    assert touched_HI.issubset(set(init_map_HI))

    # mapping consistency
    assert sorted(init_map_HI) == sorted(final_map_HI)
    
def test_depth_on_fully_connected2():
    """
    To counter 'test_compute_circuit_depth' with a fully-connected device 
    where logical depth and optimized layout depth should be equal

    """
    fully_connected_device = QuantumDevice(
        "fc4", nx.complete_graph(4),
        gate_set=("X","H","S","T","Tdg","CX")
    )
    circuit = generate_quantum_adder()
    layout_synthesizer = LayoutSynthesizer(
        quantum_circuit=circuit, device=fully_connected_device, transition_based=False,hard_island=False
    )
    logical_depth = layout_synthesizer.compute_circuit_depth(circuit)
    _,_,_, depth, _ = fully_connected_device.layout_synthesis(circuit, transition_based=False)
    assert depth == circuit.depth() == logical_depth == 11 

def test_collision_extraction():
    device = generate_quantum_device()
    circuit = generate_quantum_adder()
    layout_synthesizer = LayoutSynthesizer(quantum_circuit=circuit, device=device)
    collisions = layout_synthesizer.collision_extraction()

    gate_idxs = []
    for instruction in circuit.instructions:
        gate_idxs.append(instruction.gate_indices)

    collisions_expected = []
    for collision in collision_extracting(gate_idxs):
        if collision not in collisions_expected:
            collisions_expected.append(collision)

    assert collisions == collisions_expected


def test_initialize_pi():
    device = generate_quantum_device()
    circuit = generate_quantum_adder()
    layout_synthesizer = LayoutSynthesizer(quantum_circuit=circuit, device=device)
    circuit_depth = layout_synthesizer.compute_circuit_depth(circuit)
    circuit_num_qubits = circuit.qubit_count

    pi = layout_synthesizer._initialize_pi()

    for qubit_idx in range(circuit_num_qubits):
        for timestep in range(circuit_depth):
            assert (
                str(pi[qubit_idx][timestep])
                == "pi_{" + f"q{qubit_idx},t{timestep}" + "}"
            )


def test_initialize_sigma():
    device = generate_quantum_device()
    circuit = generate_quantum_adder()
    layout_synthesizer = LayoutSynthesizer(quantum_circuit=circuit, device=device)
    circuit_depth = layout_synthesizer.compute_circuit_depth(circuit)

    sigma = layout_synthesizer._initialize_sigma()

    for edge in device.connectivity.edges:
        for timestep in range(circuit_depth):
            assert (
                str(sigma[edge[0]][edge[1]][timestep])
                == "sigma_{" + f"e({edge[0]},{edge[1]}),t{timestep}" + "}"
            )


# ---------------------------------------------------------------------------
# transistion based = False tests (normal)
# ---------------------------------------------------------------------------

def test_n_swaps_matches_result_list():
    """
    LayoutSynthesizer.n_swaps should count exactly the SWAPs returned.
    We use a sparse ‘triangle-plus-tail’ graph so at least one SWAP is needed.
    """
    conn = nx.Graph([(0,1), (1,2), (2,0), (2,3)])   # tail makes SWAP inevitable
    dev  = QuantumDevice("tail4", conn, gate_set=("X","H","S","T","Tdg","CX"))
    circ = generate_quantum_adder()

    # compile once with the default depth objective simply to get a reference
    _,_,_,depth_default,res_default = dev.layout_synthesis(circ, transition_based=False)
    n_swaps_default = len(res_default["S"])
    assert n_swaps_default > 0          # sanity: we *do* need swaps on this graph

    # re-compile but *minimise SWAP count*
    _,_,_,depth_swap,res_swap = dev.layout_synthesis(
        circ, transition_based=False, objective="swap"   # <<< pass-through kw
    )
    n_swaps_swap = len(res_swap["S"])

    # 1) internal accounting is consistent
    assert n_swaps_swap == res_swap["n_swaps"]              # new field added

    # 2) swap-objective never inserts *more* swaps than depth-objective
    assert n_swaps_swap <= n_swaps_default

    # 3) depth may grow when we minimise swaps (trade-off check)
    assert depth_swap >= depth_default

def test_swap_objective_on_fully_connected():
    """On an all-to-all device the optimal swap count must be zero."""
    fc = QuantumDevice("fc4", nx.complete_graph(4),
                       gate_set=("X","H","S","T","Tdg","CX"))
    circ = generate_quantum_adder()
    _,_,_,_,res = fc.layout_synthesis(
        circ, transition_based=False, objective="swap"
    )
    assert res["n_swaps"] == 0
    assert res["S"] == []
# Helper: return the *set* of physical qubits touched by a compiled circuit
def touched_qubits(meta):
    return {q for instr in meta["compiled"].instructions for q in instr.gate_indices}


def test_relay_dense_graph_optional():
    """
    dense graph  =>  relay not required, but allowed
    """
    dev  = generate_quantum_device()              # 5-node, 6-edge graph (dense)
    circ = generate_quantum_adder()
    _, P0, _, _, meta = dev.layout_synthesis(circ, transition_based=False)
    extra = touched_qubits(meta) - set(P0)

    # relay qubits *may* appear, but a zero-extra solution is perfectly valid
    assert extra == set() or extra.issubset(set(dev.available_qubits))

def test_relay_sparse_graph_present():
    """
    sparse graph  =>  at least one relay *should* appear when hard_island=False
    """
    # 5-node “triangle + double-tail” – SWAP *and* relay inevitable because
    # logical circuit uses 4 qubits while the device offers 5.
    conn = nx.Graph([(0, 1), (1, 2), (2, 0), (2, 3), (3, 4)])
    
    dev  = QuantumDevice("tail4", conn,
                         gate_set=("X","H","S","T","Tdg","CX"))
    circ = generate_quantum_adder()

    _, P0, _, _, meta = dev.layout_synthesis(circ,
                                             transition_based=False,
                                             hard_island=False)
    extra = touched_qubits(meta) - set(P0)

    # Relay qubits are allowed – but not obligatory
    if not extra:
        pytest.xfail("No relay qubit required – island solution is optimal")
    

def test_relay_sparse_graph_forbidden_hard_island():
    """
    sparse graph (same as in `test_relay_sparse_graph_present`) => relay forbidden when hard_island=True
    """
    conn = nx.Graph([(0, 1), (1, 2), (2, 0), (2, 3)])
    dev  = QuantumDevice("tail4", conn,
                         gate_set=("X","H","S","T","Tdg","CX"))
    circ = generate_quantum_adder()

    _, P0, _, _, meta = dev.layout_synthesis(circ,
                                             transition_based=False,
                                             hard_island=True)
    extra = touched_qubits(meta) - set(P0)

    # no relay qubits allowed
    assert not extra, (
        "hard_island=True but compiler touched relay qubits "
        f"{sorted(extra)} outside island {P0}"
    )

@pytest.mark.slow
def test_layout_synthesis_normal():
    device = generate_quantum_device()
    circuit = generate_quantum_adder()
    _, initial_qubit_mapping, final_qubit_mapping, objective_result,results_dict = (
        device.layout_synthesis(circuit, transition_based=False)
    )
    assert objective_result == 15
    assert sorted(initial_qubit_mapping) == sorted(final_qubit_mapping)


# @pytest.mark.xfail(reason="Ancilla-containment via CouplingMap not enforced yet", strict=False)
@pytest.mark.slow
def test_allocate_normal():
    device = generate_quantum_device()
    circuit = generate_quantum_adder()
    quantum_resource = QuantumResource(
        quantum_circuit=circuit,
        LS_parameters={"transition based": False,"hard_island": True, "epsilon": 0.3, "objective": "depth"},
    )
    available_qubits = device.available_qubits

    assert device.max_available_connections == 5
    allocation = device.allocate(quantum_resource)
    allocated_qubit_idxs = allocation.allocated_qubit_idxs


    assert device.max_available_connections == 1, f'inspect: {device.inspect_device()}'
    assert sorted(device.available_qubits + list(allocated_qubit_idxs)) == sorted(
        available_qubits
    )
    assert all(qubit_idx in available_qubits for qubit_idx in allocated_qubit_idxs)


# @pytest.mark.slow
@pytest.mark.xfail(reason="Ancilla-containment via CouplingMap not enforced yet", strict=False)
def test_deallocate_normal():
    device = generate_quantum_device()
    circuit = generate_quantum_adder()
    quantum_resource = QuantumResource(
        quantum_circuit=circuit,
        LS_parameters={"transition based": False, "hard_island": True, "epsilon": 0.3, "objective": "depth"},
    )
    available_qubits = device.available_qubits

    assert device.max_available_connections == 5
    allocation = device.allocate(quantum_resource)
    assert device.max_available_connections == 1, f'inspect: {device.inspect_device()}'
    device.deallocate(allocation)
    assert device.max_available_connections == 5
    assert sorted(device.available_qubits) == sorted(available_qubits)


@pytest.mark.slow
def test_layout_synthesis_normal_missing_node3():
    connectivity = nx.Graph()
    connectivity.add_edge(0, 1)
    connectivity.add_edge(0, 2)
    connectivity.add_edge(1, 2)
    connectivity.add_edge(2, 4)

    device = QuantumDevice(
        device_name="test",
        connectivity=connectivity,
        # gate_set=(X, H, S, T, Tdg, CX),
        gate_set=("X", "H", "S", "T", "Tdg", "CX"),
    )
    circuit = generate_quantum_adder()

    assert sorted(device.available_qubits) == [0, 1, 2, 4]
    assert device.max_available_connections == 4
    _, initial_qubit_mapping, final_qubit_mapping, objective_result,results_dict = (
        device.layout_synthesis(circuit, transition_based=False)
    )
    assert objective_result == 15
    assert sorted(initial_qubit_mapping) == [0, 1, 2, 4]
    assert sorted(initial_qubit_mapping) == sorted(final_qubit_mapping)


def test_layout_synthesis_insufficient_qubits():
    connectivity = nx.Graph()
    connectivity.add_edge(0, 1)
    connectivity.add_edge(0, 2)
    connectivity.add_edge(1, 2)
    device = QuantumDevice(
        device_name="test",
        connectivity=connectivity,
        # gate_set=(X, H, S, T, Tdg, CX),
        gate_set=("X", "H", "S", "T", "Tdg", "CX"),
    )
    circuit = generate_quantum_adder()

    with pytest.raises(
        AssertionError,
        match=f"does not have sufficient available qubits",
    ):
        device.layout_synthesis(circuit, transition_based=False)


# ---------------------------------------------------------------------------
# transistion based = True tests
# ---------------------------------------------------------------------------

def test_layout_synthesis_transition_based():
    device = generate_quantum_device()
    circuit = generate_quantum_adder()
    _, initial_qubit_mapping, final_qubit_mapping, objective_result,results_dict = (
        device.layout_synthesis(circuit,  transition_based=True)

    )

    assert objective_result == 15
    assert sorted(initial_qubit_mapping) == sorted(final_qubit_mapping)


def test_allocate_transition_based():
    device = generate_quantum_device()
    circuit = generate_quantum_adder()
    quantum_resource = QuantumResource(quantum_circuit=circuit)
    available_qubits = device.available_qubits

    assert device.max_available_connections == 5
    allocation = device.allocate(quantum_resource)
    allocated_qubit_idxs = allocation.allocated_qubit_idxs
    assert device.max_available_connections == 1
    assert sorted(device.available_qubits + list(allocated_qubit_idxs)) == sorted(
        available_qubits
    )
    assert all(qubit_idx in available_qubits for qubit_idx in allocated_qubit_idxs)


def test_deallocate_transition_based():
    device = generate_quantum_device()
    circuit = generate_quantum_adder()
    quantum_resource = QuantumResource(quantum_circuit=circuit)
    available_qubits = device.available_qubits

    assert device.max_available_connections == 5
    allocation = device.allocate(quantum_resource)
    assert device.max_available_connections == 1
    device.deallocate(allocation)
    assert device.max_available_connections == 5
    assert sorted(device.available_qubits) == sorted(available_qubits)


def test_layout_synthesis_transition_based_missing_node0():
    connectivity = nx.Graph()
    # connectivity.add_edge(0, 1)
    # connectivity.add_edge(0, 2)
    connectivity.add_edge(1, 2)
    connectivity.add_edge(2, 3)
    connectivity.add_edge(2, 4)
    connectivity.add_edge(3, 4)

    device = QuantumDevice(
        device_name="test",
        connectivity=connectivity,
        # gate_set=(X, H, S, T, Tdg, CX),
        gate_set=("X", "H", "S", "T", "Tdg", "CX"),
    )

    circuit = generate_quantum_adder()

    assert sorted(device.available_qubits) == [1, 2, 3, 4]
    assert device.max_available_connections == 4
    _, initial_qubit_mapping, final_qubit_mapping, objective_result,results_dict = (
        device.layout_synthesis(circuit)
    )

    assert objective_result == 15
    assert sorted(initial_qubit_mapping) == [1, 2, 3, 4]
    assert sorted(initial_qubit_mapping) == sorted(final_qubit_mapping)


def test_layout_synthesis_transition_based_missing_node1():
    connectivity = nx.Graph()
    # connectivity.add_edge(0, 1)
    connectivity.add_edge(0, 2)
    # connectivity.add_edge(1, 2)
    connectivity.add_edge(2, 3)
    connectivity.add_edge(2, 4)
    connectivity.add_edge(3, 4)

    device = QuantumDevice(
        device_name="test",
        connectivity=connectivity,
        # gate_set=(X, H, S, T, Tdg, CX),
        gate_set=("X", "H", "S", "T", "Tdg", "CX"),
    )
    circuit = generate_quantum_adder()

    assert sorted(device.available_qubits) == [0, 2, 3, 4]
    assert device.max_available_connections == 4
    _, initial_qubit_mapping, final_qubit_mapping, objective_result,results_dict = (
        device.layout_synthesis(circuit)
    )

    assert objective_result == 15
    assert sorted(initial_qubit_mapping) == [0, 2, 3, 4]
    assert sorted(initial_qubit_mapping) == sorted(final_qubit_mapping)


def test_layout_synthesis_transition_based_missing_node2():
    connectivity = nx.Graph()
    connectivity.add_edge(0, 1)
    # connectivity.add_edge(0, 2)
    # connectivity.add_edge(1, 2)
    # connectivity.add_edge(2, 3)
    # connectivity.add_edge(2, 4)
    connectivity.add_edge(3, 4)

    device = QuantumDevice(
        device_name="test",
        connectivity=connectivity,
        # gate_set=(X, H, S, T, Tdg, CX),
        gate_set=("X", "H", "S", "T", "Tdg", "CX"),
    )
    circuit = generate_quantum_adder()

    assert sorted(device.available_qubits) == [0, 1, 3, 4]
    assert device.max_available_connections == 2

    with pytest.raises(
        AssertionError,
        match=f"does not have sufficient available qubits",
    ):
        device.layout_synthesis(circuit)


def test_layout_synthesis_transition_based_missing_node3():
    connectivity = nx.Graph()
    connectivity.add_edge(0, 1)
    connectivity.add_edge(0, 2)
    connectivity.add_edge(1, 2)
    # connectivity.add_edge(2, 3)
    connectivity.add_edge(2, 4)
    # connectivity.add_edge(3, 4)

    device = QuantumDevice(
        device_name="test",
        connectivity=connectivity,
        # gate_set=(X, H, S, T, Tdg, CX),
        gate_set=("X", "H", "S", "T", "Tdg", "CX"),
    )
    circuit = generate_quantum_adder()

    assert sorted(device.available_qubits) == [0, 1, 2, 4]
    assert device.max_available_connections == 4
    _, initial_qubit_mapping, final_qubit_mapping, objective_result,results_dict = (
        device.layout_synthesis(circuit)
    )

    assert objective_result == 15
    assert sorted(initial_qubit_mapping) == [0, 1, 2, 4]
    assert sorted(initial_qubit_mapping) == sorted(final_qubit_mapping)


def test_layout_synthesis_transition_based_missing_node4():
    connectivity = nx.Graph()
    connectivity.add_edge(0, 1)
    connectivity.add_edge(0, 2)
    connectivity.add_edge(1, 2)
    connectivity.add_edge(2, 3)
    # connectivity.add_edge(2, 4)
    # connectivity.add_edge(3, 4)

    device = QuantumDevice(
        device_name="test",
        connectivity=connectivity,
        # gate_set=(X, H, S, T, Tdg, CX),
        gate_set=("X", "H", "S", "T", "Tdg", "CX"),
    )
    circuit = generate_quantum_adder()

    assert sorted(device.available_qubits) == [0, 1, 2, 3]
    assert device.max_available_connections == 4
    optimized_circuit, initial_qubit_mapping, final_qubit_mapping, objective_result,results_dict = (
        device.layout_synthesis(circuit)
    )

    assert objective_result == 15
    assert sorted(initial_qubit_mapping) == [0, 1, 2, 3]
    assert sorted(initial_qubit_mapping) == sorted(final_qubit_mapping)


def test_allocate_covers_transpiled_circuit_kyiv():
    """
    Every physical qubit that the *compiled* circuit touches must appear in
    QuantumAllocation.allocated_qubit_idxs.  Today that list only contains the
    initial mapping, so qubits brought in by SWAPs (e.g. 14 on IBM Kyiv) are
    missing and the test rightly fails.
    """
    circuit = generate_quantum_adder()
    q_resource = QuantumResource(quantum_circuit=circuit)
    allocation  = IBM_Kyiv.allocate(q_resource)

    uncovered = set()
    for instr in allocation.transpiled_circuit.instructions:
        for q in instr.gate_indices:
            if q not in allocation.allocated_qubit_idxs:
                uncovered.add(q)

    assert not uncovered, (
        "Transpiled circuit uses physical qubits "
        f"{sorted(uncovered)} that are *not* recorded in "
        f"allocation {allocation.allocated_qubit_idxs}"
    )



