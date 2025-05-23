

import pytest
from olsq.solve import collision_extracting
import networkx as nx
from qrew.simulation.refactor.resources import (
    QuantumDevice,
    QuantumResource,
)
from qrew.simulation.refactor.quantum import QuantumCircuit,QuantumInstruction, LayoutSynthesizer
from qrew.simulation.refactor.quantum_gates import *
from qrew.simulation.refactor.broker import Broker
from qrew.simulation.refactor.devices.quantum_devices import (
    IBM_Kyiv,     # same device that triggered the (14,) qubit bug
)
import itertools
# def test_a():
#     assert 1 == 1


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
    
def test_depth_on_fully_connected():
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


@pytest.mark.slow
def test_layout_synthesis_normal():
    device = generate_quantum_device()
    circuit = generate_quantum_adder()
    _, initial_qubit_mapping, final_qubit_mapping, objective_result,results_dict = (
        device.layout_synthesis(circuit, transition_based=False)
    )
    assert objective_result == 15
    assert sorted(initial_qubit_mapping) == sorted(final_qubit_mapping)


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


# ---------------------------------------------------------------------------
# 1) dense graph  ⇒  relay not required, but allowed
# ---------------------------------------------------------------------------
def test_relay_dense_graph_optional():
    dev  = generate_quantum_device()              # 5-node, 6-edge graph (dense)
    circ = generate_quantum_adder()
    _, P0, _, _, meta = dev.layout_synthesis(circ, transition_based=False)
    extra = touched_qubits(meta) - set(P0)

    # relay qubits *may* appear, but a zero-extra solution is perfectly valid
    assert extra == set() or extra.issubset(set(dev.available_qubits))


# ---------------------------------------------------------------------------
# 2) sparse graph  ⇒  at least one relay *should* appear when hard_island=False
# ---------------------------------------------------------------------------
def test_relay_sparse_graph_present():
    # 4-node “triangle + tail” makes at least one SWAP inevitable
    conn = nx.Graph([(0, 1), (1, 2), (2, 0), (2, 3)])
    dev  = QuantumDevice("tail4", conn,
                         gate_set=("X","H","S","T","Tdg","CX"))
    circ = generate_quantum_adder()

    _, P0, _, _, meta = dev.layout_synthesis(circ,
                                             transition_based=False,
                                             hard_island=False)
    extra = touched_qubits(meta) - set(P0)

    # relay qubits expected
    assert extra, "sparse graph should trigger at least one relay qubit"


# ---------------------------------------------------------------------------
# 3) same sparse graph  ⇒  relay forbidden when hard_island=True
# ---------------------------------------------------------------------------
def test_relay_sparse_graph_forbidden_hard_island():
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