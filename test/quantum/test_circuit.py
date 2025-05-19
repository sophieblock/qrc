import networkx as nx
from olsq.solve import collision_extracting
import pytest

from qrew.simulation.refactor.resources import (
    QuantumDevice,
    QuantumResource,
)
from qrew.simulation.refactor.quantum import QuantumCircuit,QuantumInstruction, LayoutSynthesizer
from qrew.simulation.refactor.quantum_gates import *

from qrew.simulation.refactor.devices.quantum_devices import (
    IBM_Kyiv,     # same device that triggered the (14,) qubit bug
)

# def test_a():
#     assert 1 == 1


# device: line4_device’      
#                          
#                     0 –– 1 –– 2 –– 3     
# def line4_device():
#     g = nx.Graph([(0, 1), (1, 2), (2,3)])
#     return QuantumDevice(
#         "line4", g, gate_set=("H", "CX", "SWAP"),
#     )

# def leaf_cnot():
#     qc = QuantumCircuit(qubit_count=2)   # logical q0, q1
#     qc.add_instruction(QuantumInstruction(H(),  (0,)))
#     qc.add_instruction(QuantumInstruction(H(),  (1,)))
#     qc.add_instruction(QuantumInstruction(CX(), (0, 1)))   # needs 0-2 edge
#     return qc

# @pytest.mark.parametrize("hard_island", [True, False])
# def test_line4(hard_island):
#     dev   = line4_device()
#     circ  = leaf_cnot()

#     res = QuantumResource(
#         quantum_circuit=circ,
#         LS_parameters={
#             "transition based": False, "epsilon": 0.3,
#             "objective": "depth",      "hard_island": hard_island,
#         },
#     )

#     # ----------------  first allocation  -----------------------------------
#     alloc1 = dev.allocate(res)
#     touched1 = {q for inst in alloc1.transpiled_circuit.instructions
#                    for q in inst.gate_indices}
#     assert touched1 == set(alloc1.allocated_qubit_idxs)

#     # ----------------  second allocation  ----------------------------------
#     alloc2 = dev.allocate(res)
#     touched2 = {q for inst in alloc2.transpiled_circuit.instructions
#                    for q in inst.gate_indices}
#     assert touched2 == set(alloc2.allocated_qubit_idxs)

#     # after two concurrent reservations no connected component ≥2 remains
#     with pytest.raises(AssertionError,
#                        match="does not have sufficient available qubits"):
#         dev.allocate(res)

#     # clean-up
#     dev.deallocate(alloc1)
#     dev.deallocate(alloc2)
#     assert sorted(dev.available_qubits) == [0, 1, 2, 3]




def star5_device():
    """5-qubit star: centre 0, leaves 1-4."""
    g = nx.Graph([(0, 1), (0, 2), (0, 3), (0, 4)])
    return QuantumDevice(
        device_name="star5",
        connectivity=g,
        gate_set=("H", "CX", "SWAP"),
    )

def far_cnot_circuit():
    """
    2-logical-qubit circuit whose CX is *not* directly connected
    on the star leaves; it forces a relay through qubit 0 unless forbidden.
    """
    qc = QuantumCircuit(qubit_count=2)
    qc.add_instruction(QuantumInstruction(gate=H(),  qubit_indices=(0,)))
    qc.add_instruction(QuantumInstruction(gate=H(),  qubit_indices=(1,)))
    qc.add_instruction(QuantumInstruction(gate=CX(), qubit_indices=(0, 1)))
    return qc


def test_relay_qubit_policy_contract_1():
    dev   = star5_device()
    circ  = far_cnot_circuit()
    # print(f"circ instructions: {circ.instructions}")
    
  
    # wrap in QuantumResource because allocate() expects one
    res = QuantumResource(
        quantum_circuit=circ,
        LS_parameters={"transition based": False,
                       "epsilon": 0.3,
                       "objective": "depth",
                       "hard_island": True}
    )
    alloc = dev.allocate(res)
    touched = {
        q for instr in alloc.transpiled_circuit.instructions
        for q in instr.gate_indices
    }
    # relay qubit 0 was indeed used
    assert 0 in touched
    # the new guard in allocate() guarantees every touched qubit is recorded
    assert touched == set(alloc.allocated_qubit_idxs)


def test_relay_qubit_policy_contract_2():
    hard_island = False
    dev   = star5_device()
    circ  = far_cnot_circuit()
    print(f"circ instructions: {circ.instructions}")
    

    # wrap in QuantumResource because allocate() expects one
    res = QuantumResource(
        quantum_circuit=circ,
        LS_parameters={"transition based": False,
                       "epsilon": 0.3,
                       "objective": "depth",
                       "hard_island": False}
    )
    alloc = dev.allocate(res)
    touched = {
        q for instr in alloc.transpiled_circuit.instructions
        for q in instr.gate_indices
    }
    # relay qubit 0 was indeed used
    assert 0 in touched
    # the new guard in allocate() guarantees every touched qubit is recorded
    assert touched == set(alloc.allocated_qubit_idxs)

# def test_gate_set_is_set():
#     qc = QuantumCircuit(2)
#     qc.add_instruction(gate=X(), indices=(0,))
#     qc.add_instruction(gate=X(), indices=(0,))      # duplicate on purpose
#     assert qc.gate_set == {"X"}


@pytest.fixture
def device():
    return generate_quantum_device()

@pytest.fixture
def circuit():
    return generate_quantum_adder()
def qubits_touched(qc):
    """Return *all* physical qubits that appear in `qc`."""
    return {q for ins in qc.instructions for q in ins.gate_indices}


# ---------------------------------------------------------------------
#  parametrised life-cycle test
# ---------------------------------------------------------------------
@pytest.mark.parametrize(
    "hard_island, expected_depth, max_after, expected_qubits",
    [
        # old behaviour – optimiser must stay inside the 4-node island
        (True,  15, 1, 4),
        # relay allowed – optimiser may (or may not) add the 5th node
        (False, 15, 0, 5),
    ],
    ids=["hard_island", "relay_allowed"],
)
def test_layout_allocate_deallocate_normal(
    device,
    circuit,
    hard_island,
    expected_depth,
    max_after,
    expected_qubits,
):

    # ------------------------------------------------------------------
    # layout_synthesis  (depth & island invariants)
    # ------------------------------------------------------------------
    (compiled1,
     init_map1,
     final_map1,
     depth1,
     res1) = device.layout_synthesis(
        circuit,
        transition_based=False,
        hard_island=hard_island,
    )

    assert depth1 == expected_depth
    assert sorted(init_map1) == sorted(final_map1)

    touched1 = qubits_touched(compiled1)

    if hard_island:
        # optimiser must stay strictly within the initial island
        assert touched1.issubset(set(init_map1))
    # else: relay qubits allowed – nothing to assert here

    # ------------------------------------------------------------------
    # allocate()  (scheduler bookkeeping)
    # ------------------------------------------------------------------
    avail_before = device.available_qubits.copy()
    assert device.max_available_connections == 5          # toy connectivity

    q_resource = QuantumResource(
        quantum_circuit=circuit,
        LS_parameters={
            "transition based": False,
            "epsilon": 0.3,
            "objective": "depth",
            "hard_island": hard_island,
        },
    )
    alloc = device.allocate(q_resource)

    # allocator reserves every qubit its own compilation touched
    touched2 = qubits_touched(alloc.transpiled_circuit)
    assert len(touched2) == expected_qubits
    assert sorted(device.available_qubits + alloc.allocated_qubit_idxs) == sorted(
        avail_before
    )
    assert device.max_available_connections == max_after
    assert sorted(alloc.allocated_qubit_idxs) == sorted(touched2)

    # the device-side check inside allocate() has already enforced
    # the relay-free invariant when hard_island=True, so no extra
    # assertion is required here.

    # ------------------------------------------------------------------
    # deallocate()
    # ------------------------------------------------------------------
    device.deallocate(alloc)
    assert sorted(device.available_qubits) == sorted(avail_before)
    assert device.max_available_connections == 5

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


def test_compute_circuit_depth():
    device = generate_quantum_device()
    circuit = generate_quantum_adder()
    layout_synthesizer = LayoutSynthesizer(
        quantum_circuit=circuit, device=device, transition_based=False
    )
    circuit_depth = layout_synthesizer.compute_circuit_depth(circuit)
    assert circuit_depth == 11


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

    pi = layout_synthesizer.initialize_pi()

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

    sigma = layout_synthesizer.initialize_sigma()

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
    assert device.max_available_connections == 1
    assert sorted(device.available_qubits + list(allocated_qubit_idxs)) == sorted(
        available_qubits
    )
    assert all(qubit_idx in available_qubits for qubit_idx in allocated_qubit_idxs)


@pytest.mark.slow
def test_deallocate_normal():
    device = generate_quantum_device()
    circuit = generate_quantum_adder()
    quantum_resource = QuantumResource(
        quantum_circuit=circuit,
        LS_parameters={"transition based": False, "epsilon": 0.3, "objective": "depth"},
    )
    available_qubits = device.available_qubits

    assert device.max_available_connections == 5
    allocation = device.allocate(quantum_resource)
    assert device.max_available_connections == 1
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