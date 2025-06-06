

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

# def test_a():
#     assert 1 == 1



def _make_tiny_swap_resource():
    """
    A 2-qubit CX is enough – it forces the allocator to reserve both qubits
    on a 2-node device.
    """
    qc = QuantumCircuit(qubit_count=2)
    qc.add_instruction(QuantumInstruction(X(),  (0,)))
    qc.add_instruction(QuantumInstruction(CX(), (0, 1)))
    return QuantumResource(quantum_circuit=qc)      # default LS-parameters


def _make_singleton_toy_device():
    g = nx.Graph([(0, 1)])
    dev = QuantumDevice("toy-2q-singleton", g, gate_set=("X", "CX"))
    dev._is_singleton = True          # triggers deepcopy inside Broker
    return dev


def test_broker_makes_fresh_copy_and_resets():
    """
    • Broker₁ should mutate **its own** copy of the device.
    • Broker₂ gets a brand-new, fully-reset copy – so it can allocate again.
    • The original singleton object stays pristine throughout.
    """
    device_singleton = _make_singleton_toy_device()
    resource         = _make_tiny_swap_resource()

    # ---- first broker -----------------------------------------------------
    broker1 = Broker(quantum_devices=[device_singleton])
    alloc1  = broker1.request_allocation(resource)
    # inside broker1 the device has no free qubits left
    assert broker1.quantum_devices[0].available_qubits == []

    # original singleton object was *deep-copied* → still fully available
    assert device_singleton.available_qubits == [0, 1]

    # ---- second broker ----------------------------------------------------
    broker2 = Broker(quantum_devices=[device_singleton])
    # starts from a clean slate
    assert broker2.quantum_devices[0].available_qubits == [0, 1]

    alloc2  = broker2.request_allocation(resource)
    # again, broker2’s private copy is now depleted
    assert broker2.quantum_devices[0].available_qubits == []
    # original singleton still untouched
    assert device_singleton.available_qubits == [0, 1]


def test_broker_reset_even_without_singleton_flag():
    """
    If the device is *not* marked as a singleton (no deepcopy), the Broker
    still calls .reset() so every run starts from a clean state.
    """
    # create a normal (non-singleton) device and exhaust it via Broker₁
    g = nx.Graph([(0, 1)])
    dev = QuantumDevice("toy-2q", g, gate_set=("X", "CX"))
    res = _make_tiny_swap_resource()

    broker1 = Broker(quantum_devices=[dev])
    broker1.request_allocation(res)
    assert broker1.quantum_devices[0].available_qubits == []

    # Broker₂ receives *the very same* object but reset() is called → available
    broker2 = Broker(quantum_devices=[dev])
    assert broker2.quantum_devices[0].available_qubits == [0, 1]

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
