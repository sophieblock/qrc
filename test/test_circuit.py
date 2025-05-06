import networkx as nx
from olsq.solve import collision_extracting
import pytest

from workflow.simulation.refactor.resources.quantum_resources import (
    QuantumDevice,
    QuantumResource,
)
from workflow.simulation.refactor.quantum import *
from workflow.simulation.refactor.quantum_gates import *


def test_a():
    assert 1 == 1


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
    _, initial_qubit_mapping, final_qubit_mapping, objective_result = (
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
        LS_parameters={"transition based": False, "epsilon": 0.3, "objective": "depth"},
    )
    available_qubits = device.available_qubits

    assert device.max_available_connections == 5
    allocation = device.allocate(quantum_resource)
    allocated_qubit_idxs = allocation.allocated_qubit_idxs
    assert device.max_available_connections == 1
    assert sorted(device.available_qubits + allocated_qubit_idxs) == sorted(
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
    _, initial_qubit_mapping, final_qubit_mapping, objective_result = (
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
        match=f"Device {device.name} does not have sufficient available qubits",
    ):
        device.layout_synthesis(circuit, transition_based=False)


def test_layout_synthesis_transition_based():
    device = generate_quantum_device()
    circuit = generate_quantum_adder()
    _, initial_qubit_mapping, final_qubit_mapping, objective_result = (
        device.layout_synthesis(circuit)
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
    _, initial_qubit_mapping, final_qubit_mapping, objective_result = (
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
    _, initial_qubit_mapping, final_qubit_mapping, objective_result = (
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
        match=f"Device {device.name} does not have sufficient available qubits",
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
    _, initial_qubit_mapping, final_qubit_mapping, objective_result = (
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
    optimized_circuit, initial_qubit_mapping, final_qubit_mapping, objective_result = (
        device.layout_synthesis(circuit)
    )

    assert objective_result == 15
    assert sorted(initial_qubit_mapping) == [0, 1, 2, 3]
    assert sorted(initial_qubit_mapping) == sorted(final_qubit_mapping)
