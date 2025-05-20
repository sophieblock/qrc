import pytest
import networkx as nx

from qrew.simulation.refactor.quantum_gates import SWAP,CX
from qrew.simulation.refactor.quantum import QuantumCircuit, QuantumInstruction
from qrew.simulation.refactor.devices.quantum_devices import QuantumDevice
from qrew.simulation.refactor.q_interop.transpilers import QiskitCompiler


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
    assert res["swaps"] == []


# tests/test_layout_synthesizer.py
import networkx as nx
import pytest

from qrew.simulation.refactor.q_interop.transpilers import QiskitCompiler
from qrew.simulation.refactor.resources.quantum_resources import QuantumDevice
from qrew.simulation.refactor.quantum import QuantumCircuit, QuantumInstruction
from qrew.simulation.refactor.quantum_gates import CX, SWAP


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def make_line_device(n=3, *, opt_level=0):
    """Create a trivial n-qubit line device with the requested transpile OL."""
    g = nx.Graph([(i, i + 1) for i in range(n - 1)])
    qc = QiskitCompiler(
        basis_gates=("ECR", "I", "RZ", "SX", "X", "CX"),
        optimization_level=opt_level,
        layout_method="trivial",
        routing_method=None,          # leave routing to LayoutSynthesizer
    )
    return QuantumDevice(device_name=f"Line{n}_O{opt_level}",
                         connectivity=g,
                         compiler=qc)


# logical circuit used in every parameter set
LOGICAL = QuantumCircuit(
    qubit_count=3,
    instructions=[
        QuantumInstruction(CX(),   (0, 1)),
        QuantumInstruction(SWAP(), (0, 2)),
        QuantumInstruction(CX(),   (0, 1)),
    ],
)


# ---------------------------------------------------------------------------
# parameterised test matrix
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("opt_level", [0, 1, 2])
@pytest.mark.parametrize("transition_based", [True, False])
@pytest.mark.parametrize("hard_island", [True, False])
def test_layout_synthesizer_modes(opt_level, transition_based, hard_island):
    """
    A smoke-test verifying that LayoutSynthesizer returns a self-consistent
    solution for *every* combination of π/σ mode and optimization_level.
    """
    dev = make_line_device(opt_level=opt_level)

    # run LS
    circ, init_map, fin_map, depth, meta = dev.layout_synthesis(
        LOGICAL,
        transition_based=transition_based,
        hard_island=hard_island,
        objective="depth",
    )
    swaps = meta["swaps"]

    # ------------------------------------------------------------------ checks
    # 1) swap_duration learned must be non-negative
    assert dev.swap_duration >= 0, "swap_duration should be computed"

    # 2) reported depth must match the circuit’s own depth()
    assert depth == circ.depth(), "returned depth ≠ circuit.depth()"

    # 3) if hard_island=True, every σ-SWAP edge must stay within initial island
    if hard_island:
        initial_island = set(init_map)
        offending = [e for e, _ in swaps if not set(e).issubset(initial_island)]
        assert not offending, \
            f"hard_island violated by SWAP(s) on edges {offending}"

    # 4) transition_based ⇒ depth should be ≤ logical_depth + (#σ * swap_duration)
    if transition_based:
        logical_depth = LOGICAL.depth()
        upper_bound = logical_depth + len(swaps) * dev.swap_duration
        assert depth <= upper_bound, \
            "transition-based depth larger than conservative upper bound"

    # 5) no duplicate physical indices inside π-maps
    assert len(set(init_map)) == len(init_map), "π₀ not injective"
    assert len(set(fin_map))  == len(fin_map),  "π_f not injective"

    # 6) every SWAP edge actually exists on the device
    for edge, _ in swaps:
        assert tuple(edge) in dev.connectivity.edges or tuple(edge[::-1]) in dev.connectivity.edges, \
            f"σ-edge {edge} not present in device topology"