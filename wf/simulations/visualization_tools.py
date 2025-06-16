from typing import TYPE_CHECKING
from .simulation.refactor.data import Data, DataSpec
from .simulation.refactor.schema import Signature, Flow
from .simulation.refactor.builder import DanglingT,LeftDangle,RightDangle,Connection, Port,ProcessInstance
if TYPE_CHECKING:
    from .simulation.refactor.data import Data, DataSpec
    from .simulation.refactor.schema import Signature, Flow
    from .simulation.refactor.builder import DanglingT,LeftDangle,RightDangle,Connection, Port,ProcessInstance
    from .simulation.refactor.data_types import QBit,CBit,TensorType,MatrixType, check_dtypes_consistent
    # from .simulation.refactor.graph import DirectedEdge,Network

import itertools
from numpy import inf as INFINITY
from typing import List
import html

import IPython.display
import ipywidgets
import pydot
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from IPython.display import display, SVG
from functools import cached_property
import sympy
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from .util.log import logging
logger = logging.getLogger(__name__)
from graphviz import Digraph
import tempfile
import subprocess
import platform
from IPython.display import Image, display
from IPython.display import display as display_fig
import cirq


_BASE_GATES = {
    # 1-qubit, no parameter
    "H":   cirq.H,
    "X":   cirq.X,
    "Y":   cirq.Y,
    "Z":   cirq.Z,
    "S":   cirq.S,
    "T":   cirq.T,
    "Tdg": cirq.T**-1,          # dagger of T

    # 1-qubit, parameter
    "RX":  cirq.rx,             # factories
    "RY":  cirq.ry,
    "RZ":  cirq.rz,

    # 2-qubit, no parameter
    "CX":  cirq.CNOT,
    "CZ":  cirq.CZ,
    "SWAP": cirq.SWAP,

    # 2-qubit, parameter
    "CZPow": cirq.CZPowGate,    # will be called with exponent
    # Add more as needed…
}


def _cirq_gate_from_instruction(instr):
    """
    Return the Cirq gate/operation corresponding to *instr* (QuantumInstruction).

    For 1-qubit parameterised gates we call the factory with the angle.
    For CZPow we instantiate with exponent=instr.gate.param.
    """
    name = instr.gate.name

    if name in {"RX", "RY", "RZ"}:
        theta = getattr(instr.gate, "param")
        return _BASE_GATES[name](theta)

    if name == "CZPow":
        exponent = getattr(instr.gate, "param")
        return _BASE_GATES[name](exponent=exponent)

    if name not in _BASE_GATES:
        raise ValueError(f"Gate '{name}' not (yet) supported by translate_c_to_cirq")

    return _BASE_GATES[name]


# ────────────────────────────────────────────────────────────────────────────
# 2.  Main translation routine
# ────────────────────────────────────────────────────────────────────────────
def translate_c_to_cirq(
    qc,
    *,
    noise_model=None,
    save_measurements=False,
    include_idle_identity=True,
):

    n_qubits = qc.qubit_count
    qubits   = cirq.LineQubit.range(n_qubits)

    ops = []

    if include_idle_identity and n_qubits:
        ops.append(cirq.Moment(cirq.I.on_each(*qubits)))

    meas_count = 0

    for instr in qc.instructions:
        gate_name = instr.gate.name

        # 0.  Resolve the Cirq gate or factory
        if gate_name == "MEASURE":
            key = str(meas_count) if save_measurements else None
            ops.append(cirq.measure(qubits[instr.gate_indices[0]], key=key))
            meas_count += 1
            continue

        cirq_gate = _cirq_gate_from_instruction(instr)

        # 1-qubit vs 2-qubit dispatch
        if len(instr.gate_indices) == 1:
            q0 = qubits[instr.gate_indices[0]]
            ops.append(cirq_gate(q0) if callable(cirq_gate) else cirq_gate.on(q0))

        elif len(instr.gate_indices) == 2:
            q0, q1 = (qubits[i] for i in instr.gate_indices)
            # Cirq convention: for CX the first is *control*, second *target*
            if callable(cirq_gate):
                ops.append(cirq_gate(q0, q1))
            else:
                ops.append(cirq_gate.on(q0, q1))
        else:
            raise NotImplementedError(
                f"Gates on >2 qubits not supported (instruction={instr})"
            )

        # (Optional) insert noise here in future — parity with Tangelo stub.
        if noise_model and gate_name in noise_model.noisy_gates:
            raise NotImplementedError("Noise model insertion not implemented yet")

    return cirq.Circuit(ops)


def _assign_ids_to_nodes_and_edges(process_instances, all_ports):
    to_id: Dict[Any, str] = {}
    ids: Set[str] = set()
    disambiguator = 0
    def add(item: Any, desired_id: str):
        nonlocal disambiguator
        if item in to_id:
            # raise ValueError(f"Item {item} was already added to the ID mapping.")
            return
        if desired_id not in ids:
            unique_id = desired_id
        else:
            unique_id = f'{desired_id}_G{disambiguator}'
            disambiguator += 1

        ids.add(unique_id)
        to_id[item] = unique_id
    # print(f"process_instances: {process_instances}")
    for pinst in process_instances:
        
        add(pinst, f'{pinst.process.__class__.__name__}')
        for groupname,_ in pinst.process.signature.groups():
            add((pinst,groupname),groupname)
    # 2) **new**: also give every possible port in those groups an ID,
    #    so that when add_pinst does Port(pinst,reg,idx) those show up here
    
    # for pinst in process_instances:
    #     for _, groupregs in pinst.process.signature.groups():
    #         for reg in groupregs:
    #             for idx in reg.all_idxs():
    #                 port = Port(pinst, reg, idx)
    #                 add(port, reg.name)
    for port in all_ports:
        add(port,f'{port.reg.name}')

    return to_id
def _parition_registers_in_a_group(
    data_regs: Iterable[Data], pinst: ProcessInstance
) -> Tuple[List[Port], List[Port], List[Port]]:
    """Construct and sort the expected Ports for a given data group.

    Since we expect the input data to be in a group, we assert that
    if they are THRU data there are not LEFT and RIGHT registers as well.


    """
    lefts = []
    rights = []
    thrus = []
    for data in data_regs:
        for idx in data.all_idxs():
            dataport = Port(pinst, data, idx)
            if data.flow is Flow.LEFT:
                lefts.append(dataport)
            elif data.flow is Flow.RIGHT:
                rights.append(dataport)
            else:
                assert data.flow is Flow.THRU
                thrus.append(dataport)

    if len(thrus) > 0:
        if len(lefts) > 0 or len(rights) > 0:
            raise ValueError(
                "A data group containing THRU data registers cannot "
                "also contain LEFT and RIGHT data registers."
            )

    return lefts, rights, thrus

from pathlib import Path
import os

def display_mod(process, type: str="dtype",  show_bookkeeping=True):
    if type.lower() == 'dtype':
        IPython.display.display(ModuleDrawer(process, label_type="dtype", show_bookkeeping=show_bookkeeping))
    elif type.lower() == 'nbytes':
        IPython.display.display(ModuleDrawer(process, label_type="nbytes", show_bookkeeping=show_bookkeeping))
    elif type.lower() == 'shape':
        IPython.display.display(ModuleDrawer(process, label_type="shape", show_bookkeeping=show_bookkeeping))


def display_mod(process, type: str="dtype",  show_bookkeeping=True):
    edge_label = type.lower()
    IPython.display.display(ModuleDrawer(process, label_type=edge_label, show_bookkeeping=show_bookkeeping))
    # if type.lower() == 'dtype':
    #     IPython.display.display(ModuleDrawer(process, label_type="dtype", show_bookkeeping=show_bookkeeping))
    # elif type.lower() == 'nbytes':
    #     IPython.display.display(ModuleDrawer(process, label_type="nbytes", show_bookkeeping=show_bookkeeping))
    # elif type.lower() == 'shape':
    #     IPython.display.display(ModuleDrawer(process, label_type="shape", show_bookkeeping=show_bookkeeping))


def save_mod(process, label_type: str="dtype",  show_bookkeeping=True, filename = None, output_dir = None):
    edge_label = label_type.lower()
    drawer = ModuleDrawer(process, label_type=edge_label,show_bookkeeping=show_bookkeeping,)
    drawer.render(display=False, save_fig=True, filename=filename,output_dir=output_dir)

class ModuleDrawer:
    """A class to encapsulate methods for displaying or saving a Workflow as a graph using pydot."""

    def __init__(self, cnode, label_type="dtype", show_bookkeeping = True):
        cnode = cnode.as_composite()
        self._cnode = cnode
        self._ports = cnode.all_ports
        self._pinsts = cnode.pinsts
        self.label_type = label_type  # New attribute to select between dtype and data_width
        self.show_bookkeeping = show_bookkeeping
        self.ids = _assign_ids_to_nodes_and_edges(self._pinsts,self._ports)
        # logger.debug(f"ids: {self.ids}")
    def _repr_svg_(self):
        # Jupyter will call this automatically and render the SVG inline
        return self.get_svg_bytes().decode('utf-8')
    def _repr_png_(self):
        # Fallback if the notebook prefers PNG
        return self.get_graph().create(format='png')
    def get_dangle_node(self, port: Port) -> pydot.Node:
        """Overridable method to create a Node representing dangling Ports."""
        return pydot.Node(self.ids[port], label=port.pretty(), shape='plaintext')

    def add_dangles(
        self, graph: pydot.Graph, signature: Signature, dangle: DanglingT
    ) -> pydot.Graph:
        """Add nodes representing dangling indices to the graph.

        We wrap this in a subgraph to align (rank=same) the 'nodes'
        """
        # print(f"{dangle}")
        if dangle is LeftDangle:
            regs = signature.lefts()
        elif dangle is RightDangle:
            regs = signature.rights()
        else:
            raise ValueError()

        subg = pydot.Subgraph(rank='same')
        for reg in regs:
            for idx in reg.all_idxs():
                
                subg.add_node(self.get_dangle_node(Port(dangle, reg, index=idx)))
        graph.add_subgraph(subg)
        return graph
    
  
    def get_thru(self, thru: Port) -> str:
        """Overridable method for generating a <TR> representing a THRU dataport.

        This should have a `colspan="2"` to make sure there aren't separate left and right
        cells / dataports.
        """
        return (
            f'  <TR><TD colspan="2" port="{self.ids[thru]}">'
            f'{html.escape(thru.pretty())}</TD></TR>\n'
        )
    def get_port_label(self,port: Port):
        from qrew.simulation.refactor.builder import Split, Join
        if not self.show_bookkeeping:
            if isinstance(port.process_instance, ProcessInstance) and isinstance(port.process_instance.process, (Split, Join)):
                # logger.debug(f'No port label for {port.process_instance}')
                return ''
            
        # logger.debug(f'Port label for {port.process_instance} {isinstance(port.process_instance,Split)}: {port.pretty()}')
        return port.pretty()

    def _register_td(self, port: Optional[Port], *, with_empty_td: bool, rowspan: int = 1) -> str:
        """Return the html code for an individual <TD>.

        This includes some factored-out complexity which aims to correctly pad cells that
        are empty due to a differing number of left and right dataports.

        Args:
            port: The optional port.
            with_empty_td: If `soq` is `None`, return an empty `<TD>` in its place if
                this is set to True. Otherwise, omit the empty TD and rely on the rowspan arguments.
            rowspan: If greater than `1`, include the `rowspan` html attribute on the TD to
                span multiple rows.
        """

        if port is None:
            if with_empty_td:
                return '<TD></TD>'
            else:
                return ''

        if rowspan != 1:
            assert rowspan > 1
            rowspan_html = f'rowspan="{rowspan}"'
        else:
            rowspan_html = ''

        return f'<TD {rowspan_html} port="{self.ids[port]}">{html.escape(self.get_port_label(port))}</TD>'

    def _get_register_tr(
        self,
        left: Optional[Port],
        right: Optional[Port],
        *,
        with_empty_td: bool = True,
        left_rowspan: int = 1,
        right_rowspan: int = 1,
    ) -> str:
        """Return the html code for a <TR> where `left` and `right` may be `None`.

        Args:
            left: The optional left dataport.
            right: the optional right dataport.
            with_empty_td: If `left` or `right` is `None`, put an empty `<TD>` in its place if
                this is set to True. Otherwise, omit the empty TD and rely on the rowspan arguments.
            left_rowspan: If greater than `1`, include the `rowspan` html attribute on left TDs to
                span multiple rows.
            right_rowspan: If greater than `1`, include the `rowspan` html attribute on right TDs to
                span multiple rows.
        """
        # print(f" - left: {left}, right: {right}")
        tr_code = '  <TR>'
        tr_code += self._register_td(left, rowspan=left_rowspan, with_empty_td=with_empty_td)
        tr_code += self._register_td(right, rowspan=right_rowspan, with_empty_td=with_empty_td)
        tr_code += '</TR>\n'
        return tr_code
    def get_pinst_header_text(self, pinst: ProcessInstance):
        from qrew.simulation.refactor.builder import Split, Join
        if not self.show_bookkeeping:
            if isinstance(pinst.process, (Split, Join)):
                return ''
        return f'<font point-size="10">{html.escape(str(pinst.process))}</font>'
    def add_pinst(self, graph: pydot.Graph, pinst: ProcessInstance) -> pydot.Graph:
        """Process and add a bloq instance to the Graph."""
        # print(f"adding binst: {pinst}")
        label = '<'  # graphviz: start an HTML section
        label += '<TABLE  BORDER="0" CELLBORDER="1" CELLSPACING="0" >\n'

        label += f'  <TR><TD colspan="2">{self.get_pinst_header_text(pinst=pinst)}</TD></TR>\n'

        for groupname, groupregs in pinst.process.signature.groups():
            lefts, rights, thrus = _parition_registers_in_a_group(groupregs, pinst)

            # Special case: all registers are THRU and we don't need different left and right
            # columns.
            if len(thrus) > 0:
                for t in thrus:
                    label += self.get_thru(t)
                continue

        
            n_surplus_lefts = max(0, len(lefts) - len(rights))
            n_surplus_rights = max(0, len(rights) - len(lefts))
            n_common = min(len(lefts), len(rights))

            if n_common >= 1:
               
                # add all but the last common rows. Both lefts[i] and rights[i] are non-None.
                for i in range(n_common - 1):
                    label += self._get_register_tr(lefts[i], rights[i])

                # For the last common row, we need to include an increased rowspan for
                # to pad the less-full column.
                label += self._get_register_tr(
                    lefts[n_common - 1],
                    rights[n_common - 1],
                    left_rowspan=n_surplus_rights + 1,
                    right_rowspan=n_surplus_lefts + 1,
                )

                # For the rest of the registers, we don't include empty TDs
                # because we used the rowspan argument above.
                with_empty_td = False
            else:
                # No common rows; no place to include rowspan arguments so we pad with empty TDs
                with_empty_td = True

            # Add the rest of the registers.
            for l, r in itertools.zip_longest(lefts[n_common:], rights[n_common:], fillvalue=None):
                label += self._get_register_tr(l, r, with_empty_td=with_empty_td)

        label += '</TABLE>'
        label += '>'  # graphviz: end the HTML section
        # print(f"adding pydot node: {self.ids[pinst]}")
        graph.add_node(pydot.Node(self.ids[pinst], label=label, shape='plain'))
        return graph
    @staticmethod
    def _fmt_dtype(dtype):
        return str(dtype)


    def cxn_edge(self, left_id: str, right_id: str, cxn: Connection) -> pydot.Edge:
        """Overridable method to style a pydot.Edge for connecionts."""
        l, r = cxn.left.reg, cxn.right.reg
        logger.debug(f"cxn: {cxn}. l.dtype: {l.dtype}, l.shape: {l.shape}, l.nbytes: {l.dtype.nbytes}, \nr.dtype: {r.dtype}, r.shape: {r.shape}, r.dtype.nbytes: {r.dtype.nbytes}")
        if self.label_type == 'shape':
            # l, r = cxn.left.reg, cxn.right.reg
            if l == r:
                # print(f"l == r")
                # cxn_label =  self._fmt_dtype(l)
                cxn_label = str(r.shape)
            
            if isinstance(l.dtype,(TensorType,MatrixType)) and not isinstance(r.dtype,(TensorType,MatrixType)):
                cxn_label = f'{l.dtype.shape}-{r.shape}'
            elif isinstance(l.dtype,(TensorType,MatrixType)) and isinstance(r.dtype,(TensorType,MatrixType)):
                cxn_label = f'{l.dtype.shape}-'
            elif not isinstance(l.dtype,(TensorType,MatrixType)) and isinstance(r.dtype,(TensorType,MatrixType)):
                cxn_label = f'-{r.dtype.shape}'
            else:
                cxn_label = f"{l.shape}->{r.shape}"
        elif self.label_type == 'dtype':
            l_dtype, r_dtype = l.dtype, r.dtype
            if l_dtype == r_dtype:
                # print(l, str(l))
                cxn_label =  self._fmt_dtype(l_dtype)
            elif l_dtype.data_width == 1:
                cxn_label =  self._fmt_dtype(l_dtype if isinstance(l_dtype,(QBit,CBit)) else r_dtype)
            else:
                # TODO: choose 'wider' dtype
                assert check_dtypes_consistent(l_dtype,r_dtype)
                cxn_label =  f'{self._fmt_dtype(l_dtype)}-{self._fmt_dtype(r_dtype)}'

            
        elif self.label_type == 'nbytes':
            l_bytes, r_bytes = l.dtype.nbytes, r.dtype.nbytes
            if l_bytes == 0 or r_bytes == 0:
                l_width, r_width = l.dtype.data_width, r.dtype.data_width
                cxn_label = str(r_width)
            else:
                # print(cxn.left.data.bit_length)
                # print(cxn.left.data)
                cxn_label = str(r_bytes)
        else: 
            cxn_label = cxn.left.reg.total_bits() 
        # print(cxn_label,self.label_type)
        # logger.debug(f"cxn label: {cxn_label}, shape: {cxn.shape}")
        assert cxn_label is not None, f"cxn_label cannot be None for label_type={self.label_type}. cxn label: {cxn_label}, shape: {cxn.shape}"
        return pydot.Edge(left_id, right_id, label=cxn_label)

    def add_cxn(self, graph: pydot.Graph, cxn: Connection) -> pydot.Graph:
        """Process and add a connection to the Graph.

        Connections are specified using a `:` delimited set of ids. The first element
        is the node. For most Node instances, the second element is
        the port. The final element is the compass direction of where exactly
        the connecting line should be anchored.

        For DangleT nodes, there aren't any 'Port's so the second element is omitted.
        """

        if cxn.left.process_instance is LeftDangle:
            left = f'{self.ids[cxn.left]}:e'
        else:
            left = f'{self.ids[cxn.left.process_instance]}:{self.ids[cxn.left]}:e'
        # print(cxn.dest_nodes.process_instance)
        if cxn.right.process_instance is RightDangle:
            right = f'{self.ids[cxn.right]}:w'
        else:
            right = f'{self.ids[cxn.right.process_instance]}:{self.ids[cxn.right]}:w'

        graph.add_edge(self.cxn_edge(left, right, cxn))
        return graph
    
    def get_graph(self) -> pydot.Dot:
        """Get the graphviz graph representing the Bloq.

        This is the main entry-point to this class.
        """
        graph = pydot.Dot('my_graph', graph_type='digraph', rankdir='LR')
        graph = self.add_dangles(graph, self._cnode.signature, LeftDangle)

        for pinst in self._pinsts:
            graph = self.add_pinst(graph, pinst)

        graph = self.add_dangles(graph, self._cnode.signature, RightDangle)

        for cxn in self._cnode.connections:
            graph = self.add_cxn(graph, cxn)

        return graph


    def get_svg_bytes(self) -> bytes:
        """Generate the SVG representation of the network."""
        graph = self.get_graph()
        return graph.create(prog='dot', format='svg')
    def get_svg(self) -> IPython.display.SVG:
        """Get an IPython SVG object displaying the graph."""
        return IPython.display.SVG(self.get_svg_bytes())
    def save_graph(self, filename, display=False, output_dir=None):
        """
        Saves the graph to the specified directory and optionally displays it.

        Args:
            filename (str): Name of the file to save (with extension, e.g., 'graph.png').
            display (bool): If True, opens the saved file in the default viewer.
            output_dir (str or Path, optional): Base directory to save the graph. If None, defaults to 'figures'.
        """
        # Determine base directory
        if output_dir is None:
            dir_path = Path("figures")
        else:
            dir_path = Path(output_dir)
        # Ensure the directory exists
        dir_path.mkdir(parents=True, exist_ok=True)

        filepath = dir_path / filename
        ext = filepath.suffix.lower()

        # Generate Graphviz graph
        graph = self.get_graph()

        # Save the graph in the appropriate format
        if ext == ".svg":
            graph.write_svg(str(filepath))
        elif ext == ".png":
            graph.write_png(str(filepath))
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

        print(f"Graph saved to {filepath}")

        # Optionally display the saved file
        if display:
            if platform.system() == "Darwin":  # macOS
                subprocess.run(["open", str(filepath)])
            elif platform.system() == "Windows":  # Windows
                subprocess.run(["start", str(filepath)], shell=True)
            else:  # Linux/Unix
                subprocess.run(["xdg-open", str(filepath)])

    def render(self, filename=None, display=False, save_fig=True, output_dir=None):
        """
        Render the graph. Optionally save to a directory and/or display it
        """
        if filename and save_fig:
            # Save the graph using the provided filename and output directory
            self.save_graph(filename, display=display, output_dir=output_dir)
        elif display:
            # Display the graph without saving
            graph = self.get_graph()
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                tmpfile.close()
                graph.write_png(tmpfile.name)

                # Open the temporary file
                if platform.system() == "Darwin":  # macOS
                    subprocess.run(["open", tmpfile.name])
                elif platform.system() == "Windows":  # Windows
                    subprocess.run(["start", tmpfile.name], shell=True)
                else:  # Linux/Unix
                    subprocess.run(["xdg-open", tmpfile.name])
        elif filename and not save_fig:
            print(f"Filename '{filename}' provided but save_fig is set to False. Skipping save.")
# def normalize_process_call(
#     target: Callable,
#     args: Tuple[Any, ...],
#     kwargs: Optional[Dict[str, Any]] = None,
#     *,
#     arg_types: Optional[Tuple[Any, ...]] = None,
#     kwarg_types: Optional[Dict[str, Any]] = None,
# ) -> Dict[str, Any]:
#     """
#     A unified function that attempts to "bind" user inputs to either:
#         (1) A Process subclass's signature (like MatrixMult)
#         (2) A torch operator (like torch.add)
#         (3) A normal Python function
#     Returns a dictionary {param_name -> actual_argument} if successful,
#     or raises an error if the binding fails.
    
#     1. If 'target' is an instance of Process, we:
#        - Build an inspect.Signature from 'process.signature'
#        - Attempt signature.bind(*args, **kwargs)
#        - Validate each input Data with the RegisterSpecs
#        - Return {param_name: Data}

#     2. If 'target' is a torch op (OpOverload/OpOverloadPacket) or a BuiltinFunctionType,
#        we use `normalize_function(...)` from operator_schemas. If that fails or returns None,
#        we do a final fallback to normal Python signature binding.

#     3. Otherwise, fallback to a normal Python function check with inspect.signature(target).


   
#     NOTE: Might need to unify that `Data` objects are always real args if target is a Process subclass,
#     or raw python if target is a torch op. Right now we return the user objects as the binding result
    
#     NOTE: This approach merges the advanced multiple-dispatch logic
#     in operator_schemas.py with custom logic for Process.

#     Args:
#         target (Union[Process, callable]): The function or Process object
#         args (Tuple[Any]): The positional args
#         kwargs (Optional[Dict[str, Any]]): The keyword args
#         arg_types (Optional[Tuple[Any]]): Additional type hints for disambiguation
#         kwarg_types (Optional[Dict[str, Any]]): Additional type hints for disambiguation

#     Returns:
#         A dict {param_name -> Data or python object} with resolved inputs, or raises an error.
#     """
#     if kwargs is None:
#         kwargs = {}
#     normalized_data: Dict[str, Any] = {}
  
#     from torch._library.infer_schema import infer_schema
#     # print(infer_schema)
#     if isinstance(target, Process):
#         # Build an inspect.Signature from the process signature
#         fx_sig = build_inspect_signature_from_process_signature(target.signature)

#         param_names = [p.name for p in target.signature.lefts()]  
#         logger.debug(f'Process {target} has fx sig: {fx_sig} w/ parameter names: {param_names}')
#         # Attempt to bind 
#         args_for_bind = [d.data for d in args]  # or just 'args' if they're Data
#         kwargs_for_bind = {k: (v.data if hasattr(v, "data") else v) for k, v in kwargs.items()}
#         try:
#             bound = fx_sig.bind(*args_for_bind, **kwargs_for_bind)
#             bound.apply_defaults()
#         except TypeError as te:
#             matchobj = re.search(r"missing a required argument: '(.*?)'", str(te))
#             if matchobj:
#                 missing_param = matchobj.group(1)
#                 raise ValueError(f"Missing data for parameter {missing_param}") from te
#             raise ValueError(f"Failed to bind arguments to Process {target}: {te}") from te
#         # Reconstruct param_name -> original Data 
#         param_list = list(fx_sig.parameters.keys())
#         used_positional_count = len(bound.args)
#         for idx, pname in enumerate(param_list):
#             if idx < used_positional_count:
#                 data_obj = args[idx]
#             else:
#                 data_obj = kwargs.get(pname, None)
#             if data_obj is None:
#                 raise ValueError(f"Missing data for parameter {pname}")
           

#             normalized_data[pname] = data_obj
#             logger.debug(f"{pname}: {type(data_obj)}")
#             # print(pname, type(data_obj))
#         # Additional typed checks using the process's signature
#         # e.g. 'signature.validate_data_with_register_specs(inputs)'
#         # or your own logic with regspec.matches_data
#         for k,v in normalized_data.items():
#             logger.debug(f"{k}: {type(v)}")
       
#         target.signature.validate_data_with_register_specs(
#             [normalized_data[p.name] for p in target.signature.lefts()]
#         )

#         #         )
#         return normalized_data
#     ################################################################
#     # B) If not a Process
#     #    => Attempt the PyTorch / plain function approach
#     ################################################################

#     from torch.fx.operator_schemas import (
#         get_signature_for_torch_op,
#         _args_kwargs_to_normalized_args_kwargs,
#         OpOverload,
#         OpOverloadPacket,
#     )
#     if (not isinstance(target, OpOverloadPacket)
#         and not isinstance(target, OpOverload)
#         and not isinstance(target, types.BuiltinFunctionType)):
#         # Just use Python's inspect
#         sig = inspect.signature(inspect.unwrap(target))
#         bound = sig.bind(*args, **kwargs)
#         bound.apply_defaults()
#         return dict(bound.arguments)

   
#     torch_op_sigs = get_signature_for_torch_op(target)
#     if not torch_op_sigs:
#         # No known signatures => can't bind
#         raise ValueError(f"No signature found for PyTorch op: {target}")

#     # 2.1) Try to find *all* matches by attempting sig.bind()
#     matched_schemas = []
#     for candidate_sig in torch_op_sigs:
#         try:
#             candidate_sig.bind(*args, **kwargs)
#             matched_schemas.append(candidate_sig)
#         except TypeError:
#             continue

#     if len(matched_schemas) == 0:
#         # No valid schema matched => can't unify
#         raise ValueError(
#             f"No valid overload for {target} with arguments={args}, kwargs={kwargs}"
#         )
#     elif len(matched_schemas) == 1:
#         # Exactly one match => perfect
#         chosen_sig = matched_schemas[0]
#     else:
#         # Multiple matches => we need arg_types/kwarg_types to break ties
#         if not arg_types and not kwarg_types:
#             # Raise an error for ambiguity
#             raise ValueError(
#                 f"Ambiguous call to {target}. Multiple overloads matched, "
#                 "and you did not provide arg_types/kwarg_types to disambiguate."
#             )

#         # 2.2) Second pass: do type-based filtering
#         # We'll see which schemas match your provided param-type hints
#         # We require that every param name's type passes type_matches().
#         filtered = []
#         for candidate_sig in matched_schemas:
#             # 2.2.1) bind param names from the signature to the arg_types
#             try:
#                 # Trick: We'll do a parallel "bind" but with arg_types instead of real values
#                 # so we can check if param.annotation is consistent with arg_type
#                 bound_type_check = candidate_sig.bind(*(arg_types or ()), **(kwarg_types or {}))
#             except TypeError:
#                 # If we can't bind the types, skip
#                 continue

#             # 2.2.2) For each param => check type_matches(signature_annot, actual_type)
#             # But recall that we might not have any official annotation in these
#             # Torch schemas. If so, skip or treat as a pass. Example code:
#             all_good = True
#             for name, user_type in bound_type_check.arguments.items():
#                 param = candidate_sig.parameters[name]
#                 if param.annotation is not inspect.Parameter.empty:
#                     if not type_matches(param.annotation, user_type):
#                         all_good = False
#                         break
#                 # else no annotation => ignore
#             if all_good:
#                 filtered.append(candidate_sig)

#         if len(filtered) == 0:
#             raise ValueError(
#                 f"Could not find a matching schema for {target} even after "
#                 f"type-based disambiguation. arg_types={arg_types} kwarg_types={kwarg_types}"
#             )
#         elif len(filtered) > 1:
#             raise ValueError(
#                 f"Still ambiguous: multiple overloads match the provided arg_types={arg_types} "
#                 f"and kwarg_types={kwarg_types} for {target}. Overloads: {filtered}"
#             )
#         else:
#             chosen_sig = filtered[0]

#     # -------------------------------------------------------------------------
#     # If we get here, we have a single chosen_sig => let's do a final bind and return a dict
#     # -------------------------------------------------------------------------
#     bound = chosen_sig.bind(*args, **kwargs)
#     bound.apply_defaults()
#     # Convert BoundArguments to a plain dict
#     return dict(bound.arguments)
