# from .builder import Port,Connection, Flow, Data, ProcessInstance, DanglingT, LeftDangle, RightDangle
from .data import Data, Result, DataSpec
from .schema import RegisterSpec,Flow,Signature
from .graph import Node, DirectedEdge
from .process import Process
from .data_types import *

from qrew.simulation.data_types import DataType

import numpy as np
from numpy.typing import NDArray

from typing import Union,Dict,List,overload,Iterator, Optional,Set,Hashable,TypeVar,Mapping,Tuple, Iterable, Callable, FrozenSet, Sequence,Any

from attrs import field, frozen

import networkx as nx

from functools import cached_property


from ..util.log import get_logger,logging

logger = get_logger(__name__)

from torch.fx.experimental.symbolic_shapes import ShapeEnv as TorchShapeEnv
import torch
from torch import SymInt, SymBool, SymFloat 
from torch.fx.experimental.sym_node import method_to_operator, SymNode
from torch.fx.experimental.recording import record_shapeenv_event
from torch._guards import ShapeGuard, Source
from torch.fx.experimental.symbolic_shapes import (
    free_unbacked_symbols,
    DimDynamic,
    SymbolicContext,
    guard_int,
    StatelessSymbolicContext,
    DimConstraints
)
from typing_extensions import TypeGuard
import sympy
from sympy import Symbol
from attrs import define, frozen, field
from .unification_tools import canonicalize_dtype


class DanglingT:
    """The type of the singleton objects `LeftDangle` and `RightDangle`.

    These objects are placeholders for the `node_instance` field of a `Port` that represents
    an "external wire". We can consider `Ports` of this type to represent input or
    output data of a `Node` or 'Network'.
    """

    def __init__(self, name: str):
        self._name = name

    def __repr__(self):
        return self._name

    def process_is(self, t) -> bool:
        """DanglingT.process_is(...) is always False.
        """
        return False

LeftDangle = DanglingT("LeftDangle")
RightDangle = DanglingT("RightDangle")


@define
class ProcessInstance:
    """
        A class for a unique instance Process object of a Node within a Network.

        Attributes:
            process: Process
            index: Arbitrary index to disambiguate this instance from other Node's
            with the same Process model within a network. 
    
    """
    process: 'Process'
    i: int
    def __str__(self):
        return f"{self.process}<{self.i}>"
    
    def process_is(self,t) -> bool:
        return isinstance(self.process,t)
    def __hash__(self):
        # Hash only the object id + the unique index i, ignoring the
        # actual fields of process (which might be unhashable).
        return hash((id(self.process), self.i))
    def __eq__(self, other):
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (self.i == other.i) and (self.process is other.process)
        
def _to_tuple(x):
    """mypy-compatible attrs converter for CompositeMod.connections"""
    return tuple(x)

def _to_set(x: Iterable[ProcessInstance]) -> Set[ProcessInstance]:
    """mypy-compatible attrs converter for CompositeMod.pinsts"""
    return set(x)

@define
class Port:
    
    """

    One half of a connection.
    
    process_instance: The ProcessInstance to which this port belongs.
    reg: The RegisterSpec that this Port is an instance of.
    index: RegisterSpec objs with non-empty 'shape' attributes are multi-dimensional
            and so a Port should explicitely index the instantiation of one element
            of the multi-dimensional Data obj.
    """


    process_instance: Union[ProcessInstance, DanglingT]
    reg: 'RegisterSpec'
    index: Tuple[int,...] = field(converter=_to_tuple,default=tuple())
    
    @index.validator
    def _check_idx(self, attribute, value):
        if len(value) != len(self.reg.shape):
            raise ValueError(f"Bad index shape {value} for {self.reg}.")
        for i, shape in zip(value, self.reg.shape):
            if i >= shape:
                raise ValueError(f"Bad index {i} for {self.reg}.")
        
    def pretty(self) -> str:
        label = self.reg.name
        if len(self.index) > 0:
            return f'{label}[{", ".join(str(i) for i in self.index)}]'
        return label
    
    def __hash__(self):
        # define a stable hash based on your fields
        return hash((self.process_instance, self.reg, self.index))

    def __eq__(self, other):
        if not isinstance(other, Port):
            return NotImplemented
        return (
            self.process_instance == other.process_instance
            and self.reg == other.reg
            and self.index == other.index
        )
    def __str__(self):
        return f'{self.process_instance}.{self.pretty()}'
    def __repr__(self):
        return f'{self.process_instance}.{self.pretty()}'
    
@frozen
class Connection:
    """A connection between two `Port`s.

    Quantum data flows from left to right. The graph implied by a collection of `Connections`s
    is directed. Does this still apply for classical?
    """

    left: Port
    right: Port

    @cached_property
    def shape(self) -> int:

        ls = self.left.reg.bitsize
        rs = self.right.reg.bitsize

        if ls != rs:
            raise ValueError(f"Invalid Connection {self}: shape mismatch: {ls} != {rs}")
        return ls

    def __str__(self) -> str:
        return f'{self.left} -> {self.right}'

_PortType = TypeVar('_PortType', bound=np.generic)
PortT = Union[Port, NDArray[_PortType]]
PortInT = Union[Port, NDArray[_PortType], Sequence[Port]]
def _shape_of(dt: DataType | Any) -> tuple[int, ...]:
    """Return tensor/matrix shape, or () when scalar."""
    return tuple(getattr(dt, "shape", ()))

def _element_dtype_of(dt: DataType | Any) -> DataType:
    """Tensor/Matrix → element_type; otherwise return dt itself."""
    return canonicalize_dtype(dt.element_type) if hasattr(dt, "element_type") else dt
@define
class Split(Process):
    """ Splits one register of a given dtype into an array of e.g. CBit or QBit.

    If, say, dtype= CBit() with shape=(10,10), we produce an output array
    of 'element dtype' with shape=(10,10), if that is your design.

    For a purely atomic approach, you do an "atomic" input register
    and a "multi-wire" output register. If the user wants to
    fully decompose it further, they can do so in a build_composite(...).

    We'll keep it simple: one left RegisterSpec, one right RegisterSpec
    with shape that matches your “split” logic.
    """

    dtype: Any = field(converter=canonicalize_dtype)
    @dtype.validator
    def _validate_dtype(self, attribute, value):
        # Reject any symbolic data_width
        from sympy import Basic as SymExpr

        width = getattr(value, "data_width", None)
        if isinstance(width, SymExpr) or (hasattr(width, "free_symbols") and width.free_symbols):
            raise ValueError(f"Cannot split with symbolic data_width: {width}")

    @cached_property
    def signature(self) -> Signature:
        """
        INPUT  (LEFT)   : one atomic register carrying `self.dtype`
        OUTPUT (RIGHT)  : an *array* of wires whose length equals the
                          **top-level element count** of the input value.
        Rules
        -----
        • If `dtype` is TensorType/MatrixType  
          –  Each output wire carries the *element_type*  
          –  Fan-out =  product(dtype.shape)  (flatten the tensor)
        •	Scalar inputs (e.g. CInt(4), QInt(8)) 
            –  Each output wire carries one CBit or QBit, respectively  
            –  Fan-out =  dtype.data_width  (bit / qubit splitter)
            
            This is analogous to Qualtran: one "left" register is dtype,
            the "right" register is an array of QBit or CBit of shape=(data_width,).
        """
      
        left_reg = RegisterSpec("arg", dtype=self.dtype, shape=(), flow=Flow.LEFT)
        total_wires = (
            self.dtype.total_bits
            if hasattr(self.dtype, "total_bits")
            else self.dtype.data_width
        )
        shp          = _shape_of(self.dtype)
        elem_dt      = _element_dtype_of(self.dtype)
        elem_width   = elem_dt.data_width if isinstance(elem_dt, DataType) else 1
        n_elems      = prod(shp) or 1
        logger.debug(f'dtype: {self.dtype}, elem_dt: {elem_dt} --> total_wires: {total_wires}\n - isinstance(self.dtype, CType): {isinstance(self.dtype, CType)} ')
        # pick out_dtype (one bit / qubit per split)
        if isinstance(self.dtype, (TensorType, MatrixType)):
            out_dtype = elem_dt
            fan_out   = prod(_shape_of(self.dtype)) or 1
            logger.debug(f"{self.dtype} with element dtype: {self.dtype.element_type} output dtype: {out_dtype}, elem_width: {elem_width}, n_elems: {n_elems}, fan_out: {fan_out}")
       
        elif isinstance(self.dtype, CType):
            
            out_dtype = CBit()
            fan_out   = self.dtype.data_width
            logger.debug(f"{self.dtype}, output dtype: {out_dtype}, fan_out: {fan_out}")
        elif isinstance(self.dtype, QType):
            out_dtype = QBit()
            fan_out   = self.dtype.data_width
            logger.debug(f"{self.dtype}, output dtype: {out_dtype}, fan_out: {fan_out}")
        else:
            raise TypeError(f"Split cannot infer element_type from {self.dtype}")
        # logger.debug(f'total_wires: {total_wires}')
        if fan_out == 1:
            right_shape = ()
        else:
            right_shape = (fan_out,)
        right_reg = RegisterSpec(
            name="arg",
            dtype=out_dtype,
            shape=right_shape,
            flow=Flow.RIGHT,
        )
        return Signature([left_reg, right_reg])
    
    def set_expected_input_properties(self):
        """
        Possibly define a dictionary-based property if you want old checks.
        """
        self.expected_input_properties = [{"Data Type": type(self.dtype), "Usage": "Split"}]
    def __repr__(self):
        return 'Split'
    def __hash__(self):
        return hash((self.dtype,self.num_splits))
    def __eq__(self, other):
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (self.dtype,) == (other.dtype,)

@define  
class Join(Process):
    """
    Join an *array* of wires into a single atomic register of `dtype`.

    INPUT  (LEFT)
        • Tensor/Matrix target
              – if its element-type is *multi-bit* classical
                    in_shape = (*tensor.shape, elem_width)
                    wire_dtype = CBit()
              – else
                    in_shape = tensor.shape
                    wire_dtype = element_type
        • Scalar CType
              in_shape = (dtype.data_width,)
              wire_dtype = CBit()
        • Scalar QType
              in_shape = (dtype.data_width,)
              wire_dtype = QBit()
        • CBit / QBit target
              in_shape = ()
              wire_dtype = same CBit / QBit

    OUTPUT (RIGHT)
        a single atomic register carrying `self.dtype`
    """

    dtype: DataType = field(converter=canonicalize_dtype)

    @cached_property
    def signature(self) -> Signature:
        # Helper lambdas
        shp       = _shape_of(self.dtype)              # tuple
        elem_dt   = _element_dtype_of(self.dtype)      # DataType | type
        elem_bits = elem_dt.data_width if isinstance(elem_dt, DataType) else 1

        # ----------- decide wire_dtype and in_shape -------------------
        if isinstance(self.dtype, (TensorType, MatrixType)):
            wire_dtype = elem_dt
            in_shape   = shp or ()

        elif isinstance(self.dtype, CType):
            # Re-instantiate the same CType with its bit_width
            in_dtype   = self.dtype.__class__(bit_width=self.dtype.bit_width)

            if isinstance(self.dtype, CBit):
                wire_dtype = CBit()
                in_shape   = (self.dtype.data_width,)
            else:
                wire_dtype = in_dtype            # multi-bit classical scalar
                in_shape   = (self.dtype.data_width,)

        elif isinstance(self.dtype, QType):
            wire_dtype = QBit()
            in_shape   = (self.dtype.data_width,)

        else:
            raise TypeError(f"Join cannot handle dtype {self.dtype!r}")

        # ---------------- build Signature -----------------------------
        left_reg  = RegisterSpec("arg", dtype=wire_dtype, shape=in_shape, flow=Flow.LEFT)
        right_reg = RegisterSpec("arg", dtype=self.dtype,      flow=Flow.RIGHT)
        return Signature([left_reg, right_reg])
    # @cached_property
    # def signature(self) -> Signature:
    #     shp          = _shape_of(self.dtype)
    #     elem_dt      = _element_dtype_of(self.dtype)
    #     elem_width   = elem_dt.data_width if isinstance(elem_dt, DataType) else 1
    #     n_elems      = prod(shp) or 1

    #     # ---------- determine wire_dtype & in_shape --------------------
    #     if isinstance(self.dtype, (TensorType, MatrixType)):
    #         wire_dtype  = _element_dtype_of(self.dtype)
    #         in_shape  = _shape_of(self.dtype) or ()

    #         # if isinstance(elem_dt, CType) and elem_w > 1:
    #         #     # one CBit per *bit* of every element
    #         #     wire_dtype = CBit()
    #         #     in_shape   = (*self.dtype.shape, elem_w) if n_elems != 1 else (elem_w,)
    #         # else:
    #         #     wire_dtype = elem_dt
    #         #     in_shape   = self.dtype.shape

    #     elif isinstance(self.dtype, CType):
    #         in_dtype  = self.dtype.__class__(**getattr(self.dtype, "__dict__", {}))
    #         if isinstance(self.dtype, CBit):
    #             wire_dtype = CBit()
    #             in_shape   = (self.dtype.data_width,)
    #         else:
    #             wire_dtype = in_dtype
    #             in_shape   = (self.dtype.data_width,)

    #     elif isinstance(self.dtype, QType):

    #         wire_dtype = QBit()
    #         in_shape   = (self.dtype.data_width,)

    #     else:
    #         raise TypeError(f"Join cannot handle dtype {self.dtype!r}")

    #     # ---------- LEFT array register --------------------------------
    #     left_reg = RegisterSpec(
    #         name="arg",
    #         dtype=wire_dtype,
    #         shape=in_shape,
    #         flow=Flow.LEFT,
    #     )

    #     # ---------- RIGHT atomic register ------------------------------
    #     right_reg = RegisterSpec(
    #         name="arg",
    #         dtype=self.dtype,
    #         shape=(),
    #         flow=Flow.RIGHT,
    #     )

    #     return Signature([left_reg, right_reg])
    @dtype.validator
    def _validate_dtype(self, attribute, value):
        if value.is_symbolic():
            raise ValueError(f"{self} Cannot split with symbolic data_width.")
    def set_expected_input_properties(self):
        # If you want dictionary-based checks, define them here
        self.expected_input_properties = [{"Data Type": type(self.dtype), "Usage": "Join"}]
    def __repr__(self):
        return 'Join'
    def __hash__(self):  return hash(self.dtype)
    def __eq__(self, other):
        return isinstance(other, Join) and self.dtype == other.dtype

def _to_set(x: Iterable[ProcessInstance]) -> FrozenSet[ProcessInstance]:

    return frozenset(x)

@define
class CompositeMod(Process):
    """
    CompositeMod represents a composite (i.e. decomposed) process block.
    It is constructed from a set of Connections, a Signature (of the composite),
    and (optionally) a set of ProcessInstances.

    For logging and debugging purposes, it also stores the normalized mapping
    from the parent Process (which maps generic register names such as arg0, arg1, etc.
    to Data objects that include legacy "Usage" keys).

    Note: This class does not merge legacy input_data with the signature mapping;
    it only displays them side-by-side.
    """
    connections:Tuple[Connection, ...] = field(converter=_to_tuple)
    _signature: Signature
    pinsts: FrozenSet[ProcessInstance] = field(converter=_to_set)
    normalized_map: Optional[Dict[str, Data]] = field(default=None)  # store parent's normalized_map


    @cached_property
    def all_ports(self) -> Set[Port]:
        """A set of all 'Port's present in the computational graph."""
        ports = {cxn.left for cxn in self.connections}
        ports |= {cxn.right for cxn in self.connections}
        return set(ports)
   
    def final_ports(self) -> Dict[str,PortT]:
        """
        Final output ports

        """
        g=self._pinst_graph
        if RightDangle not in g:
            return {}
        
        final_preds, _ = _pinst_to_cxns(RightDangle, pinst_graph=g)
        return _cxns_to_port_dict(
            self.signature.rights(),
            final_preds,
            get_me=lambda x: x.right,
            get_assign=lambda x: x.left,
        )
        
    def as_composite(self):
        """Returns itself as it is already a composite block."""
        return self
    
    @cached_property
    def _pinst_graph(self) -> nx.DiGraph:
        """Get a cached version of this composite Module's ProcessInstance graph.

        The ProcessInstance graph (or pinst_graph) records edges between bloq instances
        and stores the `Connection` (i.e. Port-Port) information on an edge attribute
        named `cxns`.

        
        """
   
        return _create_graph(self.connections, self.pinsts)
    def iter_process_connections(
        self,
    ) -> Iterator[Tuple[ProcessInstance, List[Connection], List[Connection]]]:
        """Iterate over Process and their connections in topological order.

        Yields:
            A process instance, its predecessor connections, and its successor connections. The
            process instances are yielded in a topologically-sorted order. The predecessor
            and successor connections are lists of `Connection` objects feeding into or out of
            (respectively) the pinst. Dangling nodes are not included as the pinst (but
            connections to dangling nodes are included in predecessors and successors).
            Every connection that does not involve a dangling node will appear twice: once as
            a predecessor and again as a successor.
        """
       
        g = self._pinst_graph
        
        # for process_instance in greedy_topological_sort(g):
        for process_instance in nx.topological_sort(g):
            if isinstance(process_instance, DanglingT):
                continue
            pred_cxns, succ_cxns = _pinst_to_cxns(process_instance, pinst_graph=g)
            yield process_instance, pred_cxns, succ_cxns


    def iter_composite(
            self,
    ) -> Iterator[Tuple[ProcessInstance, Dict[str, PortT], Tuple[PortT, ...]]]:
        """
        Iterate over process instances and their input ports. 

        This method is helpful for "adding from" this existing composite process. You must
        use `map_ports` to map this CompositeMod's ports to the correct ones for the
        new process.
        """
        for pinst, preds, succs in self.iter_process_connections():
            in_ports = _cxns_to_port_dict(
                pinst.process.signature.lefts(),
                preds,
                get_me=lambda x: x.right,
                get_assign=lambda x: x.left,
            )
            out_ports = tuple(_to_port(pinst, reg) for reg in pinst.process.signature.rights())
            yield pinst, in_ports, out_ports

    
    @staticmethod
    def _debug_pinst(g: nx.DiGraph, pinst: Union[ProcessInstance, DanglingT]):
        # Helper to format a process instance
        def simple_name(pi):
            if isinstance(pi, ProcessInstance):
                # Assumes each process instance has an attribute 'i'
                return f"{pi.process.__class__.__name__}<{pi.i}>"
            return str(pi)
        
        # Header for the current process instance
        header = simple_name(pinst) if isinstance(pinst, ProcessInstance) else str(pinst)
        lines = [header]
        
        pred_cxns, succ_cxns = _pinst_to_cxns(pinst, pinst_graph=g)
        for pred_cxn in pred_cxns:
            # Use simple_name to format the process instance on the left side of the connection.
            lines.append(
                f"  {simple_name(pred_cxn.left.process_instance)}.{pred_cxn.left.pretty()} -> {pred_cxn.right.pretty()}"
            )
        for succ_cxn in succ_cxns:
            # And similarly for the process instance on the right side.
            lines.append(
                f"  {succ_cxn.left.pretty()} -> {simple_name(succ_cxn.right.process_instance)}.{succ_cxn.right.pretty()}"
            )
        return lines
    
    def debug_text(self) -> str:
        """Print connection information to assist in debugging.

        The output will be a topologically sorted list of BloqInstances with each
        topological generation separated by a horizontal line. Each bloq instance is followed
        by a list of its incoming and outgoing connections. Note that all non-dangling
        connections are represented twice: once as the output of a binst and again as the input
        to a subsequent binst.
        """
        g = self._pinst_graph
        gen_texts = []
        for gen in nx.topological_generations(g):
            gen_lines = []
            for pinst in gen:
                if isinstance(pinst, DanglingT):
                    continue

                gen_lines.extend(self._debug_pinst(g, pinst))

            if gen_lines:
                gen_texts.append('\n'.join(gen_lines))

        delimited_gens = ('\n' + '-' * 20 + '\n').join(gen_texts)
        return delimited_gens
    
    @property
    def describe(self):
        description = self.debug_text()
        logger.debug(description)
        # print(description)
    def print_tabular(self):
        """
        Print a table of the sub-processes (ProcessInstances) in this CompositeMod,
        adding a 'legacy usage' column that attempts to show the original usage key
        from the parent's Data objects (self.normalized_map). We do NOT rely on
        the register name being the same as the param_name. Instead, we try two steps:

        1) Direct name match: If port.reg.name is in normalized_map, we use that Data's usage.
        2) Fallback shape/dtype match: If no direct name match, we see if exactly one
        entry in normalized_map has the same shape (and optional dtype) as this port.
        If so, we use that usage. Otherwise, we print '-'.

        This approach does not require adding usage to RegisterSpec.
        """
        from rich.pretty import pretty_repr
        import numpy as np

        # Build a dictionary param_name -> (Data, usage_str) from self.normalized_map
        # so we can do fallback shape-based matching if direct name match fails.
        def get_usage_str(dobj: Data) -> str:
            # logger.debug(f'dobj: {dobj}')
            # prefer Data.properties["Usage"], else fallback to Data.metadata.hint
            usage = dobj.properties.get("Usage") or dobj.metadata.hint
            return usage if usage else "-"

        if self.normalized_map is None:
            # logger.debug("No normalized_map found on CompositeMod; cannot display legacy usage.")
            usage_map = {}
        else:
            usage_map = {
                k: (v, get_usage_str(v))  # store the Data object & the usage string
                for k, v in self.normalized_map.items()
            }
        # print(f"usage_map: {usage_map}, self.normalized_map: {self.normalized_map}")
        lines = []
        header = f"{'opcode':<13}  {'name':<25}  {'legacy usage':<15}  {'target':<20}  {'args':<20}  {'type':<15}  {'kwargs':<10}"
        lines.append(header)
        lines.append("-" * len(header))

        def find_usage_for_port(port: Port) -> str:
            g = self._pinst_graph
            # logger.debug(f'     -> port: {port}, port.pinst: {port.process_instance} port.reg: {port.reg}\n     ')
            next_node = next(g.neighbors(port.process_instance), None)
            neighbors_list = list(g.neighbors(port.process_instance))

            # logger.debug(f'     -> port: {port}, port.pinst: {port.process_instance} port.reg: {port.reg}\n          - neighbors_list: {neighbors_list}, next_node: {next_node}')
            if next_node and not isinstance(next_node,DanglingT):
                # if next_node.process.inputs:
                #     # logger.debug(f'     port: {port} -> next_node: {repr(next_node.process.signature)}')
                #     # usage = next_node.process.spec_to_usage[port.reg.name]
                #     # data_id = next_node.process.spec_to_input_id[port.reg.name]
                    
                #     usage = next_node.process.spec_to_usage[port.reg.name]
                #     logger.debug(f'     port: {port} -> next_node: {next_node} with id: {data_id}, legacy usage: {usage}')

                #     return usage
                return ''
            return ''

        # 1) Placeholders: Ports whose process_instance is LeftDangle
        for port in self.all_ports:
            if port.process_instance is LeftDangle:
                # logger.debug(f'port: {port.pretty()}')
                opcode_str = "placeholder"
                name_str = port.reg.name
                legacy_usage = find_usage_for_port(port)
                target_str = name_str
                args_str = "()"
                type_str = port.reg.dtype.__name__ if hasattr(port.reg.dtype, "__name__") else str(port.reg.dtype)
                kwargs_str = "{}"
                lines.append(f"{opcode_str:<13}  {name_str:<25}  {legacy_usage:<15}  {target_str:<20}  {args_str:<20}  {type_str:<15}  {kwargs_str:<10}")

        # 2) Process nodes
        g = self._pinst_graph
        for pinst in nx.topological_sort(g):
            if isinstance(pinst, DanglingT):
                continue
            opcode_str = "call_process"
            name_str = str(pinst)
            target_str = pinst.process.__class__.__name__
            usage_set = set()
            arg_list = []
            for pred in g.pred[pinst]:
                cxn_list = g.edges[pred, pinst]['cxns']
                for cx in cxn_list:
                    # logger.debug(f'pinst: {pinst}, right.pinst: {cx.right.process_instance}, left.pinst: {cx.left.process_instance} right.reg: {cx.right.reg}')
                    # logger.debug(f'cx: {type(cx.left.process_instance)}')
                    # usage_str = find_usage_for_port(cx.right)
                    # usage_set.add(usage_str)
                    if not isinstance(cx.left.process_instance,DanglingT):
                        arg_list.append(cx.left.pretty())
                    else:
                        arg_list.append(cx.right.pretty())
            usage_str = ", ".join(sorted(u for u in usage_set if u != "-")) or "-"
            args_str = f"({', '.join(arg_list)})"
            type_str = ""
            kwargs_str = "{}"
            lines.append(f"{opcode_str:<13}  {name_str:<25}  {usage_str:<15}  {target_str:<20}  {args_str:<20}  {type_str:<15}  {kwargs_str:<10}")

        # 3) Outputs: right side from RightDangle with flow=Flow.RIGHT
        for cxn in self.connections:
            if cxn.right.process_instance is RightDangle and cxn.right.reg.flow == Flow.RIGHT:
                opcode_str = "output"
                name_str = cxn.right.pretty()  # e.g. "OUT"
                usage_str = find_usage_for_port(cxn.right)
                target_str = "output"
                args_str = f"({cxn.left.pretty()},)"
                type_str = cxn.left.reg.dtype.__name__ if hasattr(cxn.left.reg.dtype, "__name__") else str(cxn.left.reg.dtype)
                kwargs_str = "{}"
                lines.append(f"{opcode_str:<13}  {name_str:<25}  {usage_str:<15}  {target_str:<20}  {args_str:<20}  {type_str:<15}  {kwargs_str:<10}")

        table_str = "\n".join(lines)
        # print(table_str)
        logger.debug(table_str)
    

    def print_tabular_fx(self) -> None:
        """
        Print a table of the composite graph in a style analogous to `torch.fx.Graph.print_tabular()`,
        with columns for opcode, name, target, args, data_type, shape, prev_nodes, next_nodes, etc.
        
        This version auto-sizes columns based on the widest entry so everything aligns nicely.
        """

        # We define the columns we want:
        columns = [
            "opcode",      # e.g. 'placeholder', 'call_process', or 'output'
            "name",        # e.g. "LeftDangle", "TwoBitOp<0>", "RightDangle"
            "target",      # e.g. "TwoBitOp", "Atom", or "output"
            "args",        # e.g. (LeftDangle.a, LeftDangle.b) ...
            "data_type",   # e.g. "CBit", "QAny(3)", or "None"
            # "shape",       # e.g. "()" or "(3,)" or "N/A"
            # "prev_nodes",  # e.g. the node names feeding into this one (excl. dangle)
            # "next_nodes",  # e.g. the node names that come after
        ]

        # We'll accumulate rows as a list of dicts, one per node
        rows = []

        def node_name(pinst_or_dangle):
            if pinst_or_dangle is LeftDangle:
                return "LeftDangle"
            if pinst_or_dangle is RightDangle:
                return "RightDangle"
            if isinstance(pinst_or_dangle, ProcessInstance):
                return f"{pinst_or_dangle.process.__class__.__name__}<{pinst_or_dangle.i}>"
            return str(pinst_or_dangle)  # fallback

        g = self._pinst_graph
        topo_nodes = list(nx.topological_sort(g))

        for node in topo_nodes:
            if node is LeftDangle:
                # placeholders
                row = dict.fromkeys(columns, "")
                row["opcode"] = "placeholder"
                row["name"]   = node_name(node)
                row["target"] = "LeftDangle"
                row["args"]   = "()"
                # For placeholders, data_type/shape might be gleaned by seeing which RIGHT edges exist
                row["data_type"] = "-"
                # row["shape"]     = "-"
                # Predecessors? None. Successors? 
                succs = list(g.successors(node))
                # row["prev_nodes"] = "None"
                # row["next_nodes"] = ", ".join(node_name(s) for s in succs if s not in (LeftDangle, RightDangle))
                rows.append(row)

            elif node is RightDangle:
                # outputs
                row = dict.fromkeys(columns, "")
                row["opcode"] = "output"
                row["name"]   = node_name(node)
                row["target"] = "RightDangle"
                row["args"]   = "()"
                row["data_type"] = "-"
                # row["shape"]     = "-"
                preds = list(g.predecessors(node))
                # row["prev_nodes"] = ", ".join(node_name(p) for p in preds if p not in (LeftDangle, RightDangle))
                # row["next_nodes"] = "None"
                rows.append(row)

            else:
                # Normal node = ProcessInstance
                pinst = node
                row = dict.fromkeys(columns, "")
                row["opcode"] = "call_process"
                row["name"]   = node_name(pinst)
                row["target"] = pinst.process.__class__.__name__

                # Gather its incoming edges to figure out args, data_type, shape, prev_nodes, etc.
                pred_cxns, succ_cxns = _pinst_to_cxns(pinst, g)

                # a) Collect arg list for row["args"]
                # Usually from the left ports
                arg_list = []
                data_types = []
                shapes = []
                for cx in pred_cxns:
                    # e.g. "LeftDangle.d1 -> TwoBitOp<0>.a"
                    left_port = cx.left
                    arg_list.append(f"{node_name(left_port.process_instance)}.{left_port.pretty()}")
                    # data_type, shape
                    dt = str(left_port.reg.dtype)
                    data_types.append(dt)
                    if left_port.reg.shape:
                        shapes.append(str(left_port.reg.shape))
                    else:
                        shapes.append("()")

                row["args"] = "(" + ", ".join(arg_list) + ")" if arg_list else "()"

                # b) For demonstration, if you want to store just the first or combined
                #    data_type / shape:
                row["data_type"] = ";".join(set(data_types)) if data_types else "-"
                # row["shape"]     = ";".join(set(shapes))     if shapes else "-"

                # c) prev_nodes, next_nodes
                prev_nodes = set(cx.left.process_instance for cx in pred_cxns if cx.left.process_instance not in (LeftDangle, RightDangle))
                next_nodes = set(cx.right.process_instance for cx in succ_cxns if cx.right.process_instance not in (LeftDangle, RightDangle))
                # row["prev_nodes"] = ", ".join(node_name(n) for n in prev_nodes) if prev_nodes else "None"
                # row["next_nodes"] = ", ".join(node_name(n) for n in next_nodes) if next_nodes else "None"

                rows.append(row)

        #
        # Now we compute the max width for each column, then print them aligned
        #
        col_widths = {}
        for col in columns:
            col_widths[col] = max(len(col), max(len(str(r[col])) for r in rows))

        # Print header
        def format_cell(col_name, text):
            width = col_widths[col_name]
            return f"{text:<{width}}"

        header_line = "  ".join(format_cell(c, c) for c in columns)
        sep_line = "  ".join("-" * col_widths[c] for c in columns)

        print(header_line)
        print(sep_line)

        for row in rows:
            line = "  ".join(format_cell(c, row[c]) for c in columns)
            print(line)
    def print_tabular_fx2(self) -> None:
        """
        Print a table of the composite graph in a style analogous to `torch.fx.Graph.print_tabular()`,
        with columns like: opcode, name, target, args, etc.
        """
        # Prepare columns
        headers = [
            "opcode",     # e.g. 'placeholder', 'call_process', or 'output'
            "name",       # symbolic name for this node or register
            "target",     # e.g. the Process class name if it's a normal node
            "args",       # a tuple of input ports
            "type",       # optional: might store the data type
            "Prev Node",  # which node(s) feed into this
            "Next Node",  # which node(s) this feeds into
            "Input Nodes" # the actual inputs
        ]
        col_line = "  ".join(f"{h:<20}" for h in headers)

        lines = [col_line, "-" * len(col_line)]

        # Helper to get the textual name for a node or dangling
        def node_name(pinst_or_dangle):
            if pinst_or_dangle is LeftDangle:
                return "LeftDangle"
            if pinst_or_dangle is RightDangle:
                return "RightDangle"
            if isinstance(pinst_or_dangle, ProcessInstance):
                return str(pinst_or_dangle)  # e.g. "TwoBitOp<0>"
            return str(pinst_or_dangle)

        # Build a topological ordering of nodes
        g = self._pinst_graph
        topo = list(nx.topological_sort(g))

        for node in topo:
            # Distinguish placeholders (LeftDangle) vs normal vs outputs (RightDangle)
            if node is LeftDangle:
                # All left registers => placeholders
                # Gather the associated left regs
                opcode = "placeholder"
                name_ = node_name(node)
                target = "LeftDangle"
                # Gather edges from LeftDangle -> real node
                # The "args" = empty in this example
                # type_ = "?" or we can skip
                succs = list(g.successors(node))
                next_nodes = ",".join(node_name(s) for s in succs)
                # For 'placeholder', no "Prev Node"
                # For "Input Nodes", probably none
                row = f"{opcode:<20}  {name_:<20}  {target:<20}  ()         ?          None       {next_nodes:<10}  []"
                lines.append(row)

            elif node is RightDangle:
                # gather edges from something -> RightDangle
                opcode = "output"
                name_ = node_name(node)
                target = "output"
                preds = list(g.predecessors(node))
                prev_nodes = ",".join(node_name(p) for p in preds)
                row = f"{opcode:<20}  {name_:<20}  {target:<20}  ()         ?    {prev_nodes:<10}  None          []"
                lines.append(row)

            else:
                # It's a ProcessInstance (a normal node)
                pinst = node
                opcode = "call_process"
                name_ = node_name(pinst)
                # e.g. "TwoBitOp" or "Atom" or "NAtomParallel"
                target = pinst.process.__class__.__name__
                # Gather its input ports from predecessor edges
                pred_cxns, succ_cxns = _pinst_to_cxns(pinst, g)  # your helper that returns edges
                prev_nodes = set(cx.left.process_instance for cx in pred_cxns if cx.left.process_instance not in (LeftDangle, RightDangle))
                next_nodes = set(cx.right.process_instance for cx in succ_cxns if cx.right.process_instance not in (LeftDangle, RightDangle))
                prev_str = ",".join(node_name(n) for n in prev_nodes)
                next_str = ",".join(node_name(n) for n in next_nodes)

                # The 'args' = each input port’s name. Or you can store them as (cx.left.pretty()) etc.
                arg_list = []
                for cx in pred_cxns:
                    # E.g. "LeftDangle.a -> TwoBitOp<0>.a"
                    # We'll store the left side’s name
                    arg_list.append(cx.left.pretty())
                args_str = f"({', '.join(arg_list)})"

                # If you want a 'type' column, you can do pinst.process.signature or something
                type_ = "None"

                # Build the row
                row = f"{opcode:<20}  {name_:<20}  {target:<20}  {args_str:<20}  {type_:<20}  {prev_str:<20}  {next_str:<20}  {arg_list}"
                lines.append(row)

        # Print them out
        table_str = "\n".join(lines)
        print(table_str)
    def __str__(self):

        return f'CompositeMod([{len(self.pinsts)} subbloqs...])'

def _create_graph(
    cxns: Iterable[Connection], nodes: Iterable[ProcessInstance] = ()
) -> nx.DiGraph:
    """Create a NetworkX graph with support for multiple destinations per edge."""
    pinst_graph = nx.DiGraph()
    for cxn in cxns:
        source = cxn.left.process_instance
        edge = (source, cxn.right.process_instance)
        
        if edge in pinst_graph.edges:
            pinst_graph.edges[edge]['cxns'].append(cxn)
        else:
            pinst_graph.add_edge(*edge, cxns=[cxn])
    pinst_graph.add_nodes_from(nodes)
    return pinst_graph

def _pinst_to_cxns(
    pinst: Union[ProcessInstance, DanglingT], pinst_graph: nx.DiGraph
):
    """Helper method to extract all predecessor and successor Connections for a binst."""
    pred_cxns: List[Connection] = []
    for pred in pinst_graph.pred[pinst]:
        pred_cxns.extend(pinst_graph.edges[pred, pinst]['cxns'])

    succ_cxns: List[Connection] = []
    for succ in pinst_graph.succ[pinst]:
        succ_cxns.extend(pinst_graph.edges[pinst, succ]['cxns'])

    return pred_cxns, succ_cxns



def _cxns_to_port_dict(
    regs: Iterable[RegisterSpec],
    cxns: Iterable[Connection],
    get_me: Callable[[Connection], Port],
    get_assign: Callable[[Connection], Port],
) -> Dict[str, PortT]:
    """Helper function to get a dictionary of ports from a list of connections.

    Args:
        regs: Left or right RegisterSpec objs (used as a reference to initialize multidimensional
            registers correctly).
        cxns: Predecessor or successor connections from which we get the ports of interest.
        get_me: A function that says which port is used to derive keys for the returned
            dictionary. Generally: if `cxns` is predecessor connections, this will return the
            `right` element of the connection and opposite of successor connections.
        get_assign: A function that says which port is used to derive the values for the
            returned dictionary. 

    Returns:
        port_dict: A dictionary mapping RegisterSpec.name to the selected ports
    """
    port_dict: Dict[str, PortT] = {}

    # Initialize multi-dimensional dictionary values.
    for reg in regs:
        if reg.shape:
            port_dict[reg.name] = np.empty(reg.shape, dtype=object)

    # In the abstract: set `port_dict[me] = assign`. Specifically: use the RegisterSpec.name as
    # keys and handle multi-dimensional registers.
    for cxn in cxns:
        me = get_me(cxn)
        assign = get_assign(cxn)

        if me.reg.shape:
            port_dict[me.reg.name][me.idx] = assign 
        else:
            port_dict[me.reg.name] = assign

    return port_dict

class _IgnoreAvailable:
    """Used as an argument in `_to_port` to ignore any `available.add()` tracking."""

    def add(self, x: Hashable):
        pass

def _to_port(
    process_instance: Union[ProcessInstance, DanglingT],
    reg: RegisterSpec,
    available: Union[Set[Port], _IgnoreAvailable] = _IgnoreAvailable(),
) -> PortT:
    """ 
    Create the data port or array of ports for a spec object.

    Args:
        process_instance: The output dataport's process instance.
        reg: The RegisterSpec object instance.
        available: Use for bookkeeping.

    Returns:
        A single Port or an array of Ports for the given RegisterSpec object.
    """

    # Handle shaped data (non-symbolic)
    if reg.shape:
        ports = np.empty(reg.shape, dtype=object)
        for idx in reg.all_idxs():
            port = Port(process_instance, reg, index=idx)
            ports[idx] = port
            available.add(port)
        return ports

        

    # Fallback for unshaped data
    port = Port(process_instance, reg)
    available.add(port)
    return port

def _process_ports(registers: Iterable['RegisterSpec'], 
                       in_ports: Mapping[str,PortInT], 
                       debug_str: str,
                       func: Callable[[Port, RegisterSpec, Tuple[int, ...]], None],):
    """
    Processes the connection between input Ports in the context of register specs
    expected by a process's signature.

    This implements the following outer loop and calls 'func(indexed_port, spec, index)' for
    every `spec` and corresponding ports (from `in_ports`) in the input.

    This is where we should perform input validation to make sure that a set of spec names used as keys
    for `in_ports` is identical to the set of inputs passed in.

    Args:
        registers: Iterable of RegisterSpec (schema) expected by the process.
        in_ports: Dictionary of input Port objects provided to the process.
        func: Callback function that handles the connection between a Port and a RegisterSpec object.

    Raises:
        ValueError: If a required input is missing or if an unexpected input is provided.

    TODO: Figure out if this is where we should actually handle the "user" inputs (or Process inputs for that matter)
    and map to the signature? Unsure.
    """
    unchecked_names: Set[str] = set(in_ports.keys())
   
    logger.debug(f"Processing port keys: {unchecked_names}")

    
    for reg in registers:
        # Get the ID of the expected data register
        reg_name = reg.name
        
        
        try:
            in_port = np.asarray(in_ports[reg_name])
        except KeyError:
            raise KeyError(f"Incorrect port name '{reg_name}' not in: {in_ports}")
       
        
        
        unchecked_names.remove(reg_name)
        # logger.debug(f"Resolved shape for Data {data_id}: {data_register.resolve_shape()}")

        
        for left_index in reg.all_idxs():
            logger.debug(f"Processing index {left_index} for reg '{reg}'")
            indexed_port = in_port[left_index]
            assert isinstance(indexed_port,Port), indexed_port

            
            func(indexed_port, reg, left_index)
            # We need to add a RegisterSpec consistentancy check here. Partly why I think that 
            # this function may be of use at runtime with user input args. 
            if not check_dtypes_consistent(indexed_port.reg.dtype, reg.dtype, severity=DTypeCheckingSeverity.LOOSE,classical_level= C_PromoLevel.PROMOTE,quantum_level= Q_PromoLevel.LOOSE):
                extra_str = (
                    f"{indexed_port.reg.name}: {indexed_port.reg.dtype} vs {reg.name}: {reg.dtype}"
                )
                raise ValueError(
                    f"{debug_str} register dtypes are not consistent {extra_str}."
                )
    if unchecked_names:
        raise ValueError(f"Unexpected input provided for data ID: {reg_name}. {[reg.name for reg in registers]}")
   

class ProcessBuilder:
    def __init__(self, immutable: bool = False):
        self._pinsts: Set[ProcessInstance] = set()
        self._edges = []
        self._connections = []
        self._data = []
        self._index = 0
        self._available: Set[Port] = set()
        self.immutable = immutable
    
    def add_register_from_datatype(self,reg: Union[str, RegisterSpec], dtype: Optional[DataType] = None):
        """
        Add a new typed register to the composite process being built.

        """
        # logger.debug(f"Adding register {reg}")
        if isinstance(reg,RegisterSpec):
            if dtype is not None:
                raise ValueError(f"data type must be specified ...")
            
        else:
            if not isinstance(reg, str):
                raise ValueError(f"data type must at least be specified.. {reg}")
            
            if not isinstance(reg,DataType):
                raise ValueError(f"dtpe must be specified and it should be a DataType object.")
            reg = RegisterSpec(name = reg, dtype = dtype)
        self._data.append(reg)
        if reg.flow & Flow.LEFT:
            # logger.debug(f"Converting Data obj to Port obj")
            return _to_port(LeftDangle, reg, available=self._available)
        return None
    @overload
    def add_register(self, reg: RegisterSpec, bitsize: None = None) -> Union[None, PortT]: ...

    @overload
    def add_register(self, reg: str, bitsize: int) -> PortT: ...

    def add_register(
        self, reg: Union[str, RegisterSpec], bitsize: Optional[int] = None
    ) -> Union[None, PortT]:
        """
        """
        if bitsize is not None:
            return self.add_register_by_datatype(reg, CBit() if bitsize == 1 else CAny(bitsize))
        return self.add_register_from_datatype(reg)

    
    @classmethod
    def from_signature(
        cls, signature: Signature, immutable: bool = True
    ) -> Tuple['ProcessBuilder', Dict[str,PortT]]:
        """
            Constructs a ProcessBuilder with a pre-defined signature.

            *not implemnted yet* This constructor is used to decompose larger 
            Nodes/Process objects into their subroutine, if e.g. you're decomposing 
            an existing Node and need the signatures to match. For exampple,  
            `Node.decompose()` or 'Process.decompose().
        """
        builder = cls(immutable = False)
        initial_ports: Dict[str, PortT] = {}
        for reg in signature:
            if reg.flow & Flow.LEFT:
   
                dataport = builder.add_register_from_datatype(reg)
                assert dataport is not None
                initial_ports[reg.name] = dataport
            else:
                builder.add_register_from_datatype(reg)
        
        builder.immutable = immutable
        # logger.debug(f"Returning initial ports: {initial_ports}")
        return builder, initial_ports
    def add_d(self, process_model, **in_ports: PortInT):
        """
            Add a new Process instance to the computational DAG. 

            Returns:
                A dictionary mapping right (output) Data.ids to PortInT
        """
        
        pinst = ProcessInstance(process_model, i = self._new_index())
        # logger.debug(f"     - add_d: {process_model} instance: {pinst}")
        return dict(self._add_process_instance(pinst, in_ports=in_ports))
    def add_t(self, process_model: Process, **in_ports: PortInT) -> Tuple[PortT,...]:
        """
        Add a new process instance to the compute graph. Always return a tuple of Port objects.
        returns:
            a `Port` or an array thereof for each output register ordered according to
            process_model.signature. 

        """
        pinst = ProcessInstance(process_model, i = self._new_index())
        # logger.debug(f"New ProcessInstance: {pinst}")
        return tuple(port for _,port in self._add_process_instance(pinst, in_ports=in_ports))
    
    def add(self, process_model: Process, **in_ports: PortInT):
        """
        Add a new process instance to the compute graph. Should be the main function for building
        a CompositeMod. Each call will add the new process instance, wire up the ports from prior
        operations in the new process, and returns to be used for subsequent ops.

        Should raise an error if provided an invalid addition. data ports must be used exactly 
        once and they must match the 'Register' specifications of the process_model.

        """
        # logger.debug(f'in_ports: {in_ports}')
        outs = self.add_t(process_model, **in_ports)
        if len(outs) == 0:
            return None
        if len(outs) == 1:
            return outs[0]
        return outs
    
    def add_process(self, process_model, **in_ports: PortInT):
        """ 
            Add a new node to the control-flow graph. It should add the new node
            (carrying a process model), wire up the data ports from the prior
            Node/operation's into the new Node, and then returns the new dataports
            to be used in subsequent Node's (tuple)

            Port objects must be used exactly once and ports must match the 'Data' 
            specifications of the new process.

            **in_ports: 
        
        """
        
        process_instance = ProcessInstance(process_model, i = self._new_index())
        # logger.debug(f"Adding Process Instance {process_instance}. inports: {in_ports}")
        # for i,k in in_ports.items():
        #     logger.debug(f"     - {i}: {type(k[0][0])} {k[0]}")

        outs = tuple(port for _,port in self._add_process_instance(process_instance, in_ports=in_ports))

        # logger.debug(f"Ports wired up from prior opss into new Process ({len(outs)}): {outs}")
        if len(outs) == 0:
            return None
        if len(outs) == 1:
            return outs[0]
        return outs
    
    def add_node(self, node: Node, **in_ports: PortInT) -> Dict[str,PortT]:
        """
            returns a dictionary mapping right (output) data names to PortT
        """
        process_instance = ProcessInstance(node.process, i=self._new_index())
        return dict(self._add_process_instance(process_instance,in_ports))

    
    
    def _add_cxn(
        self,
        process_instance: Union[ProcessInstance, DanglingT],
        indexed_port: Port,
        data: Data,
        idx: Tuple[int, ...],
    ) -> None:
        """Helper function to be used as the base for the `func` argument of `_process_ports`.

        This creates a connection between the provided input `indexed_port` to the current process instance's
        `(data, idx)`.
        """
        # logger.debug(f"available: {[type(tmp) for tmp in self._available]}")
        try:
            # print(f"indexed_port: {indexed_port}")
            self._available.remove(indexed_port)
        except KeyError:
            # print(f"process instance: {process_instance} {isinstance(process_instance, DanglingT)}")
            if isinstance(process_instance,DanglingT):
                process = process_instance
            else:
                process = process_instance.process

            
            raise ValueError(
                f"{indexed_port} is not an available Port for `{process}.{data.id}`."
            ) from None
      

       
        out_port = Port(process_instance, data, idx)
        cxn = Connection(
                left=indexed_port,
                right=out_port 
        )
        # print(f"New connection between {indexed_port} and {out_port}: {cxn}")
        self._connections.append(cxn)
    




    def _add_process_instance(self,process_instance:ProcessInstance, in_ports: Mapping[str, PortInT]
    ) -> Iterator[Tuple[str,PortT]]:
        """ Add a process instance.
        
        """
        
        self._pinsts.add(process_instance)
        
        process = process_instance.process
        # logger.debug(f'process: {process}')
        # logger.debug(f'{process.signature}')
        def _add_edge(indexed_port: Port, spec: RegisterSpec, index: Tuple[int,...]):
            # Create a connection between the indexed port to the current process instance's (data, idx)
            # logger.debug(f"Adding edge: indexed port: {indexed_port}, process_instance: {process_instance}, idx: {index}")
            self._add_cxn(process_instance, indexed_port, spec, index)
        # logger.debug(f"process.signature.lefts: {process.signature._lefts}")
        # Process input Ports based on the signature and connect them
        _process_ports(
            registers=process.signature.lefts(),
            in_ports=in_ports,
            debug_str=str(process),
            func=_add_edge
        )
        
        yield from (
            (spec.name, _to_port(process_instance, spec, available=self._available))
            for spec in process.signature.rights()
        )

    def finalize(self, *args, normalized_map=None, **final_ports: PortT) -> CompositeMod:
        # Helper function to build a new RIGHT-flow RegisterSpec for a final output
        def _infer_match(name: str, port: PortT) -> RegisterSpec:
            """
            Create a RegisterSpec that matches the shape/dtype of `port`,
            labeled with the given `name` and Flow.RIGHT.
            """
            if isinstance(port, Port):
                return RegisterSpec(
                    name=name,
                    dtype=port.reg.dtype,
                    flow=Flow.RIGHT
                )
            # If it's an array of Ports, pick the first element's dtype, etc.
            # TODO: handle shaped arrays to adapt further
        
            first_port = port.reshape(-1)[0]
            return RegisterSpec(
                name=name,
                dtype=first_port.reg.dtype,
                shape=port.shape,
                flow=Flow.RIGHT
            )
        # Helper function that adds a connection from the final port to RightDangle
        def _finish_edges(indexed_port: Port, spec: RegisterSpec, idx: Tuple[int, ...]):
            """
            Called by _process_ports for each final output register.
            Creates a connection from 'indexed_port' to a RightDangle port.
            """
            self._add_cxn(RightDangle, indexed_port, spec, idx)
        # def _infer_match(name: str, port: PortT) -> RegisterSpec:
        #     """Go from Port -> RegisterSpec, but use a specific name for the data."""
        #     if isinstance(port, Port):
        #         return RegisterSpec(name=name,dtype=port.reg.dtype, flow=Flow.RIGHT)

        #     # raise TypeError(f"Unexpected type for port: {type(port)}. name: {name} Data: {port}")
        #     # print(f"name: {name}, port: {port} {port.data.shape} {dir(port)}")
        #     # logger.debug(f"")
        #     # print(Data(data=port.data,id = name, properties=port.properties, shape=port.data.shape,flow=Flow.RIGHT))
        #     right_data = RegisterSpec(name=name,dtype=port.reshape(-1)[0].reg.dtype,shape=port.shape,flow=Flow.RIGHT)
            
        #     return right_data
        
        # def _finish_edges(indexed_port: Port, data: Data, idx: Tuple[int, ...]):
        #     # close over `RightDangle`
        #     return self._add_cxn(RightDangle, indexed_port, data, idx)
        # If items from `final_ports` do not already exist in `self._data`,
        # add them with Flow.RIGHT to the signature data array.
        right_side_names = [spec.name for spec in self._data if spec.flow & Flow.RIGHT]

        for name, port in final_ports.items():
            if name not in right_side_names:
                # Create a new RegisterSpec for this final output
                new_spec = _infer_match(name, port)
                self._data.append(new_spec)

        
        signature = Signature(self._data)

        _process_ports(
            registers=signature.rights(),
            in_ports=final_ports,
            debug_str='Finalizing',
            func=_finish_edges
        )
        
        return CompositeMod(
            connections=self._connections,
            signature=signature,
            pinsts=self._pinsts,
            normalized_map=normalized_map  # Pass along the parent's normalized_map
        )

      

    @staticmethod
    def map_ports(
        ports: Dict[str, PortT], port_map: Iterable[Tuple[PortT, PortT]]
    ) -> Dict[str, PortT]:
        """
        Map `ports` according to `port_map`
        
        Args:
            ports: A dictionary mapping `RegisterSpec` names (keys) to `Port`s (or arrays
                of `Port`s) where the values of this dictionary will be mapped.
            port_map: An iterable of (old_port, new_port) tuples that inform how to
                perform the mapping. Note that this is a list of tuples (not a dictionary)
                because `old_port` may be an unhashable numpy array of Port.

        Returns:
            A mapped version of `ports`.
        """
        # flatten out any numpy arrays
        flat_soq_map: Dict[Port, Port] = {}
        for old_ports, new_ports in port_map:
            if isinstance(old_ports, Port):
                assert isinstance(new_ports, Port), new_ports
                flat_soq_map[old_ports] = new_ports
                continue

            assert isinstance(old_ports, np.ndarray), old_ports
            assert isinstance(new_ports, np.ndarray), new_ports
            assert old_ports.shape == new_ports.shape, (old_ports.shape, new_ports.shape)
            for o, n in zip(old_ports.reshape(-1), new_ports.reshape(-1)):
                flat_soq_map[o] = n

        # Then use vectorize to use the flat mapping.
        def _map_port(port: Port) -> Port:
            # Helper function to map an individual port.
            return flat_soq_map.get(port, port)

        # Use `vectorize` to call `_map_port` on each element of the array.
        vmap = np.vectorize(_map_port, otypes=[object])

        def _map_ports(ports: PortT) -> PortT:
            if isinstance(ports, Port):
                return _map_port(ports)
            return vmap(ports)

        return {name: _map_ports(ports) for name, ports in ports.items()}    
        
    def _new_index(self):
        """Generate a new unique index for ProcessInstance."""
        index = self._index
        self._index += 1
        return index
    
    def split(self, port: Port, num_parts: Optional[int] = None) -> List[PortT]:
        """Split a register into multiple smaller registers."""
        dt = port.reg.dtype

        # split_ports = []
        # for i in range(num_parts):
        #     split_ports.append(self.add_process(Split(size=1), reg=reg))
        if not isinstance(port, Port):
            raise ValueError(f'Expects a single dataport to split')
        # if isinstance(dt, (TensorType, MatrixType)) and (prod(dt.shape) or 1) <= 1:
        #     raise ValueError(f"Cannot split single-element tensor {dt}")
        # logger.debug(f"port: {port},\nreg: {port.reg},\ndtype: {port.reg.dtype}")
        return self.add(Split(dtype=dt), arg=port)

    def join(self, ports: List[PortT], dtype: Optional[DataType] = None) -> PortT:
        """Concatenate multiple ports into a single register."""
        # try:
        #     ports = np.asarray(ports)
        #     (n,) = ports.shape
        # except AttributeError:
        #     raise ValueError("Can on merge/join equal-shaped ports. (potentiall only size 1?)")
        ports = np.asarray(ports)
        (n,) = ports.shape
        if dtype is None:
            wire_dts = [p.reg.dtype for p in ports.reshape(-1)]
            # ensure mutual consistency under LOOSE severity
            for a, b in itertools.combinations(wire_dts, 2):
                if not check_dtypes_consistent(
                    a, b,
                        severity=DTypeCheckingSeverity.LOOSE,
                        classical_level=C_PromoLevel.CAST,
                        quantum_level=Q_PromoLevel.LOOSE,
                    ):
                        raise ValueError(f"Inconsistent dtypes in join: {wire_dts!r}")
                # pick a representative base
                base = wire_dts[0]
                # if it’s a TensorType, we’re really flattening an array-of-arrays,
                # so append the new axis:
                if isinstance(base, TensorType):
                    dtype = TensorType(shape=(*base.shape, n), element_type=base.element_type)
                else:
                    # for scalars/bitfields, composing n wires → a wider scalar:
                    new_width = base.data_width * n
                    cls = type(base)
                    # assume constructor signature `(bit_width, …)`
                    dtype = cls(**{**base.__dict__, "bit_width": new_width})



            
        logger.debug(f"ports.shape: {ports.shape},\ndtype: {dtype}")
        return self.add(Join(dtype=dtype), arg=ports)
    
    