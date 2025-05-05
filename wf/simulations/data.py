"""

This file contains the class definition for containing
and describing data. The Data class is a container for
arbitrary data and their properties which are utilized,
updated and passed along a problem network.

"""
from typing import List, Dict, Tuple, Iterable, Union, Sequence, Optional,overload

import numpy as np
import itertools
from dataclasses import dataclass, field
from typing import Optional, Union, Tuple, Dict
import sympy
from attrs import field, frozen

import enum
from torch.fx.experimental.symbolic_shapes import (
    free_unbacked_symbols,
    DimDynamic,
    SymbolicContext
)
import torch
from torch.fx.experimental.symbolic_shapes import ShapeEnv as TorchShapeEnv
from torch.fx.experimental.sym_node import method_to_operator, SymNode
from torch import SymInt, SymBool, SymFloat
from torch.fx.experimental.symbolic_shapes import StatelessSymbolicContext
from torch.fx.experimental.recording import record_shapeenv_event
from torch._guards import ShapeGuard, Source


from .data_types import MatrixType,DataType,TensorType, CBit,CAny,QAny

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Type

from sympy import Symbol

from collections import defaultdict
from typing_extensions import TypeGuard

from ...util.log import logging
logger = logging.getLogger(__name__)


SymbolicInt = Union[int, sympy.Expr]
NEXT_AVAILABLE_DATA_ID = 0

def get_next_data_id():
    global NEXT_AVAILABLE_DATA_ID
    next_id = NEXT_AVAILABLE_DATA_ID
    #print(next_id, format(next_id, "X").zfill(7))
    NEXT_AVAILABLE_DATA_ID+=1
    #return "D" + format(next_id, "X").zfill(2)
    return str(next_id)




def create_symtype(cls, pytype, shape_env, arg,source = None, duck=True):
    from torch._dynamo.source import ConstantSource
    if source == None:
        source = ConstantSource(f"{len(shape_env.var_to_val)}")
        # source = ConstantSource(f"{arg.id}")


    symbol = shape_env.create_symbol(
        arg,
        source=source,
        dynamic_dim=DimDynamic.DUCK if duck else DimDynamic.DYNAMIC,
        constraint_dim=None,
    )
    return cls(
        SymNode(
            symbol,
            shape_env,
            pytype,
            hint=arg,
        )
    )

def create_symint(shape_env, i: int, duck=True):
    return create_symtype(SymInt, int, shape_env, i, duck=duck)

def create_symbool(shape_env, b: bool):
    return create_symtype(SymBool, bool, shape_env, b)

def create_symfloat(shape_env, f: float):
    return create_symtype(SymFloat, float, shape_env, f)


def is_symbolic(
    val: Union[int, SymInt, float, SymFloat, bool, SymBool]
) -> TypeGuard[Union[SymInt, SymFloat, SymBool]]:
    if isinstance(val, (int, float, bool)):
        return False
    return val.node.is_symbolic()

map_symnode_to_type = {
    int:SymInt,
    float:SymFloat,
    bool:SymBool,
    
}

NEXT_AVAILABLE_ENV_ID = 0

def get_next_env_id():
    global NEXT_AVAILABLE_ENV_ID
    next_id = NEXT_AVAILABLE_ENV_ID
    #print(next_id, format(next_id, "X").zfill(7))
    NEXT_AVAILABLE_ENV_ID+=1
    #return "D" + format(next_id, "X").zfill(2)
    return str(next_id)

class ShapeEnv(TorchShapeEnv):
    def __init__(self, **kwargs):
        # Initialize the base class
        kwargs.setdefault('specialize_zero_one', True) 
        super().__init__(**kwargs)
        self.custom_constraints = {}
        self.id = 'ENV_'+get_next_env_id()
        from torch.fx.experimental.validator import translation_validation_enabled
        # print(f"translation validation enabled? {translation_validation_enabled()}")
   

    def create_symbolic_int(self, val, source: Source, symbolic_type: str):
        """
        Create a symbolic integer for scalar data types.
        Uses `make_symbol` with appropriate symbolic type prefixes for custom types.
        """
        sym_type = {
            "bit": fresh_bit,
            "qbit": fresh_qbit
        }.get(symbolic_type, fresh_int)
        
        symbolic_expr = sympy.Symbol(sym_type())
        # logger.debug(f"Creating symbolic integer {symbolic_expr} for value {val}, source {source}")
        # print(self._create_symbol_for_source(source))
        return self.create_symintnode(symbolic_expr, hint=val, source=source)
    
    def _create_symbol_for_source(self, source: Source) -> Optional[sympy.Symbol]:

        srcname = source.name()
        if source not in self.source_to_symbol:
            self.source_to_symbol[srcname] = sympy.Symbol(srcname, integer=True)
        return self.source_to_symbol[srcname]
    
    def create_symbolic_size(self, val,dtype, source: Source, dynamic_dim=DimDynamic.STATIC):
        """Creates a symbolic size value with a specified source and dynamic dimension"""
        # ex_size = tuple(self._maybe_specialize_sym_int_with_hint(sz) for sz in data.size())
        # logger.debug(f"Attempting to create symbol for {val}, source= {source}, dtype={dtype}")
        return create_symtype(map_symnode_to_type[dtype], dtype, self, arg=val, source=source, duck=True)
    def create_symbolic_sizes(
        self,
        shape: Sequence[int],
        source: Source,
        constraint_sizes: Optional[SymbolicContext] = None,
        symbolic_context: Optional[SymbolicContext] = None,
    ) -> List[sympy.Expr]:
        # print(self)
        from torch._dynamo.source import TensorPropertySource, TensorProperty

        if isinstance(shape, tuple):
            ex_data = torch.empty(*shape, device="meta")
        else:
            ex_data = shape

        ex_size = tuple(self._maybe_specialize_sym_int_with_hint(sz) for sz in ex_data.size())

        # Validate and synchronize environments for symbolic integers
        for sz in ex_size:
            if isinstance(sz, SymInt) and sz.node.shape_env is not self:
                try:
                    sz.node.shape_env.check_equal(self)
                except AssertionError as e:
                    raise ValueError(
                        f"Mismatch between shape environments: {sz.node.shape_env} vs {self}. Details: {e}"
                    )

        # logger.debug(f"- ShapeEnv.create_symbolic_sizes - ex_size create: {ex_size} from the ex_data {ex_data}")
        
        if symbolic_context is None:
            dim = len(ex_size)
            dynamic_dims = [DimDynamic.DUCK] * dim
            constraint_sizes = constraint_sizes if constraint_sizes else [None] * dim
            symbolic_context = StatelessSymbolicContext(
                dynamic_sizes=dynamic_dims,
                constraint_sizes=constraint_sizes,
            )
        assert len(ex_size) == len(dynamic_dims), "Shape and dynamic dims length mismatch"

        size: List[sympy.Expr] = self._produce_dyn_sizes_from_int_tuple(ex_size, source, symbolic_context)
        sym_sizes = [
            self.create_symintnode(
                sym,
                hint=hint,
                source=TensorPropertySource(source, TensorProperty.SIZE, i),
            )
            for i, (sym, hint) in enumerate(zip(size, ex_size))
        ]

        return tuple(sym_sizes)
    
    def create_symbolic_strides(self, strides: Sequence[int], source: Source) -> List[sympy.Expr]:
        """Generates symbolic strides"""
        from torch._dynamo.source import TensorPropertySource, TensorProperty
        
        return tuple(self.create_symintnode(
            sympy.Integer(stride),
            hint=stride,
            source=TensorPropertySource(source, TensorProperty.STRIDE, i)
        ) for i, stride in enumerate(strides))
    
    def create_symbolic_storage_offset(self, offset: int, source: Source) -> sympy.Expr:
        """Generates symbolic storage offset"""
        from torch._dynamo.source import TensorPropertySource, TensorProperty
        
        return self.create_symintnode(
            self.create_symbol(
                offset,
                TensorPropertySource(source, TensorProperty.STORAGE_OFFSET),
                dynamic_dim=DimDynamic.DUCK,
                constraint_dim=None,
            ),
            hint=offset,
            source=TensorPropertySource(source, TensorProperty.STORAGE_OFFSET)
        )
   
    @record_shapeenv_event()
    def create_symintnode(
            self,
            sym: "sympy.Expr",
            *,
            hint: Optional[int],
            source: Optional[Source] = None,
    ):
        """Create a SymInt value from a symbolic expression

        If you know what the current hint value of the SymInt to be created
        is, pass it into hint.  Otherwise, pass None and we will make our best
        guess

        """
        source_name = source.name() if source else None
        # logger.debug(f" source: {source}, name: {source_name}")
        if self._translation_validation_enabled and source is not None:
            # Create a new symbol for this source.
            symbol = self._create_symbol_for_source(source)
            assert symbol is not None

            # Create a new FX placeholder and Z3 variable for 'symbol'.
            fx_node = self._create_fx_placeholder_and_z3var(symbol, int)

            # Add an equality assertion for the newly created symbol and 'sym'.
            self._add_assertion(sympy.Eq(symbol, sym))
        else:
            fx_node = None
        out: Union[int, SymInt]
        if isinstance(sym, sympy.Integer):
            if hint is not None:
                assert int(sym) == hint
            out = int(sym)
        else:
            # How can this occur? When we mark_unbacked, we end up with a real
            # tensor that has hints for all sizes, but we MUST NOT create a
            # SymNode with a hint, because we're hiding the hint from our eyes
            # with the unbacked Symbol.  And in fact, the hint compute may be
            # inconsistent with size oblivious tests.
            if free_unbacked_symbols(sym):
                hint = None
            out = SymInt(SymNode(sym, self, int, hint, fx_node=fx_node))
        return out
    def add_constraint(self, symbol: Symbol, constraint: sympy.Expr):
        """Add a custom constraint to the environment """
        self.custom_constraints[symbol] = constraint

    def enforce_custom_constraints(self):
        """Validate all custom constraints"""
        for symbol, constraint in self.custom_constraints.items():
            if not constraint.subs(self.sym_to_val):
                raise ValueError(f"Constraint violated for symbol {symbol}: {constraint}")

    def __repr__(self):
        return f"{self.id}"



def constrain_unify(a: torch.SymInt, b: torch.SymInt) -> None:

    if not isinstance(a, SymInt):
        if not isinstance(b, SymInt):
            assert a == b
            return
        else:
            shape_env = b.node.shape_env
    else:
        shape_env = a.node.shape_env

    shape_env._constrain_unify(a, b)



@dataclass
class FreshSupply:
    prefix: str
    fresh: int = 0

    def __call__(self):
        r = f"{self.prefix}{self.fresh}"
        self.fresh += 1
        return r


fresh_var = FreshSupply("v")
fresh_int = FreshSupply("i")
fresh_bit = FreshSupply("b")
fresh_qbit =FreshSupply("q")
fresh_size = FreshSupply("s")


class Flow(enum.Flag):
    """
    Denotes LEFT, RIGHT or THRU data.

    LEFT data serve as input lines (input data, that only goes in not out of the task ex. discard data) 
    to the Process. RIGHT data objs are output lines (date is generated within the Node so 
    there's only an edge out) from the Process. THRU are both input and output data i.e. flows in, flows out.

    LEFT and RIGHT Data imply data at the 'entering' or 'exiting' a Task/Operation 
    
    """
    LEFT = enum.auto()
    RIGHT = enum.auto()
    THRU = LEFT | RIGHT


def canonicalize_dtype(value):
    """
    Convert legacy or ambiguous dtype specifications to the canonical in‑house type.
    For example, if value is np.ndarray or a list-of-floats type, return TensorType.
    Otherwise, return value unmodified.
    """
    # If the provided type is a list (or a list of types), pick the first element.
    if isinstance(value, list):
        value = value[0]
    # If the value is the numpy.ndarray type, convert to TensorType.
    if value is np.ndarray:
        return TensorType
    # If the value is list[float] (using PEP 585 notation, for example),
    # you might check by its __origin__ or __name__ if available.
    try:
        # If value has an __origin__ and it is list, then assume TensorType is desired.
        if hasattr(value, "__origin__") and value.__origin__ == list:
            return TensorType
    except Exception:
        pass
    return value

from attrs import field,define

@define
class DataSpec:
    """
    Minimal metadata
    """
    hint: Optional[str] = None
    dtype: Optional[Any] = field(default=None, converter=canonicalize_dtype)
    shape: Optional[tuple] = None
    # elem_dtype: Optional[Any] = None  # New field for element dtype
    def __attrs_post_init__(self):
        if isinstance(self.hint, str):
            self.hint = self.hint.strip()
        # Infer element dtype if not explicitly set
        # if self.dtype and self.elem_dtype is None:
        #     if isinstance(self.dtype, (list, tuple)):
        #         self.elem_dtype = promote_element_types(*self.dtype)
        #     else:
        #         self.elem_dtype = self.dtype
    def is_compatible_with(self, other: "DataSpec") -> bool:
        """
        Compare this (actual) vs. 'other' (expected).
        If 'other.hint' is set, we do usage check; same for data_type, ndim.
        Now also compare 'other.extra' fields => must match exactly.
        """
        # 1) usage check
        if other.hint:
            if isinstance(other.hint, (tuple, list)):
                if self.hint not in other.hint:
                    return False
            else:
                if self.hint != other.hint:
                    return False

        # 2) data_type check
        if other.dtype:
            if isinstance(other.dtype, (tuple, list)):
                if self.dtype not in other.dtype:
                    return False
            else:
                if self.dtype != other.dtype:
                    return False


        return True
def _is_array_like(obj: Any) -> bool:
    return isinstance(obj, (list, np.ndarray, torch.Tensor))
from attrs import define, field


from typing import Protocol
from torch._inductor.virtualized import OpsValue
import functools
from torch._prims_common import ELEMENTWISE_TYPE_PROMOTION_KIND, type_to_dtype


class DTypeVar(Protocol):
    @property
    def dtype(self) -> torch.dtype:
        ...


DTypeArg = Union[DTypeVar, torch.types.Number, str, OpsValue]

@functools.lru_cache(None)
def get_promoted_dtype(
    *args: Sequence[tuple[torch.dtype, bool]],
    type_promotion_kind: Optional[ELEMENTWISE_TYPE_PROMOTION_KIND] = None,
):
    def construct_input(inp):
        if inp[1]:
            return torch.empty([], dtype=inp[0])
        else:
            return torch.empty([1], dtype=inp[0])

    inps = [construct_input(arg) for arg in args]
    _, dtype = torch._prims_common.elementwise_dtypes(
        *inps,
        type_promotion_kind=(
            type_promotion_kind
            if type_promotion_kind
            else ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
        ),
    )
    return dtype


def promote_types(
    args: Sequence[DTypeArg],
    type_promotion_kind: Optional[ELEMENTWISE_TYPE_PROMOTION_KIND] = None,
):
    dtype_prop_candidates = []

    for arg in args:
        assert not isinstance(arg, str)
        if isinstance(arg, OpsValue):
            arg = arg.value
            assert isinstance(arg, torch._prims_common.Number) or hasattr(arg, "dtype")

        if isinstance(arg, torch._prims_common.Number):
            dtype_prop_candidates.append((type_to_dtype(type(arg)), True))
            continue

        dtype_prop_candidates.append((arg.dtype, getattr(arg, "is_scalar", False)))

    dtype = get_promoted_dtype(
        *dtype_prop_candidates,
        type_promotion_kind=type_promotion_kind,
    )

    return dtype
def promote_element_types(*dtypes: Union[type, torch.dtype, np.dtype]) -> Union[type, torch.dtype, np.dtype]:
    """
    Given multiple element dtypes, returns the most precise common dtype.

    - Uses PyTorch & NumPy type promotion where applicable.
    - Falls back to Python's built-in type hierarchy when needed.
    - If only one dtype is provided, returns that dtype directly.
    - When the input types were provided as Python types (e.g. int, float),
      the returned promoted type is converted back to the equivalent Python type.
    
    Examples:
        promote_element_types(int, float) -> float
        promote_element_types(torch.int32, torch.float32) -> torch.float32
        promote_element_types(np.int32, np.float64) -> np.dtype('float64')
        promote_element_types(np.dtype('<U1')) -> str

    """
    dtype_map = {
        int: torch.int32,
        float: torch.float32,
        bool: torch.bool,
        str: str,  # strings remain as str
    }
    # Map torch dtypes back to Python builtins.
    torch_to_py = {
        torch.int32: int,
        torch.int64: int,
        torch.float32: float,
        torch.float64: float,
        torch.bool: bool,
    }
    # First, convert any Python types to their torch equivalents if possible.
    converted_dtypes = [dtype_map.get(dt, dt) for dt in dtypes]

   
    # logger.debug(f"promote_element_types: input dtypes: {dtypes}, converted: {converted_dtypes}")

    if not converted_dtypes:
        raise ValueError("No dtypes provided for promotion.")
    def convert_result(dt):
        if isinstance(dt, torch.dtype):
            return torch_to_py.get(dt, dt)
        return dt
    # If only one dtype is provided, use it directly.
    if len(converted_dtypes) == 1:
        result = converted_dtypes[0]
        # For a single NumPy string dtype, return str.
        if isinstance(result, np.dtype) and result.kind in ['U', 'S']:
            return str
        return convert_result(result)

    # If all are NumPy dtypes, use NumPy promotion.
    elif all(isinstance(dt, np.dtype) for dt in converted_dtypes):
        res = np.result_type(*converted_dtypes)
        # logger.debug(f"res.kind: {res.kind}, res.kind in ['U', 'S']: {res.kind in ['U', 'S']}")
        if res.kind in ['U', 'S']:
            result = str
        if res.kind in ['i', 'u']:
            return int
        if res.kind in ['f']:
            return float
        return res
    # If all inputs are torch dtypes:
    if all(isinstance(dt, torch.dtype) for dt in converted_dtypes):
        result = converted_dtypes[0]
        for dt in converted_dtypes[1:]:
            result = torch.promote_types(result, dt)
        return convert_result(result)
    # Mixed types: use iterative promotion.
    result = converted_dtypes[0]
    for dt in converted_dtypes[1:]:
        # logger.debug(f"Mixed types")
        # If both are torch dtypes:
        if isinstance(result, torch.dtype) and isinstance(dt, torch.dtype):
            result = promote_types(result, dt)
            # result = promote_types([result, dt], type_promotion_kind=None)
        # If both are NumPy dtypes:
        elif isinstance(result, np.dtype) and isinstance(dt, np.dtype):
            result = np.result_type(result, dt)
            if result.kind in ['U', 'S']:
                result = str
        # If one is a Python type from dtype_map, use its mapping.
        elif isinstance(dt, type) and dt in dtype_map:
            result = dtype_map[dt]
        else:
            raise TypeError(f"Unsupported dtype in promotion: {dt}")
    return convert_result(result)
def _extract_element_types(val: Any) -> List[Any]:
    """
    Extracts all possible element types from `val`, handling lists, tensors, matrices, etc.
    """
    if isinstance(val, torch.Tensor):
        return [val.dtype]
    if isinstance(val, np.ndarray):
        return [val.dtype]
    if isinstance(val, (list, tuple)):
        elem_types = set()
        for elem in val:
            elem_types.add(type(elem))
        return list(elem_types)

    return [type(val)]  # Fallback to single type


class Data:
    def __init__(self, 
                data: Any =1, 
                properties: Optional[Dict[str, Any]] = None,
                id: Optional[str] = None,
                shape: Optional[tuple] = tuple(),
                flow=Flow.THRU, 
                shape_env=None, 
                source = None
            ):
        self.data = data
        self.id = id or "d" + get_next_data_id() 
        self.properties = properties or {}
        # If no Data Type provided, do a fallback
        
        dtype_in = self.properties.get("Data Type", None)
        if not dtype_in:
            # TODO: THIS IS WHERE THE MISSING OUTPUT EDGE ERROR IS ARISING. 
            try:
                dtype_in = self._infer_data_type(self.data)
                # self.properties["Data Type"] = dtype_in 
                self.properties["Data Type"] = type(data)
            except TypeError:

                self.properties["Data Type"] = type(data)


        
        self.flow = flow
        self.shape_env = shape_env or ShapeEnv()
        self._shape = shape if shape else self._infer_shape(data)
        self.source = source
        self._symbolic_shape = None

        # logger.debug(f'infered shape: {self._infer_shape(data)}')
        

        hint = self.properties.get("Usage", None)
        if not hint:
            hint = dtype_in
            # hint = self._default_hint(self.data)
        self.metadata = DataSpec(
            hint=hint,
            dtype=dtype_in,
            shape=self._shape
        )
        
        self._initialize_shape_env()

  
    
    def _initialize_shape_env(self):
        """Initialize symbolic shape for the Data object."""
        # logger.debug(f"Initializing {self.shape_env} with {self}")
        if not self.shape_env:
            # logger.warning(f"Data {self.id} has no shape_env. Skipping symbolic init.")
            self._symbolic_shape = ()
            return
        from torch._dynamo.source import ConstantSource
        if self.source is None:
            self.source = ConstantSource(self.id)
        if isinstance(self.data, MatrixType):
            self._meta_shape = self.data.shape
            self._shape = self.data.shape
            self._symbolic_shape = self.shape_env.create_symbolic_sizes(
                self._shape,
                source=self.source
            )
        elif isinstance(self.data, TensorType):
            self._symbolic_shape = self.shape_env.create_symbolic_sizes(self.data.shape)
        elif isinstance(self.data, CAny):
            self._shape = tuple()
            self._symbolic_shape = self.shape_env.create_symbolic_int(
                self.data.num_units,
                source=self.source,
                symbolic_type="bit"
            )
        elif isinstance(self.data, QAny):
            self._symbolic_shape = self.shape_env.create_symbolic_int(self.data.num_units,
                source=self.source,
                symbolic_type="qbit"
            )
            self._shape = tuple()
        else:
            self._shape = tuple()
            self._symbolic_shape = tuple()

    @property
    def bit_length(self):
        return self.data.num_units
    @property
    def bitsize(self) -> int:

        if isinstance(self.data, DataType):
            return self.data.bitsize
        elif isinstance(self.data, int):  # Fallback for integers
            return self.data.bit_length()
        raise AttributeError(f"Cannot determine bitsize for data type: {type(self.data)}")

    @property
    def shape(self):
        return self._shape
    

    @property
    def symbolic_shape(self):
        return self._symbolic_shape
    # def all_idxs(self):
    #     """Generate all index tuples based on the node's shape."""
    #     shape = self.shape
    #     return itertools.product(*[range(int(dim)) for dim in shape])
    
    def __hash__(self):
        """Hash method for Data to allow inclusion in hash-based collections."""
        return hash((self.id, self.shape, self.flow))
    
    def __eq__(self, other):
        # Define equality to allow comparison between Data objects
        if isinstance(other, Data):
            return self.shape == other.shape and self.properties == other.properties
        return False
    def __repr__(self):
        if isinstance(self.symbolic_shape,tuple) and len(self.symbolic_shape)>0:
            return f"Data('{self.id}', {self.shape}, {self.flow})"
        return f"Data('{self.id}',data={self.data}, shape={self.shape} properties={self.properties} {self.flow})"
    def _infer_shape(self, val: Any) -> Tuple[int, ...]:
        if isinstance(val, np.ndarray):
            return val.shape
        if isinstance(val, torch.Tensor):
            return tuple(val.shape)
        if isinstance(val, (list,tuple)):
            return (len(val),)
        if hasattr(self.data, "shape"):
            return self.data.shape
        return ()
    def _infer_data_type(self, val: Any) -> DataType:
        """
        Infer a canonical DataType from the given value.
        If val is a torch.Tensor or np.ndarray, return an instance of TensorType.
        For other types, fallback to int, str, etc.
        """

        # If already a DataType instance, use it.
        if isinstance(val, DataType):
           
            # logger.debug(f"  User input {self.id} is already a DataType instance.")
            return val
        # For torch.Tensor and np.ndarray, instantiate a TensorType with inferred shape and element type.
        # For torch.Tensor, use the tensor's dtype via promote_element_types.
        if isinstance(val, torch.Tensor):
            shape = self._infer_shape(val)
            promoted = promote_element_types(val.dtype)
            inferred = TensorType(shape=shape, element_dtype=promoted, val=val)
            # logger.debug(f"  User input {self.id} -> Inferred TensorType from torch.Tensor: {inferred}")
            return inferred
        
        # For np.ndarray, extract the element types and promote them.
        if isinstance(val, np.ndarray):
            shape = self._infer_shape(val)
            elem_types = _extract_element_types(val)  # This returns a list (e.g. [val.dtype])
            promoted = promote_element_types(*elem_types)
            
            inferred = TensorType(shape=shape, element_dtype=promoted, val=val) 
            # logger.debug(f"  User input {self.id} -> Inferred TensorType from np.ndarray: {inferred}")
            return inferred
        # For lists/tuples, you may want to do something similar (this depends on your design)
        if isinstance(val, (list, tuple)):
            # logger.debug(f'{val} is a list or tuple')
            if isinstance(val,list):
                def ret_type(x):
                    return list[x]
            else:
                def ret_type(x):
                    return tuple[x, ...]
                 
            if len(val) == 0:
                return ret_type(Any)
            # base_type = val[0]
            base_type = type(val[0])
            for elem in val:
                if issubclass(type(elem), base_type):
                    continue
                elif issubclass(base_type, type(elem)):
                    base_type = type(elem)

                else:
                    return ret_type(Any)
            return ret_type(base_type)
        if isinstance(val, int):
            return int
        if isinstance(val, str):
            return str
        if isinstance(val, float):
            return float
        raise TypeError(f"Cannot infer valid data type from input data. val: {val}")
    def __repr__(self):
        try:
            return f'Data(`{self.id}`, usage: {self.properties["Usage"]}, meta: {self.metadata})'
  
        except KeyError:
            return f'Data(`{self.id}`, usage: None , meta: {self.metadata})'



def find_symbol_binding_data(data_list: Iterable[Data]):
    """
    Find and map unique symbolic shapes from the list of Data objects
    into their primary symbols within the ShapeEnv.
    """
    symbol_to_data = {}
    for data in data_list:
        for sym in data.symbolic_shape:
            if isinstance(sym, SymInt) and isinstance(sym.node.expr, sympy.Symbol):
                if sym.node.expr not in symbol_to_data:
                    symbol_to_data[sym.node.expr] = data
    return symbol_to_data

class Signature:
    """An ordered collection of input, output, and thru data registers for a Process."""

    def __init__(self, registers: Iterable[Data]):
        registers = list(registers)
        formatted_registers = ",\n    ".join([f"{data}" for data in registers])
        logger.debug("Initializing Signature with registers:\n    [%s]", formatted_registers)
        
        self._registers = tuple(registers)
        # logger.debug(f"Signature registers: {self._registers}")
        self._lefts = self._dedupe(
            (data.id, data) for data in self._registers if data.flow & Flow.LEFT
        )
       
        
        self._rights = self._dedupe(
            (data.id, data) for data in self._registers if data.flow & Flow.RIGHT
        )

        # TODO: The current unification logic using `_constrain_unify` assumes
        # all SymInt objects reference the same ShapeEnv. The error occurs because
        # `SymInt` instances from different ShapeEnvs can't be directly unified without 
        # deeper handling of inter-ShapeEnv constraints. Likely solution: 
        # Implement an intermediary constraint propagation method that 
        # synchronizes across environments or creates a shared environment.
        # self._unify_shape_envs() 
        

    @staticmethod
    def _dedupe(kv_iter: Iterable[Tuple[str, Dict]]) -> Dict[str, Dict]:
        """Construct a dictionary, but check that there are no duplicate keys."""
        d = {}
        for k, v in kv_iter:
            if k in d:
                # Generate a unique key by appending a number or UUID
                unique_key = f"{k}_{len(d)+1}"
                d[unique_key] = v
                logger.warning(f"Data {k} is specified more than once per side. Renaming to {unique_key}.")
            else:
                d[k] = v
            #logger.debug("Register %s added to the dictionary.", k)
        return d
    
    @classmethod
    def build(cls, *args, **kwargs) -> 'Signature':
        """Construct a Signature from either keyword arguments or Data objects."""
        if args and kwargs:
            raise TypeError("Cannot mix positional and keyword arguments in Signature.build")

        if args:
            # Assume args is an iterable of Data
            if len(args) != 1 or not isinstance(args[0], (list, tuple, set)):
                raise TypeError("Positional arguments should be a single iterable of Data objects")
            return cls(args[0])

        # Keyword mode
        shape_env = ShapeEnv(duck_shape=False)
        return cls(
            Data(id=k, data=CBit()) if v == 1 else Data(id=k, data=v, shape_env=shape_env)
            for k, v in kwargs.items()
            if v is not None
        )

    @classmethod
    def build_from_properties(cls, input_props, output_props) -> 'Signature':
        """Build a Signature based on a Process's input and output properties.
        
        
        """
        output_props = output_props or []  # Safely handle None
        
        registers = []
        logger.debug(f"***** Building from props")
        for i, prop in enumerate(input_props):
            logger.debug(f"input {i}: {prop}")
            # Should all be through
            # registers.append(Data(prop["Data Type"], properties=prop, id = f"{prop["Usage"]+get_next_data_id()}"))
            registers.append(Data(prop["Data Type"], properties=prop))
            # logger.debug("  - Added LEFT register: %s", registers[-1])
        
        for i, prop in enumerate(output_props):
            
            # registers.append(Data(prop["Data Type"],properties=prop, id = f"{str(prop["Usage"])+get_next_data_id()}", flow = Flow.RIGHT))
            registers.append(Data(prop["Data Type"],properties=prop, flow = Flow.RIGHT))
            # logger.debug("  - Added RIGHT register: %s", registers[-1])

        signature = cls(registers)
        # logger.debug("      - Signature built successfully: %s", signature)
        return signature
    
    @classmethod
    def build_from_data(cls, inputs, output_props=None) -> "Signature":
        """
        Build a Signature based on a Process's input data and output properties (if provided).

        - Input data objects are marked as `Flow.LEFT` or `Flow.THRU` depending on whether they are
        reused in the output.
        - Outputs not present in the inputs are created as new `Data` objects with `Flow.RIGHT`.

        Args:
            inputs (list[Data]): Input data objects.
            output_props (list[dict], optional): List of output properties dictionaries.
                Each dictionary specifies properties for an output `Data` object.

        Returns:
            Signature: A Signature object representing the data flow.
        """
        registers = []

        # Step 1: Handle input data (Flow.LEFT or Flow.THRU)
        input_usage_map = {data.properties.get("Usage"): data for data in inputs}
        for data_in in inputs:
            registers.append(data_in)  # Add inputs with default flow (LEFT)

        # Step 2: Process outputs based on output properties
        if output_props:
            for i, prop in enumerate(output_props):
                usage = prop.get("Usage")

                # If an input's "Usage" matches an output's "Usage", mark it as THRU
                if usage in input_usage_map:
                    input_usage_map[usage].flow = Flow.THRU
                else:
                    # Create a new Data object for outputs not in inputs
                    registers.append(
                        Data(
                            data=prop.get("Data Type"),
                            properties=prop,
                            flow=Flow.RIGHT,
                        )
                    )

        # Build the Signature with updated registers
        signature = cls(registers)
        # logger.debug("Signature built successfully: %s", signature)
        return signature

   
    def _unify_shape_envs(self):
        """TODO: Unify ShapeEnv instances among the Data objects in the signature."""
        shape_envs = set(data.shape_env for data in self._registers if hasattr(data, 'shape_env'))
        logger.debug(f"{shape_envs}")
        if len(shape_envs) > 1:
            logger.info(f"Multiple ShapeEnvs detected: {shape_envs}. Unifying them.")
            primary_env = next(iter(shape_envs))
            
            # Map symbols to their primary data
            symbol_to_data = find_symbol_binding_data(self._registers)
            
            for symbol, data in symbol_to_data.items():
                for other_data in self._registers:
                    if other_data.shape_env is not primary_env:
                        for sym1, sym2 in zip(data.symbolic_shape, other_data.symbolic_shape):
                            primary_env._constrain_unify(sym1, sym2)
                    
                        # Move shape_env of the other_data
                        other_data.shape_env = primary_env
                        logger.info(f"Unified Data {other_data.id} into primary ShapeEnv {primary_env.id}.")

    
    def lefts(self):
        """Iterable over all registers that appear on the LEFT as input."""

       
        yield from self._lefts.values()

    def rights(self):
        """Iterable over all registers that appear on the RIGHT as output."""

        yield from self._rights.values()

    def get_left(self, name: str) -> Dict:
        """Get a left register by name."""
        logger.debug("  Fetching LEFT register by name: %s", name)
        return self._lefts[name]

    def get_right(self, name: str) -> Dict:
        """Get a right register by name."""
        logger.debug("  Fetching RIGHT register by name: %s", name)
        return self._rights[name]
    
    def groups(self) -> Iterable[Tuple[str, List[Data]]]:
        """Iterate over data groups by name.

        Data objects with shared names (but differing `.flow` attributes) can be implicitly grouped.
        """
        groups = defaultdict(list)
        for reg in self._registers:
            groups[reg.id].append(reg)

        yield from groups.items()

    

    def __repr__(self):
        return f'Signature({repr(self._registers)})'

    

    def __contains__(self, item):
        return item in self._registers
    @overload
    def __getitem__(self, key: int) -> Data:
        pass

    @overload
    def __getitem__(self, key: slice) -> Tuple[Data, ...]:
        pass
    def __getitem__(self, key):
        return self._registers[key]
    def __iter__(self) -> Iterable[Dict]:
        yield from self._registers

    def __len__(self) -> int:
        return len(self._registers)

    def __eq__(self, other) -> bool:
        return self._registers == other._registers
    



class Result:
    def __init__(
        self,
        network_idx,
        start_time,
        end_time=None,
        device_name=None,
        memory_usage=None,
        qubits_used=None,
        circuit_depth=None,
        success_probability=None,
    ):
        self.network_idx: int = network_idx
        self.start_time: float = start_time
        self.end_time: float = end_time
        self.device_name: str = device_name
        self.memory_usage: float = memory_usage
        self.qubits_used: Tuple[int] = qubits_used
        self.circuit_depth: int = circuit_depth
        self.success_probability: float = success_probability
