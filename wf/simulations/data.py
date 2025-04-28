
from typing import List, Dict, Tuple, Iterable, Union, Sequence, Optional,overload, Any,Optional
import itertools

import builtins
import numpy as np
from dataclasses import dataclass
import torch
import sympy
from attrs import field, validators,define
import re

from .symbolic_shape import ShapeEnv
from .unification_tools import (ALLOWED_BUILTINS, 
                            canonicalize_dtype, 
                            get_nested_shape_and_numeric, 
                            dim_is_int_or_dyn,
                            shape_is_tuple,
                            _extract_element_types,
                            create_data_type_hint
)
from .dtypes import *
from ...util.log import logging
logger = logging.getLogger(__name__)

NEXT_AVAILABLE_DATA_ID = 0

def get_next_data_id():
    global NEXT_AVAILABLE_DATA_ID
    next_id = NEXT_AVAILABLE_DATA_ID
    #print(next_id, format(next_id, "X").zfill(7))
    NEXT_AVAILABLE_DATA_ID+=1
    #return "D" + format(next_id, "X").zfill(2)
    return str(next_id)


SymbolicInt = Union[int, sympy.Expr]

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

def _sanitize_data_hint(hint_str: str) -> str:
    hint_str = hint_str.strip()  # Remove leading/trailing whitespace first.
    sanitized = re.sub(r"[^0-9a-zA-Z_]", "_", hint_str)
    if not sanitized:
        sanitized = "var"
    if sanitized[0].isdigit():
        sanitized = f"_{sanitized}"
    return sanitized

@define
class DataSpec:
    """
    Minimal metadata for a Data object.

    This class captures the expected properties of the underlying data payload,
    specifically:
      - **Data Type:** The canonical type (e.g. a TensorType instance) that includes
        the data’s inherent multi-dimensional layout. We refer to this layout as the "data shape."
      - **Shape:** The actual shape of the data payload (the "data shape").

    Note: This is distinct from the wire shape used in workflow registers (see RegisterSpec),
    which defines how many parallel wires or ports the data is split into when flowing
    through a process.
    """
    hint: Optional[str] = field(default=None, converter=_sanitize_data_hint)
    dtype: Optional[Any] = field(default=None, converter=canonicalize_dtype)
    shape: Optional[tuple] = field(
        default=tuple(),
        validator=validators.deep_iterable(
            member_validator=dim_is_int_or_dyn,
            iterable_validator=shape_is_tuple
        )
    )
    # elem_dtype: Optional[Any] = None  # New field for element dtype
    def __attrs_post_init__(self):
        if isinstance(self.hint, str):
            self.hint = self.hint.strip()
            # safe_name = hint.replace(" ", "_")
            # self.hint = safe_name
        
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
def create_type_hint(x):
    """
    Produces a type hint for the given argument.

    The :func:`create_type_hint` looks for a type hint compatible with the input argument `x`.

    If `x` is a `list` or `tuple`, it looks for an object in the list whose type is a superclass
    of the rest, and uses that as `base_type` for the `List` or `Tuple` to be returned.
    If no such object is found, it defaults to `List[Any]`.

    If `x` is neither a `list` nor a `tuple`, it returns `x`.
    """
    import warnings, inspect, typing
    from typing import Any, Union, List, Tuple
    

    try:
        if isinstance(x, (list, tuple)):
            # todo(chilli): Figure out the right way for mypy to handle this
            if isinstance(x, list):

                def ret_type(x):
                    return list[x]  # type: ignore[valid-type]

            else:

                def ret_type(x):
                    return tuple[x, ...]  # type: ignore[valid-type]

            if len(x) == 0:
                return ret_type(Any)
            base_type = x[0]
            for t in x:
                if issubclass(t, base_type):
                    continue
                elif issubclass(base_type, t):
                    base_type = t
                else:
                    return ret_type(Any)
            return ret_type(base_type)
    except Exception:
        # We tried to create a type hint for list but failed.
        warnings.warn(
            f"We were not able to successfully create type hint from the type {x}"
        )
    return x

class Data:
    """
    Encapsulates a data payload (e.g. a NumPy array, torch.Tensor, or list) along with its metadata.

    **Key Points:**
      - The **DataSpec** (stored in self.metadata) captures the inherent multi-dimensional layout
        of the payload, i.e. its "data shape". For example, a 2×3 NumPy array has a data shape of (2, 3).
      - This underlying data shape is separate from the "wire shape" defined in a workflow's RegisterSpec.
        When wiring data between processes, you might split the outer dimension of the data (e.g., splitting
        a (2,3) array into 2 wires, each carrying a 1D vector of length 3). In that case, the RegisterSpec's
        shape would be (2,) (the wire shape), while the dtype (e.g. TensorType((3,))) carries the inner data shape.
    """
    def __init__(
        self,
        data: Any,
        properties: Optional[Dict[str, Any]] = None,
        id: Optional[str] = None,
        shape: Optional[tuple] = None,
        shape_env=None,
        source = None
    ):
        self.id = id or "d" + get_next_data_id()  
        self.data = data
        self.properties = properties or {}

        self._shape = shape if shape else self._infer_shape(data)
        # logger.debug(f'{self.id}, data {self.data}\n - infered shape: {self._infer_shape(data)}')
        # If no Data Type provided, do a fallback
        
        dtype_in = self.properties.get("Data Type", None)
        if not dtype_in:
            
            dtype_in = self._infer_data_type(self.data)
            self.properties["Data Type"] = dtype_in
            # logger.debug(f' - infered dtype: {dtype_in}')
        if _is_array_like(data):
            element_types = _extract_element_types(data)
            promoted_elem_dtype = promote_element_types(*element_types) if element_types else None
            canonical_res = canonicalize_dtype(self.data)
            # type_hint_res = create_data_type_hint(self.data)
            logger.debug(f"dtype_in: {dtype_in}. element_types:{element_types}, promoted_elem_dtype: {promoted_elem_dtype}, _shape: {self._shape}")
       
        hint = self.properties.get("Usage", None)
        if not hint:
            hint = self._default_hint(self.data)
        # self.properties["Usage"] = _sanitize_data_hint(hint)
        self.metadata = DataSpec(
            hint=hint,
            dtype=dtype_in,
            shape=self._shape
        )
        # logger.debug(f' - Meta: {self.metadata}')
        
        self.shape_env = shape_env or ShapeEnv()

        self.source = source
        self._symbolic_shape = None
        self._initialize_shape_env()
        # logger.debug(f"Created Data('{self.id}', meta={self.metadata})")
    
    @property
    def shape(self):
        return self._shape
    @property
    def sym_shape(self):
        return self._symbolic_shape
    @property
    def ndim(self):
        return len(self.shape)
    def _infer_data_type(self, val: Any) -> DataType:
        """
        Infer a canonical DataType from the given value.
        
        This function infers the type of the input data and returns a canonical DataType object.
        It uses the following ordering:
        
        1. If val is already an instance of DataType, return it directly.
        2. If val is a bare type (e.g. int, float, str) and is one of ALLOWED_BUILTINS, return it.
        3. If val is a torch.Tensor, create and return a TensorType instance using its shape and
        promoted dtype.
        4. If val is a numpy.ndarray, similarly return a TensorType instance with an inferred element dtype.
        5. If val is a list or tuple, attempt to infer a type annotation for its elements.
        6. If val is an instance of int, float, or str (i.e. a scalar value), return the corresponding type.
        7. Otherwise, raise a TypeError.
        
        This ordering ensures that bare types (which are critical for expected input properties)
        are handled before any conversion that might otherwise misinterpret them.
        """
        # 1. If already a DataType instance, return it.
        if isinstance(val, DataType):
            return val

        # 2. If val is a bare type (e.g. int, float, str), return it if allowed.
        if isinstance(val, type):
            if val in ALLOWED_BUILTINS:
                return val
            else:
                raise TypeError(f"Unsupported bare type: {val}")

        # 3. If val is a torch.Tensor.
        if isinstance(val, torch.Tensor):
            shape = self._infer_shape(val)
            promoted = promote_element_types(val.dtype)
            return TensorType(shape=shape, element_type=promoted, val=val)

        # 4. If val is a numpy.ndarray.
        if isinstance(val, np.ndarray):
            shape = self._infer_shape(val)
            if val.dtype.kind in ('U', 'S'):
                elem_dt = str
            elif val.dtype.kind in ('i',):
                elem_dt = int
            elif val.dtype.kind in ('f',):
                elem_dt = np.float32 if val.dtype == np.float32 else np.float64
            else:
                elem_dt = float
            return TensorType(shape=shape, element_type=elem_dt)

        # 5. If val is a list or tuple.
        if isinstance(val, (list, tuple)):
            if len(val) == 0:
                # Return a generic type: list[Any] or tuple[Any, ...]
                return list[Any] if isinstance(val, list) else tuple[Any, ...]
            base_type = type(val[0])
            for elem in val:
                if not isinstance(elem, base_type):
                    base_type = object
                    break
            return list[base_type] if isinstance(val, list) else tuple[base_type, ...]
        
        # 6. If val is a scalar int, float, or str.
        if isinstance(val, int):
            return int
        if isinstance(val, float):
            return float
        if isinstance(val, str):
            return str

        # 7. Otherwise, we cannot infer the data type.
        raise TypeError(f"Cannot infer valid data type from input data. val: {val}")
    def _infer_shape(self, val: Any) -> Tuple[int, ...]:
        if isinstance(val, np.ndarray):
            return val.shape
        if isinstance(val, torch.Tensor):


            return tuple(val.shape)
        if isinstance(val, (list,tuple)):
            shape,is_numeric = get_nested_shape_and_numeric(val)
            logger.debug(f"shape: {shape}, is_numeric: {is_numeric}")
            return (len(val),)
        if hasattr(self.data, "shape"):
            return self.data.shape
        return ()
  
    def _default_hint(self, val: Any) -> str:
        """Heuristic to guess usage if user didn't supply it."""
       
        if isinstance(val, (torch.Tensor,np.ndarray)):
            return "tensor"
        if isinstance(val, int):
            return "int"
        if isinstance(val, str):
            return "str"
        if isinstance(val, tuple):
            return "tuple"
        if isinstance(val, list):
            return "list"
        return "any"
    def _initialize_shape_env(self):
        """Initialize symbolic shape for the Data object."""
        # logger.debug(f"Initializing {self.shape_env} with {self}")
        if not self.shape_env:
            logger.warning(f"Data {self.id} has no shape_env. Skipping symbolic init.")
            self._symbolic_shape = ()
            return
        from torch._dynamo.source import ConstantSource
        if self.source is None:
            self.source = ConstantSource(self.id)
        if isinstance(self.metadata.dtype, MatrixType):
            # self._meta_shape = self.metadata.shape
           
            self._symbolic_shape = self.shape_env.create_symbolic_sizes(
                self._shape,
                source=self.source
            )
        elif isinstance(self.metadata.dtype, TensorType):
            # self._meta_shape = self.metadata.shape
            self._symbolic_shape = self.shape_env.create_symbolic_sizes(self.data.shape,source=self.source)
        elif isinstance(self.metadata.dtype, CUInt):
       
            self._symbolic_shape = self.shape_env.create_symbolic_int(
                self.data.num_units,
                source=self.source,
                symbolic_type="bit"
            )
        elif isinstance(self.metadata.dtype, QAny):
            self._symbolic_shape = self.shape_env.create_symbolic_int(self.data.num_units,
                source=self.source,
                symbolic_type="qbit"
            )
   
        else:
  
            self._symbolic_shape = tuple()

    @property
    def bit_length(self):
        """
        The Data objs 'number of units'

        For example, if the data is a single scalar value, then bit_length is 1. If
        the data represents a vector (or wire) of sub-elements, then bit_length 
        should be equal to the number of sub-elements.
        """
        return self.metadata.dtype.num_units
    @property
    def bitsize(self) -> int:
        """
        The total number of bits used for one logical element of data, i.e., the intrinsic bit-width
        of one sub-element multiplied by the bit_length. That is, if one sub-element of a data object
        is B bits and there are U sub-elemements per "element," then bitsize = B x U
        """

        if isinstance(self.metadata.dtype, DataType):
            # tmp_instance = self.metadata.dtype
            # print(f'{self} bitlength {self.bit_length}, {builtins.int(self).bit_length()} returning {tmp_instance} bitsize ({tmp_instance.bitsize}) versus {self.metadata.dtype.bitsize}')
            return self.metadata.dtype.num_elements
        elif isinstance(self.metadata.dtype, int):  # Fallback for integers
            # tmp_instance = self.metadata.dtype(self._shape, element_type=float)
            # print(f'{self} returning {tmp_instance} bitsize ({tmp_instance.bitsize}) versus {self.metadata.dtype.bitsize}')
            return self.metadata.dtype.bit_length()
        raise AttributeError(f"Cannot determine bitsize for {self}")
   
    def total_bits(self):
        """
        Return the total number of bits based on the Data's type.
        Delegates to metadata.dtype.total_bits() if available; otherwise, if raw data is a torch.Tensor,
        computes the bits from the tensor; else raises an error.
        """
        if hasattr(self.metadata.dtype, "total_bits"):
            return self.metadata.dtype.total_bits
        
        if isinstance(self.data, torch.Tensor):
            return self.data.nelement() * self.data.element_size() * 8
        raise AttributeError(f"Data object with meta {self.metadata} ({self.metadata.dtype.num_elements}) does not support total_bits")
    def all_idxs(self):
        """Generate all index tuples based on the node's shape."""
        if hasattr(self.metadata, "all_idxs"):
            return self.metadata.all_idxs()
        shape = self.shape
        return itertools.product(*[range(int(dim)) for dim in shape])
    # def all_idxs(self):
    #     """
    #     Generate all index tuples based on the shape.
    #     Delegates to metadata.dtype.all_idxs() if available.
    #     """
    #     if hasattr(self.metadata, "all_idxs"):
    #         return self.metadata.all_idxs()
    #     raise AttributeError("Data object does not support all_idxs")
    def __repr__(self):
        try:
            return f'Data(`{self.id}`, usage: {self.properties["Usage"]}, meta: {self.metadata})'
  
        except KeyError:
            return f'Data(`{self.id}`, usage: None , meta: {self.metadata})'

    def __str__(self):
        try:
            return f'Data(`{self.id}`, usage: {self.properties["Usage"]}, meta: {self.metadata})'
  
        except KeyError:
            return f'Data(`{self.id}`, usage: None , meta: {self.metadata})'

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
