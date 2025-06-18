"""

This file contains the class definition for containing
and describing data. The Data class is a container for
arbitrary data and their properties which are utilized,
updated and passed along a problem network.

"""
from typing_extensions import TypeGuard
from typing import List, Dict, Tuple, Iterable, Union, Sequence, Optional,overload, Any,Optional
import itertools

import builtins
import numpy as np
from dataclasses import dataclass
import torch
import sympy
from attrs import field, validators,define
import re

from .symbolic_shape import ShapeEnv,create_symint, create_symfloat, create_symbool,create_symtype, SymInt, SymBool, SymFloat
from .unification_tools import (ALLOWED_BUILTINS, 
                            canonicalize_dtype, 
                            # get_nested_shape_and_numeric, 
                        
                            _extract_element_types,
                            create_data_type_hint
)
from .data_types import *

from ..util.log import get_logger,logging
logger = get_logger(__name__)

SymbolicInt = Union[int, sympy.Expr]
NEXT_AVAILABLE_DATA_ID = 0

def get_next_data_id():
    global NEXT_AVAILABLE_DATA_ID
    next_id = NEXT_AVAILABLE_DATA_ID
    #print(next_id, format(next_id, "X").zfill(7))
    NEXT_AVAILABLE_DATA_ID+=1
    #return "D" + format(next_id, "X").zfill(2)
    return str(NEXT_AVAILABLE_DATA_ID)




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
            # dtype_in = self._infer_data_type(self.data)
            # self.properties["Data Type"] = type(data)
            try:
                dtype_in = self._infer_data_type(self.data)
                # self.properties["Data Type"] = dtype_in 
                self.properties["Data Type"] = type(data)
            except TypeError:

                self.properties["Data Type"] = type(data)


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
                self.data.data_width,
                source=self.source,
                symbolic_type="bit"
            )
        elif isinstance(self.metadata.dtype, QAny):
            self._symbolic_shape = self.shape_env.create_symbolic_int(self.data.data_width,
                source=self.source,
                symbolic_type="qbit"
            )
   
        else:
  
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
    
    # def __hash__(self):
    #     """Hash method for Data to allow inclusion in hash-based collections."""
    #     return hash((self.id, self.shape))
    
    # def __eq__(self, other):
    #     # Define equality to allow comparison between Data objects
    #     if isinstance(other, Data):
    #         return self.shape == other.shape and self.properties == other.properties
    #     return False
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
            inferred = TensorType(shape=shape, element_type=promoted, val=val)
            # logger.debug(f"  User input {self.id} -> Inferred TensorType from torch.Tensor: {inferred}")
            return inferred
        
        # For np.ndarray, extract the element types and promote them.
        if isinstance(val, np.ndarray):
            shape = self._infer_shape(val)
            elem_types = _extract_element_types(val)  # This returns a list (e.g. [val.dtype])
            promoted = promote_element_types(*elem_types)
            
            inferred = TensorType(shape=shape, element_type=promoted, val=val) 
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
            return f'Data(`{self.id}`, usage: {self.properties["Usage"]})'
  
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