from attrs import field, validators,define, Factory
import types
from typing import Any, Dict,List, Dict, Tuple, Iterable, Union, Union, Optional,overload,cast
from pprint import pformat
import numpy as np
import itertools
import enum
import re
import sympy
from collections import defaultdict
from torch import SymInt, SymBool, SymFloat
from .data_types import *

from .utilities import InitError
from ...util.log import get_logger,logging
logger = get_logger(__name__)


def get_flow_spec(flow):
    if flow == Flow.LEFT:
        return 'InSpec'
    if flow == Flow.THRU:
        return 'ThruSpec'
    if flow == Flow.RIGHT:
        return 'OutSpec'

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


DEBUG_INIT_LOGS = False  # Set to True to enable __attrs_post_init__ debug output.

def _sanitize_name(name: str) -> str:
    """
    Convert 'Pivot Idx' -> 'Pivot_Idx', remove invalid Python chars,
    and ensure it doesn't start with a digit. 
    This ensures we never store a RegisterSpec name that breaks 
    Python's identifier logic or has spaces.
    """
    # Replace invalid chars with underscores
    sanitized = re.sub(r"[^0-9a-zA-Z_]", "_", name)
    if not sanitized:
        sanitized = "var"
    # If it starts with digit, prepend underscore
    if sanitized[0].isdigit():
        sanitized = f"_{sanitized}"
    # return sanitized.lower()
    return sanitized
import inspect
ALLOWED_BUILTINS = [int, float, str, bool, list, tuple, set]

def _compute_domain_from_dtype(dtype: Any) -> str:
    """
    Returns "Q" if `dtype` or its element_type represents a quantum type, otherwise "C".
    This peels off composite wrappers (TensorType, MatrixType, etc.).
    """
    # 1) If it's a DataType instance, dive into element_type if present
    if isinstance(dtype, DataType):
        if hasattr(dtype, "element_type"):
            return _compute_domain_from_dtype(dtype.element_type)
        # classical DataTypes subclass CType
        if isinstance(dtype, CType):
            return "C"
        # everything else (QBit, QInt, QUInt, QAny, …) is quantum
        return "Q"

    # 2) If it's a DataType class, same logic
    if inspect.isclass(dtype) and issubclass(dtype, DataType):
        if issubclass(dtype, CType):
            return "C"
        return "Q"

    # 3) Builtins, GenericAlias, etc. → classical
    return "C"


def _compute_domain(self) -> str:
    """
    Attrs Factory hook: inspects this RegisterSpec's own .dtype
    and delegates into _compute_domain_from_dtype.
    """
    return _compute_domain_from_dtype(self.dtype)

def _compute_is_symbolic(dtype: Any) -> bool:
    """
    Delegates to dtype.is_symbolic() when dtype is a DataType,
    else returns False for builtins / GenericAlias.
    """
    from .data_types import is_symbolic as _dtype_is_symbolic
    if isinstance(dtype, DataType):
        return _dtype_is_symbolic(dtype)
    return False
def dim_is_int_or_dyn(inst, attribute, dim):
    """
    Custom member validator to check that each dimension is either an int or Dyn.
    """
    if not (isinstance(dim, int) or dim == Dyn):
        raise ValueError(f"Invalid shape dimension: {dim}. Must be an int or Dyn.")

def shape_is_tuple(inst, attribute, value):
    """
    Ensures the shape is a tuple.
    """
    if not isinstance(value, tuple):
        raise TypeError(f"Expected tuple for shape, got {type(value)}")

def canonicalize_dtype(value):
    """
    Convert legacy or ambiguous dtype specifications to the canonical in‑house type.

    For example, if 'value' is a NumPy array or a torch tensor, this function returns a TensorType
    instance with the inferred shape and element_type determined from the value's dtype.
    
    In particular:
      - For a np.ndarray:
          * If value.dtype.kind is 'U' or 'S' (string types), element_type is set to str.
          * If value.dtype.kind is 'i' (integer), element_type is set to int.
          * If value.dtype.kind is 'f' (floating point), element_type is set to np.float32 for float32,
            or np.float64 otherwise.
      - For a torch.Tensor, similar logic applies based on the tensor’s dtype.

    Otherwise, the value is returned unmodified.
    """
    import numpy as np
    import torch
    
    if isinstance(value, DataType):
        return value

    if isinstance(value, types.GenericAlias):
        logger.debug(f" -> value is a GenericAlias: {value}")
        return value

    if isinstance(value, (list, tuple)):
        logger.debug(f"{value} -> value is a (list, tuple): {value}")
        shape, is_numeric = get_nested_shape_and_numeric(value)
        if is_numeric:
            return TensorType(shape)
    


    # Corrected: use a tuple of types instead of a list.
    if isinstance(value, (np.ndarray, torch.Tensor)):
        
        if isinstance(value, np.ndarray):
            if value.dtype.kind in ('U', 'S'):
                elem_dt = str
            elif value.dtype.kind in ('i',):
                elem_dt = int
            elif value.dtype.kind in ('f',):
                elem_dt = np.float32 if value.dtype == np.float32 else np.float64
            else:
                elem_dt = float
            logger.debug(f'elem_dt: {elem_dt}')
            return TensorType(shape=value.shape, element_type=elem_dt)
        else:  # torch.Tensor branch
            if value.dtype in (torch.int32, torch.int64):
                elem_dt = int
            elif value.dtype == torch.float32:
                elem_dt = np.float32
            elif value.dtype == torch.float64:
                elem_dt = np.float64
            else:
                elem_dt = float
            logger.debug(f'elem_dt: {elem_dt}')
            return TensorType(shape=tuple(value.shape), element_type=elem_dt)
    if not isinstance(value, type) and hasattr(value, "shape"):
        shape_val = getattr(value, "shape")
        if callable(shape_val):
            shape_val = shape_val()
        try:
            shape_tuple = tuple(int(x) for x in shape_val)
        except Exception as e:
            raise ValueError(f"Error converting shape from object {value}: {e}")
        return TensorType(shape_tuple)

    if value is np.ndarray:
        return TensorType
    if value is torch.Tensor:
        return TensorType

    try:
        if hasattr(value, "__origin__") and value.__origin__ == list:
            return TensorType
    except Exception:
        pass

    return value

def get_nested_shape_and_numeric(val):
    """
    Recursively compute the shape of a nested list/tuple and whether its
    innermost elements are numeric (int/float/NumPy numeric).
    
    Returns:
      (shape_tuple, is_numeric)
        shape_tuple: e.g. (2, 3) if val is [[1, 2, 3], [4, 5, 6]]
        is_numeric: True if all leaves are numeric.
    """
    import numpy as np

    # Base case: if not list/tuple, shape=(), is_numeric=whether val is int/float
    if not isinstance(val, (list, tuple)):
        return (), isinstance(val, (int, float, np.integer, np.floating))
    if not val:
        # e.g. an empty list => shape (0,) is_numeric=True (arbitrary choice)
        return (), True

    # Recursively get shape/is_numeric from the first element
    subshape, is_subnumeric = get_nested_shape_and_numeric(val[0])

    # Then we must confirm each subsequent element has the same shape & numeric status
    # for a strict "rectangular" interpretation:
    for item in val[1:]:
        s, is_num = get_nested_shape_and_numeric(item)
        if s != subshape or not is_num:
            return (), False
    return (len(val),) + subshape, is_subnumeric


def get_shape(v) -> Tuple[Any, ...]:
    """
    Converter for the _shape attribute.
    - If v is one of the ALLOWED_BUILTINS, wrap it in a single-element tuple.
    - If v has a "shape" attribute, retrieve it (calling it if callable) and convert
      each dimension to an int unless it equals Dyn.
    - Otherwise, assume v is iterable and convert each element (unless it's Dyn).
    """
    # logger.debug(f"get_shape for {v}")
    if v in ALLOWED_BUILTINS:
        return (v,)
    if hasattr(v, "shape"):
        shape_val = getattr(v, "shape")
        if callable(shape_val):
            shape_val = shape_val()
        out = []
        for x in shape_val:
            if x == Dyn:
                out.append(Dyn)
            elif isinstance(x, (np.integer, int)):
                out.append(int(x))
            else:
                out.append(x)
        return tuple(out)
    try:
        out = []
        for x in v:
            if x == Dyn:
                out.append(Dyn)
            else:
                out.append(int(x))
        return tuple(out)
    except TypeError:
        raise ValueError(f"Cannot split with symbolic data_width: {v}")

@define
class RegisterSpec:
    """
    Defines a workflow register for passing data between processes.

    name: identifier
    dtype:The canonical (or user-specified) type for data on this register.
                For example, a TensorType instance (e.g. TensorType((3,), element_dtype=float))
                that encapsulates the inherent multi-dimensional layout (the "data shape")
                for the data carried by each wire.
    _shape: A tuple representing the "wire shape" (i.e. the number of parallel wires or ports)
                  in the workflow graph. For instance, if you want to split a (2,3) data payload
                  into 2 wires (each carrying a 1D vector of length 3), then _shape would be (2,).
    flow: Flow Indicates whether the register is used for input (LEFT), output (RIGHT), or both (THRU).
    variadic:  Boolean flag indicating whether multiple data objects can be bound to this register.


    __attrs_post_init__() behavior:
        - If `dtype` is a DataType subclass and a non-empty wire shape is provided,
        then `dtype` is instantiated with that shape. This ensures that the dtype
        is concrete (e.g. TensorType((2,2))) for resource estimation.
        - Otherwise, builtin types or GenericAliases are accepted as-is.
    note: The **data shape** is embedded in the `dtype` (e.g. TensorType((3,)) for each wire),
        while the **wire shape** is given by the `_shape` attribute (e.g. (2,) for 2 wires).
    """

    name: str = field(converter=_sanitize_name)
    dtype: Any = field(converter=canonicalize_dtype)
    _shape: Tuple[Any, ...] = field(
        default=(),
        converter=get_shape,
        validator=validators.deep_iterable(
            member_validator=dim_is_int_or_dyn,
            iterable_validator=shape_is_tuple,
        ),
    )
    flow: Flow = field(default=Flow.THRU)
    variadic: bool = field(default=False)
     
    domain: str = field(
        init=False,
        default=Factory(_compute_domain, takes_self=True),
    )
    is_symbolic: bool = field(
        init=False,
        default=Factory(_compute_is_symbolic, takes_self=True),
    )

    
    def __attrs_post_init__(self):
        """
        Post-initialization hook for RegisterSpec.

        Behavior:
        - If dtype is a DataType instance:
            * (Legacy behavior) Previously, if _shape was empty and dtype had a defined shape,
                we would adopt that shape. For our split/join test, we want to *not* do that.
            * Therefore, we deliberately skip adopting dtype.shape if _shape is empty,
                keeping the register "atomic" (i.e. _shape remains ()).
        - If dtype is a DataType class and a non-empty _shape is provided, instantiate it.
        - Otherwise, builtin types or GenericAliases are accepted.
        """
        import inspect
        init_logs = []
        
        if isinstance(self.dtype, DataType):
            # Instead of adopting dtype.shape when _shape is empty, we leave _shape as provided.
            # (This means that if _shape is not explicitly given, it remains (), making the register atomic.)
            init_logs.append(f"Registering DataType INSTANCE without inheriting shape: {repr(self)}")
        elif inspect.isclass(self.dtype) and issubclass(self.dtype, DataType):
            if self.shape and len(self.shape) > 0:
                self.dtype = self.dtype(self.shape)
                init_logs.append(f" -> Instantiated DataType from subclass with shape {self.shape}: {repr(self)}")
            else:
                init_logs.append(f" -> Registering DataType SUBCLASS without shape: {repr(self)}")
        elif isinstance(self.dtype, types.GenericAlias):
            init_logs.append(f" -> Registering GenericAlias type: {repr(self)}")
        elif self.dtype in ALLOWED_BUILTINS:
            init_logs.append(f" -> Registering built-in base type: {repr(self)}")
        else:
            raise TypeError(
                f"name '{self.name}' dtype must be a DataType (instance or subclass), "
                f"a builtin type in ALLOWED_BUILTINS, or a GenericAlias. Got {self.dtype} of type {type(self.dtype)}. Class? {inspect.isclass(self.dtype)}, subclass of DataType? {issubclass(self.dtype, DataType)}"
            )
        
        # Validate that _shape (exposed via self.shape) is a tuple.
        if not isinstance(self.shape, tuple):
            raise TypeError(f"shape attr must be a tuple, got {type(self.shape)}")
        # init_logs.append(f"Initialized RegisterSpec: {self}")
        # logger.debug(f"Initialized RegisterSpec: {repr(self)}")
        if DEBUG_INIT_LOGS:
            for msg in init_logs:
                logger.debug(msg)
    @_shape.validator
    def _check_shape(self, attribute, value):
        """
        Validator for _shape: every element must be an int or a dynamic placeholder.
        """
        for dim in value:
            if not (isinstance(dim, int) or dim == Dyn):
                raise ValueError(f"Invalid shape dimension: {dim}. Must be an int or Dyn.")
    @property
    def symbolic_shape(self) -> Tuple[SymInt, ...]:
        return self._shape
    @property
    def shape(self) -> Tuple[int, ...]:
        # if is_symbolic(*self._shape):
        #     raise ValueError(f"{self} is symbolic. Cannot get real-valued shape.")
        return cast(Tuple[int, ...], self._shape)
    
    @property
    def bitsize(self) -> int:
        """Number of bits (or qubits) per single value of this register's data_type."""
        return self.dtype.data_width
    
    
    def all_idxs(self):
        """Generate all index tuples based on the node's shape."""
        shape = self.shape
        yield from itertools.product(*[range(int(dim)) for dim in shape])
    
    def total_bits(self) -> int | sympy.Basic:
        """
        Total logical bits carried by this register:
            • if dtype defines `.total_bits` (e.g. TensorType, MatrixType)
              use that value;
            • else fall back to per-element width (`self.bitsize`).
        Then multiply by the wire-shape fan-out (self._shape).
        """
        # 1) bits required by *one* data payload on this wire
        payload_bits = (
            self.dtype.total_bits          # TensorType, MatrixType, etc.
            if hasattr(self.dtype, "total_bits")
            else self.bitsize              # scalar dtypes
        )

        # 2) fan-out if this RegisterSpec has multiple parallel wires
        fan_out = prod(self.symbolic_shape or (1,))
        
        return payload_bits * fan_out
    
    def __eq__(self, other):
        if not isinstance(other, RegisterSpec):
            return NotImplemented
        if self.name != other.name or self.flow != other.flow:
            return False
        # unify dtype
        if self._same_dtype(other) and (self._shape == other._shape):
            return True
        return False

    
    def _same_dtype(self, other):
        """
        Compare self.dtype vs. other.dtype using is_consistent_data_type, with
        some extra logic for class vs. instance of DataType.
        """
        import inspect
        from .data_types import TensorType, Dyn

        a, b = self.dtype, other.dtype

        # 1) If both are TensorType → shape‐only compare (Dyn wildcard)
        if isinstance(a, TensorType) and isinstance(b, TensorType):
            s1, s2 = a.shape, b.shape
            if len(s1) != len(s2):
                return False
            for d1, d2 in zip(s1, s2):
                if not (d1 == d2 or d1 == Dyn or d2 == Dyn):
                    return False
            return True

        # 2) If *neither* is a DataType class/instance, just do normal equality
        is_a_dt = isinstance(a, DataType) or (inspect.isclass(a) and issubclass(a, DataType))
        is_b_dt = isinstance(b, DataType) or (inspect.isclass(b) and issubclass(b, DataType))
        if not (is_a_dt or is_b_dt):
            return a == b

        # 3) If both are concrete DataType instances, rely on their __eq__
        if isinstance(a, DataType) and isinstance(b, DataType):
            return a == b

        # 4) Finally fall back to your unification routine
        return check_dtypes_consistent(a, b)
    def __hash__(self):
        # Combine the same fields used in __eq__.
        return hash((self.name, self.dtype, self._shape, self.flow))
  

    def matches_data_list(self, data_objs: List["Data"]) -> bool:
        """
        If variadic=True, we try to match *all* data_objs in a row.
        If variadic=False, we expect exactly one `Data`.
        """
        if not self.variadic:
            # Expect exactly one item:
            if len(data_objs) != 1:
                return False
            return self.matches_data(data_objs[0])

        # else if variadic => each data_obj must match
        for d in data_objs:
            if not self.matches_data(d):
                return False
        return True
    def matches_data(self, data_obj) -> bool:
        """
        Check if data_obj is compatible with this RegisterSpec.
        We compare the normalized (canonicalized) dtypes.
        """
        from .data import canonicalize_dtype  # use the canonicalizer from data.py

        # Retrieve the actual type from the Data's metadata.
        arg_dtype = data_obj.metadata.dtype
        # logger.debug(f'Attempting to match input data {arg_dtype} to expected: {self.dtype}')
        if isinstance(arg_dtype, MatrixType):
            logger.debug(f"Casting MatrixType {arg_dtype} to TensorType for comparison.")
            arg_dtype = TensorType(arg_dtype.shape)
        if isinstance(self.dtype, (list,tuple)) and not isinstance(self.dtype, types.GenericAlias):
            # If *any* item in the list is consistent, we accept it.
            return any(is_consistent_data_type(opt, arg_dtype) for opt in self.dtype)
    
        if isinstance(self.dtype, (list, tuple, set, int, float)):
            if arg_dtype not in self.dtype:
                logger.debug(f'{arg_dtype} not in {self.dtype}')
                return False
        # Normalize both expected and actual types.
        norm_expected = canonicalize_dtype(self.dtype)
        norm_actual = canonicalize_dtype(arg_dtype)
        
        # If either normalized value is not a type (e.g. an instance),
        # then compare their classes.
        if not isinstance(norm_expected, type) and hasattr(norm_expected, "__class__"):
            norm_expected = norm_expected.__class__
        if not isinstance(norm_actual, type) and hasattr(norm_actual, "__class__"):
            norm_actual = norm_actual.__class__
        
        # logger.debug(
        #     f"RegisterSpec.matches_data: for spec '{self.name}', "
        #     f"normalized expected type: {norm_expected}, normalized actual type: {norm_actual}"
        # )
        
        # Now compare the normalized types.
        if norm_expected != norm_actual:
            logger.debug(f"Type mismatch in matches_data: expected {norm_expected}, got {norm_actual}")
            return False

        # Optionally check shape if needed (here commented out to match legacy behavior):
        # if self._shape and data_obj.shape != self._shape:
        #     return False

        return True
   
    def _initialize_shape_env(self):
        """
        If data is e.g. MatrixType, unify self._shape with the shape_env
        by creating symbolic sizes.
        If data is e.g. CAny or QAny, create a symbolic int for the bit/qubit count, etc.
        Otherwise, if it's a raw torch.Tensor, also unify shape with shape_env.
        """
        # logger.debug(f"Initializing {self.shape_env} with {self}")
        if not self.shape_env:
            # logger.warning(f"Data {self.id} has no shape_env. Skipping symbolic init.")
            self._symbolic_shape = ()
            return
        from torch._dynamo.source import ConstantSource

        if self.source is None:
            self.source = ConstantSource(self.id)

        if self.metadata.dtype == MatrixType or self.metadata.dtype == TensorType:
            # We unify the static .shape from data.metadata
            # and create symbolic sizes in shape_env
           
            # Create symbolic representation
            # logger.debug(f'Creating symbolic represetnations of N-dim objects')
            self._symbolic_shape = self.shape_env.create_symbolic_sizes(
                self.shape, source=self.source
            )

        elif isinstance(self.data, (CAny,CBit)):
            # A bag of classical bits
            # shape = empty tuple, but the bitcount is data.data_width
            self._symbolic_shape = self.shape_env.create_symbolic_int(
                self.data.data_width,
                source=self.source,
                symbolic_type="bit"
            )
           

        elif isinstance(self.data, (QAny,QBit)):
            # A bag of qubits
            self._symbolic_shape = self.shape_env.create_symbolic_int(
                self.data.data_width,
                source=self.source,
                symbolic_type="qbit"
            )
       
        else:
            # Fallback: no symbolic shape
            # logger.debug(f'Defaulting symbolic shape to ()')
            self._symbolic_shape = ()

    def __str__(self) -> str:
        return f'{get_flow_spec(self.flow)}(name={self.name}, dtype={self.dtype}, shape={self.shape})'
        

    def __repr__(self) -> str:
        """
        Fallback if someone calls repr():
        """
        """
        Return a concise, human-readable summary, e.g.:
          arg0: TensorType (shape=(3, 3)) [in/out]
        """
        # Decide how to show shape. If empty or None, you could say "unspecified".
        if self.shape:
            shape_str = f"(shape={self.shape})"
        # elif hasattr(self.dtype,DataType) and hasattr(self.dtype,'shape'):
        #     shape_str = f'{self.dtype.shape}'
        else:
            shape_str = ''
        # If dtype is a class, show its __name__; if it's an instance, show the class name.
        if isinstance(self.dtype, type):
            dtype_str = self.dtype.__name__
        else:
            dtype_str = self.dtype


        # Indicate if it’s a variadic spec
        variadic_label = " (variadic)" if self.variadic else ""

        return f"{self.name}: {dtype_str} {shape_str} {self.flow} {variadic_label}"
    

from typing import Union


class Signature:
    """
    Signature is a collection of RegisterSpecs, partitioned into LEFT, RIGHT, and/or THRU flows.
    
    Key functionality:
    - `build(...)`: Construct from either (a) an iterable of `RegisterSpec` objects, or (b) dictionary-like 
      declarations of simple types (like `x=1` => a single CBit).
    - `build_from_properties(...)`: For processes that define only the expected_input_properties and output_properties attributes 
    - `build_from_specs(...)`: For typed `InputSpec` lists (new approach).  - `build_from_data(...)`: Merges or re-flags existing Data objects according to usage or property logic.
    - lefts(), rights(): Return RegisterSpecs that flows in/out, ignoring THRU.
    - get_left(...), get_right(...): Directly access data by ID for quick lookups in code or logs.
    
     [Yellow] – We can reorder lines or comment them out, add any new functions, but do NOT remove any code.

    **What we want**:
    1. **No 'in_0'/'out_0'** style naming in the new approach. Instead, we want "arg_0", "arg_1", etc., 
    letting 'flow' itself show whether it's effectively input or output (or THRU).
    2. We keep the old code commented out—**NOT** deleted—so if we revert, we can restore it easily.
    3. Flow (LEFT, RIGHT, THRU) still matters for typed checks. But if usage is purely “entering,” 
    the user said it might really be THRU in some contexts. We can choose to set everything to THRU 
    except for guaranteed outputs we set to RIGHT, etc.
    """
   
    def __init__(self, registers: Iterable[RegisterSpec]):
        # registers = list(registers)
        # formatted_registers = ",\n    ".join([f"{data}" for data in registers])
        # logger.debug("Initializing Signature with registers:\n    [%s]", formatted_registers)
        
        self._registers = tuple(registers)
        # logger.debug(f"Signature registers: {self._registers}")
        self._lefts = self._dedupe(
            (reg.name, reg) for reg in self._registers if reg.flow & Flow.LEFT
        )
       
        
        self._rights = self._dedupe(
            (reg.name, reg) for reg in self._registers if reg.flow & Flow.RIGHT
        )
        self.usage_map = {}

      
        

    @staticmethod
    def _dedupe(kv_iter: Iterable[Tuple[str, Dict]]) -> Dict[str, Dict]:
        """Construct a dictionary, but check that there are no duplicate keys."""
        d = {}
        for k, v in kv_iter:
            if k in d:
                # raise ValueError(f"RegisterSpec {k} is specified more than once per side.")
                # TODO: Decide whether this function causes issues if we auto rename like below. For now, raise error.
                # # Generate a unique key by appending a number or UUID
                unique_key = f"{k}_{len(d)+1}"
                d[unique_key] = v
                logger.warning(f"RegisterSpec {k} is specified more than once per side. Renaming to {unique_key}.")
            else:
                d[k] = v
            # logger.debug("Register %s added to the dictionary.", k)

        # logger.debug(f'dict: {d}')
        return d
    
    @classmethod
    def build(cls, *args, **registers: Union[int,sympy.Expr]) -> 'Signature':
        """
        Construct a Signature from either keyword arguments or RegisterSpec objects.
        [YELLOW] this is taken directly from Qualtran's Siganture and should remain the same.
        We dont want to wonder too far from their implementation of signature since its part of visualization.
        """
        if args and registers:
            raise TypeError("Cannot mix positional and keyword arguments in Signature.build")

        if args:
            # Assume args is an iterable of Data
            if len(args) != 1 or not isinstance(args[0], (list, tuple, set)):
                raise TypeError("Positional arguments should be a single iterable of RegisterSpec objects")
            logger.debug(f"args: {args}")
            return cls(args[0])
        def convert_value(v):
            if isinstance(v, DataType):
                return v
            # 2) A builtin type -> map to default CType
            if v is int:
                return CUInt(v)
            if v is float:
                return CFloat(v)
            # 3) Literal 1 -> single bit
            if v == 1:
                return CBit()
            # For now, preserve the old logic for numbers.
            return CBit() if v == 1 else CUInt(v)

        return cls(
            RegisterSpec(name=k, dtype=convert_value(v))
            for k, v in registers.items() if v
        )
     

    
    @classmethod
    def build_from_dtypes(cls, **data_types: DataType):
        """
        Takes explicit DataType objects (or something convertible to one) 
        and builds a Signature from them
        """
        # maybe use ALLOWED_BUILTINS for built in dtypes (like int, float,etc.)
        return cls(RegisterSpec(name=k,dtype=v) for k,v in data_types.items() if v in ALLOWED_BUILTINS or v.data_width)
    @classmethod
    def build_from_properties(cls, input_props, output_props) -> 'Signature':
        """
        Build a Signature from input and output property dictionaries.
        This implementation assigns input register names as "arg0", "arg1", ...,
        and for outputs, only registers those outputs whose "Usage" is not already
        present in the input properties. (This prevents duplicating data that simply flows through.)
        For a single new output, it is named "OUT"; if multiple, they are named "arg_out0", "arg_out1", etc.
        
        If a property dictionary's "Data Type" value is a list, the first element is taken as the canonical type.
        If the canonical type is numpy.ndarray (as returned by type(np.array([]))), it is replaced with TensorType.
        The "shape" is taken from the property if provided, or defaults to an empty tuple.
        All input registers are assigned Flow.THRU and all output registers Flow.RIGHT.
        """
        registers = []
        import copy
        import numpy as np
        
        
        logger.debug("Building .signature from Process properties *****")
        
        input_usages = set()
        # Process input properties
        if input_props:
            for i, prop in enumerate(input_props):
                name = f"arg{i}"
                usage = prop.get("Usage")
                if usage is not None:
                    input_usages.add(usage)
                dtype = prop.get("Data Type")
                if isinstance(dtype, list):
                    dtype = dtype[0]
                if dtype is np.ndarray:
                    dtype = TensorType
                shape = copy.deepcopy(prop.get("shape", ()))
                # logger.debug(f"Attempting to regsiter {name} with dtype: {dtype}")
                registers.append(RegisterSpec(name=name, dtype=dtype, shape=shape, flow=Flow.THRU))
        
        # Process output properties: only add outputs for properties whose "Usage" is new.
        if output_props:
            out_props_to_add = []
            for prop in output_props:
                usage = prop.get("Usage")
                # Only add if this usage is not already in the inputs.
                if usage not in input_usages:
                    out_props_to_add.append(prop)
                # else:
                    # logger.debug(f"Skipping output property for Usage '{usage}' because it already exists in inputs.")
            if len(out_props_to_add) == 1:
                name = "OUT"
                prop = out_props_to_add[0]
                dtype = prop.get("Data Type")
                if isinstance(dtype, list):
                    dtype = dtype[0]
                shape = copy.deepcopy(prop.get("shape", ()))
                registers.append(RegisterSpec(name=name, dtype=dtype, shape=shape, flow=Flow.RIGHT))
            elif len(out_props_to_add) > 1:
                for i, prop in enumerate(out_props_to_add):
                    name = f"arg_out{i}"
                    dtype = prop.get("Data Type")
                    if isinstance(dtype, list):
                        dtype = dtype[0]
                    shape = copy.deepcopy(prop.get("shape", ()))
                    registers.append(RegisterSpec(name=name, dtype=dtype, shape=shape, flow=Flow.RIGHT))
        
        return cls(registers)
   
    @classmethod
    def build_from_data(cls, inputs: list["Data"], output_props: list[Dict[str, Any]]) -> "Signature":
        """
        Build a Signature from the given list of Data objects and output properties.
        
        For each input Data object:
        - Use its metadata.hint if available; otherwise, use its 'Usage' property.
        - Sanitize the chosen name using _sanitize_name.
        - Use the Data's metadata.dtype (or fallback to type(data.data)) as dtype.
        - Determine the wire shape using get_shape() on data.metadata.shape if available,
            otherwise default to an empty tuple (scalar).
        - Set the flow to Flow.THRU for input registers.
        
        For output properties:
        - For each dictionary in output_props, process similarly to build a RegisterSpec with flow RIGHT.
        - If an output property’s usage can be unified with an input (i.e. an input register with
            that usage already exists), skip creation of a new RegisterSpec.
        
        Returns:
            A new Signature instance built from the combined input and output RegisterSpec objects.
        """
       
        registers = []
        
        # Build input registers from the provided Data objects.
        for data in inputs:
            # Determine the name: prefer metadata.hint (already sanitized) if available.
            if hasattr(data, "metadata") and data.metadata and getattr(data.metadata, "hint", None):
                name = _sanitize_name(data.metadata.hint)
            else:
                # If a Row Idx is provided, combine it with Usage.
                usage = data.properties.get("Usage")
                if usage is not None and "Row Idx" in data.properties:
                    name = _sanitize_name(f"{usage} {data.properties['Row Idx']}")
                elif usage is not None:
                    name = _sanitize_name(usage)
                else:
                    name = data.id
        
            # Determine the dtype: prefer metadata.dtype if available.
            dtype = data.metadata.dtype if (hasattr(data, "metadata") and data.metadata and hasattr(data.metadata, "dtype")) else None
            if dtype is None:
                dtype = type(data.data)
        
            # Determine the wire shape: if metadata.shape is available, convert it; otherwise, use ().
            if hasattr(data, "metadata") and data.metadata and getattr(data.metadata, "shape", None) is not None:
                shape = get_shape(data.metadata.shape)
            else:
                shape = ()
        
            # Create the RegisterSpec for input (flow THRU).
            reg = RegisterSpec(name=name, dtype=dtype, shape=shape, flow=Flow.THRU)
            registers.append(reg)
        
        # Process output properties if provided.
        if output_props:
            for j, prop in enumerate(output_props):
                usage = prop.get("Usage")
                if isinstance(usage, (tuple, list)):
                    canonical_usage = usage[0] if usage else f"arg_out{j}"
                elif usage:
                    canonical_usage = usage
                else:
                    canonical_usage = f"arg_out{j}"
        
                out_name = _sanitize_name(canonical_usage)
                
                # Determine dtype and shape for output from the property dict.
                dtype = prop.get("Data Type")
                shape = get_shape(prop.get("shape", ()))
                
                # Before creating a new output register, check if an input register with the same usage exists.
                input_usages = [reg.name for reg in registers]
                if out_name in input_usages:
                    continue
                reg = RegisterSpec(name=out_name, dtype=dtype, shape=shape, flow=Flow.RIGHT)
                registers.append(reg)
        
        return cls(registers)
   
    def validate_data_with_register_specs(self, data_objs: List["Data"]) -> bool:
        """
        Now supports variadic RegisterSpec. For each spec in self.lefts(), we:
        - If spec.variadic=False => consume exactly one data from data_objs
        - If spec.variadic=True  => consume the remainder (or partial) of data_objs
        (or we can define a more advanced partitioning logic if we have multiple variadic specs).
        Raises ValueError on mismatch, or returns True if all pass.
        """
        # We only validate "left-flow" registers here, but you can combine left+thru if desired.
        specs = list(self.lefts())
        data_idx = 0
        n_data = len(data_objs)

        for i, spec in enumerate(specs):
            # If no data left to consume, but we still have specs => error
            if data_idx >= n_data:
                raise ValueError(f"Not enough data for spec '{spec.name}'. Expecting more inputs.")
            
            if not spec.variadic:
                data_obj = data_objs[data_idx]
                
                # Must match exactly one data
                # logger.debug(f"spec: {spec}, data: {data_objs[data_idx]}")
                if not spec.matches_data(data_objs[data_idx]):
                    raise InitError(
                        f"Typed check mismatch for data {data_objs[data_idx].id} w\ meta: {data_objs[data_idx].metadata} against spec: {spec}.\n"
                        f"  - Expected data_type={spec.dtype}, shape={spec.shape}\n"
                        f"  - Got data_type={data_objs[data_idx].metadata.dtype}, shape={data_objs[data_idx].shape}"
                    )
                data_idx += 1
            else:
                # spec is variadic => gather all remaining inputs
                chunk = data_objs[data_idx:]
                if not chunk:
                    # If chunk is empty, but we have a variadic spec => check zero-arg success
                    if not spec.matches_data_list([]):
                        raise ValueError(f"Variadic spec {spec.name} could not match empty list of data.")
                else:
                    if not spec.matches_data_list(chunk):
                        raise ValueError(
                            f"Variadic spec {spec.name} mismatch. "
                            f"Failed to match data from index {data_idx}..{n_data-1}"
                        )
                # We consume ALL remaining data
                data_idx = n_data
                # If you only allow one variadic spec total, break after
                # break

        # If we haven't consumed all data_objs, that means we had leftover inputs
        if data_idx < n_data:
            raise ValueError(
                f"Extra data not consumed by signature. We used {data_idx} of {n_data} inputs."
            )

        return True
   
    def lefts(self):
        """Iterable over all registers that appear on the LEFT as input."""

       
        yield from self._lefts.values()

    def rights(self):
        """Iterable over all registers that appear on the RIGHT as output."""

        yield from self._rights.values()

    def get_left(self, name: str) -> Dict:
        """Get a left register by name."""
        # logger.debug("  Fetching LEFT register by name: %s", name)
        return self._lefts[name]

    def get_right(self, name: str) -> Dict:
        """Get a right register by name."""
        # logger.debug("  Fetching RIGHT register by name: %s", name)
        return self._rights[name]
    
    def groups(self) -> Iterable[Tuple[str, List[RegisterSpec]]]:
        """Iterate over data groups by name.

        RegisterSpec objects with shared names (but differing `.flow` attributes) can be implicitly grouped.
        """
        groups = defaultdict(list)
        for reg in self._registers:
            groups[reg.name].append(reg)

        yield from groups.items()

    
    def __repr__(self):
        return f'Signature({repr(self._registers)})'
    

    def __contains__(self, item):
        return item in self._registers
    @overload
    def __getitem__(self, key: int) -> RegisterSpec:
        pass

    @overload
    def __getitem__(self, key: slice) -> Tuple[RegisterSpec, ...]:
        pass
    def __getitem__(self, key):
        return self._registers[key]
    def __iter__(self) -> Iterable[Dict]:
        yield from self._registers

    def __len__(self) -> int:
        return len(self._registers)

    def __eq__(self, other) -> bool:
        return self._registers == other._registers

    