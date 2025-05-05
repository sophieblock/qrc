
from typing import Any, Dict, Optional,List, Dict, Tuple, Iterable, Union, Sequence, Optional,overload,cast
from pprint import pformat
import numpy as np
import itertools
import sympy
from torch.fx.experimental.symbolic_shapes import (
    free_unbacked_symbols,
    DimDynamic,
    SymbolicContext,
    guard_int,
    StatelessSymbolicContext,
   

)
from collections import defaultdict
import inspect
from torch._subclasses.fake_tensor import FakeTensor
import torch
from torch.fx.experimental.symbolic_shapes import ShapeEnv as TorchShapeEnv
from torch.fx.experimental.sym_node import method_to_operator, SymNode
from torch import SymInt, SymBool, SymFloat
from torch.utils._sympy.numbers import int_oo
import torch.fx
import torch.fx.traceback as fx_traceback
from torch.fx.experimental.recording import record_shapeenv_event, replay_shape_env_events, FakeTensorMeta
from torch._guards import ShapeGuard, Source
from torch._guards import ShapeGuard, Source
from torch.fx.experimental.symbolic_shapes import (
    free_unbacked_symbols,
    DimDynamic,
    SymbolicContext
)
from torch.fx.experimental.symbolic_shapes import ShapeEnv as TorchShapeEnv

import sympy
from typing import Dict, List, Union
from torch.fx.experimental.recording import record_shapeenv_event
from typing_extensions import TypeGuard

from .data_types import (
    DataType,
    NDimDataType,
    MatrixType,
    TensorType,
    CBit,
    CAny,
    QAny,
    QBit
)
from .unification_tools import (ALLOWED_BUILTINS, 
                            canonicalize_dtype, 
                            get_shape, 
                            dim_is_int_or_dyn,
                            shape_is_tuple,
                            is_consistent_data_type
)
from .utilities import InitError
from ...util.log import logging
logger = logging.getLogger(__name__)
from attrs import field


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


def constrain_unify(a: torch.SymInt, b: torch.SymInt) -> None:
    """
    Given two SymInts, constrain them so that they must be equal.  NB:
    this will not work with SymInts that represent nontrivial expressions
    (yet!)
    """
    if not isinstance(a, SymInt):
        if not isinstance(b, SymInt):
            assert a == b
            return
        else:
            shape_env = b.node.shape_env
    else:
        shape_env = a.node.shape_env

    shape_env._constrain_unify(a, b)

import enum

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



from attrs import frozen, define

import types
ALLOWED_BUILTINS = [int, float, str, bool, list, tuple, set]

def flow_to_str(flow: Flow) -> str:
    """
    Helper to convert a Flow enum (LEFT, RIGHT, THRU) to a short text label.
    """
    if flow == Flow.LEFT:
        return "input"
    elif flow == Flow.RIGHT:
        return "output"
    elif flow == Flow.THRU:
        return "in/out"
    else:
        # If someone combined flags in an unexpected way, just show the bits
        return str(flow)
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

@define
class RegisterSpec:
    """
    A refined RegisterSpec that can handle:
      • DataType classes (MatrixType, TensorType, etc.)
      • DataType instances (MatrixType(2,3))
      • Built-in Python types (int, str, float, etc.)
      • Generic aliases like list[int], tuple[str, ...]
    """
    name: str
    dtype: type = field(converter=canonicalize_dtype)
    # dtype: any  # e.g. <class 'TensorType'>, <class 'int'>, list[int], or an actual DataType instance
    _shape: Tuple[SymInt, ...] = field(
        default=tuple(),
        converter=lambda v: (v,) if v in ALLOWED_BUILTINS else tuple(v)
    )
    flow: Flow = Flow.THRU
    variadic: bool = False

    def __attrs_post_init__(self):
        """
        We allow the following:
          1) A DataType instance (e.g. TensorType((2,3)))
          2) A DataType subclass (e.g. <class 'TensorType'>)
          3) A builtin Python type or a 'GenericAlias' (list[int], tuple[str], etc.)
        Everything else => raise ValueError
        """
        import inspect
        
        # If it's an actual DataType instance => OK
        if isinstance(self.dtype, DataType):
            
            logger.debug(f' -> Registering DataType INSTANCE: {repr(self)}')
            return  # e.g. self.dtype is CBit()

        # If it's a class that inherits DataType => OK
        if inspect.isclass(self.dtype) and issubclass(self.dtype, DataType):
            logger.debug(f' -> Registering DataType SUBCLASS: {self}')
            return  # e.g. <class 'TensorType'>

        # If it's a GenericAlias (like list[int]) => OK
        if isinstance(self.dtype, types.GenericAlias):
            # e.g. list[int], tuple[float], etc. Accept it
            logger.debug(f' -> Registering type GenericAlias: `{self}`')
            return

        # If it's one of the standard built-in types we want to allow
        ALLOWED_BUILTINS = [int, float, str, bool, list, tuple, set]
        if self.dtype in ALLOWED_BUILTINS:
            logger.debug(f' -> Registering base type: {self}')
            return 

        # If we got here => not recognized
        raise ValueError(
            f"name '{self.name}' dtype must be a DataType or a builtin type in ALLOWED_BUILTINS, "
            f"or a GenericAlias. Got {self} {self.dtype} of type {type(self.dtype)}"
        )

        
    @property
    def shape_symbolic(self) -> Tuple[SymInt, ...]:
        return self._shape
    @property
    def shape(self) -> Tuple[int, ...]:
        
        return cast(Tuple[int, ...], self._shape)
    @property
    def num_units(self) -> int:
        """
        Returns the number of 'units' (e.g. matrix elements, bits, qubits, etc.).
        Delegates to self.data.num_units if self.data is a DataType.
        """
        if isinstance(self.dtype, DataType):
            return self.dtype.num_units
        elif isinstance(self.dtype, int):
            # e.g. a single integer might be 1 unit
            return 1
        # fallback / error
        return 0  # or raise ValueError
    
    @property
    def bitsize(self) -> int:
        if isinstance(self.dtype, type) and issubclass(self.dtype, NDimDataType):
            # Dynamically create an instance so we can read bitsize
            tmp_instance = self.dtype(self._shape, element_dtype=float)
            return tmp_instance.bitsize
        elif issubclass(self.dtype, DataType):
            # It's already an instance
            return self.dtype.bitsize
        raise AttributeError(f"Cannot determine bitsize for data type: {type(self)}")

    # @property
    # def memory_bytes(self) -> int:
    #     """
    #     If we want to track memory usage in bytes:
    #     If data is an NDimDataType or another specialized type, we can use its memory_in_bytes.
    #     """
    #     if isinstance(self.data, NDimDataType):
    #         return self.data.memory_in_bytes
    #     # If it's not an NDimDataType, you can do something else or raise an error
    #     return 0  # or fallback logic
    
    
    def all_idxs(self):
        """Generate all index tuples based on the node's shape."""
        shape = self.shape
        return itertools.product(*[range(int(dim)) for dim in shape])
    
    # def __eq__(self, other):
    #     if other.__class__ is not self.__class__:
    #         return NotImplemented
    #     # if not isinstance(other, RegisterSpec):
    #     #     return False
    #     return (
    #         self.name == other.name
    #         and self.dtype == other.dtype
    #         and self._shape == other._shape
    #         and self.flow == other.flow
    #     )
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
        # If both are already DataType instances:
        if isinstance(self.dtype, DataType) and isinstance(other.dtype, DataType):
            return is_consistent_data_type(self.dtype, other.dtype)

        # If one is a DataType class (e.g. TensorType) and the other is an instance (e.g. TensorType((3,3))):
        import inspect
        if inspect.isclass(self.dtype) and issubclass(self.dtype, DataType) and isinstance(other.dtype, DataType):
            # e.g. self.dtype == TensorType, other.dtype == TensorType((3,3))
            # We rely on is_consistent_data_type to see if the instance is of that class, ignoring shape for now.
            return is_consistent_data_type(self.dtype, other.dtype)

        if inspect.isclass(other.dtype) and issubclass(other.dtype, DataType) and isinstance(self.dtype, DataType):
            return is_consistent_data_type(self.dtype, other.dtype)

        # Otherwise, fallback to normal equality or the unification tool:
        return is_consistent_data_type(self.dtype, other.dtype)

    def __hash__(self):
        # Combine the same fields used in __eq__.
        return hash((self.name, self.dtype, self._shape, self.flow))
    # def __repr__(self):
    #     # print(self.shape, self.symbolic_shape)
    #     if isinstance(self.shape,tuple) and len(self.shape)>0:
    #         return f"Spec('{self.name}',shape: {self.shape}, dtype={self.dtype})"
    #     return f"Spec('{self.name}', dtype={self.dtype}, _shape={self._shape}, {self.flow})"

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
        actual_dtype = data_obj.metadata.dtype
        # Normalize both expected and actual types.
        norm_expected = canonicalize_dtype(self.dtype)
        norm_actual = canonicalize_dtype(actual_dtype)
        
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

        if self.metadata.dtype == NDimDataType or self.metadata.dtype == MatrixType or self.metadata.dtype == TensorType:
            # We unify the static .shape from data.metadata
            # and create symbolic sizes in shape_env
           
            # Create symbolic representation
            # logger.debug(f'Creating symbolic represetnations of N-dim objects')
            self._symbolic_shape = self.shape_env.create_symbolic_sizes(
                self.shape, source=self.source
            )

        elif isinstance(self.data, (CAny,CBit)):
            # A bag of classical bits
            # shape = empty tuple, but the bitcount is data.num_units
            self._symbolic_shape = self.shape_env.create_symbolic_int(
                self.data.num_units,
                source=self.source,
                symbolic_type="bit"
            )
           

        elif isinstance(self.data, (QAny,QBit)):
            # A bag of qubits
            self._symbolic_shape = self.shape_env.create_symbolic_int(
                self.data.num_units,
                source=self.source,
                symbolic_type="qbit"
            )
       
        else:
            # Fallback: no symbolic shape
            # logger.debug(f'Defaulting symbolic shape to ()')
            self._symbolic_shape = ()

    def __str__(self) -> str:
        """
        Return a concise, human-readable summary, e.g.:
          arg0: TensorType (shape=(3, 3)) [in/out]
        """
        # Decide how to show shape. If empty or None, you could say "unspecified".
        if self.shape:
            shape_str = f"(shape={self.shape})"
        else:
            shape_str = ''

        # If dtype is a class, show its __name__; if it's an instance, show the class name.
        if isinstance(self.dtype, type):
            dtype_str = self.dtype.__name__
        else:
            dtype_str = self.dtype.__class__.__name__

        # Flow decode
        flow_str = self.flow

        # Indicate if it’s a variadic spec
        variadic_label = " (variadic)" if self.variadic else ""

        return f"{self.name}: {dtype_str} {shape_str} [{flow_str}]{variadic_label}"

    # def __repr__(self) -> str:
    #     """
    #     Fallback if someone calls repr():
    #     """
    #     return str(self)
from typing import Union


class Signature:
    """
    Signature is an ordered collection of RegisterSpecs, partitioned into LEFT, RIGHT, and/or THRU flows.
    
    Key functionality:
    - `build(...)`: Construct from either (a) an iterable of `RegisterSpec` objects, or (b) dictionary-like 
      declarations of simple types (like `x=1` => a single CBit).
    - `build_from_properties(...)`: For processes that define input_props and output_props.
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

      
        

    @staticmethod
    def _dedupe(kv_iter: Iterable[Tuple[str, Dict]]) -> Dict[str, Dict]:
        """Construct a dictionary, but check that there are no duplicate keys."""
        d = {}
        for k, v in kv_iter:
            if k in d:
                # Generate a unique key by appending a number or UUID
                unique_key = f"{k}_{len(d)+1}"
                d[unique_key] = v
                logger.warning(f"RegisterSpec {k} is specified more than once per side. Renaming to {unique_key}.")
            else:
                d[k] = v
            #logger.debug("Register %s added to the dictionary.", k)
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

        # for k,v in registers.items():
        #     logger.debug(f"{k}, v: {v}")
        # return cls(
        #     RegisterSpec(name=k, dtype=CBit()) if v == 1 else RegisterSpec(name=k, dtype=v)
        #     for k, v in kwargs.items() if v
        # )
        return cls(
            RegisterSpec(name=k, dtype=CBit() if v == 1 else CAny(v))
            for k, v in registers.items() if v
        )
    
    
    @classmethod
    def builder_from_dtypes(cls, **data_types: DataType):
        return cls(RegisterSpec(name=k,dtype=v) for k,v in data_types.items() if v.num_units)
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
        from .schema import Flow  # Ensure Flow is imported
        from workflow.simulation.refactor.data_types import TensorType
        
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
                # logger.debug(f"Attempting to register {name} with dtype: {dtype}")
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
    # @classmethod
    # def build_from_properties(cls, input_props, output_props) -> 'Signature':
    #     """
    #     Build a Signature from input and output property dictionaries.
    #     This implementation assigns input register names as "arg0", "arg1", ...,
    #     and output register names as "OUT" if there is one output, or "arg_out0", "arg_out1", ... if multiple.
        
    #     If a property dictionary's "Data Type" value is a list, the first element is taken as the canonical type.
    #     If the canonical type is numpy.ndarray (as returned by type(np.array([]))), it is replaced with TensorType.
    #     The "shape" is taken from the property if provided, or defaults to an empty tuple.
    #     All input registers are assigned Flow.THRU and all output registers Flow.RIGHT.
    #     """
        
    #     registers = []
    #     import copy
    #     import numpy as np
    #     from .schema import Flow  # Ensure Flow is imported
    #     # Import our canonical type
    #     from workflow.simulation.refactor.data_types import TensorType
        
    #     logger.debug("Building .signature from Processproperties *****")
        
    #     # Process input properties
    #     if input_props:
    #         for i, prop in enumerate(input_props):
    #             name = f"arg{i}"
    #             dtype = prop.get("Data Type")
    #             # If a list of types is provided, choose the first one.
    #             if isinstance(dtype, list):
    #                 dtype = dtype[0]
    #             # Convert numpy.ndarray to TensorType if needed
    #             if dtype is np.ndarray:
    #                 dtype = TensorType
    #             # Allow an optional 'shape' property; default to empty tuple.
    #             shape = copy.deepcopy(prop.get("shape", ()))
    #             registers.append(RegisterSpec(name=name, dtype=dtype, shape=shape, flow=Flow.THRU))
        
    #     # Process output properties
    #     if output_props:
    #         if len(output_props) == 1:
    #             # Single output: name it "OUT"
    #             name = "OUT"
    #             prop = output_props[0]
    #             dtype = prop.get("Data Type")
    #             if isinstance(dtype, list):
    #                 dtype = dtype[0]
    #             # if dtype is np.ndarray:
    #             #     dtype = TensorType
    #             shape = copy.deepcopy(prop.get("shape", ()))
    #             registers.append(RegisterSpec(name=name, dtype=dtype, shape=shape, flow=Flow.RIGHT))
    #         else:
    #             # Multiple outputs: name them "arg_out0", "arg_out1", etc.
    #             for i, prop in enumerate(output_props):
    #                 name = f"arg_out{i}"
    #                 dtype = prop.get("Data Type")
    #                 if isinstance(dtype, list):
    #                     dtype = dtype[0]
    #                 # if dtype is np.ndarray:
    #                 #     dtype = TensorType
    #                 shape = copy.deepcopy(prop.get("shape", ()))
    #                 registers.append(RegisterSpec(name=name, dtype=dtype, shape=shape, flow=Flow.RIGHT))
        
    #     return cls(registers)
   
  
    @classmethod
    def build_from_data(cls, inputs, output_props=None) -> "Signature":
        """
        SOON TO BE REMOVED -> USE .build(), .build_from_dtypes(), or build_from_properties()
        instead
       
        """
        from rich.pretty import pretty_repr
        from .unification_tools import create_data_type_hint
        registers = []

        # Step 1: Handle input data (Flow.LEFT or Flow.THRU)
        input_usage_map = {data.properties.get("Usage"): data for data in inputs}
        logger.debug(f"Input usage map:")
        logger.info(pretty_repr(input_usage_map))
        for data_in in inputs:
            name = data_in.id
            dtype = data_in.metadata.dtype

            shape = data_in.metadata.dtype
            registers.append(RegisterSpec(name=name, dtype=dtype, shape=shape))

            # registers.append(data_in)  # Add inputs with default flow (LEFT)

        # Step 2: Process outputs based on output properties
        if output_props:
            for i, prop in enumerate(output_props):
                usage = prop.get("Usage")

                # If an input's "Usage" matches an output's "Usage", mark it as THRU
                if usage in input_usage_map:
                    logger.debug(f"{usage} in {input_usage_map}, settng flow attr to THRU")
                    input_usage_map[usage].flow = Flow.THRU
                else:
                    # Create a new Data object for outputs not in inputs
                    name = f'out_{i}'
                    # logger.debug(f"Generating output ReigsterSpec {name} with props: {prop}")
                    dtype = create_data_type_hint(prop.get("Data Type", None))
                    shape = prop.get("shape", ())
                    logger.debug(f"Generating output ReigsterSpec {name} with props: {prop}\n dtype - {dtype}")
                    registers.append(RegisterSpec(name=name, dtype=dtype, shape=shape, flow=Flow.RIGHT))
                    # registers.append(
                    #     RegisterSpec(
                    #         name=prop.get("Usage", None) or f'out_{i}',
                    #         dtype=prop,
                    #         flow=Flow.RIGHT,
                    #     )
                    # )

        # Build the Signature with updated registers
        signature = cls(registers)
        # logger.debug("Signature built successfully: %s", signature)
        return signature
   
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
    