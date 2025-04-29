from attrs import field, validators,define
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
from .dtypes import *

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

    # ------------------------------- metadata -----------------------------
    name:   str   = field(converter=_sanitize_name)
    dtype:  Any   = field(converter=canonicalize_dtype)
    _shape: Tuple[Any, ...] = field(
        default=tuple(),
        converter=get_shape,
        validator=validators.deep_iterable(
            member_validator=dim_is_int_or_dyn,
            iterable_validator=shape_is_tuple,
        ),
    )
    flow:    Flow = field(default=Flow.THRU)
    variadic: bool = field(default=False)

    # ------------------------------- derived ------------------------------
    domain:  str   = field(init=False)         # 'Q' or 'C'
    is_symbolic: bool = field(init=False)
    
    # ----------------------------------------------------------------------
    #  Post-init: domain tagging, dtype instantiation, safety checks
    # ----------------------------------------------------------------------
    def __attrs_post_init__(self):
        import inspect
        init_logs = []
        # ---- 2) determine domain ('Q' or 'C')
        if isinstance(self.dtype, TensorType):
            inner = self.dtype.element_type
            self.domain = "Q" if isinstance(inner, QType) else "C"
            # forbid mixed element/domain mismatch
            if self.domain == "Q" and isinstance(inner, CType):
                raise TypeError(
                    "TensorType with classical element_type cannot live on a quantum wire"
                )
        else:
            self.domain = "Q" if isinstance(self.dtype, QType) else "C"

        if isinstance(self.dtype, DataType):
            # Instead of adopting dtype.shape when _shape is empty, we leave _shape as provided.
            # (This means that if _shape is not explicitly given, it remains (), making the register atomic.)
            init_logs.append(f"Registering DataType INSTANCE without inheriting shape: {repr(self)}")
        # ---- 1) instantiate dtype if user passed a DataType *class*
        elif inspect.isclass(self.dtype) and issubclass(self.dtype, DataType):
            # Only instantiate if the dtype constructor expects a shape argument
            try:
                self.dtype = self.dtype(self._shape)
                init_logs.append(f" -> Instantiated DataType from subclass with shape {self.shape}: {repr(self)}")
            except TypeError:
                # Constructor didn’t accept a shape – leave as class
                init_logs.append(f" -> Registering DataType SUBCLASS without shape: {repr(self)}")
                pass
        elif isinstance(self.dtype, types.GenericAlias):
            init_logs.append(f" -> Registering GenericAlias type: {repr(self)}")
        elif self.dtype in ALLOWED_BUILTINS:
            init_logs.append(f" -> Registering built-in base type: {repr(self)}")
        else:
            raise TypeError(
                f"name '{self.name}' dtype must be a DataType (instance or subclass), "
                f"a builtin type in ALLOWED_BUILTINS, or a GenericAlias. Got {self.dtype} of type {type(self.dtype)}"
            )

        if DEBUG_INIT_LOGS:
            for msg in init_logs:
                logger.debug(msg)
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
    def all_idxs(self):
        """Generate all index tuples based on the node's shape."""
        shape = self.shape
        yield from itertools.product(*[range(int(dim)) for dim in shape])
    
    # def all_idxs(self):
    #     """Generate all index tuples based on the node's shape."""
    #     shape = self.shape
    #     return itertools.product(*[range(int(dim)) for dim in shape])
    
    # ------------------------------------------------------------------
    #  Equality / hashing (name, dtype, shape, flow)
    # ------------------------------------------------------------------
    def __eq__(self, other):
        if not isinstance(other, RegisterSpec):
            return NotImplemented
        return (
            self.name == other.name
            and self.flow == other.flow
            and is_consistent_data_type(self.dtype, other.dtype)
            and self._shape == other._shape
        )

    def __hash__(self):
        return hash((self.name, self.flow, str(self.dtype), self._shape))


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

   # ------------------------------------------------------------------
    #  Matching helpers  (unchanged logic, but use domain tag)
    # ------------------------------------------------------------------
    def matches_data(self, data_obj) -> bool:
        arg_dtype = data_obj.metadata.dtype

        # must both be classical or both quantum
        if (isinstance(arg_dtype, QType)) != (self.domain == "Q"):
            return False

        # ------ allow “any-classical” to match any same-width classical type ------
        if isinstance(arg_dtype, CAny) and not isinstance(self.dtype, QType):
            return arg_dtype.data_width == self.dtype.data_width

        # (symmetrically, you could allow QAny ↔ QUInt/QInt if you need)
        if isinstance(arg_dtype, QAny) and isinstance(self.dtype, QType):
            return arg_dtype.data_width == self.dtype.data_width

        # fallback to your existing unification logic
        return is_consistent_data_type(self.dtype, arg_dtype)

    def matches_data_list(self, data_objs: List["Data"]) -> bool:
        if not self.variadic and len(data_objs) != 1:
            return False
        return all(self.matches_data(d) for d in data_objs)
   

    def __str__(self) -> str:
        return f'{get_flow_spec(self.flow)}(name={self.name}, dtype={self.dtype}, shape={self.shape})'
        

    # ------------------------------------------------------------------
    #  Printable
    # ------------------------------------------------------------------
    def __repr__(self):
        shape = f"(shape={self._shape})" if self._shape else ""
        return (
            f"{self.name}: {self.dtype}{shape} "
            f"[{self.flow.name}] domain={self.domain}"
            f"{' variadic' if self.variadic else ''}"
        )
    

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

        This method is useful when your Process subclass only specifies 
        expected_input_properties/output_properties attributes. 
        It creates RegisterSpec objects from dictionaries. It’s a good choice when 
        you want to let the process define its “contract” via simple dicts.
        """
        registers = []
        import copy
        
        
        # logger.debug("Building .signature from Process properties *****")
        
        # Possibly a helper unify_type_list, we skip for brevity
        def unify_type_list(type_list):
            # e.g. if MatrixType in type_list and TensorType in type_list => unify to TensorType
            # if single element => return it, else return type_list
            if len(type_list) == 1:
                return type_list[0]
            from workflow.simulation.refactor.dtypes import MatrixType, TensorType
            if MatrixType in type_list and TensorType in type_list:
                return TensorType
            return type_list

        usage_map = {}   # usage -> RegisterSpec
        registers = []

        # ------------------- 1) Process INPUTS as THRU registers -------------------
        for i, prop in enumerate(input_props or []):
            usage = prop.get("Usage", "")
            dtype = prop.get("Data Type")
            shape = copy.deepcopy(prop.get("shape", ()))

            # Possibly unify a list of types
            if isinstance(dtype, list):
                dtype = unify_type_list(dtype)

            # Name = usage if provided, else arg{i}
            if usage:
                name = _sanitize_name(usage)
            else:
                name = f"arg{i}"

            reg = RegisterSpec(
                name=name,
                dtype=dtype,
                shape=shape,
                flow=Flow.THRU,  # input is considered in/out
            )
            registers.append(reg)

            # If usage is non-empty, record it so we can unify outputs that match
            if usage:
                usage_map[usage] = reg

        # ------------------- 2) Process OUTPUTS with single vs. multiple logic -------------------
        out_props = output_props or []
        if len(out_props) == 1:
            # Single output special case
            single = out_props[0]
            usage = single.get("Usage", "")
            dtype = single.get("Data Type")
            shape = copy.deepcopy(single.get("shape", ()))

            if isinstance(dtype, list):
                dtype = unify_type_list(dtype)

            if usage:
                # Check if there's an existing input register with the same usage
                existing_reg = usage_map.get(usage)
                logger.debug(f"usage: {usage}, existing_reg: {existing_reg}")
                if existing_reg and existing_reg.flow == Flow.THRU:
                    # It's already in/out => skip making a new RegisterSpec
                    logger.debug(f"Unifying output usage '{usage}' with existing THRU register '{existing_reg.name}'.")
                else:
                    # Either not found or flow != THRU => create a new RIGHT register
                    out_name = _sanitize_name(usage)
                    reg = RegisterSpec(name=out_name, dtype=dtype, shape=shape, flow=Flow.RIGHT)
                    registers.append(reg)
            else:
                # usage is empty => fallback to "OUT"
                reg = RegisterSpec(name="OUT", dtype=dtype, shape=shape, flow=Flow.RIGHT)
                registers.append(reg)
        else:
            # Multiple outputs => name them "arg_out0", "arg_out1", etc. if usage is missing
            for j, prop in enumerate(out_props):
                usage = prop.get("Usage", "")
                dtype = prop.get("Data Type")
                shape = copy.deepcopy(prop.get("shape", ()))

                if isinstance(dtype, list):
                    dtype = unify_type_list(dtype)

                if usage:
                    existing_reg = usage_map.get(usage)
                    if existing_reg and existing_reg.flow == Flow.THRU:
                        # logger.debug(f"Unifying output usage '{usage}' with existing THRU register '{existing_reg.name}'.")
                        continue  # Skip creating a new register
                    else:
                        out_name = _sanitize_name(usage)
                else:
                    # usage empty => fallback naming
                    out_name = f"arg_out{j}"

                reg = RegisterSpec(name=out_name, dtype=dtype, shape=shape, flow=Flow.RIGHT)
                registers.append(reg)

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

    