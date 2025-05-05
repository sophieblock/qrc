import torch
import inspect
from typing import Optional, Tuple,Dict, Any, Union,Callable
from torch import SymInt, SymBool, SymFloat
from typing_extensions import TypeGuard

from typing import Any, List
import types
import numpy as np

import torch
from torch.fx.operator_schemas import (
        OpOverload,
        OpOverloadPacket,
        type_matches,
        _args_kwargs_to_normalized_args_kwargs
    )

from .data_types import *
from ...util.log import logging
logger = logging.getLogger(__name__)

ALLOWED_BUILTINS = [int, float, str, bool, list, tuple, set]

def build_inspect_signature_from_process_signature(signature: "Signature") -> inspect.Signature:
    """
    Convert the left-flow portion of `signature` into a Python Signature object:
      - Non-variadic specs => normal positional params
      - If exactly one is variadic => we represent it with a *args param
      - We skip specs that are purely flow=RIGHT, so we don't demand them at call-time
    """
    params = []
    # logger.debug(f"Signature: {signature}")
    # for reg in signature:
    #     if reg.flow & Flow.LEFT:
    #         print("reg.flow & Flow.LEFT: ",reg.flow & Flow.LEFT)
    #     logger.debug(reg.flow)
    left_specs = [str(s) for s in signature if s.flow & Flow.LEFT]
    # print(f"left_specs: {left_specs}")
    # print(f"right_specs: {[str(s) for s in signature if s.flow & Flow.RIGHT]}")
    # If you allow multiple variadic specs, you'd need more advanced logic.
    # We assume at most one is variadic for now.
    variadic_seen = False

    for i, spec in enumerate(signature.lefts()):
        if not spec.variadic:
            # a normal positional parameter
            # logger.debug(f'spec name: {spec.name}, {spec.name.isidentifier()}')
            params.append(
                inspect.Parameter(
                    name=spec.name,
                    kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            )
        else:
            # If we already saw a variadic, we can’t add a second *args
            if variadic_seen:
                raise ValueError("Multiple variadic specs found; not supported by this approach.")
            variadic_seen = True
            # Represent it with a *args param 
            params.append(
                inspect.Parameter(
                    name=spec.name,
                    kind=inspect.Parameter.VAR_POSITIONAL,  # means *<spec.name>
                )
            )
            # We don't create subsequent params, or we keep scanning if you want
            # For a single-variadic approach, we can just continue or break
            # break

    return inspect.Signature(parameters=params)


def normalize_process_call(
    target: Callable,
    args: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]] = None,
    *,
    arg_types: Optional[Tuple[Any, ...]] = None,
    kwarg_types: Optional[Dict[str, Any]] = None,
    force_kwarg_mode: bool = False,
) -> Dict[str, Any]:
    """
    A unified function that attempts to "bind" user inputs to either:
      (1) A Process subclass's signature (e.g. MatrixMult)
      (2) A torch operator (e.g. torch.add)
      (3) A normal Python function.
    
    Returns a dictionary {param_name -> actual_argument} with resolved inputs,
    or raises an error if binding fails.
    
    For Process targets, it builds an inspect.Signature from process.signature,
    binds the provided Data inputs (unwrapping .data when necessary), and validates
    the binding against the process's RegisterSpecs.
    
    For torch ops, it first uses get_signature_for_torch_op to retrieve candidate schemas,
    attempts a first-pass binding, and if multiple schemas match, uses the provided
    arg_types/kwarg_types to disambiguate. Finally, it binds the chosen signature and
    returns dict(bound.arguments) so that parameter names (e.g. "input", "other") are used.
    
    For all other callables, it falls back to inspect.signature(target).
    
    Args:
        target (Union[Process, callable]): The function or Process object.
        args (Tuple[Any]): The positional arguments.
        kwargs (Optional[Dict[str, Any]]): The keyword arguments.
        arg_types (Optional[Tuple[Any]]): Additional type hints for disambiguation.
        kwarg_types (Optional[Dict[str, Any]]): Additional type hints for disambiguation.
        force_kwarg_mode (bool): If True, converts the final bound mapping to a kwargs-only representation.
    
    Returns:
        A dict {param_name -> Data or python object} with resolved inputs.
    """
    # print(f"\n=== Entering normalize_process_call where force_kwarg_mode={force_kwarg_mode} ===")
    from torch.fx.operator_schemas import get_signature_for_torch_op

    if kwargs is None:
        kwargs = {}
    normalized_data: Dict[str, Any] = {}
  
    from torch._library.infer_schema import infer_schema
    # logger.debug(f'target: {target}\n - args: {args}\n - types: {arg_types}\n - to bind: {args_for_bind}')
    # logger.debug(f'target type: {type(target)}\n - args: {args}\n - types: {arg_types}')
    # (1) If target is a Process => build an inspect.Signature from process.signature
    if isinstance(target, Process):
        fx_sig = build_inspect_signature_from_process_signature(target.signature)
        param_names = [p.name for p in target.signature.lefts()]
        args_for_bind = list(args)
        kwargs_for_bind = {k: (v.data if hasattr(v, "data") else v) for k, v in kwargs.items()}
        try:
            bound = fx_sig.bind(*args_for_bind, **kwargs_for_bind)
            bound.apply_defaults()
        except TypeError as te:
            matchobj = re.search(r"missing a required argument: '(.*?)'", str(te))
            logger.debug(f"matchobj: {matchobj}")
            if matchobj:
                missing_param = matchobj.group(1)
                raise ValueError(f"Missing data for parameter {missing_param}") from te
            raise ValueError(f"Failed to bind arguments to Process {target}: {te}") from te
        param_list = list(fx_sig.parameters.keys())
        used_positional_count = len(bound.args)
        for idx, pname in enumerate(param_list):
            if idx < used_positional_count:
                data_obj = args[idx]
            else:
                data_obj = kwargs.get(pname, None)
            if data_obj is None:
                raise ValueError(f"Missing data for parameter {pname}")
            normalized_data[pname] = data_obj

        target.signature.validate_data_with_register_specs(
            [normalized_data[p.name] for p in target.signature.lefts()]
        )
        # logger.debug(f"target falls into branch (1) since subclass of Process. Resulting normalized keys: {normalized_data.keys()}")
       
    # (2) If target is a torch op => do advanced schema approach, skip normal signature fallback
    elif (isinstance(target, OpOverloadPacket)
          or isinstance(target, OpOverload)
          or isinstance(target, types.BuiltinFunctionType)):
        normalized_data = _normalize_torch_op(target, args, kwargs, arg_types, kwarg_types)
        logger.debug(f"target falls into branch (3) since of type: {type(target)}. Resulting normalized keys: {normalized_data.keys}")
    else:
        # Fallback: use inspect.signature on unwrapped target.

        sig = inspect.signature(inspect.unwrap(target))
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        normalized_data = dict(bound.arguments)
        logger.debug(f" no branch fallback for type: {type(target)}. Resulting normalized keys: {normalized_data.keys}")



     # Finally, if force_kwarg_mode is True, rebind to kwargs-only.
    if force_kwarg_mode:
        if isinstance(target, Process):
            fx_sig = build_inspect_signature_from_process_signature(target.signature)
        elif (isinstance(target, OpOverloadPacket)
              or isinstance(target, OpOverload)
              or isinstance(target, types.BuiltinFunctionType)):
            sigs = get_signature_for_torch_op(target)
            fx_sig = sigs[0] if sigs else None
        else:
            fx_sig = inspect.signature(inspect.unwrap(target))
        if fx_sig is not None:
            bound_kw = fx_sig.bind(**normalized_data)
            bound_kw.apply_defaults()
            normalized_data = dict(bound_kw.arguments)
    # logger.debug(f"Final normalized result: {normalized_data}")
    return normalized_data


def _extract_element_types(val: Any) -> List[Any]:
    """
    Extracts all possible element types from `val`, handling lists, tensors, matrices, etc.
    """
    import numpy as np
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

def _shape_is_consistent(spec_shape, data_shape):
    # same length => each slot either matches or is Dyn
    if len(spec_shape.__args__) != len(data_shape.__args__):
        return False
    for sdim, ddim in zip(spec_shape.__args__, data_shape.__args__):
        if sdim == Dyn or ddim == Dyn:
            continue
        if sdim != ddim:
            return False
    return True

def type_matches(signature_type: Any, argument_type: Any):
    import inspect
    from typing import Any, Union
    import numbers
    import warnings
    sig_origin_type = getattr(signature_type, "__origin__", signature_type)

    if signature_type is argument_type:
        return True

    # Union types in signature. Given type needs to match one of the
    # contained types in the Union
    if sig_origin_type is Union and signature_type != argument_type:
        sig_contained = signature_type.__args__
        return any(type_matches(c, argument_type) for c in sig_contained)

    if getattr(signature_type, "__origin__", None) is list:
        sig_el_type = signature_type.__args__[0]

        # int can be promoted to list[int]
        if argument_type is int and sig_el_type is int:
            return True

        if not inspect.isclass(sig_el_type):
            warnings.warn(
                f"Does not support nested parametric types, got {signature_type}. Please file a bug."
            )
            return False
        if getattr(argument_type, "__origin__", None) is list:
            return issubclass(argument_type.__args__[0], sig_el_type)

        def is_homogeneous_tuple(t):
            if getattr(t, "__origin__", None) is not tuple:
                return False
            contained = t.__args__
            if t.__args__ == ((),):  # Tuple[()].__args__ == ((),) for some reason
                return True
            return all((c is Ellipsis) or issubclass(c, sig_el_type) for c in contained)

        # Tuple[T] is accepted for List[T] parameters
        return is_homogeneous_tuple(argument_type)

    # Dtype is an int in schemas
    if signature_type is int and argument_type is torch.dtype:
        return True

    if signature_type is numbers.Number and argument_type in {int, float}:
        return True
    if inspect.isclass(argument_type) and inspect.isclass(signature_type):
        return issubclass(argument_type, signature_type)

    return False
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

def is_consistent_data_type(dtype_a, dtype_b):
    """
    Determines if dtype_a (expected) is consistent with dtype_b (actual).
    """
    from workflow.simulation.refactor.data_types import (
        DataType, NDimDataType, TensorType, MatrixType, Dyn
    )
    # logger.debug(f"dtype a: {dtype_a}, {dtype_a == Dyn} dtype b: {dtype_b}")

    # If dtype_a is a MatrixType or a subclass of NDimDataType => treat it as TensorType
    if isinstance(dtype_a, MatrixType) or \
       (isinstance(dtype_a, type) and issubclass(dtype_a, NDimDataType) and dtype_a is not NDimDataType):
        if not isinstance(dtype_a, type) and hasattr(dtype_a, 'shape'):
            dtype_a = TensorType(shape=dtype_a.shape, element_dtype=dtype_a.element_dtype)
        else:
            # If it's the class itself, we unify it to TensorType the class
            dtype_a = TensorType

    # If dtype_b is a MatrixType or a subclass of NDimDataType => treat it as TensorType
    if isinstance(dtype_b, MatrixType) or \
       (isinstance(dtype_b, type) and issubclass(dtype_b, NDimDataType) and dtype_b is not NDimDataType):
        if not isinstance(dtype_b, type) and hasattr(dtype_b, 'shape'):
            dtype_b = TensorType(shape=dtype_b.shape, element_dtype=dtype_b.element_dtype)
        else:
            dtype_b = TensorType

    if dtype_a == dtype_b:
        return True

    if dtype_a == Dyn or dtype_b == Dyn:
        return True

    if isinstance(dtype_a, (list, tuple)):
        return any(is_consistent_data_type(e, dtype_b) for e in dtype_a)

    if dtype_a == TensorType:
        if dtype_b == list or (hasattr(dtype_b, "__origin__") and dtype_b.__origin__ == list):
            return True
    if dtype_b == TensorType:
        if dtype_a == list or (hasattr(dtype_a, "__origin__") and dtype_a.__origin__ == list):
            return True

    # Case 1: Both are DataType classes (not instances)
    if (isinstance(dtype_a, type) and issubclass(dtype_a, DataType)) \
       and (isinstance(dtype_b, type) and issubclass(dtype_b, DataType)):

        if dtype_a is dtype_b:
            return True
        
        if issubclass(dtype_a, NDimDataType) and issubclass(dtype_b, NDimDataType):
            return True

        return False

    # Case 2: Expected is a DataType class, actual is an instance
    if isinstance(dtype_a, type) and issubclass(dtype_a, DataType):
        return isinstance(dtype_b, dtype_a)

    if isinstance(dtype_b, type) and issubclass(dtype_b, DataType):
        return isinstance(dtype_a, dtype_b)

    # Case 3: If both are TensorType instances, check shape consistency (we may not want this at this level)
    if isinstance(dtype_a, TensorType) and isinstance(dtype_b, TensorType):
        if len(dtype_a.shape) != len(dtype_b.shape):
            return False
        return all(
            is_consistent_data_type(exp_dim, act_dim)
            for exp_dim, act_dim in zip(dtype_a.shape, dtype_b.shape)
        )

    return False

def create_data_type_hint(x: Any) -> Any:
    """
    A custom version of PyTorch’s create_type_hint
    If 'x' is a list or tuple, we produce a typed annotation (like list[int]) 
    or something if you want. If 'x' is a bare scalar => int, float, etc.
    If 'x' is a torch.Tensor => e.g. TensorType or something.

    source: torch.fx
    """
    # This example is partial. Tweak as needed for your logic.


    # If it's a list or tuple, try to deduce a homogeneous element type
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            # e.g. list[Any]
            return list[Any] if isinstance(x, list) else tuple[Any, ...]
        # find a single “base_type” that is the super-type of all elements
        base_type = type(x[0])
        for elem in x:
            # if 'issubclass(type(elem), base_type)' we unify
            if not issubclass(type(elem), base_type):
                # fallback
                return list[Any] if isinstance(x, list) else tuple[Any, ...]
        # if we got here => homogeneous => we produce list[base_type]
        if isinstance(x, list):
            return list[base_type]
        else:
            return tuple[base_type, ...]
    else:
        if inspect.isclass(x) and issubclass(x, DataType):
            return x
        # Not a list/tuple => return type(x).
        # Or if x is a torch.Tensor => maybe 'TensorType'
        if isinstance(x, torch.Tensor):
            return torch.Tensor  # or a custom type
        
        if x in ALLOWED_BUILTINS:
            return x
        return type(x)




def is_symbolic(
    val: Union[int, SymInt, float, SymFloat, bool, SymBool]
) -> TypeGuard[Union[SymInt, SymFloat, SymBool]]:
    if isinstance(val, (int, float, bool)):
        return False
    return val.node.is_symbolic()

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
    instance with the inferred shape and element_dtype determined from the value's dtype.
    
    In particular:
      - For a np.ndarray:
          * If value.dtype.kind is 'U' or 'S' (string types), element_dtype is set to str.
          * If value.dtype.kind is 'i' (integer), element_dtype is set to int.
          * If value.dtype.kind is 'f' (floating point), element_dtype is set to np.float32 for float32,
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
            return TensorType(shape=value.shape, element_dtype=elem_dt)
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
            return TensorType(shape=tuple(value.shape), element_dtype=elem_dt)
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
        raise ValueError(f"Invalid shape value: {v}")