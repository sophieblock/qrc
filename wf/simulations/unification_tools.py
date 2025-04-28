import torch
import inspect
from typing import Optional, Tuple,Dict, Any, Union,Callable
from torch import SymInt, SymBool, SymFloat
from typing_extensions import TypeGuard
from numpy.typing import NDArray
from typing import Any, List
import types
import numpy as np
from ...util.log import logging
logger = logging.getLogger(__name__)
# from .data_types import DataType, TensorType, MatrixType, Dyn
from .dtypes import *
ALLOWED_BUILTINS = [int, float, str, bool, list, tuple, set]
# def create_graph_signature(
#     fx_g: torch.fx.GraphModule,
#     fw_metadata: ViewAndMutationMeta,
#     in_spec: pytree.TreeSpec,
#     out_spec: pytree.TreeSpec,
#     *,
#     user_args_flat: list[Tensor],
#     params_and_buffers_flat: list[Tensor],
#     param_names: list[str],
#     buffer_names: list[str],
#     trace_joint: bool,
#     num_user_fw_outs: Optional[int],
#     loss_index: Optional[int],
# ) -> GraphSignature:
#     # Retrieve graph input names
#     graph_input_names = _graph_input_names(fx_g)
#     # Retrieve graph output names
#     graph_output_names = _graph_output_names(fx_g)

#     num_params_buffers = len(param_names) + len(buffer_names)
#     num_tokens = len(fw_metadata.tokens)
#     # We have enough restrictions on the graph (no de-duping, synthetic bases, etc),
#     # Such that # graph inps = # user inps + # params + # buffers
#     num_user_args = len(graph_input_names) - num_params_buffers - num_tokens

#     if trace_joint:
#         assert num_user_fw_outs is not None
#         num_fw_outs = num_user_fw_outs + fw_metadata.num_mutated_inp_runtime_indices
#         backward_output_names = graph_output_names[num_fw_outs:]

#         grad_index = itertools.count(0)
#         gradients_to_parameters = {
#             backward_output_names[next(grad_index)]: param_names[i]
#             for i, param in enumerate(params_and_buffers_flat)
#             if param.requires_grad
#         }

#         gradients_to_user_inputs = {
#             backward_output_names[next(grad_index)]: graph_input_names[
#                 i + len(params_and_buffers_flat)
#             ]
#             for i, user_input in enumerate(user_args_flat)
#             if user_input.requires_grad
#         }

#         assert len(gradients_to_parameters) + len(gradients_to_user_inputs) == len(
#             backward_output_names
#         )

#         # Check that we have fully accounted for all graph outputs
#         backward_signature = BackwardSignature(
#             gradients_to_parameters,
#             gradients_to_user_inputs,
#             graph_output_names[loss_index],
#         )
#     else:
#         backward_signature = None
#         num_user_fw_outs = (
#             len(graph_output_names)
#             - fw_metadata.num_mutated_inp_runtime_indices
#             - num_tokens
#         )

#     return GraphSignature.from_tracing_metadata(
#         in_spec=in_spec,
#         out_spec=out_spec,
#         graph_input_names=graph_input_names,
#         graph_output_names=graph_output_names,
#         view_mutation_metadata=fw_metadata,
#         named_parameters=param_names,
#         named_buffers=buffer_names,
#         num_user_inputs=num_user_args,
#         num_user_outputs=num_user_fw_outs,
#         loss_index=loss_index,
#         backward_signature=backward_signature,
#     )


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
    from workflow.simulation.refactor.dtypes import (
        DataType,CType, TensorType, MatrixType, Dyn
    )
    # logger.debug(f"dtype a: {dtype_a}, {dtype_a == Dyn} dtype b: {dtype_b}")

    # If dtype_a is a MatrixType or a subclass of NDimDataType => treat it as TensorType
    if isinstance(dtype_a, MatrixType) or \
       (isinstance(dtype_a, type) and issubclass(dtype_a, CType) and dtype_a is not CType):
        if not isinstance(dtype_a, type) and hasattr(dtype_a, 'shape'):
            dtype_a = TensorType(shape=dtype_a.shape, element_type=dtype_a.element_type)
        else:
            # If it's the class itself, we unify it to TensorType the class
            dtype_a = TensorType

    # If dtype_b is a MatrixType or a subclass of NDimDataType => treat it as TensorType
    if isinstance(dtype_b, MatrixType) or \
       (isinstance(dtype_b, type) and issubclass(dtype_b, CType) and dtype_b is not CType):
        if not isinstance(dtype_b, type) and hasattr(dtype_b, 'shape'):
            dtype_b = TensorType(shape=dtype_b.shape, element_type=dtype_b.element_type)
        else:
            dtype_b = TensorType

    # Now proceed with normal checks
    if dtype_a == dtype_b:
        return True

    if dtype_a == Dyn or dtype_b == Dyn:
        return True

    # If expected is a list or tuple, check each element
    if isinstance(dtype_a, (list, tuple)):
        return any(is_consistent_data_type(e, dtype_b) for e in dtype_a)

    # TensorType vs list => consider them consistent
    if dtype_a == TensorType:
        if dtype_b == list or (hasattr(dtype_b, "__origin__") and dtype_b.__origin__ == list):
            return True
    if dtype_b == TensorType:
        if dtype_a == list or (hasattr(dtype_a, "__origin__") and dtype_a.__origin__ == list):
            return True

    # Case 1: Both are DataType classes (not instances)
    if (isinstance(dtype_a, type) and issubclass(dtype_a, DataType)) \
       and (isinstance(dtype_b, type) and issubclass(dtype_b, DataType)):
        # If they're literally the same class, OK
        if dtype_a is dtype_b:
            return True
        # If both are "some NDimDataType" or child => consider that consistent
        if issubclass(dtype_a, CType) and issubclass(dtype_b, CType):
            return True
        # else fallback
        return False

    # Case 2: Expected is a DataType class, actual is an instance
    if isinstance(dtype_a, type) and issubclass(dtype_a, DataType):
        return isinstance(dtype_b, dtype_a)

    # Or the reverse
    if isinstance(dtype_b, type) and issubclass(dtype_b, DataType):
        return isinstance(dtype_a, dtype_b)

    # Case 3: If both are TensorType instances, check shape consistency
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
        raise ValueError(f"Invalid shape value: {v}")
