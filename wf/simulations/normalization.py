import inspect
import re
import types
from typing import Optional, Tuple,Dict, Any, Union,Callable
from dataclasses import dataclass

from numpy import inf as INFINITY

import copy

from .utilities import all_dict1_vals_in_dict2_vals


from .utilities import InitError
from ...assert_checks import gen_mismatch_dict
from torch.fx.operator_schemas import (
      
        OpOverload,
        OpOverloadPacket,
        type_matches,
        _args_kwargs_to_normalized_args_kwargs
    )
from ...util.log import logging
logger = logging.getLogger(__name__)

# from workflow.simulation.refactor.process import Process

def build_inspect_signature_from_process_signature(signature: "Signature") -> inspect.Signature:
    """
    Convert the left-flow portion of `signature` into a Python Signature object:
      - Non-variadic specs => normal positional params
      - If exactly one is variadic => we represent it with a *args param
      - We skip specs that are purely flow=RIGHT, so we don't demand them at call-time
    """
    params = []
    logger.debug(f"Signature: {signature}")
    # for reg in signature:
    #     if reg.flow & Flow.LEFT:
    #         logger.debug("reg.flow & Flow.LEFT: ",reg.flow & Flow.LEFT)
    #     logger.debug(reg.flow)
    left_specs = [s for s in signature if s.flow & Flow.LEFT]
    logger.debug(f"left_specs: {left_specs}")
    # If you allow multiple variadic specs, you'd need more advanced logic.
    # We assume at most one is variadic for now.
    variadic_seen = False

    for i, spec in enumerate(left_specs):
        if not spec.variadic:
            # a normal positional parameter
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
    logger.debug(f"\n=== Entering normalize_process_call where force_kwarg_mode={force_kwarg_mode} ===")
    from torch.fx.operator_schemas import get_signature_for_torch_op
    

    if kwargs is None:
        kwargs = {}
    normalized_data: Dict[str, Any] = {}
  
    from torch._library.infer_schema import infer_schema
    # logger.debug(f'target: {target}\n - args: {args}\n - types: {arg_types}\n - to bind: {args_for_bind}')
    logger.debug(f'target type: {type(target)}\n - args: {args}\n - types: {arg_types}')
    # (1) If target is a Process => build an inspect.Signature from process.signature
    if isinstance(target, Process):
        fx_sig = build_inspect_signature_from_process_signature(target.signature)
        # Debug: print the signature and its parameters
        logger.debug("\nGenerated inspect.Signature for process:")
        logger.debug(f"fx_sig: {fx_sig}")
        # logger.debug("Parameters:")
        # for name, param in fx_sig.parameters.items():
        #     logger.debug(f"  {name}: {param}")
        param_names = [p.name for p in target.signature.lefts()]
        logger.debug(f"Process {target} with param names: {param_names}")
        args_for_bind = list(args)
        logger.debug(f"args_for_bind: {args_for_bind}")
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
        # Additional typed checks using the process's signature
        # e.g. 'signature.validate_data_with_register_specs(inputs)'
        # or your own logic with regspec.matches_data
        # for k,v in normalized_data.items():
        #     logger.debug(f"{k}: {v} of type {type(v)}")
       
        target.signature.validate_data_with_register_specs(
            [normalized_data[p.name] for p in target.signature.lefts()]
        )
        logger.debug(f"target falls into branch (1) since subclass of Process. Resulting normalized keys: {normalized_data.keys}")
   
    # Branch 3: Torch operators / built-in functions.
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
    logger.debug(f"Final normalized result: {normalized_data}")
    return normalized_data
from torch._jit_internal import boolean_dispatched


# Helper for torch ops.
def _normalize_torch_op(
    target: Callable,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    arg_types: Optional[Tuple[Any, ...]],
    kwarg_types: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Normalizes a torch op (or built-in function) using candidate schemas.
    If multiple schemas match, uses provided type hints for disambiguation,
    or attempts to use infer_schema as a fallback.
    """
    import inspect
    from torch.fx.operator_schemas import get_signature_for_torch_op, type_matches, _args_kwargs_to_normalized_args_kwargs
    torch_op_sigs = get_signature_for_torch_op(target)
    logger.debug("\nGenerated inspect.Signature for torch op:")
    logger.debug(f"fx_sig: {torch_op_sigs}")
    if not torch_op_sigs:
        raise ValueError(f"No signature found for PyTorch op: {target}")
    matched_schemas = []
    for candidate_sig in torch_op_sigs:
        try:
            candidate_sig.bind(*args, **kwargs)
            matched_schemas.append(candidate_sig)
        except TypeError:
            continue
    if len(matched_schemas) == 0:
        raise ValueError(f"No valid overload for {target} with arguments={args}, kwargs={kwargs}")
    elif len(matched_schemas) == 1:
        chosen_sig = matched_schemas[0]
    else:
        # Multiple matches: try type-hint disambiguation if provided.
        if arg_types is not None or kwarg_types is not None:
            arg_types = arg_types if arg_types else ()
            kwarg_types = kwarg_types if kwarg_types else {}
            filtered = []
            for candidate_sig in matched_schemas:
                try:
                    bound_type_check = candidate_sig.bind(*arg_types, **kwarg_types)
                except TypeError:
                    continue
                all_good = True
                for name, user_type in bound_type_check.arguments.items():
                    param = candidate_sig.parameters[name]
                    if param.annotation is not inspect.Parameter.empty:
                        if not type_matches(param.annotation, user_type):
                            all_good = False
                            break
                if all_good:
                    filtered.append(candidate_sig)
            if len(filtered) == 0:
                raise ValueError(f"Could not find a matching schema for {target} even after type-based disambiguation. arg_types={arg_types} kwarg_types={kwarg_types}")
            elif len(filtered) > 1:
                # If the string representations are identical, choose the first.
                rep = str(filtered[0])
                if all(str(s) == rep for s in filtered):
                    chosen_sig = filtered[0]
                else:
                    raise ValueError(f"Still ambiguous: multiple overloads match the provided arg_types={arg_types} and kwarg_types={kwarg_types} for {target}. Overloads: {filtered}")
            else:
                chosen_sig = filtered[0]
        else:
            # Fallback: use infer_schema for disambiguation.
            try:
                from torch._library.infer_schema import infer_schema
                op_name = getattr(target, '__name__', None) or "unknown"
                inferred = infer_schema(target, op_name=op_name, mutates_args=())
                logger.debug(f"Inferred schema: {inferred}")
                filtered = [s for s in matched_schemas if str(s) == inferred]
                if len(filtered) == 1:
                    chosen_sig = filtered[0]
                else:
                    schema_printouts = "\n".join(str(s) for s in matched_schemas)
                    raise RuntimeError(f"Ambiguous schema after using infer_schema. Please provide explicit argument types. Available schemas:\n{schema_printouts}")
            except Exception as e:
                logger.debug(f"Error using infer_schema for disambiguation: {e}")
                raise RuntimeError("Multiple matching schemas found and could not disambiguate.") from e
        bound = chosen_sig.bind(*args, **kwargs)
        bound.apply_defaults()
        normalized_data = dict(bound.arguments)
        return normalized_data
    bound = matched_schemas[0].bind(*args, **kwargs)
    bound.apply_defaults()
    return dict(bound.arguments)
from torch.fx.operator_schemas import ArgsKwargsPair
# In-house helper analogous to the torch.fx version.
def _args_kwargs_to_normalized_args_kwargs(
    sig: inspect.Signature,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    normalize_to_only_use_kwargs: bool,
) -> Optional[ArgsKwargsPair]:
    """
    Given a signature, args, and kwargs, returns an ArgsKwargsPair after binding and applying defaults.
    Supports only POSITIONAL_OR_KEYWORD and KEYWORD_ONLY parameters.
    """
    supported_parameter_types = {
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    }
    if any(p.kind not in supported_parameter_types for p in sig.parameters.values()):
        if list(sig.parameters.keys()) != ["input", "from", "to", "generator"]:
            return None
    bound_args = sig.bind(*args, **kwargs)
    bound_args.apply_defaults()
    new_kwargs: Dict[str, Any] = {}
    new_args: list[Any] = []
    for i, param in enumerate(sig.parameters):
        if not normalize_to_only_use_kwargs and i < len(args):
            new_args.append(bound_args.arguments[param])
        else:
            new_kwargs[param] = bound_args.arguments[param]
    
    return ArgsKwargsPair(tuple(new_args), new_kwargs)
