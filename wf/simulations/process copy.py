# -*- coding: utf-8 -*-
""" This file contains class definitions for the base Process
class that specifies how individual processes/tasks are to 
be executed/updated.

"""
import inspect
import re

from typing import List, TYPE_CHECKING, Optional, Tuple,Dict, Any, Union,Callable
from dataclasses import dataclass

from numpy import inf as INFINITY

import copy

from .utilities import all_dict1_vals_in_dict2_vals
from .unification_tools import is_consistent_data_type
from .data import Data,DataSpec
from .dtypes import DataType,TensorType, MatrixType, NDimDataType
from .register import RegisterSpec,Flow,Signature
from .utilities import InitError
from ...assert_checks import gen_mismatch_dict
if TYPE_CHECKING:

    from .data import Data, Result, DataSpec
    from .register import RegisterSpec,Flow,Signature

    from .resources.resources import Resource
    from .resources.quantum_resources import QuantumAllocation
    from .resources.classical_resources import ClassicalAllocation
    from .quantum import QuantumCircuit
    
from torch.fx.operator_schemas import type_matches
import types
from attrs import define, field
from ...util.log import logging
logger = logging.getLogger(__name__)


def parse_metadata(properties: Optional[Dict[str, Any]]) -> DataSpec:
    """
    Convert a raw properties dict into a typed DataSpec instance.
    - Normalize 'Usage' key (case-insensitive, formatting).
    - Infer 'Data Type' if missing.
    """
    usage = properties.get("Usage", None)
    dtype = properties.get("Data Type", None)

    # Normalize usage (capitalize for consistency)
    usage = usage.strip().capitalize() if usage else None

    # Infer Data Type if missing
    if dtype is None and "data" in properties:
        dtype = type(properties["data"])  # Infer from the `data` attribute
    elif isinstance(dtype, type):
        dtype = dtype  # Leave it as-is if it's already a type
    elif isinstance(dtype, DataType):
        dtype = dtype.__class__  # Extract the type from the DataType object
    else:
        dtype = None

    return DataSpec(
        usage=usage,
        data_type=dtype,  # Keep this as a type object
        extra={k: v for k, v in properties.items() if k not in {"Usage", "Data Type"}},
    )



from typing import NamedTuple
import torch
from torch.fx.operator_schemas import (
      
        OpOverload,
        OpOverloadPacket,
        type_matches,
        _args_kwargs_to_normalized_args_kwargs
    )


def get_signature_for_process_op(process: Callable, return_schemas: bool = True):
    """
    Source: taken from torch.fx.operator_schemas.py (get_signature_for_torch_op(...) function)
    
    torch docstring: Given an operator on the `torch` namespace, return a list of `inspect.Signature`
    objects corresponding to the overloads of that op.. May return `None` if a signature
    could not be retrieved.

    Args:
        op (Callable): An operator on the `torch` namespace to look up a signature for

    Returns:
        Optional[List[inspect.Signature]]: A list of signatures for the overloads of this
            operator, or None if the operator signatures could not be retrieved. If
            return_schemas=True, returns a tuple containing the optional Python signatures
            and the optional TorchScript Function signature
    """
    if isinstance(op, OpOverload):
        schemas = [op._schema]
    elif isinstance(op, OpOverloadPacket):
        schemas = [getattr(op, overload)._schema for overload in op.overloads()]
    else:
        override = _manual_overrides.get(op)
        if override:
            return (override, None) if return_schemas else None

        aten_fn = torch.jit._builtins._find_builtin(op)

        if aten_fn is None:
            return (None, None) if return_schemas else None
        schemas = torch._C._jit_get_schemas_for_operator(aten_fn)

    signatures = [_torchscript_schema_to_signature(schema) for schema in schemas]
    return (signatures, schemas) if return_schemas else signatures

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
                print(f"Inferred schema: {inferred}")
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
    #         print("reg.flow & Flow.LEFT: ",reg.flow & Flow.LEFT)
    #     logger.debug(reg.flow)
    left_specs = [str(s) for s in signature if s.flow & Flow.LEFT]
    print(f"left_specs: {left_specs}")
    print(f"right_specs: {[str(s) for s in signature if s.flow & Flow.RIGHT]}")
    # If you allow multiple variadic specs, you'd need more advanced logic.
    # We assume at most one is variadic for now.
    variadic_seen = False

    for i, spec in enumerate(signature.lefts()):
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
    print(f"\n=== Entering normalize_process_call where force_kwarg_mode={force_kwarg_mode} ===")
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
        # Debug: logger.debug the signature and its parameters
        # logger.debug("\nGenerated inspect.Signature for process:")
        # logger.debug(f"fx_sig: {fx_sig}")
        # logger.debug("Parameters:")
        # for name, param in fx_sig.parameters.items():
        #     logger.debug(f"  {name}: {param}")
        param_names = [p.name for p in target.signature.lefts()]
        # logger.debug(f"Process {target} with param names: {param_names}")
        args_for_bind = list(args)
        # logger.debug(f"args_for_bind: {args_for_bind}")
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
        logger.debug(f"target falls into branch (1) since subclass of Process. Resulting normalized keys: {normalized_data.keys()}")

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
from torch._jit_internal import boolean_dispatched





def _decompose_helper(process: 'Process'):
    from .builder import ProcessBuilder

    builder, initial_ports = ProcessBuilder.from_signature(process.signature, immutable=False)
    out_ports = process.build_composite(builder=builder, **initial_ports)
    return builder.finalize(**out_ports)
# @define
class Process:
    """Abstract base class for process models.

    This abstract base class lays out the basic functionality
    for basic process classes which will inherit from Process and
    are wrapped by Node.

    This class is responsible for:
    1) Storing `inputs` (list of `Data`) and matching them to `expected_input_properties`.
    2) Building or inferring `signature` (left, right data).
    3) Validating input data properties. If mismatches occur, it raises `InitError`.
    
    [Green] - FULL FREEDOM
    
    Legacy (required) attributes: Constructing a Process requires inheriting from this base class
    and constructing the required methods:
        - "set_expected_input_properties"
        - "set_required_resources"
        - "set_output_properties"
        - "compute_flop_count"
        - "validate_data"
        - "update"
        - "generate_output" 

    
    Notes from legacy (pre- signature/RegisterSpec): 
    - the attributes output_properties are only used when connecting 
    the output Nodes of one network with the input Nodes of another network. In most 
    cases they won't be necessary if the Node/Process are neither input/output 
    Nodes of a larger Network. It will be useful to include nevertheless for reuseability.  
    """

    UNSTARTED = "UNSTARTED"
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"

    def __init__(
        self,
        **kwargs,
    ):
        
        self.inputs: Optional[List[Data]] = kwargs.get("inputs", None)
        logger.debug(f"{self} initialized with inputs: {self.inputs}")
        
        self.required_resources: Optional[Any] = kwargs.get("required_resources", [])
        self.output_properties: Optional[List[Dict[str, Any]]]  = kwargs.get("output_properties", [])
        self.expected_input_properties: Optional[List[Dict[str, Any]]]  = kwargs.get("expected_input_properties", [])
    
        self.dynamic: bool = kwargs.get("dynamic", False)
        self.status: str = Process.UNSTARTED

        self.time_till_completion: float = float("inf")
        #     self._validate_inputs()
        self._signature = None
        # self.input_data = {}
        # self.shape_env = kwargs.get("shape_env", ShapeEnv())

        # Make sure we have some default expected and output properties
        if not self.expected_input_properties:
            try:
                self.set_expected_input_properties()
            except NotImplementedError:
                self.expected_input_properties = self._derive_input_props_from_signature()
        if not self.output_properties:
            try:
                self.set_output_properties()
            except NotImplementedError:
                self.output_properties = self.infer_output_properties()



        self.result = None
        
        
       
        # If user provided inputs, do the usual validations
        if self.inputs:
            from rich.pretty import pretty_repr
      
            self.input_data = self._build_legacy_input_mapping(self.inputs)
            logger.debug("Legacy self.input_data automatically generated:")
            
            logger.info(pretty_repr(self.input_data))
            
           
            self.normalized_map = self._build_normalized_mapping(self.inputs)
            logger.debug("\nNormalized mapping (normalize_process_call() return) dict:")
            logger.info(pretty_repr(self.normalized_map))
            

            self.normalized_inputs = self._build_ordered_normalized_inputs(self.signature, self.normalized_map, self.inputs)
            # logger.debug("\nNormalized input (self.normalized_inputs) list:")
            # logger.info(pretty_repr(self.normalized_inputs))
            self._build_mappings()
        


            if not self.validate_data_properties_new():
                raise InitError(
                    f"""Input data properties {[input.properties for input in self.inputs]} 
                do not satisfy expected properties: {self.expected_input_properties} 
                for process {self.__class__.__name__}"""
                )
            # if not self.validate_data_properties_original():
            #     raise InitError(
            #         f"""Input data properties {[input.properties for input in self.inputs]} 
            #     do not satisfy expected properties: {self.expected_input_properties} 
            #     for process {self.__class__.__name__}"""
            #     )

         

            # If no resources, let child define
            if not self.required_resources:
                self.set_required_resources()
        else:
            self.input_data = {}

        # logger.debug(f"{self} signature: {self.signature}")
       
        # logger.debug(f'{self} initialized (inputs: {self.inputs} with signature: {self.signature}')

    @property
    def signature(self):
        """
        Builds or returns a cached `Signature` from self.inputs and self.output_properties.
        If no inputs, we build from expected_input_properties + output_properties.
        """
        
        from .register import Signature


        if self._signature is not None:
            return self._signature
        

        if self._signature is None:
            logger.debug(f"\n===  Initializing `{self}` Signature === ")
            if self.inputs and not self.expected_input_properties:
                logger.debug(f"Defining signature from inputs: {self.inputs}. Any expected inputs? {self.expected_input_properties}")
                self._signature = Signature.build_from_data(self.inputs, self.output_properties)
            else:
                self._signature = Signature.build_from_properties(
                    input_props=self.expected_input_properties,
                    output_props=self.output_properties
                )
       
        return self._signature
    
    # --- Helper methods for input mapping and logging ---
    def _build_mappings(self):
        # Build additional mappings: data_id_to_spec and usage_to_spec
        self.input_id_to_spec, self.spec_to_input_id = {}, {}
        self.usage_to_spec,self.spec_to_usage = {},{}
        for arg, data_obj in self.normalized_map.items():
            self.input_id_to_spec[data_obj.id] = arg
            self.spec_to_input_id[arg] = data_obj.id
            usage = data_obj.properties.get("Usage")
            if usage:
                self.usage_to_spec[usage] = arg
                self.spec_to_usage[arg] = usage
            else:
                # this might cause issues...
                # self.usage_to_spec[data_obj.metadata.hint] = arg
                self.spec_to_usage[arg] = data_obj.metadata.hint
        
        description = (
                f"{self} initialized with the following map attributes:\n"
                f" - input_id_to_spec: {self.input_id_to_spec} \n"
                f" - spec_to_input_id: {self.spec_to_input_id}\n"
                f" - usage_to_spec: {self.usage_to_spec}\n"
                f" - spec_to_usage: {self.spec_to_usage}\n"
            )
        logger.debug(description)


    def _build_legacy_input_mapping(self, inputs: List[Data]) -> Dict[str, Any]:
        mapping = {}
        for data in inputs:
            usage = data.properties.get("Usage")
            if usage is not None:
                if usage == "Reduction Row":
                    key = usage
                elif "Row Idx" in data.properties:
                    key = f"{usage} {data.properties['Row Idx']}"
                else:
                    key = usage
            else:
                key = data.id
            mapping[key] = data.data
        return mapping
    
    def _build_normalized_mapping(self, inputs: List[Data]) -> Dict[str, Data]:
        # Here we assume normalize_process_call() exists and returns a mapping by signature register names.
        from .process import normalize_process_call
        return normalize_process_call(target=self, args=tuple(inputs), kwargs={})

    def _build_ordered_normalized_inputs(self, signature: Signature, norm_map: Dict[str, Data], inputs: List[Data]) -> List[Data]:
        ordered = []
        for reg in signature.lefts():
            if reg.name in norm_map:
                # logger.debug(f'{reg.name} in mapping, adding to ordered list')
                ordered.append(norm_map[reg.name])
            else:
                logger.debug(f'{reg.name} not in mapping, attempting fallback')
                # Fallback: search in inputs by matching 'Usage'
                fallback = next((d for d in inputs if d.properties.get("Usage") == reg.name), None)
                if fallback:
                    ordered.append(fallback)
                else:
                    ordered.append(None)
                    logger.warning(f"Normalized mapping missing key for register {reg.name}")
        return ordered

    
    def infer_output_properties(self):
        """
        If no explicit output_properties are set, try building from the signature’s RIGHT data.
        """
        if self.signature:
            # if not self.inputs:
            #     self.inputs = list(self.signature.lefts())
            # logger.debug(f"inputs: {list(self.signature.lefts())}")
            output_signature = self.signature.rights()
            ret = []
            for s in output_signature:
                ret.append({"Data Type": s.dtype})
            return ret
        else:
            return []
    
   
    def _derive_input_props_from_signature(self) -> List[Dict[str, Any]]:
        """
        Build a dictionary-based expected_input_properties from
        the signature's left-flow RegisterSpecs. If no signature, return [].
        """
        if self.signature is None:
            return []
        logger.debug(f'{self} signature: {self.signature}')
        input_props = []
        for regspec in self.signature.lefts():
            print(regspec)
            # Example: "Data Type" => regspec.dtype
            # "Usage" => we could store regspec.name or something else
            prop_dict = {
                "Data Type": regspec.dtype,
             
            }
            input_props.append(prop_dict)
        logger.debug(f' - Infered input props from sig: {input_props}')
        return input_props
    

    ## --------------------------- VALIDATION ---------------------------------- ##
    def validate_data_properties_new(self) -> bool:
        """
        Validate that each left-register in the process signature matches its corresponding input Data object.
        
        This method relies on the Signature – which is composed of RegisterSpec objects –
        and the ordering produced by _build_ordered_normalized_inputs. For each RegisterSpec 
        (on the LEFT flow), it checks that the corresponding Data object's metadata.dtype and shape 
        are consistent with the RegisterSpec (using RegisterSpec.matches_data()).
        
        If any mismatch is detected, an InitError is raised with details. Otherwise, returns True.
        """
        # Build the ordered list of normalized inputs from the current signature and normalized mapping.
        normalized_inputs = self._build_ordered_normalized_inputs(self.signature, self.normalized_map, self.inputs)
        left_specs = list(self.signature.lefts())
        
        # Ensure the number of normalized inputs matches the number of left-side registers.
        if len(normalized_inputs) != len(left_specs):
            raise InitError(
                f"Mismatch in number of inputs: found {len(normalized_inputs)} normalized inputs but "
                f"expected {len(left_specs)} registers in the signature."
            )
        
        errors = []
        # Iterate over each left-register and compare it with the corresponding input.
        for idx, spec in enumerate(left_specs):
            data_obj = normalized_inputs[idx]
            if data_obj is None:
                errors.append(f"Input for register '{spec.name}' is missing.")
            elif not spec.matches_data(data_obj):
                errors.append(
                    f"Input for register '{spec.name}' (Data ID: {data_obj.id}) does not match the expected type.\n"
                    f"  - Expected: {spec.dtype} with wire shape {spec.shape}\n"
                    f"  - Got: {data_obj.metadata.dtype} with data shape {data_obj.metadata.shape}"
                )
        if errors:
            full_error = "\n".join(errors)
            logger.error("Data property validation failed:\n%s", full_error)
            raise InitError("Data property validation failed:\n" + full_error)
        return True
   

    def _raise_data_property_error(self, mismatches: Dict[str, Dict[str, Dict[str, Any]]]):
        """
        Raises an InitError with a detailed error message about mismatched data properties.
        """
        error_lines = ["Input data does not satisfy expected properties :"]
        
        for input_id, details in mismatches.items():
            error_lines.append(f"\n- {input_id}:")
            for prop, mismatch in details.items():
                error_lines.append(f"    Property '{prop}': Expected {mismatch['expected']}, Actual {mismatch['actual']}")

        raise InitError("\n".join(error_lines))



    # --- Abstract methods to be implemented by subclasses ---
    def validate_data(self) -> bool:
        """Process specific verification that ensures input data has
        correct specifications. Will be unique to each process
        """

        raise NotImplementedError
    def validate_inputs(self):
        pass
    def set_required_resources(self):
        """Sets the appropriate resources required to run the process. This method should update
        self.required_resources to be a list of Resources required for the Process to successfully run.
        Will be unique for each process.

        Default: No resources required!
        """

        self.required_resources = []

    def set_expected_input_properties(self):
        """Sets the expected input properties of the Process to execute/update successfully.
        This method should update self.expected_input_properties to be a list of dictionaries, with each
        dictionary specifying the expected properties of an input Data instance. A Process that requires
        N inputs will require N such dictionaries specifying its properties. Will be unique to each Process.
        """

        raise NotImplementedError

    def set_output_properties(self):
        """Sets the expected properties of the generated outputs of the Process.
        This method should update self.output_properties to be a list of dictionaries,
        with each dictionary specifying the expected properties of each output Data
        """

        raise NotImplementedError
    def update(self):

        raise NotImplementedError

    def generate_output(self) -> List["Data"]:
        """After completing updates, the process will wrap its results
        and return a (unordered) set of Data states to be processed by Node.
        The expected output will be unique to the process.
        """

        raise NotImplementedError

    def set_required_resources(self):
        """Default: No resources required."""
        self.required_resources = []

    def decompose(self):
        return _decompose_helper(self)
    
    def build_composite(self, builder, **ports):
        raise NotImplementedError(f"{self} has no decomposition.")
    
    def as_composite(self):
        from .builder import ProcessBuilder
        builder, init_ports = ProcessBuilder.from_signature(self.signature)
        # logger.debug(f"As composite operation. Initial ports: {init_ports}")
        return builder.finalize(**builder.add_d(self,**init_ports))

    def __repr__(self):
        return self.__class__.__name__
    @property
    def describe(self):
        
        if self.status == Process.COMPLETED:
            description = (
                f"Process: {self}\n"
                # f" - inputs: {self.inputs} \n - outputs: {self.generate_output()}\n"
                f" - inputs: {self.inputs} \n"
                f" - status: {self.status}\n"
                f" - dyn: {self.dynamic}\n"
                f" - signature: {self.signature}\n"
            )
        else:
            description = (
                f"Process: {self}\n"
                f" - inputs: {self.inputs} \n - outputs: {[]}\n"
                f" - status: {self.status}\n"
                f" - dyn: {self.dynamic}\n"
                f" - signature: {self.signature}\n"
            )

        logger.debug(description)
        # print(description)
    def log_legacy_vs_normalized(self):
        """
        Compare the 'legacy' dictionary self.input_data vs. the new signature-based
        mapping self.normalized_map. Logs them side by side for clarity.

        This updated version first tries to match data by usage:
        1) Build a dict { usage -> param_name } from self.normalized_map, using data_obj.properties["Usage"].
        2) For each usage in self.input_data, if usage is found in that dict, we log a direct match.
        3) If usage is not found or missing, we do a fallback shape-based approach for array-like data.

        This means that scalar entries like "Column Idx" or "Pivot Idx" will now match
        the normalized param if both have the same usage string, eliminating the
        "No direct match found" messages for those scalars.
        """
        logger.debug("Comparing legacy input_data vs. normalized_map:\n")

        # 1) Build a usage->param_name map from normalized_map
        usage_to_param = {}
        for param_name, data_obj in self.normalized_map.items():
            usage = data_obj.properties.get("Usage")
            if usage:
                usage_to_param[usage] = param_name

        # 2) Now log each item in self.input_data with potential matches in normalized_map
        for usage, raw_data in self.input_data.items():
            logger.debug(f"Legacy: Usage '{usage}' -> raw_data type: {type(raw_data)}, shape: {getattr(raw_data, 'shape', None)}")
            # if usage in usage_to_param:
            #     logger.debug(f"   Normalized mapping: parameter '{usage_to_param[usage]}' holds the data with Usage '{usage}'.")
            # else:
            #     logger.debug("   No normalized mapping found for this usage.")
            shape_info = getattr(raw_data, 'shape', None)
            # logger.debug(f"  usage='{usage}' -> raw_data type: {type(raw_data)}, shape={shape_info}")

            # First try usage-based matching:
            if usage in usage_to_param:
                param_name = usage_to_param[usage]
                # logger.debug(f"    usage-based match => normalized param '{param_name}'")
                logger.debug(f"   usage-based match => parameter '{usage_to_param[usage]}' holds the data with Usage '{usage}'.")
                continue

            # 3) If usage-based matching didn’t succeed or usage is missing,
            #    fallback to shape-based matching for array-like data
            #    (only if raw_data has a shape attribute).
            if shape_info is not None:
                # Attempt to find a param referencing the same shape
                potential_params = []
                for param_name, data_obj in self.normalized_map.items():
                    nm_shape = getattr(data_obj.data, 'shape', None)
                    if nm_shape == shape_info:
                        potential_params.append(param_name)

                if potential_params:
                    logger.debug(f"    shape-based match => param(s): {potential_params}")
                else:
                    logger.debug(f"    No direct match found by shape or usage.")
            else:
                logger.debug(f"    No direct match found by usage or shape (scalar?).")

        logger.debug("\nDone comparing input_data vs. normalized_map.\n")

class ClassicalProcess(Process):
    def __init__(
        self,
        inputs=None,
        expected_input_properties=None,
        required_resources=None,
        output_properties=None,
        dynamic=False,
        shape_env=None, # Add this line
    ):
        super().__init__(
            inputs=inputs,
            expected_input_properties=expected_input_properties,
            required_resources=required_resources,
            output_properties=output_properties,
            dynamic=dynamic,
            shape_env=shape_env  # Pass it along
        )
        if inputs != None:
            self.flops = self.compute_flop_count()

    def compute_flop_count(self) -> int:
        """Computes the number of required flops for this process to complete given
        valid inputs from self.inputs. Will be unique to each Process.
        """

        raise NotImplementedError

    def _compute_classical_process_update_time(
        self, allocation: "ClassicalAllocation"
    ) -> float:
        # assert isinstance(
        #     allocation, ClassicalAllocation
        # ), f"Expected classical allocation for process {self}"

        time_till_completion = self.flops / allocation.clock_frequency
        return time_till_completion
    

class QuantumProcess(Process):
    def __init__(
        self,
        inputs=None,
        expected_input_properties=None,
        required_resources=None,
        output_properties=None,
        dynamic=False,
    ):
        super().__init__(
            inputs=inputs,
            expected_input_properties=expected_input_properties,
            required_resources=required_resources,
            output_properties=output_properties,
            dynamic=dynamic,
        )

        self.mps: MatrixProductState = None
        if inputs != None:
            self.circuit: "QuantumCircuit" = self.compute_circuit()
            # self.__get_measure_results()

    def compute_circuit(self) -> int:
        """Computes the number of required flops for this process to complete given
        valid inputs from self.inputs. Will be unique to each Process.
        """

        raise NotImplementedError

    def __get_measure_results(self):
        self.measure_results = dict()
        for instruction in self.circuit.instructions:
            if instruction.gate.name == "MEASURE":
                qubit_idx = instruction.gate_indices[0]
                self.measure_results[qubit_idx] = self.__measure(qubit_idx)

    def __measure(self, qubit_idx):
        if self.mps is None:
            self.__build_mps()

        return self.mps.measure(site=qubit_idx, get="outcome", inplace=True)

    def __build_mps(self):
        pass

    def _compute_quantum_process_update_time(
        self, allocation: "QuantumAllocation"
    ) -> float:
        # assert isinstance(
        #     allocation, QuantumAllocation
        # ), f"Expected quantum allocation for process {self}"

        time_till_completion = {
            qubit_idx: 0 for qubit_idx in allocation.allocated_qubit_idxs
        }
        for instruction in allocation.transpiled_circuit.instructions:
            gate_idxs = instruction.gate_indices
            assert all(
                gate_idx in allocation.allocated_qubit_idxs for gate_idx in gate_idxs
            ), f"Gate indices {gate_idxs} of instruction {instruction.gate} is not within allocated gate indices {allocation.allocated_qubit_idxs}"

            if instruction.gate.name == "MEASURE":
                assert (
                    len(gate_idxs) == 1
                ), f"Measure instruction should only act on one qubit"
                time_till_completion[
                    gate_idxs[0]
                ] += allocation.device_connectivity.nodes[gate_idxs[0]][
                    "measurement duration"
                ]

            elif len(gate_idxs) == 1:
                time_till_completion[
                    gate_idxs[0]
                ] += allocation.device_connectivity.nodes[gate_idxs[0]][
                    "1Q gate times"
                ][
                    instruction.gate.name
                ]

            else:
                gate_duration = allocation.device_connectivity[gate_idxs[0]][
                    gate_idxs[1]
                ][f"{instruction.gate.name} duration"]
                time_till_completion[gate_idxs[0]] += gate_duration
                time_till_completion[gate_idxs[1]] += gate_duration

        return max(time_till_completion.values())

    def _compute_sucess_probability(self, allocation: "QuantumAllocation") -> float:
        # assert isinstance(
        #     allocation, QuantumAllocation
        # ), f"Expected quantum allocation for process {self}"

        success_probability = 1.0
        for instruction in allocation.transpiled_circuit.instructions:
            gate_idxs = instruction.gate_indices
            if instruction.gate.name == "MEASURE":
                assert (
                    len(gate_idxs) == 1
                ), f"Measure instruction should only act on one qubit"
                success_probability *= (
                    1
                    - allocation.device_connectivity.nodes[gate_idxs[0]][
                        "measurement duration"
                    ]
                )

            elif len(gate_idxs) == 1:
                success_probability *= (
                    1
                    - allocation.device_connectivity.nodes[gate_idxs[0]][
                        "1Q gate errors"
                    ][instruction.gate.name]
                )

            else:
                success_probability *= (
                    1

                    - allocation.device_connectivity[gate_idxs[0]][gate_idxs[1]][
                        f"{instruction.gate.name} error"
                    ]
                )

        return success_probability

