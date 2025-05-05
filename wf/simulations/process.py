# -*- coding: utf-8 -*-
""" This file contains class definitions for the base Process
class that specifies how individual processes/tasks are to 
be executed/updated.

"""
from rich.pretty import pretty_repr
from typing import List, TYPE_CHECKING, Optional, Tuple,Dict, Any, Union,Callable
from numpy import inf as INFINITY
from quimb.tensor import MatrixProductState

import copy
from attrs import define
from .utilities import all_dict1_vals_in_dict2_vals
from .data import Data,ShapeEnv
from .schema import Flow,Signature
from .unification_tools import build_inspect_signature_from_process_signature
if TYPE_CHECKING:
    from .data import Data,Flow,Signature,ShapeEnv
    from .resources.resources import Resource
    from .resources.quantum_resources import QuantumAllocation
    from .resources.classical_resources import ClassicalAllocation
    from .quantum import QuantumCircuit


from ...util.log import logging

logger = logging.getLogger(__name__)


class InitError(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

def _decompose_helper(process: 'Process'):
    from .builder import ProcessBuilder

    builder, initial_ports = ProcessBuilder.from_signature(process.signature, immutable=False)
    out_ports = process.build_composite(builder=builder, **initial_ports)
    return builder.finalize(**out_ports)

@define
class Process:
    """Abstract base class for process models.

    This abstract base class lays out the basic functionality
    for basic process classes which will inherit from Process and
    are wrapped by Node in graph.py.

    """

    UNSTARTED = "UNSTARTED"
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"

    def __init__(self, **kwargs):
        self.inputs: list[Data] = kwargs.get("inputs", None)
        
        self.required_resources: list[Resource] = kwargs.get("required_resources", None)
        self.output_properties: list[dict] = kwargs.get("output_properties", None)
        self.expected_input_properties: list[dict] = kwargs.get("expected_input_properties", []) 
        self.dynamic: bool = kwargs.get("dynamic", False)
        self.time_till_completion: float = float("inf")
        self.status: str = Process.UNSTARTED
        self.shape_env = kwargs.get("shape_env", ShapeEnv())

        self._signature = None


        if not self.expected_input_properties:
            self.set_expected_input_properties()
        if not self.output_properties:
            try:
                self.set_output_properties()
                # logger.debug(f"Output props provided by Process: {self.output_properties}")
            except NotImplementedError:
                self.output_properties = self.infer_output_properties()
                # logger.debug(f"Infering output from signature: {self.output_properties}")
        
        logger.debug(f'{self} provided inputs: {self.inputs}')

        if self.inputs != None:
            if not hasattr(self, 'input_data'):
                self.input_data = self._build_legacy_input_mapping(self.inputs)
                logger.debug("Legacy self.input_data automatically generated:")
            
                logger.info(pretty_repr(self.input_data))
            # self.normalized_map = self._build_normalized_mapping(self.inputs)
            # logger.debug("\nNormalized mapping (normalize_process_call() return) dict:")
            # logger.info(pretty_repr(self.normalized_map))
            if not self.validate_data_properties():
                raise InitError(
                    f"""Input data properties {[input.properties for input in self.inputs]} 
                does not satisfy expected properties: {self.expected_input_properties} 
                for process {self.__class__.__name__}"""
                )
    

            if not self.required_resources:
                self.set_required_resources()
    # @property
    # def signature(self):
    #     from .schema import Signature
    #     if self._signature is not None:
    #         return self._signature
    #     # Initialize signature based on inputs or expected properties.
    #     logger.debug(f"\n===  Initializing `{self}` Signature === ")
    #     # if self.inputs:
    #     if self.inputs and not self.expected_input_properties:
    #         logger.debug(f"Defining signature from inputs: {self.inputs}. Expected inputs: {self.expected_input_properties}")
    #         self._signature = Signature.build_from_data(self.inputs, self.output_properties)
    #     else:
    #         self._signature = Signature.build_from_properties(
    #             input_props=self.expected_input_properties,
    #             output_props=self.output_properties
    #         )
    #     return self._signature

    @property
    def signature(self) -> Signature:
        """Allow lazy access to the signature."""
        if self._signature is None:
            if self.inputs:
                # logger.debug(f"Building signature for {self}")
                # self._signature = self.build_signature()
                self._signature = Signature.build_from_data(self.inputs, self.output_properties)
            else:
                self._signature = Signature.build_from_properties(
                    input_props=self.expected_input_properties,
                    output_props=self.output_properties,
                )
        else:
            logger.debug(f"Accessing cached signature for {self}")
        return self._signature
    

    # --- Helper methods for input mapping and logging ---
    def _build_mappings(self):
        self.input_id_to_spec = {}
        self.spec_to_input_id = {}
        self.usage_to_spec = {}
        self.spec_to_usage = {}
        self.hint_to_spec = {}
        self.spec_to_hint = {}

        for arg, data_obj in self.normalized_map.items():
            self.input_id_to_spec[data_obj.id] = arg
            self.spec_to_input_id[arg] = data_obj.id

            usage = data_obj.properties.get("Usage")
            if usage:
                self.usage_to_spec[usage] = arg
                self.spec_to_usage[arg] = usage

            hint = None
            if hasattr(data_obj, "metadata") and hasattr(data_obj.metadata, "hint") and data_obj.metadata.hint:
                hint = data_obj.metadata.hint
            if hint:
                self.hint_to_spec[hint] = arg
                self.spec_to_hint[arg] = hint
            else:
                self.spec_to_hint[arg] = usage

        description = (
            f"{self} initialized with the following map attributes:\n"
            f" - input_id_to_spec: {self.input_id_to_spec} \n"
            f" - spec_to_input_id: {self.spec_to_input_id}\n"
            f" - spec_to_hint: {self.spec_to_hint}\n"
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
                ordered.append(norm_map[reg.name])
            else:
                # Fallback: search in inputs by matching 'Usage'
                fallback = next((d for d in inputs if d.properties.get("Usage") == reg.name), None)
                if fallback:
                    ordered.append(fallback)
                else:
                    ordered.append(None)
                    logger.warning(f"Normalized mapping missing key for register {reg.name}")
        return ordered
    
    def _hypothetical_unify_mapping(self, norm_map: Dict[str, Data]) -> Dict[str, Any]:
        unified = {}
        for data_obj in norm_map.values():
            usage = data_obj.properties.get("Usage", data_obj.id)
            unified[usage] = data_obj.data
        return unified
    
    def infer_output_properties(self):
        """Infer output properties based on the signature."""
        # If no explicit output_properties are set, try building from the signature’s RIGHT data.
        if self.signature:
            if self.inputs is None:
                self.inputs = list(self.signature.lefts())
            output_signature = self.signature.rights()
            return [out.properties for out in output_signature if out.flow == Flow.RIGHT]
        else:
            raise ValueError("Cannot infer outputs without a valid signature")
        
    # Remaining methods are meant to be implemented (overwritten by subclasses)
    def set_required_resources(self):
        """Sets the appropriate resources required to run the process. This method should update
        self.required_resources to be a list of Resources required for the Process to successfully run.
        Will be unique for each process.
        """

        raise NotImplementedError

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

    def validate_data_properties_new(self) -> bool:
        """
        Ensures that the input Data objects satisfy the expected properties defined in
        self.expected_input_properties. Instead of relying on positional order, we build a
        mapping from normalized inputs using their "Usage" property. This allows inputs to
        be matched with expected properties in a robust, order‐independent way.
        """
        errors = []
        logger.debug("Starting validate_data_properties()")
        
        if not self.expected_input_properties:
            logger.debug("No expected input properties defined; skipping validation.")
            return True

        # Use normalized_inputs if available; otherwise, fallback to self.inputs.
        if hasattr(self, "normalized_inputs"):
            input_list = self.normalized_inputs if self.normalized_inputs else None
            if not input_list:
                err_msg = "No input data available for validation."
                logger.error(err_msg)
                raise InitError(err_msg)
        
        # Build a mapping from "Usage" to the corresponding Data object.
        # (If there are duplicates, later entries will override earlier ones.)
        usage_to_data = {}
        for data_obj in input_list:
            usage = data_obj.properties.get("Usage")
            if usage:
                usage_to_data[usage] = data_obj
            else:
                usage_to_data[data_obj.id] = data_obj

        logger.debug(f"Built usage_to_data mapping: {{ {', '.join(f'{k}: {v.id}' for k,v in usage_to_data.items())} }}")

        # Iterate over expected properties (each expected property dict should have a "Usage" key)
        for expected in self.expected_input_properties:
            exp_usage = expected.get("Usage")
            if not exp_usage:
                logger.warning("Expected property dict missing 'Usage' key. Skipping this entry.")
                continue

            data_obj = usage_to_data.get(exp_usage)
            if data_obj is None:
                err_msg = f"No normalized input found for expected usage: {exp_usage}"
                logger.error(err_msg)
                errors.append(err_msg)
                continue

            logger.debug(f"Validating expected usage '{exp_usage}' with Data ID: {data_obj.id}")
            # Now compare each property in the expected dict with the actual value.
            for prop_name, expected_value in expected.items():
                actual_value = data_obj.properties.get(prop_name, None)
                logger.debug(f"  Comparing property '{prop_name}' for Data {data_obj.id}: expected={expected_value}, actual={actual_value}")
                if prop_name == "Data Type":
                    if not is_consistent_data_type(expected_value, actual_value):
                        err_msg = (f"'{prop_name}' mismatch for Data {data_obj.id} with meta: {data_obj.metadata}\n"
                                   f"    Expected {expected_value}, Actual {actual_value}")
                        logger.error(err_msg)
                        errors.append(err_msg)
                    else:
                        logger.debug(f"  '{prop_name}' matched for Data {data_obj.id}")
                elif prop_name == "Usage":
                    actual_usage = actual_value or data_obj.metadata.hint
                    if expected_value == "Matrix":
                        # If expected usage is "Matrix", force a match for 2D data if needed.
                        if (actual_usage is None or actual_usage.lower() == "tensor") and hasattr(data_obj, "ndim") and data_obj.ndim == 2:
                            actual_usage = "Matrix"
                            data_obj.properties["Usage"] = "Matrix"
                            logger.debug(f"  Forced usage to 'Matrix' for Data {data_obj.id} based on 2D shape")
                    if actual_usage != expected_value:
                        err_msg = (f"Usage mismatch for Data {data_obj.id} with meta: {data_obj.metadata}\n"
                                   f"    Expected {expected_value}, Actual {actual_usage}")
                        logger.error(err_msg)
                        errors.append(err_msg)
                    else:
                        logger.debug(f"  Usage matched for Data {data_obj.id}")
                else:
                    if expected_value != actual_value:
                        err_msg = (f"'{prop_name}' mismatch for Data {data_obj.id} with meta: {data_obj.metadata}\n"
                                   f"    Expected {expected_value}, Actual {actual_value}")
                        logger.error(err_msg)
                        errors.append(err_msg)
                    else:
                        logger.debug(f"  Property '{prop_name}' matched for Data {data_obj.id}")

        if errors:
            logger.error("Validation errors found in validate_data_properties():")
            for err in errors:
                logger.error(err)
            raise InitError("Input data does not satisfy expected properties :\n\n" + "\n".join(errors))

        logger.debug("All input data properties validated successfully.")
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

    def validate_data_properties(self) -> bool:
        """Basic verification that input data has the expected input
        properties
        """

        self.inputs_copy = copy.copy(self.inputs)
        result = all(
            self.expected_property_in_inputs(expected_property)
            for expected_property in self.expected_input_properties
        )
        del self.inputs_copy
        return result

    def expected_property_in_inputs(self, expected_property: dict) -> bool:
        for data in self.inputs_copy:
            # logger.debug(f"expected_property.items(): {expected_property.items()}")
            if all(
                all_dict1_vals_in_dict2_vals(data.properties.get(key, None), val)
                for key, val in expected_property.items()
            ):
                self.inputs_copy.remove(data)
                return True
        return False

    def validate_data(self) -> bool:
        """Process specific verification that ensures input data has
        correct specifications. Will be unique to each process
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
            # f"inputs: {self.inputs}; outputs: {self.generate_output()}\n"
            f"inputs: {self.inputs}\n"
            f"status: {self.status}\n"
            f"dyn: {self.dynamic}\n"
        )
        else:
            description = (
                f"Process: {self}\n"
                f"inputs: {self.inputs}\n"
                f"status: {self.status}\n"
                f"dyn: {self.dynamic}\n"
            )
        logger.debug(description)


class ClassicalProcess(Process):
    def __init__(
        self,
        inputs=None,
        expected_input_properties=None,
        required_resources=None,
        output_properties=None,
        dynamic=False,
        shape_env = None,
    ):
        super().__init__(
            inputs=inputs,
            expected_input_properties=expected_input_properties,
            required_resources=required_resources,
            output_properties=output_properties,
            dynamic=dynamic,
             shape_env = shape_env,
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
        logger.debug(f'{self} requires {self.flops}. With clock freq={ allocation.clock_frequency:.2e}, time til completion set to: {time_till_completion:.2e}')
        return time_till_completion


class QuantumProcess(Process):
    def __init__(
        self,
        inputs=None,
        expected_input_properties=None,
        required_resources=None,
        output_properties=None,
        dynamic=False,
        shape_env = None,
    ):
        super().__init__(
            inputs=inputs,
            expected_input_properties=expected_input_properties,
            required_resources=required_resources,
            output_properties=output_properties,
            dynamic=dynamic,
            shape_env = shape_env,
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

def normalize_process_call(
    target: Callable,
    args: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]] = None,
    *,
    arg_types: Optional[Tuple[Any, ...]] = None,
    kwarg_types: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    A unified function that attempts to "bind" user inputs to either:
        (1) A Process subclass's signature (like MatrixMult)
        (2) A torch operator (like torch.add)
        (3) A normal Python function
    Returns a dictionary {param_name -> actual_argument} if successful,
    or raises an error if the binding fails.
    
    1. If 'target' is an instance of Process, we:
       - Build an inspect.Signature from 'process.signature'
       - Attempt signature.bind(*args, **kwargs)
       - Validate each input Data with the RegisterSpecs
       - Return {param_name: Data}

    2. If 'target' is a torch op (OpOverload/OpOverloadPacket) or a BuiltinFunctionType,
       we use `get_signature_for_torch_op(...)` to retrieve schemas. We do a first pass to see
       which schemas can bind the real arguments, then if multiple match, use `arg_types/kwarg_types`
       to disambiguate. Finally, we do a `bound = chosen_sig.bind(*args, **kwargs)` and return
       `dict(bound.arguments)` so that param names match e.g. ("input", "other").

    3. Otherwise, fallback to a normal Python function check with inspect.signature(target).
    
    NOTE: This merges the advanced multiple-dispatch logic in operator_schemas.py
    with your custom logic for Process. We skip `inspect.signature` for built-in ops
    to avoid "no signature found" errors.

    Args:
        target (Union[Process, callable]): The function or Process object
        args (Tuple[Any]): The positional args
        kwargs (Optional[Dict[str, Any]]): The keyword args
        arg_types (Optional[Tuple[Any]]): Additional type hints for disambiguation
        kwarg_types (Optional[Dict[str, Any]]): Additional type hints for disambiguation

    Returns:
        A dict {param_name -> Data or python object} with resolved inputs, or raises an error.
    """
    import inspect
    import re
    import types
    from typing import Any, Dict, Tuple, Optional, Callable

    if kwargs is None:
        kwargs = {}
    normalized_data: Dict[str, Any] = {}
  
    from torch._library.infer_schema import infer_schema
    # print(infer_schema)

    # (1) If target is a Process => build an inspect.Signature from process signature
    if isinstance(target, Process):
        fx_sig = build_inspect_signature_from_process_signature(target.signature)

        param_names = [p.name for p in target.signature.lefts()]  
        # logger.debug(f'Process {target} with param names: {param_names}')
        # Attempt to bind 
        args_for_bind = [d.data for d in args]  # or just 'args' if they're Data
        kwargs_for_bind = {k: (v.data if hasattr(v, "data") else v) for k, v in kwargs.items()}
        try:
            bound = fx_sig.bind(*args_for_bind, **kwargs_for_bind)
            bound.apply_defaults()
        except TypeError as te:
            matchobj = re.search(r"missing a required argument: '(.*?)'", str(te))
            if matchobj:
                missing_param = matchobj.group(1)
                raise ValueError(f"Missing data for parameter {missing_param}") from te
            raise ValueError(f"Failed to bind arguments to Process {target}: {te}") from te
        # Reconstruct param_name -> original Data 
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
            # logger.debug(f"{pname}: {data_obj}")
            # print(pname, type(data_obj))

        # Additional typed checks using the process's signature
        # e.g. 'signature.validate_data_with_register_specs(inputs)'
        # or your own logic with regspec.matches_data
        # for k,v in normalized_data.items():
        #     logger.debug(f"{k}: {v} of type {type(v)}")
       
        target.signature.validate_data_with_register_specs(
            [normalized_data[p.name] for p in target.signature.lefts()]
        )
        return normalized_data
    
    # (2) If target is a torch op => do advanced schema approach, skip normal signature fallback
    from torch.fx.operator_schemas import (
        get_signature_for_torch_op,

        OpOverload,
        OpOverloadPacket,
       
    )
    if (isinstance(target, OpOverloadPacket)
        or isinstance(target, OpOverload)
        or isinstance(target, types.BuiltinFunctionType)):
        torch_op_sigs = get_signature_for_torch_op(target)
        if not torch_op_sigs:
            # No known signatures => can't bind
            raise ValueError(f"No signature found for PyTorch op: {target}")

        # 2.1) Try to find *all* matches by attempting sig.bind(*args, **kwargs)
        matched_schemas = []
        for candidate_sig in torch_op_sigs:
            try:
                candidate_sig.bind(*args, **kwargs)
                matched_schemas.append(candidate_sig)
            except TypeError:
                continue

        if len(matched_schemas) == 0:
            # No valid schema matched => can't unify
            raise ValueError(
                f"No valid overload for {target} with arguments={args}, kwargs={kwargs}"
            )
        elif len(matched_schemas) == 1:
            # Exactly one match => perfect
            chosen_sig = matched_schemas[0]
        else:
            # Multiple matches => we need arg_types/kwarg_types to break ties
            if not arg_types and not kwarg_types:
                # Raise an error for ambiguity
                raise ValueError(
                    f"Ambiguous call to {target}. Multiple overloads matched, "
                    "and you did not provide arg_types/kwarg_types to disambiguate."
                )

            # 2.2) Second pass: do type-based filtering
            # We'll see which schemas match your provided param-type hints
            # We require that every param name's type passes type_matches().
            filtered = []
            for candidate_sig in matched_schemas:
                try:
                    # We'll do a parallel "bind" but with arg_types instead of real values
                    bound_type_check = candidate_sig.bind(*(arg_types or ()), **(kwarg_types or {}))
                except TypeError:
                    continue

                all_good = True
                for name, user_type in bound_type_check.arguments.items():
                    param = candidate_sig.parameters[name]
                    if param.annotation is not inspect.Parameter.empty:
                        if not type_matches(param.annotation, user_type):
                            all_good = False
                            break
                    # else no annotation => ignore
                if all_good:
                    filtered.append(candidate_sig)

            if len(filtered) == 0:
                raise ValueError(
                    f"Could not find a matching schema for {target} even after "
                    f"type-based disambiguation. arg_types={arg_types} kwarg_types={kwarg_types}"
                )
            elif len(filtered) > 1:
                raise ValueError(
                    f"Still ambiguous: multiple overloads match the provided arg_types={arg_types} "
                    f"and kwarg_types={kwarg_types} for {target}. Overloads: {filtered}"
                )
            else:
                chosen_sig = filtered[0]

        # Finally, we have a single chosen_sig => do a final bind
        bound = chosen_sig.bind(*args, **kwargs)
        bound.apply_defaults()
        return dict(bound.arguments)

    # (3) If not a Process or a torch op => fallback to normal Python signature
    sig = inspect.signature(inspect.unwrap(target))
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()
    return dict(bound.arguments)
