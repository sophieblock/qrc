

from typing import List, Dict, Tuple, Iterable, Union, Sequence, Optional,overload

import numpy as np
import itertools
from dataclasses import dataclass, field
from typing import Optional, Union, Tuple, Dict, Callable
import sympy
from attrs import field, frozen
from torch._logging import dtrace_structured, LazyString, structured, trace_structured
from torch._utils_internal import signpost_event
import enum
from torch.fx.experimental.symbolic_shapes import (
    free_unbacked_symbols,
    DimDynamic,
    SymbolicContext,
    guard_int,
    StatelessSymbolicContext,
    EqualityConstraint, 
    DimList,
    Constraint, 
    ShapeGuardPythonPrinter,
    ConstraintViolationError,
    DimConstraints,
    DimConstraint,
    StrictMinMaxConstraint,
    RelaxedUnspecConstraint,
    SubclassSymbolicContext

)
from torch._subclasses.meta_utils import is_sparse_any
from torch.utils._sympy.functions import (
    Mod,
    PythonMod,
)
from torch._subclasses.fake_tensor import FakeTensor
import torch
from torch.fx.experimental.symbolic_shapes import ShapeEnv as TorchShapeEnv
from torch.fx.experimental.sym_node import method_to_operator, SymNode
from torch import SymInt, SymBool, SymFloat

from torch.fx.experimental.recording import record_shapeenv_event, replay_shape_env_events, FakeTensorMeta
from torch._guards import ShapeGuard, Source
import inspect

# from .dtypes import MatrixType,DataType, TensorType, CBit,CAny,QAny,QBit

from .data_types import MatrixType,DataType, TensorType, CBit,CAny,QAny,QBit

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Type

from sympy import Symbol

from collections import defaultdict
from typing_extensions import TypeGuard

from ...util.log import logging
logger = logging.getLogger(__name__)

   
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




def hint_int(a: Union[torch.SymInt, int], fallback: Optional[int] = None) -> int:
    """
    Retrieve the hint for an int (based on the underlying real values as observed
    at runtime).  If no hint is available (e.g., because data dependent shapes),
    if fallback is not None, use that instead (otherwise raise an error).
    """
    if isinstance(a, torch.SymInt):
        return a.node.require_hint(fallback)
    assert type(a) is int, a
    return a

def expect_true(a, skip: int = 0):

    # Assume that a boolean is true for the purposes of subsequent symbolic
    # reasoning.  This will keep track of corresponding runtime checks to verify
    # that the result is upheld: either as a regular guard, or as a special set
    # of asserts which are triggered when an unbacked SymInt is allocated.
    #
    # DO NOT use this function for these cases:
    #
    #  - This is inappropriate for "branching" conditions (where both
    #    true and false result in valid programs).  We will always assume
    #    the condition evaluates true, and so it will never be possible
    #    to trace the false condition when you use it.  For true branching
    #    on unbacked SymInts, you must use torch.cond; if you incorrectly
    #    use expect_true in this case, you will make the false branch
    #    unreachable (as we will simply assume that only the true branch
    #    is ever exercised).
    #
    #  - This is inappropriate for situations where you know some other system
    #    invariant guarantees that this property holds, since you don't
    #    really need to insert a runtime check in that case.  Use something
    #    like constrain_range in that case -> source: torch.fx.experimental.symbolic_shapes.py
    if isinstance(a, SymBool):
        # TODO: check perf implications of this
        frame = inspect.currentframe()
        for _ in range(skip + 1):  # always run this loop at least once
            frame = frame.f_back
        return a.node.expect_true(frame.f_code.co_filename, frame.f_lineno)
    assert type(a) is bool, a
    return a

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

def create_symtype(cls, pytype, shape_env, arg,source = None, duck=True):
    from torch._dynamo.source import ConstantSource
    if source == None:
        source = ConstantSource(f"{len(shape_env.var_to_val)}")
        # source = ConstantSource(f"{arg.id}")


    symbol = shape_env.create_symbol(
        arg,
        source=source,
        dynamic_dim=DimDynamic.DUCK if duck else DimDynamic.DYNAMIC,
        constraint_dim=None,
    )
    return cls(
        SymNode(
            symbol,
            shape_env,
            pytype,
            hint=arg,
        )
    )

def create_symint(shape_env, i: int, duck=True):
    return create_symtype(SymInt, int, shape_env, i, duck=duck)

def create_symbool(shape_env, b: bool):
    return create_symtype(SymBool, bool, shape_env, b)

def create_symfloat(shape_env, f: float):
    return create_symtype(SymFloat, float, shape_env, f)


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

NEXT_AVAILABLE_ENV_ID = 0

def get_next_env_id():
    global NEXT_AVAILABLE_ENV_ID
    next_id = NEXT_AVAILABLE_ENV_ID
    #print(next_id, format(next_id, "X").zfill(7))
    NEXT_AVAILABLE_ENV_ID+=1
    #return "D" + format(next_id, "X").zfill(2)
    return str(next_id)

class ShapeEnv(TorchShapeEnv):
    """
    DESIGN:
    -------
    This module extends torch's `ShapeEnv` to unify symbolic shapes across `Data` objects.
    Introduces custom logic (e.g., create_symbolic_int for bits, qubits). Additional 
    constraints can added here.
    
    """
    def __init__(self, **kwargs):
        # Initialize the base class
        kwargs.setdefault('specialize_zero_one', True) 
        super().__init__(**kwargs)
        self.custom_constraints = {}
        self.id = 'ENV_'+get_next_env_id()
        # export TORCHDYNAMO_TRANSLATION_VALIDATION=1 to set confi.translation_validation = True
        # from torch.fx.experimental.validator import translation_validation_enabled, _assert_z3_installed_if_tv_set
        # assert self._translation_validation_enabled, f'{_assert_z3_installed_if_tv_set()}'
        # print(f"translation validation enabled? {translation_validation_enabled()}")
        # logger.debug(f'New shape env: {self}')
   

    def create_symbolic_int(self, val, source: Source, symbolic_type: str):
        """
        Create a symbolic integer for scalar data types.
        Uses `make_symbol` with appropriate symbolic type prefixes for custom types.
        """
        sym_type = {
            "bit": fresh_bit,
            "qbit": fresh_qbit
        }.get(symbolic_type, fresh_int)
        
        symbolic_expr = sympy.Symbol(sym_type())
        # logger.debug(f"Creating symbolic integer {symbolic_expr} for value {val}, source {source}")
        # print(self._create_symbol_for_source(source))
        return self.create_symintnode(symbolic_expr, hint=val, source=source)
    
    def _create_symbol_for_source(self, source: Source) -> Optional[sympy.Symbol]:
        # if not self._translation_validation_enabled:
        #     return None
        srcname = source.name()
        if source not in self.source_to_symbol:
            self.source_to_symbol[srcname] = sympy.Symbol(srcname, integer=True)
        return self.source_to_symbol[srcname]
    
    def create_symbolic_size(self, val,dtype, source: Source, dynamic_dim=DimDynamic.STATIC):
        """Creates a symbolic size value with a specified source and dynamic dimension."""
        # ex_size = tuple(self._maybe_specialize_sym_int_with_hint(sz) for sz in data.size())
        # logger.debug(f"Attempting to create symbol for {val}, source= {source}, dtype={dtype}")
        return create_symtype(map_symnode_to_type[dtype], dtype, self, arg=val, source=source, duck=True)
    def create_symbolic_sizes(
        self,
        shape: Sequence[int],
        source: Source,
        constraint_sizes: Optional[SymbolicContext] = None,
        symbolic_context: Optional[SymbolicContext] = None,
    ) -> List[sympy.Expr]:
        # print(self)
        from torch._dynamo.source import TensorPropertySource, TensorProperty
        # logger.debug(f"creating symbolic shape. Inputs:\n   - shape: {shape}\n   - source: {source}\n   - constraint_sizes: {constraint_sizes}")
        if isinstance(shape, tuple):
            if len(shape) == 0:
                # 0D shape => meta-tensor with zero dimension
                ex_data = torch.empty([], device="meta")
            else:
                ex_data = torch.empty(*shape, device="meta")
        else:
            ex_data = shape

        ex_size = tuple(self._maybe_specialize_sym_int_with_hint(sz) for sz in ex_data.size())
        dim = len(ex_size)

        # Validate and synchronize environments for symbolic integers
        for sz in ex_size:
            if isinstance(sz, SymInt) and sz.node.shape_env is not self:
                try:
                    sz.node.shape_env.check_equal(self)
                except AssertionError as e:
                    raise ValueError(
                        f"Mismatch between shape environments: {sz.node.shape_env} vs {self}. Details: {e}"
                    )

        # logger.debug(f"- ShapeEnv.create_symbolic_sizes - ex_size create: {ex_size} from the ex_data {ex_data}")
        
        if symbolic_context is None:
            
            dynamic_dims = [DimDynamic.DUCK] * dim
            constraint_sizes = constraint_sizes if constraint_sizes else [None] * dim
            symbolic_context = StatelessSymbolicContext(
                dynamic_sizes=dynamic_dims,
                constraint_sizes=constraint_sizes,
            )
        assert len(ex_size) == len(dynamic_dims), "Shape and dynamic dims length mismatch"

        size: List[sympy.Expr] = self._produce_dyn_sizes_from_int_tuple(ex_size, source, symbolic_context)
        sym_sizes = [
            self.create_symintnode(
                sym,
                hint=hint,
                source=TensorPropertySource(source, TensorProperty.SIZE, i),
            )
            for i, (sym, hint) in enumerate(zip(size, ex_size))
        ]

        return tuple(sym_sizes)
    
    def create_symbolic_strides(self, strides: Sequence[int], source: Source) -> List[sympy.Expr]:
        """Generates symbolic strides."""
        from torch._dynamo.source import TensorPropertySource, TensorProperty
        
        return tuple(self.create_symintnode(
            sympy.Integer(stride),
            hint=stride,
            source=TensorPropertySource(source, TensorProperty.STRIDE, i)
        ) for i, stride in enumerate(strides))
    
    def create_symbolic_storage_offset(self, offset: int, source: Source) -> sympy.Expr:
        """Generates symbolic storage offset."""
        from torch._dynamo.source import TensorPropertySource, TensorProperty
        
        return self.create_symintnode(
            self.create_symbol(
                offset,
                TensorPropertySource(source, TensorProperty.STORAGE_OFFSET),
                dynamic_dim=DimDynamic.DUCK,
                constraint_dim=None,
            ),
            hint=offset,
            source=TensorPropertySource(source, TensorProperty.STORAGE_OFFSET)
        )
   
    @record_shapeenv_event()
    def create_symintnode(
            self,
            sym: "sympy.Expr",
            *,
            hint: Optional[int],
            source: Optional[Source] = None,
    ):
        """Create a SymInt value from a symbolic expression

        If you know what the current hint value of the SymInt to be created
        is, pass it into hint.  Otherwise, pass None and we will make our best
        guess

        """
        source_name = source.name() if source else None
        # logger.debug(f" source: {source}, name: {source_name}")
        if self._translation_validation_enabled and source is not None:
            # Create a new symbol for this source.
            symbol = self._create_symbol_for_source(source)
            assert symbol is not None

            # Create a new FX placeholder and Z3 variable for 'symbol'.
            fx_node = self._create_fx_placeholder_and_z3var(symbol, int)

            # Add an equality assertion for the newly created symbol and 'sym'.
            self._add_assertion(sympy.Eq(symbol, sym))
        else:
            fx_node = None
        out: Union[int, SymInt]
        if isinstance(sym, sympy.Integer):
            if hint is not None:
                assert int(sym) == hint
            out = int(sym)
        else:
            # How can this occur? When we mark_unbacked, we end up with a real
            # tensor that has hints for all sizes, but we MUST NOT create a
            # SymNode with a hint, because we're hiding the hint from our eyes
            # with the unbacked Symbol.  And in fact, the hint compute may be
            # inconsistent with size oblivious tests.
            if free_unbacked_symbols(sym):
                hint = None
            out = SymInt(SymNode(sym, self, int, hint, fx_node=fx_node))
        return out
    def add_constraint(self, symbol: Symbol, constraint: sympy.Expr):
        """Add a custom constraint to the environment."""
        self.custom_constraints[symbol] = constraint
    def enforce_even_dimension(self, sym):
        from sympy import Eq
        # sym is a sympy.Symbol
        # symbol = self._create_symbol_for_source(source)
        # assert symbol is not None
        # self._add_assertion(sympy.Eq(symbol, sym))

        eq_expr = Eq(Mod(sym, 2), 0)
        self.custom_constraints[sym] = eq_expr
        self.dim_constraints.add(eq_expr)
    def enforce_custom_constraints(self):
        """Validate all custom constraints."""
        for symbol, constraint in self.custom_constraints.items():
            if not constraint.subs(self.sym_to_val):
                raise ValueError(f"Constraint violated for symbol {symbol}: {constraint}")
    def merge(self, other: "ShapeEnv", *, raise_on_conflict: bool = True, do_post_check: bool = True):
        """
        Attempt a robust merge of 'other' ShapeEnv into 'self'. That means:
        1) For every symbol in other.var_to_val, if we don’t have it, adopt it; 
            if we do, unify or raise conflict. 
        2) Merge var_to_range by intersecting ranges if both define them.
        3) Merge guards (skip duplicates).
        4) Merge unbacked_renamings, deferred_runtime_asserts, etc.
        5) Merge custom_constraints if we can. If a symbol is repeated with 
            conflicting constraints, raise or log a conflict.
        6) If do_post_check=True, we do a final “consistency” pass that tries 
            to produce guards and see if it fails. If it fails, either raise 
            or revert if you want.

        :param raise_on_conflict: If True, raise an error on direct dimension conflicts 
            (like x=2 vs x=3). If False, keep 'self' and log the conflict.
        :param do_post_check: If True, try produce_guards(...) to ensure the environment 
            is still valid. If produce_guards fails (ConstraintViolationError), 
            we raise that error.


        """
        pass
    def __repr__(self):
        return f"{self.id}(mapping: {self.var_to_val}, num_guards={len(self.guards)})"
    def id(self):
        return self.id
    def check_equal(self, other: "ShapeEnv"):
        from torch.fx.experimental.recording import shape_env_check_state_equal
        # A custom list of fields that you consider 'non-state'
        # shape_env_check_state_equal()) compares two ShapeEnv, ignoring non-state fields and remapping certain internal data. 
        # We want to adapt this logic to compare two of your shape envs for equality.
        ignore_fields = ['var_to_stack', 'log', 'logger']
        def map_value(k, v):
            # If the 'guards' contain a (expr, origin) tuple, unify it or just keep expr.
            if k == 'guards' and isinstance(v, list):
                # keep only the sympy expression
                return tuple(expr for (expr, origin) in v)
            return v

        shape_env_check_state_equal(self, other, ignore_fields, map_value)
