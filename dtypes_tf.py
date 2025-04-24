"""Quantum data type definitions (TensorFlow TypeSpec–adapted).

In this version the abstract QDType is re–imagined as QDTypeSpec, a subclass of
tf.TypeSpec. A value of a QDTypeSpec is represented as a 1-D Boolean tensor whose
length equals the number of qubits (bitsize). The various concrete types (QBit, QInt,
QUInt, BQUInt, QAny, and QFxp) implement conversion methods (to_bits, from_bits, etc.)
as well as the TypeSpec interface.
"""
import abc
from enum import Enum
from functools import cached_property
from typing import Any, Iterable, List, Literal, Optional, Sequence, TYPE_CHECKING, Union
import math
import attrs
import numpy as np
from fxpmath import Fxp
from numpy.typing import NDArray

from qualtran.symbolics import bit_length, is_symbolic, SymbolicInt

if TYPE_CHECKING:
    import galois

import abc
from enum import Enum
from typing import Any, Iterable, List, Sequence, Union, Optional

import numpy as np
import tensorflow as tf
# --- Stub definitions for symbolic support ---
# In the original code, SymbolicInt and is_symbolic are imported from qualtran.symbolics.
SymbolicInt = Union[int, tf.Tensor]  # For our purposes, an int or a Tensor representing a symbol.
def is_symbolic(*args, **kwargs) -> bool:
    # Very naive: if any argument is not an int, we treat it as symbolic.
    return any(not isinstance(arg, int) for arg in args)
class _DynType:
    """
    _DynType defines a type which stands for the absence of type information.
    """

    def __init__(self) -> None:
        self.__name__ = "_DynType"

    def __eq__(self, other):
        return isinstance(other, self.__class__)

    def __str__(self):
        return "Dyn"

    def __repr__(self):
        return "Dyn"
    def __hash__(self):
        # Return a constant hash because all instances are "equal"
        # This is consistent with __eq__.
        return hash("_DynType")



Dyn = _DynType()


# class DataType(abc.ABC):
#     """
#     Abstract parent for all data types (classical or quantum).

#     Each subclass must implement:
#     - num_units -> int: The total “element count” or bit/qubit count (depending on the type).
#     - bitsize: Total bits required to store one instance of this data type.
#     - to_bits(...) / from_bits(...): For converting this data type to and from a bit-level representation.
#     - get_domain(): If feasible, yields all possible values (e.g., for small classical types).
#     - to_units(...) / from_units(...): Splits or reconstructs the data into smaller “units.”

#     This design ensures that the shape or size of the data is primarily stored here, making
#     the `Data` class in `data.py` simpler in handling dynamic aspects like symbolic shapes.
#     """
#     @property
#     @abc.abstractmethod
#     def num_units(self) -> int:
#         """
#         Number of "fundamental units" (bits, qubits, or something else)
#         required to represent a single instance of this data type.
#         """

#     @property
#     @abc.abstractmethod
#     def bitsize(self) -> int:
#         """
#         Total number of bits needed to represent one logical element of a given data instance.
#         Could be `num_units` * 1 for qubits, or rows*cols*8 for a float64 matrix, etc.
#         """

#     @abc.abstractmethod
#     def to_bits(self, x) -> List[int]:
#         """Convert a single value x to its bit representation."""

#     def to_bits_array(self, x_array: NDArray[Any]) -> NDArray[np.uint8]:
#         """Yields an NDArray of bits corresponding to binary representations of the input elements.

#         Often, converting an array can be performed faster than converting each element individually.
#         This operation accepts any NDArray of values, and the output array satisfies
#         `output_shape = input_shape + (self.bitsize,)`.
#         """
#         return np.vectorize(
#             lambda x: np.asarray(self.to_bits(x), dtype=np.uint8), signature='()->(n)'
#         )(x_array)
#     @abc.abstractmethod
#     def from_bits(self, bits: Sequence[int]) -> Any:
#         """Combine bits to form a single value x."""
    
#     def from_bits_array(self, bits_array: NDArray[np.uint8]):

#         """Combine individual bits to form classical values.

#         Often, converting an array can be performed faster than converting each element individually.
#         This operation accepts any NDArray of bits such that the last dimension equals `self.bitsize`,
#         and the output array satisfies `output_shape = input_shape[:-1]`.
#         """
#         return np.vectorize(self.from_bits, signature='(n)->()')(bits_array)

#     @abc.abstractmethod
#     def get_classical_domain(self) -> Iterable[Any]:
#         """Yield all possible values representable by this type (if feasible)."""

#     # @abc.abstractmethod
#     # def to_units(self, x) -> List[int]:
#     #     """Yields individual units (e.g., elements) corresponding to x."""

#     # @abc.abstractmethod
#     # def from_units(self, units: Sequence[int]) -> Any:
#     #     """Combine individual units to reconstruct x."""

#     @abc.abstractmethod
#     def assert_valid_classical_val(self, val: Any, debug_str: str = 'val'):
#         """Raises an exception if `val` is not a valid classical value for this type.

#         Args:
#             val: A classical value that should be in the domain of this QDType.
#             debug_str: Optional debugging information to use in exception messages.
#         """
#     def __str__(self):
#         return f"{self.__class__.__name__}({self.num_units})"

#     def __repr__(self):
#         return str(self)


class QDTypeSpec(tf.TypeSpec, abc.ABC):
    """Abstract base class for quantum data types using TensorFlow's TypeSpec.

    A value of a QDTypeSpec is represented as a 1-D boolean tensor of length equal
    to the number of qubits (bitsize).
    """

    @property
    @abc.abstractmethod
    def num_qubits(self) -> int:
        """Number of qubits (bitsize) for a single instance of this type."""
    
    @abc.abstractmethod
    def get_classical_domain(self) -> Iterable[Any]:
        """Yields all possible classical (computational basis state) values representable
        by this type."""
    @abc.abstractmethod
    def to_bits(self, x: Any) -> List[int]:
        """Yields individual bits corresponding to binary representation of x"""
    
    def to_bits_array(self, x_array: np.ndarray) -> np.ndarray:
        """Yields an NDArray of bits corresponding to binary representations of the input elements.

        Often, converting an array can be performed faster than converting each element individually.
        This operation accepts any NDArray of values, and the output array satisfies
        `output_shape = input_shape + (self.bitsize,)`.
        """
        return np.vectorize(
            lambda x: np.asarray(self.to_bits(x), dtype=np.uint8),
            signature='()->(n)'
        )(x_array)
    
    @abc.abstractmethod
    def from_bits(self, bits: Sequence[int]) -> Any:
        """Convert a sequence of bits to form x"""
    
    def from_bits_array(self, bits_array: np.ndarray) -> np.ndarray:
        """Combine individual bits to form classical values.

        Often, converting an array can be performed faster than converting each element individually.
        This operation accepts any NDArray of bits such that the last dimension equals `self.bitsize`,
        and the output array satisfies `output_shape = input_shape[:-1]`.
        """
        return np.vectorize(self.from_bits, signature='(n)->()')(bits_array)
    
    @abc.abstractmethod
    def assert_valid_classical_val(self, val: Any, debug_str: str = 'val'):
        """Raises an exception if `val` is not a valid classical value for this type.

        Args:
            val: A classical value that should be in the domain of this QDType.
            debug_str: Optional debugging information to use in exception messages.
        """
    def assert_valid_classical_val_array(self, val_array: np.ndarray, debug_str: str = 'val'):
        """Raises an exception if `val_array` is not a valid array of classical values
        for this type.

        Often, validation on an array can be performed faster than validating each element
        individually.

        Args:
            val_array: A numpy array of classical values. Each value should be in the domain
                of this QDType.
            debug_str: Optional debugging information to use in exception messages.
        """
        for val in val_array.reshape(-1):
            self.assert_valid_classical_val(val, debug_str=debug_str)
    
    @abc.abstractmethod
    def is_symbolic(self) -> bool:
        """Return True if this type is parameterized by symbolic values."""
    
    def iteration_length_or_zero(self) -> SymbolicInt:
        return getattr(self, 'iteration_length', 0)
    
    # --- TensorFlow TypeSpec interface ---
    @property
    def value_type(self):
        # Represent a value as a tf.Tensor of booleans with shape (num_qubits,)
        shape = (self.num_qubits,) if self.num_qubits is not None else (None,)
        return tf.Tensor

    def _component_specs(self):
        dim = self.num_qubits if self.num_qubits is not None else None
        return tf.TensorSpec(shape=(dim,), dtype=tf.bool)
    
    def _to_components(self, value):
        return value  # assume value already is a tf.Tensor
    
    def _from_components(self, components):
        return components
    
    def _serialize(self):
        # For serialization, include only the bitsize.
        return (self.num_qubits,)
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.num_qubits})"

# -----------------------------------------------------------------------------
# Concrete QDTypeSpec subclasses.
# -----------------------------------------------------------------------------

class QBit(QDTypeSpec):
    """A single qubit."""
    @property
    def num_qubits(self) -> int:
        return 1

    def get_classical_domain(self) -> Iterable[int]:
        return (0, 1)

    def to_bits(self, x: Any) -> List[int]:
        self.assert_valid_classical_val(x)
        return [int(x)]

    def from_bits(self, bits: Sequence[int]) -> int:
        if len(bits) != 1:
            raise ValueError("QBit expects exactly one bit")
        return bits[0]

    def assert_valid_classical_val(self, val: Any, debug_str: str = 'val'):
        if val not in (0, 1):
            raise ValueError(f"Bad QBit value {val} in {debug_str}")

    def is_symbolic(self) -> bool:
        return False

    def __str__(self):
        return "QBit()"


class QAny(QDTypeSpec):
    """Opaque bag-of-qubits (no specific classical domain)."""
    def __init__(self, bitsize: SymbolicInt):
        self.bitsize = bitsize

    @property
    def num_qubits(self) -> int:
        return self.bitsize

    def get_classical_domain(self) -> Iterable[Any]:
        raise TypeError(f"Ambiguous domain for QAny({self.bitsize}). Use a more specific type.")
    
    def to_bits(self, x: Any) -> List[int]:
        # Delegate to QUInt conversion.
        return QUInt(self.bitsize).to_bits(x)
    
    def from_bits(self, bits: Sequence[int]) -> Any:
        return QUInt(self.bitsize).from_bits(bits)
    
    def assert_valid_classical_val(self, val: Any, debug_str: str = 'val'):
        # No validation is provided.
        pass
    
    def is_symbolic(self) -> bool:
        return is_symbolic(self.bitsize)
    
    def __str__(self):
        return f"QAny({self.bitsize})"


class QInt(QDTypeSpec):
    """Signed integer using two's complement (big–endian)."""
    def __init__(self, bitsize: SymbolicInt):
        self.bitsize = bitsize

    @property
    def num_qubits(self) -> int:
        return self.bitsize

    def is_symbolic(self) -> bool:
        return is_symbolic(self.bitsize)

    def get_classical_domain(self) -> Iterable[int]:
        max_val = 1 << (self.bitsize - 1)
        return range(-max_val, max_val)

    def to_bits(self, x: int) -> List[int]:
        if is_symbolic(self.bitsize):
            raise ValueError("Cannot compute bits with symbolic bitsize")
        self.assert_valid_classical_val(x)
        # np.binary_repr returns a string in big–endian order.
        bitstr = np.binary_repr(x, width=self.bitsize)
        return [int(b) for b in bitstr]

    def from_bits(self, bits: Sequence[int]) -> int:
        if len(bits) != self.bitsize:
            raise ValueError("Bit sequence length mismatch for QInt")
        sign = bits[0]
        if sign == 0:
            return int("".join(str(b) for b in bits), 2)
        else:
            # Two's complement: invert bits and add one.
            inverted = ''.join('1' if b == 0 else '0' for b in bits)
            return - (int(inverted, 2) + 1)

    def assert_valid_classical_val(self, val: Any, debug_str: str = 'val'):
        if not isinstance(val, (int, np.integer)):
            raise ValueError(f"{debug_str} must be an integer, got {val!r}")
        if val < -(2 ** (self.bitsize - 1)):
            raise ValueError(f"Value {val} too small for {self}")
        if val >= 2 ** (self.bitsize - 1):
            raise ValueError(f"Value {val} too large for {self}")

    def __str__(self):
        return f"QInt({self.bitsize})"


class QUInt(QDTypeSpec):
    """Unsigned integer (big–endian)."""
    def __init__(self, bitsize: SymbolicInt):
        self.bitsize = bitsize

    @property
    def num_qubits(self) -> int:
        return self.bitsize

    def is_symbolic(self) -> bool:
        return is_symbolic(self.bitsize)

    def get_classical_domain(self) -> Iterable[int]:
        return range(2 ** self.bitsize)

    def to_bits(self, x: int) -> List[int]:
        self.assert_valid_classical_val(x)
        bitstr = format(x, f'0{self.bitsize}b')
        return [int(b) for b in bitstr]

    def from_bits(self, bits: Sequence[int]) -> int:
        return int("".join(str(b) for b in bits), 2)

    def assert_valid_classical_val(self, val: Any, debug_str: str = 'val'):
        if not isinstance(val, (int, np.integer)):
            raise ValueError(f"{debug_str} must be an integer, got {val!r}")
        if val < 0:
            raise ValueError(f"Negative value {val} not allowed for {self}")
        if val >= 2 ** self.bitsize:
            raise ValueError(f"Value {val} too large for {self}")

    def __str__(self):
        return f"QUInt({self.bitsize})"


class BQUInt(QDTypeSpec):
    """Bounded unsigned integer with an iteration length."""
    def __init__(self, bitsize: SymbolicInt, iteration_length: Optional[SymbolicInt] = None):
        self.bitsize = bitsize
        self.iteration_length = iteration_length if iteration_length is not None else (2 ** bitsize)
        if not is_symbolic(self.bitsize, self.iteration_length):
            if self.iteration_length > 2 ** self.bitsize:
                raise ValueError(
                    f"Iteration length {self.iteration_length} too large for bitsize {self.bitsize}"
                )

    @property
    def num_qubits(self) -> int:
        return self.bitsize

    def is_symbolic(self) -> bool:
        return is_symbolic(self.bitsize, self.iteration_length)

    def get_classical_domain(self) -> Iterable[int]:
        if isinstance(self.iteration_length, int):
            return range(self.iteration_length)
        raise ValueError(f"Domain not defined for symbolic iteration_length {self.iteration_length}")

    def assert_valid_classical_val(self, val: Any, debug_str: str = 'val'):
        if not isinstance(val, (int, np.integer)):
            raise ValueError(f"{debug_str} must be an integer, got {val!r}")
        if val < 0:
            raise ValueError(f"Negative value {val} encountered in {debug_str}")
        if val >= self.iteration_length:
            raise ValueError(f"Value {val} exceeds iteration length {self.iteration_length} in {debug_str}")

    def to_bits(self, x: int) -> List[int]:
        self.assert_valid_classical_val(x, debug_str='val')
        return QUInt(self.bitsize).to_bits(x)

    def from_bits(self, bits: Sequence[int]) -> int:
        return QUInt(self.bitsize).from_bits(bits)

    def __str__(self):
        return f"BQUInt({self.bitsize}, {self.iteration_length})"


class QFxp(QDTypeSpec):
    r"""Fixed–point type representing a real number.

    A fixed–point number is represented by a total of bitsize bits, with num_frac bits
    for the fractional part. For signed numbers, two's complement is used.
    """
    def __init__(self, bitsize: SymbolicInt, num_frac: SymbolicInt, signed: bool = False):
        self.bitsize = bitsize
        self.num_frac = num_frac
        self.signed = signed
        if not is_symbolic(self.bitsize, self.num_frac):
            if self.signed and self.bitsize == self.num_frac:
                raise ValueError("For signed QFxp, bitsize must be greater than num_frac.")
            if self.bitsize < self.num_frac:
                raise ValueError("bitsize must be >= num_frac.")

    @property
    def num_qubits(self) -> int:
        return self.bitsize

    @property
    def num_int(self) -> SymbolicInt:
        return self.bitsize - self.num_frac

    def is_symbolic(self) -> bool:
        return is_symbolic(self.bitsize, self.num_frac)

    def _int_qdtype(self) -> Union[QInt, QUInt]:
        return QInt(self.bitsize) if self.signed else QUInt(self.bitsize)

    def get_classical_domain(self) -> Iterable[int]:
        return self._int_qdtype().get_classical_domain()

    def to_bits(self, x: Any) -> List[int]:
        return self._int_qdtype().to_bits(x)

    def from_bits(self, bits: Sequence[int]) -> int:
        return self._int_qdtype().from_bits(bits)
    def _fxp_to_bits(
        self, x: Union[float, Fxp], require_exact: bool = True, complement: bool = True
    ) -> List[int]:
        """Yields individual bits corresponding to binary representation of `x`.

        Args:
            x: The value to encode.
            require_exact: Raise `ValueError` if `x` cannot be exactly represented.
            complement: Use twos-complement rather than sign-magnitude representation of negative values.

        Raises:
            ValueError: If `x` is negative but this `QFxp` is not signed.
        """
        if require_exact:
            self._assert_valid_classical_val(x)
        if x < 0 and not self.signed:
            raise ValueError(f"unsigned QFxp cannot represent {x}.")
        if self.signed and not complement:
            sign = int(x < 0)
            x = abs(x)
        fxp = x if isinstance(x, Fxp) else Fxp(x)
        bits = [int(x) for x in fxp.like(self.fxp_dtype_template()).bin()]
        if self.signed and not complement:
            bits[0] = sign
        return bits

    def _from_bits_to_fxp(self, bits: Sequence[int]) -> Fxp:
        """Combine individual bits to form x"""
        bits_bin = "".join(str(x) for x in bits[:])
        fxp_bin = "0b" + bits_bin[: -self.num_frac] + "." + bits_bin[-self.num_frac :]
        return Fxp(fxp_bin, like=self.fxp_dtype_template())
    def assert_valid_classical_val(self, val: Any, debug_str: str = 'val'):
        self._int_qdtype().assert_valid_classical_val(val, debug_str=debug_str)

    def to_fixed_width_int(self, x: Union[float, Any], *, require_exact: bool = False, complement: bool = True) -> int:
        # For simplicity, interpret x as a float and scale it.
        # fixed_val = round(x * (2 ** self.num_frac))
        # return self._int_qdtype().from_bits(self._int_qdtype().to_bits(fixed_val))
        bits = self._fxp_to_bits(x, require_exact=require_exact, complement=complement)
        return self._int_qdtype().from_bits(bits)
    
    def float_from_fixed_width_int(self, x: int) -> float:
        return x / (2 ** self.num_frac)
    def fxp_dtype_template(self) -> Fxp:
        """A template of the `Fxp` data type for classical values.

        To construct an `Fxp` with this config, one can use:
        `Fxp(float_value, like=QFxp(...).fxp_dtype_template)`,
        or given an existing value `some_fxp_value: Fxp`:
        `some_fxp_value.like(QFxp(...).fxp_dtype_template)`.

        The following Fxp configuration is used:
         - op_sizing='same' and const_op_sizing='same' ensure that the returned
           object is not resized to a bigger fixed point number when doing
           operations with other Fxp objects.
         - shifting='trunc' ensures that when shifting the Fxp integer to
           left / right; the digits are truncated and no rounding occurs
         - overflow='wrap' ensures that when performing operations where result
           overflows, the overflowed digits are simply discarded.

        Support for `fxpmath.Fxp` is experimental, and does not hook into the classical
        simulator protocol. Once the library choice for fixed-point classical real
        values is finalized, the code will be updated to use the new functionality
        instead of delegating to raw integer values (see above).
        """
        if is_symbolic(self.bitsize) or is_symbolic(self.num_frac):
            raise ValueError(
                f"Cannot construct Fxp template for symbolic bitsizes: {self.bitsize=}, {self.num_frac=}"
            )

        return Fxp(
            None,
            n_word=self.bitsize,
            n_frac=self.num_frac,
            signed=self.signed,
            op_sizing='same',
            const_op_sizing='same',
            shifting='trunc',
            overflow='wrap',
        )
    def __str__(self):
        if self.signed:
            return f'QFxp({self.bitsize}, {self.num_frac}, True)'
        else:
            return f'QFxp({self.bitsize}, {self.num_frac})'

from typing import Iterable, Any, Union, List, Sequence, Optional, Tuple
import itertools
import numpy as np
from numpy.typing import NDArray
from torch import SymInt
import torch
import sympy
from enum import Enum
import abc
from fxpmath import Fxp
from attrs import define, field,frozen, validators

from ...util.log import logging
logger = logging.getLogger(__name__)

from qualtran.symbolics.types import is_symbolic, SymbolicComplex, SymbolicFloat, SymbolicInt
from typing import TypeVar

SymbolicT = TypeVar('SymbolicT', SymbolicInt, SymbolicFloat, SymbolicComplex)


class _DynType:
    """
    _DynType defines a type which stands for the absence of type information.
    """

    def __init__(self) -> None:
        self.__name__ = "_DynType"

    def __eq__(self, other):
        return isinstance(other, self.__class__)

    def __str__(self):
        return "Dyn"

    def __repr__(self):
        return "Dyn"
    def __hash__(self):
        # Return a constant hash because all instances are "equal"
        # This is consistent with __eq__.
        return hash("_DynType")



Dyn = _DynType()


# For clarity, the following are the base classes:
# DataType: The common abstract interface.
# QType: Marker base for quantum types (with property num_qubits = num_elements).
# CType: Marker base for classical types (with property num_bits = num_elements).

class DataType(metaclass=abc.ABCMeta):
    """
    Unified interface for data types.
    
    Fundamental metric:
      - data_width: Bit width of data words.
        For classical types, this is the number of bits.
        For quantum types, this is the number of qubits.
        
      - element_size (in bytes) = data_width / 8.
      - total_units for a register with shape (d1, d2, ..., dn) is data_width × (d1*d2*...*dn)
    """
    @property
    @abc.abstractmethod
    def data_width(self) -> int:
        """Return the number of fundamental units (bits or qubits) per value."""
    
    @abc.abstractmethod
    def get_classical_domain(self) -> Iterable[Any]:
        """Yield all possible classical values representable by this type."""
    
    @abc.abstractmethod
    def to_bits(self, x) -> List[int]:
        """Convert a value to its bit-level representation (list of 0s and 1s)."""
    @abc.abstractmethod
    def from_bits(self, bits: Sequence[int]) -> Any:
        """Reconstructs a value from its bit-level representation."""
    
    def to_bits_array(self, x_array: NDArray[Any]) -> NDArray[np.uint8]:
        """
        Vectorized conversion: given an NDArray of x values, returns an NDArray of bits.
        Output shape: input_shape + (self.bitsize,).
        """
        return np.vectorize(
            lambda x: np.asarray(self.to_bits(x), dtype=np.uint8), signature='()->(n)'
        )(x_array)
    

    def from_bits_array(self, bits_array: NDArray[np.uint8]) -> NDArray[Any]:
        """
        Vectorized reconstruction: given an NDArray of bits (last dimension = self.bitsize),
        returns an NDArray of values with shape equal to input_shape[:-1].
        """
        return np.vectorize(self.from_bits, signature='(n)->()')(bits_array)

    @abc.abstractmethod
    def assert_valid_classical_val(self, val: Any, debug_str: str = 'val'):
        """Raises an exception if val is not a valid classical value for this type."""
    
    def assert_valid_classical_val_array(self, val_array: NDArray[Any], debug_str: str = 'val'):
        """Validates an array of values; by default, validates each element."""
        for val in val_array.reshape(-1):
            self.assert_valid_classical_val(val, debug_str)
    
    def is_symbolic(self) -> bool:
        """Returns True if this data type is parameterized with symbolic objects."""
        if hasattr(self, "num_qubits"):
            return is_symbolic(self.num_qubits)
        if hasattr(self,"bit_width"):
            return is_symbolic(self.bit_width)
        return False
    def iteration_length_or_zero(self) -> SymbolicInt:
        """Returns the iteration length if defined, or else 0."""
        return getattr(self, 'iteration_length', 0)
    
    def __str__(self):
        return f"{self.__class__.__name__}({self.bitsize})"
    
class QType(DataType, metaclass=abc.ABCMeta):
    """Abstract base for quantum data types."""

    @property
    def data_width(self) -> int:
        # For quantum types, bitsize is the number of qubits
        return self.num_qubits
    
    @abc.abstractmethod
    def to_bits(self, classical_val) -> list[int]:
        """
        If we treat the quantum register in a basis-state interpretation,
        map that basis-state integer to bits. 
        For fully quantum states (superpositions), 'to_bits' is not always
        strictly meaningful, but in Qualtran they define it as the 
        classical domain the type *can* represent.
        """
    @abc.abstractmethod
    def from_bits(self, bits: list[int]):
        """Returns the classical value that these bits represent, if any."""

class CType(DataType):
    """Parent for purely classical data types."""

    @abc.abstractmethod
    def to_bits(self, val) -> list[int]:
        """Convert a classical value (e.g. int, float) to a list of bits."""
    
    @abc.abstractmethod
    def from_bits(self, bits: list[int]):
        """Inverse of to_bits()."""
   
# ----------------------------------------------------------------
# Quantum Data Type Implementations
# The fundamental unit is a qubit

    
@frozen
class QBit(QType):
    """Quantum bit (qubit) type.
    
    """
    # num_qubits: int = 1  # A qubit is one unit.

    def data_width(self):
        return 1
    def get_classical_domain(self) -> Iterable[Any]:
        """Yield all possible values representable by this type."""
        yield from (0, 1)
    def assert_valid_classical_val(self, val: int, debug_str: str = 'val'):
        if not (val == 0 or val == 1):
            raise ValueError(f"Bad {self} value {val} in {debug_str}")

    def to_bits(self, x) -> List[int]:
        """Convert a value to its binary representation."""
        self.assert_valid_classical_val(x)
        return [int(x)]

    def from_bits(self, bits: Sequence[int]) -> int:
        """Convert bits to a value."""
        if len(bits) != 1:
            raise ValueError("Invalid bit sequence for a qubit. Must have exactly 1 bit.")
        return bits[0]

    # def assert_valid_classical_val_array(
    #     self, val_array: NDArray[np.integer], debug_str: str = 'val'
    # ):
    #     if not np.all((val_array == 0) | (val_array == 1)):
    #         raise ValueError(f"Bad {self} value array in {debug_str}")

    # def __repr__(self) -> str:
    #     return f"<QBit {self.index}>"
    # def __hash__(self) -> int:
    #     return hash(self.index)
    # def __eq__(self, other: object) -> bool:
    #     return isinstance(other, QBit) and other.index == self.index
    
    # def __str__(self):
    #     return f"q{id(self)}" if self.index is None else str(self.index)
    
@frozen
class QAny(QType):
    """Opaque bag-of-qubits type."""
    num_qubits: SymbolicInt
    
   
    @property
    def data_width(self) -> int:
        return self.num_qubits

    def get_classical_domain(self) -> Iterable[Any]:
        raise TypeError(f"Ambiguous domain for {self}. Please use a more specific type.")

    def to_bits(self, x) -> List[int]:
        # For QAny, delegate to the corresponding unsigned integer representation.
        return QUInt(self.data_width).to_bits(x)
    
    def from_bits(self, bits: Sequence[int]) -> int:
        return QUInt(self.data_width).from_bits(bits)
    
    def to_bits_array(self, x_array: NDArray[np.integer]) -> NDArray[np.uint8]:
        """Returns the big-endian bitstrings specified by the given integers.

        Args:
            x_array: An integer or array of unsigned integers.
        """
        if is_symbolic(self.num_qubits):
            raise ValueError(f"Cannot compute bits for symbolic {self.num_qubits=}")

        if self.num_qubits > 64:
            # use the default vectorized `to_bits`
            return super().to_bits_array(x_array)

        w = int(self.num_qubits)
        x = np.atleast_1d(x_array)
        if not np.issubdtype(x.dtype, np.uint):
            assert np.all(x >= 0)
            assert np.iinfo(x.dtype).bits <= 64
            x = x.astype(np.uint64)
        assert w <= np.iinfo(x.dtype).bits
        mask = 2 ** np.arange(w - 1, 0 - 1, -1, dtype=x.dtype).reshape((w, 1))
        return (x & mask).astype(bool).astype(np.uint8).T

    def from_bits(self, bits: Sequence[int]) -> int:
        """Combine individual bits to form x"""
        return int("".join(str(x) for x in bits), 2)

    def from_bits_array(self, bits_array: NDArray[np.uint8]) -> NDArray[np.integer]:
        """Returns the integer specified by the given big-endian bitstrings.

        Args:
            bits_array: A bitstring or array of bitstrings, each of which has the 1s bit (LSB) at the end.
        Returns:
            An array of integers; one for each bitstring.
        """
        bitstrings = np.atleast_2d(bits_array)
        if bitstrings.shape[1] != self.data_width:
            raise ValueError(f"Input bitsize {bitstrings.shape[1]} does not match {self.data_width=}")

        if self.data_width > 64:
            # use the default vectorized `from_bits`
            return super().from_bits_array(bits_array)

        basis = 2 ** np.arange(self.data_width - 1, 0 - 1, -1, dtype=np.uint64)
        return np.sum(basis * bitstrings, axis=1, dtype=np.uint64)
   
    def get_classical_domain(self) -> Iterable[Any]:
        """Return the domain of possible values (not supported for QAny)."""
        raise NotImplementedError("Ambiguous.")
    def assert_valid_classical_val(self, val, debug_str: str = 'val'):
        pass

    def assert_valid_classical_val_array(self, val_array, debug_str: str = 'val'):
        pass


    

@frozen
class QInt(QType):
    """Signed Integer of a given width bitsize.

    A two's complement representation is used for negative integers.
    Here (and throughout Qualtran), we use a big-endian bit convention. The most significant
    bit is at index 0.
    Domain: -2^(data_width-1) to 2^(data_width-1)-1.
    """

    num_qubits: SymbolicInt
    
    @property
    def data_width(self):
        return self.num_qubits
    
    def get_classical_domain(self) -> Iterable[int]:
        max_val = 1 << (self.num_qubits - 1)
        return range(-max_val, max_val)
    
    def to_bits(self, x: int) -> List[int]:
        """Yields individual bits corresponding to binary representation of x"""
        if is_symbolic(self.num_qubits):
            raise ValueError("Cannot compute bits with symbolic bitsize.")
        self.assert_valid_classical_val(x)
        bitstring = np.binary_repr(int(x) & ((1 << int(self.num_qubits)) - 1), width=int(self.num_qubits))
        return [int(b) for b in bitstring]
    
    def from_bits(self, bits: Sequence[int]) -> int:
        """Combine individual bits to form x"""
        sign = bits[0]
        # For two's complement, if sign bit is 1, then value is negative.
        x = (0 if self.data_width == 1 else QUInt(self.data_width - 1).from_bits(
            [1 - b if sign else b for b in bits[1:]]
        ))
        return ~x if sign else x
        # if not bits:
        #     return 0
        # val = 0
        # for bit in bits:
        #     val = (val << 1) | int(bit)
        # if bits[0] == 1:
        #     val -= (1 << (len(bits) if is_symbolic(self.data_width) else int(self.data_width)))
        # self.assert_valid_classical_val(val)
        # return val

    def assert_valid_classical_val(self, val: int, debug_str: str = 'val'):
        if not isinstance(val, (int, np.integer)):
            raise ValueError(f"{debug_str} should be an integer, not {val!r}")
        if val < -(2 ** (self.data_width - 1)):
            raise ValueError(f"Too-small classical {self}: {val} encountered in {debug_str}")
        if val >= 2 ** (self.data_width - 1):
            raise ValueError(f"Too-large classical {self}: {val} encountered in {debug_str}")
    def assert_valid_classical_val_array(
        self, val_array: NDArray[np.integer], debug_str: str = 'val'
    ):
        if np.any(val_array < -(2 ** (self.data_width - 1))):
            raise ValueError(f"Too-small classical {self}s encountered in {debug_str}")
        if np.any(val_array >= 2 ** (self.data_width - 1)):
            raise ValueError(f"Too-large classical {self}s encountered in {debug_str}")

    
    
    def __str__(self):
        return f"QInt({self.data_width})"
    



@frozen
class QUInt(QType):
    """Unsigned integer of a given width bitsize which wraps around upon overflow.

    Any intended wrap around effect is expected to be handled by the developer, similar
    to an unsigned integer type in C.

    Here (and throughout Qualtran), we use a big-endian bit convention. The most significant
    bit is at index 0.
    Domain: 0 to 2^(unit_count)-1.
    Attributes:
        bitsize: The number of qubits used to represent the integer.
        
    """
    num_qubits: SymbolicInt
    
    @property
    def data_width(self) -> int:
        return self.num_qubits
    
    def get_classical_domain(self) -> Iterable[Any]:
        return range(2 ** self.data_width)
    
    def to_bits(self, x: int) -> List[int]:
        """Yields individual bits corresponding to binary representation of x"""
        self.assert_valid_classical_val(x)
        bstr = format(x, f"0{self.num_qubits}b")
        return [int(b) for b in bstr]
    
    def from_bits(self, bits: Sequence[int]) -> int:
        """Combine individual bits to form x"""
        if len(bits) != self.num_qubits:
            raise ValueError(f"Expected {self.num_qubits} bits, got {len(bits)}.")
        return int("".join(str(b) for b in bits), 2)
    
    
    def to_bits_array(self, x_array: NDArray[np.integer]) -> NDArray[np.uint8]:
        """Returns the big-endian bitstrings specified by the given integers.

        Args:
            x_array: An integer or array of unsigned integers.
        """
        if is_symbolic(self.num_qubits):
            raise ValueError(f"Cannot compute bits for symbolic bitsize {self.num_qubits}")
        if self.data_width > 64:
            return super().to_bits_array(x_array)
        w = int(self.num_qubits)
        x = np.atleast_1d(x_array)
        if not np.issubdtype(x.dtype, np.uint):
            assert np.all(x >= 0)
            x = x.astype(np.uint64)
        
        mask = 2 ** np.arange(w - 1, -1, -1, dtype=x.dtype).reshape((w, 1))
        return (x & mask).astype(bool).astype(np.uint8).T
    def from_bits_array(self, bits_array: NDArray[np.uint8]) -> NDArray[np.integer]:
        """Returns the integer specified by the given big-endian bitstrings.

        Args:
            bits_array: A bitstring or array of bitstrings, each of which has the 1s bit (LSB) at the end.
        Returns:
            An array of integers; one for each bitstring.
        """
        bitstrings = np.atleast_2d(bits_array)
        if bitstrings.shape[1] != self.data_width:
            raise ValueError(f"Expected bitsize {self.data_width}, got {bitstrings.shape[1]}")
        if self.num_qubits > 64:
            return super().from_bits_array(bits_array)
        basis = 2 ** np.arange(int(self.data_width) - 1, -1, -1, dtype=np.uint64)
        return np.sum(basis * bitstrings, axis=1, dtype=np.uint64)
    def assert_valid_value(self, val: Any, debug_str: str = "val"):
        if not isinstance(val, (int, np.integer)):
            raise ValueError(f"{debug_str} must be an integer, got {val!r}")
        if val < 0 or val >= (1 << self.data_width):
            raise ValueError(f"{debug_str}={val} is out of range for QUInt({self.data_width}).")

    def assert_valid_classical_val(self, val: int, debug_str: str = 'val'):
        if not isinstance(val, (int, np.integer)):
            raise ValueError(f"{debug_str} should be an integer, not {val!r}")
        if val < 0:
            raise ValueError(f"Negative classical value encountered in {debug_str}")
        if val >= 2**self.data_width:
            raise ValueError(f"Too-large classical value encountered in {debug_str}")

    def assert_valid_classical_val_array(
        self, val_array: NDArray[np.integer], debug_str: str = 'val'
    ):
        if np.any(val_array < 0):
            raise ValueError(f"Negative classical values encountered in {debug_str}")
        if np.any(val_array >= 2**self.data_width):
            raise ValueError(f"Too-large classical values encountered in {debug_str}")

    def __str__(self):
        return f'QUInt({self.num_qubits})'
# ----------------------------------------------------------------
# Classical Data Types
# These types are intended to represent classical values and registers.
# They inherit from DataType and implement all required methods.

def _element_size(dtype):
    """
    Returns the element size for a dtype, in bytes
    """
    if not isinstance(dtype, torch.dtype):
        raise RuntimeError(f"expected torch.dtype, but got {type(dtype)}")

    if dtype.is_complex:
        return torch.finfo(dtype).bits >> 2
    elif dtype.is_floating_point:
        return torch.finfo(dtype).bits >> 3
    elif dtype == torch.bool:
        # NOTE: torch.bool is not supported in torch.iinfo()
        return 1
    else:
        return torch.iinfo(dtype).bits >> 3



   

# Classical Bit
@frozen
class CBit(CType):
    """ Represents a single classical bit (0 or 1).
    - num_units=1, bitsize=1.
    - to_bits(...) checks if x in {0,1}.
    """
    bit_width: int = 1  # A classical bit is one unit.


    def to_bits(self, x: int) -> List[int]:
        if x not in (0, 1):
            raise ValueError("Invalid value for a classical bit. Must be 0 or 1.")
        return [x]

    def from_bits(self, bits: Sequence[int]) -> int:
        if len(bits) != 1:
            raise ValueError("Invalid bit sequence for a classical bit. Must have exactly 1 bit.")
        return bits[0]

    def get_classical_domain(self) -> Iterable[Any]:
        """Return the domain of all possible values for a classical bit."""
        return [0, 1]
    
    def assert_valid_classical_val(self, val: Any, debug_str: str = 'val'):
        if val not in (0, 1):
            raise ValueError(f"{debug_str} must be 0 or 1 for CBit; got {val}")
        
    def is_symbolic(self) -> bool:
        return False


@frozen
class CAny(CType):
   
    bit_width: Union[int, sympy.Expr]

    @property
    def element_dtype(self):
        return CBit()

    @property
    def data_width(self) -> int:
        return self.bit_width

    def to_bits(self, x: int) -> List[int]:
        """Convert an integer to its bit representation."""
        if not isinstance(x, int) or x < 0 or x >= 2**self.bit_width:
            raise ValueError(f"Invalid value for CAny with {self.bit_width} bits.")
        return [int(b) for b in bin(x)[2:].zfill(self.bit_width)]

    def from_bits(self, bits: Sequence[int]) -> int:
        """Reconstruct an integer from its bit representation."""
        if len(bits) != self.bit_width:
            raise ValueError(f"Expected {self.bit_width} bits; got {len(bits)}.")
        return int("".join(map(str, bits)), 2)


    def get_classical_domain(self) -> Iterable[Any]:
        """Enumerate all possible values representable by this type."""
        if is_symbolic(self.bit_width):
            raise ValueError("Can't enumerate domain for symbolic bit-size.")
        return range(2 ** self.bit_width)
   
    def assert_valid_classical_val(self, val: Any, debug_str: str = 'val'):
        if not isinstance(val, int) or val < 0 or val >= 2**self.bit_width:
            raise ValueError(f"{debug_str} must be a valid integer for {self.bit_width} bits. Got: {val}")


@frozen
class CInt(CType):
    """
    Classical signed integer (two's complement).
    """
    bit_width: Union[int, sympy.Expr, SymbolicInt]

    @property
    def data_width(self):
        return self.bit_width
    
    def get_classical_domain(self) -> Iterable[int]:
        half = 1 << (self.bit_width - 1)
        return range(-half, half)

    def to_bits(self, x: int) -> List[int]:
        if self.is_symbolic():
            raise ValueError(f"Cannot convert to bits: {self.data_width} is symbolic")
        self.assert_valid_classical_val(x)
        if x < 0:
            # Convert negative value using two's complement.
            x = (1 << self.bit_width) + x
        return [int(b) for b in format(x, f"0{self.bit_width}b")]

    def from_bits(self, bits: Sequence[int]) -> int:
        # TODO: Should we return CUInt type?
        # like: x = (
        #     0
        #     if self.bitsize == 1
        #     else CUInt(self.bitsize - 1).from_bits([1 - x if sign else x for x in bits[1:]])
        # )
        x = int("".join(str(b) for b in bits), 2)
        if bits[0] == 1:  # If MSB is 1, interpret as negative.
            x -= (1 << self.bit_width)
        
        return x

    def assert_valid_classical_val(self, val: Any, debug_str: str = 'val'):
        # TODO: check the comment logic:
        # if not isinstance(val, (int, np.integer)):
        #     raise TypeError(f"{debug_str} must be int for {self}")
        # if not self.is_symbolic():
        #     min_val = -(1 << (int(self.bitsize) - 1))
        #     max_val = (1 << (int(self.bitsize) - 1)) - 1
        #     if val < min_val or val > max_val:
        #         raise ValueError(f"Bad {debug_str}: {val} is out of range for {self} (must be in [{min_val}, {max_val}])")
        # If symbolic, we can't check exact bounds, but Python int is unbounded so assume okay if type is int.
    
        if not isinstance(val, int):
            raise ValueError(f"{debug_str} must be an integer.")
        lower = -(1 << (self.bit_width - 1))
        upper = 1 << (self.bit_width - 1)
        if not (lower <= val < upper):
            raise ValueError(f"{debug_str} must be in [{lower}, {upper}).")
    
    def __str__(self):
        return f"CInt({self.bit_width})"

@frozen
class CUInt(CType):
    """
    Classical unsigned integer of a given bit width (bitsize)
    """
    bit_width: SymbolicInt = field(validator=validators.instance_of((int, sympy.Expr)))
    # (Assume bitsize > 0; could add a validator for positivity)

    def get_classical_domain(self) -> Iterable[int]:
        return range(2 ** self.bit_width)

    def to_bits(self, x: int) -> List[int]:
        self.assert_valid_classical_val(x)
        return [int(b) for b in format(x, f"0{self.bit_width}b")]

    def from_bits(self, bits: Sequence[int]) -> int:
        return int("".join(str(b) for b in bits), 2)

    def assert_valid_classical_val(self, val: Any, debug_str: str = 'val'):
        if not isinstance(val, int):
            raise ValueError(f"{debug_str} must be an integer.")
        if val < 0 or val >= 2 ** self.bit_width:
            raise ValueError(f"{debug_str} must be in the range [0, 2**{self.bit_width}).")
        
class CFixed(DataType):
    def __init__(self, total_bits: int, frac_bits: int, signed: bool = True):
        assert total_bits >= 1 and 0 <= frac_bits <= total_bits
        self.total_bits = total_bits
        self.frac_bits = frac_bits
        self.signed = signed
        # For convenience, compute number of integer bits:
        self.int_bits = total_bits - frac_bits

    def data_width(self) -> int:
        return self.total_bits

    def to_bits(self, value: float) -> List[int]:
        # Convert float to fixed-point integer representation
        scale = 1 << self.frac_bits  # 2^frac_bits
        scaled_val = int(round(value * scale))
        # Mask into range:
        if self.signed:
            # Allow negative values: range [-2^(int_bits-1), 2^(int_bits-1)-1] for integer part
            max_val = (1 << (self.total_bits-1)) - 1
            min_val = -(1 << (self.total_bits-1))
            if scaled_val > max_val or scaled_val < min_val:
                # Overflow handling (saturation or wrap-around; here we wrap)
                scaled_val &= (1 << self.total_bits) - 1
        else:
            # unsigned range [0, 2^total_bits - 1]
            scaled_val %= (1 << self.total_bits)
        # Now get binary (two's complement if signed and negative)
        bits = []
        if self.signed and scaled_val < 0:
            scaled_val = (1 << self.total_bits) + scaled_val
        for i in range(self.total_bits):
            bits.append((scaled_val >> i) & 1)
        return bits

    def from_bits(self, bits: List[int]) -> float:
        if len(bits) != self.total_bits:
            raise ValueError(f"CFixed expects {self.total_bits} bits")
        # Reconstruct integer from bits (LSB-first list assumed)
        raw = 0
        for i, b in enumerate(bits):
            raw |= (b & 1) << i
        # Interpret two's complement if signed:
        if self.signed and (raw >> (self.total_bits - 1)) == 1:
            raw -= (1 << self.total_bits)
        # Convert back to float by scaling down
        value = raw / float(1 << self.frac_bits)
        return value


import struct    
class CUFixed(DataType):
    """
    - Unsigned fixed-point (CUFixed): If needed, we can similarly 
    define CUFixed(total, frac) (or use CFixed(..., signed=False) as above) to represent only non-negative values
    with more range for the integer part. It would interpret all bit patterns as non-negative values (no sign bit).
    
    - 	Quantum analog: A QFxp(total, frac) would typically be stored in total qubits, and operations 
    on it emulate fixed-point arithmetic quantumly. The classical type here 
    would be used in hybrid algorithms to simulate those values or to 
    specify constants. Both share the same bit-width and interpretation, 
    ensuring any classical post-processing (like interpreting a measurement 
    outcome as a fixed-point number) is straightforward.

    """

    pass
class CFloat(DataType):
    """
    Description: CFloat represents a classical floating-point number of a specified precision (e.g., 
    32-bit single precision, 64-bit double precision, etc.). This is a higher-level numeric type 
    compared to CFixed. It’s included to model classical computations that use floating-point arithmetic 
    (for instance, in classical pre/post-processing around quantum steps, or for hardware modules like math coprocessors). 
    
    Note: There is no direct quantum counterpart (since quantum algorithms typically don’t 
    use IEEE 754 floats natively), but CFloat might be used purely on the classical side of a hybrid workflow.

	•	Precision and Format: By default, we can align CFloat(n) with IEEE 754 standard formats for common sizes:
	•	CFloat(32): 32-bit single precision (1 sign bit, 8 exponent bits, 23 fraction bits).
	•	CFloat(64): 64-bit double precision (1 sign, 11 exponent, 52 fraction).
	•	Potentially CFloat(16) for half precision (1 sign, 5 exponent, 10 fraction, per IEEE 754 half).
	•	If n is not one of these standard sizes (or if we choose a simpler model), we must define how to split exponent/fraction bits. One approach: let the user specify explicitly exponent bits and mantissa bits, or assume a reasonable default for non-standard sizes.
    
    """

    def __init__(self, bits: int = 32):
        assert bits in (16, 32, 64, 128), "Unsupported float size"
        self.bits = bits
        # Determine format for struct packing (assuming IEEE754 binary16,32,64,128)
        if bits == 16:
            self.format = 'e'  # half precision (binary16) if supported
        elif bits == 32:
            self.format = 'f'  # single precision
        elif bits == 64:
            self.format = 'd'  # double precision
        elif bits == 128:
            self.format = 'g'  # (nonstandard in Python struct; might use decimal or numpy)
        # Note: Python's struct 'e' requires Python 3.9+ for half precision.
    def data_width(self) -> int:
        return self.bits

    def to_bits(self, value: float) -> List[int]:
        # Handle half precision via numpy if not directly supported
        if self.bits == 128:
            # Use decimal or numpy for quad precision as needed (placeholder).
            raise NotImplementedError("128-bit float conversion not implemented")
        packed = struct.pack(self.format, value)
        # Convert bytes to bit list
        bit_list = []
        for byte in packed:
            for i in range(8):
                bit_list.append((byte >> i) & 1)
        return bit_list

    def from_bits(self, bits: List[int]) -> float:
        if len(bits) != self.bits:
            raise ValueError(f"CFloat expects {self.bits} bits")
        # Reconstruct bytes from bits
        bytes_arr = bytearray()
        for i in range(0, len(bits), 8):
            byte = 0
            for j in range(8):
                byte |= (bits[i+j] & 1) << j
            bytes_arr.append(byte)
        if self.bits == 128:
            raise NotImplementedError("128-bit float conversion not implemented")
        value = struct.unpack(self.format, bytes(bytes_arr))[0]
        return value
    

class CString(DataType):
    def __init__(self, max_length: int):
        self.max_length = max_length  # in characters
        # Each char 1 byte (8 bits). Could allow different encoding in future.

    def data_width(self) -> int:
        return 8 * self.max_length

    def to_bits(self, value: str) -> List[int]:
        # Encode string in ASCII (or UTF-8 truncated to 1 byte per char for simplicity)
        if len(value) > self.max_length:
            raise ValueError("String exceeds max_length")
        # Pad the string to max_length (with null bytes for example)
        s = value.ljust(self.max_length, '\x00')
        bits = []
        for ch in s:
            byte = ord(ch) & 0xFF  # one byte
            # append 8 bits (LSB-first) for this char
            for i in range(8):
                bits.append((byte >> i) & 1)
        return bits

    def from_bits(self, bits: List[int]) -> str:
        if len(bits) != 8 * self.max_length:
            raise ValueError(f"CString expects {8 * self.max_length} bits")
        chars = []
        for i in range(0, len(bits), 8):
            byte = 0
            for j in range(8):
                byte |= (bits[i+j] & 1) << j
            chars.append(chr(byte))
        s = "".join(chars)
        # Remove padding nulls at the end
        s = s.rstrip("\x00")
        return s
    

"""CTensor (Tensor / Array Type)

Description: CTensor(element_type, shape) represents a multi-dimensional array of elements, where element_type is another DataType (could be a primitive like CInt or even another tensor for nested arrays), and shape is a tuple/list of dimensions. This is a generalized array type (vectors, matrices, etc.), analogous to how PyGears uses Array[Type, N] for vectors ￼. It aligns with the concept of an n-qubit register on the quantum side (which can be seen as an array of qubits or structured as multiple data items). While Qualtran’s QDType typically describes the type of a single item, it allows specifying a “shape” for registers of that type in a bloq’s signature. CTensor encapsulates the idea of a typed collection in the classical realm.
	•	Quantum relationship: There isn’t a specific QTensor class in Qualtran; instead, one would use e.g. QInt(4) with a shape (5,) to represent an array of 5 quantum 4-bit integers, or simply QBit() with shape (n,) for an n-qubit register. Our CTensor plays the classical analog: e.g., an array of 5 classical 4-bit integers could be CTensor(CInt(4), shape=(5,)). Thus, conceptually, CTensor(qdtype, shape) ~ array of quantum data vs CTensor(cdtype, shape) ~ array of classical data. Both share the need to handle shapes and potentially dynamic dimensions.
	•	Shape specification: We allow shape to be a tuple or list of dimension sizes. Each dimension can be:
	•	A non-negative integer (for a fixed-size dimension).
	•	A special marker for unknown/dynamic size. (For example, None or some SymDim object.)
	•	In our design, we incorporate dynamic shape support inspired by PyTorch’s symbolic shapes (ShapeEnv) ￼. This means a dimension can be left unspecified until runtime. For instance, CTensor(CInt(32), shape=[None]) could represent “a vector of 32-bit ints of unknown length”. In such cases, data_width() cannot return a concrete number. We might define it to return a symbolic expression (e.g., 8 * n for an unknown n), or simply raise an exception or return None to indicate “not fixed”. A more elaborate solution is to parametrize the type by a symbol (like CTensor(CInt(32), shape=('N',)) and carry that symbol). For this initial design, using None as placeholder is sufficient to indicate dynamic shape.
	•	data_width calculation: If all dimensions are fixed integers, then data_width = element_type.data_width() * (product of all dims). For example, CTensor(CBit(), shape=(8,)) (vector of 8 bits) has width 8. CTensor(CInt(16), shape=(4,4)) (4x4 matrix of 16-bit ints) has width 16 * 16 = 256 bits. If any dimension is dynamic/unknown, data_width() could either:
	•	Return a symbolic form (if we have a symbolic dimension system), or
	•	Return None or raise an error, meaning “this type’s width isn’t fully defined until the shape is instantiated”.
	•	We may include an attribute or method to check is_fully_defined (returns False if shape has an unknown).
	•	to_bits / from_bits: If the shape is fixed, these methods can flatten the multi-dimensional array into a bit list (e.g., row-major order by default) and reconstruct it. If shape is dynamic, we might require the user to supply a runtime shape or the actual data to determine how to flatten. Possibly, the to_bits of a dynamic tensor could include a length prefix or assume the provided value (like a Python list/np.array) knows its length.
	•	Use cases: CTensor covers a wide range of data:
	•	Vectors, matrices for classical linear algebra within algorithms.
	•	Buffers of data (e.g., an array of samples).
	•	Multi-qubit classical registers (like storing quantum measurement outcomes in an array).
	•	In hardware terms, this corresponds to memory or grouped wires (like a bus).
	•	It provides a unified way to treat multi-element data rather than having to define separate types for every array size.

Code – CTensor class:"""
import math
from functools import reduce
from operator import mul

class CTensor(DataType):
    def __init__(self, element_type: DataType, shape: list):
        # shape can include int or None for dynamic dims
        assert isinstance(element_type, DataType)
        self.element_type = element_type
        self.shape = list(shape)  # make a mutable copy

    def data_width(self) -> int:
        # If any dimension is None (unknown), we cannot determine width
        if any(d is None for d in self.shape):
            raise RuntimeError("CTensor has dynamic shape; data_width is undefined until shape is concrete")
        total_elems = 1
        for d in self.shape:
            total_elems *= d  # multiply all dimensions
        return total_elems * self.element_type.data_width()

    def to_bits(self, value) -> List[int]:
        # Flatten the multi-dimensional value (which could be a nested list or numpy array)
        # For simplicity, assume `value` is a nested list matching the shape.
        bits = []
        def flatten(elem, current_shape):
            if len(current_shape) == 0:
                # elem is a scalar (no further dimensions)
                bits.extend(self.element_type.to_bits(elem))
            else:
                # traverse into each sub-element
                expected_len = current_shape[0]
                if expected_len is None:
                    # If dynamic, infer length from the value (len of list)
                    expected_len = len(elem)
                if len(elem) != expected_len:
                    raise ValueError("Tensor value shape mismatch")
                for sub in elem:
                    flatten(sub, current_shape[1:])
        flatten(value, self.shape)
        return bits

    def from_bits(self, bits: List[int]):
        # Reconstruct nested list from flat bits according to shape
        if any(d is None for d in self.shape):
            raise RuntimeError("Cannot reconstruct CTensor with dynamic shape without additional info")
        # Determine number of elements and element bit-width
        elem_width = self.element_type.data_width()
        total_elems = 1
        for d in self.shape:
            total_elems *= d
        expected_bits = total_elems * elem_width
        if len(bits) != expected_bits:
            raise ValueError(f"Bit length {len(bits)} does not match tensor size {expected_bits}")
        # Build nested list by slicing bits for each element
        flat_elems = [self.element_type.from_bits(bits[i*elem_width:(i+1)*elem_width])
                      for i in range(total_elems)]
        # Now reshape flat_elems into the given multi-dimensional shape
        def build_nested(flat_list, shape_dims):
            if not shape_dims:  # no more dims, just a scalar
                return flat_list.pop(0)
            dim = shape_dims[0]
            return [build_nested(flat_list, shape_dims[1:]) for _ in range(dim)]
        nested_value = build_nested(flat_elems, self.shape.copy())
        return nested_value
    
"""
This implementation handles fixed shapes. If shape contains None, we disallow bit conversion without concrete info. In practice, one might carry the actual shape along with the value for dynamic cases.
	•	Dynamic shapes: We drew inspiration from PyTorch’s dynamic shape handling ￼. In a full implementation, we could have a mechanism (like a ShapeEnv) to manage symbolic dimensions. For example, we could allow a SymDim('N') to represent an unknown size named N, and if two such tensors interact we could propagate constraints (similar to how PyTorch unifies symbolic shapes). For now, our design is ready for such an extension: by marking dimensions as None (unknown), we acknowledge that the type’s size isn’t fully defined. This is useful for writing generic algorithms; e.g., a function that accepts CTensor(CBit(), shape=[None]) can operate on a bit-string of any length. The actual size would be known at runtime or at a higher level of the program.
	•	Quantum comparison: In quantum circuits, one often deals with a collection of qubits (like an array of qubits). While in theory we could have a QTensor(QBit(), 5) for 5 qubits, typically frameworks just treat it as “5 qubits” without a distinct type object. But conceptually, having a type for an array of qubits or quantum ints is useful in analysis. Indeed, Qualtran’s Signature uses shapes for registers. So both classical and quantum sides need to reason about multi-item groupings. By using CTensor and letting quantum types also accept a shape parameter, we keep consistent modeling.

"""

"""
CStruct (Structured Record) and Quantum Composite Types

Description: CStruct is a classical composite type analogous to a C/Struct or a record in a programming language: it groups multiple named fields, each of which can be a different data type. For example, one might define CStruct({"header": CUInt(4), "payload": CUInt(8)}) representing a 4-bit header and 8-bit payload packed together (total 12 bits). This is very useful for modeling hardware registers with bit fields, or grouping related data (similar to how PyGears uses tuples/structs ￼).
	•	Quantum relationship: Quantum data can also be grouped (e.g., a quantum algorithm might have a register that is conceptually (QInt(4) key, QInt(4) data) as an 8-qubit state split into two parts). While Qualtran does not have an explicit QStruct class, it allows an operation (Bloq) to have multiple quantum inputs of possibly different types. A Signature in Qualtran can hold an aggregate of varied types labeled by name. Our CStruct mirrors this classical side: it can carry both numeric and non-numeric types in one bundle, and by having field names, it aligns with how quantum circuit “signatures” or interfaces label separate quantum registers. If needed, one could define a similar QStruct in the future for symmetry, but it may be unnecessary if quantum interfaces handle multiple named types inherently.
	•	Layout and Padding: We assume packed struct (no padding bits between fields) by default, to keep data_width simply the sum of field widths. This matches hardware modeling in bit-precise DSLs (e.g., SystemVerilog struct packed behaves this way, and PyGears’ Tuple is essentially a concatenation of fields). If needed, one could later introduce alignment rules or explicit padding fields for particular architectures, but that would be additional metadata. Initially, CStruct will be purely bit-packed for simplicity and clarity in resource counting (just add up the bits).
	•	Semantic info (field names): CStruct not only carries the types of each field but also their names. This is important for readability and usage – e.g., we can access packet.fields["header"] easily, and when generating documentation or diagrams, the field names can be shown. It also aids correctness: by naming fields, we reduce the chance of mixing up bit positions, and can automatically extract sub-bitstreams by field. The presence of names doesn’t change the bit-level behavior, but it’s crucial semantic metadata (the user explicitly asked if field names, layout, etc., are carried – our answer is yes, we carry field names and define the layout as packed).
	•	Interface: data_width() = sum of each field’s data_width(). to_bits(value) expects value to be provided as a dictionary or object with attributes for each field, and it concatenates each field’s bits (likely in a fixed order, say the order of definition). We must decide an ordering: e.g., we could define that the first field’s bits come in the least significant part or most significant part. In hardware, when packing structs, typically the order in a packed struct is from MSB to LSB as declared. For our bitlist (LSB-first convention we used earlier for numbers), we might append fields in an agreed sequence. Let’s say we concatenate in the order fields were defined, with the first field’s bits first in the list (which would correspond to the lower-index bits if we treat bit[0] as LSB overall). The exact convention just needs to be consistent between to_bits and from_bits.
	•	domain_size: The number of possible distinct structs is the product of domain_sizes of the fields (assuming fields are independent). Since each field is just a segment of bits, effectively domain_size = 2^total_bits if all combinations of field bits are allowed (which they are, since we pack without cross-bit constraints). But if some fields have restricted ranges (like a field might be an CInt(3) but maybe only certain values are valid in context), that’s semantic beyond the type system. So by type alone, domain is product of each field’s domain.

Code – CStruct class:

"""
class CStruct(DataType):
    def __init__(self, fields: dict):
        """
        fields: OrderedDict or dict of field_name -> DataType.
        We will preserve insertion order if a regular dict (Python 3.7+ guarantees dict order).
        """
        # Ensure all values are DataType
        for k,v in fields.items():
            assert isinstance(v, DataType), f"Field {k} is not a DataType"
        self.fields = dict(fields)  # preserve order of insertion (py3.7+)
        # Compute field order and bit offsets if needed for later (not strictly necessary for packed)
        self.field_order = list(self.fields.keys())

    def data_width(self) -> int:
        # Sum of field widths
        return sum(field_type.data_width() for field_type in self.fields.values())

    def to_bits(self, value: dict) -> List[int]:
        # Expect value as a dict mapping field names to their values
        bits = []
        for name in self.field_order:
            field_type = self.fields[name]
            field_val = value[name]
            field_bits = field_type.to_bits(field_val)
            bits.extend(field_bits)
        return bits

    def from_bits(self, bits: List[int]):
        # Split bit list according to field sizes (assuming order in field_order)
        result = {}
        idx = 0
        for name in self.field_order:
            field_type = self.fields[name]
            w = field_type.data_width()
            field_bits = bits[idx: idx+w]
            if len(field_bits) < w:
                raise ValueError("Bit list too short for struct fields")
            result[name] = field_type.from_bits(field_bits)
            idx += w
        return result
    

"""
This packs fields in the given order. For example, if we have fields = {"a": CUInt(3), "b": CBit()}, then to_bits({"a":5, "b":1}) will produce bits of a followed by bits of b, resulting in a 4-bit list. We could document that "a"’s bits come first (bit0–bit2), and "b" is bit3, for instance. If a different bit ordering is needed (like reversing field order for endianness reasons), one can adjust the implementation. The main point is that CStruct cleanly encapsulates heterogeneous data under one type.
	•	Extensibility: We might want methods to get or set a field by name, or iterate fields, etc., but those are supplemental. For hardware layout, if alignment/padding were needed, we could insert padding fields explicitly (e.g., a field name like _pad0 with CUInt(pad_bits)). Since we focus on bit-accurate modeling, we leave out implicit padding entirely.
	•	Quantum composite: While we don’t explicitly implement a QStruct, one could imagine similar behavior: grouping multiple QDType together. Qualtran’s approach is to use tuples of types in a signature rather than a single struct type. If in a later design we wanted to formalize a quantum struct, it would conceptually do the same (concatenate qubits of subtypes). For now, the classical struct serves in classical control or data structures that accompany quantum logic.

"""

"""CUnion (Tagged Union / Variant) vs Quantum Variants

Description: CUnion represents a classical variant type – a value that can take one of several types at a time. It’s akin to a union in C (but we will include a tag to indicate which alternative is present, making it more like a sum type in typed languages). For instance, CUnion({"x": CUInt(16), "y": CUInt(8)}) could represent a 16-bit value or an 8-bit value. Internally, such a union needs to store enough bits for either option and a tag to distinguish them. This is useful in modeling scenarios where a classical signal or data path can carry different types of payloads (for example, a multiplexed bus carrying either an address or data with different sizes). PyGears explicitly supports unions as tagged unions, where a control bit (or bits) selects the interpretation ￼.
	•	Quantum relationship: Quantum computing doesn’t have a direct analog of a tagged union, because a quantum state can be in a superposition of types (which is a different concept). However, one could simulate a union on a quantum level by allocating an extra qubit(s) as a tag and using controlled operations – effectively treating the combined system as a direct sum of Hilbert spaces. In practice, we might not define a QUnion explicitly. If needed, a quantum algorithm that conditionally uses one type of register vs another could be modeled by a larger register and controls (similar to how one would implement a union in classical hardware). For our design, CUnion is mainly a classical type, but if a quantum algorithm has distinct branches with different data, a classical union type might be used in the hybrid workflow to represent the classical information about “which branch/type”.
	•	Encoding: To implement a union in hardware or bits, we typically do:
	•	Tag field: an integer tag to identify which variant is active. If there are N variants, we need ⌈log2 N⌉ bits for the tag (or exactly N bits for one-hot encoding, but binary encoding is more compact).
	•	Data field: enough bits to hold the largest variant. Commonly, we allocate memory equal to the max size of any variant so that any of them fits. In a hardware sense, the circuits for the smaller variant might leave the higher-order bits unused. Alternatively, one could pack variants in the exact needed bits and interpret the tag to decide how to slice the bits – but that complicates a uniform representation. It’s simpler to treat the union as a fixed size equal to max(bit-width of options) plus tag bits.
	•	In PyGears’ example Union[Uint[16], Uint[8]], they used 1 control bit (since 2 variants) and a data field of 16 bits (the larger) ￼. The smaller 8-bit value occupies the lower 8 bits of the 16-bit field when active, and the extra bits might be don’t-care. We will follow this approach: data width = tag_bits + max(option widths). The tag will be an unsigned integer indicating the index of the active type (e.g., 0 for the first type, 1 for second, etc.). We’ll use binary encoding of tag.
	•	Interface: data_width() = tag_width + max(option.data_width()). By default, tag_width = ceil(log2(N)) where N is number of variants. (We could also allow specifying a custom tag width or encoding if needed, but binary minimal encoding is standard.) to_bits(value) would accept something like a tuple (tag, val) or a special object indicating which field is set. Perhaps more conveniently, we can require value be a tuple of form ("field_name", field_value), i.e., explicitly stating which variant. Then to_bits will:
	•	Determine which variant’s name matches, set the tag bits accordingly, then encode the value using that field’s type into bits, and pad or extend to full data width.
	•	If the provided variant’s bit-width is smaller than the max, we must decide how to arrange it. E.g., we can put it in the lower-order bits and pad the rest with zeros. This aligns with treating the union memory as if it were a struct with overlapping fields – typically all variants start at bit0 position in hardware, and if one is shorter, it just doesn’t use the higher bits.
	•	from_bits(bits) will read the tag bits to know which variant is active, then decode the data bits accordingly and return a tuple or similar identifying the active variant and value.
	•	domain_size: If we consider only semantically valid states, the number of possible values is the sum of the domain_sizes of each variant (since only one variant is valid at a time). However, if we treat the union as raw bits (with a particular encoding scheme), some bit patterns might be unused (e.g., if using binary tags, some tag values might be invalid if not all combinations are used; or if using a padded data field, certain high-order bits are irrelevant when a smaller variant is active). Ideally, our from_bits would never produce an “invalid” result because any bit pattern will be interpreted somehow (though possibly meaningless if an invalid tag is present, but we could restrict tag range). For simplicity, we assume tag encoding covers exactly the needed variants (no invalid tag values), so every bit pattern corresponds to exactly one variant’s value. Then domain_size = 2^data_width (full bit space) in raw terms, but effectively = sum of each variant’s domain (which is slightly less if not power of two, but the unused combos are just those extra tag patterns if any).

Code – CUnion class:


"""

import math

class CUnion(DataType):
    def __init__(self, variants: dict):
        """
        variants: dict of variant_name -> DataType.
        """
        assert len(variants) >= 2, "Union needs at least two variants"
        for v in variants.values():
            assert isinstance(v, DataType), "Variant type must be a DataType"
        self.variants = dict(variants)
        self.variant_order = list(self.variants.keys())
        self.tag_bits = math.ceil(math.log2(len(self.variant_order)))
        # If len is power of two, all tag patterns used; if not, some unused patterns exist
        # Compute max data bits among variants:
        self.max_bits = max(v.data_width() for v in self.variants.values())

    def data_width(self) -> int:
        return self.tag_bits + self.max_bits

    def to_bits(self, value) -> List[int]:
        """
        Expect value as tuple (variant_name, variant_value).
        Alternatively, could accept a dict {name: value} with exactly one key.
        """
        if isinstance(value, tuple) and len(value) == 2:
            var_name, var_val = value
        elif isinstance(value, dict) and len(value)==1:
            var_name = next(iter(value.keys()))
            var_val = value[var_name]
        else:
            raise ValueError("Union value must be (name, value) or {name: value}")
        if var_name not in self.variant_order:
            raise ValueError(f"{var_name} is not a valid variant name")
        tag_index = self.variant_order.index(var_name)
        # Encode tag (binary)
        tag_bits_list = [(tag_index >> i) & 1 for i in range(self.tag_bits)]
        # Encode data
        data_type = self.variants[var_name]
        data_bits_list = data_type.to_bits(var_val)
        # Pad or truncate data_bits_list to max_bits
        if len(data_bits_list) < self.max_bits:
            # pad with zeros for remaining bits (assuming LSB-first ordering, pad at MSB side)
            data_bits_list.extend([0] * (self.max_bits - len(data_bits_list)))
        elif len(data_bits_list) > self.max_bits:
            data_bits_list = data_bits_list[:self.max_bits]  # truncate if somehow oversized
        # Combine tag and data bits. Decide order: let's put tag bits in front (e.g., as LSBs).
        # We can place tag as the lowest-order bits or highest-order bits. For simplicity, prepend tag bits.
        bits = []
        bits.extend(tag_bits_list)
        bits.extend(data_bits_list)
        return bits

    def from_bits(self, bits: List[int]):
        if len(bits) != self.tag_bits + self.max_bits:
            raise ValueError(f"CUnion expects {self.tag_bits + self.max_bits} bits")
        # Extract tag and data parts (assuming we put tag first in to_bits)
        tag_bits_list = bits[:self.tag_bits]
        data_bits_list = bits[self.tag_bits:]
        # Decode tag
        tag_val = 0
        for i, b in enumerate(tag_bits_list):
            tag_val |= (b & 1) << i
        if tag_val >= len(self.variant_order):
            raise ValueError("Invalid tag value for CUnion")
        var_name = self.variant_order[tag_val]
        data_type = self.variants[var_name]
        # The data_bits_list might be larger than the type's width (if padded).
        # We pass only the needed lower bits to from_bits of the variant type.
        w = data_type.data_width()
        data_bits_for_val = data_bits_list[:w]
        var_val = data_type.from_bits(data_bits_for_val)
        return (var_name, var_val)
    

"""
In this implementation:
	•	We compute tag_bits as the minimum bits to encode all variant indices. For example, 2 variants -> 1 tag bit, 3-4 variants -> 2 tag bits (with one unused combination if 3 variants), etc.
	•	We combine tag bits followed by data bits in the bit list. (We made an arbitrary but consistent choice to put tag first in the list, which would correspond to it being the least significant bits if you treat the concatenation as an integer. Alternatively, one could append tag at the end. The exact bit-order convention can be adjusted globally, but this is a detail.)
	•	from_bits uses the tag to dispatch to the correct variant’s from_bits.

Example: Suppose CUnion({"a": CUInt(16), "b": CUInt(8)}). Then tag_bits=1, max_bits=16, data_width=17. A value ("b", 0xAB) would produce:
	•	tag for “b” (index 1) = [1]
	•	data bits for 0xAB (8-bit) = e.g. [1,0,1,0,1,0,1,1] (LSB-first for 0xAB). Padded to 16 bits by adding 8 zeros: total 16 data bits.
	•	Combined bits = [1] + [1,0,1,0,1,0,1,1, 0,0,0,0,0,0,0,0] (17 bits).
If those bits are fed to from_bits, it reads tag=1 => variant “b”, then takes the first 8 bits of data part (it knows b is 8-bit) which are the original 0xAB bits, decodes to 0xAB. Result ("b", 0xAB).
	•	Use cases: Unions allow modeling classical control signals that can carry different kinds of info. For example, a quantum program might have a classical return value that is either an integer or an error flag; this could be a union of an int and an error code type. In hardware, a bus might on one cycle carry a 32-bit address (variant A) and on another a 32-bit data (variant B); a union type can represent the bus data with tag indicating what it is. The tagging ensures we don’t misinterpret one as the other.
	•	One-hot alternative: We used binary tags for compactness. Some hardware protocols use one-hot encoding (e.g., separate lines for each variant). That would use N tag bits for N variants, and one of them is 1 to indicate the active. Our framework could accommodate that by either a different encoding mode or by treating one-hot as a CUInt(N) with exactly one bit set (but that doesn’t enforce single-bit at type level). For now, binary tags are fine; they minimize data_width.
	•	Quantum notes: If one attempted a quantum union, the tag qubit could be entangled with the data qubits (for example, if tag |0> means the data qubits represent a 16-bit int and tag |1> means an 8-bit int plus 8 junk qubits). But this is complicated and usually avoided by structuring algorithms differently. So likely we won’t directly use CUnion on the quantum side, but it’s a powerful classical modeling tool.

UnknownType and Dynamic/Undefined Dimensions

To support dynamic dimensions and future extension, we introduce placeholders for unknown types or shapes:
	•	UnknownType: an instance of DataType that represents a type that is not yet defined or is context-dependent. This could be used in a generic algorithm before the concrete type is substituted. For example, if we had a function that should work on any type, its parameter might be annotated as UnknownType until bound. In practice, UnknownType.data_width() might raise an error or return 0 or -1 to signify “unknown.” It’s mainly a placeholder to keep the type system consistent (you can put an UnknownType in a struct or use it in a union, to be filled in later). It might also represent a type variable in templates.
	•	Dynamic shape indicator: For arrays/tensors, as discussed, we allow None in the shape tuple to indicate an unknown dimension length. Internally, we could treat None as a special sentinel or replace it with a SymDim object if more detail is needed (like a symbol name or constraints). The presence of None means the CTensor is not fully resolved. We demonstrated how CTensor.data_width() checks for this and can throw an error if asked to produce a concrete width ￼. In a more advanced design, we could integrate with a ShapeEnv that tracks relations: e.g., if a function says it returns a tensor of shape (N,) given an input of shape (N,), the UnknownType or symbolic N ties them together.
	•	Undefined type usage: In hardware modeling, an undefined type might correspond to a black box module’s input/output that we haven’t decided the width of yet (maybe waiting for some configuration). By allowing an UnknownType, we can still construct a larger structure and fill in later. This is analogous to using generics or templates. For now, we simply acknowledge it.

Code – UnknownType:
"""
class UnknownType(DataType):
    def __init__(self, description: str = "unknown"):
        self.description = description

    def data_width(self) -> int:
        raise RuntimeError(f"Data width of {self.description} is not yet known")

    def to_bits(self, value):
        raise RuntimeError(f"Cannot convert unknown type {self.description}")

    def from_bits(self, bits: List[int]):
        raise RuntimeError(f"Cannot reconstruct unknown type {self.description}")
    

"""
This class will throw if used incorrectly. In practice, one wouldn’t call those methods until the unknown is resolved to a concrete type. One might replace an UnknownType with a real DataType during a binding phase of the workflow.
	•	“Unknown shape”: We did not create a separate UnknownShapeType because we handle it within CTensor by marking shape parts as unknown. Alternatively, we could have a special case like CTensor(UnknownType(), shape=...) if the element type itself is unknown (which could happen if you know you have an array of some type but not which type yet). Both element type and shape could be unknown in theory – but at that point, nearly everything is dynamic.
	•	Future integration: These dynamic aspects are placeholders for more complex features. They ensure our system is ready for dynamic shapes, even if the initial implementation might restrict how they can be used (e.g., not allowing certain operations until resolved). The idea parallels PyTorch’s DynamicDim and shape functions, albeit at a simpler capacity here.

Semantic Information vs Resource Estimation Considerations

The classical types defined above serve two primary purposes:
	1.	Resource Estimation: At a minimum, each type knows how many bits it comprises, which is crucial for estimating resource usage (qubits, classical memory bits, gate counts for operations on them, etc.). For example, a CInt(32) tells us 32 classical bits are needed, a CStruct of certain fields tells us the total bits for that composite, and so on. This aligns with Qualtran’s goal of detailed resource accounting for quantum algorithms – by extending types to classical data, we can also estimate classical processing resources integrated into the algorithm.
	2.	Semantic Richness: Beyond bit counts, our types carry meaningful metadata:
	•	Field names in CStruct give semantic labels to bits.
	•	CFixed knows how to interpret those bits as a scaled real number (and how many are fractional).
	•	CUnion encodes the intent that only one of the variants is valid at a time, and includes the tag mechanism.
	•	CString encapsulates character encoding.
	•	CTensor explicitly represents multi-dimensional structure instead of treating everything as a flat bit heap.
This semantic info is useful for correctness and clarity. It allows the framework to do type-checking (e.g., prevent connecting a 16-bit int to a 13-bit fixed-point without explicit conversion), to print human-readable structure (like “struct {header, payload}”), or to interface with external systems (e.g., generating VHDL/Verilog types or C data structures from these definitions).

It’s worth noting that in purely classical hardware description (like SystemVerilog or PyGears), these types would eventually be lowered to bits and logic. For instance, PyGears can automatically generate a struct/union in SystemVerilog from its type description ￼ ￼. In our hybrid context, we keep the types abstract in the modeling layer, and we could generate appropriate lower-level representations for simulation or compilation.

IEEE 754 vs Simplified Floats: We mentioned the choice for CFloat. If the primary goal is resource estimation, one might say: a 32-bit float is just 32 bits, and perhaps count a floating-point adder as some fixed cost. The internal format (IEEE vs not) might not matter for counting bits, but it does matter if we simulate or if we derive circuits. We opted to use IEEE where possible to maintain realism. Another approach for simplicity is to treat CFloat as a struct of sign/exponent/mantissa bits (which is effectively what IEEE is) – one could even model it as CStruct({"sign": CBit(), "exponent": CUInt(e_bits), "fraction": CUInt(m_bits)}). That might ease certain analyses (like handling each part separately for a custom floating-point arithmetic circuit). However, it complicates treating it as a single value. So for now, CFloat wraps the standard behavior; advanced users could break it down if needed.

Summary: We have defined a robust set of classical data types – CBit, CInt/CUInt, CFixed, CFloat, CString, CTensor, CStruct, CUnion – each mirroring or extending concepts from Qualtran’s quantum types and from classical hardware data modeling frameworks (e.g., custom-width integers and fixed-point from HeteroCL ￼, struct/union/array from PyGears and HDLs ￼ ￼). All types share a common interface (DataType) ensuring that operations like bit conversion and width calculation are uniformly accessible ￼ ￼. We have also built in support for dynamic shapes and undefined types to make the system future-proof for more dynamic algorithm descriptions.

These classical types will enable a modular design where quantum and classical components can be specified with analogous detail. For example, a hybrid algorithm might be described with a signature containing both QInt(8) quantum inputs and CInt(8) classical parameters – both appear as 8-bit integers conceptually, but one is quantum. The symmetry in the type system makes such representations clear and helps in analyzing how classical pre/post-processing intertwines with quantum processing in a full workflow.  ￼ ￼
"""

   
import math

class CUnion(DataType):
    """
    In this implementation:
    	•	We compute tag_bits as the minimum bits to encode all variant indices. For example, 
        2 variants -> 1 tag bit, 3-4 variants -> 2 tag bits (with one unused combination if 3 variants), etc.
        •	We combine tag bits followed by data bits in the bit list. (We made an arbitrary but consistent 
        choice to put tag first in the list, which would correspond to it being the least significant bits 
        if you treat the concatenation as an integer. Alternatively, one could append tag at the end. The 
        exact bit-order convention can be adjusted globally, but this is a detail.)
    """
    def __init__(self, variants: dict):
        """
        variants: dict of variant_name -> DataType.
        """
        assert len(variants) >= 2, "Union needs at least two variants"
        for v in variants.values():
            assert isinstance(v, DataType), "Variant type must be a DataType"
        self.variants = dict(variants)
        self.variant_order = list(self.variants.keys())
        self.tag_bits = math.ceil(math.log2(len(self.variant_order)))
        # If len is power of two, all tag patterns used; if not, some unused patterns exist
        # Compute max data bits among variants:
        self.max_bits = max(v.data_width() for v in self.variants.values())

    def data_width(self) -> int:
        return self.tag_bits + self.max_bits

    def to_bits(self, value) -> List[int]:
        """
        Expect value as tuple (variant_name, variant_value).
        Alternatively, could accept a dict {name: value} with exactly one key.
        """
        if isinstance(value, tuple) and len(value) == 2:
            var_name, var_val = value
        elif isinstance(value, dict) and len(value)==1:
            var_name = next(iter(value.keys()))
            var_val = value[var_name]
        else:
            raise ValueError("Union value must be (name, value) or {name: value}")
        if var_name not in self.variant_order:
            raise ValueError(f"{var_name} is not a valid variant name")
        tag_index = self.variant_order.index(var_name)
        # Encode tag (binary)
        tag_bits_list = [(tag_index >> i) & 1 for i in range(self.tag_bits)]
        # Encode data
        data_type = self.variants[var_name]
        data_bits_list = data_type.to_bits(var_val)
        # Pad or truncate data_bits_list to max_bits
        if len(data_bits_list) < self.max_bits:
            # pad with zeros for remaining bits (assuming LSB-first ordering, pad at MSB side)
            data_bits_list.extend([0] * (self.max_bits - len(data_bits_list)))
        elif len(data_bits_list) > self.max_bits:
            data_bits_list = data_bits_list[:self.max_bits]  # truncate if somehow oversized
        # Combine tag and data bits. Decide order: let's put tag bits in front (e.g., as LSBs).
        # We can place tag as the lowest-order bits or highest-order bits. For simplicity, prepend tag bits.
        bits = []
        bits.extend(tag_bits_list)
        bits.extend(data_bits_list)
        return bits

    def from_bits(self, bits: List[int]):

        """
        from_bits uses the tag to dispatch to the correct variant’s from_bits.

        Example: Suppose CUnion({"a": CUInt(16), "b": CUInt(8)}). Then tag_bits=1, max_bits=16, data_width=17. A value ("b", 0xAB) would produce:
            •	tag for “b” (index 1) = [1]
            •	data bits for 0xAB (8-bit) = e.g. [1,0,1,0,1,0,1,1] (LSB-first for 0xAB). Padded to 16 bits by adding 8 zeros: total 16 data bits.
            •	Combined bits = [1] + [1,0,1,0,1,0,1,1, 0,0,0,0,0,0,0,0] (17 bits).
        If those bits are fed to from_bits, it reads tag=1 => variant “b”, then takes the first 8 bits of data 
        part (it knows b is 8-bit) which are the original 0xAB bits, decodes to 0xAB. Result ("b", 0xAB).
        """
        if len(bits) != self.tag_bits + self.max_bits:
            raise ValueError(f"CUnion expects {self.tag_bits + self.max_bits} bits")
        # Extract tag and data parts (assuming we put tag first in to_bits)
        tag_bits_list = bits[:self.tag_bits]
        data_bits_list = bits[self.tag_bits:]
        # Decode tag
        tag_val = 0
        for i, b in enumerate(tag_bits_list):
            tag_val |= (b & 1) << i
        if tag_val >= len(self.variant_order):
            raise ValueError("Invalid tag value for CUnion")
        var_name = self.variant_order[tag_val]
        data_type = self.variants[var_name]
        # The data_bits_list might be larger than the type's width (if padded).
        # We pass only the needed lower bits to from_bits of the variant type.
        w = data_type.data_width()
        data_bits_for_val = data_bits_list[:w]
        var_val = data_type.from_bits(data_bits_for_val)
        return (var_name, var_val)