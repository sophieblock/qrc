from __future__ import annotations

from typing import overload,Iterable, Any, Union, List, Sequence, Optional, Tuple
import itertools
import numpy as np
from numpy.typing import NDArray
from torch import SymInt
import sympy
import torch
from fxpmath import Fxp
from enum import Enum
import abc
import struct

from attrs import define, field, validators,frozen

from ...util.log import logging
logger = logging.getLogger(__name__)


from typing import TypeVar

SymbolicFloat = Union[float, sympy.Expr]
"""A floating point value or a sympy expression."""

SymbolicInt = Union[int, sympy.Expr]
"""An integer value or a sympy expression."""

SymbolicComplex = Union[complex, sympy.Expr]
"""A complex value or a sympy expression."""


T = TypeVar('T')
from typing_extensions import TypeIs
@frozen
class Shaped:
    """Symbolic value for an object that has a shape.

    A Shaped object can be used as a symbolic replacement for any object that has an
    attribute `shape`, for example numpy `NDArrays`. Each dimension can be either
    a positive integer value or a sympy expression.

    For the symbolic variant of a tuple or sequence of values, see `HasLength`.

    This is useful to do symbolic analysis of Bloqs whose call graph only depends on the shape
    of the input, but not on the actual values. For example, T-cost of the `QROM` Bloq depends
    only on the iteration length (shape) and not on actual data values. In this case, for the
    bloq attribute `data`, we can use the type:

    source: qualtran
    """

    shape: tuple[SymbolicInt, ...] = field(validator=validators.instance_of(tuple))

    def is_symbolic(self):
        return True


@frozen
class HasLength:
    """Symbolic value for an object that has a length.

    This is used as a "symbolic" tuple. The length can either be a positive integer
    or a sympy expression. For example, if a bloq attribute is a tuple of ints,
    we can use the type:

    ```py
    values: Union[tuple, HasLength]
    ```

    For the symbolic variant of a NDArray, see `Shaped`.

    Note that we cannot override __len__ and return a sympy symbol because Python has
    special treatment for __len__ and expects you to return a non-negative integers.

    See https://docs.python.org/3/reference/datamodel.html#object.__len__ for more details.
    source: qualtran
    """

    n: SymbolicInt

    def is_symbolic(self):
        return True
@overload
def is_symbolic(
    arg: Union[T, sympy.Expr, Shaped, HasLength], /
) -> TypeIs[Union[sympy.Expr, Shaped, HasLength]]: ...


@overload
def is_symbolic(*args) -> bool: ...


def is_symbolic(*args) -> Union[TypeIs[Union[sympy.Expr, Shaped, HasLength]], bool]:
    """Returns whether the inputs contain any symbolic object.

    Returns:
        True if any argument is either a sympy object,
        or implements the `is_symbolic` method which returns True.
    """

    if len(args) != 1:
        return any(is_symbolic(arg) for arg in args)

    (arg,) = args
    if isinstance(arg, sympy.Basic):
        return True

    checker = getattr(arg, 'is_symbolic', None)
    if checker is not None:
        return checker()

    return False


SymbolicT = TypeVar('SymbolicT', SymbolicInt, SymbolicFloat, SymbolicComplex)



def prod(args: Iterable[SymbolicT]) -> SymbolicT:
    ret: SymbolicT = 1
    for arg in args:
        ret = ret * arg
    return ret



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
        return hash("_DynType")

Dyn = _DynType()


class DataType(abc.ABC):
    """
    Abstract parent for all data types (classical or quantum).

    Each subclass must implement:
    - num_units -> int: The total “element count” or bit/qubit count (depending on the type).
    - data_width: Total bits required to store one instance of this data type.
    - to_bits(...) / from_bits(...): For converting this data type to and from a bit-level representation.
    - get_classical_domain(): If feasible, yields all possible values (e.g., for small classical types).
    - to_units(...) / from_units(...): Splits or reconstructs the data into smaller “units.”

    This design ensures that the shape or size of the data is primarily stored here, making
    the `Data` class in `data.py` simpler in handling dynamic aspects like symbolic shapes.
    """
    @property
    @abc.abstractmethod
    def data_width(self) -> int:
        """
        Number of "fundamental units" (bits, qubits, or something else)
        required to represent a single instance of this data type.
        """

    @property
    @abc.abstractmethod
    def data_width(self) -> int:
        """
        Total number of bits needed to represent one full instance.
        Could be `num_units` * 1 for qubits, or rows*cols*8 for a float64 matrix, etc.
        """

    @abc.abstractmethod
    def to_bits(self, x) -> List[int]:
        """Convert a single value x to its bit representation."""

    def to_bits_array(self, x_array: NDArray[Any]) -> NDArray[np.uint8]:
        """Yields an NDArray of bits corresponding to binary representations of the input elements.

        Often, converting an array can be performed faster than converting each element individually.
        This operation accepts any NDArray of values, and the output array satisfies
        `output_shape = input_shape + (self.data_width,)`.
        """
        return np.vectorize(
            lambda x: np.asarray(self.to_bits(x), dtype=np.uint8), signature='()->(n)'
        )(x_array)
    @abc.abstractmethod
    def from_bits(self, bits: Sequence[int]) -> Any:
        """Combine bits to form a single value x."""
    
    def from_bits_array(self, bits_array: NDArray[np.uint8]):

        """Combine individual bits to form classical values.

        Often, converting an array can be performed faster than converting each element individually.
        This operation accepts any NDArray of bits such that the last dimension equals `self.data_width`,
        and the output array satisfies `output_shape = input_shape[:-1]`.
        """
        return np.vectorize(self.from_bits, signature='(n)->()')(bits_array)

    @abc.abstractmethod
    def get_classical_domain(self) -> Iterable[Any]:
        """Yield all possible values representable by this type (if feasible)."""

    

    @abc.abstractmethod
    def assert_valid_classical_val(self, val: Any, debug_str: str = 'val'):
        """Raises an exception if `val` is not a valid classical value for this type.

        Args:
            val: A classical value that should be in the domain of this QDType.
            debug_str: Optional debugging information to use in exception messages.
        """
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
        return f"{self.__class__.__name__}({self.data_width})"
    def __format__(self, spec: str) -> str:          # <-- add this
        """Delegate formatting to the string representation."""
        return format(str(self), spec)    

# ----------------------------------------------------------------
# Classical Data Types
# These types are intended to represent classical values and registers.
# They inherit from DataType and implement all required methods.

def _bits_for_dtype(dt) -> int:
    """
    Return the bit-width for a scalar dtype:
        - torch.dtype (via torch.iinfo/finfo)
        - numpy dtypes or scalar classes
        - Python built-ins (float => 64, int => 64)
    """
    import torch, numpy as np
    if isinstance(dt, torch.dtype):
        if dt.is_floating_point:
            return torch.finfo(dt).bits
        if dt == torch.bool:
            return 1
        return torch.iinfo(dt).bits
    if isinstance(dt, np.dtype):
        return dt.itemsize * 8
    try:
        npdt = np.dtype(dt)
        return npdt.itemsize * 8
    except Exception:
        pass
    
    # Possibly default to 64 (or 32)
    if dt is float:
        return 32
    if dt is int: 
        return 32
    
    return 32  



def _element_type_converter(et: Any) -> 'DataType':
    """
    Field converter for `element_type`.
    If user does not supply one (None), default to CFloat(32).
    Otherwise, if already a DataType, return it.
    Else, raise TypeError.
    """

    if isinstance(et, DataType):
        return et
    # raise TypeError(f"element_type must be a DataType or None, got {et}")
    et_size = _bits_for_dtype(et)
    
    return CFloat(et_size)


def _to_symbolic_int(v):
    """Accept int, SymPy, Dyn, or str → SymPy.Symbol."""
    if isinstance(v, str):
        return sympy.symbols(v, positive=True, integer=True)
    return v

# ──────────────────────────────────────────────────────────────────────────────
#  Light-weight VALUE helper – orthogonal to the dtype hierarchy
# ──────────────────────────────────────────────────────────────────────────────
from enum import Enum

class BitNumbering(Enum):
    MSB = "msb"          # bit-0 is most-significant (big-endian bit order)
    LSB = "lsb"          # bit-0 is least-significant (little-endian bit order)



class BitString:
    
    """
    A mutable *value* container that knows nothing about resource checks.
    It stores an `integer` and (optionally) a `dtype` whose width it must fit.

    """
    # ------------------------------------------------------------------ ctor
    def __init__(
        self,
        integer: int = 0,
        *,
        nbits: int | None = None,
        numbering: BitNumbering = BitNumbering.MSB,
        dtype: "DataType | None" = None,
    ):
        self.numbering = numbering
        self._value: int = int(integer)
        self.nbits: int = nbits or max(1, self._value.bit_length())
        self.dtype: "DataType | None" = dtype
        # if a dtype is supplied, make sure the value fits
        if dtype is not None:
            dtype.assert_valid_classical_val(self._value, "BitStringView.integer")
            self.nbits = max(self.nbits, dtype.data_width)

    # identical helpers for array / binary if you want them later …
    # ---------------------------------------------------------------- integer
    @property
    def integer(self) -> int:
        return self._value

    @integer.setter
    def integer(self, other: int):
        self._value = int(other)
        self.nbits = max(self.nbits, self._value.bit_length())

    # ---------------------------------------------------------------- binary/array
    @property
    def binary(self) -> str:
        w = max(self.nbits, 1)
        bits = format(self._value, f"0{w}b")
        return bits[::-1] if self.numbering is BitNumbering.LSB else bits

    @binary.setter
    def binary(self, other: str):
        if other.startswith("0b"):
            other = other[2:]
        if self.numbering is BitNumbering.LSB:
            other = other[::-1]
        self._value = int(other, 2)
        self.nbits = max(self.nbits, len(other))
    
    @property
    def array(self) -> list[int]:
        return self.bits  # alias
    

    @property
    def bits(self) -> list[int]:
        width = max(self.nbits, 1)
        bitstr = format(self._value, f"0{width}b")
        if self.numbering is BitNumbering.LSB:
            bitstr = bitstr[::-1]
        return [int(b) for b in bitstr]

    @bits.setter
    def bits(self, seq: list[int]):
        if self.numbering is BitNumbering.LSB:
            seq = seq[::-1]
        self._value = int("".join(str(b) for b in seq), 2)
        self.nbits = max(self.nbits, len(seq))

    @classmethod
    def from_int(
        cls,
        integer: int,
        *,
        nbits: Optional[int] = None,
        numbering: BitNumbering = BitNumbering.MSB,
        dtype: Optional["DataType"] = None,
    ) -> "BitString":
        # basic integer?
        return cls(
            integer=integer,
            nbits=nbits if nbits is not None else None,
            numbering=numbering,
            dtype=dtype,
        )

    @classmethod
    def from_binary(
        cls,
        binary: str | "BitString",
        *,
        nbits: Optional[int] = None,
        numbering: BitNumbering = BitNumbering.MSB,
    ) -> "BitString":
        if isinstance(binary, cls):
            return cls.from_bitstring(binary, nbits=nbits)
        b = binary[2:] if binary.startswith("0b") else binary
        width = max(nbits or 0, len(b))
        inst = cls(integer=0, nbits=width, numbering=numbering)
        inst.integer = int(b[::-1] if numbering is BitNumbering.LSB else b, 2)
        return inst
    @classmethod
    def from_array(
        cls,
        array: list | "BitStringView",
        *,
        nbits: Optional[int] = 0,
        numbering: BitNumbering = BitNumbering.MSB,
        dtype: Optional["DataType"] = None,
    ) -> "BitStringView":
        if isinstance(array, cls):
            return cls.from_int(
                integer=array.integer,
                nbits=nbits or array.nbits,
                numbering=array.numbering,
                dtype=array.dtype,
            )
        
        width = max(nbits, len(list(array)))
        result = cls(integer=0, nbits=width, numbering=numbering, dtype=dtype)
        
        return result
    @classmethod
    def from_array(
        cls,
        array: Sequence[int] | "BitString",
        *,
        nbits: Optional[int] = None,
        numbering: BitNumbering = BitNumbering.MSB,
        dtype: Optional["DataType"] = None,
    ) -> "BitString":
        if isinstance(array, cls):
            return cls.from_bitstring(array, nbits=nbits)
        width = max(nbits or 0, len(array))
        inst = cls(integer=0, nbits=width, numbering=numbering, dtype=dtype)
        seq = list(array)
        if numbering is BitNumbering.LSB:
            seq = seq[::-1]
        inst.integer = int(''.join(str(b) for b in seq), 2)
        return inst
    @classmethod
    def from_bitstring(
        cls,
        other: "BitString",
        *,
        nbits: Optional[int] = None,
    ) -> "BitString":
        width = max(nbits or 0, other.nbits)
        return cls(
            integer=other.integer,
            nbits=width,
            numbering=other.numbering,
            dtype=other.dtype,
        )

    # ---------------------------------------------------------------- widen helper
    def widen_to_dtype(self, dtype: "DataType") -> "BitString":
        """Attach a dtype and—if needed—grow `nbits` to `dtype.data_width`."""
        if dtype.data_width < self.nbits:
            raise ValueError(
                f"value needs {self.nbits} bits but dtype supplies only "
                f"{dtype.data_width}"
            )
        self.dtype = dtype
        self.nbits = dtype.data_width
        return self   #  fluent API

    # ---------------------------------------------------------------- misc dunder
    # Arithmetic & equality remain byte-for-byte the same…
    def __eq__(self, other) -> bool:
        if isinstance(other, BitString):
            return (self.integer, self.nbits) == (other.integer, other.nbits)
        if isinstance(other, int):
            return self.integer == other
        if isinstance(other, str):
            return self.binary() == other.lstrip("0b")
        return NotImplemented
    def __hash__(self) -> int:
        return hash((self.integer, self.nbits, self.numbering))

    def __add__(self, other: "BitString") -> "BitString":
        if not isinstance(other, BitString):
            raise TypeError(f"Cannot add {type(other)} to BitString")
        return BitString.from_int(
            self.integer + other.integer,
            nbits=max(self.nbits, other.nbits),
        )
    def __len__(self):
        return self.nbits
    def __int__(self) -> int:
        return self.integer
    def __repr__(self):
        return f"BitString({self.integer}, nbits={self.nbits}, order={self.numbering.name})"


# ----------------------------------------------------------------------
#  LSB-variant (for testing parity with tequila.BitStringLSB)
# ----------------------------------------------------------------------
class BitStringLSB(BitString):
    @property
    def numbering(self) -> BitNumbering:
        return BitNumbering.LSB


import attrs



def _update_nbits(inst, attribute, value):
    # figure out how many bits we need for inst._value
    needed = ceil(log2((inst._value or 0) + 1)) if inst._value is not None else 0
    inst._nbits = max(inst._nbits or 0, needed)
    return value


def _on_set_integer(inst, attribute, value):
    inst._value = int(value)
    _update_nbits(inst, attribute, inst._value)
    return inst._value


def _on_set_binary(inst, attribute, value):
    s = value
    if s.startswith("0b"):
        s = s[2:]
    if inst.numbering is BitNumbering.LSB:
        inst._value = int(s[::-1], 2)
    else:
        inst._value = int(s, 2)
    _update_nbits(inst, attribute, inst._value)
    return value

def _on_set_array(inst, attribute, value):
    seq = list(value)
    if inst.numbering is BitNumbering.LSB:
        inst._value = int("".join(str(x) for x in seq[::-1]), 2)
    else:
        inst._value = int("".join(str(x) for x in seq), 2)
    _update_nbits(inst, attribute, inst._value)
    return seq
def validate_dtype(instance,attribute, value):
    instance.assert_valid_classical_val(instance.integer, "BitStringView.integer")

def _sync_nbits(instance, attribute, value):
    # grow nbits whenever integer changes
    instance.nbits = max(instance.nbits, value.bit_length())
    return value

def _enforce_dtype(instance, attribute, value):
    # enforce dtype width and grow nbits
    if value is not None:
        value.assert_valid_classical_val(instance.integer, "BitStringView.integer")
        instance.nbits = max(instance.nbits, value.data_width)
    return value
def _sync_derived(instance, attribute, value):
    """
    Class‐level on_setattr hook: whenever integer, nbits, or numbering changes,
    recompute the derived `binary` and `array` fields.
    """
    if attribute.name in ("integer", "nbits", "numbering"):
        bits = format(instance.integer, f"0{max(instance.nbits, 1)}b")
        if instance.numbering is BitNumbering.LSB:
            bits = bits[::-1]
        # Bypass hooks when updating derived fields
        object.__setattr__(instance, "binary", bits)
        object.__setattr__(instance, "array", [int(b) for b in bits])
    return value
from attrs import field, validate, validators, converters,Factory
# possibly add order = True which adds  __lt__, __le__, __gt__, and __ge__ methods that behave like __eq__  and allow instances to be ordered
@attrs.define(
    slots=True,
    kw_only=True,
    eq=False, 
    order=False, 
    hash=False,
    on_setattr=[attrs.setters.convert, attrs.setters.validate],
)
class BitStringView:

    _value: int = field(init=False, default=0, repr=False)
    # _nbits: int = field(default=0, repr=False)
    # public “fields” that drive everything via on_setattr
    
    
    numbering: BitNumbering = attrs.field(
        default=BitNumbering.MSB,
        validator=attrs.validators.instance_of(BitNumbering),
    )
    integer: int = field(default=0, 
                         converter=int, 
                         validator=attrs.validators.instance_of(int),
                         on_setattr=_sync_nbits,
    )
    binary: str = field(init=False, default="", on_setattr=_on_set_binary)
    array: List[int] = field(
        init=False, factory=list, on_setattr=_on_set_array
    )
    dtype: "DataType | None"= field(
        default=None,
        validator=validate_dtype,
        on_setattr=_enforce_dtype,
        )
    # nbits: int = field(
    #     default=Factory(lambda self: max(1, self.integer.bit_length()), takes_self=True),
    #     validator=validators.instance_of(int),
    # )
    nbits: int = attrs.field(
        default=attrs.Factory(lambda self: max(1, self.integer.bit_length()), takes_self=True),
        validator=attrs.validators.instance_of(int),
    )
   
    def __attrs_post_init__(self):
        # Compute derived fields once at init
        _sync_derived(self, attrs.fields(BitStringView).integer, self.integer)

    @dtype.validator
    def _validate_dtype(self, attribute, value):
        if value is not None:
            value.assert_valid_classical_val(self.integer, "BitStringView.integer")
            self.nbits = max(self.nbits, value.data_width)
    def binary(self) -> str:
        """
        Get the bitstring, zero-padded to nbits, respecting endianness.
        """
        bits = format(self.integer, f"0{max(self.nbits, 1)}b")
        return bits[::-1] if self.numbering is BitNumbering.LSB else bits
    def bits(self) -> list[int]:
        """
        List of bits (ints) from MSB->LSB or reversed.
        """
        return [int(b) for b in self.binary()]

    def array(self) -> list[int]:
        """Alias for bits()."""
        return self.bits()
    
    @classmethod
    def from_int(
        cls,
        integer: int,
        *,
        nbits: Optional[int] = None,
        numbering: BitNumbering = BitNumbering.MSB,
        dtype: Optional["DataType"] = None,
    ) -> "BitStringView":
        return cls(
            integer=integer,
            nbits=nbits if nbits is not None else None,
            numbering=numbering,
            dtype=dtype,
        )

    @classmethod
    def from_binary(
        cls,
        binary: str | "BitStringView",
        *,
        nbits: Optional[int] = None,
        numbering: BitNumbering = BitNumbering.MSB,
    ) -> "BitStringView":
        if isinstance(binary, cls):
            return cls.from_int(
                integer=binary.integer,
                nbits=nbits,
                numbering=binary.numbering,
                dtype=binary.dtype,
            )
        b = binary[2:] if binary.startswith("0b") else binary
        width = max(nbits or 0, len(b))
        inst = cls(integer=0, nbits=width, numbering=numbering)
        inst.integer = int(b[::-1] if numbering is BitNumbering.LSB else b, 2)
        return inst

    @classmethod
    def from_array(
        cls,
        array: Sequence[int] | "BitStringView",
        *,
        nbits: Optional[int] = None,
        numbering: BitNumbering = BitNumbering.MSB,
        dtype: Optional["DataType"] = None,
    ) -> "BitStringView":
        if isinstance(array, cls):
            return cls.from_int(
                integer=array.integer,
                nbits=nbits,
                numbering=array.numbering,
                dtype=array.dtype,
            )
        width = max(nbits or 0, len(array))
        inst = cls(integer=0, nbits=width, numbering=numbering, dtype=dtype)
        seq = list(array)
        if numbering is BitNumbering.LSB:
            seq = seq[::-1]
        inst.integer = int(''.join(str(b) for b in seq), 2)
        return inst

    def widen_to_dtype(self, dtype: "DataType") -> "BitStringView":
        """Attach a dtype and grow nbits if needed."""
        if dtype.data_width < self.nbits:
            raise ValueError(
                f"value needs {self.nbits} bits but dtype supplies only {dtype.data_width}"
            )
        self.dtype = dtype
        return self

    def __eq__(self, other) -> bool:
        if isinstance(other, BitStringView):
            return (self.integer, self.nbits) == (other.integer, other.nbits)
        if isinstance(other, int):
            return self.integer == other
        if isinstance(other, str):
            return self.binary() == other.lstrip("0b")
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.integer, self.nbits, self.numbering))

    def __add__(self, other: "BitStringView") -> "BitStringView":
        if not isinstance(other, BitStringView):
            raise TypeError(f"Cannot add {type(other)} to BitStringView")
        return BitStringView.from_int(
            integer=self.integer + other.integer,
            nbits=max(self.nbits, other.nbits),
            numbering=self.numbering,
            dtype=self.dtype,
        )

    def __len__(self) -> int:
        return self.nbits

    def __repr__(self) -> str:
        return (
            f"BitStringView({self.integer}, "
            f"nbits={self.nbits}, order={self.numbering.name})"
        )

    def __int__(self) -> int:
        return self.integer

@define(slots=True, kw_only=True,
              on_setattr=[attrs.setters.convert, attrs.setters.validate])
class BitStringViewLSB(BitStringView):
    # override numbering to LSB and remove it from init
    numbering: BitNumbering = attrs.field(default=BitNumbering.LSB, init=False)




class CType(DataType, metaclass=abc.ABCMeta):
    """Parent for purely classical data types.
    - element_size (in bytes) = data_width / 8.
    """

    @abc.abstractmethod
    def to_bits(self, val) -> list[int]:
        """Convert a classical value (e.g. int, float) to a list of bits."""
    
    @abc.abstractmethod
    def from_bits(self, bits: list[int]):
        """ Inverse of to_bits()."""
   
    @property
    def nbytes(self):
        """total bytes"""
        self.data_width // 8
    
    def to_bitstring(
        self,
        value: int | Sequence[int] | "BitStringView" = 0,
        *,
        numbering: BitNumbering = BitNumbering.MSB,
    ) -> "BitStringView":
        """
        Return a BitStringView initialised with *value* and widened
        to *this* dtype.  Works for every classical subtype.
        """
        if isinstance(value, BitStringView):
            return value.widen_to_dtype(self)

        if isinstance(value, (int, np.integer)):
            bs = BitStringView.from_int(value, numbering=numbering)
        else:  # assume list/ndarray of bits
            bs = BitStringView.from_array(value, numbering=numbering)

        return bs.widen_to_dtype(self)


    def __str__(self):
        return f"{self.__class__.__name__}({self.data_width})"
@define
class NumericType(CType, metaclass=abc.ABCMeta):
    """
    Marker base for *numeric* classical types (integers, floats, fixed‐point, bits).
    Non-numeric classical types (String, Struct) remain direct subclasses of CType.
    """
    def to_bitstring(self, value=0, *, numbering=BitNumbering.MSB):
        return CType.to_bitstring(self, value, numbering=numbering)