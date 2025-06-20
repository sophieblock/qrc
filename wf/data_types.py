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

from ...util.log import get_logger
logger = get_logger(__name__)


from typing import TypeVar


# ──────────────────────────────────────────────────────────────────────────────
#  Light-weight VALUE helper – orthogonal to the dtype hierarchy
# ──────────────────────────────────────────────────────────────────────────────
class BitNumbering(Enum):
    MSB = 0          # bit-0 is most-significant (big-endian bit order)
    LSB = 1         # bit-0 is least-significant (little-endian bit order)


from attrs import define,field, validate, validators, converters,Factory,setters



def validate_dtype(instance,attribute, value):
    instance.assert_valid_classical_val(instance.integer, "BitStringView.integer")
def _post_setattr(instance, attribute, value):
    """
    on_setattr hook: keep nbits in sync and validate dtype whenever integer or dtype changes.
    """
    name = attribute.name
    if name == "integer":
        # grow nbits based on new integer
        new_bits = value.bit_length() or 1
        if instance.nbits is None:
            object.__setattr__(instance, "nbits", new_bits)
        else:
            object.__setattr__(instance, "nbits", max(instance.nbits, new_bits))
    elif name == "dtype":
        # validate and grow nbits based on dtype
        if value is not None:
            value.assert_valid_classical_val(instance.integer, "BitStringView.integer")
            object.__setattr__(instance, "nbits", max(instance.nbits or 0, value.data_width))
    return value

# possibly add order = True which adds  __lt__, __le__, __gt__, and __ge__ methods that behave like __eq__  and allow instances to be ordered
@define(
    slots=True,
    kw_only=True,
    on_setattr=[setters.convert, setters.validate, _post_setattr],
)
class BitStringView:
    integer: int = field(
        default=0,
        converter=int,
        validator=validators.instance_of(int),
    )

    nbits: Optional[int] = field(
        default=None,
        validator=validators.optional(validators.instance_of(int)),
    )

    numbering: BitNumbering = field(
        default=BitNumbering.MSB,
        validator=validators.instance_of(BitNumbering),
    )

    dtype: Optional["DataType"] = field(
        default=None,
    )

    def __attrs_post_init__(self):
        # initialize nbits if not provided
        if self.nbits is None:
            object.__setattr__(self, "nbits", self.integer.bit_length() or 1)
        # enforce dtype if given
        if self.dtype is not None:
            self.dtype.assert_valid_classical_val(self.integer, "BitStringView.integer")
            object.__setattr__(self, "nbits", max(self.nbits, self.dtype.data_width))

    def binary(self) -> str:
        bits = format(self.integer, f"0{self.nbits}b")
        return bits[::-1] if self.numbering is BitNumbering.LSB else bits

    def bits(self) -> list[int]:
        return [int(b) for b in self.binary()]

    def array(self) -> list[int]:
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
        if isinstance(integer, cls):
            # preserve existing bits and dtype if not overridden
            return cls.from_bitstring(
                other=integer,
                nbits=nbits or integer.nbits,
                numbering=numbering,
                dtype=dtype or integer.dtype,
            )
        return cls(
            integer=integer,
            nbits=nbits,
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
                nbits=nbits or binary.nbits,
                numbering=binary.numbering,
                dtype=binary.dtype,
            )
        b = binary[2:] if binary.startswith("0b") else binary
        width = max(nbits or 0, len(b))
        inst = cls(integer=0, nbits=width, numbering=numbering)
        inst.integer = int(
            b[::-1] if numbering is BitNumbering.LSB else b,
            2,
        )
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
                nbits=nbits or array.nbits,
                numbering=array.numbering,
                dtype=array.dtype,
            )
        seq = list(array)
        width = max(nbits or 0, len(seq))
        inst = cls(integer=0, nbits=width, numbering=numbering, dtype=dtype)
        if numbering is BitNumbering.LSB:
            seq = seq[::-1]
        inst.integer = int("".join(str(b) for b in seq), 2)
        return inst

    @classmethod
    def from_bitstring(
        cls,
        other: "BitStringView",
        *,
        nbits: Optional[int] = None,
        numbering: BitNumbering | None = None,
        dtype: Optional["DataType"] = None,
    ) -> "BitStringView":
        width = max(nbits or 0, other.nbits)
        return cls(
            integer=other.integer,
            nbits=width,
            numbering=numbering if numbering is not None else other.numbering,
            dtype=dtype if dtype is not None else other.dtype,
        )

    @classmethod
    def msb(
        cls,
        integer: int | "BitStringView",
        *,
        nbits: Optional[int] = None,
        dtype: Optional["DataType"] = None,
    ) -> "BitStringView":
        return cls.from_int(
            integer=integer,
            nbits=nbits,
            numbering=BitNumbering.MSB,
            dtype=dtype,
        )

    @classmethod
    def lsb(
        cls,
        integer: int | "BitStringView",
        *,
        nbits: Optional[int] = None,
        dtype: Optional["DataType"] = None,
    ) -> "BitStringView":
        return cls.from_int(
            integer=integer,
            nbits=nbits,
            numbering=BitNumbering.LSB,
            dtype=dtype,
        )
    def with_numbering(self, numbering: BitNumbering) -> "BitStringView":
        """
        Return a new BitStringView with the same integer, nbits and dtype
        but a different bit-ordering.
        """
        return type(self).from_bitstring(
            other=self,
            nbits=self.nbits,
            numbering=numbering,
            dtype=self.dtype,
        )


    def widen_to_dtype(self, dtype: "DataType") -> "BitStringView":
        dtype.assert_valid_classical_val(self.integer, "BitStringView.integer")
        object.__setattr__(self, "nbits", max(self.nbits, dtype.data_width))
        return self

    def __eq__(self, other) -> bool:
        if isinstance(other, BitStringView):
            return (
                self.integer,
                self.nbits,
                self.numbering,
            ) == (
                other.integer,
                other.nbits,
                other.numbering,
            )
        if isinstance(other, int):
            return self.integer == other
        if isinstance(other, str):
            return self.binary() == other.lstrip("0b")
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.integer, self.nbits, self.numbering))

    def __add__(
        self,
        other: "BitStringView",
    ) -> "BitStringView":
        if not isinstance(other, BitStringView):
            raise TypeError(f"Cannot add {type(other)} to BitStringView")
        return BitStringView.from_int(
            integer=self.integer + other.integer,
            nbits=max(self.nbits, other.nbits),
            numbering=self.numbering,
        )

    def __len__(self) -> int:
        return self.nbits

    def __int__(self) -> int:
        return self.integer

    def __repr__(self) -> str:
        return (
            f"BitStringView({self.integer}, "
            f"nbits={self.nbits}, order={self.numbering.name})"
        )


SymbolicFloat = Union[float, sympy.Expr]
"""A floating point value or a sympy expression."""

SymbolicInt = Union[int, sympy.Expr]
"""An integer value or a sympy expression."""

SymbolicComplex = Union[complex, sympy.Expr]
"""A complex value or a sympy expression."""


T = TypeVar('T')
from typing_extensions import TypeIs

SymbolicT = TypeVar('SymbolicT', SymbolicInt, SymbolicFloat, SymbolicComplex)



@overload
def is_symbolic(
    arg: Union[T, sympy.Expr, SymInt,SymbolicT], /
) -> TypeIs[Union[sympy.Expr, SymInt,SymbolicT]]: ...


@overload
def is_symbolic(*args) -> bool: ...


def is_symbolic(*args) -> Union[TypeIs[Union[sympy.Expr, SymInt,SymbolicT]], bool]:
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
        """
        Combine individual bits to form classical values.

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
    
    def __format__(self, spec: str) -> str:        

        return format(str(self), spec)    


# -------------------------------- Helper methods --------------------------------

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


# ----------------------------------------------------------------
# Classical Data Types
# These types are intended to represent classical values and registers.
# They inherit from DataType and implement all required methods.

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
        numbering: BitNumbering = BitNumbering.MSB) -> "BitStringView":
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
    
# Classical Bit
@frozen
class CBit(NumericType):
    """ 
    Represents a single classical bit (0 or 1).
    """
    # bit_width: int = 1  # A classical bit is one unit.
    @property
    def data_width(self):
        return 1
    
    @property
    def bit_width(self):
        return 1
    
    @property
    def nbytes(self):
        # TODO: I actually dont know if this is correct but it needs to be defined regardless 
        return self.data_width
    
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
    def __str__(self):
        return 'CBit()'
 

@frozen
class CInt(NumericType):
    """
    Classical signed integer (two's complement).
    """
    bit_width: SymbolicInt = field(converter=_to_symbolic_int)

    @property
    def data_width(self):
        return self.bit_width
    
    @property
    def nbytes(self) -> int:       
        return self.data_width // 8
    
    def get_classical_domain(self) -> Iterable[int]:
        half = 1 << (self.bit_width - 1)
        return range(-half, half)

    def to_bits(self, value: int) -> List[int]:
        bits = []
        self.assert_valid_classical_val(value)
        for i in reversed(range(self.bit_width)):
            bits.append((value >> i) & 1)
        return bits

    def from_bits(self, bits: Sequence[int]) -> int:
        sign_bit = bits[0]
        bit_string = "".join(str(b) for b in bits)
        if sign_bit == 0:
            # non-negative
            return int(bit_string, 2)
        else:
            # negative, two's complement decode
            # invert bits => add 1 => negative
            inverted = ''.join('1' if b == '0' else '0' for b in bit_string)
            return - (int(inverted, 2) + 1)
    

    def assert_valid_classical_val(self, val: Any, debug_str: str = 'val'):
        if not isinstance(val, (int, np.integer)):
            raise ValueError(f"{debug_str} must be an integer.")
        lower = -(1 << (self.bit_width - 1))
        upper = 1 << (self.bit_width - 1)
        if not (lower <= val < upper):
            raise ValueError(f"{debug_str} must be in [{lower}, {upper}).")
        
    def __str__(self):
        return f"CInt({self.bit_width})"


@frozen
class CUInt(NumericType):
    """
    Classical unsigned integer of a given bit width (bitsize)
    """
    bit_width: SymbolicInt = field(default=32)

    @property
    def data_width(self):
        return self.bit_width
    @property
    def nbytes(self) -> int:       
        return self.data_width // 8
    @bit_width.validator
    def _check_bit_width(self, attribute, value):
        """
        Validator for _shape: every element must be an int or a dynamic placeholder.
        """
        
        if not (isinstance(value, int) or isinstance(value,SymbolicInt) or value == Dyn):
            raise ValueError(f"Invalid bit_width for CUInt: {value}. Must be an int, SymbolicInt, or Dyn.")
    def get_classical_domain(self) -> Iterable[int]:
        return range(2 ** self.bit_width)

    def to_bits(self, value: int) -> List[int]:
        if not (0 <= value < (1 << self.bit_width)):
            raise ValueError("CUInt value out of range")
        # return [(value >> i) & 1 for i in range(self.bit_width)]
        width = int(self.bit_width)
        bitstr = format(int(value), f'0{width}b')
        return [int(b) for b in bitstr]

    def from_bits(self, bits: Sequence[int]) -> int:
        if len(bits) != self.bit_width:
            raise ValueError(f"CUInt expects {self.bit_width} bits")
        # val = 0
        # for i, b in enumerate(bits):
        #     val |= (b & 1) << i
        # return val
        return int("".join(str(b) for b in bits), 2)

    def to_bits_array(self, x_array: NDArray[np.integer]) -> NDArray[np.uint8]:
        """Returns the big-endian bitstrings specified by the given integers.

        Args:
            x_array: An integer or array of unsigned integers.
        """
        if is_symbolic(self.bit_width):
            raise ValueError(f"Cannot compute bits for symbolic {self.bit_width=}")

        if self.bit_width > 64:
            # use the default vectorized `to_bits`
            return super().to_bits_array(x_array)

        w = int(self.bit_width)
        x = np.atleast_1d(x_array)
        if not np.issubdtype(x.dtype, np.uint):
            assert np.all(x >= 0)
            assert np.iinfo(x.dtype).bits <= 64
            x = x.astype(np.uint64)
        assert w <= np.iinfo(x.dtype).bits
        mask = 2 ** np.arange(w - 1, 0 - 1, -1, dtype=x.dtype).reshape((w, 1))
        return (x & mask).astype(bool).astype(np.uint8).T
    def from_bits_array(self, bits_array: NDArray[np.uint8]) -> NDArray[np.integer]:
        """Returns the integer specified by the given big-endian bitstrings.

        Args:
            bits_array: A bitstring or array of bitstrings, each of which has the 1s bit (LSB) at the end.
        Returns:
            An array of integers; one for each bitstring.
        """
        bitstrings = np.atleast_2d(bits_array)
        if bitstrings.shape[1] != self.bit_width:
            raise ValueError(f"Input bitsize {bitstrings.shape[1]} does not match {self.bit_width=}")

        if self.bit_width > 64:
            # use the default vectorized `from_bits`
            return super().from_bits_array(bits_array)

        basis = 2 ** np.arange(self.bit_width - 1, 0 - 1, -1, dtype=np.uint64)
        return np.sum(basis * bitstrings, axis=1, dtype=np.uint64)

    def assert_valid_classical_val(self, val: int, debug_str: str = 'val'):
        if not isinstance(val, (int, np.integer)):
            raise ValueError(f"{debug_str} should be an integer, not {val!r}")
        if val < 0:
            raise ValueError(f"Negative classical value encountered in {debug_str}")
        if val >= 2**self.data_width:
            raise ValueError(f"Too-large classical value encountered in {debug_str}")

    def __str__(self):
        return f"CUInt({self.bit_width})"

@frozen
class CFxp(NumericType):
    """

    The classical counterpart CFxp(total_bits, num_frac, signed=True) 
    has the same bit-format but holds a classical value. Both share the 
    interpretation: one value is encoded in a fixed number of bits where 
    a certain subset are fractional. Qualtran’s documentation states, 
    “A real number can be approximately represented in fixed point 
    using num_int bits for the integer part and num_frac bits for the 
    fractional part… If the real number is signed we store negative values 
    in two’s complement form.” 


    Representation: We can define CFxp(total, frac) such that:
	•	total = total bits (including sign if signed).
	•	frac = number of fractional bits (to the right of the binary point).
	•	num_int = total - frac (number of integer bits, including sign bit if signed) ￼.
	•	Example: CFxp(6,4) has 6 bits total, 4 fractional, hence 2 integer bits. 

    If signed, one of those integer bits is the sign, so it can represent values 
    from -2^(1) to 2^(1)-2^(-4) in steps of 2^-4.
    """
    bit_width: int
    num_frac: int
    signed: bool = False
    def __attrs_post_init__(self):
        if not is_symbolic(self.data_width) and self.data_width == 1 and self.signed:
            raise ValueError("data_width must be > 1.")
        if not is_symbolic(self.bit_width) and not is_symbolic(self.num_frac):
            if self.signed and self.bit_width == self.num_frac:
                raise ValueError("num_frac must be less than bitsize if the CFxp is signed.")
            if self.bit_width < self.num_frac:
                raise ValueError("bit_width must be >= num_frac.")
        # assert self.bit_width >= 1 and 0 <= self.num_frac <= self.bit_width

    @property
    def data_width(self) -> int:
        return self.bit_width
    @property
    def nbytes(self) -> int:       
        return self.data_width // 8

    @property
    def num_int(self) -> int:
        return self.bit_width - self.num_frac
    def to_fixed_width_int(self, x: float, require_exact: bool = False) -> int:
        scale = 1 << self.num_frac
        scaled_val = int(round(x * scale))
        # handle sign and range if self.signed
        if self.signed:
            max_val = (1 << (self.bit_width - 1)) - 1
            min_val = -(1 << (self.bit_width - 1))
            if scaled_val > max_val or scaled_val < min_val:
                scaled_val &= (1 << self.bit_width) - 1
        else:
            scaled_val %= (1 << self.bit_width)
        return scaled_val
    def to_bits(self, x: int) -> List[int]:
        if self.signed:
            return CInt(self.bit_width).to_bits(x)
        else:
            return CUInt(self.bit_width).to_bits(x)
    def from_bits(self, bits: Sequence[int]) -> int:
        if self.signed:
            return CInt(self.bit_width).from_bits(bits)
        else:
            return CUInt(self.bit_width).from_bits(bits)
    def float_from_fixed_width_int(self, x: int) -> float:
        return x / (2 ** self.num_frac)
    def get_classical_domain(self) -> Iterable[int]:
        if self.signed:
            if self.is_symbolic():
                raise TypeError("Domain for symbolic CFxp is not enumerable.")
            half = 1 << (int(self.bit_width) - 1)
            return range(-half, half)
        else:
            if self.is_symbolic():
                raise TypeError("Domain for symbolic CFxp is not enumerable.")
            return range(0, 1 << int(self.bit_width))

    
    def assert_valid_classical_val(self, val: Any, debug_str: str = 'val'):
        if not isinstance(val, (int, float)):
            raise ValueError(f"{debug_str} must be int or float for {self}")
        # No strict bounding unless you want to saturate or wrap. Basic check could be done here if desired.
    def fxp_dtype_template(self) -> Fxp:
        if is_symbolic(self.data_width) or is_symbolic(self.num_frac):
            raise ValueError(
                f"Cannot construct Fxp template for symbolic bitsizes: {self.data_width=}, {self.num_frac=}"
            )
        return Fxp(
            None,
            n_word=self.data_width,
            n_frac=self.num_frac,
            signed=self.signed,
            op_sizing='same',
            const_op_sizing='same',
            shifting='trunc',
            overflow='wrap',
        )
    def __str__(self):
        if self.signed:
            return f'CFxp({self.bit_width}, {self.num_frac}, True)'
        else:
            return f'CFxp({self.bit_width}, {self.num_frac})'
@frozen
class CFloat(NumericType):
    """
    CFloat represents a classical floating-point number (IEEE754).
    """
    bit_width: int = 32

    def __attrs_post_init__(self):
        if self.bit_width not in (8, 16, 32, 64):
            raise ValueError(f"Unsupported float size: {self.bit_width}")

    @property
    def data_width(self) -> int:
        return self.bit_width
    @property
    def nbytes(self) -> int:       
        return self.data_width // 8
    def to_bits(self, value: float) -> List[int]:
        if self.bit_width == 128:
            raise ValueError("128-bit float conversion not implemented")
        packed = struct.pack(self._fmt, value)
        bit_list = []
        for byte in packed:
            for i in range(8):
                bit_list.append((byte >> i) & 1)
        return bit_list

    def from_bits(self, bits: List[int]) -> float:
        if len(bits) != self.bit_width:
            raise ValueError(f"CFloat expects {self.bit_width} bits")
        bytes_arr = bytearray()
        for i in range(0, len(bits), 8):
            byte = 0
            for j in range(8):
                byte |= (bits[i+j] & 1) << j
            bytes_arr.append(byte)
        if self.bit_width == 128:
            raise NotImplementedError("128-bit float conversion not implemented")
        return struct.unpack(self._fmt, bytes(bytes_arr))[0]

    @property
    def _fmt(self) -> str:
        if self.bit_width == 16:
            return 'e'  # half precision
        elif self.bit_width == 32:
            return 'f'
        elif self.bit_width == 64:
            return 'd'
        else:
            return 'g'  # fallback for 128, but not fully supported

    def get_classical_domain(self) -> Iterable[Any]:
        # For real hardware, domain is continuous, so enumerating is not feasible.
        return []

    def assert_valid_classical_val(self, val: Any, debug_str: str = 'val'):
        if not isinstance(val, float):
            raise ValueError(f"{debug_str} must be float for {self}")
    def __repr__(self):
        return f'CFloat({self.bit_width})'

@frozen
class TensorType(NumericType):
    """
    Represents a rank-N generalized tensor with specified dimensions and element data type ( A base class for N-dimensional data types (matrices, tensors, etc.).)
    Holds:
      - shape: A tuple of dimension sizes (e.g., (rows, cols))
      - element_type: The Python or NumPy type (float, int, etc.)
      - val: Optionally store an actual NumPy array (or any underlying data).
    """
    shape: Tuple[int, ...] = field(validator = validators.instance_of(tuple))
    element_type: DataType = field(
        default=float,
        converter=_element_type_converter
    )
    val: Optional[NDArray] = None
    @property
    def data_width(self) -> SymbolicInt:
        """
        Bit-width per element of this tensor.
        Satisfies the abstract DataType.data_width contract.
        """
        if isinstance(self.element_type, DataType):
            return self.element_type.data_width
        return _bits_for_dtype(self.element_type)
    def nelement(self):
        return prod(self.shape) 
    def element_size(self) -> SymbolicInt:
        """
        Bytes per element = data_width (bits) // 8.
        """
        return self.data_width // 8
    @property
    def nbytes(self) -> SymbolicInt:
        """
        Total bytes in the tensor = num_elements × bytes per element.
        """
        return self.element_size() * self.nelement()
    @property
    def rank(self) -> int:
        return len(self.shape)
    @property
    def total_bits(self) -> SymbolicInt:
        """
        Total bits in the tensor = num_elements × bits per element.
        """
        return self.data_width * self.nelement()
    def get_classical_domain(self) -> Iterable[Any]:
        """Yields all possible values representable by this tensor type."""
        if self.dtype is not None:
            return itertools.product(self.dtype.get_classical_domain(), repeat=np.prod(self.shape))
        else:
            raise NotImplementedError("Domain enumeration not supported for arbitrary tensor types.")

    @property
    def bytes_per_element(self) -> int:
        """bits_per_element // 8 if you want a direct integer # of bytes."""
        return self.bit_width // 8
    @property
    def num_units(self) -> int:
        """Total number of elements = product of all dims."""
        return np.prod(self.shape) if self.shape else 1
    @property
    def memory_in_bytes(self) -> int:
        """Total memory usage = num_units * bytes_per_element."""
        return int(self.num_units * self.element_size())
   
    
    def multiply(self, other: "TensorType") -> "TensorType":
        """
        Example of a naive "broadcast multiply" shape logic (like an Einstein summation).
        """
        new_shape = np.broadcast_shapes(self.shape, other.shape)
        return TensorType(shape=new_shape, element_type=self.element_type)
    def to_bits(self, x) -> List[int]:
        """
        Convert a single piece of data to bits.
        For an N-dimensional type, x might be an NDArray or sub-structure.
        """
        # Example: flatten x, convert each element to bit (very naive)
        # You can refine for HPC usage
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        bits = []
        for element in x.flatten():
            # Turn each element into an 8-bit representation:
            # extremely naive approach
            val_as_int = int(element)  # This is a simplification
            bits.extend([int(b) for b in np.binary_repr(val_as_int, width=self.data_width)])
        return bits
    def from_bits(self, bits: Sequence[int]) -> np.ndarray:
        if any(d is None for d in self.shape):
            raise RuntimeError("Cannot reconstruct TensorType with dynamic shape without additional info")
        elem_width = self.element_type.data_width
        total_elems = 1
        for d in self.shape:
            total_elems *= d
        expected_bits = total_elems * elem_width
        if len(bits) != expected_bits:
            raise ValueError(f"Bit length {len(bits)} does not match tensor size {expected_bits}")

        flat_elems = [
            self.element_type.from_bits(bits[i*elem_width:(i+1)*elem_width])
            for i in range(total_elems)
        ]

        def build_nested(flat_list, shape_dims):
            if not shape_dims:
                return flat_list.pop(0)
            dim = shape_dims[0]
            return [build_nested(flat_list, shape_dims[1:]) for _ in range(dim)]
        shape_copy = self.shape
        return build_nested(flat_elems, shape_copy)
   
    def to_units(self, x) -> List[int]:
        """
        'Units' might mean each element of the ND array, as an int.
        Or a more advanced representation. We'll do a naive approach:
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        return x.flatten().tolist()

    def from_units(self, bits: Sequence[int]) -> np.ndarray:
        """
        Reconstruct ND array from a flat list of elements.
        """
        # This is extremely naive, purely demonstrative
        arr_size = self.num_units
        element_bit_size = self.data_width

        if len(bits) != arr_size * element_bit_size:
            raise ValueError(
                f"Expected {arr_size * element_bit_size} bits; got {len(bits)}"
            )

        arr = np.zeros(arr_size, dtype=np.int32)  # default int, just for example
        for i in range(arr_size):
            chunk = bits[i * element_bit_size : (i + 1) * element_bit_size]
            as_str = "".join(str(b) for b in chunk)
            as_int = int(as_str, 2)
            arr[i] = as_int
        return arr.reshape(self.shape)
    def assert_valid_classical_val(self, val: Any, debug_str: str = 'val'):
        """
        Validates if `val` is a valid classical value for this data type.

        Args:
            val: The value to validate.
            debug_str: Debugging string for error messages.
        """
        if not isinstance(val, np.ndarray):
            raise TypeError(f"{debug_str} must be a NumPy array, got {type(val)} instead.")

        if val.shape != self.shape:
            raise ValueError(f"{debug_str} must have shape {self.shape}, got {val.shape}.")

        if val.dtype != self.element_type:
            raise TypeError(
                f"{debug_str} must have dtype {self.element_type}, got {val.dtype}."
            )
    def __getitem__(self, idx: int) -> Any:
        """
        Dual indexing logic:
        - If val is not None, we return val[idx]
        - Otherwise, we do shape-based indexing => shape[idx]
        """
        if self.val is not None:
            # If it's an array, do array indexing
            if isinstance(self.val, np.ndarray):
                return self.val[idx]
            # If it's e.g. a tuple or list => textual data
            return self.val[idx]
        # shape-based indexing for purely numeric scenario (val=None)
        return self.shape[idx]
    def __repr__(self):
        dtype_name = self.element_type.__name__ if isinstance(self.element_type, type) else str(self.element_type)
        return f"TensorType({self.shape}, dtype={dtype_name})"
    def __str__(self):
        # dtype_name = self.element_type.__name__ if isinstance(self.element_type, type) else str(self.element_type)
        return f"TensorType({self.shape})"
   
    def __len__(self) -> int:
        """
        If val is not None => len(val) (like textual data).
        Else => len(shape) for numeric scenario (test_ndimdataype_len_and_getitem).
        """
        if self.val is not None:
            # If it's an array, return shape[0], or if it's a list/tuple, return len(val)
            if isinstance(self.val, (list, tuple)):
                return len(self.val)
            if isinstance(self.val, np.ndarray):
                return self.val.shape[0]  # or e.g. self.shape[0]
        return len(self.shape)
    def __hash__(self) -> int:
        return hash((self.__class__, self.shape, self.element_type))
@frozen
class MatrixType(TensorType):
    """
    Legacy type. We will be moving away from this, but for now, the helper will unify it with TensorType to
    pass tests ritten with just MatrixType: unification_tools.is_consistant_data_type(dtypeA, dtypeB).
    """
    def __init__(
        self,
        rows_or_shape: Union[int, Tuple[int, int]] = SymInt,
        cols: int = SymInt,
        element_type: Any = float,
    ):
        # If rows_or_shape is already a 2D tuple, use that as the shape directly
        if isinstance(rows_or_shape, tuple):
            if len(rows_or_shape) != 2:
                raise ValueError(
                    f"MatrixType expects a 2D shape, got {rows_or_shape}"
                )
            shape = rows_or_shape
        else:
            # Otherwise, treat `rows_or_shape` as the `rows` integer
            shape = (rows_or_shape, cols)

        # Now call the parent initializer with the final shape
        super().__init__(shape=shape, element_type=element_type)


    @property
    def rows(self) -> int:
        return self.shape[0]

    @property
    def cols(self) -> int:
        return self.shape[1]

    def multiply(self, other: "MatrixType") -> "MatrixType":
        """Matrix multiplication shape logic (rows x cols)."""
        if self.cols != other.rows:
            raise ValueError(
                f"Matrix multiply dimension mismatch: {self.rows}x{self.cols} vs {other.rows}x{other.cols}"
            )
        # Return a new MatrixType with shape (self.rows, other.cols)
        return MatrixType(self.rows, other.cols, element_type=self.element_type)

    def __str__(self):
        # dtype_name = self.element_type.__name__ if isinstance(self.element_type, type) else str(self.element_type)
        # return f"Matrix(({self.rows}, {self.cols}), dtype={dtype_name})"
        return f"Matrix({self.rows}, {self.cols})"
    def __repr__(self):
        dtype_name = self.element_type.__name__ if isinstance(self.element_type, type) else str(self.element_type)
        return f"Matrix(({self.rows}, {self.cols}), dtype={dtype_name})"


@frozen
class String(CType):
    max_length: int

    def __attrs_post_init__(self):
        if self.max_length < 0:
            raise ValueError("max_length cannot be negative.")
    @property
    def data_width(self) -> int:
        return 8 * self.max_length
    @property
    def nbytes(self) -> int:       
        return self.max_length
    def to_bits(self, value: str) -> List[int]:
        if len(value) > self.max_length:
            raise ValueError("String exceeds max_length")
        s = value.ljust(self.max_length, '\x00')
        bits = []
        for ch in s:
            byte = ord(ch) & 0xFF
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
        return s.rstrip("\x00")

    def get_classical_domain(self) -> Iterable[Any]:
        # Potentially all strings of length <= max_length, which is huge. Return empty or partial.
        return []

    def assert_valid_classical_val(self, val: Any, debug_str: str = 'val'):
        if not isinstance(val, str):
            raise ValueError(f"{debug_str} must be a string for {self}")
@frozen
class Struct(CType):
    """
    Struct is a classical composite type akin to a C/struct record.
    """
    fields: dict[str, DataType, CType]

    def __attrs_post_init__(self):
        # validate fields
        for k, v in self.fields.items():
            if not isinstance(v, DataType):
                raise TypeError(f"Field {k!r} is not a DataType")
        object.__setattr__(self, "field_order", list(self.fields.keys()))

    @property
    def data_width(self) -> int:
        # sum of each field's data_width
        return sum(ft.data_width for ft in self.fields.values())
    @property
    def nbytes(self) -> int:                      # <── NEW
        return sum(
            getattr(ft, "nbytes", ft.data_width // 8)
            for ft in self.fields.values()
        )
    @property
    def total_bits(self) -> int:                         # new
        """Recursive bit count (matches TensorType.total_bits)."""
        return sum(
            getattr(ft, "total_bits", ft.data_width)     # field may be composite
            for ft in self.fields.values()
        )
    @property
    def bit_width(self) -> int:
        # alias for compatibility with check_dtypes_consistent
        return self.data_width

    def to_bits(self, value: dict[str, Any]) -> List[int]:
        bits: List[int] = []
        for name in self.field_order:
            field_type = self.fields[name]
            field_val  = value[name]
            bits.extend(field_type.to_bits(field_val))
        return bits

    def from_bits(self, bits: List[int]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        idx = 0
        for name in self.field_order:
            field_type = self.fields[name]
            w = field_type.data_width
            chunk = bits[idx : idx + w]
            if len(chunk) < w:
                raise ValueError("Bit list too short for struct fields")
            result[name] = field_type.from_bits(chunk)
            idx += w
        return result

    def get_classical_domain(self) -> Iterable[Any]:
        # not used for consistency checks
        return []

    def assert_valid_classical_val(self, val: Any, debug_str: str = "val"):
        if not isinstance(val, dict):
            raise ValueError(f"{debug_str} must be a dict")
        for name in self.field_order:
            if name not in val:
                raise ValueError(f"Missing field '{name}' in {debug_str}")
            self.fields[name].assert_valid_classical_val(val[name], f"{debug_str}.{name}")



# ----------------------------------------------------------------
# Quantum Data Type Implementations
# The fundamental unit is a qubit
# @define(eq=False)
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
   

@frozen
class QBit(QType):
    """Quantum bit (qubit) type.
    
    """
    # num_qubits: int = 1  # A qubit is one unit.
    @property
    def data_width(self):
        return 1
    @property
    def num_qubits(self):
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
    def __str__(self):
        return 'QBit()'
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
        x = (
            0
            if self.num_qubits == 1
            else QUInt(self.num_qubits - 1).from_bits([1 - x if sign else x for x in bits[1:]])
        )
        return ~x if sign else x


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
    

@frozen
class BQUInt(QType):
    """Unsigned integer whose values are bounded within a range.

    LCU methods often make use of coherent for-loops via UnaryIteration, iterating over a range
    of values stored as a superposition over the `SELECT` register. Such (nested) coherent
    for-loops can be represented using a `Tuple[Register(dtype=BQUInt),
    ...]` where the i'th entry stores the bitsize and iteration length of i'th
    nested for-loop.

    One useful feature when processing such nested for-loops is to flatten out a composite index,
    represented by a tuple of indices (i, j, ...), one for each selection register into a single
    integer that can be used to index a flat target register. An example of such a mapping
    function is described in Eq.45 of https://arxiv.org/abs/1805.03662. A general version of this
    mapping function can be implemented using `numpy.ravel_multi_index` and `numpy.unravel_index`.

    Attributes:
        bitsize: The number of qubits used to represent the integer.
        iteration_length: The length of the iteration range.
    """
    bitsize: SymbolicInt
    iteration_length: SymbolicInt = field()
    def __attrs_post_init__(self):
        if not self.is_symbolic():
            if self.iteration_length > 2**self.bitsize:
                raise ValueError(
                    "BQUInt iteration length is too large for given bitsize. "
                    f"{self.iteration_length} vs {2**self.bitsize}"
                )
   
    @iteration_length.default
    def _default_iteration_length(self):
        return 2 ** self.bitsize
    
   
    @property
    def num_qubits(self) -> int:
        return self.bitsize
    def get_classical_domain(self) -> Iterable[Any]:
        if isinstance(self.iteration_length, int):
            return range(self.iteration_length)
        raise ValueError(f"Iteration length {self.iteration_length} is symbolic.")
    
    def assert_valid_classical_val(self, val: int, debug_str: str = 'val'):
        if not isinstance(val, (int, np.integer)):
            raise ValueError(f"{debug_str} must be an integer; got {val!r}")
        if val < 0 or val >= self.iteration_length:
            raise ValueError(f"{debug_str}={val} out of bounds for BQUInt({self.bitsize}, {self.iteration_length}).")
    
    def to_bits(self, x: int) -> List[int]:
        self.assert_valid_classical_val(x, debug_str='val')
        return QUInt(self.bitsize).to_bits(x)
    
    def from_bits(self, bits: Sequence[int]) -> int:
        return QUInt(self.bitsize).from_bits(bits)
    
    def is_symbolic(self) -> bool:
        return is_symbolic(self.bitsize, self.iteration_length)
    
    def __str__(self):
        return f"BQUInt({self.bitsize}, {self.iteration_length})"






@frozen
class QFxp(QType):
    """Fixed point type to represent real numbers in a quantum register.

    A fixed point number is represented using a total bitsize split into an integer part and a fractional part.
    If signed, negative numbers are stored in two's complement.
    """
    num_qubits: SymbolicInt
    num_frac: SymbolicInt
    signed: bool = False

    def __attrs_post_init__(self):
        if not is_symbolic(self.data_width) and self.data_width == 1 and self.signed:
            raise ValueError("data_width must be > 1.")
        if not is_symbolic(self.data_width) and not is_symbolic(self.num_frac):
            if self.signed and self.data_width == self.num_frac:
                raise ValueError("num_frac must be less than bit_width if the QFxp is signed.")
            if self.data_width < self.num_frac:
                raise ValueError("bit_width must be >= num_frac.")

    @property
    def data_width(self):
        return self.num_qubits

    @property
    def num_int(self) -> int:
        """Number of bits for the integral part."""
        return self.data_width - self.num_frac

    def is_symbolic(self) -> bool:
        return is_symbolic(self.num_qubits, self.num_frac)

    @property
    def _int_dtype(self) -> Union[QUInt, QInt]:
        """The corresponding dtype for the raw integer representation."""
        return QInt(self.num_qubits) if self.signed else QUInt(self.num_qubits)

    def get_classical_domain(self) -> Iterable[int]:
        yield from self._int_dtype.get_classical_domain()

    def to_bits(self, x) -> List[int]:
        return self._int_dtype.to_bits(x)

    def from_bits(self, bits: Sequence[int]):
        return self._int_dtype.from_bits(bits)

    def assert_valid_classical_val(self, val: int, debug_str: str = 'val'):
        self._int_dtype.assert_valid_classical_val(val, debug_str)

    def to_fixed_width_int(
        self, x: Union[float, Fxp], *, require_exact: bool = False, complement: bool = True
    ) -> int:
        bits = self._fxp_to_bits(x, require_exact=require_exact, complement=complement)
        return self._int_dtype.from_bits(bits)

    def float_from_fixed_width_int(self, x: int) -> float:
        return x / (2 ** self.num_frac)

    def fxp_dtype_template(self) -> Fxp:
        if is_symbolic(self.num_qubits) or is_symbolic(self.num_frac):
            raise ValueError(
                f"Cannot construct Fxp template for symbolic bitsizes: {self.num_qubits=}, {self.num_frac=}"
            )
        return Fxp(
            None,
            n_word=self.num_qubits,
            n_frac=self.num_frac,
            signed=self.signed,
            op_sizing='same',
            const_op_sizing='same',
            shifting='trunc',
            overflow='wrap',
        )

    def _get_classical_domain_fxp(self) -> Iterable[Fxp]:
        for x in self._int_dtype.get_classical_domain():
            yield Fxp(x / 2**self.num_frac, like=self.fxp_dtype_template())

    def _fxp_to_bits(
        self, x: Union[float, Fxp], require_exact: bool = True, complement: bool = True
    ) -> List[int]:
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
        bits_bin = "".join(str(x) for x in bits[:])
        fxp_bin = "0b" + bits_bin[: -self.num_frac] + "." + bits_bin[-self.num_frac :]
        return Fxp(fxp_bin, like=self.fxp_dtype_template())

    def _assert_valid_classical_val(self, val: Union[float, Fxp], debug_str: str = 'val'):
        fxp_val = val if isinstance(val, Fxp) else Fxp(val)
        if fxp_val.get_val() != fxp_val.like(self.fxp_dtype_template()).get_val():
            raise ValueError(
                f"{debug_str}={val} cannot be accurately represented using Fxp {fxp_val}"
            )

    def __str__(self):
        if self.signed:
            return f"QFxp({self.num_qubits}, {self.num_frac}, True)"
        return f"QFxp({self.num_qubits}, {self.num_frac})"




# ----------------------------------------------------------------
# Add explanation of the rest of code functionality

def _is_classical(dtype:DataType) -> bool:
    assert isinstance(dtype,DataType)
    return isinstance(dtype,CType)
def is_classical(dt):
    return _is_classical(dt)
def _is_quantum(dtype:DataType) -> bool:
    assert isinstance(dtype,DataType)
    return isinstance(dtype,QType)

def is_quantum(dt):
    return _is_quantum(dt)
def _cross_domain(a, b) -> bool:
    """True if one dtype is quantum and the other classical."""
    return (_is_quantum(a) and _is_classical(b)) or \
           (_is_classical(a) and _is_quantum(b))

QAnyInt =(QInt, QUInt, BQUInt,)
QAnyUInt =(QUInt, BQUInt,)

class Q_PromoLevel(Enum):
    """ 
    Quantum-domain type-checking levels. Controls how strictly two quantum dtypes must match to be considered consistent

     - STRICT: Require exact dtype match, (same class, same num qubits). Only single bit conversions are allowed.
     - ANY: Disallow numeric type conversions (int vs. fxp), but still allow single-bit or QAny
            conversions if dimensions match.
     - LOOSE: Allow a broad range of QType conversions between, i.e., QInt <-> QFxp, 

    TODO: Rename `ANY` to be less ambiguous. 
    """
    STRICT  = 0   # exact match only
    ANY = 1   # within-domain widening only
    LOOSE = 2   # PROMOTE or bit-cast within domain


class C_PromoLevel(Enum):
    """ 
    Classical-domain type-checking levels.
    """
    STRICT  = 0   # exact dtype match only
    PROMOTE = 1   #  widening & int→float promotion (Torch-style)
    CAST = 2   # bit-cast within classical domain (same total width)



class DTypeCheckingSeverity(Enum):
    STRICT = 0   # old “exact match”
    ANY    = 1   # old “no numeric promotion”  (middle)
    LOOSE  = 2   # old “broad numeric promotion”


def _check_quint_qfxp_consistent(q_u: QUInt, qfxp: QFxp) -> bool:
    """
    Example logic: a QUInt is consistent with a QFxp if:
      - QFxp is unsigned,
      - same num_qubits,
      - either wholly integer (num_frac=0) or wholly fractional (num_int=0).
    Adjust this logic to suit your needs. 
    """
    if qfxp.signed:
        return False
    if q_u.num_qubits != qfxp.num_qubits:
        return False
    # For QFxp, num_int = num_qubits - num_frac
    # => If num_frac = 0, purely integer; if num_int=0, purely fractional
    if (qfxp.num_frac == 0) or (qfxp.num_int == 0):
        return True
    return False


def _check_cuint_cfixed_consistent(cu: CUInt, cfix: CFxp) -> bool:
    """
    Example logic: a CUInt is consistent with a CFxp if:
      - cfix is not signed (or allow it?), 
      - same bit_width,
      - cfix.num_frac == 0 or cfix.num_int == 0, etc.
    Adjust as needed.
    """
    if cfix.signed:
        return False
    if _is_symbolic_dim(cu.bit_width) or _is_symbolic_dim(cfix.bit_width):
        return True
    if cu.bit_width != cfix.bit_width:
        return False
    # If cfix.num_frac == 0 => pure integer
    return (cfix.num_frac == 0 or cfix.num_int == 0)


def _check_tensor_consistency(a: TensorType, b: TensorType, severity: DTypeCheckingSeverity) -> bool:
    """
    Decide if two TensorTypes are consistent under the given severity.
    - Must have the same rank.
    - Each dimension must either be equal or one side is Dyn (wildcard).
    - Element types are then recursively checked.
    """
    if a.rank != b.rank:
        return False
    
    for dim_a, dim_b in zip(a.shape, b.shape):
        if dim_a == dim_b:
            continue
        if dim_a == Dyn or dim_b == Dyn:
            continue
        return False
    
    return check_dtypes_consistent(a.element_type, b.element_type, severity)



def _map_to_domain(level: DTypeCheckingSeverity) -> tuple[C_PromoLevel, Q_PromoLevel]:
    """Return (classical_level, quantum_level) for a given global level."""
    if level is DTypeCheckingSeverity.STRICT:
        return (C_PromoLevel.STRICT, Q_PromoLevel.STRICT)
    if level is DTypeCheckingSeverity.ANY:
        return (C_PromoLevel.PROMOTE, Q_PromoLevel.ANY)
    if level is DTypeCheckingSeverity.LOOSE:
        return (C_PromoLevel.CAST,    Q_PromoLevel.LOOSE)
    
    raise ValueError(f"Unknown severity {level}")

def _is_symbolic_dim(w) -> bool:
    return w is Dyn or isinstance(w, sympy.Basic) or not isinstance(w, int)


def _width_equal_or_symbolic(w1, w2) -> bool:
    """Return True if widths *exactly* match – or one is Dyn/symbolic."""
    
    # symbolic = lambda w: w is Dyn or _is_symbolic_dim(w)
    return (w1 == w2) or _is_symbolic_dim(w1) or _is_symbolic_dim(w2)



def check_dtypes_consistent(
    dtype_a: DataType,
    dtype_b: DataType,
    severity: DTypeCheckingSeverity = DTypeCheckingSeverity.LOOSE,
    *,
    classical_level: C_PromoLevel | None = None,
    quantum_level:   Q_PromoLevel | None = None,
) -> bool:
    """
    Checks whether dtype_a and dtype_b are consistent under a chosen severity level.
    - Translate `severity` to (C_PromoLevel, Q_PromoLevel)  
    - Handle cross-domain bridge cases early  
    Args:
        dtype_a: The dtype to check against the reference.
        dtype_b: The reference dtype
        severity: Global type-check knob 
        classical_level  : explicit override (C_PromoLevel) or None
        quantum_level    : explicit override (Q_PromoLevel) or None
    Returns:
        True if consistent, False otherwise.

    TODO: We should probably split checks into two different functions for either (a) classical
    of (b) quantum, such that later we may introduce (c) hybrid

    """
    c_lvl_default, q_lvl_default = _map_to_domain(severity)
    c_lvl = classical_level if classical_level is not None else c_lvl_default
    q_lvl = quantum_level   if quantum_level   is not None else q_lvl_default

    if dtype_a == dtype_b or dtype_a is Dyn or dtype_b is Dyn:
        return True
    a_is_q, b_is_q = _is_quantum(dtype_a), _is_quantum(dtype_b)

    if a_is_q ^ b_is_q:
        # Only QBit ↔ CBit is allowed outside STRICT
        if isinstance(dtype_a, QBit) and isinstance(dtype_b, CBit) or \
           isinstance(dtype_b, QBit) and isinstance(dtype_a, CBit):
            return q_lvl is not Q_PromoLevel.STRICT
        return False  # all other Q↔C pairs forbidden

    
    # ====================== quantum domain ==================================
    if a_is_q: 
        same_dtypes = dtype_a == dtype_b
        same_n_qubits = dtype_a.num_qubits == dtype_b.num_qubits
        logger.debug(f'dtype a: {dtype_a}, num qubits: {dtype_a.num_qubits},')
        logger.debug(f'dtype b: {dtype_b}, num qubits: {dtype_b.num_qubits}')
        if same_dtypes or same_n_qubits and dtype_a.num_qubits == 1:
            return True
        same_subclass = type(dtype_a) is type(dtype_b)
        width_ok      = _width_equal_or_symbolic(dtype_a.data_width,
                                                 dtype_b.data_width)
        logger.debug(f'same_subclass: {same_subclass}, width_ok: {width_ok}')
        if q_lvl is Q_PromoLevel.STRICT:
            return False

        if q_lvl is Q_PromoLevel.ANY:
            if isinstance(dtype_a, QAny) or isinstance(dtype_b, QAny):
                return same_n_qubits
            return False
        if isinstance(dtype_a, QAny) or isinstance(dtype_b, QAny):
            # QAny -> any dtype and any dtype -> QAny
            return width_ok
        if isinstance(dtype_a,QAnyInt) and isinstance(dtype_b,QAnyInt):
            return width_ok
        elif isinstance(dtype_a, QAnyUInt) and isinstance(dtype_b, QFxp):
            # unsigned Fxp which is wholly an integer or < 1 part is a uint.
            return _check_quint_qfxp_consistent(dtype_a, dtype_b)
        elif isinstance(dtype_b, QAnyUInt) and isinstance(dtype_a, QFxp):
            # unsigned Fxp which is wholy an integer or < 1 part is a uint.
            return _check_quint_qfxp_consistent(dtype_b, dtype_a)
       
        return False

    # =============================== CLASSICAL =======================
    # ------------------------------------------------------------------
    #  Tensor / Matrix handling
    # ------------------------------------------------------------------
    if isinstance(dtype_a, MatrixType):
        dtype_a = TensorType(shape=dtype_a.shape, element_type=dtype_a.element_type)
    if isinstance(dtype_b, MatrixType):
        dtype_b = TensorType(shape=dtype_b.shape, element_type=dtype_b.element_type)
    if getattr(dtype_a, "data_width", None) == 1 and getattr(dtype_b, "data_width", None) == 1:
        return True
    if isinstance(dtype_a, TensorType) and isinstance(dtype_b, TensorType):
        return _check_tensor_consistency(dtype_a, dtype_b,severity)
    same_dtype = type(dtype_a) is type(dtype_b)
    width_ok = _width_equal_or_symbolic(dtype_a.data_width,
                                        dtype_b.data_width)

    # 3.1 STRICT  – identical object OR both widths symbolic in same class
    if c_lvl is C_PromoLevel.STRICT:
        if dtype_a == dtype_b:
            return True
        if same_dtype and _is_symbolic_dim(getattr(dtype_a, "bit_width", None)) \
                   and _is_symbolic_dim(getattr(dtype_b, "bit_width", None)):
            return True
        return False

    
    # print(f'Level: {severity} - CLevel: {c_lvl}, width_ok: {width_ok}')
    # print(f'dtype a: {dtype_a}, bit_width: {dtype_a.data_width},')
    # print(f'dtype b: {dtype_b}, bit_width: {dtype_b.data_width}')
    
    
    
    if isinstance(dtype_a, Struct) and isinstance(dtype_b, Struct):
        if set(dtype_a.fields) != set(dtype_b.fields):
            return False
        for name in dtype_a.fields:
            if not check_dtypes_consistent(
                dtype_a.fields[name],
                dtype_b.fields[name],
                severity,
            ):
                return False
        return True
    # PROMOTE – symbolic wildcard OR widening int→float
    if c_lvl is C_PromoLevel.PROMOTE:
        # symbolic width passes within same class
        if same_dtype and (_is_symbolic_dim(getattr(dtype_a, "data_width", None))
                      or _is_symbolic_dim(getattr(dtype_b, "data_width", None))):
            return True
        
        # int / uint → float widening
        int_to_float = (
            (isinstance(dtype_a, (CInt, CUInt)) and isinstance(dtype_b, CFloat))
            or (isinstance(dtype_b, (CInt, CUInt)) and isinstance(dtype_a, CFloat))
        )
        if int_to_float and width_ok:
            return True
        
        return False
    
    

    if c_lvl is C_PromoLevel.CAST:
        # a) everything PROMOTE already allows
        if check_dtypes_consistent(dtype_a, dtype_b,
                                   severity,  # keep same severity
                                   classical_level=C_PromoLevel.PROMOTE,
                                   quantum_level=q_lvl):
            return True
        # b) bit-cast signed↔unsigned
        if same_dtype and width_ok:
            return True
        if (isinstance(dtype_a, CInt) and isinstance(dtype_b, CUInt) or
            isinstance(dtype_b, CInt) and isinstance(dtype_a, CUInt)) and width_ok:
            return True
        # c) uint ↔ fixed-point (frac==0)
        if isinstance(dtype_a, CUInt) and isinstance(dtype_b, CFxp):
            return _check_cuint_cfixed_consistent(dtype_a, dtype_b)
        if isinstance(dtype_b, CUInt) and isinstance(dtype_a, CFxp):
            return _check_cuint_cfixed_consistent(dtype_b, dtype_a)
        # d) int ↔ float reinterpret when widths equal
        if width_ok and (
            (isinstance(dtype_a, CInt) and isinstance(dtype_b, CFloat)) or
            (isinstance(dtype_b, CInt) and isinstance(dtype_a, CFloat))
        ):
            return True
        return False  # CAST failed
    # everything else fails
    return False
