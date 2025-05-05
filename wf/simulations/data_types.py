from typing import Iterable, Any, Union, List, Sequence, Optional, Tuple
import itertools
import numpy as np
from numpy.typing import NDArray
from torch import SymInt
import sympy
from enum import Enum
import abc

from attrs import define, field, validators

from ...util.log import logging
logger = logging.getLogger(__name__)

"""
CHANGES & NEW DESIGN:

    1.  Introduced a new hierarchy of `DataType` classes (e.g., `MatrixType`, `TensorType`) that store
        static shapes (e.g., `(rows, cols)` for matrices) and handle resource/shape logic (bitsize, memory usage).
    2.  `NDimDataType` is an abstract base for multi-dimensional types, providing a default method for
        calculating bits, memory usage, and shape-based logic.
    3.  Specialized types like `MatrixType`, `TensorType`, `CBit`, `CAny`, and `QAny` each override or
        extend the base functionality as needed. 
    4.  By placing shape logic in data types, we keep the code modular — `Data` is responsible for hooking
        these static shapes into dynamic symbolic shape environments, while `DataType` provides the
        “blueprint” for how many units/bits are needed.

FUNCTIONALITY OVERVIEW:

`DataType` (abstract):
    - Parent interface for any classical/quantum data representation
    - Exposes `num_units`, `bitsize`, `to_bits(...)`, and other resource or domain logic

`NDimDataType`:
    - A rank-N base class that implements standard shape-based computations (e.g., total bits = 
        `num_elements * bits_per_element`).
    - `MatrixType` and `TensorType` inherit from here, providing 2D or ND shapes

`CBit`, `CAny`, `QBit`, `QAny`:
    - Specialized classes for single bits, flexible bitcounts, single qubits, or flexible qubit counts

"""
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


class DataType(abc.ABC):
    """
    Abstract parent for all data types (classical or quantum).

    Each subclass must implement:
    - num_units -> int: The total “element count” or bit/qubit count (depending on the type).
    - bitsize: Total bits required to store one instance of this data type.
    - to_bits(...) / from_bits(...): For converting this data type to and from a bit-level representation.
    - get_domain(): If feasible, yields all possible values (e.g., for small classical types).
    - to_units(...) / from_units(...): Splits or reconstructs the data into smaller “units.”

    This design ensures that the shape or size of the data is primarily stored here, making
    the `Data` class in `data.py` simpler in handling dynamic aspects like symbolic shapes.
    """
    @property
    @abc.abstractmethod
    def num_units(self) -> int:
        """
        Number of "fundamental units" (bits, qubits, or something else)
        required to represent a single instance of this data type.
        """

    @property
    @abc.abstractmethod
    def bitsize(self) -> int:
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
        `output_shape = input_shape + (self.bitsize,)`.
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
        This operation accepts any NDArray of bits such that the last dimension equals `self.bitsize`,
        and the output array satisfies `output_shape = input_shape[:-1]`.
        """
        return np.vectorize(self.from_bits, signature='(n)->()')(bits_array)

    @abc.abstractmethod
    def get_domain(self) -> Iterable[Any]:
        """Yield all possible values representable by this type (if feasible)."""

    @abc.abstractmethod
    def to_units(self, x) -> List[int]:
        """Yields individual units (e.g., elements) corresponding to x."""

    @abc.abstractmethod
    def from_units(self, units: Sequence[int]) -> Any:
        """Combine individual units to reconstruct x."""

    @abc.abstractmethod
    def assert_valid_classical_val(self, val: Any, debug_str: str = 'val'):
        """Raises an exception if `val` is not a valid classical value for this type.

        Args:
            val: A classical value that should be in the domain of this QDType.
            debug_str: Optional debugging information to use in exception messages.
        """
    def __str__(self):
        return f"{self.__class__.__name__}({self.num_units})"

    def __repr__(self):
        return str(self)

@define
class NDimDataType(DataType):
    """
    A base class for N-dimensional data types (matrices, tensors, etc.).
    Holds:
      - shape: A tuple of dimension sizes (e.g., (rows, cols))
      - element_dtype: The Python or NumPy type (float, int, etc.)
      - val: Optionally store an actual NumPy array (or any underlying data).
    """

    shape: Tuple[int, ...]
    element_dtype: Any = float
    val: Optional[np.ndarray] = None

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def num_elements(self) -> int:
        """Total number of elements = product of all dims."""
        return np.prod(self.shape) if self.shape else 1

    @property
    def bits_per_element(self) -> int:
        """
        default:
            - If float or int => 8 bits (just an example).
            - If specialized (float32 => 32 bits, etc.), we can override or add a lookup
        """
        if self.element_dtype in (float, int):
            return 8   # For demonstration: 8 bits per float or int
        return 1      # e.g., if it's a single bit type or something else

    @property
    def total_bits(self) -> int:
        """num_elements * bits_per_element"""
        return self.num_elements * self.bits_per_element

    @property
    def bytes_per_element(self) -> int:
        """bits_per_element // 8 if we want a direct integer # of bytes"""
        return self.bits_per_element // 8

    @property
    def memory_in_bytes(self) -> int:
        """Total memory usage = num_elements * bytes_per_element"""
        return int(self.num_elements * self.bytes_per_element)

    @property
    def num_units(self) -> int:
        """
        We assume 'units' = total elements for a general array. We can allow for
        'units' to be 'bits' or 'qubits' or 'float elements' (i.e., override it if 'unit' means something else)

        """
        return self.num_elements

    @property
    def bitsize(self) -> int:
        """Implements the abstract property from DataType"""
        return self.total_bits

    def _generate_random_data(self) -> np.ndarray:
        """
        By default, create a random array with shape 'self.shape'
        in double precision. Override if we want float32, etc.
        """
        return np.random.randn(*self.shape).astype(np.float64)

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
            bits.extend([int(b) for b in np.binary_repr(val_as_int, width=self.bits_per_element)])
        return bits

    def from_bits(self, bits: Sequence[int]) -> np.ndarray:
        """
        Reconstruct an NDArray from a sequence of bits.
        For demonstration, assume bits_per_element = 8 and each set of 8 bits => 1 int/float.
        """
        # This is extremely naive, purely demonstrative
        arr_size = self.num_elements
        element_bit_size = self.bits_per_element

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

    def get_domain(self) -> Iterable[Any]:
        """Potentially huge domain for large shape. We'll skip a real enumeration here."""
        # For demonstration, we won't enumerate all possibilities
        return []

    def to_units(self, x) -> List[int]:
        """
        'Units' might mean each element of the ND array, as an int.
        Or a more advanced representation. We'll do a naive approach:
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        return x.flatten().tolist()

    def from_units(self, units: Sequence[int]) -> np.ndarray:
        """
        Reconstruct ND array from a flat list of elements.
        """
        if len(units) != self.num_elements:
            raise ValueError(f"Expected {self.num_elements} elements; got {len(units)}")
        return np.array(units).reshape(self.shape)
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

        if val.dtype != self.element_dtype:
            raise TypeError(
                f"{debug_str} must have dtype {self.element_dtype}, got {val.dtype}."
            )
    
@define
class MatrixType(NDimDataType):
    """
    A 2D matrix type specialized from `NDimDataType`
    """

    def __init__(self, rows: int=SymInt, cols: int=SymInt, element_dtype: Any = float, val=None):
        shape = (rows, cols)
        super().__init__(shape=shape, element_dtype=element_dtype, val=val)

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
        return MatrixType(self.rows, other.cols, element_dtype=self.element_dtype)

    def __str__(self):
        return f"Matrix({self.rows}, {self.cols}, dtype={self.element_dtype})"

@define
class TensorType(DataType):
    """
    Represents a rank-N generalized tensor with specified dimensions and element data type ( A base class for N-dimensional data types (matrices, tensors, etc.).)
    Holds:
      - shape: A tuple of dimension sizes (e.g., (rows, cols))
      - element_dtype: The Python or NumPy type (float, int, etc.)
      - val: Optionally store an actual NumPy array (or any underlying data).
    """
    shape: Tuple[int, ...] = field(validator = validators.instance_of(tuple))
    element_dtype: Any = float
    val: Optional[NDArray] = None
    def __attrs_post_init__(self):
        pass
        
    @property
    def num_units(self) -> int:
        """Total number of elements = product of all dims."""
        return np.prod(self.shape) if self.shape else 1
    def get_domain(self) -> Iterable[Any]:
        """Yields all possible values representable by this tensor type."""
        if self.dtype is not None:
            return itertools.product(self.dtype.get_domain(), repeat=np.prod(self.shape))
        else:
            raise NotImplementedError("Domain enumeration not supported for arbitrary tensor types.")

    @property
    def bits_per_element(self) -> int:
        """
        For resource tracking. By default:
          - If float or int => 8 bits
          - If specialized (float32 => 32 bits, etc.), you can override or add a lookup.
        """
        if self.element_dtype in (float, int):
            return 8 
        return 1    
    @property
    def bitsize(self) -> int:
        """Return the bitsize per element."""
        if isinstance(self.element_dtype, CBit):
            return 1  # CBit is 1 bit
        return 8 * self.bytes_per_element
    @property
    def total_bits(self) -> int:
        """num_units * bits_per_element."""
        return self.num_units * self.bits_per_element
    @property
    def bytes_per_element(self) -> int:
        """bits_per_element // 8 if you want a direct integer # of bytes."""
        return self.bits_per_element // 8
    @property
    def memory_in_bytes(self) -> int:
        """Total memory usage = num_units * bytes_per_element."""
        return int(self.num_units * self.bytes_per_element)
    @property
    def rank(self) -> int:
        return len(self.shape)
    
    def multiply(self, other: "TensorType") -> "TensorType":
        """
        Example of a naive "broadcast multiply" shape logic (like an Einstein summation).
        """
        new_shape = np.broadcast_shapes(self.shape, other.shape)
        return TensorType(shape=new_shape, element_dtype=self.element_dtype)
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
            bits.extend([int(b) for b in np.binary_repr(val_as_int, width=self.bits_per_element)])
        return bits
    def from_bits(self, bits: Sequence[int]) -> np.ndarray:
        """
        Reconstruct an NDArray from a sequence of bits.
        For demonstration, assume bits_per_element = 8 and each set of 8 bits => 1 int/float.
        """
        # This is extremely naive, purely demonstrative
        arr_size = self.num_units
        element_bit_size = self.bits_per_element

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
    def to_units(self, x) -> List[int]:
        """
        'Units' might mean each element of the ND array, as an int.
        Or a more advanced representation. We'll do a naive approach:
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        return x.flatten().tolist()

    def from_units(self, units: Sequence[int]) -> np.ndarray:
        """
        Reconstruct ND array from a flat list of elements.
        """
        if len(units) != self.num_units:
            raise ValueError(f"Expected {self.num_units} elements; got {len(units)}")
        return np.array(units).reshape(self.shape)
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

        if val.dtype != self.element_dtype:
            raise TypeError(
                f"{debug_str} must have dtype {self.element_dtype}, got {val.dtype}."
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
        dtype_name = self.element_dtype.__name__ if isinstance(self.element_dtype, type) else str(self.element_dtype)
        return f"TensorType({self.shape}, dtype={dtype_name})"
    def __str__(self):
        # dtype_name = self.element_dtype.__name__ if isinstance(self.element_dtype, type) else str(self.element_dtype)
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
        return hash((self.__class__, self.shape, self.element_dtype))
from qualtran.symbolics.types import is_symbolic, SymbolicComplex, SymbolicFloat, SymbolicInt
from typing import TypeVar

SymbolicT = TypeVar('SymbolicT', SymbolicInt, SymbolicFloat, SymbolicComplex)

# Classical Bit
@define
class CBit(DataType):
    """ Represents a single classical bit (0 or 1).
    - num_units=1, bitsize=1.
    - to_bits(...) checks if x in {0,1}.
    """

    @property
    def num_units(self) -> int:
        return 1

    @property
    def bitsize(self) -> int:
        return 1

    def to_bits(self, x: int) -> List[int]:
        if x not in (0, 1):
            raise ValueError("Invalid value for a classical bit. Must be 0 or 1.")
        return [x]

    def from_bits(self, bits: Sequence[int]) -> int:
        if len(bits) != 1:
            raise ValueError("Invalid bit sequence for a classical bit. Must have exactly 1 bit.")
        return bits[0]

    def to_units(self, x: int) -> List[int]:
        """Units for a classical bit are equivalent to its bit representation."""
        return self.to_bits(x)

    def from_units(self, units: Sequence[int]) -> int:
        """Reconstruct the classical bit from its units."""
        return self.from_bits(units)

    def get_domain(self) -> Iterable[Any]:
        """Return the domain of all possible values for a classical bit."""
        return [0, 1]
    def assert_valid_classical_val(self, val: Any, debug_str: str = 'val'):
        if val not in (0, 1):
            raise ValueError(f"{debug_str} must be 0 or 1. Got: {val}")
    def __hash__(self) -> int:
        return hash(("CBit",))

    def __repr__(self) -> str:
        return "CBit()"

    def __str__(self) -> str:
        return "CBit"
    
@define
class CAny(DataType):
    """
    Abstract bag of classical bits with a flexible bitcount.
    - bits: The number of bits in the bag, possibly symbolic.
    - bitsize = bits.
    - to_bits(...) / from_bits(...) to interpret integer values up to 2^bits - 1.
    """
    bits: SymInt

    @property
    def element_dtype(self):
        return CBit()

    @property
    def num_units(self) -> int:
        return self.bits

    @property
    def bitsize(self) -> int:
        return self.bits

    def to_bits(self, x: int) -> List[int]:
        """Convert an integer to its bit representation."""
        if not isinstance(x, int) or x < 0 or x >= 2**self.bits:
            raise ValueError(f"Invalid value for CAny with {self.bits} bits.")
        return [int(b) for b in bin(x)[2:].zfill(self.bits)]

    def from_bits(self, bits: Sequence[int]) -> int:
        """Reconstruct an integer from its bit representation."""
        if len(bits) != self.bits:
            raise ValueError(f"Expected {self.bits} bits; got {len(bits)}.")
        return int("".join(map(str, bits)), 2)

    def to_units(self, x: int) -> List[int]:
        """Units for CAny are equivalent to its bit representation."""
        return self.to_bits(x)

    def from_units(self, units: Sequence[int]) -> int:
        """Reconstruct the value from its units."""
        return self.from_bits(units)

    def get_domain(self) -> Iterable[Any]:
        """Enumerate all possible values representable by this type."""
        return range(2**self.bits)
    def assert_valid_classical_val(self, val: Any, debug_str: str = 'val'):
        if not isinstance(val, int) or val < 0 or val >= 2**self.bits:
            raise ValueError(f"{debug_str} must be a valid integer for {self.bits} bits. Got: {val}")
    def __hash__(self) -> int:
        return hash((self.__class__, self.bits))
    def __repr__(self) -> str:
        return f"CAny({self.bits})"

    def __str__(self) -> str:
        return f"CAny({self.bits})"

@define
class QBit(DataType):
    """Quantum bit (qubit) type.
    
    """

    @property
    def num_units(self) -> int:
        return 1

    @property
    def bitsize(self) -> int:
        return 1

    def to_bits(self, x) -> List[int]:
        """Convert a value to its binary representation."""
        if x not in (0, 1):
            raise ValueError("Invalid value for a qubit. Must be 0 or 1.")
        return [x]

    def from_bits(self, bits: Sequence[int]) -> int:
        """Convert bits to a value."""
        if len(bits) != 1:
            raise ValueError("Invalid bit sequence for a qubit. Must have exactly 1 bit.")
        return bits[0]

    def to_units(self, x) -> List[int]:
        """Convert a value to a list of individual units."""
        return self.to_bits(x)

    def from_units(self, units: Sequence[int]) -> int:
        """Reconstruct a value from individual units."""
        return self.from_bits(units)

    def get_domain(self) -> Iterable[Any]:
        """Yield all possible values representable by this type."""
        return [0, 1]  # A qubit can only represent 0 or 1
    def out(self) -> str:
        return str(self.index)
    def assert_valid_classical_val(self, val: int, debug_str: str = 'val'):
        if not (val == 0 or val == 1):
            raise ValueError(f"Bad {self} value {val} in {debug_str}")

    def __repr__(self) -> str:
        return f"<QBit {self.index}>"
    def __hash__(self) -> int:
        return hash(self.index)
    def __eq__(self, other: object) -> bool:
        return isinstance(other, QBit) and other.index == self.index
    
    def __str__(self):
        return f"q{id(self)}" if self.index is None else str(self.index)
    
@define
class QAny(DataType):
    """Opaque bag-of-qubits type."""
    qubits: Union[int, SymbolicInt] = field()
    # qubits: SymbolicInt
    @property
    def element_dtype(self):
        return QBit()
    @property
    def num_units(self) -> int:
        return self.qubits

    @property
    def bitsize(self) -> int:
        """Assume each qubit has a 1-to-1 mapping with bitsize."""
        return self.qubits

    def get_classical_domain(self) -> Iterable[Any]:
        return range(2**self.bitsize)
    def to_bits(self, x: int) -> List[int]:
        """Yields individual bits corresponding to binary representation of x"""
        self.assert_valid_classical_val(x)
        return [int(x) for x in f'{int(x):0{self.bitsize}b}']
    def to_bits_array(self, x_array: NDArray[np.integer]) -> NDArray[np.uint8]:
        """Returns the big-endian bitstrings specified by the given integers.

        Args:
            x_array: An integer or array of unsigned integers.
        """
        if is_symbolic(self.bitsize):
            raise ValueError(f"Cannot compute bits for symbolic {self.bitsize=}")

        if self.bitsize > 64:
            # use the default vectorized `to_bits`
            return super().to_bits_array(x_array)

        w = int(self.bitsize)
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
        if bitstrings.shape[1] != self.bitsize:
            raise ValueError(f"Input bitsize {bitstrings.shape[1]} does not match {self.bitsize=}")

        if self.bitsize > 64:
            # use the default vectorized `from_bits`
            return super().from_bits_array(bits_array)

        basis = 2 ** np.arange(self.bitsize - 1, 0 - 1, -1, dtype=np.uint64)
        return np.sum(basis * bitstrings, axis=1, dtype=np.uint64)
    def to_units(self, x) -> List[int]:
        """Yield individual units corresponding to x."""
        raise NotImplementedError("QAny does not support unit-level conversion.")

    def from_units(self, units: Sequence[int]) -> Any:
        """Combine individual units to reconstruct x."""
        raise NotImplementedError("QAny does not support reconstruction from units.")

    def get_domain(self) -> Iterable[Any]:
        """Return the domain of possible values (not supported for QAny)."""
        raise NotImplementedError("Domain is too large to enumerate for QAny.")
    def assert_valid_classical_val(self, val, debug_str: str = 'val'):
        pass

    def assert_valid_classical_val_array(self, val_array, debug_str: str = 'val'):
        pass

    def __hash__(self) -> int:
        return hash((self.__class__, self.qubits))
    def __repr__(self) -> str:
        return f"QAny({self.qubits})"

    def __str__(self) -> str:
        return f"QAny({self.qubits})"