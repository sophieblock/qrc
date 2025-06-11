# In-House Data Types Reference  
**Path:** `qrew/simulation/data_types.py`

---

## 1. Bit-Level Utilities

### BitNumbering  
Defines bit-order when viewing or constructing bitstrings.
```python
    class BitNumbering(Enum):
        MSB = 0  # bit-0 is most-significant (big-endian)
        LSB = 1  # bit-0 is least-significant (little-endian)
```
### BitStringView  
Immutable view of a non-negative integer as a fixed-width bitstring, with optional dtype validation.
**Fields**  
- `integer: int` — stored value  
- `nbits: int` — bit-width (auto-grows when you set a larger integer or dtype)  
- `numbering: BitNumbering` — MSB vs LSB ordering  
- `dtype: Optional[DataType]` — if set, enforces that `integer` fits in `dtype.data_width`

**Constructors**  
- `from_int`, `from_binary`, `from_array`, `from_bitstring`

**Inspectors**  
- `binary()` → raw bit-string (`str`)  
- `bits()` → `List[int]`  
- `array()` → same as `bits()`

**Converters**  
- `widen_to_dtype(dtype)` → grow `nbits` to at least `dtype.data_width`

**Operators & Hooks**  
- `__eq__`, `__hash__`, `__add__`, `__len__`, `__int__`, `__repr__`  
- `_post_setattr` keeps `nbits` in sync and validates against `dtype`

---

## 2. Symbolic & Dynamic Support

- **`SymbolicFloat = Union[float, sympy.Expr]`**  
- **`SymbolicInt   = Union[int,   sympy.Expr]`**  
- **`SymbolicComplex = Union[complex, sympy.Expr]`**  
- **`Dyn`** — singleton placeholder for “dynamic” (unknown) sizes  
- **`is_symbolic(x)`** — returns `True` if `x` or its attributes contain SymPy expressions  
- **`prod(iterable)`** — product, supports symbolic multiplication  

---

## 3. Bit-Width Helpers

- `_bits_for_dtype(dt)`  
  Returns bit-width for:
    - `torch.dtype` via `torch.iinfo/finfo`
    - `numpy.dtype` via `.itemsize`
    - Python built-ins (`int→32`, `float→32`)

- `_element_type_converter(et)`  
  Converts user-provided element types to a `DataType` (defaults to `CFloat` if needed)

- `_to_symbolic_int(v)`  
  Converts `str` → SymPy symbol, passes through ints and SymPy Expr

---

## 4. Abstract Base: `DataType`

Defines a uniform interface for all data types.

    class DataType(abc.ABC):
        @property
        @abc.abstractmethod
        def data_width(self) -> int: ...
        @abc.abstractmethod
        def to_bits(self, x) -> List[int]: ...
        @abc.abstractmethod
        def from_bits(self, bits: Sequence[int]) -> Any: ...
        @abc.abstractmethod
        def get_classical_domain(self) -> Iterable[Any]: ...
        @abc.abstractmethod
        def assert_valid_classical_val(self, val, debug_str='val'): ...
        # Provided: to_bits_array, from_bits_array, assert_valid_classical_val_array,
        # is_symbolic(), iteration_length_or_zero(), __str__, __format__

---

## 5. Classical Data Types  

_Inherit: `CType` → `NumericType` → `DataType`_

### CBit  
Single bit (0 or 1)

    class CBit(NumericType):
        @property
        def data_width(self): return 1
        def to_bits(self, x: int) -> List[int]: ...
        def from_bits(self, bits: List[int]) -> int: ...
        def get_classical_domain(self) -> Iterable[int]: ...
        def assert_valid_classical_val(self, val, debug_str='val'): ...

### CInt(n)  
Signed two’s-complement integer of `n` bits  
- Domain `[-2^(n−1), 2^(n−1))`  
- Big-endian bit order

    class CInt(NumericType):
        bit_width: SymbolicInt
        @property data_width -> bit_width
        def to_bits(self, value: int) -> List[int]: ...
        def from_bits(self, bits: List[int]) -> int: ...
        def get_classical_domain(self) -> Iterable[int]: ...
        def assert_valid_classical_val(self, val, debug_str='val'): ...

### CUInt(n)  
Unsigned integer of `n` bits  
- Domain `[0, 2^n)`

    class CUInt(NumericType):
        bit_width: SymbolicInt
        @property data_width -> bit_width
        def to_bits(self, value: int) -> List[int]: ...
        def from_bits(self, bits: List[int]) -> int: ...
        def get_classical_domain(self) -> Iterable[int]: ...
        def assert_valid_classical_val(self, val, debug_str='val'): ...

### CFxp(total_bits, num_frac, signed=False)  
Fixed-point with `total_bits`, `num_frac` fractional bits  

    class CFxp(NumericType):
        bit_width: int
        num_frac: int
        signed: bool
        @property data_width -> bit_width
        @property num_int -> bit_width − num_frac
        def to_fixed_width_int(self, x: float, require_exact=False) -> int: ...
        def float_from_fixed_width_int(self, x: int) -> float: ...
        def to_bits(self, x: int) -> List[int]: ...
        def from_bits(self, bits: List[int]) -> int: ...
        def get_classical_domain(self) -> Iterable[int]: ...
        def assert_valid_classical_val(self, val, debug_str='val'): ...

### CFloat(n)  
IEEE-754 float of width `n ∈ {8,16,32,64}` bits  

    class CFloat(NumericType):
        bit_width: int
        @property data_width -> bit_width
        def to_bits(self, value: float) -> List[int]: ...
        def from_bits(self, bits: List[int]) -> float: ...
        def get_classical_domain(self) -> Iterable[Any]: ...
        def assert_valid_classical_val(self, val, debug_str='val'): ...

---

## 6. N-Dimensional Containers

### TensorType(shape: Tuple[int,…], element_type: DataType|type)  

Represents an N-D tensor of elements of `element_type`.

- **Properties**  
  - `shape`, `element_type`, optional `val: np.ndarray`  
  - `data_width`, `nelement()`, `element_size()`, `nbytes`, `total_bits`, `rank`

- **Methods**  
  - `to_bits(x)` / `from_bits(bits)`  
  - `multiply(other)` → broadcast-style shape  
  - `assert_valid_classical_val(val)` → checks type, shape, dtype

---

## 7. Quantum Data Types  

_Inherit: `QType` → `DataType`_

### QType  
Abstract base; `data_width = num_qubits`

### QBit  
Single qubit

    class QBit(QType):
        @property num_qubits -> 1
        def to_bits(self, x) -> List[int]: ...
        def from_bits(self, bits: List[int]) -> int: ...
        def get_classical_domain(self) -> Iterable[int]: ...
        def assert_valid_classical_val(self, val, debug_str='val'): ...

### QInt(n)  
Signed quantum integer of `n` qubits  

### QUInt(n)  
Unsigned quantum integer of `n` qubits  

### QFxp(total_bits, num_frac, signed=False)  
Quantum fixed-point  

### BQUInt(bitsize, extent)  
Bounded unsigned quantum integer  

### QAny(n)  
Opaque bag of `n` qubits  

---

## 8. Testing Coverage  

Tests in `test/test_dtypes.py` verify:

- Primitive behavior (`data_width`, `to_bits`/`from_bits`)  
- Array conversions (`to_bits_array`, `from_bits_array`)  
- Domain checks & validation  
- Symbolic & dynamic widths  
- Classical vs. quantum consistency  
- Container round-trips for `TensorType`
