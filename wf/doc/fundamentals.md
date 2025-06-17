# `DataType` — the universal interface for register element types
_In-house data types <code class="filepath">./data_types.py</code>._

Every concrete dtype (classical *or* quantum) derives from `DataType`. Think of it as the “adapter” that lets high-level operations treat qubits or bits like well-typed scalars.


- **Abstract properties/methods**
  
  - `data_width` → how many fundamental units (bits or qubits) per element
  - `to_bits(x)` → `[int, …]` for a single value
  - `from_bits(bits)` → reconstruct a single value
  - `get_classical_domain()` → iterable of representable classical values (if enumerable)
  - `assert_valid_classical_val(val)` → raise if `val` out of domain
- **Provided helpers**
  
  - `to_bits_array(x_array)` and `from_bits_array(bits_array)` via NumPy-vectorization
  - `assert_valid_classical_val_array(...)`
  - `is_symbolic()` / `iteration_length_or_zero()`
  - `__str__` and `__format__` delegate to the class name + width



#### Hooks

> `_post_setattr` and `validate_dtype` keep `nbits` in sync whenever you assign to `integer` or `dtype`.


---

## Quantum Data Types (`QType`)

_All inherit from `QType` → `DataType`._

| QType (constructor)   | `data_width` / `num_qubits` | Classical domain†                                                               | Purpose & notes                                                                                                             |
| ---------------------------- | ----------------------------- | --------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `QBit()`                   | `1`                         | `{0, 1}`                                                                                    | Single qubit viewed in the computational basis. Simplest building block for larger registers.                               |
| `QInt(n)`                  | `n`                         | $[-2^{\,n-1},\,2^{\,n-1})$                                                                  | *Signed* two’s-complement integer. High (bit 0) is the sign-bit. Arithmetic wraps mod $2^n$.                               |
| `QUInt(n)`                 | `n`                         | $[0,\,2^{\,n})$                                                                             | *Unsigned* integer. Developer manages wrap-around semantics on overflow (C-style).                                          |
| `BQUInt(bitsize, L)`       | `bitsize`                   | $[0,\,L)$ with $L\le 2^{\texttt{bitsize}}$                                                  | Bounded unsigned integer; ideal for coherent for-loop indices.`iteration_length = L` may be symbolic.                       |
| `QAny(n)`                  | `n`                         | *Ambiguous* — delegates to `QUInt(n)` when coerced                                         | Opaque register of*n* qubits used when the specific dtype is unknown/irrelevant. Avoid when a precise domain matters.       |
| `QFxp(w, f, signed=False)` | `w`                         | Unsigned →$[0,\,2^{w-f})$  Signed → $[-2^{w-f-1},\,2^{w-f-1})$  **Step** $2^{-f}$ | Fixed-point real number with`f` fractional bits. Backed by `QUInt` (unsigned) or `QInt` (signed).  `float = int · 2^{-f}`. |

- `to_bits` / `from_bits` treat basis-state interpretations of qubit registers.
  In a basis-state register, the value <span class="token literal">42</span>
  is stored as a two’s-complement integer, while
  <span class="token type">QAny</span> represents an untyped qubit bundle.

## Classical Data Types (`CType`)

_Concrete classical dtypes all derive from `CType` → `NumericType` → `DataType` and therefore share the universal bit-level API (`to_bits`, `from_bits`, `assert_valid_classical_val`, …)_

- **Primitive bit and integers**: `CBit`, `CInt`, `CUInt`
- **Fixed-point, IEEE float, strings**: `CFxp`, `CFloat`, `String`

#### Scalar / “Atomic” Classical Dtypes

| CType (constructor)          | `data_width` / `bit_width`    | Classical domain†                                                                                                                             | Purpose & notes                                                                              |
| ----------------------------------- | ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `CBit()`                          | $n=1$                         | `{0, 1}`                                                                                                                                                  | Single classical bit.                                                                        |
| `CInt(n)`                         | $n$                           | $[-2^{\,n-1},\,2^{\,n-1})$                                                                                                                                | Signed two’s-complement integer with$n$ bits. Sign-bit at MSB; wraps mod $2^n$.             |
| `CUInt(n)`                        | $n$                           | $[0,\,2^{\,n})$                                                                                                                                           | Unsigned integer with$n$ bits. Developer handles overflow (wrap or error).                   |
| `CFxp(total, frac, signed=False)` | `w=total`                     | Unsigned →$[0,\,2^{\,\texttt{total-frac}})$Signed → $[-2^{\,\texttt{total-frac-1}},\,2^{\,\texttt{total-frac-1}})$**Step** $2^{-\texttt{frac}}$ | Fixed-point real number (`frac` fractional bits). Re-uses `CInt`/`CUInt` for raw bit access. |
| `CFloat(n)`                       |$n\in\{8,16,32,64\}$| IEEE-754 range for the chosen width                                                                                                                       | Classical floating-point; (de)serialises via`struct.pack` / `struct.unpack`.                 |
| `String(max_len)`                 | `w = 8 × max_len`            | All ASCII strings of length ≤`max_len`                                                                                                                   | Null-padded byte string. Each char occupies one byte.                                        |

<sup>†</sup> *“Classical domain” = set of basis-state values that can be represented exactly; superpositions are a quantum artefact and therefore not listed here.*

## Composite / Container Classical Dtypes

### `Struct(fields: dict[str, DataType])`

Packed **record / struct** akin to a C `struct` or Rust `struct`.

#### Properties

* `fields` – ordered mapping `name → dtype` (order preserved)
* `data_width` – sum of field widths
* `nbytes`, `total_bits` – byte / bit totals (recursive if a field is itself composite)
* `field_order` – list of keys in declaration order

#### Core Methods

* `to_bits(value_dict)` – concatenates each field’s bitstring
* `from_bits(bits)` – slices the concatenation back into a `dict`
* `assert_valid_classical_val(value_dict)` – field-wise validation

```python
# A composite structure {id: uint16, mass: float32, label: 8-char string}
particle = Struct(
    fields={
        "id":    CUInt(16),
        "mass":  CFloat(32),
        "label": String(8),
    }
)

print(f"{particle}  ––  is_classical: {is_classical(particle)}")
print(f"    - data_width (bits): {particle.data_width}")
print(f"    - nbytes:            {particle.nbytes}")
print(f"    - total_bits:        {particle.total_bits}")
```

Out:

<pre style="
  background:#f4f4f4;          /* light-grey terminal look  */
  color:#3a3a3a;               /* darker text               */
  font-family:'Fira Code', monospace;
  font-size:0.9rem;
  line-height:1.5;
">
Struct(112)  ––  is_classical: True
    - data_width (bits): 112
    - nbytes:            14
    - total_bits:        112
</pre>

## <b> N-Dim Data Types </b>

### `TensorType(shape: Tuple[int, …], element_type: DataType | type)`

Represents an **N-dimensional** array whose **elements** are themselves scalar dtypes
(e.g. `CFloat(32)`, `CUInt(8)`). Conceptually similar to a NumPy ndarray that carries a
bit-level contract.

##### Properties

* `shape`, `element_type`, optional **`val: np.ndarray`**
* `data_width` – bits **per element**
* `rank` – `len(shape)`
* `nelement()` – total number of elements
* `element_size()` / **`bytes_per_element`** – bytes per element (`data_width // 8`)
* **`nbytes`** – total bytes (`nelement × bytes_per_element`)
* **`total_bits`** – `nelement × data_width` (alias `memory_in_bytes × 8`)

##### Methods

* `to_bits(x)` / `from_bits(bits)` – naïve flatten ↔ reconstruct
* `multiply(other)` – broadcast-style shape multiplication
* `assert_valid_classical_val(val)` – validates `val.shape` & `val.dtype`
* Standard `DataType` helpers (`to_bits_array`, `__str__`, …) come for free.

#### Arithmetic helpers

`TensorType` introduces numerical helpers that *respect shape broadcasting*
for both concrete **and symbolic** dimensions:

| Helper                                        | Shape rule                                                                   |
| ----------------------------------------------- | ------------------------------------------------------------------------------ |
| `multiply(other)`                             | NumPy-style**element-wise** or broadcast multiply.                           |
| `add(other)`, `subtract(other)` † | Same broadcast logic as`multiply`.                                           |
| `dot(other)` †                    | Tensor contraction of the last axis of`self` with the first axis of `other`. |
| `reshape(new_shape)`                          | Returns a view with identical total bits.                                    |
| `transpose(*axes)`                            | Reorders axes without changing storage size.                                 |

<sup>†</sup> Only `multiply` is implemented today; `add`, `subtract`, and
`dot` are **recommended** extensions to give users a full algebraic toolkit.
All helpers must raise a clear `ValueError` when dimension symbols are
incompatible (e.g. `Dyn × 4` versus concrete `3 × 4`).

Example of a tensor with float32 element dtype:

```python
tensor = TensorType(shape=(3, 4, 4), element_type=CFloat(32))
print(f"{tensor}  ––    rank: {tensor.rank} | is_classical: {is_classical(tensor)} ")
print(f"    - nelements: {tensor.nelement()}")
print(f"    - bytes per element: {tensor.bytes_per_element}")
print(f"    - total bytes = {tensor.nbytes}")
print(f"    - total bits = {tensor.total_bits}")
```

Out:

<pre style="
  background:#f4f4f4;          /* light-grey terminal look  */
  color:#3a3a3a;               /* darker text               */
  font-family:'Fira Code', monospace;
  font-size:0.9rem;
  line-height:1.5;
">
TensorType((3, 4, 4))  ––    rank: 3 | is_classical: True 
    - nelements: 48
    - bytes per element: 4
    - total bytes = 192
    - total bits = 1536
</pre>

### `MatrixType(rows, cols, element_type=float)`

A **rank-2 convenience subclass** of `TensorType` (will eventually be folded into it).

#### Extras on top of `TensorType`

* Properties: `rows`, `cols`
* `multiply(other)` – **matrix multiplication** shape check
  (`self.cols == other.rows`); returns `MatrixType(self.rows, other.cols, …)`
* Should also expose (future work):
  `add`, `subtract` (element-wise), `transpose`, `inverse` (when square).

```python
A = MatrixType(2, 3, element_type=CFloat(32))
B = MatrixType(3, 1, element_type=CFloat(32))
C = A.multiply(B)
print("A:", A, "| B:", B, "| A×B:", C)
```

Out:

<pre style="
  background:#f4f4f4;          /* light-grey terminal look  */
  color:#3a3a3a;               /* darker text               */
  font-family:'Fira Code', monospace;
  font-size:0.9rem;
  line-height:1.5;
">
A: Matrix(2, 3) | B: Matrix(3, 1) | A×B: Matrix(2, 1)
</pre>

---

#### API Exposed by *All* Classical Dtypes

| Category                 | Methods / Properties                                             |
| -------------------------- | ------------------------------------------------------------------ |
| **Representation**       | `data_width`, `nbytes`, `to_bits(x)`, `from_bits(bits)`          |
| **Validation**           | `assert_valid_classical_val(val)` and array variant              |
| **Domain introspection** | `get_classical_domain()` (enumerable types only)                 |
| **Bulk helpers**         | Vectorised`to_bits_array`, `from_bits_array` (NumPy)             |
| **Debug UX**             | `__str__`, `__format__` → e.g. `CInt(32)`, `TensorType((3, 3))` |

> **Tip:** When you introduce a new classical dtype, implement the scalar API first (`to_bits`, `from_bits`, `assert_valid_classical_val`).
> The array helpers and pretty-printing come automatically via the base class.

---

## Bit-level Utilities

These tiny helpers sit **orthogonally** to the `DataType` hierarchy: they let
you view any integer (or list of bits) as a *bit-string* with an explicit width
and ordering, and they give every classical `DataType` a convenient
`to_bitstring` adapter.

### `BitNumbering` &nbsp;–&nbsp; bit-ordering enum
| Numbering flag | int (example) | `binary()` string ( `nbits = 4` ) | `bits()` → `List[int]` |
| -------------- | ------------- | --------------------------------- | ----------------------- |
| `MSB` (big-endian) | **13** | `1101` | `[1, 1, 0, 1]` |
| `LSB` (little-endian) | **13** | `1011` | `[1, 0, 1, 1]` |

*With `MSB` the left-most character is **bit 0** (most significant);  
with `LSB` the **right-most** character is bit 0, so the string is reversed.*


### class `BitStringView`

A tiny, immutable wrapper around:
- `integer: int` — underlying Python int
- `nbits: int` — declared bit-width (auto-grows when you set a larger integer or to fit dtype)
- `numbering: BitNumbering` — ordering of bits (`MSB` or `LSB`)
- `dtype: Optional[DataType]` — if set, validates that the integer fits in that dtype

#### **Class constructors**  — alternate entry points
  - `from_int(int, nbits=?, numbering=?, dtype=?)`:  Quick literal → view (optionally widen). Accepts Python `int` or another bit view obj
  - `from_binary(str, nbits=?, numbering=?)`: User CLI / config files. Binary string (`0b…` prefix optional).
  - `from_array(Sequence[int], nbits=?, numbering=?, dtype=?)`: Interop with bit-lists from other libs
  - `from_bitstring(other, nbits=?, numbering=?, dtype=?)`: Clone while tweaking metadata (Existing `BitStringView`)
  - `msb(…) / lsb(…)`: Convenience helper/ Alias to `from_int` with fixed ordering

#### **Inspectors**
  - `binary()` → Zero-padded, raw bit‐string (ordering respected)
  - `bits()` / `array()` → `List[int]` in display order

#### **Converters**
  - `with_numbering(numbering: BitNumbering)`: *new* `BitStringView` with the same integer/value, nbits and dtype
        but a different bit-ordering
  - `widen_to_dtype(dtype: DataType)`:  Mutates `nbits` to at least `dtype.data_width`, sets new dtype, validates value

**Magic**: `__int__`, `__len__`, `__add__`, `__eq__`, `__hash__`, `__repr__`: Make it behave like an `int` that still “remembers” its width & order


`BitStringView` is a **presentation/transport** layer:  
it can wrap *any* integer—typed or not—without pulling in the whole
`DataType` hierarchy.  That makes it ideal for:

* logging & debugging (`logger.debug("%s", bs.binary())`)
* CLI/GUI fields that accept `0b…` / `0x…` literals
* serialisers/deserialisers that need explicit endianness
* teaching examples where you want to show the raw bits

*Adopting `BitStringView` consistently throughout the codebase eliminates
a whole class of endianness and width-mismatch bugs—treat it as the
binary analogue of `str`.*  

---

#### Utility Functions & Constants

- `is_symbolic(x)`
  Detects SymPy or user-defined symbolic widths.
- `prod(iterable)`
  Product, supports symbolic multiplication.
- `Dyn`
  A singleton placeholder for “dynamic” (unknown) sizes.
- `_bits_for_dtype(dt)`, `_element_type_converter(et)`, `_to_symbolic_int(v)`
  Internal converters to unify Python/NumPy/Torch types, default element types, and string → SymPy symbol.

#### Other Relevant Helpers
- `is_classical(dtype)` / `is_quantum(dtype)` 
  make it trivial to branch logic based on the data‑type _domain_
```python
for dtype in [CBit(), QBit(), tensor_int8, sym_tensor]:
    print(f"{dtype:20} | classical={is_classical(dtype):5} | quantum={is_quantum(dtype):5}")
```
out:
<pre style="
  background:#f4f4f4;          /* light-grey terminal look  */
  color:#3a3a3a;               /* darker text               */
  font-family:'Fira Code', monospace;
  font-size:0.9rem;
  line-height:1.5;
">
CBit()               | classical=    1 | quantum=    0
QBit()               | classical=    0 | quantum=    1
TensorType((3, 4))   | classical=    1 | quantum=    0
TensorType((m, n))   | classical=    1 | quantum=    0
</pre>
---

---

# Compilation Layer <code class="filepath">./schema.py</code>

## `RegisterSpec`

<pre style="background:#272822;color:#f8f8f2;padding:0.8em;">
class RegisterSpec(
    name: str,
    dtype: DataType | type | alias,
    _shape: tuple = (),
    flow: Flow = Flow.THRU,
    variadic: bool = False
)
</pre>

Describes one **logical wire** (or bundle of wires) in a process graph.
It carries every piece of compile-time metadata needed for

* static **type-checking**,
* **resource estimation** (bits / qubits), and
* **symbolic-shape** reasoning when dimensions are unknown (`Dyn`, SymPy, Torch SymInt).

### Key Properties / methods

| property         | description                                                                            |
| ------------------ | ---------------------------------------------------------------------------------------- |
| `shape`          | Wire fan-out tuple — may contain`int`, `Dyn`, or SymPy symbols. Empty `()` ⇒ scalar. |
| `symbolic_shape` | Same as`shape`, but coerced to Torch `SymInt` where available.                         |
| `bitsize`        | Bits**per payload** (`dtype.data_width`).                                              |
| `domain`         | `"Q"` if the dtype (or its `element_type`) is quantum, else `"C"`.                     |
| `is_symbolic`    | `True` if any dimension or dtype width is symbolic.                                    |
| `total_bits()`   | `payload_bits × fan_out` — full logical width of the wire bundle.                    |
| `all_idxs()`     | Generator yielding every index tuple under the rectangular`shape`.                     |

#### Flow flags

| Enum`Flow`   | Meaning                            |
| -------------- | ------------------------------------ |
| `Flow.LEFT`  | Input-only register   (〈—)       |
| `Flow.RIGHT` | Output-only register   (—〉)      |
| `Flow.THRU`  | Pass-through (input**and** output) |

`Signature` partitions its registers into **lefts** (inputs) and **rights** (outputs) using these flags.

---

### 1 · Domain & resource example

```python
from qrew.simulation.schema import RegisterSpec, Flow
from qrew.simulation.data_types import *

qbit_reg  = RegisterSpec("qb", QBit(), flow=Flow.THRU)
cbit_reg  = RegisterSpec("cb", CBit())
tensor_c  = RegisterSpec("tc",
              TensorType((2,3), element_type=CInt(8)),
              flow=Flow.THRU)

assert [qbit_reg.domain, cbit_reg.domain, tensor_c.domain] == ["Q","C","C"]
assert qbit_reg.total_bits()  == 1      # 1 qubit
assert tensor_c.total_bits() == 6 * 8   # 48 bits
```

<b>Formula:</b>

$$
\text{total\_bits} \;=\;
\bigl(\texttt{dtype.total\_bits} \text{ or } \texttt{dtype.data\_width}\bigr)
\times
\prod(\texttt{shape})
$$

Works seamlessly with `Dyn` and SymPy symbols.

---

### 2 · Wire-shape ≠ Data-shape

Two wires; each carries a length-3 vector of 8-bit ints:

```python
reg = RegisterSpec(
    "scalar8x4",
    dtype=TensorType((3,), element_type=CInt(8)),
    shape=(2,),                 # fan-out
    flow=Flow.LEFT
)
print(f"Spec: {repr(reg)}")
print(f"__str__: {str(reg)}")
print(f"   – bitsize:            {reg.bitsize}")
print(f"   – total_bits():       {reg.total_bits()}")
print(f"   – shape:              {reg.shape}")
print(f"   – dtype:              {reg.dtype}")
print(f"   – dtype.shape:        {reg.dtype.shape}")
```

Out:

<pre style="
  background:#f4f4f4;          /* light-grey terminal look  */
  color:#3a3a3a;               /* darker text               */
  font-family:'Fira Code', monospace;
  font-size:0.9rem;
  line-height:1.5;
">
Spec: scalar8x4: TensorType((3,)) (shape=(2,)) Flow.LEFT 
__str__: InSpec(name=scalar8x4, dtype=TensorType((3,)), shape=(2,))
   – bitsize:            8
   – total_bits():       48
   – shape:              (2,)
   – dtype:              TensorType((3,))
   – dtype.shape:        (3,)
</pre>

> **To Do:** add example when `shape == ()`, a dtype *class* is **not** auto-instantiated.
> Scalars stay scalar.

---

## class `Signature`

An **ordered collection** of `RegisterSpec`s that partitions a process interface
into **inputs** (`Flow.LEFT`), **outputs** (`Flow.RIGHT`), and **through‑wires**
(`Flow.THRU`), i.e., a `Signature` defines the inputs/outputs of an operational schema. It is functional and does not access "states" like parameters (i.e. no raw data like Data). Should be a property of all `Process` subclasses & `CompositeMod` instances.

```python
sig = Signature.build(
      x=QBit(),              # LEFT (default)
      y=QBit(),
      result=RegisterSpec("result", QBit(), flow=Flow.RIGHT)
  )
```

#### Construction Helpers

| classmethod                                 | purpose                                                |
| --------------------------------------------- | -------------------------------------------------------- |
| build(**kwargs)                             | Quick ad-hoc signature from keyword args or specs.     |
| build_from_dtypes(**types)                  | Like`build` but forces each value to a concrete dtype. |
| build_from_properties(inp_props, out_props) | Convert Process property dicts to a signature.         |
| build_from_data(inputs, out_props)          | Infer from runtime`Data` + output metadata.            |

#### Core API

| member                              | description                                   |
| ------------------------------------- | ----------------------------------------------- |
| lefts() / rights()                  | Iterate over input- or output-flow specs.     |
| get_left(name) / get_right(name)    | Dict-style access by register name.           |
| groups()                            | Yield`(name, [spec…])` grouped by identifier |
| validate_data_with_register_specs() | Runtime check of`Data` against left specs.    |
| Sequence helpers                    | `sig[i]`, `len(sig)`, iteration, membership.  |

A `Signature` is pure **metadata**; it never contains actual payloads,
only the rules that payloads must satisfy.
------------------------------------------

### Helper Utilities

* `_sanitize_name(name)` — strips spaces/illegal chars, ensures no leading digit.
* `canonicalize_dtype(value)` — normalises builtin / NumPy / torch dtypes
  to in‑house `DataType` objects (`TensorType`, etc.).
* `get_shape(v)` — converts any “shape‑like” value into a canonical `tuple`.
* `qubit_count_for(reg)` — returns `int(reg.total_bits())` for quantum regs,
  else `0`.

