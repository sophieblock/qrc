
---
# In-house data types <code class="filepath">./data_types.py</code>


### class `BitNumbering`  
An `Enum` for bit-order interpretation.

```python
class BitNumbering(Enum):
    MSB = 0  # bit-0 is most-significant (big-endian)
    LSB = 1  # bit-0 is least-significant (little-endian)
```
> CBit()

### class `BitStringView`
A thin, immutable wrapper around an integer + bit-width + bit-order + optional `DataType`.

#### Fields
- `integer: int` — the stored value  
- `nbits: int` — number of bits (auto-grows when you set a larger integer or dtype)  
- `numbering: BitNumbering` — ordering of bits  
- `dtype: Optional[DataType]` — if set, validates that the integer fits in that dtype  

#### Key methods
- **Class constructors**  
  - `from_int(...)`, `from_binary(...)`, `from_array(...)`, `from_bitstring(...)` — alternate entry points  
- **Inspectors**  
  - `binary()` → raw bit‐string  
  - `bits()` → `List[int]`  
  - `array()` → same as `bits()`  
- **Converters**  
  - `widen_to_dtype(dtype)` → grow `nbits` to at least `dtype.data_width`  
- **Magic**  
  - `__eq__`, `__hash__`  
  - `__add__` (integer addition)  
  - `__len__` (returns `nbits`)  
  - `__int__`, `__repr__`

#### Hooks
> `_post_setattr` and `validate_dtype` keep `nbits` in sync whenever you assign to `integer` or `dtype`.  
### class `DataType`
Abstract parent for all data types (classical and quantum).

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

---

## Quantum Data Types (`QType`)

_All inherit from `QType` → `DataType`._

| QType&nbsp;(constructor)      | `data_width` / `num_qubits` | Classical domain<sup>†</sup>                               | Purpose & notes                                                                                                   |
|-------------------------------|-----------------------------|------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| `QBit()`                      | `1`                         | `{0, 1}`                                                   | Single qubit viewed in the computational basis. Simplest building block for larger registers.                      |
| `QInt(n)`                     | `n`                         | $[-2^{\,n-1},\,2^{\,n-1})$                                 | *Signed* two’s-complement integer. High (bit 0) is the sign-bit. Arithmetic wraps mod $2^n$.                       |
| `QUInt(n)`                    | `n`                         | $[0,\,2^{\,n})$                                            | *Unsigned* integer. Developer manages wrap-around semantics on overflow (C-style).                                 |
| `BQUInt(bitsize, L)`          | `bitsize`                   | $[0,\,L)$ with $L\le 2^{\texttt{bitsize}}$                 | Bounded unsigned integer; ideal for coherent for-loop indices. `iteration_length = L` may be symbolic.             |
| `QAny(n)`                     | `n`                         | *Ambiguous* — delegates to `QUInt(n)` when coerced        | Opaque register of *n* qubits used when the specific dtype is unknown/irrelevant. Avoid when a precise domain matters. |
| `QFxp(w, f, signed=False)`    | `w`                         | Unsigned → $[0,\,2^{w-f})$  <br>Signed → $[-2^{w-f-1},\,2^{w-f-1})$  <br>**Step** $2^{-f}$ | Fixed-point real number with `f` fractional bits. Backed by `QUInt` (unsigned) or `QInt` (signed).  `float = int · 2^{-f}`. |

- `to_bits` / `from_bits` treat basis-state interpretations of qubit registers.
In a basis-state register, the value <span class="token literal">42</span>
is stored as a two’s-complement integer, while
<span class="token type">QAny</span> represents an untyped qubit bundle.

## Classical Data Types (`CType`)


_Concrete classical dtypes all derive from `CType` → `NumericType` → `DataType` and therefore share the universal bit-level API (`to_bits`, `from_bits`, `assert_valid_classical_val`, …)_

- **Primitive bit and integers**: `CBit`, `CInt`, `CUInt`

- **Fixed-point, IEEE float, strings**: `CFxp`, `CFloat`, `String`

#### Scalar / “Atomic” Classical Dtypes

| CType&nbsp;(constructor)                   | `data_width` / `bit_width` | Classical domain<sup>†</sup>                             | Purpose & notes                                                                                     |
|-------------------------------------------|---------------------------|----------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| `CBit()`                                  | $n=1$                       | `{0, 1}`                                                 | Single classical bit.                                                                               |
| `CInt(n)`                                 | $n$                       | $[-2^{\,n-1},\,2^{\,n-1})$                               | Signed two’s-complement integer with $n$ bits. Sign-bit at MSB; wraps mod $2^n$.                                  |
| `CUInt(n)`                                | $n$                    | $[0,\,2^{\,n})$                                          | Unsigned integer with $n$ bits. Developer handles overflow (wrap or error).                                       |
| `CFxp(total, frac, signed=False)`         | `w=total`                   | Unsigned → $[0,\,2^{\,\texttt{total-frac}})$<br>Signed → $[-2^{\,\texttt{total-frac-1}},\,2^{\,\texttt{total-frac-1}})$<br>**Step** $2^{-\texttt{frac}}$ | Fixed-point real number (`frac` fractional bits). Re-uses `CInt`/`CUInt` for raw bit access.        |
| `CFloat(n)`                               | $$n\in\{8,16,32,64\}$$       | IEEE-754 range for the chosen width                      | Classical floating-point; (de)serialises via `struct.pack` / `struct.unpack`.                       |
| `String(max_len)`                         | `w = 8 × max_len`             | All ASCII strings of length ≤ `max_len`                  | Null-padded byte string. Each char occupies one byte.                                               |


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

| Helper | Shape rule |
|--------|------------|
| `multiply(other)` | NumPy-style **element-wise** or broadcast multiply. |
| `add(other)`, `subtract(other)` <sup>†</sup> | Same broadcast logic as `multiply`. |
| `dot(other)` <sup>†</sup> | Tensor contraction of the last axis of `self` with the first axis of `other`. |
| `reshape(new_shape)` | Returns a view with identical total bits. |
| `transpose(*axes)` | Reorders axes without changing storage size. |

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
| Category              | Methods / Properties |
|-----------------------|----------------------|
| **Representation**    | `data_width`, `nbytes`, `to_bits(x)`, `from_bits(bits)` |
| **Validation**        | `assert_valid_classical_val(val)` and array variant |
| **Domain introspection** | `get_classical_domain()` (enumerable types only) |
| **Bulk helpers**      | Vectorised `to_bits_array`, `from_bits_array` (NumPy) |
| **Debug UX**          | `__str__`, `__format__` → e.g. `CInt(32)`, `TensorType((3, 3))` |

> **Tip:** When you introduce a new classical dtype, implement the scalar API first (`to_bits`, `from_bits`, `assert_valid_classical_val`).  
> The array helpers and pretty-printing come automatically via the base class.

---

Utility Functions & Constants

- `is_symbolic(x)`  
  Detects SymPy or user-defined symbolic widths.

- `prod(iterable)`  
  Product, supports symbolic multiplication.

- `Dyn`  
  A singleton placeholder for “dynamic” (unknown) sizes.

- `_bits_for_dtype(dt)`, `_element_type_converter(et)`, `_to_symbolic_int(v)`  
  Internal converters to unify Python/NumPy/Torch types, default element types, and string → SymPy symbol.
---

# Compilation Layer <code class="filepath">./schema.py</code> 
## `RegisterSpec`
<code class="signature">class RegisterSpec(name: str, dtype: DataType | type | alias, _shape: (), flow: Flow = Flow.THRU, variadic: bool = False)</code>


Describes a single **logical wire** (or bundle of wires) in a workflow graph. It stores *all* compile-time information needed for:

* **type checking** (matching `Data` at runtime),  
* **resource estimation** (qubit / bit counts), and  
* **symbolic-shape reasoning** when some dimensions are dynamic.
#### Properties

| property          | description                                                                  |
|-------------------|------------------------------------------------------------------------------|
| `shape`           | Tuple of concrete wire dimensions (Public accessor to `_shape`). Could be a tuple of `int` / `Dyn` / sympy  dimensions. An  empty `()` means a scalar wire/register          |
| `symbolic_shape`  | Same as `shape` but with  placeholders (`SymInt` for torch‑symbolic sizes)   |
| `bitsize`         | Bits *or qubits* **per single payload** (`dtype.data_width`).                |
| `domain`          | `"Q"` if the register’s dtype (or its element_type) is quantum; else `"C"`.  |
| `is_symbolic`     | `True` if any dimension or dtype size is symbolic.                           |
| `total_bits()`    | `payload_bits × fan_out` — total logical bits conveyed by the wire‑bundle.   |
| `all_idxs()`      | Generator over every index tuple given the rectangular `shape`.              |

---

`Flow` is a tiny `enum.Flag` used **only** to tag a register as input / output:

| Enum `Flow` | Meaning | Bitwise Behaviour |
|-------------|---------|-------------------|
| `Flow.LEFT`   | Register is **input-only** to a process. | `Flow.LEFT` |
| `Flow.RIGHT`     | Register is **output-only** from a process. | `Flow.RIGHT` |
| `Flow.THRU`      | Register is both input **and** output (pass-through). | `Flow.LEFT \| Flow.RIGHT` |


Internally these flags let the `Signature` split registers into **lefts** (inputs) and **rights** (outputs).

#### Behaviour Notes


* **Post-init coercion** – if `dtype` is a *class* and `_shape` is non-empty, the class is **instantiated** with that shape so later code always sees a concrete dtype instance.
* **Equality** (`__eq__`) – compares name, flow, shape and *compatible* dtypes (with `Dyn` wildcards considered equal for `TensorType`).
* **Data matching** – `matches_data()` / `matches_data_list()` validate real `Data` objects against the spec; if `variadic=True` the spec may absorb an arbitrary number of sequential `Data` arguments.


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

| classmethod                               | purpose                                                 |
|-------------------------------------------|---------------------------------------------------------|
| build(**kwargs)                           | Quick ad-hoc signature from keyword args or specs.      |
| build_from_dtypes(**types)                | Like `build` but forces each value to a concrete dtype. |
| build_from_properties(inp_props, out_props)| Convert Process property dicts to a signature.         |
| build_from_data(inputs, out_props)        | Infer from runtime `Data` + output metadata.            |


#### Core API

| member                              | description                                   |
|-------------------------------------|-----------------------------------------------|
| lefts() / rights()                  | Iterate over input- or output-flow specs.     |
| get_left(name) / get_right(name)    | Dict-style access by register name.           |
| groups()                            | Yield `(name, [spec…])` grouped by identifier |
| validate_data_with_register_specs() | Runtime check of `Data` against left specs.   |
| Sequence helpers                    | `sig[i]`, `len(sig)`, iteration, membership.  |

A `Signature` is pure **metadata**; it never contains actual payloads,
only the rules that payloads must satisfy.
---

### Helper Utilities

* `_sanitize_name(name)` — strips spaces/illegal chars, ensures no leading digit.  
* `canonicalize_dtype(value)` — normalises builtin / NumPy / torch dtypes
  to in‑house `DataType` objects (`TensorType`, etc.).  
* `get_shape(v)` — converts any “shape‑like” value into a canonical `tuple`.  
* `qubit_count_for(reg)` — returns `int(reg.total_bits())` for quantum regs,
  else `0`.
