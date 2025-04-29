import pytest
import numpy as np
import random 
import math

from workflow.simulation.refactor.dtypes import *

random.seed(1234)


def bits_to_int_reference(bs: np.ndarray) -> int:
    # bs is a 1D array of 0/1, big-endian
    return int("".join(str(int(b)) for b in bs.tolist()), 2)
def int_to_bits_reference(num: int, bit_count: int) -> np.ndarray:
    """Convert a non-negative integer to its big-endian bit array."""
    assert 0 <= num < 2**bit_count, f"{num} out of range for {bit_count} bits"
    return np.array([(num >> i) & 1 for i in reversed(range(bit_count))], dtype=np.uint8)

def assert_to_and_from_bits_array_consistent(dtype, values: np.ndarray):
    values = np.asanyarray(values)
    bits_array = dtype.to_bits_array(values)
    # Check element-wise conversion.
    for val, bits in zip(values.reshape(-1), bits_array.reshape(-1, dtype.data_width)):
        expected_bits = np.array(dtype.to_bits(val))
        np.testing.assert_array_equal(bits, expected_bits)
    # Check roundtrip conversion.
    values_rt = dtype.from_bits_array(bits_array)
    np.testing.assert_array_equal(values_rt, values)

def test_bit():
    qbit = QBit()
    assert qbit.num_qubits == 1
    assert qbit.data_width == 1
    assert str(qbit) == 'QBit()'

    cbit = CBit()
    assert cbit.bit_width == 1
    assert cbit.data_width == 1
    assert str(CBit()) == 'CBit()'

@pytest.mark.parametrize(
    "dtype_cls, arg, attr_name, expected_str",
    [
        (QInt,   8, "data_width", "QInt(8)"),
        (CInt,   8,   "data_width", "CInt(8)"),
        (QUInt,  8, "data_width", "QUInt(8)"),
        (CUInt,  8,   "data_width", "CUInt(8)"),
    ],
)
def test_int_types(dtype_cls, arg, attr_name, expected_str):
    # concrete
    dt = dtype_cls(arg)
    assert getattr(dt, attr_name) == arg
    assert str(dt) == expected_str

    # symbolic
    n = sympy.symbols("x")
    dt_sym = dtype_cls(n)
    assert getattr(dt_sym, attr_name) == n
    assert str(dt_sym) == f"{dtype_cls.__name__}(x)"
    assert is_symbolic(dtype_cls(sympy.Symbol("x")))

@pytest.mark.parametrize(
    "dtype_cls, bits, frac, signed, str_repr, num_int",
    [
        # quantum
        (QFxp, 16, 15, False, "QFxp(16, 15)", 1),
        (QFxp, 16, 15, True,  "QFxp(16, 15, True)", 1),
        # classical
        (CFxp, 16, 15, False, "CFxp(16, 15)", 1),
        (CFxp, 16, 15, True,  "CFxp(16, 15, True)", 1),
    ],
)
def test_fxp_types(dtype_cls, bits, frac, signed, str_repr, num_int):
    dt = dtype_cls(bits, frac, signed) if signed else dtype_cls(bits, frac)
    assert str(dt) == str_repr
    # quantum uses .num_qubits, classical .num_bits
    attr = "num_qubits" if dtype_cls is QFxp else "bit_width"
    assert getattr(dt, attr) == bits
    assert dt.num_int == num_int
    

    tpl = dt.fxp_dtype_template().dtype
    # same template naming for both quantum & classical
    suffix = "s" if signed else "u"
    assert tpl == f"fxp-{suffix}{bits}/{frac}"

    # invalid params
    with pytest.raises(ValueError, match="data_width must be > 1.") if dtype_cls is QFxp else pytest.raises(ValueError):
        dtype_cls(1, 1, True)
    # num_frac too large
    with pytest.raises(ValueError, match="num_frac must be less than"):
        dtype_cls(4, 4, True)
    # bit_width too small
    with pytest.raises(ValueError, match="bit_width must be >="):
        dtype_cls(4, 5)

    # symbolic
    b, f = sympy.symbols("b f")
    sym_dt = dtype_cls(b, f, True)
    assert getattr(sym_dt, attr) == b
    assert sym_dt.num_int == b - f
    assert is_symbolic(dtype_cls(*sympy.symbols("x y")))

@pytest.mark.parametrize(
    "dtype",
    [
        QBit(), CBit(),
        QInt(4), CInt(4),
        QUInt(4), CUInt(4),
    ],
)
def test_domain_and_validation_arr(dtype):
    arr = np.array(list(dtype.get_classical_domain()))
    dtype.assert_valid_classical_val_array(arr)

@pytest.mark.parametrize(
    "dtype, bad_vals",
    [
        (QBit(),    [-1, '|0>']),
        (CBit(),    [-1, '|0>']),
        (QInt(4),   [-9]),
        (CInt(4),   [-9]),
        (QUInt(3),  [8, -1]),
        (CUInt(3),  [8, -1]),
    ],
)
def test_validation_errs(dtype, bad_vals):
    for val in bad_vals:
        with pytest.raises(ValueError):
            dtype.assert_valid_classical_val(val)


@pytest.mark.parametrize(
    "dtype, good_arr, bad_arr",
    [
        (QBit(),    [0,1], [-1,1]),
        (CBit(),    [0,1], [-1,1]),
    ],
)
def test_validate_arrays(dtype, good_arr, bad_arr):
    rs = np.random.RandomState(52)
    arr_good = rs.choice(good_arr, size=(10,4))
    dtype.assert_valid_classical_val_array(arr_good)

    arr_bad = rs.choice(bad_arr, size=(10,4))
    with pytest.raises(ValueError):
        dtype.assert_valid_classical_val_array(arr_bad)
# ---------------------------------------------------------------------------
#  Symbolic / Dyn width cases with explicit per-domain overrides
# ---------------------------------------------------------------------------
def test_symbolic_int_cases():
    """
    Exhaustively verify consistency for
      • symbolic widths (SymPy)      – CInt(n)  vs  CInt('n')
      • Dyn width                    – CInt(Dyn)
    across every combination of global, classical, and quantum levels.
    """
    # -------- symbolic & Dyn dtypes ---------------------------------
    n, m   = sympy.symbols("n m", positive=True, integer=True)
    sym_ci = CInt(n)
    sym_ci2 = CInt("n")           # string → SymPy symbol 'n'
    sym_qi = QInt(m)

    dyn_ciA = CInt(Dyn)
    dyn_ciB = CInt(Dyn)
    ci8     = CInt(8)

    # quick alias
    def ok(a, b, g, c=None, q=None):
        return check_dtypes_consistent(a, b, g,
                                       classical_level=c,
                                       quantum_level=q)

    # 1) cross-domain symbolic must be False everywhere ----------------
    for g in DTypeCheckingSeverity:
        assert not ok(sym_ci, sym_qi, g)

    # 2) symbolic-vs-symbolic CInt is True for ALL ladder combos -------
    for g in DTypeCheckingSeverity:
        # global only
        assert ok(sym_ci, sym_ci2, g)
        # explicit single-domain overrides
        for c in list(C_PromoLevel) + [None]:
            assert ok(sym_ci, sym_ci2, g, c=c)
        for q in list(Q_PromoLevel) + [None]:
            assert ok(sym_ci, sym_ci2, g, q=q)
        # explicit both-domain overrides
        for c in C_PromoLevel:
            for q in Q_PromoLevel:
                assert ok(sym_ci, sym_ci2, g, c=c, q=q)

    # 3) Dyn width rules ----------------------------------------------
    #    Dyn vs Dyn  → always True
    for g in DTypeCheckingSeverity:
        assert ok(dyn_ciA, dyn_ciB, g)

    #    Dyn vs concrete width: allowed in ANY+LOOSE (PROMOTE/CAST),
    #    forbidden in STRICT
    assert not ok(dyn_ciA, ci8, DTypeCheckingSeverity.STRICT)
    assert ok(dyn_ciA,  ci8, DTypeCheckingSeverity.ANY)
    assert ok(dyn_ciA,  ci8, DTypeCheckingSeverity.LOOSE)
# 6) consistency with “Any” type
@pytest.mark.parametrize(
    "dtype, any_type",
    [
        # quantum
        (QFxp(4,4), QAny(4)),
        (QInt(4),   QAny(4)),
        (QInt(Dyn),   QAny(4)),
        (QUInt(4),  QAny(4)),
        (BQUInt(4,5), QAny(4)),
        # classical
        
        # (CInt(4),   CInt(4)),
        # (CUInt(4),  CInt(4)),
        # (CFxp(4,4), CInt(4)),
        # (CUInt(8),  CFxp(8,0)),
    ],
)
def test_any_consistency(dtype, any_type):
    assert check_dtypes_consistent(dtype, any_type)

@pytest.mark.parametrize(
    "dtype_a, dtype_b, severity, expected",
    [
        # STRICT: only identical bit‐width & type
        (CInt(8),  CInt(8),   DTypeCheckingSeverity.STRICT, True),
        (CInt(8),  CFloat(8), DTypeCheckingSeverity.STRICT, False),
        (CFxp(8,4,True),  CFxp(8,4,True), DTypeCheckingSeverity.STRICT, True),
        (CFxp(8,4,True),  CFxp(8,3,True), DTypeCheckingSeverity.STRICT, False),

        # PROMOTE: allow int → float if same width; nothing else
        (CInt(8),  CFloat(8), DTypeCheckingSeverity.ANY,    True),
        (CInt(8),  CUInt(8),  DTypeCheckingSeverity.ANY,    False),

        # CAST: allow bit‐cast within same width
        (CUInt(8), CFxp(8,0), DTypeCheckingSeverity.LOOSE,  True),
        (CFloat(32), CInt(32), DTypeCheckingSeverity.LOOSE, True),

        # Negative checks
        (CUInt(8), CFxp(8,0), DTypeCheckingSeverity.ANY,    False),
        (CInt(8),   CFloat(8), DTypeCheckingSeverity.LOOSE, True),  # float‐int also ok under CAST
    ],
)
def test_classical_promo_levels(dtype_a, dtype_b, severity, expected):
    """
    Verify C_PromoLevel behavior via the global DTypeCheckingSeverity:
      - STRICT  ↔ C_PromoLevel.STRICT
      - ANY     ↔ C_PromoLevel.PROMOTE
      - LOOSE   ↔ C_PromoLevel.CAST
    """
    result = check_dtypes_consistent(dtype_a, dtype_b, severity)
    assert result is expected


@pytest.mark.parametrize(
    "dtype, counterpart_cls",
    [
        (QUInt(4), QFxp),
        (BQUInt(4, 5),QFxp),
        (CUInt(4), CFxp),
    ],
)
def test_type_errors_fxp_uint(dtype, counterpart_cls):
   
    # valid width 4 & 0
    assert check_dtypes_consistent(dtype, counterpart_cls(4,4))
    assert check_dtypes_consistent(dtype, counterpart_cls(4,0))
    # others fail
    assert not check_dtypes_consistent(dtype, counterpart_cls(4,2))
    assert not check_dtypes_consistent(dtype, counterpart_cls(4,3, True))
    assert not check_dtypes_consistent(dtype, counterpart_cls(4,0, True))



@pytest.mark.parametrize('qdtype', [QInt(4)])
def test_type_errors_fxp_int(qdtype):
    assert not check_dtypes_consistent(qdtype, QFxp(4, 0))
    assert not check_dtypes_consistent(qdtype, QFxp(4, 4))


def test_type_errors_fxp():
    assert not check_dtypes_consistent(QFxp(4, 4), QFxp(4, 0))
    assert not check_dtypes_consistent(QFxp(4, 3, signed=True), QFxp(4, 0))
    assert not check_dtypes_consistent(QFxp(4, 3), QFxp(4, 0))


    assert not check_dtypes_consistent(QFxp(8, 4), QFxp(8, 3))

@pytest.mark.parametrize(
    'qdtype_a', [QUInt(4), BQUInt(4, 5), QInt(4)]
)
@pytest.mark.parametrize(
    'qdtype_b', [QUInt(4), BQUInt(4, 5),QInt(4), ]
)
def test_qtype_errors_matrix(qdtype_a, qdtype_b):
    if qdtype_a == qdtype_b:
        assert check_dtypes_consistent(qdtype_a, qdtype_b)
    elif isinstance(qdtype_a, (QInt, QUInt, BQUInt)) and isinstance(qdtype_b, (QInt, QUInt, BQUInt)):
        assert check_dtypes_consistent(qdtype_a, qdtype_b)
    else:
        assert not check_dtypes_consistent(qdtype_a, qdtype_b)

#————————————————————————————————————————————————————————————————————————————————————————
# 2) CFloat round-trip bits (16, 32, 64) and error on 128
#————————————————————————————————————————————————————————————————————————————————————————
@pytest.mark.parametrize("width, vals", [
    (16, [0.5, -0.5, 1.0, -1.0]),
    (32, [3.14, -2.71]),
    (64, [1e-100, -1e100]),
])
def test_cfloat_to_and_from_bits_roundtrip(width, vals):
    dt = CFloat(bit_width=width)
    for v in vals:
        bits = dt.to_bits(v)
        assert len(bits) == width
        recovered = dt.from_bits(bits)
        # allow small rounding error for IEEE floats
        assert pytest.approx(v, rel=1e-6, abs=1e-12) == recovered

def test_cfloat_unsupported_width():
    with pytest.raises(ValueError):
        CFloat(bit_width=128).to_bits(0.0)

@pytest.mark.parametrize(
    'cdtype_a', [CUInt(4),  CInt(4)]
)
@pytest.mark.parametrize(
    'cdtype_b', [CUInt(4), CInt(4), ]
)
def test_ctype_errors_matrix(cdtype_a, cdtype_b):
    if cdtype_a == cdtype_b:
        assert check_dtypes_consistent(cdtype_a, cdtype_b)
    elif isinstance(cdtype_a, (CInt, CUInt)) and isinstance(cdtype_b, (CInt, CUInt)):
        assert check_dtypes_consistent(cdtype_a, cdtype_b)
    else:
        assert not check_dtypes_consistent(cdtype_a, cdtype_b)     
        
@pytest.mark.parametrize(
    "dtype_cls, args",
    [
        (QInt, (4,)),
        (CInt, (4,)),
    ],
)
def test_int_to_and_from_bits(dtype_cls, args):
    dt = dtype_cls(*args)
    # domain per type
    domain = list(dt.get_classical_domain())
    # bit‐roundtrip for every value
    for x in domain:
        bits = dt.to_bits(x)
        assert dt.from_bits(bits) == x

    # out‐of‐range must raise
    with pytest.raises(ValueError):
        dt.to_bits(max(domain) + 1)

    # array version
    assert_to_and_from_bits_array_consistent(dt, domain)


@pytest.mark.parametrize(
    "dtype_cls, args",
    [
        (QUInt, (4,)),
        (CUInt, (4,)),
    ],
)
def test_uint_to_and_from_bits(dtype_cls, args):
    dt = dtype_cls(*args)
    domain = list(dt.get_classical_domain())

    for x in domain:
        bits = dt.to_bits(x)
        assert dt.from_bits(bits) == x

    with pytest.raises(ValueError):
        dt.to_bits(max(domain) + 1)

    assert_to_and_from_bits_array_consistent(dt, domain)

def assert_bits_roundtrip(dtype, values: np.ndarray):
    bits_arr = dtype.to_bits_array(values)
    values_rt = dtype.from_bits_array(bits_arr)
    np.testing.assert_equal(values_rt, values)

# --- Multidimensional simulation for dtypes ---
@pytest.mark.parametrize("q_dtype, c_dtype", [
    (QBit(), CBit()),
    (QInt(5), CInt(5)),
    (QUInt(5), CUInt(5)),
    (QFxp(5, 3, signed=True), CFxp(5, 3, signed=True)),
])
def test_multidimensional_sim_for_dtypes(q_dtype, c_dtype):
    # For both quantum and classical, get the classical domain if enumerable.

    try:
        q_domain = list(q_dtype.get_classical_domain())
    except Exception:
        q_domain = None
    try:
        c_domain = list(c_dtype.get_classical_domain())
    except Exception:
        c_domain = None

    # If the domain is enumerable, test roundtrip conversion.
    if q_domain is not None:
        q_values = np.array(q_domain)
        # print(q_values)
        assert_bits_roundtrip(q_dtype, q_values)
    if c_domain is not None:
        c_values = np.array(c_domain)
        # print(c_values)
        assert_bits_roundtrip(c_dtype, c_values)
        
    # Additionally, for types with the same fundamental unit count, the roundtrip should yield the same results.
    # (This ensures that even if implementations differ, the interface produces equivalent outputs.)
    if q_domain is not None and c_domain is not None:
        np.testing.assert_equal(np.array(q_domain), np.array(c_domain))

def test_multidimensional_sim_for_large_int():
    # Use an 100-bit signed integer type for quantum and classical.
    q_dtype = QInt(100)
    c_dtype = CInt(100)

    # Values to test: a sample of large positive and negative integers.
    values = np.array([2**88 - 1, 2**12 - 1, 2**54 - 1, 1 - 2**72, 1 - 2**62])

    # Perform roundtrip conversion for each type.
    assert_bits_roundtrip(q_dtype, values)
    assert_bits_roundtrip(c_dtype, values)
    # And check that the two dtypes produce identical roundtrip results.
    q_rt = q_dtype.from_bits_array(q_dtype.to_bits_array(values))
    c_rt = c_dtype.from_bits_array(c_dtype.to_bits_array(values))
    np.testing.assert_equal(q_rt, c_rt)


def test_bits_to_int():
    rs = np.random.RandomState(52)
    bitstrings = rs.choice([0, 1], size=(100, 23))

    nums = QUInt(23).from_bits_array(bitstrings)
    assert nums.shape == (100,)

    for num, bs in zip(nums, bitstrings):
        ref_num = bits_to_int_reference(bs)
        assert num == ref_num

    # single-bitstring case:
    single_bs = np.array([[1, 0]])       # note: shape should be (1, 2)
    single_nums = QUInt(2).from_bits_array(single_bs)
    assert single_nums.shape == (1,)
    assert single_nums[0] == 2

def test_int_to_bits():
    rs = np.random.RandomState(52)
    nums = rs.randint(0, 2**23 - 1, size=(100,), dtype=np.uint64)

    bitstrings = QUInt(23).to_bits_array(nums)
    assert bitstrings.shape == (100, 23)

    for num, bs in zip(nums, bitstrings):
        ref_bs = int_to_bits_reference(int(num), bit_count=23)
        np.testing.assert_array_equal(ref_bs, bs)

    # out-of-bounds values should raise
    with pytest.raises(AssertionError):
        QUInt(8).to_bits_array(np.array([4, -2], dtype=np.int64))


def test_bounded_quint_to_and_from_bits():
    bquint4 = BQUInt(4, 12)
    assert [*bquint4.get_classical_domain()] == [*range(0, 12)]
    assert list(bquint4.to_bits(10)) == [1, 0, 1, 0]
    with pytest.raises(ValueError):
        BQUInt(4, 12).to_bits(13)

    assert_to_and_from_bits_array_consistent(bquint4, range(0, 12))


@pytest.mark.parametrize(
    "dtype_cls, args",
    [
        (QBit, ()),
        (CBit, ()),
    ],
)
def test_bit_to_and_from_bits(dtype_cls, args):
    dt = dtype_cls(*args)
    for v in (0, 1):
        assert dt.from_bits(dt.to_bits(v)) == v
    with pytest.raises(ValueError):
        dt.to_bits(2)
    assert_to_and_from_bits_array_consistent(dt, [0, 1])

def test_qany_to_and_from_bits():
    assert list(QAny(4).to_bits(10)) == [1, 0, 1, 0]

    assert_to_and_from_bits_array_consistent(QAny(4), range(16))

@pytest.mark.parametrize(
    "dtype_cls, args",
    [
        (QFxp, (6, 4, False)),
        (CFxp, (6, 4, False)),
        (QFxp, (6, 4, True)),
        (CFxp, (6, 4, True)),
    ],
)
def test_fxp_to_and_from_bits(dtype_cls, args):
    dt = dtype_cls(*args)
    # pick a handful of representable values
    samples = list(dt.get_classical_domain())[:8]
    assert_to_and_from_bits_array_consistent(dt, samples)


def test_qfxp_to_and_from_bits():
    assert_to_and_from_bits_array_consistent(
        QFxp(4, 3, False), [QFxp(4, 3, False).to_fixed_width_int(x) for x in [1 / 2, 1 / 4, 3 / 8]]
    )
    assert_to_and_from_bits_array_consistent(
        QFxp(4, 3, True),
        [
            QFxp(4, 3, True).to_fixed_width_int(x)
            for x in [1 / 2, -1 / 2, 1 / 4, -1 / 4, -3 / 8, 3 / 8]
        ],
    )


def test_qfxp_to_fixed_width_int():
    assert QFxp(6, 4).to_fixed_width_int(1.5) == 24 == 1.5 * 2**4
    assert QFxp(6, 4, signed=True).to_fixed_width_int(1.5) == 24 == 1.5 * 2**4
    assert QFxp(6, 4, signed=True).to_fixed_width_int(-1.5) == -24 == -1.5 * 2**4


def test_qfxp_from_fixed_width_int():
    qfxp = QFxp(6, 4)
    
    for x_int in qfxp.get_classical_domain():
        x_float = qfxp.float_from_fixed_width_int(x_int)
        x_int_roundtrip = qfxp.to_fixed_width_int(x_float)
        assert x_int == x_int_roundtrip
    cfxp = CFxp(6, 4)
    for x_int in cfxp.get_classical_domain():
        x_float = cfxp.float_from_fixed_width_int(x_int)
        x_int_roundtrip = cfxp.to_fixed_width_int(x_float)
        assert x_int == x_int_roundtrip

    for float_val in [1.5, 1.25]:
        assert qfxp.float_from_fixed_width_int(qfxp.to_fixed_width_int(float_val)) == float_val

        assert cfxp.float_from_fixed_width_int(cfxp.to_fixed_width_int(float_val)) == float_val



def test_fxp_consistency():
    # For fixed-point types, test that QFxp(6,4,signed=True) and CFxp(6,4,signed=True) yield the same bit conversion.
    qfxp = QFxp(6, 4, signed=True)
    
    cfxp = CFxp(6, 4, signed=True)
    for float_val in [1.5, -1.5, 0.75, -0.75]:
        # Convert float to fixed-width integer
        q_int = qfxp.to_fixed_width_int(float_val)
        c_int = cfxp.to_fixed_width_int(float_val)
        # Both should be the same integer.
        assert q_int == c_int, f"Fixed-width integer conversion mismatch for {float_val}: {q_int} vs {c_int}"
        # Now compare bit conversion:
        q_bits = qfxp.to_bits(q_int)
        c_bits = cfxp.to_bits(c_int)
        assert q_bits == c_bits, f"to_bits mismatch for fixed-point value {float_val}: {q_bits} vs {c_bits}"
        # Roundtrip conversion:
        q_val = qfxp.from_bits(q_bits)
        c_val = cfxp.from_bits(c_bits)
        assert q_val == c_val, f"Roundtrip conversion mismatch for fixed-point value {float_val}: {q_val} vs {c_val}"

    # Test vectorized conversion for fixed-point types.
    qfxp = QFxp(6, 4, signed=True)
    cfxp = CFxp(6, 4, signed=True)

    # Create an array of fixed-width integer representations
    float_vals = [1.5, -1.5, 0.75, -0.75, 0.0, 1.0]
    q_ints = np.array([qfxp.to_fixed_width_int(x) for x in float_vals])
    c_ints = np.array([cfxp.to_fixed_width_int(x) for x in float_vals])
    # They should match:
    np.testing.assert_array_equal(q_ints, c_ints)
    # Now test roundtrip using to_bits_array and from_bits_array:
    assert_to_and_from_bits_array_consistent(qfxp, q_ints)
    assert_to_and_from_bits_array_consistent(cfxp, c_ints)

def test_qfxp_to_and_from_bits_using_fxp():
    # QFxp: Negative numbers are stored as twos complement
    qfxp_4_3 = QFxp(4, 3, True)
    assert list(qfxp_4_3._fxp_to_bits(0.5)) == [0, 1, 0, 0]
    assert qfxp_4_3._from_bits_to_fxp(qfxp_4_3._fxp_to_bits(0.5)).get_val() == 0.5
    assert list(qfxp_4_3._fxp_to_bits(-0.5)) == [1, 1, 0, 0]
    assert qfxp_4_3._from_bits_to_fxp(qfxp_4_3._fxp_to_bits(-0.5)).get_val() == -0.5
    assert list(qfxp_4_3._fxp_to_bits(0.625)) == [0, 1, 0, 1]
    assert qfxp_4_3._from_bits_to_fxp(qfxp_4_3._fxp_to_bits(+0.625)).get_val() == +0.625
    assert qfxp_4_3._from_bits_to_fxp(qfxp_4_3._fxp_to_bits(-0.625)).get_val() == -0.625
    assert list(qfxp_4_3._fxp_to_bits(-(1 - 0.625))) == [1, 1, 0, 1]
    assert qfxp_4_3._from_bits_to_fxp(qfxp_4_3._fxp_to_bits(0.375)).get_val() == 0.375
    assert qfxp_4_3._from_bits_to_fxp(qfxp_4_3._fxp_to_bits(-0.375)).get_val() == -0.375
    with pytest.raises(ValueError):
        _ = qfxp_4_3._fxp_to_bits(0.1)
    assert list(qfxp_4_3._fxp_to_bits(0.7, require_exact=False)) == [0, 1, 0, 1]
    assert list(qfxp_4_3._fxp_to_bits(0.7, require_exact=False, complement=False)) == [0, 1, 0, 1]
    assert list(qfxp_4_3._fxp_to_bits(-0.7, require_exact=False)) == [1, 0, 1, 1]
    assert list(qfxp_4_3._fxp_to_bits(-0.7, require_exact=False, complement=False)) == [1, 1, 0, 1]

    with pytest.raises(ValueError):
        _ = qfxp_4_3._fxp_to_bits(1.5)

    assert (
        qfxp_4_3._from_bits_to_fxp(qfxp_4_3._fxp_to_bits(1 / 2 + 1 / 4 + 1 / 8))
        == 1 / 2 + 1 / 4 + 1 / 8
    )
    assert (
        qfxp_4_3._from_bits_to_fxp(qfxp_4_3._fxp_to_bits(-1 / 2 - 1 / 4 - 1 / 8))
        == -1 / 2 - 1 / 4 - 1 / 8
    )
    with pytest.raises(ValueError):
        _ = qfxp_4_3._fxp_to_bits(1 / 2 + 1 / 4 + 1 / 8 + 1 / 16)

    for qfxp in [QFxp(4, 3, True), QFxp(3, 3, False), QFxp(7, 3, False), QFxp(7, 3, True)]:
        for x in qfxp._get_classical_domain_fxp():
            assert qfxp._from_bits_to_fxp(qfxp._fxp_to_bits(x)) == x

    assert list(QFxp(7, 3, True)._fxp_to_bits(-4.375)) == [1] + [0, 1, 1] + [1, 0, 1]
    assert list(QFxp(7, 3, True)._fxp_to_bits(+4.625)) == [0] + [1, 0, 0] + [1, 0, 1]


def test_iter_bits():
    assert QUInt(2).to_bits(0) == [0, 0]
    assert QUInt(2).to_bits(1) == [0, 1]
    assert QUInt(2).to_bits(2) == [1, 0]
    assert QUInt(2).to_bits(3) == [1, 1]


def test_iter_bits_twos():
    assert QInt(4).to_bits(0) == [0, 0, 0, 0]
    assert QInt(4).to_bits(1) == [0, 0, 0, 1]
    assert QInt(4).to_bits(-2) == [1, 1, 1, 0]
    assert QInt(4).to_bits(-3) == [1, 1, 0, 1]
    with pytest.raises(ValueError):
        _ = QInt(2).to_bits(100)

@pytest.mark.skip(reason="Too many")
@pytest.mark.parametrize('val', [random.uniform(-1, 1) for _ in range(10)])
@pytest.mark.parametrize('width', [*range(2, 20, 2)])
@pytest.mark.parametrize('signed', [True, False])
def test_fixed_point(val, width, signed):
    if (val < 0) and not signed:
        with pytest.raises(ValueError):
            _ = QFxp(width + int(signed), width, signed=signed)._fxp_to_bits(
                val, require_exact=False, complement=False
            )
    else:
        bits = QFxp(width + int(signed), width, signed=signed)._fxp_to_bits(
            val, require_exact=False, complement=False
        )
        if signed:
            sign, bits = bits[0], bits[1:]
            assert sign == (1 if val < 0 else 0)
        val = abs(val)
        approx_val = math.fsum([b * (1 / 2 ** (1 + i)) for i, b in enumerate(bits)])
        assert math.isclose(val, approx_val, abs_tol=1 / 2**width), (
            f'{val}:{approx_val}:{width}',
            bits,
        )
        with pytest.raises(ValueError):
            _ = QFxp(width, width).to_fixed_width_int(-val)
        bits_from_int = QUInt(width).to_bits(QFxp(width, width).to_fixed_width_int(val))
        assert bits == bits_from_int



def test_any_disallows_numeric_conversions():
    """Check that severity=PROMOTE doesn't allow e.g. QInt <-> QFxp or CInt <-> CFxp.
       We already tested single-bit in other tests. 
    """
    qi4 = QInt(4)
    qf4 = QFxp(4, 2, signed=False)

    assert not check_dtypes_consistent(qi4, qf4, DTypeCheckingSeverity.ANY), f'{qi4} and {qf4} should be consistent with DTypeCheckingSeverity.STRICT'

    # If PROMOTE disallows numeric conversions, we expect false here:
    assert not check_dtypes_consistent(qi4, qf4, DTypeCheckingSeverity.STRICT), f'{qi4} and {qf4} should be consistent with DTypeCheckingSeverity.STRICT'

    ci8 = CInt(8)
    cf8_3 = CFxp(bit_width=8, num_frac=3, signed=True)
    assert not check_dtypes_consistent(ci8, cf8_3, DTypeCheckingSeverity.ANY), f'{ci8} and {cf8_3} should be consistent with DTypeCheckingSeverity.PROMOTE'

@pytest.mark.parametrize(
    "dtype_a, dtype_b, severity, expected",
    [
        # --------- C_PromoLevel.STRICT  ---------
        (CInt(1),  CBit(),  C_PromoLevel.STRICT,  True), 
        (CInt(16),  CInt(16),   C_PromoLevel.STRICT, True),   # identical OK
        (CInt(16),  CFloat(16), C_PromoLevel.STRICT, False),  # int ≠ float
        (CInt(16),  CUInt(16),  C_PromoLevel.STRICT, False),  # signed ≠ unsigned

        # --------- C_PromoLevel.PROMOTE (global ANY) -----------
        (CInt(1),  CBit(),  C_PromoLevel.PROMOTE,  True), 
        (CInt(16),  CFloat(16), C_PromoLevel.PROMOTE,    True),   # int → float widen
        (CInt(16),  CUInt(16),  C_PromoLevel.PROMOTE,    False),  # int ↔ uint still blocked
        (CFloat(32), CFloat(16), C_PromoLevel.PROMOTE,   False),  # float widths differ

        # --------- C_PromoLevel.CAST (global LOOSE) ------------
        (CInt(1),  CBit(),  C_PromoLevel.CAST,  True), 
        (CInt(32),  CUInt(32),  C_PromoLevel.CAST,  True),   # signed ↔ unsigned bit-cast
        (CUInt(8),  CFxp(8, 0), C_PromoLevel.CAST,  True),   # uint ↔ fixed-pt (frac=0)
        (CFloat(32), CInt(32),  C_PromoLevel.CAST,  True),   # float ↔ int re-interpret
        (CFloat(16), CFxp(16, 5), C_PromoLevel.CAST, False), # float ≠ fxp even with same width
    ],
)
def test_classical_promo_matrix(dtype_a, dtype_b, severity, expected):
    """
    Matrix verifying behaviour of STRICT / PROMOTE / CAST for classical types.

    Ladder mapping (as implemented in dtypes.py):
        STRICT → C_PromoLevel.STRICT
        ANY    → C_PromoLevel.PROMOTE
        LOOSE  → C_PromoLevel.CAST
    """
    assert check_dtypes_consistent(dtype_a, dtype_b, classical_level=severity) is expected


@pytest.mark.parametrize(
    "dtype_a, dtype_b, clevel, expected",
    [
        # --------- C_PromoLevel.STRICT  ---------
        (CInt(16),  CInt(16),   C_PromoLevel.STRICT,  True),
        (CInt(16),  CFloat(16), C_PromoLevel.STRICT,  False),
        (CInt(16),  CUInt(16),  C_PromoLevel.STRICT,  False),

        # --------- C_PromoLevel.PROMOTE  ---------
        (CInt(16),  CFloat(16), C_PromoLevel.PROMOTE, True),   # int → float widen
        (CInt(16),  CUInt(16),  C_PromoLevel.PROMOTE, False),  # int ↔ uint blocked
        (CFloat(32), CFloat(16), C_PromoLevel.PROMOTE, False), # float widths differ

        # --------- C_PromoLevel.CAST  -------------
        (CInt(32),  CUInt(32),  C_PromoLevel.CAST,    True),   # bit-cast signed ↔ unsigned
        (CUInt(8),  CFxp(8, 0), C_PromoLevel.CAST,    True),   # uint ↔ fxp(frac=0)
        (CFloat(32), CInt(32),  C_PromoLevel.CAST,    True),   # float ↔ int reinterpret
        (CFloat(16), CFxp(16, 5), C_PromoLevel.CAST,  False),  # float ≠ fxp
    ],
)
def test_classical_promo_matrix_explicit(dtype_a, dtype_b, clevel, expected):
    """
    Same matrix as before, but drives the checker via explicit `classical_level=`.
    No quantum_level override is provided (default mapping applies).
    """
    assert (
        check_dtypes_consistent(
            dtype_a,
            dtype_b,
            severity=DTypeCheckingSeverity.LOOSE,  # global level can be anything; we override with clevel
            classical_level=clevel,
        )
        is expected
    )




def test_classical_dtypes_with_dimension():
    matrix = MatrixType((3, 3))
    assert matrix == MatrixType(3, 3)

    # ndim = NDimDataType(shape=(3, 4), element_type=float)
    # assert repr(ndim) == "NDim(shape=(3, 4), dtype=float)"
    matrix = MatrixType(2, 3, element_type=CFloat(32))

    assert repr(matrix) == "Matrix((2, 3), dtype=CFloat(32))"
    matrix = MatrixType(2, 3, element_type=float)
    assert repr(matrix) == "Matrix((2, 3), dtype=CFloat(64))"

    tensor = TensorType(shape=(3, 4, 5), element_type=int)
    assert tensor.shape == (3, 4, 5)
    assert len(tensor) == tensor.rank == 3
    assert repr(tensor) == "TensorType((3, 4, 5), dtype=CFloat(64))", f'{repr(tensor)}'

    tensor = TensorType(shape=(3, 4, 5), element_type=np.float64)
    assert repr(tensor) == "TensorType((3, 4, 5), dtype=CFloat(64))", f'{str(tensor)}'


def test_MatrixType_init_fail():
    with pytest.raises(ValueError):
        _ = MatrixType((1,)) 

def test_cstring_basic():
    dt = CString(max_length=3)
    # exact length
    s = "Hi!"
    bits = dt.to_bits(s)
    assert len(bits) == 3*8
    assert dt.from_bits(bits) == "Hi!"
    # shorter string → padded with NUL
    bits2 = dt.to_bits("A")
    assert dt.from_bits(bits2) == "A"
    # empty → all NUL
    bits3 = dt.to_bits("")
    assert dt.from_bits(bits3) == ""
    # too long → error
    with pytest.raises(ValueError):
        dt.to_bits("toolong")

def test_cstruct_fields():
    """Field names and per-field dtypes must match for all severities."""
    structA = CStruct(fields={"x": CInt(4), "y": CUInt(4)})
    structB = CStruct(fields={"x": CInt(4), "y": CUInt(4)})

    structMismatch = CStruct(fields={"x": CInt(4), "y": CUInt(5)})   # width diff
    structExtra    = CStruct(fields={"x": CInt(4), "z": CBit()})      # key diff
    structQuantum  = CStruct(fields={"x": CInt(4), "y": QBit()})      # cross-domain

    for sev in DTypeCheckingSeverity:
        # identical ⇒ always consistent
        assert check_dtypes_consistent(structA, structB, sev)

        # width mismatch, extra key, cross-domain ⇒ never consistent
        assert not check_dtypes_consistent(structA, structMismatch, sev)
        assert not check_dtypes_consistent(structA, structExtra,    sev)
        assert not check_dtypes_consistent(structA, structQuantum,  sev)

    
def test_cstruct_fields2():
    """Check that CStruct fields must match to be consistent. 
       We can also test some field-level differences.
    """
    structA = CStruct(fields={
        "x": CInt(4),
        "y": CUInt(4)
    })
    structB = CStruct(fields={
        "x": CInt(4),
        "y": CUInt(4)
    })
    structMismatch = CStruct(fields={
        "x": CInt(4),
        "y": CUInt(5)
    })
    structExtra = CStruct(fields={
        "x": CInt(4),
        "z": CBit()
    })

    # Identical => consistent in all severities
    for sev in DTypeCheckingSeverity:
        assert check_dtypes_consistent(structA, structB, sev) is True
        # Mismatch => false
        assert check_dtypes_consistent(structA, structMismatch, sev) is False
        # Different field keys => false
        assert check_dtypes_consistent(structA, structExtra, sev) is False

def test_ndim_dtype_consistency():
    assert check_dtypes_consistent(TensorType, TensorType)
    assert check_dtypes_consistent(MatrixType, MatrixType)
    assert check_dtypes_consistent(TensorType((1,)), TensorType((1,)))
    
    assert check_dtypes_consistent(MatrixType((2,2)), TensorType((2,2)))
    # reverse
    assert check_dtypes_consistent(TensorType((2,2)), MatrixType((2,2)))

    

    assert check_dtypes_consistent(TensorType((Dyn,1)), TensorType((5,1)))

    n = sympy.symbols('n', positive=True, integer=True)
    m = sympy.symbols('m', positive=True, integer=True)
    assert check_dtypes_consistent(TensorType((n,m)), TensorType((Dyn, Dyn)))



# def test_instance_vs_class():
#     tensor_instance = TensorType((2, 3))
#     assert check_dtypes_consistent(TensorType, tensor_instance)

#     matrix_instance = MatrixType()
#     assert check_dtypes_consistent(MatrixType, matrix_instance)


# def test_list_of_types():
#     assert check_dtypes_consistent([MatrixType, TensorType], TensorType)
#     assert check_dtypes_consistent([MatrixType, TensorType], MatrixType)
#     assert not check_dtypes_consistent([MatrixType, TensorType], int)

def test_is_consistent_tensortype():
    t1 = TensorType((2, 4))
    t2 = TensorType((3, 4))
    assert not check_dtypes_consistent(t1, t2)

    t3 = TensorType((2, 3), element_type=float)
    t4 = TensorType((3, 2), element_type=float)
    assert not check_dtypes_consistent(t3, t4)

    assert not check_dtypes_consistent(t1, CBit())

    tensor1 = TensorType((2, 4))
    tensor2 = TensorType((2, 4))
    assert check_dtypes_consistent(tensor1, tensor2)

    t1 = TensorType((2, 3), element_type=float)
    t2 = MatrixType(2, 3, element_type=float)
    assert check_dtypes_consistent(t1, t2)


def test_tensortype_properties_and_bit_roundtrip():
    # a 2×2 tensor of bits
    shape = (2, 2)
    dt = TensorType(shape=shape, element_type=CBit())

    # data_width per element = 1
    assert dt.data_width == 1
    # nbytes = (1 bit // 8) * 4 elements = 0 bytes (integer division)
    assert dt.element_size() == 0
    assert dt.nelement() == 4
    assert dt.total_bits == 4
    assert dt.rank == 2

    # build a nested Python list
    value = [[0, 1], [1, 0]]
    # flatten → bits
    bits = dt.to_bits(value)
    assert bits == [0,1,1,0]
    # reconstruct with from_bits
    nested = dt.from_bits(bits)
    assert nested == [[0,1],[1,0]]

def test_tensortype_assert_valid_classical_val():
    dt = TensorType(shape=(2,3), element_type=CBit())
    # must be a NumPy array
    with pytest.raises(TypeError):
        dt.assert_valid_classical_val([[0,1,0],[1,0,1]])
    # wrong shape
    arr = np.zeros((2,2), dtype=bool)
    with pytest.raises(ValueError):
        dt.assert_valid_classical_val(arr)
    # wrong dtype (expect Python int/bit via CBit)
    arr2 = np.zeros((2,3), dtype=int)
    with pytest.raises(TypeError):
        dt.assert_valid_classical_val(arr2)

def test_tensor_multiply_broadcast():
    t1 = TensorType(shape=(2,1), element_type=CBit())
    t2 = TensorType(shape=(1,3), element_type=CBit())
    t3 = t1.multiply(t2)
    # broadcast to (2,3)
    assert t3.shape == (2,3)
    assert isinstance(t3, TensorType)

def test_tensor_broadcast_mult():
    tensor1 = TensorType(shape=(2, 3, 4), element_type=float)
    tensor2 = TensorType(shape=(1, 3, 1), element_type=float)
    result = tensor1.multiply(tensor2)
    assert result.shape == (2, 3, 4)

def _element_size(dtype):
    """
    Returns the element size for a dtype, in bytes
    """
    if not isinstance(dtype, torch.dtype):
        raise RuntimeError(f"expected torch.dtype, but got {type(dtype)}")

    if dtype.is_complex:
        print(torch.finfo(dtype).bits)
        return torch.finfo(dtype).bits >> 2
    elif dtype.is_floating_point:
        print(torch.finfo(dtype).bits)
        return torch.finfo(dtype).bits >> 3
    elif dtype == torch.bool:
        return 1
    else:
        print(torch.iinfo(dtype).bits)
        return torch.iinfo(dtype).bits >> 3

def test_tensortype_memory_calculations():
    torch_tensor_int8 = torch.ones((2, 2, 2), dtype=torch.int8)
    # print("[int8]: ", f"_element_size({str(torch_tensor_int8.dtype)}): {_element_size(torch_tensor_int8.dtype)}",
    #       f"tensor.element_size(): {torch_tensor_int8.element_size()}",
    #       f"tensor.nbytes: {torch_tensor_int8.nbytes}, total_bits={torch_total_bits(torch_tensor_int8)} tensor.untyped_storage().element_size: {torch_tensor_int8.untyped_storage().element_size()}")
    tensor = TensorType(shape=torch_tensor_int8.shape, element_type=torch.int8)
    assert tensor.data_width == 8, f'tensor should have data_width = 8. Got: {tensor.data_width}'
    assert tensor.nelement() == 8, f'tensor should have nelements = {torch_tensor_int8.nelement()}. Got: {tensor.nelement()}'
    assert tensor.element_size() == 1, f'tensor should have element_size = {torch_tensor_int8.element_size()}. Got: {tensor.element_size()}'
    assert tensor.nbytes == torch_tensor_int8.nbytes, f'tensor should have nbytes = {torch_tensor_int8.nbytes}. Got: {tensor.nbytes}'
    assert tensor.total_bits == torch_total_bits(torch_tensor_int8), f'tensor should have total_bits = {torch_total_bits(torch_tensor_int8)}. Got: {tensor.total_bits}'

    torch_tensor_float32 = torch.ones((2, 2, 2), dtype=torch.float32)
    # print("[float32]: ", f"_element_size({str(torch_tensor_float32.dtype)}): {_element_size(torch_tensor_float32.dtype)}",
    #       f"tensor.element_size(): {torch_tensor_float32.element_size()}",
    #       f"tensor.nbytes: {torch_tensor_float32.nbytes}, total_bits={torch_total_bits(torch_tensor_float32)} tensor.untyped_storage().element_size: {torch_tensor_float32.untyped_storage().element_size()}")
    tensor = TensorType(shape=torch_tensor_float32.shape, element_type=torch.float32)
    assert tensor.data_width == 32, f'tensor should have data_width = 32. Got: {tensor.data_width}'
    assert tensor.nelement() == torch_tensor_float32.nelement(), f'tensor should have nelements = {torch_tensor_float32.nelement()}.Got: {tensor.nelement()}'
    assert tensor.element_size() == torch_tensor_float32.element_size(), f'tensor should have element_size = {torch_tensor_float32.element_size()}. Got: {tensor.element_size()}'
    assert tensor.nbytes == torch_tensor_float32.nbytes, f'tensor should have nbytes = {torch_tensor_float32.nbytes}. Got: {tensor.nbytes}'
    assert tensor.total_bits == torch_total_bits(torch_tensor_float32), f'tensor should have total_bits = {torch_total_bits(torch_tensor_float32)}. Got: {tensor.total_bits}'

    torch_tensor_int8 = torch.ones((2,3), dtype=torch.int8)
    tensor_c  = TensorType((2,3), element_type=CInt(8))
    print("[int8]: ", f"_element_size({str(torch_tensor_int8.dtype)}): {_element_size(torch_tensor_int8.dtype)}",
          f"tensor.element_size(): {torch_tensor_int8.element_size()}, got: {tensor_c.total_bits}",
          f"tensor.nbytes: {torch_tensor_int8.nbytes}, total_bits={torch_total_bits(torch_tensor_int8)} tensor.untyped_storage().element_size: {torch_tensor_int8.untyped_storage().element_size()}")
    
    assert tensor_c.total_bits == torch_total_bits(torch_tensor_int8), f'tensor should have total_bits = {torch_total_bits(torch_tensor_int8)}. Got: {tensor_c.total_bits}'


@pytest.mark.parametrize("dtype,expected_element_size,expected_bits_per_element,expected_total_bits", [
    
    (torch.int8, 1, 8, 64),# torch.int8: each element is 1 byte → 1*8 = 8 bits per element, total = 8*8 = 64 bits.
    # (torch.quint8,  1, 8,  64),     # 1 byte -> 8 bits per element.
    # (torch.qint8,   1, 8,  64),     # 1 byte -> 8 bits per element.
    (torch.float32, 4, 32, 256),    # torch.float32: each element is 4 bytes → 4*8 = 32 bits per element, total = 8*32 = 256 bits.
    (torch.int32,   4, 32, 256),    # 4 bytes -> 32 bits per element.
    # (torch.qint32,  4, 32, 256),    # 4 bytes -> 32 bits per element.
    (torch.uint32, 4, 32, 256),# torch.uint32: each element is 4 bytes → 4*8 = 32 bits per element, total = 8*32 = 256 bits.
    (torch.float64, 8, 64, 512),    # 8 bytes -> 64 bits per element, 8*64 = 512 bits total.
    (torch.int64,   8, 64, 512),    # 8 bytes -> 64 bits per element.
    
    
])
def test_torch_dtype_resource_values(dtype, expected_element_size, expected_bits_per_element, expected_total_bits):
    shape = (2, 2, 2)
    tensor = torch.ones(shape, dtype=dtype)
    # Built-in element size in bytes
    es = tensor.element_size()
    # Total number of elements
    ne = tensor.nelement()
    # Total number of bytes (should equal ne * es)
    nbytes = tensor.nbytes
    # Total bits using built-in nbytes
    total_bits = torch_total_bits(tensor)
    
    assert es == expected_element_size, (
        f"For {dtype}, expected element_size {expected_element_size} bytes but got {es} bytes."
    )
    assert nbytes == ne * es, (
        f"For {dtype}, expected nbytes {ne*es} but got {nbytes}."
    )
    # Bits per element is bytes per element times 8.
    assert es * 8 == expected_bits_per_element, (
        f"For {dtype}, expected bits per element {expected_bits_per_element} but got {es*8}."
    )
    assert total_bits == expected_total_bits, (
        f"For {dtype}, expected total bits {expected_total_bits} but got {total_bits}."
    )
    # The raw storage is in bytes; its element size should be 1.
    storage = tensor.untyped_storage()
    assert storage.element_size() == 1, (
        f"For {dtype}, expected untyped_storage.element_size() 1 but got {storage.element_size()}."
    )
    
def torch_total_bits(tensor: torch.Tensor) -> int:
    """
    Computes the total number of bits in a tensor.
    PyTorch’s tensor.nbytes gives the total number of bytes.
    Multiply by 8 to obtain total bits.
    """
    return tensor.nbytes * 8

def torch_bit_length(dtype: torch.dtype) -> int:
    """
    For a given torch.dtype, compute the bit-length per element.
    This is defined as tensor.element_size() * 8.
    """
    # Create a 1-element tensor of the given dtype.
    t = torch.ones((1,), dtype=dtype)
    return t.element_size() * 8
# --- Tests for Qualtran Register total_bits via QDType mapping ---
@pytest.mark.parametrize("dtype, qdtype, expected_bit_length", [
    (torch.float32, QFxp(32, 16, signed=True), 32),
    (torch.int32,   QInt(32),                   32),

])
def test_torch_vs_quantum_bit_length(dtype, qdtype, expected_bit_length):
    # Compute torch bit_length.
    torch_bl = torch_bit_length(dtype)
    
    # For our Qualtran type, we assume num_qubits equals the bit_length.
    qualtran_bl = qdtype.num_qubits
    assert qualtran_bl == torch_bl, f"For Qualtran dtype {qdtype}, expected num_qubits {torch_bl} but got {qualtran_bl}."
