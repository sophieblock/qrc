"""
`DataType` Initialization Tests
   - Tests that validate the proper construction of in‑house data types (DataType,
     `CFixed`, `TensorType`, `CBit`, `QBit`,...).

"""
import pytest
import numpy as np

import torch

from workflow.simulation.refactor.schema import (
    RegisterSpec, Flow, Signature
)
from workflow.simulation.refactor.dtypes import *


from workflow.simulation.refactor.unification_tools import (
    type_matches, is_consistent_data_type,canonicalize_dtype
)
from workflow.assert_checks import assert_to_and_from_bits_array_consistent

def make_symbolic_cint(sym_name='n') -> CInt:
    n = sympy.Symbol(sym_name, positive=True, integer=True)
    return CInt(bit_width=n)

def make_dyn_cint() -> CInt:
    # If you treat Dyn as a bit_width
    return CInt(bit_width=Dyn)

# Similarly for quantum:
def make_symbolic_qint(sym_name='m') -> QInt:
    m = sympy.Symbol(sym_name, positive=True, integer=True)
    return QInt(num_qubits=m)

def make_dyn_qint() -> QInt:
    return QInt(num_qubits=Dyn)



def test_qint():
    qint_8 = QInt(8)
    assert qint_8.num_qubits == 8
    assert str(qint_8) == 'QInt(8)'
    n = sympy.symbols('x')
    qint_8 = QInt(n)
    assert qint_8.num_qubits == n
    assert str(qint_8) == 'QInt(x)'
    assert is_symbolic(QInt(sympy.Symbol('x')))

def test_quint():
    """Checks that QUInt(8) displays as expected and can handle symbolic widths."""
    qint_8 = QUInt(8)
    assert str(qint_8) == 'QUInt(8)'
    assert qint_8.num_qubits == 8

    # Should not raise
    QUInt(1)

    n = sympy.symbols('x', positive=True, integer=True)
    quint_sym = QUInt(n)
    assert quint_sym.num_qubits == n
    # is_symbolic(QUInt(sympy.Symbol('x'))) => True
    assert is_symbolic(quint_sym) is True

    # Now we test QInt(n) vs. QUInt(n) => should pass under LOOSE severity
    qint_sym = QInt(n)
    # By default check_dtypes_consistent uses severity=LOOSE.
    assert check_dtypes_consistent(qint_sym, quint_sym) is True, \
        "QInt(n) <-> QUInt(n) must pass under LOOSE if they have the same symbolic num_qubits."
def test_quint_dyn():
    """Similar to above, but uses Dyn to ensure it passes under LOOSE."""
    qdyn = QUInt(Dyn)
    qdyn2 = QInt(Dyn)
    # LOOSE => accept
    assert check_dtypes_consistent(qdyn, qdyn2) is True

    # ANY => numeric conversions disallowed, but we want to be flexible with Dyn?
    # If you'd like ANY to fail, we can confirm:
    assert check_dtypes_consistent(qdyn, qdyn2, DTypeCheckingSeverity.ANY) is False

    # STRICT => must be identical => fail
    assert check_dtypes_consistent(qdyn, qdyn2, DTypeCheckingSeverity.STRICT) is False

def test_quint_strict():
    """Under STRICT, QInt(4) vs QUInt(4) should fail unless they're literally the same type."""
    qi4 = QInt(4)
    qu4 = QUInt(4)
    assert check_dtypes_consistent(qi4, qu4, DTypeCheckingSeverity.STRICT) is False

def test_quint_any():
    """Under ANY, we want no numeric conversions (like QInt <-> QUInt),
    unless single-bit or QAny is in play."""
    qi5 = QInt(5)
    qu5 = QUInt(5)
    assert check_dtypes_consistent(qi5, qu5, DTypeCheckingSeverity.ANY) is False

def test_quint_loose_numeric():
    """Under LOOSE, numeric conversions are accepted for matching widths or symbolic widths."""
    qi5 = QInt(5)
    qu5 = QUInt(5)
    # LOOSE => pass
    assert check_dtypes_consistent(qi5, qu5) is True

def test_single_bit_any():
    """ANY severity still allows single-bit cross-compat (QBit vs CBit) or QBit vs QBit with data_width=1."""
    from workflow.simulation.refactor.dtypes import QBit, CBit
    qb = QBit()
    cb = CBit()
    # check that data_width=1 => consistent
    assert qb.data_width == 1
    assert cb.data_width == 1
    # ANY => single-bit cross => True
    assert check_dtypes_consistent(qb, cb, DTypeCheckingSeverity.ANY) is True

def test_qfxp():
    qfp_16 = QFxp(16, 15)
    assert str(qfp_16) == 'QFxp(16, 15)'
    assert qfp_16.num_qubits == 16
    assert qfp_16.num_int == 1
    assert qfp_16.fxp_dtype_template().dtype == 'fxp-u16/15'

    qfp_16 = QFxp(16, 15, signed=True)
    assert str(qfp_16) == 'QFxp(16, 15, True)'
    assert qfp_16.num_qubits == 16
    assert qfp_16.num_int == 1
    assert qfp_16.fxp_dtype_template().dtype == 'fxp-s16/15'

    with pytest.raises(ValueError, match="num_qubits must be > 1."):
        QFxp(1, 1, signed=True)
    QFxp(1, 1, signed=False)
    with pytest.raises(ValueError, match="num_frac must be less than.*"):
        QFxp(4, 4, signed=True)
    with pytest.raises(ValueError, match="bit_width must be >= .*"):
        QFxp(4, 5)
    b = sympy.symbols('b')
    f = sympy.symbols('f')
    qfp = QFxp(b, f)
    assert qfp.num_qubits == b
    assert qfp.num_int == b - f
    qfp = QFxp(b, f, True)
    assert qfp.num_qubits == b
    assert qfp.num_int == b - f
    assert is_symbolic(QFxp(*sympy.symbols('x y')))


@pytest.mark.parametrize("severity", [
    DTypeCheckingSeverity.STRICT,
    DTypeCheckingSeverity.ANY,
    DTypeCheckingSeverity.LOOSE
])
def test_single_bit_types(severity):
    """Check how single-bit quantum vs classical types behave in each severity setting."""
    qbit = QBit()
    cbit = CBit()

    # For STRICT, typically we only allow QBit == QBit or CBit == CBit.
    # For ANY or LOOSE, we might allow QBit vs CBit to pass or fail based on our code.
    consistent_q_q = check_dtypes_consistent(qbit, qbit, severity)
    consistent_c_c = check_dtypes_consistent(cbit, cbit, severity)
    consistent_q_c = check_dtypes_consistent(qbit, cbit, severity)

    # qbit vs. qbit => always True, because dtype_a == dtype_b
    assert consistent_q_q is True, f'{severity}: dtype_a: {qbit} should be consistent to dtype_b: {qbit}'
    # cbit vs. cbit => always True
    assert consistent_c_c is True, f'{severity}: dtype_a: {cbit},  should be consistent to dtype_b:: {cbit}'

    if severity == DTypeCheckingSeverity.STRICT:
        # We typically do not allow QBit <-> CBit
        assert consistent_q_c is False, f'dtype_a: {qbit}, dtype_b: {cbit}'
    else:
        # ANY or LOOSE might allow single-bit cross-compatibility
        # Adjust if you want different logic for ANY vs. LOOSE
        assert consistent_q_c is True, f'dtype_a: {qbit}, dtype_b: {cbit}'

def test_strict_requires_exact_match():
    """Under STRICT severity, check that QInt(4) != QInt(5), for example."""
    a = QInt(4)
    b = QInt(5)
    c = QInt(4)

    # They are not the same object, but a and c have same width => they'd pass equality check
    # if .__eq__ checks all fields. However, if your code uses data_width in equality,
    # QInt(4) == QInt(4) might be True. Let's assume it is.
    # We'll forcibly test check_dtypes_consistent.
    assert check_dtypes_consistent(a, b, DTypeCheckingSeverity.STRICT) is False
    # a vs c => same attributes => the function might see them as identical => True
    assert check_dtypes_consistent(a, c, DTypeCheckingSeverity.STRICT) is True

    # Another example: CFixed(8,4) vs CFixed(8,4) => exact match => True
    fixA = CFixed(bit_width=8, frac_bits=4, signed=True)
    fixB = CFixed(bit_width=8, frac_bits=4, signed=True)
    fixC = CFixed(bit_width=8, frac_bits=3, signed=True)
    assert check_dtypes_consistent(fixA, fixB, DTypeCheckingSeverity.STRICT) is True
    assert check_dtypes_consistent(fixA, fixC, DTypeCheckingSeverity.STRICT) is False

def test_cuint_cfixed_consistency_loose():
    """Check that we can allow CUInt(8) <-> CFixed(8, 0, signed=False) under LOOSE."""
    cu8 = CUInt(8)
    cf8_0 = CFixed(bit_width=8, frac_bits=0, signed=False)
    cf8_3 = CFixed(bit_width=8, frac_bits=3, signed=False)

    # LOOSE => might allow cuint(8) <-> cfixed(8,0)
    assert check_dtypes_consistent(cu8, cf8_0, DTypeCheckingSeverity.LOOSE) is True
    # but if frac_bits=3 => depends on your rules. Possibly disallowed?
    assert check_dtypes_consistent(cu8, cf8_3, DTypeCheckingSeverity.LOOSE) is False

def test_cstruct_fields():
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

def test_tensor_type_consistency():
    """Check shape matching for TensorType, plus element_type consistency."""
    elemA = CInt(8)
    elemB = CInt(8)
    elemC = CInt(16)
    # Suppose shape is fixed to (3,2)
    tA = TensorType(shape=(3, 2), element_type=elemA)
    tB = TensorType(shape=(3, 2), element_type=elemB)
    tC = TensorType(shape=(3, 2), element_type=elemC)  # element differs
    tD = TensorType(shape=(3, 3), element_type=elemB)  # shape differs

    # Under any severity, if shape & element_type are consistent => True
    for sev in DTypeCheckingSeverity:
        assert check_dtypes_consistent(tA, tB, sev) is True
        assert check_dtypes_consistent(tA, tC, sev) is False
        assert check_dtypes_consistent(tA, tD, sev) is False

def test_symbolic_dyn_cases():
    """Check how code handles symbolic or Dyn-based types."""
    sym_cint = make_symbolic_cint('n')
    sym_qint = make_symbolic_qint('m')

    
    assert str(sym_qint) == 'QInt(m)'
    # assert sym_qint.data_width == 'm', f'sym_qint.data_width: {sym_qint.data_width}'

    # If we compare sym_cint vs. sym_qint => one quantum, one classical => typically false
    # (Unless your code allows deeper logic.)
    assert not check_dtypes_consistent(sym_cint, sym_qint, DTypeCheckingSeverity.LOOSE)


    other_sym_cint = make_symbolic_cint('n')
    assert check_dtypes_consistent(sym_cint, other_sym_cint, DTypeCheckingSeverity.STRICT) is True

    dynA = make_dyn_cint()
    assert check_dtypes_consistent(dynA, other_sym_cint, DTypeCheckingSeverity.LOOSE) is True
    dynB = make_dyn_cint()
    assert check_dtypes_consistent(dynA, dynB, DTypeCheckingSeverity.LOOSE) is True
    assert check_dtypes_consistent(dynA, dynB, DTypeCheckingSeverity.ANY) is True

    # If we do dyn vs. non-dyn => might fail or succeed depending on your data_width equality logic
    cint8 = CInt(8)
    result = check_dtypes_consistent(dynA, cint8, DTypeCheckingSeverity.LOOSE)
    result = check_dtypes_consistent(dynA, cint8, DTypeCheckingSeverity.ANY)

    assert result is False, "We expect Dyn != 8 unless you explicitly handle it."

def test_any_disallows_numeric_conversions():
    """Check that severity=ANY doesn't allow e.g. QInt <-> QFxp or CInt <-> CFixed.
       We already tested single-bit in other tests. 
    """
    qi4 = QInt(4)
    qf4 = QFxp(4, 2, signed=False)
    # If ANY disallows numeric conversions, we expect false here:
    assert not check_dtypes_consistent(qi4, qf4, DTypeCheckingSeverity.ANY)

    ci8 = CInt(8)
    cf8_3 = CFixed(bit_width=8, frac_bits=3, signed=True)
    assert not check_dtypes_consistent(ci8, cf8_3, DTypeCheckingSeverity.ANY)

def test_cfloat_strict():
    """Check that CFloat with different bit sizes is not consistent under STRICT, but may be under LOOSE if you prefer."""
    cf32 = CFloat(32)
    cf64 = CFloat(64)
    # Strict => must match exactly
    assert not check_dtypes_consistent(cf32, cf64, DTypeCheckingSeverity.STRICT)
    # Under LOOSE => we may or may not allow. Suppose we still say no:
    assert not check_dtypes_consistent(cf32, cf64, DTypeCheckingSeverity.LOOSE)

    # But if the code says if both are CFloat we check bits => might still fail if bits differ.
    # So the above is consistent with that logic.

def test_DataType_metrics():
    classical_bits = CInt(10)
    assert classical_bits == CInt(10)
    assert str(classical_bits) == 'CInt(10)'

    # Replaced old bitsize == 16 with data_width == 10
    assert classical_bits.data_width == 10, f'Got: {classical_bits.data_width}'

    qany = QAny(10)
    assert qany == QAny(10)
    assert str(qany) == 'QAny(10)'

    assert qany.data_width == 10, f'Got: {qany.data_width}'
    assert is_consistent_data_type(qany,QAny(10)) 

def test_single_qubit_consistency():
    assert str(QBit()) == 'QBit()'
    assert check_dtypes_consistent(QBit(), QBit())
    assert check_dtypes_consistent(QBit(), QInt(1))
    assert check_dtypes_consistent(QInt(1), QInt(Dyn))
    assert check_dtypes_consistent(QInt(1), QBit())
    assert check_dtypes_consistent(QAny(1), QBit())
    assert check_dtypes_consistent(QFxp(1, 1), QBit())


# def test_match():
#     assert is_consistent_data_type(TensorType, TensorType)
#     assert is_consistent_data_type(MatrixType, MatrixType)
#     assert is_consistent_data_type(TensorType((1,)), TensorType((1,)))

#     assert is_consistent_data_type(MatrixType((2,2)), TensorType((2,2)))
#     # reverse
#     assert is_consistent_data_type(TensorType((2,2)), MatrixType((2,2)))
#     assert is_consistent_data_type(MatrixType(), MatrixType((Dyn, Dyn)))

#     assert is_consistent_data_type(TensorType, MatrixType)

#     # dynamic matches
#     assert is_consistent_data_type(CUInt(8), CUInt(Dyn))

#     assert is_consistent_data_type(Dyn, int)
#     assert is_consistent_data_type(int, Dyn)
#     assert is_consistent_data_type(TensorType((Dyn,1)), TensorType((5,1)))
#     assert is_consistent_data_type(Dyn, TensorType)
#     assert is_consistent_data_type(TensorType, Dyn)
#     assert is_consistent_data_type(Dyn, Dyn)

def test_instance_vs_class():
    tensor_instance = TensorType((2, 3))
    assert is_consistent_data_type(TensorType, tensor_instance)

    matrix_instance = MatrixType()
    assert is_consistent_data_type(MatrixType, matrix_instance)


def test_list_of_types():
    assert is_consistent_data_type([MatrixType, TensorType], TensorType)
    assert is_consistent_data_type([MatrixType, TensorType], MatrixType)
    assert not is_consistent_data_type([MatrixType, TensorType], int)

def test_is_consistent_data_type():
    t1 = TensorType((2, 4))
    t2 = TensorType((3, 4))
    assert not is_consistent_data_type(t1, t2)

    t3 = TensorType((2, 3), element_type=float)
    t4 = TensorType((3, 2), element_type=float)
    assert not is_consistent_data_type(t3, t4)

    assert not is_consistent_data_type(t1, CBit())

    tensor1 = TensorType((2, 4))
    tensor2 = TensorType((2, 4))
    assert is_consistent_data_type(tensor1, tensor2)

    t1 = TensorType((2, 3), element_type=float)
    t2 = MatrixType(2, 3, element_type=float)
    assert is_consistent_data_type(t1, t2)

def test_validation_errs_q_types():
    with pytest.raises(ValueError):
        QBit().assert_valid_classical_val(-1)

    with pytest.raises(ValueError):
        QBit().assert_valid_classical_val('|0>')  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        QUInt(3).assert_valid_classical_val(8)


    with pytest.raises(ValueError):
        QInt(4).assert_valid_classical_val(-9)

    with pytest.raises(ValueError):
        QUInt(3).assert_valid_classical_val(-1)

    with pytest.raises(ValueError):
        QUInt(3).assert_valid_classical_val(-1)

def test_register_bits_roundtrip():
    # For a given QType, ensure that converting values to bits (via to_bits_array)
    # and then back (via from_bits_array) recovers the original classical values.
    # Here we test for a couple of types.
    for qdtype in [QBit(), QInt(4), QUInt(4)]:
       
        # Get the classical domain from the QDType.
        domain = list(qdtype.get_classical_domain())
        # Convert the domain to an array (if domain is small, e.g. QBit or QUInt(4))
        if len(domain) > 0 and len(domain) <= 16:
            arr = np.array(domain)
            bits_arr = qdtype.to_bits_array(arr)
            recovered = qdtype.from_bits_array(bits_arr)
            # The roundtrip should recover the original values.
            np.testing.assert_array_equal(recovered, arr)

def test_qany_to_and_from_bits():
    assert list(QAny(4).to_bits(10)) == [1, 0, 1, 0]
    assert_to_and_from_bits_array_consistent(QAny(4), range(16))

def test_cuint_and_cbit():
    # Instantiate CBit and CUInt with bit_width=1
    cbit = CBit()
    cuint1 = CUInt(bit_width=1)

    # Test for both valid values: 0 and 1
    for value in (0, 1):
        # Get the bit representation from CBit and CUInt
        bits_from_cbit = cbit.to_bits(value)
        bits_from_cuint = cuint1.to_bits(value)

        # Assert that both produce the same result
        assert bits_from_cbit == bits_from_cuint, f"Mismatch in to_bits for value {value}"
        # Assert that the outputs are lists of int
        assert isinstance(bits_from_cbit, list)
        assert all(isinstance(b, int) for b in bits_from_cbit)

        # Check that from_bits recovers the original value for both types
        assert cbit.from_bits(bits_from_cbit) == value, f"CBit.from_bits failed for value {value}"
        assert cuint1.from_bits(bits_from_cuint) == value, f"CUInt.from_bits failed for value {value}"

def test_quint_1():
    # Instantiate QUInt with num_qubits=1
    quint1 = QUInt(num_qubits=1)

    # Test for both valid values: 0 and 1
    for value in (0, 1):
        bits = quint1.to_bits(value)
        # Assert that bits is a list of int
        assert isinstance(bits, list)
        assert all(isinstance(b, int) for b in bits)
        # Assert that from_bits correctly recovers the value
        reconstructed = quint1.from_bits(bits)
        assert reconstructed == value, f"QUInt.from_bits failed for value {value}"

def test_int_to_from_bits_consistency():
    # For QInt(4) vs CInt(4): check to_bits and from_bits roundtrip for a range of values.
    qint = QInt(4)
    cint = CInt(4)

    # test domain are the same
    q_domain = list(qint.get_classical_domain())
    c_domain = list(cint.get_classical_domain())
    assert q_domain == c_domain, f"Quantum domain {q_domain} differs from classical domain {c_domain}"

    for x in range(-8, 8):
        q_bits = qint.to_bits(x)
        c_bits = cint.to_bits(x)
        assert q_bits == c_bits, f"to_bits mismatch for value {x}: {q_bits} vs {c_bits}"
        q_val = qint.from_bits(q_bits)
        c_val = cint.from_bits(c_bits)
        assert q_val == c_val == x, f"Roundtrip conversion failed for value {x}: got {q_val} and {c_val}"

def test_uint_to_from_bits_consistency():
    # For QUInt(4) vs CUInt(4): test roundtrip conversion for all values in domain.
    quint = QUInt(4)
    cuint = CUInt(4)

    # test domain are the same
    q_domain = list(quint.get_classical_domain())
    c_domain = list(cuint.get_classical_domain())
    assert q_domain == c_domain, f"Quantum unsigned domain {q_domain} differs from classical unsigned domain {c_domain}"


    for x in range(0, 16):
        q_bits = quint.to_bits(x)
        c_bits = cuint.to_bits(x)
        assert q_bits == c_bits, f"to_bits mismatch for value {x}. Quantum {quint} to_bits: {q_bits} vs classical {cuint} to_bits: {c_bits}"
        # print(f"to_bits match for value {x}: {quint} to_bits: {q_bits} vs {cuint} to_bits: {c_bits}")
        q_val = quint.from_bits(q_bits)
        c_val = cuint.from_bits(c_bits)
        assert q_val == c_val == x, f"Roundtrip conversion failed for {x}. Quantum {quint} from_bits: {q_val} vs Classical {cuint} from_bits: {c_val}"


def test_fxp_consistency():
    # For fixed-point types, test that QFxp(6,4,signed=True) and CFxp(6,4,signed=True) yield the same bit conversion.
    qfxp = QFxp(6, 4, signed=True)
    
    cfxp = CFixed(6, 4, signed=True)
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
    cfxp = CFixed(6, 4, signed=True)
    # Create an array of fixed-width integer representations
    # For a few representative float values:
    float_vals = [1.5, -1.5, 0.75, -0.75, 0.0, 1.0]
    q_ints = np.array([qfxp.to_fixed_width_int(x) for x in float_vals])
    c_ints = np.array([cfxp.to_fixed_width_int(x) for x in float_vals])
    # They should match:
    np.testing.assert_array_equal(q_ints, c_ints)
    # Now test roundtrip using to_bits_array and from_bits_array:
    assert_to_and_from_bits_array_consistent(qfxp, q_ints)
    assert_to_and_from_bits_array_consistent(cfxp, c_ints)


def assert_bits_roundtrip(dtype, values: np.ndarray):
    bits_arr = dtype.to_bits_array(values)
    values_rt = dtype.from_bits_array(bits_arr)
    np.testing.assert_equal(values_rt, values)

# --- Multidimensional simulation for dtypes ---
@pytest.mark.parametrize("q_dtype, c_dtype", [
    (QBit(), CBit()),
    (QInt(5), CInt(5)),
    (QUInt(5), CUInt(5)),
    (QFxp(5, 3, signed=True), CFixed(5, 3, signed=True)),
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

#
def test_symbolic_instantiation():
    # Create symbolic classical types.
    n = sympy.symbols('n', positive=True, integer=True)
    c_dtype_sym = CInt(n)
    # Check that num_elements returns the symbolic n and that is_symbolic returns True.
    assert c_dtype_sym.data_width == n
    assert c_dtype_sym.is_symbolic() is True

    # Similarly for quantum type.
    q_dtype_sym = QInt(n)
    assert q_dtype_sym.data_width == n
    assert q_dtype_sym.is_symbolic() is True

# --- Tests for torch.dtype resource calculations ---

def element_size(dtype: torch.dtype) -> int:
    """
    Mimics the _element_size function from torch's code.
    For floating point types, returns: (torch.finfo(dtype).bits) >> 3.
    For integer (and quantized) types, returns: (torch.iinfo(dtype).bits) >> 3.
    """
    if dtype.is_floating_point:
        # For float, use finfo.bits divided by 8
        return torch.finfo(dtype).bits >> 3
    else:
        # For integers (or quantized types), use iinfo.bits divided by 8.
        return torch.iinfo(dtype).bits >> 3
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
    # (torch.quint8,  QUInt(8),                   8),
    # (torch.bits8,   QInt(8),                    8),
    # (torch.bits2x4, QInt(8),                    8),
    # (torch.bits4x2, QInt(8),                    8),
    # (torch.bits1x8, QInt(8),                    8),
    # (torch.bits16,  QInt(16),                  16),
])
def test_torch_vs_qualtran_bit_length(dtype, qdtype, expected_bit_length):
    # Compute torch bit_length.
    torch_bl = torch_bit_length(dtype)
    assert torch_bl == expected_bit_length, f"For torch dtype {dtype}, expected bit_length {expected_bit_length} but got {torch_bl}."
    
    # For our Qualtran type, we assume num_qubits equals the bit_length.
    qualtran_bl = qdtype.num_qubits
    assert qualtran_bl == expected_bit_length, f"For Qualtran dtype {qdtype}, expected num_qubits {expected_bit_length} but got {qualtran_bl}."
