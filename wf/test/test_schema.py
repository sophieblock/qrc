import pytest
import numpy as np
import re 
import math
import json
import sympy as sp
import torch
import sympy
from attrs import define
sep_str = 100 * '#'

import re
# from qrew.simulation.register import RegisterSpec, Flow
from qrew.simulation.schema import RegisterSpec, Signature, Flow
from qrew.simulation.data_types import *
from qrew.simulation.data import Data,promote_element_types
from qrew.simulation.unification_tools import (
    type_matches, is_consistent_data_type,canonicalize_dtype
)
from qrew.simulation.process import Process,ClassicalProcess
from qrew.assert_checks import assert_registers_match_parent, assert_multiline_equal

from qrew.visualization_tools import ModuleDrawer, _assign_ids_to_nodes_and_edges,save_mod
from qrew.simulation.builder import ProcessBuilder,ProcessInstance, CompositeMod,PortInT,PortT,Port, LeftDangle,RightDangle,Split,Join
from qrew.simulation.Process_Library.for_testing import Atom,Atom_n,TwoBitOp,SwapTwoBit, AtomChain,NAtomParallel,SplitJoin,TestParallelQCombo
figure_path = 'test_outputs/test_schema/'
# ----------------------------------------------------------------
# Compile-level tests 
# ----------------------------------------------------------------

qbit_reg  = RegisterSpec(name="qb",  dtype=QBit(), flow=Flow.THRU)
cbit_reg  = RegisterSpec(name="cb",  dtype=CBit(), flow=Flow.THRU)
tensor_c  = RegisterSpec(name="tc",  dtype=TensorType((2,3), element_type=CInt(8)), flow=Flow.THRU)



# ----------------------------------------------------------------
# RegisterSpec
# ----------------------------------------------------------------

@pytest.mark.parametrize("reg, exp", [
    (qbit_reg, "Q"),
    (cbit_reg, "C"),
    (tensor_c, "C"),
])
def test_domain_tag(reg, exp):
    assert reg.domain == exp


def test_total_bits():
    assert qbit_reg.total_bits() == 1
    # assert tensor_q.total_bits()  == 4 * 1, f'Got: {tensor_q.total_bits()}, dtpe: {tensor_q.dtype}, shape: {tensor_q.shape}'  # 4 qubits
    assert tensor_c.total_bits()  == 6 * 8  # 6 elements * 8 bits each

def torch_total_bits(tensor: torch.Tensor) -> int:
    """
    Computes the total number of bits in a tensor.
    PyTorch’s tensor.nbytes gives the total number of bytes.
    Multiply by 8 to obtain total bits.
    """
    return tensor.nbytes * 8

def test_scalar_fanout():
    reg = RegisterSpec("scalar8x4", dtype=CInt(8), shape=(4,))
    assert reg.bitsize == 8          # per element
    assert reg.total_bits() == 8 * 4 # fan-out × element

def test_tensor_fanout():
    tensor_dt = TensorType((2, 3), element_type=CBit())   # 6 bits / tensor
    reg       = RegisterSpec("tensors", dtype=tensor_dt, shape=(5,))
    # TensorType.total_bits = 6, fan-out = 5
    assert reg.total_bits() == 6 * 5

@pytest.mark.parametrize("dtype, shape, expected", [
    (QBit(),       (),     1),
    (QInt(4),      (),     4),
    (QUInt(2),     (3,),   6),   # 3 parallel wires of 2 qubits each
])
def test_quantum_total_bits(dtype, shape, expected):
    reg = RegisterSpec("q", dtype=dtype, shape=shape)
    assert reg.total_bits() == expected

def test_symbolic_dims():
    n = sympy.symbols('n', positive=True, integer=True)
    dt = TensorType((n, 2), element_type=CBit())  # 2n bits
    reg = RegisterSpec("sym", dtype=dt)           # no fan-out
    assert str(reg.total_bits()) == str(2 * n)
    m = sympy.symbols('m', positive=True, integer=True)
    dt_dyn = TensorType((m, 3), element_type=CInt(4))   # Dyn * 12 bits
    reg_dyn = RegisterSpec("dyn", dtype=dt_dyn, shape=(2,))  # fan-out ×2
    # we can only check that the symbolic multiplier is present
    assert reg_dyn.bitsize == 4
    assert reg_dyn.total_bits().free_symbols  # contains Dyn symbol

def test_multidim_register():
    r = RegisterSpec("qreg", QBit(), shape=(2, 3), flow=Flow.RIGHT)
    idxs = list(r.all_idxs())
    assert len(idxs) == 2 * 3

    assert not r.flow & Flow.LEFT
    assert r.flow & Flow.THRU
    assert r.total_bits() == 2 * 3

    assert r.flip() == RegisterSpec("qreg", QBit(), shape=(2, 3), flow=Flow.LEFT)


@pytest.mark.parametrize('n, N, m, M', [(4, 10, 5, 19), (4, 16, 5, 32)])
def test_selection_registers_indexing(n, N, m, M):
    dtypes = [BQUInt(n, N), BQUInt(m, M)]
    regs = [RegisterSpec(sym, dtype) for sym, dtype in zip(['x', 'y'], dtypes)]
    for x in range(int(dtypes[0].iteration_length)):
        for y in range(int(dtypes[1].iteration_length)):
            assert np.ravel_multi_index((x, y), (N, M)) == x * M + y
            assert np.unravel_index(x * M + y, (N, M)) == (x, y)

    assert np.prod(tuple(int(dtype.iteration_length) for dtype in dtypes)) == N * M


@pytest.mark.parametrize("torch_dtype, element_bits", [
    (torch.int8,   8),
    (torch.float32, 32),
])
def test_tensor_total_bits_vs_pytorch(torch_dtype, element_bits):
    torch_tensor = torch.ones((2, 2, 2), dtype=torch_dtype)
    dt   = TensorType(shape=torch_tensor.shape, element_type=torch_dtype)
    reg  = RegisterSpec("torch", dtype=dt)
    assert dt.total_bits == torch_total_bits(torch_tensor)
    assert reg.total_bits() == torch_total_bits(torch_tensor)

def test_matrix_resource():
    """
    Test MatrixType uses the same logiv as TensorType
    """
    t = TensorType((3,4), element_type=CInt(16)) 
    t_reg = RegisterSpec("tensor", dtype=t)
    assert t.total_bits == 12 * 16
    assert t_reg.total_bits() == 12 * 16
    mat = MatrixType((3, 4), element_type=CInt(16))    # 12 elements ×16 bits
    reg = RegisterSpec("mat", dtype=mat)
    assert mat.total_bits == 12 * 16, f'dtype.total_bits: {mat.total_bits}, tensor.total_bits: {t.total_bits}'

    assert reg.total_bits() == 12 * 16, f'dtype.total_bits: {mat.total_bits}'


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
    print("[int8]: ", f"",
          f"tensor.element_size(): {torch_tensor_int8.element_size()}, got: {tensor_c.total_bits}",
          f"tensor.nbytes: {torch_tensor_int8.nbytes}, total_bits={torch_total_bits(torch_tensor_int8)} tensor.untyped_storage().element_size: {torch_tensor_int8.untyped_storage().element_size()}")
    
    assert tensor_c.total_bits == torch_total_bits(torch_tensor_int8), f'tensor should have total_bits = {torch_total_bits(torch_tensor_int8)}. Got: {tensor_c.total_bits}'


def test_register():
    r = RegisterSpec("my_reg", QAny(5))
    assert r.name == 'my_reg'
    assert r.bitsize == 5
    assert r.shape == tuple()
    assert r.flow == Flow.THRU
    assert r.total_bits() == 5

def test_equivalent_registerspec1():
    kwargs1 = {'name': 'a', 'dtype': np.ndarray, 'shape': (2, 2)}
    kwargs2 = {'name': 'a', 'dtype': TensorType, 'shape': (2, 2)}
    reg1 = RegisterSpec(**kwargs1)
    reg2 = RegisterSpec(**kwargs2)
    assert reg1 == reg2, (
        f'np.ndarray conversion equivalence failed:\n'
        f'reg1: {reg1}\nreg2: {reg2}'
    )

    kwargs3 = {'name': 'a', 'dtype': torch.Tensor, 'shape': (2, 2)}
    reg3 = RegisterSpec(**kwargs3)
    assert reg2 == reg3, (
        f'torch.Tensor conversion equivalence failed:\n'
        f'reg2: {reg2}\nreg3: {reg3}'
    )


def test_equivalent_RegisterSpec_dtype():
    inhouse_tensor_spec = RegisterSpec('a', dtype=TensorType((2, 2)))
    should_pass = [
        RegisterSpec('dtype=ndarray', dtype=np.ndarray, shape=(2, 2)),
        RegisterSpec('dtype=torch.randn((2, 2))', dtype=torch.randn((2, 2))),
        RegisterSpec('embedded list of ints', dtype=[[1, 2], [3, 4]]),
        RegisterSpec('embedded list of ints', dtype=TensorType((Dyn, 2))),
    ]
    for reg in should_pass:
        assert inhouse_tensor_spec._same_dtype(reg), (
            f'{reg.name} conversion equivalence failed:\n'
            f'{inhouse_tensor_spec.dtype} != {reg.dtype}'
        )

@pytest.mark.parametrize('base_kwargs, modify_field, new_value, expected_msg', [
    (
        {'name': 'arg', 'dtype': int, 'shape': (3, 3)},
        'name', 'different_arg', 'name mismatch'
    ),
    (
        {'name': 'arg', 'dtype': int, 'shape': (3, 3)},
        'dtype', list[int], 'dtype mismatch'
    ),
    (
        {'name': 'cbit', 'dtype': CBit},
        'flow', Flow.LEFT, 'flow mismatch'
    )
])
def test_registerspec_inequivalence(base_kwargs, modify_field, new_value, expected_msg):
    reg1 = RegisterSpec(**base_kwargs)
    kwargs2 = dict(base_kwargs)
    kwargs2[modify_field] = new_value
    reg2 = RegisterSpec(**kwargs2)
    assert reg1 != reg2, (
        f'Expected inequivalence due to {expected_msg}.\n'
        f'reg1: {reg1}\nreg2: {reg2}\n'
        f'Field {modify_field} changed from {base_kwargs[modify_field]} to {new_value}'
    )

def test_registerspec_wire_vs_data_shape():
    """
    TODO: Actually make the following test
    Create a RegisterSpec that intends to split a (2,3) data payload into 2 wires.
    Each wire should carry a 1D vector of length 3.
      - The RegisterSpec’s wire shape should be (2,).
      - Its dtype should be a TensorType with shape (3,).
    """
    reg = RegisterSpec(
        name='x',
        dtype=TensorType((3,), element_type=float),
        shape=(2,),
        flow=Flow.LEFT
    )
    assert reg.shape == (2,), f'Expected wire shape (2,), got {reg.shape}'
    assert reg.dtype.shape == (3,), f'Expected inner data shape (3,), got {reg.dtype.shape}'


def test_registerspec_dynamic_shape():
    reg = RegisterSpec(
        name='reg',
        dtype=TensorType((3,), element_type=int),
        shape=(2,),
        flow=Flow.LEFT
    )
    reg_dyn = RegisterSpec(
        name='reg',
        dtype=TensorType((3,), element_type=int),
        shape=(Dyn,),
        flow=Flow.LEFT
    )
    assert reg_dyn.shape == (Dyn,), f'Expected (Dyn,), got {reg_dyn.shape}'
    assert check_dtypes_consistent(reg.dtype,reg_dyn.dtype), f'reg.dtype: {reg.dtype} vs. reg_dyn.dtype: {reg_dyn.dtype}'


def test_ndim_dtype_consistency():
    assert check_dtypes_consistent(TensorType, TensorType)
    assert check_dtypes_consistent(MatrixType, MatrixType)
    assert check_dtypes_consistent(TensorType((1,)), TensorType((1,)))
    
    assert check_dtypes_consistent(MatrixType((2,2)), TensorType((2,2)))
    
    # reverse
    assert check_dtypes_consistent(TensorType((2,2)), MatrixType((2,2)))

    assert check_dtypes_consistent(TensorType((Dyn,1)), TensorType((5,1)))
    # symbolic
    n = sympy.symbols('n', positive=True, integer=True)
    m = sympy.symbols('m', positive=True, integer=True)
    assert check_dtypes_consistent(TensorType((n,m)), TensorType((Dyn, Dyn)))


def test_canonicalize_dtype_numpy():
    arr = np.array([[1, 2, 3],
                    [4, 5, 6]])
    dt = canonicalize_dtype(arr)
    assert isinstance(dt, TensorType)
    assert dt.shape == arr.shape
    # assert dt.element_type == CFloat(64), f'Got: {dt.element_type}'


# def test_instance_vs_class():
#     tensor_instance = TensorType((2, 3))
#     assert check_dtypes_consistent(TensorType, tensor_instance)

#     matrix_instance = MatrixType()
#     assert check_dtypes_consistent(MatrixType, matrix_instance)


# def test_list_of_types():
#     assert check_dtypes_consistent([MatrixType, TensorType], TensorType)
#     assert check_dtypes_consistent([MatrixType, TensorType], MatrixType)
#     assert not check_dtypes_consistent([MatrixType, TensorType], int)

def test_is_consistent_data_type():
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
def test_tensor_uint8_3x1():
    dt = TensorType(shape=(3, 1), element_type=CUInt(8))

    assert dt.data_width == 8
    assert dt.element_size() == 1            # 8 bits // 8
    assert dt.nelement() == 3
    assert dt.total_bits == 24
    assert dt.rank == 2

    value = [[5], [10], [255]]
    bits  = dt.to_bits(value)
    assert bits == [
        0,0,0,0,0,1,0,1,
        0,0,0,0,1,0,1,0,
        1,1,1,1,1,1,1,1
    ]
    assert dt.from_bits(bits) == value


def test_tensor_sint4_2x2x2():
    dt = TensorType(shape=(2, 2, 2), element_type=CInt(4))

    assert dt.data_width == 4
    assert dt.element_size() == 0            # 4 bits // 8
    assert dt.nelement() == 8
    assert dt.total_bits == 32
    assert dt.rank == 3

    value = [
        [[-1, 0], [1, 2]],
        [[-4, -3], [3, 4]]
    ]
    bits = dt.to_bits(value)
    assert dt.from_bits(bits) == value


# def test_tensor_float32_vec4():
#     dt = TensorType(shape=(4,), element_type=CFloat(32))

#     assert dt.data_width == 32
#     assert dt.element_size() == 4            # 32 bits // 8
#     assert dt.nelement() == 4
#     assert dt.total_bits == 128
#     assert dt.rank == 1

#     value = [0.0, 1.0, -1.0, 3.14]
#     bits  = dt.to_bits(value)
#     assert dt.from_bits(bits) == value

def test_tensortype_assert_valid_classical_val():
    dt = TensorType(shape=(2,3), element_type=CBit())
   
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

def test_matrixtype_multiply_symbolic():
    M, N, K = sympy.symbols('m n k', positive=True, integer=True)
    sym_A = MatrixType((M, N), element_type=CInt(8))
    assert sym_A.nelement() == M*N
    assert sym_A.nbytes == M*N
    assert sym_A.total_bits == 8* M*N
    assert sym_A.is_symbolic()

    sym_B = MatrixType((N, K), element_type=CInt(8))

    sym_C =  sym_A.multiply(sym_B)
    assert sym_C.shape == (M,K), f"Expected (m,k). Got: {sym_C.shape}"
    assert sym_C.nelement() == M*K
    assert sym_C.nbytes == M*K
    assert sym_C.total_bits == 8* M*K
    assert sym_C.is_symbolic()


def test_tensor_multiply_symbolic():
    M, N, K = sympy.symbols('m n k', positive=True, integer=True)
    sym_A = TensorType(shape=(M, N), element_type=CInt(16))
    assert sym_A.nelement() == M*N
    assert sym_A.nbytes == 2* M*N
    assert sym_A.total_bits == 16* M*N
    assert sym_A.is_symbolic()

    sym_B = TensorType(shape=(N, K), element_type=CInt(16))

    sym_C =  sym_A.multiply(sym_B)
    assert sym_C.shape == (M,K), f"Expected (m,k). Got: {sym_C.shape}"



@pytest.mark.parametrize('init_kwargs1, init_kwargs2', [
    (
        {'name': 'int', 'dtype': int},
        {'name': 'int', 'dtype': int}
    ),
    (
        {'name': 'float', 'dtype': float},
        {'name': 'float', 'dtype': float}
    ),
    (
        {'name': 'float', 'dtype': np.ndarray, 'shape': (3, 3)},
        {'name': 'float', 'dtype': np.ndarray, 'shape': [3, 3]}
    ),
    (
        {'name': 'bit', 'dtype': CBit(), 'shape': (), 'flow': Flow.RIGHT},
        {'name': 'bit', 'dtype': CBit(), 'shape': (), 'flow': Flow.RIGHT}
    ),
   
    (
        {'name': 'list[int]', 'dtype': list[int], 'shape': (5,)},
        {'name': 'list[int]', 'dtype': list[int], 'shape': [5]},
    ),
])
def test_classical_register_spec_equivalence(init_kwargs1, init_kwargs2):
    reg1 = RegisterSpec(**init_kwargs1)
    reg2 = RegisterSpec(**init_kwargs2)
    assert reg1 == reg2, (
        f'Classical type equivalence failed:\n'
        f'reg1: {reg1}, dtype: {reg1.dtype}\nreg2: {reg2}, dtype: {reg2.dtype}'
    )
# TensorType vs MatrixType equality path -----------------------------------
def test_registerspec_equivalence_tensor_vs_matrix():
    reg_tensor = RegisterSpec("m", dtype=TensorType((2, 2)))
    reg_matrix = RegisterSpec("m", dtype=MatrixType(2, 2))

    assert reg_tensor == reg_matrix


# 3) Nested quantum element type drives domain --------------------------------
def test_tensor_of_qbit_has_quantum_domain():
    """
    Even a classical TensorType wrapper becomes quantum if its
    element_type is quantum.
    """
    dt_qtensor = TensorType((4,), element_type=QBit())   # 4-element tensor
    reg = RegisterSpec("qvec", dtype=dt_qtensor)

    assert reg.domain == "Q"
    assert reg.total_bits() == 4                        # 4 qubits total


# 4) Auto-instantiation happens only for fan-out wires ------------------------
def test_dtype_auto_instantiation_scalar_vs_vector():
    """
    RegisterSpec with _shape == () keeps the dtype *class* un-instantiated.
    Non-empty shape instantiates the dtype with that fan-out.
    """
    # scalar wire
    reg_scalar = RegisterSpec("x", dtype=TensorType, shape=())
    assert isinstance(reg_scalar.dtype, type), "dtype should still be a class for scalar"

    # vector wire (fan-out)
    reg_vec = RegisterSpec("y", dtype=TensorType, shape=(3,))
    assert isinstance(reg_vec.dtype, TensorType), "dtype should be an instance after fan-out"


# ----------------------------------------------------------------
# `Signature` tests 
# ----------------------------------------------------------------

def test_signature_build():
    sig = Signature.build(a=1, b=2)
    assert len(sig) == 2
    for reg in sig:
        # For registers built from simple ints, we keep shape=() atomic
        assert reg.shape == (), f'Expected shape=(), got {reg.shape}'

# 1) THRU-flow partitioning ---------------------------------------------------
def test_signature_flow_partitioning():
    # Explicitly construct the three registers first
    a_reg = RegisterSpec("a", dtype=CBit())        # THRU (default flow)
    b_reg = RegisterSpec("b", dtype=CUInt(8))      # THRU
    c_reg = RegisterSpec("c", dtype=QBit(), flow=Flow.RIGHT)

    # Build the Signature from a list, not with the kwargs helper
    sig = Signature([a_reg, b_reg, c_reg])

    assert [r.name for r in sig.lefts()]  == ["a", "b"]
    assert [r.name for r in sig.rights()] == ["a", "b", "c"]


# 2) Literal-to-dtype promotion rules ----------------------------------------
def test_literal_promotion_rules():
    """
    Signature.build() should upgrade numeric literals:
      • 1  -> CBit()
      • n>1 -> CUInt(n)
    """
    sig = Signature.build(one=1, five=5, neg_five=-5)
    one_reg, five_reg, neg_five = sig

    assert isinstance(one_reg.dtype, CBit)
    assert isinstance(five_reg.dtype, CUInt) and five_reg.dtype.bit_width == 5
    assert isinstance(neg_five.dtype, CInt) and neg_five.dtype.bit_width == 5, f'Got dtype: {neg_five.dtype}'


def test_variadic_register_matches_list():
    var_reg   = RegisterSpec("args", dtype=CBit(), variadic=True)
    fixed_reg = RegisterSpec("ret",  dtype=CBit(), flow=Flow.RIGHT)

    sig = Signature([var_reg, fixed_reg])

    data_in  = [Data(CBit(), {}) for _ in range(5)]   # 5 inputs for variadic slot
    data_out = [Data(CBit(), {})]                     # 1 output
    data_all = data_in + data_out

    # Should validate without raising
    sig.validate_data_with_register_specs(data_all)




def test_signature_with_simple_data():
    
    input_properties = [
        {'Data Type': TensorType((Dyn,)), 'Usage': 't1'}, 
        {'Data Type': TensorType((Dyn,)), 'Usage': 't2'}]
    
    output_properties = [
        {'Data Type': TensorType((Dyn,)), 'Usage': 't3'}
        ]
    
   
    signature_from_properties = Signature.build_from_properties(
        input_props=input_properties,
        output_props=output_properties,
    )
    

    d1,d2,dout = signature_from_properties._registers
    
    signature_from_data = Signature([d1,d2,dout])

    assert signature_from_data == signature_from_properties, f"Signatures do not match! \n{signature_from_data}\n{signature_from_properties}"


def test_signature_from_properties():
    
    input_properties = [
        {'Data Type': TensorType, 'Usage': 'Matrix1'}, 
        {'Data Type': TensorType, 'Usage': 'Matrix2'}]
    
    output_properties = [
        {'Data Type': TensorType, 'Usage': 'Out'}
        ]

    signature_from_properties = Signature.build_from_properties(
        input_props=input_properties,
        output_props=output_properties,
    )
    d1,d2,dout = signature_from_properties._registers
    
    signature_from_data = Signature.build([d1,d2,dout])
    assert signature_from_data == signature_from_properties, f"Signatures do not match! \n{signature_from_data}\n{signature_from_properties}"


def test_signature_roundtrip_scalar():
    #  simple classical + default THRU flow
    sig1 = Signature(
        [
            RegisterSpec("a", dtype=CBit()),
            RegisterSpec("b", dtype=CUInt(8), flow=Flow.RIGHT),
        ]
    )
    blob = sig1.to_dict()
    sig2 = Signature.from_dict(json.loads(json.dumps(blob)))   # simulate disk round-trip
    assert sig1 == sig2

def test_signature_roundtrip_tensor():
    
    dt = TensorType((2, 3), element_type=np.float32)
    sig1 = Signature(
        [
            RegisterSpec("x", dtype=dt, flow=Flow.LEFT),
            RegisterSpec("y", dtype=CBit()),
        ]
    )
    sig2 = Signature.from_dict(sig1.to_dict())   # direct, no JSON stringify
    assert sig1 == sig2

   
    input_only_reg = sig2.get_left('x')
    # assert input_only_reg.dtype == dt
    assert input_only_reg.flow is Flow.LEFT, f'Expect Flow.LEFT. Got: {input_only_reg.flow}'

    assert [r.name for r in sig2.rights()] == ["y"]
    # same check but via different method
    out_reg = sig2.get_right('y')
    assert out_reg.dtype == CBit()
    assert out_reg.flow is Flow.THRU, f'Expect Flow.THRU. Got: {out_reg.flow}'
    # assert [r.flow for r in sig2.lefts()] == [Flow.LEFT], f'Signature: {sig2} –– lefts: {list(sig2.lefts())}'
    assert [r.flow for r in sig2.rights() if r.name == "x"] == []   # x is not RIGHT
    # same check but via different method
    with pytest.raises(KeyError):
        sig2.get_right('x')

def test_signature_roundtrip_quantum_symbolic():
    # Case 3 – quantum dtype & symbolic tensor shape
    n = sp.symbols("n", positive=True, integer=True)
    sig1 = Signature(
        [
            RegisterSpec("ctrl", dtype=QBit()), # flow defauls to Flow.THRU
            RegisterSpec(
                "psi",
                dtype=TensorType((n, 2), element_type=QInt(4)),
            ),
        ]
    )
    sig2 = Signature.from_dict(sig1.to_dict())
    assert sig1 == sig2
    # confirm symbolic shape survives
    psi1 = sig1["psi"]
    psi2 = sig2["psi"]
    assert psi1.dtype.shape == psi2.dtype.shape == (n, 2)


def test_signature_roundtrip_with_props():

    # Case 4 – custom .properties payload persists
    regs = [
        RegisterSpec("data", dtype=torch.Tensor, shape=(Dyn,), flow=Flow.THRU),
        RegisterSpec("sum",  dtype=torch.Tensor, flow=Flow.RIGHT),
    ]
    sig1 = Signature(regs)
    # manually attach an auxiliary property dict as many Process builders do
    sig1.properties = {"stage": "reduce", "author": "alice"}

    sig2 = Signature.from_dict(sig1.to_dict())
    assert sig2.properties == {"stage": "reduce", "author": "alice"}
    assert sig1 == sig2

# ----------------------------------------------------------------
# `Data` tests (runtime level)
# ----------------------------------------------------------------
def test_promote_element_types_string():
    """
    Verify that promoting a NumPy string dtype results in the Python type 'str'.
    """
    import numpy as np
    result = promote_element_types(np.dtype('<U1'))
    assert result == str, f"Expected promotion of np.dtype('<U1') to be str, got {result}"

def test_promote_element_types_mixed_numbers():
    """
    Verify that promoting a mix of integer and floating NumPy dtypes results in a floating dtype.
    """
    import numpy as np
    result = promote_element_types(np.dtype('int32'), np.dtype('float64'))
    expected = np.dtype('float64')
    assert result == expected, f"Expected promotion to be {expected}, got {result}"


def test_data_string_array_element_type():
    """
    Create a Data object from a NumPy array of strings and check that
    the TensorType element_type is correctly set to str.
    """
    import numpy as np
    str_array = np.array([['1', '2'], ['3', '4']])
    # Using "Matrix" usage so the hint is set to "Matrix" (your test uses that)
    data_obj = Data(str_array, {"Usage": "Matrix"})
    # The inferred data type is a TensorType. We need to check its element_type.
    element_type = data_obj.metadata.dtype.element_type
    assert element_type == str, f"Expected element_type to be str, got {element_type}"


