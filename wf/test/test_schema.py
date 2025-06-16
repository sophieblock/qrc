import pytest
import numpy as np
import re 
import math
import torch
import sympy
from attrs import define
sep_str = 100 * '#'

# from qrew.simulation.refactor.register import RegisterSpec, Flow
from qrew.simulation.refactor.schema import RegisterSpec, Signature, Flow
from qrew.simulation.refactor.data_types import *
from qrew.simulation.refactor.data import Data,promote_element_types
from qrew.simulation.refactor.unification_tools import (
    type_matches, is_consistent_data_type,canonicalize_dtype
)
from qrew.simulation.refactor.process import Process,ClassicalProcess
from qrew.assert_checks import assert_registers_match_parent, assert_multiline_equal

from qrew.visualization_tools import ModuleDrawer, _assign_ids_to_nodes_and_edges
from qrew.simulation.refactor.builder import ProcessBuilder,ProcessInstance, CompositeMod,PortInT,PortT,Port, LeftDangle,RightDangle
from qrew.simulation.refactor.Process_Library.for_testing import Atom,Atom_n,TwoBitOp,SwapTwoBit, AtomChain,NAtomParallel,SplitJoin,TestParallelQCombo
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


# ----------------------------------------------------------------
# Signature
# ----------------------------------------------------------------

def test_signature_build():
    sig = Signature.build(a=1, b=2)
    assert len(sig) == 2
    for reg in sig:
        # For registers built from simple ints, we keep shape=() atomic
        assert reg.shape == (), f'Expected shape=(), got {reg.shape}'



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


# ----------------------------------------------------------------
# tests for builder.py
# ----------------------------------------------------------------

def test_builder():
    
    signature = Signature.build(x=1,y=1)
    x_reg, y_reg = signature
    builder, initial_ports = ProcessBuilder.from_signature(signature)
    assert initial_ports == {'x': Port(LeftDangle, x_reg), 'y': Port(LeftDangle, y_reg)}
    x, y = initial_ports['x'],initial_ports['y']
    x, y = builder.add(TwoBitOp(),a = x, b = y)
    x, y = builder.add(TwoBitOp(),a = x, b = y)
    composite_op = builder.finalize(x=x,y=y)
    inds = {process_instances.i for process_instances in composite_op.pinsts}
    assert len(inds) == 2
    assert len(composite_op.pinsts) == 2
    # composite_op.describe
    sig1 = composite_op.signature
    # assert composite_op.validate_data()

    # builder2, initial_ports2 = 
    composite_op_op = composite_op.as_composite()
    composite_op_op.describe
    sig2 = composite_op_op.signature
    assert sig1 == sig2
    
    decomposed_op = composite_op_op.as_composite()
    # decomposed_op.describe

    save_mod(decomposed_op, label_type='dtype', filename='TwoBitOp_dtype.png', output_dir=figure_path+'Atomic_Examples/')
    save_mod(decomposed_op, label_type='num_units', filename='TwoBitOp_num_units.png', output_dir=figure_path+'Atomic_Examples/')
    
def test_TwoBitOp_process():
    
    twobit_process = TwoBitOp()
    assert len(twobit_process.signature) == 2
    composite_twobit_process = twobit_process.as_composite()
   
    actual_output = composite_twobit_process.debug_text()

    expected_output = """\
TwoBitOp<0>
  LeftDangle.a -> a
  LeftDangle.b -> b
  a -> RightDangle.a
  b -> RightDangle.b"""


    assert_multiline_equal(actual_output, expected_output)



def test_atom():
    
    atomic_process = Atom()
    
    catom = atomic_process.as_composite()
    assert atomic_process.signature == catom.signature
    expected_debug_text = """\
Atom<0>
  LeftDangle.n -> n
  n -> RightDangle.n"""
    assert_multiline_equal(catom.debug_text(),expected_debug_text)

    save_mod(catom,  filename="CAtom()_dtype.png",output_dir=figure_path+'Atomic_Examples/')

def test_atom_n():
    bitsize = 4

    atom = Atom_n(n=bitsize)
    
    assert len(atom.signature) == 1
    assert len(atom.signature._lefts) == 1  # Confirm 1 input in signature
    assert len(atom.signature._rights) == 1, f"atom.signature._rights len(atom.signature._rights): {len(atom.signature._rights)}"

    catom = atom.as_composite()
    assert isinstance(catom, CompositeMod)
    assert (catom.signature == atom.signature)
    process_instances = list(catom.pinsts)
    assert len(process_instances) == 1
    assert process_instances[0].process == atom
    bag = atom.signature
    assert bag[0].bitsize == 4, f'Expected 4 got {bag[0].bitsize}'
    # catom.print_tabular()
    
    save_mod(catom,  filename="CAtom_n()_dtype.png",output_dir=figure_path+'Atomic_Examples/')
    save_mod(catom, label_type='num_units', filename="CAtom_n()_num_units.png",output_dir=figure_path+'Atomic_Examples/')

def test_iter_process_connections():
    op = SwapTwoBit()
    c_op = op.decompose() 

    save_mod(c_op,  filename="SwapTwoBit_dtype.png",output_dir=figure_path+'Atomic_Examples/')
    save_mod(c_op, label_type='num_units', filename="SwapTwoBit_num_units.png",output_dir=figure_path+'Atomic_Examples/')


    c_op2 = op.as_composite() # composited operation
    
    c_op_output = c_op.debug_text()
    
    expected_output = """\
TwoBitOp<0>
  LeftDangle.d1 -> a
  LeftDangle.d2 -> b
  b -> TwoBitOp<1>.a
  a -> TwoBitOp<1>.b
--------------------
TwoBitOp<1>
  TwoBitOp<0>.b -> a
  TwoBitOp<0>.a -> b
  a -> RightDangle.d1
  b -> RightDangle.d2"""


    assert_multiline_equal(c_op_output, expected_output)
    c_op2_output = c_op2.debug_text()
    
    expected_output2 = """\
SwapTwoBit<0>
  LeftDangle.d1 -> d1
  LeftDangle.d2 -> d2
  d1 -> RightDangle.d1
  d2 -> RightDangle.d2"""


    assert_multiline_equal(c_op2_output, expected_output2)
    
    save_mod(c_op2,  filename="composite_SwapTwoBit_dtype.png",output_dir=figure_path+'Atomic_Examples/')
    save_mod(c_op2, label_type='num_units', filename="composite_SwapTwoBit_num_units.png",output_dir=figure_path+'Atomic_Examples/')


    c_op.print_tabular()
    c_op.print_tabular_fx()
    # c_op2.print_tabular()
    assert len(list(c_op.iter_process_connections())) == len(c_op.pinsts)
    for pinst, preds, succs in c_op.iter_process_connections():
        print(pinst)
        print(f' - preds: {preds}')
        print(f' - succs: {succs}')
        assert isinstance(pinst, ProcessInstance)
        assert len(preds) > 0
        assert len(succs) > 0

def test_chained_atoms():
    save_path = figure_path + 'Atomic_Examples/'
    chained_process = AtomChain()
    # print(chained_process)
    # print(str(chained_process))
    # print(repr(chained_process))


    # chained_process.describe
    
    decomposed_chained = assert_registers_match_parent(chained_process)

    debug_output = decomposed_chained.debug_text()
    

    expected_output = """\
Atom<0>
  LeftDangle.reg -> n
  n -> Atom<1>.n
--------------------
Atom<1>
  Atom<0>.n -> n
  n -> Atom<2>.n  
--------------------
Atom<2>
  Atom<1>.n -> n
  n -> RightDangle.reg"""

    assert_multiline_equal(debug_output,expected_output)
    sig1 = decomposed_chained.signature

    # assert decomposed_chained.validate_data()
    # print(decomposed_chained.signature)

    save_mod(decomposed_chained,  filename="chained_atom_decomposed_dtype.png",output_dir=save_path + 'Series/')
    save_mod(decomposed_chained, label_type='num_units', filename="chained_atom_decomposed_num_units.png",output_dir=save_path + 'Series/')


    composite_chained = chained_process.as_composite()

    expected_output2 = """\
AtomChain<0>
  LeftDangle.reg -> reg
  reg -> RightDangle.reg"""
    assert_multiline_equal(composite_chained.debug_text(),expected_output2)
    sig2 = composite_chained.signature

    assert sig1==sig2, f"sig1: {sig1} != to sig2: {sig2}"
    
    # Draw high level of the process
    save_mod(composite_chained,  filename="highlevel_chained_atom_dtype.png",output_dir=save_path + 'Series/')
    
    # Draw high level of the process with edge labels showing data size as labels
    save_mod(composite_chained, label_type='num_units', filename="highlevel_chained_atom_num_units.png",output_dir=save_path + 'Series/')




bookkeeping_path = figure_path + 'Bookeeping_Examples/'
def test_split():
    spec = CInt(4)
    
    assert isinstance(spec,CType)
    split = Split(spec).as_composite()
    save_mod(split, label_type='dtype', filename="split_cuint4_dtype.png",output_dir=bookkeeping_path)

    save_mod(split, label_type='num_units', filename="split_cuint4_num_units.png",output_dir=bookkeeping_path)

def test_builder_splitjoin():

    builder = ProcessBuilder()
    atomic_reg = RegisterSpec(name="a_tensor", dtype=TensorType((2,),element_type=CBit()), flow=Flow.LEFT)

    in_port = builder.add_register(atomic_reg)  # returns a single Port
    assert isinstance(in_port, Port), f".add_register() should return a Port if reg.flow is LEFT/THRU. Instead, we got type(in_port) = {type(in_port)}"
    
    assert in_port.process_instance is LeftDangle
    assert isinstance(in_port.index, tuple)
    assert in_port.pretty() == 'a_tensor'
    
   
    split_out= builder.add(Split(dtype=in_port.reg.dtype), arg=in_port)
    # assert split_out.shape == (2,)
   
    # assert split_out[0].reg.dtype == CBit()

    assert str(split_out[0].process_instance) == 'Split<0>', f'Got: {str(split_out[0].process_instance)}'
    assert split_out[0].pretty() == 'arg[0]', f'Got: {split_out[0].pretty()}'
    

    join_process = Join(dtype=atomic_reg.dtype)

    joined_out = builder.add(join_process, arg=split_out)


    composite_mod = builder.finalize(a_tensor=joined_out)
    
    expected_debug_text = """\
Split<0>
  LeftDangle.a_tensor -> arg
  arg[0] -> Join<1>.arg[0]
  arg[1] -> Join<1>.arg[1]
--------------------
Join<1>
  Split<0>.arg[0] -> arg[0]
  Split<0>.arg[1] -> arg[1]
  arg -> RightDangle.a_tensor"""
    assert_multiline_equal(composite_mod.debug_text(),expected_debug_text)
    # print(composite_mod.signature)
    save_mod(composite_mod, label_type='dtype', filename="splitjoin_tensor_(2,)_dtype.png",output_dir=bookkeeping_path)
    save_mod(composite_mod, label_type='num_units', filename="splitjoin_tensor_(2,)_num_units.png",output_dir=bookkeeping_path)
    
    # Attempt to print tabular in a similar format as torch.graph_module.print_tabular()
    composite_mod.print_tabular()
    composite_mod.print_tabular_fx()

@define
class SplitJoin(Process):
    n: int 

    @property
    def signature(self):
        return Signature([RegisterSpec('x', dtype=TensorType((self.n,)))])
    def build_composite(self, builder, *,x: 'Port'):
        
        xs = builder.split(x)
        x =  builder.join(xs, dtype=x.reg.dtype)
        return {'x':x}
    
def test_SplitJoin_class():
    split_join = SplitJoin(n=2)
    
    sj = split_join.as_composite()
    
    drawer = ModuleDrawer(sj)
    drawer.render(display=False, filename="composite_dtype.png",output_dir=figure_path + 'SplitJoin()/')

    # print(split_join.decompose().debug_text())
    c_splitjoin = split_join.decompose()
    drawer = ModuleDrawer(c_splitjoin)

    drawer.render(display=False, filename="decomposed_dtype.png",output_dir=figure_path + 'SplitJoin()/')
    c_splitjoin.print_tabular()

    c_splitjoin.print_tabular_fx()

# def test_parallel_atoms():
    
#     num_parallel = 3
#     d = Data(CUInt(3), properties={"Usage": "bits"})
#     parallel_process = NAtomParallel(inputs=[d],num_parallel=num_parallel)
    
   
#     assert len(parallel_process.signature) == 1
    
#     # Test decomposition
#     cprocess = parallel_process.decompose()
#     assert cprocess.signature == parallel_process.signature

#     # for debugging
#     cprocess.print_tabular()
#     cprocess.print_tabular_fx()

#     drawer = ModuleDrawer(cprocess)
#     drawer.render(display=False, save_fig=True, filename="decomposed_dtype.png",output_dir=figure_path + 'NAtomParallel(Process)/')

    # draw with num_units as the edge labels
    # drawer = ModuleDrawer(cprocess, label_type="num_units")
    # drawer.render(display=False, save_fig=True, filename="NAtomParallel_decomposed_num_units.png")

    # # Render high-level block
    # composite_block = parallel_process.as_composite()
    # drawer = ModuleDrawer(composite_block)
    # drawer.render(display=False, save_fig=True, filename="NAtomParallel_highlevel.png")
  
    # composite_block.print_tabular()

    # composite_block.print_tabular_fx()
