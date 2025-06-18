import pytest
import numpy as np
import random 
import math

from qrew.simulation.data_types import *



def test_default_constructor_and_fields():
    bs = BitStringView()
    assert bs.integer == 0
    assert bs.nbits == 1
    assert bs.numbering == BitNumbering.MSB
    assert bs.dtype is None


def test_integer_assignment_and_nbits_growth():
    bs = BitStringView(integer=3)
    assert bs.nbits == 2  # 0b11 needs 2 bits
    bs.integer = 7
    assert bs.integer == 7
    assert bs.nbits == 3  # 0b111 needs 3 bits



@pytest.mark.parametrize("value, nbits, expected", [
    (5, 4, "0101"),
    (2, 3, "010"),
])
def test_binary_and_bits_methods_msb(value, nbits, expected):
    bs = BitStringView.from_int(value, nbits=nbits)
    assert bs.binary() == expected
    assert bs.bits() == [int(c) for c in expected] 
    assert bs.array() == [int(c) for c in expected]


@pytest.mark.parametrize("value, nbits, expected", [
    (5, 4, "1010"),
    (2, 3, "010"),
])
def test_binary_and_bits_methods_lsb(value, nbits, expected):
    bs = BitStringView.from_int(value, nbits=nbits, numbering=BitNumbering.LSB)
    assert bs.binary() == expected
    assert bs.bits() == [int(c) for c in expected]
    # `.array()` is same as `.bits()`
    assert bs.array() == [int(c) for c in expected]

def test_msb_lsb_factories():
    # msb factory
    msb3 = BitStringView.msb(3, nbits=3)
    assert isinstance(msb3, BitStringView)
    assert msb3.numbering == BitNumbering.MSB
    # lsb factory
    lsb3 = BitStringView.lsb(3, nbits=3)
    assert isinstance(lsb3, BitStringView)
    assert lsb3.numbering == BitNumbering.LSB

def test_with_numbering_switch():
    msb = BitStringView.from_int(6, nbits=3, numbering=BitNumbering.MSB)
    # '6' in 3 bits: '110'
    assert msb.binary() == "110"
    lsb = msb.with_numbering(BitNumbering.LSB)
    # reversed: '011'
    assert lsb.numbering == BitNumbering.LSB
    assert lsb.binary() == "011"
    # switch back yields original object equality
    assert lsb.with_numbering(BitNumbering.MSB) == msb

@pytest.mark.parametrize("binary, expected_int, expected_nbits", [
    ("0b101", 5, 3),
    ("1010", 10, 4),
])
def test_from_binary_methods(binary, expected_int, expected_nbits):
    bs = BitStringView.from_binary(binary)
    assert int(bs) == expected_int
    # nbits = length of string without '0b'
    length = len(binary.lstrip("0b"))
    assert bs.nbits == length
    assert bs.nbits == expected_nbits

    # default MSB
    assert bs.numbering == BitNumbering.MSB

@pytest.mark.parametrize("array, expected_int, expected_nbits", [
    ([1,0,1], 5, 3),
    ([1,0,1,0], 10, 4),
])
def test_from_array_methods(array, expected_int, expected_nbits):
    bs = BitStringView.from_array(array)
    assert int(bs) == expected_int
    assert bs.nbits == len(array)
    assert bs.nbits == expected_nbits, f'{bs} nbits not equal to {expected_nbits}. len(arr)={len(array)}'
    assert bs.array() == array
    assert bs.numbering == BitNumbering.MSB

def test_constructor_overloads():
    a = BitStringView.from_int(7, nbits=4)
    # passing a BitStringView instance back into from_int yields same object
    assert BitStringView.from_int(a) == a
    # from_array/from_binary overloads
    assert BitStringView.from_array(a.array()) == a
    assert BitStringView.from_binary(a.binary()) == a

def test_equality_hashing_and_add():
    a = BitStringView.from_int(integer=3, nbits=2)
    b = BitStringView.from_int(integer=3, nbits=2)
    c = BitStringView.from_int(integer=1, nbits=2)
    assert a == b
    assert a != c
    assert hash(a) == hash(b)
    d = a + c
    assert isinstance(d, BitStringView)
    assert int(d) == 4  # 3 + 1 = 4
    assert d.nbits >= max(a.nbits, c.nbits)

@pytest.mark.parametrize("arr", [
    [1, 0, 1, 0],
    [0, 1, 1],
    [1, 1, 0, 1],
])
def test_array_lsb_roundtrip(arr):
    """LSB‐ordered array → BitStringView → reinterpret as MSB flips the bits."""
    # raw LSB view preserves the input array
    lsb = BitStringView.from_array(arr, nbits=len(arr), numbering=BitNumbering.LSB)
    assert lsb.array() == arr

    # if we reinterpret that same bitstring as MSB, the bits reverse
    msb_from_lsb = BitStringView.from_bitstring(lsb, numbering=BitNumbering.MSB)
    assert msb_from_lsb.bits() == list(reversed(arr))

@pytest.mark.parametrize("binary, expected_int", [
    ("1010", 5),    # LSB '1010' → reverse→ '0101'→5
    ("0b011", 6),   # LSB '011'  → reverse→ '110' →6
])
def test_from_binary_lsb(binary, expected_int):
    """Ensure from_binary honors LSB ordering correctly."""
    # construct with LSB ordering
    bs = BitStringView.from_binary(binary, numbering=BitNumbering.LSB)

    # integer value should match reversed bits
    assert bs.integer == expected_int

    # and .binary() should return exactly the LSB‐ordered string (no '0b')
    raw = binary[2:] if binary.startswith("0b") else binary
    # format expected_int to width=len(raw), then reverse to LSB order
    formatted = format(expected_int, f"0{len(raw)}b")
    expected_bin = formatted[::-1]
    assert bs.binary() == expected_bin

# ----------------------------------------------------------------------
# 1) Indexing – MSB vs LSB order
# ----------------------------------------------------------------------
def test_getitem_msb_lsb():
    """
    Verify that indexing respects the chosen bit-ordering:

        • MSB view  — index 0 is the most-significant bit
        • LSB view  — index 0 is the least-significant bit
    """
    pattern = 0b1100_1010            # 0xCA
    bs_msb  = BitStringView.msb(pattern, nbits=8)
    bs_lsb  = bs_msb.with_numbering(BitNumbering.LSB)

    # explicit spot checks
    assert bs_msb[0] == 1, "MSB[0] should be the left-most ‘1’"
    assert bs_msb[7] == 0, "MSB[7] should be the right-most ‘0’"

    assert bs_lsb[0] == 0, "LSB[0] should be original LSB (right-most)"
    assert bs_lsb[7] == 1, "LSB[7] should be original MSB (left-most)"

    # order-reversal property: bs_lsb[i] == bs_msb[7-i]
    for i in range(8):
        assert bs_lsb[i] == bs_msb[7 - i], (
            f"Index mismatch at {i}: MSB[{7-i}]={bs_msb[7-i]} vs "
            f"LSB[{i}]={bs_lsb[i]}"
        )

# ----------------------------------------------------------------------
# 2) Mutation – __setitem__ in both views
# ----------------------------------------------------------------------
def test_setitem_mutation():
    """
    • Clear the MSB in an 8-bit view
    • Clear the LSB in a 4-bit LSB-ordered view
      (shows that the same index touches a *different* physical bit)
    """
    # ---- MSB view: toggle the most-significant bit --------------------
    bs = BitStringView.msb(0b1001_0011, nbits=8)   # 0x93 → 10010011
    bs[0] = 0                                       # clear MSB
    assert int(bs) == 0b0001_0011, (
        f"Expected 0b00010011 after clearing MSB, got {bs.binary()}"
    )
    assert bs.binary() == "00010011", "Binary string not updated correctly"
    assert bs.nbits == 8, "Width should stay constant after mutation"

    # ---- LSB view: toggle the least-significant bit -------------------
    bs_lsb = BitStringView.lsb(0b0101, nbits=4)     # value 5 → LSB view ‘1010’
    bs_lsb[0] = 0                                   # clear LSB (index 0 in LSB mode)
    assert int(bs_lsb) == 0b0100, (
        f"Expected integer 4 after clearing LSB, got {int(bs_lsb)}"
    )
    assert bs_lsb.binary() == "0010", (
        "LSB binary string should reflect bit-0 change"
    )
    assert bs_lsb.nbits == 4, "Width should remain 4 bits"

def test_bitstring_cbit_msb():
    dt      = CBit()
    bs      = dt.to_bitstring(1, numbering=BitNumbering.MSB)
    assert isinstance(bs, BitStringView)
    assert bs.integer == 1,  f"MSB integer should be 1, got {bs.integer}"
    assert bs.nbits == 1,  f"MSB nbits should be 1, got {bs.nbits}"
    assert bs.dtype == dt, f"dtype not set after widen_to_dtype. bs.dtype: {bs.dtype}"

def test_bitstring_cuint8_roundtrip():
    dt  = CUInt(8)
    val = 0b10110011

    msb = dt.to_bitstring(val, numbering=BitNumbering.MSB)
    assert msb.binary() == "10110011", f"MSB binary wrong: {msb.binary()}"
    
    lsb = dt.to_bitstring(val, numbering=BitNumbering.LSB)
    assert lsb.binary() == "11001101", f"LSB binary wrong: {lsb.binary()}"
    
    # round-trip via .bits()
    rebuilt = int("".join(map(str, reversed(lsb.bits()))), 2)
    assert rebuilt == val, f"Round-tripped val {rebuilt} != original {val}"


def test_bitstring_cint4_min():
    # signed min value (-8) in 4-bit CInt
    dt, val = CInt(4), -8
    bs = dt.to_bitstring(val)
    assert bs.binary() == "1000"
    assert dt.from_bits(bs.bits()) == val


def test_bitstring_no_change_if_positive():
    # widen unsigned value that already fits the dtype
    dt = CInt(4)
    bs = dt.to_bitstring(5)          
    assert bs.integer == 5
    assert bs.binary() == "0101"

def test_bitstring_cint4_negative():
    dt  = CInt(4)    
    val = -3
    bs  = dt.to_bitstring(val)         
    assert bs.nbits == 4, f"nbits should widen to 4. Got: {bs.nbits}"
    assert bs.binary() == "1101", f"Two’s-complement −3 is 1101, got {bs.binary()}. bs.info(): {bs.info()}"
    
    decoded = dt.from_bits(bs.bits())

    assert decoded == val, f"Roundtrip failed: Decoded value should equal original value: {val}. Got: {decoded}"

def test_bitstring_from_list():
    dt   = CUInt(5)
    bits = [1,0,1,0,1]                 # MSB order
    bs   = dt.to_bitstring(bits)
    assert bs.integer == 0b10101, f"Expected 0b10101, got {bs.integer:b}"
    assert bs.nbits == 5, f"nbits should be 5 after widen. Got: {bs.nbits}"


def test_widen_to_dtype(tmp_path):
    class FakeType:
        data_width = 5
        @staticmethod
        def assert_valid_classical_val(val, name):
            assert isinstance(val, int)
    bs = BitStringView.from_int(integer=1, nbits=2)
    bs2 = bs.widen_to_dtype(FakeType)
    assert bs2.nbits == 5

def test_bitstring_reuse():
    initial = BitStringView.from_int(6, nbits=3) # 110
    dt      = CUInt(4)
    widened = dt.to_bitstring(initial) # in-place widen
    assert widened is initial, "widen_to_dtype should mutate, i.e., same object"
    assert widened.nbits == 4
    assert widened.dtype == dt

def test_widen_to_cuint():
    b_msb = BitStringView.from_int(13, nbits=4, numbering=BitNumbering.MSB)
    assert b_msb.binary() == '1101'
    assert int(b_msb) == 13
    assert len(b_msb) == b_msb.nbits == 4

    b_msb.widen_to_dtype(CUInt(8))
    assert b_msb.dtype.data_width == b_msb.nbits == 8

@pytest.mark.parametrize("dtype,value", [
    (CInt(4),      -3),
    (CUInt(5),     19),
    (QBit(),        1),
    (QInt(6),     -17),
    (BQUInt(4,8),   7),
])
def test_to_bitstring_single_impl(dtype, value):
    bs = dtype.to_bitstring(value)
    # round-trip
    assert dtype.from_bits(bs.bits()) == value
    # width is fixed by dtype
    assert bs.nbits == dtype.data_width

def test_widen_to_dtype_quint():
    dtype = QUInt(8)
    bs = BitStringView.from_int(255).widen_to_dtype(dtype)
    resource_ok = dtype.to_bits(bs.integer)
    assert all(resource_ok), f'resource_ok: {resource_ok}'
    # bs = dtype.bitstring(255)   # same thing

def test_constructor():
    for i in range(15):
        bita = BitStringView.from_int(integer=i)
        # print(f"BitA: {bita.info()}")
        # bita_lsb = BitStringView.from_int(integer=i, numbering=BitNumbering.LSB)
        bita_lsb = bita.with_numbering(BitNumbering.LSB)
        bitb = BitStringView.from_int(integer=bita)
        # print(f"BitB: {bitb.info()}")
        bitc = BitStringView.from_int(integer=bita_lsb)
        # print(f"BitC: {bitc.info()}")
        bitd = BitStringView.from_array(array=bita)
        # print(f"BitD: {bitd.info()}")
        bite = BitStringView.from_array(array=bita_lsb, numbering=BitNumbering.MSB)
        # print(f"BitE: {bite.info()}")
        bitf = BitStringView.from_binary(binary=bita)
        # print(f"BitF: {bitf.info()}")
        bitg = BitStringView.from_binary(binary=bita_lsb, numbering=BitNumbering.MSB)
        # print(f"BitG: {bitg.info()}")
        assert (bita == bitb)
        assert (bita == bitc)
        assert (bita == bitd)
        assert (bita == bite)
        assert (bita == bitf)
        assert (bita == bitg)

def test_bitstrings():
   
    for i in range(15):
        bit = BitStringView.from_int(integer=i)     
        assert bit.integer == i,             f"stored integer should be {i}"
        # binary() is zero-padded to bit.nbits, strip leading zeros 
        assert bit.binary().lstrip("0") or "0" == format(i, 'b')
        assert bit.binary().lstrip("0") or "0" == bin(i)[2:]


    arrays    = [
        [0, 0, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 0, 1, 1, 1, 0],
        [1, 0, 0, 1, 1, 1, 0, 0, 1, 0],
    ]
    integers  = [
        1,
        4,
        5,
        32 + 8 + 4 + 2,              # 0b1001110
        2 + 16 + 32 + 64 + 512,      # 0b1001001110
    ]

    for idx, arr in enumerate(arrays):
        nbits = len(arr)

        bita = BitStringView.from_array(array=arr, nbits=nbits)                
        bitb = BitStringView.from_int(integer=integers[idx], nbits=nbits)    

        assert bita == bitb, f"obj equality failed at idx={idx}"
        assert bita.integer == integers[idx]
        assert int(bita) == bita.integer
        assert bita.array() == arr
        assert bitb.integer == integers[idx]
        assert bitb.array() == arr

@pytest.mark.parametrize("value", range(15))
def test_bitstring_lsb_basic(value: int):
    bit_msb = BitStringView.from_int(integer=value)                    
    bit_lsb = BitStringView.from_int(integer=value,
                                     numbering=BitNumbering.LSB)        

   
    assert bit_msb.integer == value
    assert (bit_msb.binary().lstrip("0") or "0") == format(value, "b")    
    assert (bit_msb.binary().lstrip("0") or "0") == bin(value)[2:]       
    assert bit_lsb.integer == bit_msb.integer         
    assert bit_msb != bit_lsb                                           

cases = [
    ([0, 0, 1],                    1,   4),
    ([1, 0, 0],                    4,   1),
    ([1, 0, 1],                    5,   5),
    ([1, 0, 1, 1, 1, 0],          46,  29),   # 0b101110 – MSB vs LSB
    ([1, 0, 0, 1, 1, 1, 0, 0, 1, 0], 626, 313),
]
#                ↑arr                 ↑int (MSB view)   ↑int (LSB view)

@pytest.mark.parametrize("bits_msb,int_msb,int_lsb", cases)
def test_bitstring_lsb_arrays(bits_msb, int_msb, int_lsb):
    nbits = len(bits_msb)


    bita = BitStringView.from_array(bits_msb, nbits=nbits)
    bitb = BitStringView.from_int(int_msb, nbits=nbits)

    bitc = BitStringView.from_array(
        bits_msb,
        nbits=nbits,
        numbering=BitNumbering.LSB
    )
    assert bita == bitb                    
    assert bita.integer == int_msb
    assert bita.array() == bits_msb
    assert bitb.integer == int_msb
    assert bitb.array() == bits_msb

    assert bitc.integer == int_lsb
    assert bitc.array() == bits_msb               
    assert bitc.binary() == bita.binary()      
       
@pytest.mark.parametrize("value", range(15))
def test_conversion_ints(value):
    bita = BitStringView.from_int(value)                       
    bita_lsb = BitStringView.from_int(value, numbering=BitNumbering.LSB)
    bita_converted = BitStringView.from_bitstring(bita_lsb,numbering=BitNumbering.MSB)       
    assert bita == bita_converted, f'bita: {bita.info()}, bita_converted: {bita_converted.info()}'


array_cases = [
    [0, 0, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 0, 1, 1, 1, 0],
    [1, 0, 0, 1, 1, 1, 0, 0, 1, 0],
]

@pytest.mark.parametrize("bits", array_cases)
def test_conversion_arrays(bits):
    nbits = len(bits)
    bita = BitStringView.from_array(bits, nbits=nbits)     
    bita_lsb = BitStringView.from_bitstring(bita, numbering=BitNumbering.LSB)

    assert bita_lsb.array() == list(reversed(bits)), f'bitsa_lsb: {bita_lsb.info()}, array: {bita_lsb.array()}'   
    assert bita.binary() == bita_lsb.binary()[::-1]           
    assert bita.integer == bita_lsb.integer                         

# def test_bitstringview_conversion_helpers():
#     """
#     Convert LSB -> MSB via from_bitstring
#     Convert MSB -> LSB via with_numbering
#     """
#     for i in range(15):
#         b_msb = BitStringView.from_int(integer=i)
#         assert b_msb.integer == i, f"integer mismatch @ {i}"
       
#         assert b_msb.numbering == BitNumbering.MSB

#         b_lsb = b_msb.with_numbering(BitNumbering.LSB)         
#         b_round = BitStringView.from_bitstring(b_lsb, numbering=BitNumbering.MSB)
#         assert b_round.numbering is BitNumbering.MSB
#         assert b_round == b_msb

#     # round-trip through array example
   

#     arr = [1,0,1,0,1]
#     b_msb = BitStringView.from_array(arr, numbering=BitNumbering.MSB)

#     b_lsb = BitStringView.from_bitstring(b_msb, numbering=BitNumbering.LSB)
#     assert b_lsb.array() == arr[::-1]            # order flipped logically
#     assert b_msb.binary() == b_lsb.binary()[::-1]
#     assert b_msb.integer == b_lsb.integer
#     # assert b_back.bits[::-1] == b_lsb.bits()

# @pytest.mark.parametrize("arr,int_msb,int_lsb", [
#     ([0,0,1],                    1,   4),
#     ([1,0,0],                    4,   1),
#     ([1,0,1],                    5,   5),
#     ([1,0,1,1,1,0],             46,  23),
#     ([1,0,0,1,1,1,0,0,1,0],    0b1001110010, 0b0100111001),
# ])
# def test_bitstringview_lsb(arr, int_msb, int_lsb):
#     nbits = len(arr)

#     b_msb = BitStringView.from_array(arr, nbits=nbits, numbering=BitNumbering.MSB)
#     b_lsb = BitStringView.from_array(arr, nbits=nbits, numbering=BitNumbering.LSB)

#     # MSB ↔ expected integer
#     assert b_msb.integer == int_msb
#     # LSB interprets same *visual* array reversed in weight
#     assert b_lsb.integer == int_lsb

#     # Array order is identical by construction
#     assert b_msb.array() == arr
#     assert b_lsb.array() == arr

#     # Binary strings match when you reverse the LSB one
#     assert b_msb.binary() == b_lsb.binary()[::-1]

#     # Numbering difference ⇒ objects are *not* equal
#     assert b_msb != b_lsb
# def test_bitstringview_lsb():
#     for i in range(15):
#         b = BitStringView.from_int(integer=i)
#         b_lsb = BitStringViewLSB.from_int(integer=i)
#         assert b.integer == i
#         assert b.binary() == format(i, 'b')
#         assert b.binary() == bin(i)[2:]
#         assert b_lsb.integer == b.integer
#         assert b != b_lsb

#     arrays = [
#         [0, 0, 1],
#         [1, 0, 0],
#         [1, 0, 1],
#         [1, 0, 1, 1, 1, 0],
#         [1, 0, 0, 1, 1, 1, 0, 0, 1, 0],
#     ]
#     integers = [
#         1,
#         4,
#         5,
#         32 + 8 + 4 + 2,
#         2 + 16 + 32 + 64 + 512,
#     ]
#     integers_lsb = [
#         4,
#         1,
#         5,
#         1 + 4 + 8 + 16,
#         1 + 8 + 16 + 32 + 256,
#     ]

#     for idx, arr in enumerate(arrays):
#         nbits = len(arr)

#         bita = BitStringView.from_array(array=arr, nbits=nbits)
#         bitb = BitStringView.from_int(integer=integers[idx], nbits=nbits)
#         bitc = BitStringViewLSB.from_array(array=arr, nbits=nbits)

#         assert bita == bitb
#         assert bita.integer == integers[idx]
#         assert bita.array() == arr
#         assert bitb.integer == integers[idx]
#         assert bitb.array() == arr

#         assert bitc.integer == integers_lsb[idx]
#         assert bitc.array() == arr
#         assert bitc.binary() == bita.binary()







# def test_bitstringview_msb_equivalence():
#     for i in range(15):
#         bs = BitString.from_int(i)
#         assert bs.integer == i
#         assert bs.binary == format(i, 'b').zfill(bs.nbits)
#         assert bs.binary == bin(i)[2:].zfill(bs.nbits)

#     arrays = [
#         [0,0,1],
#         [1,0,0],
#         [1,0,1],
#         [1,0,1,1,1,0],
#         [1,0,0,1,1,1,0,0,1,0],
#     ]
#     integers = [1,4,5,32+8+4+2, 2+16+32+64+512]

#     for arr, intval in zip(arrays, integers):
#         bs_a = BitString.from_array(arr)
#         bs_b = BitString.from_int(intval, nbits=len(arr))
#         assert bs_a == bs_b
#         assert bs_a.array == arr
#         assert bs_b.integer == intval


# def test_bitstring_lsb_equivalence():
#     arrays = [
#         [0,0,1],
#         [1,0,0],
#         [1,0,1],
#         [1,0,1,1,1,0],
#         [1,0,0,1,1,1,0,0,1,0],
#     ]
#     ints_msb = [1,4,5,32+8+4+2, 2+16+32+64+512]
#     ints_lsb = [4,1,5,1+4+8+16, 1+8+16+32+256]

#     for arr, im, il in zip(arrays, ints_msb, ints_lsb):
#         nbits = len(arr)

#         bs_msb = BitString.from_array(arr)
#         bs_lsb = BitStringLSB.from_array(arr)
#         assert bs_msb.array == arr
#         assert bs_lsb.array == arr
#         assert bs_msb.integer == im
#         assert bs_lsb.integer == il
#         # same visible binary string regardless of numbering
#         assert bs_msb.binary == bs_lsb.binary
