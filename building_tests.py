from attrs import frozen, field,validators
from qualtran import (
    Bloq, Signature, QAny,QBit,QDType, BloqBuilder, SoquetT, Soquet, CompositeBloq, DecomposeTypeError
)
from qualtran._infra.data_types import SymbolicInt
from typing import Callable, Optional,Dict, Any
from qualtran._infra.data_types import QDType, QAny,QBit
from attrs import define, frozen, validators
from functools import cached_property
from qualtran.drawing import show_bloq

import numpy as np
from attrs import define, frozen, field
from functools import cached_property
from typing import Dict, Any

from qualtran import Bloq, Signature, BloqBuilder, CompositeBloq
from qualtran._infra.registers import Register, Side
from qualtran._infra.data_types import QAny, QDType



@frozen(repr=False)
class Atom(Bloq):
    """An atomic bloq useful for testing (like in qualtran’s examples)."""
    tag: Optional[str] = None

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(q=1)

    def decompose_bloq(self) -> 'CompositeBloq':
        # This atom is declared "atomic" for demonstration
        raise DecomposeTypeError(f"{self} is atomic")

    def __repr__(self):
        if self.tag:
            return f"Atom({self.tag!r})"
        return "Atom()"
class NParallelCombo(Bloq):
    """A bloq that runs N copies of another sub-bloq in parallel.

    Args:
        n: How many parallel copies.
        make_subbloq: A callable returning the Bloq to be placed in each wire;
            defaults to a TestAtom if not given.
    """

    def __init__(
        self,
        n: SymbolicInt,
        make_subbloq: Callable[[], Bloq] = lambda: Atom()
    ):
        self.n = n
        self.make_subbloq = make_subbloq

    @cached_property
    def signature(self) -> Signature:
        # This says we have 1 register named 'reg' of bitsize = n,
        # so effectively QAny(n).  Just like TestParallelCombo used bitsize=3.
        return Signature.build(reg=self.n)
    def build_composite_bloq(self, bb: BloqBuilder, **soqs) -> Dict[str, Any]:
        # 1) find the single left input name from signature
        left_reg = next(iter(self.signature.lefts()))
        in_soq = soqs[left_reg.name]

        # 2) split
        splitted = bb.split(in_soq)

        # 3) sub-bloq in parallel
        for i in range(len(splitted)):
            splitted[i] = bb.add(self.make_subbloq(i), x=splitted[i])

        # 4) re-join
        out_soq = bb.join(splitted)

        # 5) store in single right register
        right_reg = next(iter(self.signature.rights()))
        soqs[right_reg.name] = out_soq
        return soqs
    # def build_composite_bloq(self, bb: BloqBuilder, **soqs) -> Dict[str, SoquetT]:
    #     # Retrieve the 'reg' soquet from the variadic arguments
    #     reg = soqs["reg"]
    #     print(f"reg: {reg}")

    #     # Split the single big register into n single-bit wires
    #     # (or n sub-registers of size 1).
    #     reg = bb.split(reg)  # now `reg` is an array of shape=(n,)
    #     print(f"split_ports: {reg}")

    #     # Add the sub-bloq in parallel on each wire
    #     for i in range(len(reg)):
    #         reg[i] = bb.add(self.make_subbloq(), q=reg[i])

    #     # Finally, join them back into a single bitsize=n register
    #     return {"reg": bb.join(reg)}



#
# 1) A child Bloq: AddConstantSpecOp
#
@frozen
class AddConstantSpecOp(Bloq):
    dtype: QDType = field(default=QAny(4))
    constant: int = 5

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register('x', self.dtype, side=Side.THRU)])

    def decompose_bloq(self) -> 'CompositeBloq':
        raise DecomposeTypeError(f"{self} is atomic")

    def on_classical_vals(self, **vals: Any) -> Dict[str, Any]:
        x_in = vals['x']
        modval = 1 << self.dtype.num_qubits
        return {'x': (x_in + self.constant) % modval}


#
# 2) A flexible parallel Bloq
#
@frozen
class NParallelCombo_Dtype(Bloq):
    input_dtype: QDType = field(default=QAny(2))

    @cached_property
    def signature(self) -> Signature:
        return Signature.build_from_dtypes(x=self.input_dtype)

    def make_subbloq(self, i: int) -> Bloq:
        # By default, do Atom()—or override in a subclass.
        # If you have 'Atom' in test_bloqs, you can do:
        from qualtran.examples.test_bloqs import Atom
        return Atom()

    def build_composite_bloq(self, bb: BloqBuilder, **soqs) -> Dict[str, Any]:
        # 1) find the single left input name from signature
        left_reg = next(iter(self.signature.lefts()))
        in_soq = soqs[left_reg.name]

        # 2) split
        splitted = bb.split(in_soq)

        # 3) sub-bloq in parallel
        for i in range(len(splitted)):
            splitted[i] = bb.add(self.make_subbloq(i), x=splitted[i])

        # 4) re-join
        out_soq = bb.join(splitted)

        # 5) store in single right register
        right_reg = next(iter(self.signature.rights()))
        soqs[right_reg.name] = out_soq
        return soqs


#
# 3) A variant that uses AddConstantSpecOp
#
@frozen
class NParallelCombo_Dtype_AddConstant(NParallelCombo_Dtype):
    constant: int = 5

    def make_subbloq(self, i: int) -> Bloq:
        return AddConstantSpecOp(dtype=self.input_dtype, constant=self.constant)


    
if __name__ == '__main__':
    # A. Basic usage with default (Atom) sub-bloq
    my_combo = NParallelCombo_Dtype(QAny(2))
    show_bloq(my_combo, "dtype")
    cbloq = my_combo.decompose_bloq()
    # show_bloq(cbloq, "my_combo_decomposed")

    # B. Using the AddConstantSpecOp sub-bloq
    add_combo = NParallelCombo_Dtype_AddConstant(input_dtype=QAny(3), constant=7)
    # show_bloq(add_combo, "add_combo")
    add_cbloq = add_combo.decompose_bloq()
    # show_bloq(add_cbloq, "add_combo_decomposed")
    # n = 2
    # parallel_combo = NParallelCombo(n=n)
    # print(f'parallel_combo.signature: {parallel_combo.signature}')
    # show_bloq(parallel_combo, 'dtype')
    # print(parallel_combo.as_composite_bloq().debug_text())
    # cbloq = parallel_combo.decompose_bloq()
    # show_bloq(cbloq, 'dtype')
    # print(f"signature: \n{cbloq.signature}")
    # print("debug_text:")
    # print(cbloq.debug_text())
