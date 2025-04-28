from ..process import Process, ClassicalProcess
from ..data import Data,DataSpec
from ..register import RegisterSpec, Signature, Flow
from ..dtypes import *
from ..builder import ProcessBuilder, Port, PortT,CompositeMod

from attrs import define, frozen
from functools import cached_property
from typing import Union, Optional, Dict

@define(repr=False)
class Atom(Process):
    """An atomic process for basic testing."""
    tag: Optional[str] = None

    
    @cached_property
    def signature(self) -> Signature:
       
        return Signature.build(n=1)
    
    def validate_data(self) -> bool:
        """Simple validation for test cases."""
        
        return all(data.properties["Data Type"] in [CBit, int] for data in self.inputs)
    

    def set_expected_input_properties(self):
        """Sets the expected input properties."""
        self.expected_input_properties = [
            {"Data Type": [CBit, int]}  # Allow specific DataTypes
        ]

    def __repr__(self):
        if self.tag:
            return f'Atom({self.tag!r})'
        else:
            return 'Atom()'

    def __str__(self):
        if self.tag:
            return f'Atom({self.tag!r})'
        return 'Atom'
    
@define(repr=False)
class Atom_n(Process):
    """An atomic process for basic testing."""
    n: Union[int, SymInt]

    def __init__(self, n: Union[int, SymInt], **kwargs):
        # Set the n attribute before calling the parent constructor.
        self.n = n
        super().__init__(**kwargs)  # Handles input validation, etc.
        
    @cached_property
    def signature(self) -> Signature:
        # Wrap n in a CBit to ensure compatibility
        return Signature.build_from_dtypes(n=CInt(bit_width=self.n))

    def validate_data(self) -> bool:
        """Simple validation for test cases."""
        
        return all(data.properties["Data Type"] in [CInt, CBit, int] for data in self.inputs)
    

    def set_expected_input_properties(self):
        """Sets the expected input properties."""
        self.expected_input_properties = [
            {"Data Type": [CInt, CBit, int]}  # Allow specific DataTypes
        ]

    def __repr__(self):
        return f'Atom({self.n})'
    
    def __str__(self):
        return f'Atom({self.n})'
    

@define
class AtomChain(Process):
    """Chain multiple `Atom` processes to run in sequence."""

   
    @cached_property
    def signature(self) -> Signature:
        return Signature.build(reg=1)

    def build_composite(self, builder: 'ProcessBuilder', reg: 'PortT') -> Dict[str, 'PortT']:
        for i in range(3):
            reg = builder.add(Atom(tag=f'atom{i}'), n=reg)
        return {'reg': reg}


    def set_expected_input_properties(self):
        """Define the expected input properties for the chained Atoms."""
        self.expected_input_properties = [
            {"Data Type": [CBit, int]}  # Define acceptable input types for the chain
        ]

    def validate_data(self) -> bool:
        """Ensures the input data can run correctly."""
        return all(input_data.properties['Data Type'] == CBit for input_data in self.inputs)



@define
class NAtomParallel(Process):
    """Parallel composition of Atom processes."""
    num_parallel: int = 1  # Declare as a field so __init__ is auto-generated

    @cached_property
    def signature(self) -> Signature:
        # This says we have 1 register named 'reg' of bitsize = n,
        # so effectively QAny(n).  Just like TestParallelCombo used bitsize=3.
        return Signature.build(bag=self.num_parallel)
   
    def build_composite(self, builder: ProcessBuilder, **ports: PortT) -> Dict[str, PortT]:

        left_reg = next(iter(self.signature.lefts()))
        reg = ports[left_reg.name]
        print(f"reg: {reg}")
        # Split the input register into individual bits
        # split_ports = builder.split(reg, self.num_parallel)
        logger.debug(f"reg: {reg}")
        # Split the single big register into n single-bit wires
        # (or n sub-registers of size 1).
        reg = builder.split(reg)  # now `reg` is an array of shape=(n,)
        logger.debug(f"split_ports: {reg}")
        # Add the sub-bloq in parallel on each wire
        for i in range(len(reg)):
            # reg[i] = builder.add(Atom(tag=f'b{i}'), n=reg[i])
            reg[i] = builder.add(Atom(), n=reg[i])

        # Finally, join them back into a single bitsize=n register
        # and place it into the single right register
        right_reg = next(iter(self.signature.rights()))
        return {right_reg.name: builder.join(reg)}

class SimpleAddProcess(ClassicalProcess):
    def set_expected_input_properties(self):
        # We'll define two integer inputs
        self.expected_input_properties = [
            {"Data Type": int, "Usage": "x"},
            {"Data Type": int, "Usage": "y"},
        ]
    def set_output_properties(self):
        # We'll define one integer output
        self.output_properties = [
            {"Data Type": int, "Usage": "OUT"}
        ]
    def validate_data(self) -> bool:
        return True
    def update(self):
        # Pretend we do x+y
        self.result = self.inputs[0].data + self.inputs[1].data
    def generate_output(self) -> list[Data]:
        return [Data(self.result, properties={"Usage": "OUT", "Data Type": int})]
#


@define
class TwoBitOp(ClassicalProcess):
    """ Maybe a + b = c FOR CLASSICLA BITS """
    @cached_property
    def signature(self) -> Signature:
        return Signature.build(a=1, b=1)
    def set_expected_input_properties(self):
        """Sets the expected input properties."""
        self.expected_input_properties = [
            {"Data Type": [CBit,int]}  # Allow specific DataTypes
        ]*2

    def decompose_bloq(self) -> 'CompositeMod':
        raise TypeError(f"{self} is atomic")
    


@define
class SwapTwoBit(ClassicalProcess):
    @cached_property
    def signature(self) -> Signature:
        return Signature.build(d1=1, d2=1)

    def build_composite(
        self, builder: 'ProcessBuilder', d1: 'Port', d2: 'Port'
    ) -> Dict[str, PortT]:
        d1, d2 = builder.add(TwoBitOp(), a=d1, b=d2)
        d1, d2 = builder.add(TwoBitOp(), a=d2, b=d1)
        return {'d1': d1, 'd2': d2}


@define
class SplitJoin(Process):
    n: int 

    @property
    def signature(self):
        return Signature([RegisterSpec('x', dtype=TensorType((self.n,)))])
    def build_composite(self, builder, *,x: 'Port'):
        # print(f"x: {x.reg}")
        xs = builder.split(x)
        x =  builder.join(xs, dtype=x.reg.dtype)
        return {'x':x}
