"""

This file contains class definitions for the broker which handles resource
allocation. The broker determines how to allocate both available
classical and quantum hardware resources to most efficiently execute
a problem or subproblem
"""
from copy import deepcopy
from .resources.resources import *
from .resources.classical_resources import *
from .resources.quantum_resources import *

from networkx.utils import graphs_equal
from typing import List


class Broker:
    def __init__(
        self,
        classical_devices: List[ClassicalDevice] | None = None,
        quantum_devices:  List[QuantumDevice]  | None = None,
    ):
        """
        A Broker always starts with *fresh* device instances:

        • If the supplied device object is a singleton that might be reused
          elsewhere, we deep-copy it so that state (e.g. qubit availability)
          is not shared between brokers.

        • Every device – copied or original – is reset so all qubits become
          available before the broker hands out allocations.
        """
        self.classical_devices = [
            self._clone_and_reset(dev) for dev in (classical_devices or [])
        ]
        self.quantum_devices = [
            self._clone_and_reset(dev) for dev in (quantum_devices or [])
        ]
    @staticmethod
    def _clone_and_reset(dev: Device):
        # deep-copy only if the project marks the class as a singleton
        dev_copy = deepcopy(dev) if getattr(dev, "_is_singleton", False) else dev

        # most of your Device classes already expose `.reset()`
        if hasattr(dev_copy, "reset"):
            dev_copy.reset()           # type: ignore[attr-defined]

        return dev_copy
    def request_allocation(self, resource: Resource) -> Allocation:
        if resource.resource_type == resource.CLASSICAL:
            device = self.get_optimal_classical_device(resource)
        else:
            device = self.get_optimal_quantum_device(resource)

        return device.allocate(resource)

    def get_optimal_classical_device(
        self, resource: ClassicalResource
    ) -> ClassicalDevice:
        highest_clock_frequency = 0
        optimal_device = None

        for device in self.classical_devices:
            if (
                device.check_if_available(resource)
                and device.clock_frequency > highest_clock_frequency
            ):
                optimal_device = device
                highest_clock_frequency = device.clock_frequency
        assert optimal_device != None, "No available devices"

        return optimal_device

    def get_optimal_quantum_device(self, resource: QuantumResource):
        max_connections = 0
        optimal_device = None

        supported_devices = self.get_supported_quantum_devices(resource)

        assert (
            len(supported_devices) != 0
        ), f"Broker does not contain any quantum devices that supports the required gate set in {resource.circuit}. "

        for device in supported_devices:
            if (
                device.check_if_available(resource)
                and device.max_available_connections > max_connections
            ):
                optimal_device = device
                max_connections = device.max_available_connections

        assert optimal_device != None, "No available devices"
        return optimal_device

    def get_supported_quantum_devices(
        self, resource: QuantumResource
    ) -> List[QuantumDevice]:
        circuit_gateset = resource.circuit.gate_set

        supported_devices = []
        logger.debug(f"Checking devices that support gateset: {circuit_gateset}")
        for device in self.quantum_devices:
            logger.debug(f"Device {device.name} with gateset: {device.gate_set}")
            

            if (
                device.connectivity.number_of_nodes()
                >= resource.circuit.qubit_count
            ):
                supported_devices.append(device)

        return supported_devices
    

    # def get_supported_quantum_devices(
    #     self, resource: QuantumResource
    # ) -> List[QuantumDevice]:
    #     circuit_gateset = resource.circuit.gate_set

    #     supported_devices = []
    #     logger.debug(f"Checking devices that support gateset: {circuit_gateset}")
    #     for device in self.quantum_devices:
    #         logger.debug(f"Device {device.name} with gateset: {device.gate_set}")
    #         if self.device_supports_circuit_gateset(device, circuit_gateset):
    #             if (
    #                 device.connectivity.number_of_nodes()
    #                 >= resource.circuit.qubit_count
    #             ):
    #                 supported_devices.append(device)

    #     return supported_devices

    def device_supports_circuit_gateset(
        self, device: QuantumDevice, circuit_gateset: List[QuantumGate]
    ):
        return all(gate in device.gate_set for gate in circuit_gateset)

    def request_deallocation(self, allocation: Allocation):
        if allocation.device_type == Allocation.CLASSICAL:
            device = self.get_classical_device(allocation)
        elif allocation.device_type == Allocation.QUANTUM:
            device = self.get_quantum_device(allocation)
        device.deallocate(allocation)

    def get_classical_device(self, allocation: ClassicalAllocation) -> Device:
        for device in self.classical_devices:
            conditions = (
                device.name == allocation.device_name,
                device.clock_frequency == allocation.clock_frequency,
            )
            if all(conditions):
                return device

        raise ValueError(
            f"Cannot find classical device allocation {allocation.device_name} in Broker {self}"
        )

    def get_quantum_device(self, allocation: QuantumAllocation) -> Device:
        for device in self.quantum_devices:
            conditions = (
                device.name == allocation.device_name,
                graphs_equal(device.connectivity, allocation.device_connectivity),
            )

            if all(conditions):
                return device

        raise ValueError(
            f"Cannot find quantum device allocation {allocation.device_name} in Broker {self}"
        )

    def __add__(self, other):
        assert isinstance(other, Broker), f"{other} must be a Broker object"
        classical_devices = list(set(self.classical_devices + other.classical_devices))
        quantum_devices = list(set(self.quantum_devices + other.quantum_devices))
        return Broker(
            classical_devices=classical_devices, quantum_devices=quantum_devices
        )
    def get_max_value(self,key):
        if key == 'Memory [B]':
            return max([x.RAM for x in self.classical_devices])
        if key == "Qubits Used":
            return max([len(x.available_qubits) for x in self.quantum_devices]) 
        
    @staticmethod
    def _clone_and_reset(dev: Device):
        # deep-copy only if the project marks the class as a singleton
        dev_copy = deepcopy(dev) if getattr(dev, "_is_singleton", False) else dev

        # most of your Device classes already expose `.reset()`
        if hasattr(dev_copy, "reset"):
            dev_copy.reset()           # type: ignore[attr-defined]

        return dev_copy