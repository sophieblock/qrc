from numpy import inf as INFINITY


NEXT_AVAILABLE_RESOURCE_ID = 0

def get_next_device_id():
    global NEXT_AVAILABLE_RESOURCE_ID
    next_id = NEXT_AVAILABLE_RESOURCE_ID
    NEXT_AVAILABLE_RESOURCE_ID+=1
    return "R" + format(next_id, "X").zfill(7)

class Allocation:
    CLASSICAL = "CLASSICAL"
    QUANTUM = "QUANTUM"

    accepted_types = [CLASSICAL, QUANTUM]

    def __init__(
        self,
        device_name,
        device_type,
    ):
        self.device_name: str = device_name
        self.device_type: str = device_type

        assert (
            self.device_type in self.accepted_types
        ), f"processor_type must be either 'Classical' or 'Quantum'"


class Resource:
    CLASSICAL = "CLASSICAL"
    QUANTUM = "QUANTUM"

    accepted_types = [CLASSICAL, QUANTUM]

    def __init__(self, resource_type):
        assert (
            resource_type in self.accepted_types
        ), f"resource_type must be either 'Classical' or 'Quantum'"
        self.resource_type = resource_type


class Device:
    CLASSICAL = "CLASSICAL"
    QUANTUM = "QUANTUM"

    accepted_types = [CLASSICAL, QUANTUM]

    def __init__(self, device_name, device_type):

        assert (
            device_type in self.accepted_types
        ), f"device_type must be either 'Classical' or 'Quantum'"
        self.name: str = device_name or get_next_device_id()
        self.device_type: str = device_type

    def allocate(self, resource: Resource) -> Allocation:

        raise NotImplementedError

    def deallocate(self, allocation: Allocation):

        raise NotImplementedError