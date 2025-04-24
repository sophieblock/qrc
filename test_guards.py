# Owner(s): ["module: dynamo"]
import functools
import weakref

import torch
import torch._dynamo

from torch._C._dynamo import guards
from torch._dynamo.convert_frame import GlobalStateGuard



RootGuardManager = guards.RootGuardManager
DictGuardManager = guards.DictGuardManager
DictSubclassGuardManager = guards.DictSubclassGuardManager
GetAttrGuardAccessor = guards.GetAttrGuardAccessor
GetItemGuardAccessor = guards.GetItemGuardAccessor
TypeGuardAccessor = guards.TypeGuardAccessor
OBJECT_ALIASING = guards.OBJECT_ALIASING
install_object_aliasing_guard = guards.install_object_aliasing_guard
NO_TENSOR_ALIASING = guards.NO_TENSOR_ALIASING
install_no_tensor_aliasing_guard = guards.install_no_tensor_aliasing_guard



x = torch.tensor(4)
weakref_x = weakref.ref(x)

default_mgr_enum = torch._dynamo.guards.GuardManagerType.GUARD_MANAGER


class Pair:
    def __init__(self, x, y):
        self.x = x
        self.y = y


global_pair = Pair(torch.randn(4), 1)


def id_type(x):
    return id(type(x))


def equals_match(x, expected):
    return x == expected


def equals_match_verbose_code_parts(expected):
    return [f"x == {expected}"]


def ge_match(x, expected):
    return x >= expected


def ge_match_verbose_code_parts(expected):
    return f"expected >= {expected}"


def less_match(x, expected):
    return x < expected


def less_match_verbose_code_parts(expected):
    return [f"expected < {expected}"]

def test_rootguard_manager():
    guard_manager = RootGuardManager()
    guard_manager.add_type_match_guard(id_type(5), ["type(x) == int"])
    guard_manager.add_lambda_guard(
        functools.partial(ge_match, expected=5),
        ge_match_verbose_code_parts(expected=5),
    )
    print(guard_manager.get_leaf_guards())
if __name__ == "__main__":
    # test_rootguard_manager()
    test_dynamic_indices_guard()
    # test_global_state_guard()