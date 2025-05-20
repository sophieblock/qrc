ep_packages = {
  "qiskit.transpiler.layout": [
    "default = qiskit.transpiler.preset_passmanagers.builtin_plugins:DefaultLayoutPassManager",
    "dense   = qiskit.transpiler.preset_passmanagers.builtin_plugins:DenseLayoutPassManager",
    "sabre   = qiskit.transpiler.preset_passmanagers.builtin_plugins:SabreLayoutPassManager",
    "trivial = qiskit.transpiler.preset_passmanagers.builtin_plugins:TrivialLayoutPassManager",
  ],
  "qiskit.transpiler.routing": [
    "basic      = qiskit.transpiler.preset_passmanagers.builtin_plugins:BasicSwapPassManager",
    "lookahead  = qiskit.transpiler.preset_passmanagers.builtin_plugins:LookaheadSwapPassManager",
    "none       = qiskit.transpiler.preset_passmanagers.builtin_plugins:NoneRoutingPassManager",
    "sabre      = qiskit.transpiler.preset_passmanagers.builtin_plugins:SabreSwapPassManager",
    "stochastic = qiskit.transpiler.preset_passmanagers.builtin_plugins:StochasticSwapPassManager",
  ],
  # add other groups if needed…
}

import pkg_resources
default_iter = pkg_resources.iter_entry_points

def hooked_iter(group, name=None):
    if group in ep_packages:
        for ep_def in ep_packages[group]:
            ep = pkg_resources.EntryPoint.parse(ep_def)
            ep.dist = pkg_resources.Distribution()  # satisfy the API
            yield ep
    else:
        yield from default_iter(group, name)

pkg_resources.iter_entry_points = hooked_iter