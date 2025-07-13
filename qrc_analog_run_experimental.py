import warnings

# Suppress PennyLane‐vs‐JAX compatibility warnings
warnings.filterwarnings(
    "ignore",
    # message=".*PennyLane is (not yet compatible|currently not compatible) with JAX versions > 0.4.28.*",
    category=RuntimeWarning,
)
import pennylane as qml
import os
import numpy
import pickle
import re
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from qiskit import *
from jax import numpy as np
import sympy
import matplotlib.pyplot as plt
import base64
from jax import numpy as jnp
import pickle

from datetime import datetime
 # Using pennylane's wrapped numpy
from sympy import symbols, MatrixSymbol, lambdify, Matrix, pprint
import jax
import numpy as old_np
from jax import random
import scipy
import pickle
import base64
import time
import os
from jax import Array
import ast
import pandas as pd
from pathlib import Path
from qiskit.circuit.library import *
from qiskit import *
from qiskit.quantum_info import *
import autograd
from pennylane.wires import Wires
import matplotlib.cm as cm
from functools import partial
from pennylane import numpy as penny_np
from jax import config
import optax
from pennylane.transforms import transform
from typing import Sequence, Callable, Union, List
from itertools import chain
from functools import partial, singledispatch
from pennylane.circuit_graph import LayerData
from pennylane.queuing import WrappedObj
from pennylane.transforms import transform
from pennylane.operation import Operation, AnyWires
from pennylane.ops import PauliRot
from pennylane.operation import (
    has_gen,
    gen_is_multi_term_hamiltonian,
    has_grad_method,
    has_nopar,
    has_unitary_gen,
    is_measurement,
    is_trainable,
    not_tape,
)
from jax import jit
import pennylane as qml
from pennylane.operation import AnyWires, Operation

#from pennylane.pulse import ParametrizedEvolution, ParametrizedHamiltonian,HardwareHamiltonian
from jax.experimental.ode import odeint
from pennylane.devices.qubit.apply_operation import _evolve_state_vector_under_parametrized_evolution,apply_parametrized_evolution
has_jax = True
diable_jit = False
config.update('jax_disable_jit', diable_jit)
#config.parse_flags_with_absl()
config.update("jax_enable_x64", True)
os.environ['JAX_TRACEBACK_FILTERING'] = 'off'
def quantum_fun(gate, input_state, qubits):
    '''
    Apply the gate to the input state and return the output state.
    '''
    qml.StatePrep(input_state, wires=[*qubits])
    gate(wires=qubits)
    return qml.density_matrix(wires=[*qubits])
def generate_dataset(
    gate, n_qubits, training_size, key, new_set=False
):
    """
    Generate a deterministic dataset of input and output states for a given gate.

    Parameters:
        gate: The quantum gate to apply.
        n_qubits: Number of qubits in the system.
        training_size: Number of training states required.
        key: JAX random key for reproducibility.
        trot_step: (Optional) Trotter step for additional determinism.
        reservoir_count: (Optional) Reservoir count for additional determinism.
        new_set: If True, generate a new dataset even for the same parameters. Default is False.

    Returns:
        Tuple (input_states, output_states).
    """
    if new_set:
        # Use the raw key to generate a new dataset
        seed = int(jax.random.randint(key, (1,), 0, 2**32 - 1)[0])
    else:
       
        # Derive a deterministic seed that ignores trot_step and reservoir_count
        key_int = int(jax.random.randint(key, (1,), 0, 2**32 - 1)[0])
        seed = hash((n_qubits, key_int)) 

    # Generate random state vectors deterministically
    X = []
    for i in range(training_size):
        folded_key = jax.random.fold_in(jax.random.PRNGKey(seed), i)
        state_seed = int(jax.random.randint(folded_key, (1,), 0, 2**32 - 1)[0])
        state_vec = random_statevector(2**n_qubits, seed=state_seed).data
        X.append(np.asarray(state_vec, dtype=jnp.complex128))

    # Generate output states using the circuit
    qubits = Wires(list(range(n_qubits)))
    dev_data = qml.device("default.qubit", wires=qubits)
    circuit = qml.QNode(quantum_fun, device=dev_data, interface="jax")

    y = [np.array(circuit(gate, x, qubits), dtype=jnp.complex128) for x in X]

    return np.asarray(X), np.asarray(y)

def create_initial_state(num_qubits, base_state):
    """
    Create an initial state for a given number of qubits.
    """
    state = np.zeros(2**num_qubits)

    if base_state == 'basis_state':
        state = state.at[0].set(1)

    elif base_state == 'GHZ_state':
        state = state.at[0].set(1 / np.sqrt(2))
        state = state.at[-1].set(1 / np.sqrt(2))

    return state


def get_rate_of_improvement(cost, prev_cost,second_prev_cost):
    
    prev_improvement = prev_cost - second_prev_cost
    current_improvement = cost - prev_cost
    acceleration = prev_improvement - current_improvement

    return acceleration


def get_base_learning_rate(grads, scale_factor=.1, min_lr=1e-5, max_lr=0.2):
    """Estimate a more practical initial learning rate based on the gradient norms."""
    grad_norm = jnp.linalg.norm(grads)
    
    initial_lr = jnp.where(grad_norm > 0, scale_factor / grad_norm, 0.1)
    
    clipped_lr = jnp.clip(initial_lr, min_lr, max_lr)
    print(f"grad_norm: {grad_norm}, initial base lr: {initial_lr:.5f}, clipped: {clipped_lr:.5f}")
    return initial_lr, clipped_lr,grad_norm
def get_slicewise_lr_trees(
    params: jnp.ndarray,
    grads: jnp.ndarray,
    N_train: int,
    N_ctrl: int,
    N_reserv: int,
    time_steps: int,
    *,
    L_ref: int = 256,
    P_ref: int = 1024,
    eta_base: float = 0.1,
    eps: float = 1e-12,
    min_lr: float = 1e-6,
    max_lr: float = 0.5,
    debug: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute one learning‐rate per Trotter slice (τ_t and its J‐block) + one for h,
    then scatter back into a full-length vector lr_tree of shape [M].

    We follow a “LARS” recipe:
      (1) Count total params: M = T + 3 + T*(N_ctrl*N_reserv).
      (2) Compute a global base η₀ := eta_base * (N_train / L_ref) * sqrt(P_ref / N_params).
      (3) For each slice t=0..T-1, gather that slice’s parameters:
          w^{(t)} = [τ_t, {J_{kℓ}^{(t)}}_{k=0..N_ctrl-1, ℓ=0..N_reserv-1} ],
          and its gradients g^{(t)} likewise.
          Then set η^{(t)} = η₀ · (‖w^{(t)}‖₂ / (‖g^{(t)}‖₂ + eps)), clipped into [min_lr, max_lr].
      (4) For the global field h (3 parameters), gather w^{(h)} = [h^x,h^y,h^z], same for g^{(h)},
          and set η^{(h)} = η₀ · (‖w^{(h)}‖₂ / (‖g^{(h)}‖₂ + eps)), clipped.
      (5) Finally, “scatter” each η into all indices belonging to that slice (for τ_t and its J’s)
          or to the h‐block.

    Returns:
      - lr_tree: array of shape [M], listing the per‐param learning rate
      - assignment_mask: an integer array of shape [M], giving a “group ID” per index:
            0…(T−1)  → slice 0…slice T−1,
           T         → the “h” group
           (no other IDs)
        (Use `assignment_mask[i]` to see which slice/h‐group index i belongs to.)
    """
    M = grads.shape[0]
    num_J = N_ctrl * N_reserv

    # 1) Compute total # params and global base η₀
    N_params = M
    eta0 = eta_base * (N_train / L_ref) * jnp.sqrt(P_ref / float(N_params))
    # note: if eta0 > max_lr, we'll clip later at per‐group level

    # 2) Build an “assignment mask” of length M:
    #    indices 0..(time_steps-1)       → each τ_t (we’ll give it group ID t)
    #    indices time_steps..time_steps+2 → the 3‐vector h (we assign ID = time_steps)
    #    indices time_steps+3..M-1       → the J‐blocks, sliced in chunks of size num_J
    idx = jnp.arange(M)
    mask_tau    = idx < time_steps
    mask_h      = (idx >= time_steps) & (idx < time_steps + 3)
    mask_J_flat = idx >= (time_steps + 3)

    # build an integer array “group” of shape [M]:
    #   group[i] = t in [0..T-1] if i is in τ‐block for slice t
    #   group[i] = T      if i is in h‐block
    #   group[i] = t      if i is in J‐block for slice t (we must compute t from index)
    #
    # Indices for J‐blocks run from (time_steps+3) up to (time_steps+3 + T*num_J -1).
    def assign_group(i):
        # i: a scalar index in [0..M-1]
        #      if i < time_steps → group = i (slice ID)
        #      elif time_steps <= i < time_steps+3 → group = time_steps  (the “h” ID)
        #      else → group = floor( (i - (time_steps+3)) / num_J )  [in 0..T-1].
        return jax.lax.cond(
            i < time_steps,
            lambda i: i,                                # slice t = i
            lambda i: jax.lax.cond(
                i < time_steps + 3,
                lambda i: time_steps,                   # “h” group
                lambda i: ((i - (time_steps + 3)) // num_J),
                i
            ),
            i
        )
    group_ids    = jax.vmap(assign_group)(idx)  # shape [M], each entry in [0..T].
    assignment_mask = group_ids  # we’ll return this so downstream code knows the grouping.

    # 3) For each group in 0..T (where group=T means “h”), collect all grads/params belonging to that group,
    #    compute ‖w_group‖₂, ‖g_group‖₂, then η_group = η₀ * (‖w‖/ (‖g‖ + ε)), clipped.
    def compute_group_lr(gid):
        # gather indices where assignment_mask == gid
        idxs = jnp.where(group_ids == gid, size=M)[0]
        # slice out params, grads at those indices
        w_block = params[idxs]
        g_block = grads[idxs]
        w_norm  = jnp.linalg.norm(w_block)
        g_norm  = jnp.linalg.norm(g_block)
        raw_lr  = eta0 * (w_norm / (g_norm + eps))
        return jnp.clip(raw_lr, min_lr, max_lr)

    # vector‐map compute_group_lr over gid = 0..T
    all_group_ids = jnp.arange(time_steps + 1)  # 0..T-1 for slices, T for “h”
    group_lr_tree = jax.vmap(compute_group_lr)(all_group_ids)  # shape [T+1]

    # 4) Scatter back: each parameter i gets lr = group_lr_tree[group_ids[i]]
    lr_tree = group_lr_tree[group_ids]  # broadcasting by integer‐indexing

    if debug:
        # Print some debugging info:
        print(f"\n--- slice‐wise LARS debug ---")
        print(f" total params  M = {M},  base η₀ = {eta0:.3e}")
        for gid in range(time_steps):
            print(f"  slice {gid:2d}: η_slice = {float(group_lr_tree[gid]):.3e}")
        print(f"  h‐group: η_h = {float(group_lr_tree[time_steps]):.3e}")
        print(f" lr_tree stats:  min={float(jnp.min(lr_tree)):.3e},  max={float(jnp.max(lr_tree)):.3e},  mean={float(jnp.mean(lr_tree)):.3e}")
        print(f"------------------------------\n")

    return lr_tree, assignment_mask


from optax._src import base as optax_base
from reduce_on_plateau import reduce_on_plateau
from typing import NamedTuple
# ------------------------------------------------------------------------------
# TRUST-COEFFICIENT  (post-2022 consensus: 0.1 %–1 % of weight-norm)
# Paper trail: LARS/LAMB [You+’20], AGC [Brock+’21], LAMB-C [Fong+’22],
#             NGD for VQCs [Suzuki+’22], etc.
# We pick the mid-range 1 %  (c = 0.01) and let verify_c() assert sanity.
TRUST_COEFF_DEFAULT = 0.01        # NOTE: replaces the old hard-wired 0.75
# ------------------------------------------------------------------------------

def reduce_on_cost_threshold(threshold, factor=.25, min_scale=1e-5):
    class S(NamedTuple):
        scale: jnp.ndarray
        done:  jnp.ndarray

    def init(_):                      # params not needed here
        return S(jnp.asarray(1.), jnp.asarray(False))

    def upd(updates, state, params=None, *, value=None, **extras):
        """Optax expects (updates, state, params, **extra)."""
        fire = (value is not None) & (value <= threshold) & (~state.done)
        
        new_scale = jnp.where(fire,
                              jnp.maximum(state.scale * factor, min_scale),
                              state.scale)
        new_state = S(new_scale, state.done | fire)
        updates = jax.tree_util.tree_map(lambda g: g * new_scale, updates)
        return updates, new_state

    return optax.GradientTransformationExtraArgs(init, upd)

# ── 2) shrink when grad variance stalls ───────────────────────────────────────
def reduce_on_grad_variance(factor=.5, patience=10, window=50,
                            var_tol=1e-6, min_scale=1e-5):
    class S(NamedTuple):
        scale:  jnp.ndarray
        mean:   jnp.ndarray
        var:    jnp.ndarray
        streak: jnp.ndarray
    α = jnp.exp(-1./window)

    def init(_):
        return S(jnp.asarray(1.), jnp.asarray(0.), jnp.asarray(0.),
                 jnp.asarray(0, jnp.int32))

    def upd(updates, state, params=None, **extras):
        g_norm = _l2_vec(updates)
        mean   = α*state.mean + (1-α)*g_norm
        var    = α*state.var  + (1-α)*(g_norm - mean)**2
        streak = jnp.where(var < var_tol, state.streak + 1, 0)

        fire   = (streak >= patience) & (state.scale > min_scale)
        scale  = jnp.where(fire,
                           jnp.maximum(state.scale * factor, min_scale),
                           state.scale)
        new_state = S(scale, mean, var, streak)
        updates   = jax.tree_util.tree_map(lambda g: g * scale, updates)
        return updates, new_state

    return optax.GradientTransformationExtraArgs(init, upd)
# ──────────────────────────────────────────────────────────────────────────────
# utils – build masks + group dictionaries for LR / grad logging
# ──────────────────────────────────────────────────────────────────────────────

from optax._src.transform import ScaleByAdamState  # already in optax
# ----------------------------------------------------------------------
try:
    # newest optax
    from optax._src.masking import MaskedNode
except ImportError:
    # for older optax versions
    from optax.transforms._masking import MaskedNode  # type: ignore
def _l2(tree):
    return jnp.sqrt(sum(jnp.sum(jnp.square(x))
                        for x in jax.tree_util.tree_leaves(tree)))

def _l2_vec(vec: jnp.ndarray) -> float:
    """Return the ℓ₂-norm as Python float (helper – JIT friendly)."""
    return float(jnp.linalg.norm(vec))
def _pytree_l2(tree):
    """
    Compute the ℓ\_2-norm of **all numeric leaves** in a *PyTree* while
    ignoring special Optax placeholders.

    Parameters
    ----------
    tree : PyTree
        Arbitrary nested structure (dict / list / tuple) whose leaves are
        ``jax.Array`` objects **or** Optax‐specific sentinels.

    Returns
    -------
    float
        \[
            \|\mathrm{tree}\|\_2
            \;=\;
            \sqrt{ \sum\_{ℓ} \|x^{(ℓ)}\|\_2^{\,2} }
        \]
        where the sum runs over every leaf
        :math:`x^{(ℓ)}` **except**
        *``optax.MaskNode``* instances and plain Python scalars
        (those scalars enter as :math:`x^{(ℓ)} = \mathrm{float}`).

    Notes
    -----
    * Each ``jax.Array`` leaf contributes its full Frobenius norm
      :math:`\sqrt{\sum_{i} x_i^2}`.
    * Designed as a helper for
      :pyfunc:`log_moments`, hence the explicit skip of ``MaskedNode`` objects
      created by ``optax.masked`` transformations.
    """
    sq_sum = 0.0
    for leaf in jax.tree_util.tree_leaves(tree):
        # skip MaskedNode or empty placeholders
        if isinstance(leaf, MaskedNode):
            continue
        if jnp.isscalar(leaf):
            sq_sum += float(leaf) ** 2
        else:                                    # ndarray / Array
            sq_sum += float(jnp.sum(jnp.square(leaf)))
    return float(jnp.sqrt(sq_sum))
def _find_adam_state(state):
    """
    Recursively search *any* nested Optax state for the first
    :class:`optax.ScaleByAdamState`.

    Parameters
    ----------
    state : Any
        The optimiser state produced by an Optax gradient transformation
        (may be an arbitrarily deep tuple of named-tuples).

    Returns
    -------
    optax.ScaleByAdamState | None
        The *first* occurrence found in a depth-first traversal, or
        ``None`` if the subtree contains no Adam moments.

    Implementation detail
    ---------------------
    The Optax masking transform wraps inner states in a ``MaskedState`` whose
    `.inner_state` attribute points to the original
    :class:`optax.ScaleByAdamState`; this helper tunnels through such wrappers.
    """
    if isinstance(state, ScaleByAdamState):
        return state
    if isinstance(state, tuple):
        for s in state:
            found = _find_adam_state(s)
            if found is not None:
                return found
    if hasattr(state, "inner_state"):
        return _find_adam_state(state.inner_state)
    return None
def log_moments(opt_state, case_num, time_steps,epoch):
    """
    Extract and log the **block-wise** Adam first and second moments at a
    given training epoch.

    Parameters
    ----------
    opt_state : tuple | NamedTuple
        The full optimiser state returned by
        :pyfunc:`get_optimizer` for *any* ``case_num`` currently supported in
        the code-base.
    case_num : int
        Optimiser scheme identifier (cf.\ :pyfunc:`get_optimizer`):
        ``1–4`` = type-grouped Adam,
        ``6–7`` = slice-wise Adam.
    time_steps : int
        T (number of Trotter slices); needed to split flat moments into
        :math:`(\boldsymbol τ,\,\boldsymbol h,\,\boldsymbol J)` blocks when
        **case ≥ 6**.
    epoch : int
        Purely cosmetic — printed only in developer debug statements.

    Returns
    -------
    dict[str, tuple[float, float]]
        ``{"t": (‖m_t‖₂, ‖v_t‖₂), "h": (‖m_h‖₂, ‖v_h‖₂),
           "J": (‖m_J‖₂, ‖v_J‖₂)}``

        where :math:`m_B,\;v_B` are the *block-restricted* Adam moments at the
        current step.

    Raises
    ------
    RuntimeError
        If no :class:`optax.ScaleByAdamState` can be found inside
        ``opt_state`` (should never happen for supported cases).
    ValueError
        If ``case_num`` ∉ {1, 2, 3, 4, 6, 7}.
    
    """
    
    if case_num in [1,2,3,4]:
        _, t_s, h_s, J_s = opt_state   # unpack MaskedStates
        def _mv(masked_state):
            adam = _find_adam_state(masked_state)
            m, v = adam.mu, adam.nu
            assert m is not None
            
            return _pytree_l2(m), _pytree_l2(v)
        # print(f"epoch {epoch}: ")
        # print(f"t_s: {t_s}")
        # print(f"h: {h_s}")
        # print(f"J: {J_s}")
        return {"t": _mv(t_s), "h": _mv(h_s), "J": _mv(J_s)}

    if case_num in (6, 7):
        # opt_state = (clip_state, adam_state, scale_state) for case-6,
        # or (clip_state, *masked_adams) for case-7
        for branch in opt_state:
            adam = _find_adam_state(branch)
            if adam is not None:
                m, v = adam.mu, adam.nu
                break
        else:
            raise RuntimeError("No Adam state found!")

        T = time_steps
        return {
            "t": (float(jnp.linalg.norm(m[:T])),
                  float(jnp.linalg.norm(v[:T]))),
            "h": (float(jnp.linalg.norm(m[T:T+3])),
                  float(jnp.linalg.norm(v[T:T+3]))),
            "J": (float(jnp.linalg.norm(m[T+3:])),
                  float(jnp.linalg.norm(v[T+3:]))),
        }

    raise ValueError("moment logging only implemented for cases 1 and 6")
# ────────────────────────────────────────────────────────────────────
# 1) coefficient-of-variation per block, works for both case-sets
# ────────────────────────────────────────────────────────────────────
def coef_variation_by_block(case_num, grads_flat,
                            time_steps, N_ctrl, N_reserv):
    """
    Compute the *coefficient of variation* (CV\_B) for every parameter block.

    CV definition
    -------------
    For a block :math:`B` with gradient vector
    :math:`\boldsymbol g_B = (g_1,\dots,g_{|B|})`
    we define

    \[
        \mathrm{CV}_B
        \;=\;
        \frac{\sigma(|\boldsymbol g_B|)}
             {\mu(|\boldsymbol g_B|)}
        \;=\;
        \frac{\sqrt{\tfrac1{|B|}\sum_i (|g_i|-\mu_B)^2}}
             {\tfrac1{|B|}\sum_i |g_i| } .
    \]

    Parameters
    ----------
    grads_flat : jax.Array
        Flattened gradient vector **before** the optimiser update
        (shape = ``(M,)``).
    case_num : int
        Determines the grouping:
        ``≤ 5`` → type-wise {τ, h, J}; ``≥ 6`` → slice-wise + h.
    time_steps, N_ctrl, N_reserv : int
        Needed to slice ``grads_flat`` consistently with
        :pyfunc:`grad_group_dict`.

    Returns
    -------
    dict[str, float]
        Mapping ``block-name → CV_B`` where
        *block-name* ∈ {``"t","h","J"``} for type grouping or
        {``"h","slice_0",…``} for slice grouping.
    
    """
    blocks = grad_group_dict(case_num, grads_flat,
                             time_steps, N_ctrl, N_reserv)
    out = {}
    for name, vec in blocks.items():
        vec = jnp.asarray(vec)
        # mu  = jnp.mean(jnp.abs(vec))
        # sig = jnp.std(jnp.abs(vec))
        mu  = jnp.median(jnp.abs(vec))            # robust/GradNorm style
        sig = jnp.sqrt(jnp.median((jnp.abs(vec) - mu)**2) + 1e-12)
        out[name] = float(sig / (mu + 1e-12))
    return out

def verify_c(theta: jnp.ndarray,
             grads: jnp.ndarray,
             block_map: dict[str, slice],
             c: float,
             *,
             per_block: bool = False,
             safe_lo: float = 0.005,       # 0.005 rad  (≈ 0.3°)
             safe_hi: float = 0.02,        # 0.02 rad   (≈ 1.1°)
             weak_floor: float = 1e-3,     # 0.001 rad  (1 mrad)
             frac_limit: float = 0.05,     # ‖Δθ‖/‖θ‖  < 5 %
             tol_global: float = 0.20,     # looser CV_B limit
             tol_block:  float = 0.05,
            #  safe_lo: float = 0.03,
            #  safe_hi: float = 0.10,
            #  weak_floor: float = 5e-3,
            #  frac_limit: float = 0.15,
            #  tol_global: float = 0.40,
            #  tol_block:  float = 0.05,
             eps: float = 1e-12):
    """
    Run the four sanity-tests T1–T4 from the spec. Instead of raising on the first
    failure, collect pass/fail for each test and still compute all metrics. Return
    a dict with 'tests' (which tests passed/failed and messages) and 'metrics'
    (values as if everything passed).
    """
    M          = theta.size
    rms0       = _l2_vec(theta) / jnp.sqrt(M)
    dtheta     = c * rms0
    g_norms    = {k: _l2_vec(grads[v]) for k, v in block_map.items()}
    max_g      = max(g_norms.values()) + eps
    eta_global = dtheta / max_g

    # compute all intermediate quantities
    shift_dom = eta_global * max_g
    min_g     = min(g_norms.values()) + eps
    shift_w   = eta_global * min_g
    frac      = dtheta / (_l2_vec(theta) + eps)
    update_norms = jnp.asarray([eta_global * g for g in g_norms.values()])
    cv_up     = float(jnp.std(update_norms) / (jnp.mean(update_norms) + eps))
    # literature: GradNorm (Chen+’18), AdaScale (Bernstein+’20) use CV≤0.2
    limit = tol_block if per_block else tol_global

    # Prepare containers
    tests: dict[str, bool] = {}
    messages: dict[str, str] = {}

    # T1 – safe-shift window on dominant block
    try:
        assert safe_lo <= shift_dom <= safe_hi
        tests["T1"] = True
        messages["T1"] = ""
    except AssertionError:
        msg = f"T1 FAIL: dominant shift {shift_dom:.3e} not in [{safe_lo},{safe_hi}]"
        tests["T1"] = False
        messages["T1"] = msg
        print(msg)

    # T2 – weak-block visibility
    try:
        assert shift_w > weak_floor
        tests["T2"] = True
        messages["T2"] = ""
    except AssertionError:
        msg = f"T2 FAIL: weakest shift {shift_w:.2e} < {weak_floor:.2e}"
        tests["T2"] = False
        messages["T2"] = msg
        print(msg)

    # T3 – fractional norm change
    try:
        assert frac < frac_limit
        tests["T3"] = True
        messages["T3"] = ""
    except AssertionError:
        msg = f"T3 FAIL: ‖Δθ‖/‖θ‖ = {frac:.3f} > {frac_limit}"
        tests["T3"] = False
        messages["T3"] = msg
        print(msg)

    # T4 – update-level CV_B
    try:
        assert cv_up <= limit
        tests["T4"] = True
        messages["T4"] = ""
    except AssertionError:
        msg = f"T4 FAIL: CV_B_update = {cv_up:.3f} > {limit}"
        tests["T4"] = False
        messages["T4"] = msg
        print(msg)

    # Assemble metrics dictionary
    metrics = {
        "RMS0":      rms0,
        "Δθ*":       dtheta,
        "η_glob":    eta_global,
        "shift_dom": shift_dom,
        "shift_weak": shift_w,
        "CV_up":     cv_up,
    }

    return {
        "tests":   tests,
        "messages": messages,
        "metrics": metrics,
    }


def _slice_masks(time_steps: int, N_ctrl: int, N_reserv: int):
    """
    Generate boolean-mask helper lambdas for **slice-wise grouping**.

    Returns
    -------
    elem_mask : Callable[[int, int], jax.Array]
        ``elem_mask(t, M)`` → length-*M* boolean mask selecting the
        :math:`τ_t` scalar **plus** all couplings
        :math:`\{J_{kℓ}^{(t)}\}` belonging to Trotter slice *t*.
    h_mask : Callable[[int], jax.Array]
        ``h_mask(M)`` → mask of the three global field parameters
        :math:`(h^x,h^y,h^z)`.
    """
    num_J = N_ctrl * N_reserv
    T     = time_steps

    def elem_mask(t, M):
        idx      = jnp.arange(M)
        tau_m    = idx == t
        start_J  = T + 3 + t * num_J
        end_J    = start_J + num_J
        J_m      = (idx >= start_J) & (idx < end_J)
        return tau_m | J_m          # τₜ  ∪  J^{(t)}

    def h_mask(M):
        idx = jnp.arange(M)
        return (idx >= T) & (idx < T + 3)

    return elem_mask, h_mask


def lr_group_dict(case_num: int,
                  lr_tree: jnp.ndarray,
                  time_steps: int,
                  N_ctrl: int,
                  N_reserv: int):
    """
    Re-shape a *flat* learning-rate vector into a dict keyed by parameter block.

    Parameters
    ----------
    lr_tree : jax.Array
        Flat vector of per-parameter learning rates (:math:`\eta_i`, 1 ≤ i ≤ M).
    case_num : int
        ``≤ 5`` → returns keys ``"t","h","J"``;  
        ``≥ 6`` → returns ``"h", "slice_0", …, "slice_{T-1}"``.
    time_steps, N_ctrl, N_reserv : int
        Geometry parameters needed for slice-wise indexing.

    Returns
    -------
    dict[str, jax.Array]
        Each entry is a *view* (no copy) into ``lr_tree`` covering exactly the
        indices of that block.
    """
    if case_num <= 5:                    # τ / h / J grouping
        return {
            "t": lr_tree[:time_steps],
            "h": lr_tree[time_steps:time_steps + 3],
            "J": lr_tree[time_steps + 3:],
        }

    # slice-wise grouping
    M                 = lr_tree.shape[0]
    elem_mask, h_mask = _slice_masks(time_steps, N_ctrl, N_reserv)
    out               = {"h": lr_tree[h_mask(M)]}
    for t in range(time_steps):
        out[f"slice_{t}"] = lr_tree[elem_mask(t, M)]
    return out


def grad_group_dict(case_num: int,
                    grads_flat: jnp.ndarray,
                    time_steps: int,
                    N_ctrl: int,
                    N_reserv: int):
    """
    Same interface as :pyfunc:`lr_group_dict` but applied to **gradients**.

    Use this inside the training loop *before* calling the optimiser so you
    capture **pre-update** gradients.

    Returns
    -------
    dict[str, jax.Array]
        Block-wise gradient subvectors, suitable for computing CV\_B or other
        diagnostics.
    """
    if case_num <= 5:
        mask_tau = jnp.arange(grads_flat.size) < time_steps
        mask_h   = (jnp.arange(grads_flat.size) >= time_steps) & \
                   (jnp.arange(grads_flat.size) < time_steps + 3)
        mask_J   = jnp.arange(grads_flat.size) >= time_steps + 3
        return {
            "t": grads_flat[mask_tau],
            "h": grads_flat[mask_h],
            "J": grads_flat[mask_J],
        }

    # slice-wise
    M                 = grads_flat.size
    elem_mask, h_mask = _slice_masks(time_steps, N_ctrl, N_reserv)
    out               = {"h": grads_flat[h_mask(M)]}
    for t in range(time_steps):
        out[f"slice_{t}"] = grads_flat[elem_mask(t, M)]
    return out



def get_groupwise_lr_trees_slicewise(
    params:        jnp.ndarray,   # flat vector of length M = T + 3 + T*(N_ctrl*N_reserv)
    grads:         jnp.ndarray,   # flat gradient vector of length M
    num_train:     int,
    NC:            int,
    time_steps:    int,
    N_ctrl:        int,
    N_reserv:      int,
    max_lr:        float = 0.2,
    debug:         bool  = False,
    target_update: float = 0.05,
    eps:           float = 1e-12,
    scale_by_num_train: bool = False,
    per_block_target_update: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""
    **Temporal slice-wise learning-rate allocation**  
    (used by *case 5* / *case 6* optimisers).

    ------------------------------------------------
    Geometry
    ^^^^^^^^
    Let

    * :math:`T`   = ``time_steps``  (number of Trotter slices)  
    * :math:`C`   = ``N_ctrl``      (control qubits)  
    * :math:`R`   = ``N_reserv``    (reservoir qubits)  
    * :math:`J = C·R`

    The flat parameter vector is ordered

    \[
      \underbrace{(\tau_0,\dots,\tau_{T-1})}\_{T}
      \;\Vert\;
      \underbrace{(h^x,h^y,h^z)}_{3}
      \;\Vert\;
      \underbrace{\bigl(J_{00}^{(0)},\dots,J_{CR-1}^{(T-1)}\bigr)}_{T·J}.
    \]

    We build **T + 1 groups**  

    * group `t` ∈ {0,…,T−1} : ``slice t``   = { τ\_t } ∪ { all *J*\*\*(t)\* }  
    * group `T`             : ``h``         = { hˣ,hʸ,hᶻ }

    and assign a *single* scalar LR :math:`\eta_{t}` (resp. :math:`\eta_{h}`)
    to every parameter in that group.

    ------------------------------------------------
    Step-by-step algorithm
    ^^^^^^^^^^^^^^^^^^^^^^

    1. **Assignment mask**  
       Build ``assignment_mask`` ∈ ℕ^M such that

       \[
         \texttt{assignment_mask}[i] \;=\;
         \begin{cases}
           t, & i \in\text{ slice }t,\\[4pt]
           T, & i \in \{h^x,h^y,h^z\}.
         \end{cases}
       \]

       This is returned so downstream code can recover the grouping.

    2. **Target update Δθ\* (per group or global)**  

       *If* ``per_block_target_update=True``:

       \[
         \Delta\theta^{(\mathrm{gid})}_\* = 0.75\;
               \frac{\lVert\boldsymbol\theta_{\mathrm{gid}}\rVert_2}
                    {\sqrt{\lvert\mathrm{gid}\rvert}}
         \quad\text{for each gid ∈ \{0,…,T\}.}
       \]

       Otherwise every group uses the scalar ``target_update`` that was
       passed in.

    3. **Trust-ratio ceiling**  

       For each group *g* compute

       \[
         \eta^{\max}_g
           = \min\Bigl(
               \texttt{max_lr},
               \frac{\Delta\theta^{(g)}_\*}{\lVert\boldsymbol g^{(g)}\rVert_2+ε}
             \Bigr).
       \]

    4. **Robust scale r\_g**  

       \[
         r_g
           \;=\;
           \bigl(\operatorname{median}|\,g^{(g)}|\;+\;
                 \operatorname{MAD}|\,g^{(g)}|\bigr)\;
           \times
           \texttt{factor},
       \]
       where the optional multiplicative **factor**

       \[
         \texttt{factor} =
         \begin{cases}
            N_C/8,& \text{if } \texttt{scale_by_num_train}\land N_{\text{train}}\ge 20\\
            N_C/4,& \text{if } 11\le N_{\text{train}}\le 15\\
            N_C/2,& \text{if } N_{\text{train}}\le 10\\
            1,&\text{otherwise}.
         \end{cases}
       \]

    5. **Per-parameter LR inside a group**

       \[
         \eta_i
           = \operatorname{clip}_{[10^{-6},\,\texttt{max_lr}]}
             \!\Bigl(
                \eta^{\max}_g\;
                \frac{r_g}{\,|g_i| + r_g + ε}
             \Bigr),\qquad
             i\in g.
       \]

    6. **Return**

       ``lr_tree`` –  flat length-M array  
       ``assignment_mask`` –  the group IDs described in (1).

    ------------------------------------------------
    Debug output
    ^^^^^^^^^^^^
    When ``debug=True`` the routine prints

    * global ‖g‖₂, median, MAD  
    * per-group :math:`\eta^{\max}_g`, r\_g, and final LR statistics  
    * a detailed per-slice table enumerating every coupling  
      :math:`J_{rc}^{(t)}` (optional but invaluable when something diverges).

    ------------------------------------------------
    Complexity
    ^^^^^^^^^^
    :math:`\mathcal O(M)` – everything is a single `jax.lax` pass over the
    flat arrays; no Python-side loops except the
    ``for gid in range(T+1)`` scatter which is JIT-optimised.
    """
    M = grads.shape[0]
    idx = jnp.arange(M)
    num_J = N_ctrl * N_reserv
    T         = time_steps
    
    grad_abs = jnp.abs(grads) + 1e-12
    
    # ——— 2) Build an “assignment_mask” in [0..T] for each index i ∈ [0..D−1] ———
    #    If i < time_steps:          group_id = i   (so slice i)
    #    If time_steps ≤ i < time_steps+3: group_id = T  (the “h” group)
    #    Else:                        group_id = (i−(T+3)) // num_J  (i.e. which J‐block)
    # --- helper to turn index → group-id -------------------------------------
    def _assign(i):
        return jax.lax.cond(
            i < T,
            lambda k: k,                                           # τ_t
            lambda k: jax.lax.cond(
                k < T + 3,
                lambda _: T,                                       # global ⃗h
                lambda q: (q - (T + 3)) // num_J,                  # J^{(t)}
                k
            ),
            i
        )

    assignment_mask = jax.vmap(_assign)(idx)          # [M] int32
    # ---------- per-group target_update ----------------------------------------
     # block-specific target_update ------------------------------------
    if per_block_target_update:
        tgt_all = jnp.asarray([
            0.75 * _rms(params[assignment_mask == gid])  # gid 0..T
            for gid in range(T + 1)
        ])
    else:
        tgt_all = jnp.full((T + 1,), float(target_update))

    grad_norm_all = jnp.linalg.norm(grad_abs)
    eta_max = jnp.minimum(max_lr,
                          target_update / (grad_norm_all + eps))

    # ---------- r_g per group ---------------------------------------------------
    # block med+MAD
    def _med_mad(x):
        med = jnp.median(x)
        mad = jnp.median(jnp.abs(x - med))
        return med + mad
    r_all = jnp.asarray([
        _med_mad(grad_abs[assignment_mask == gid])
        for gid in range(T + 1)
    ])
    if scale_by_num_train:
        factor = NC / 8 if num_train >= 20 else NC / 4 if num_train > 10 else NC / 2
        r_all *= factor

    # ---------- assemble lr_tree ------------------------------------------------
    lr_tree = jnp.zeros_like(grad_abs)
    for gid in range(T + 1):
        sel      = assignment_mask == gid
        g_block  = grad_abs[sel]
        r_g      = r_all[gid]
        eta_g    = jnp.minimum(max_lr, tgt_all[gid] / (jnp.linalg.norm(g_block) + eps))
        lr_vals  = jnp.clip(eta_g * r_g / (g_block + r_g + eps), 1e-6, max_lr)
        lr_tree  = lr_tree.at[sel].set(lr_vals)
    # ----------------------------------------------------------------------
    if debug:
        grad_norm_all = jnp.linalg.norm(grad_abs)
        eta_max_global = jnp.minimum(max_lr, target_update / (grad_norm_all + eps))

        print(f"\n--- global-norm anchor (slice-wise) ---")
        med_all = jnp.median(grad_abs)
        mad_all = jnp.median(jnp.abs(grad_abs - med_all))
        print(f"  ‖g‖₂ = {grad_norm_all:.3e},  eta_max = {float(eta_max_global):.3e}")
        print(f"  median(|g|) = {med_all:.3e},  MAD(|g|) = {mad_all:.3e}")
        print("--------------------------------------\n")

        # h-vector block
        hx, hy, hz = lr_tree[T:T+3]
        print(f"global h-vec lrs: hx={hx:.2e}, hy={hy:.2e}, hz={hz:.2e}")

        # per-slice detail
        reserv_qubits = list(range(N_ctrl, N_ctrl + N_reserv))
        ctrl_qubits   = list(range(N_ctrl))

        print("\n=== per-time-step learning-rates ===")
        for t in range(T):
            tau_lr = float(lr_tree[t])
            start  = T + 3 + t * num_J
            end    = start + num_J
            J_block = lr_tree[start:end]
            avg_J   = float(jnp.mean(J_block))

            J_elems = []
            for j, c in enumerate(ctrl_qubits):
                for i, r in enumerate(reserv_qubits):
                    idx_local = i * len(ctrl_qubits) + j
                    J_elems.append(f"J({r},{c})={float(J_block[idx_local]):.2e}")

            print(f" step {t:2d}: τ_lr={tau_lr:.2e}, avg(J_lr)={avg_J:.2e}, "
                  + ", ".join(J_elems))

        print("\n[slice-wise LR] summary:")
        for t in range(T):
            sel = assignment_mask == t
            print(f"  slice {t:2d}: mean={float(jnp.mean(lr_tree[sel])):.3e}, "
                  f"min={float(jnp.min(lr_tree[sel])):.3e}, "
                  f"max={float(jnp.max(lr_tree[sel])):.3e}")
        h_sel = assignment_mask == T
        print(f"    h-group : mean={float(jnp.mean(lr_tree[h_sel])):.3e}, "
              f"min={float(jnp.min(lr_tree[h_sel])):.3e}, "
              f"max={float(jnp.max(lr_tree[h_sel])):.3e}")
        print("-----------------------------------------------------------\n")

    return lr_tree, assignment_mask



def get_groupwise_lr_trees(
    params: jnp.ndarray,        # flat parameter vector
    grads:  jnp.ndarray,        # flat gradient vector (same length)
    num_train: int,
    NC: int,                    # == N_ctrl
    time_steps: int,            # == T
    *,
    max_lr: float = 0.2,
    debug: bool = False,
    scale_by_num_train: bool = False,
    target_update: float = 0.05,
    eps: float = 1e-12,
    factor: float = 1.0,        # kept for backward-compat
    per_block_target_update: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    **Parameter-type learning-rate allocation**  
    (used by *case 1 – 4* optimisers).

    Blocks
    ^^^^^^
    * ``τ``-block : indices [0 : T)  
    * ``h``-block : indices [T : T+3)  
    * ``J``-block : indices [T+3 : M)

    where :math:`M = T + 3 + T·(N_C·N_R)`.

    ------------------------------------------------
    Step-by-step
    ^^^^^^^^^^^^

    1. **Global RMS and optional per-block RMS**

        \[
            \mathrm{RMS}(\theta) = \frac{\lVert\boldsymbol\theta\rVert_2}{\sqrt{M}}.
        \]

        *If* ``per_block_target_update=True``  
        set :math:`Δθ_\*^{(B)} = 0.75 · \mathrm{RMS}(\theta_B)`  
        otherwise every block shares the scalar ``target_update``.

    2. **Trust-ratio ceiling**

        For each block :math:`B∈\{τ,h,J\}`

        \[
            \eta^{\max}_B = \min\!\Bigl(
                \texttt{max_lr},
                \frac{Δθ_\*^{(B)}}{\lVert\boldsymbol g_B\rVert_2 + ε}
            \Bigr).
        \]

    3. **Robust scale**

        \[
            r_B
            = \bigl(\operatorname{median}|g_B| + \operatorname{MAD}|g_B|\bigr)
                \times \texttt{scale\_fac},
        \]
        where the optional scaling factor is identical to the one documented
        for the slice-wise routine.

    4. **Per-parameter rule**

        \[
            \eta_i
            = \operatorname{clip}_{[10^{-6},\,\texttt{max_lr}]}\!
                \left(
                \eta^{\max}_B\;
                \frac{r_B}{\,|g_i| + r_B + ε}
                \right),
            \qquad i∈B.
        \]

    5. **Return**

        * ``lr_tree``              – flat length-M array  
        * ``mask_tau, mask_h, mask_J`` – boolean masks selecting each block.

    ------------------------------------------------
    Debug diagnostics
    ^^^^^^^^^^^^^^^^^
    When ``debug=True`` prints

    * global and block-wise RMS(θ), η\_max (uncapped → clipped)  
    * r\_B pre / post scaling  
    * final min / max / mean LR for each block.

    ------------------------------------------------
    Citations
    ^^^^^^^^^
    * Benzing et al., *PRX Quantum* 3 (2022) – safe angular update region  
    * Maia et al., *npj QIC* 9 (2023) – median/MAD normalisation in VQAs  
    * Wang & Kolter, *ICML* (2023) “RMS-trust ratio” – motivation for step 2.
      
    """

    # ── 1. prepare masks ─────────────────────────────────────────────
    M   = params.size
    idx = jnp.arange(M)
    c = TRUST_COEFF_DEFAULT          # ≈ 1 % of ‖θ‖ (literature-recommended)

    # global quantities
    global_rms_params = jnp.linalg.norm(params) / jnp.sqrt(M)
    grad_norm_all = jnp.linalg.norm(grads)
    global_target = c * global_rms_params
    eta_all = jnp.where(grad_norm_all > 0.0,
                        global_target / (grad_norm_all + eps),
                        1e-3)
    clipped_eta_all = jnp.minimum(eta_all, max_lr)
    
    

  
        

    mask_tau = idx < time_steps
    mask_h   = (idx >= time_steps) & (idx < time_steps + 3)
    mask_J   = idx >= time_steps + 3

    g_tau, g_h, g_J = grads[mask_tau], grads[mask_h], grads[mask_J]
    p_tau, p_h, p_J = params[mask_tau], params[mask_h], params[mask_J]

    # ── 2. block-wise target_update  (Δθ*) ────────────────────────────
    
    if per_block_target_update:
        # *Should* tighten the trust ratio separately for the slow-moving h-vector and the typically noisier J-couplings - need sources...
        def rms_block(mask):
            #  per-block RMS
            vec = params[mask]
            return jnp.linalg.norm(vec) / jnp.sqrt(vec.size)
        tgt_tau = c * rms_block(mask_tau)
        tgt_h   = c * rms_block(mask_h)
        tgt_J   = c * rms_block(mask_J)
    else:                       
        target_update = tgt_tau = tgt_h = tgt_J = c * global_rms_params


    # ---------- run T-tests ---------------------------------------------------
    block_map = {
        "t": slice(0, time_steps),
        "h": slice(time_steps, time_steps + 3),
        "J": slice(time_steps + 3, M),
    }
    test_report = verify_c(
        params,
        grads,
        block_map,
        c,
        per_block=per_block_target_update
    )
    tests = test_report.get('tests',{})
    messages = test_report.get('messages',{})
    metrics = test_report.get('metrics', {})
    if debug:
        print("\n── DEBUG ──")
        print(f"*per_block_target_upd*  = {per_block_target_update}")
        if per_block_target_update:
            print("  → Using Per-Block Target Update!")
        else:
            print("  → Using Global Target Update!")
        print(f"  max_lr                   = {max_lr:.3e}")
        print(f"  c                        = {c:.3e}")
        print(f"  GLOBAL RMS (‖θ‖/√M)      = {global_rms_params:.3e}")

        print("\n── VERIFYING c IS WELL-CHOSEN ──")
        for test_name, passed in tests.items():
            status = "PASS" if passed else "FAIL"
            msg = messages.get(test_name, "")
            if passed:
                print(f"  {test_name}: {status}")
            else:
                print(f"  {test_name}: {status}  → {msg}")
        print("\n  Metrics:")
        for metric_name, value in metrics.items():
            print(f"    {metric_name:<10} = {value:.3e}")

    # ── 3. block-wise η^max_B  (trust ratio) where the blocks are B \in {τ,h,J} ────────────────────────────
    lr_tau_uncapped = tgt_tau / (jnp.linalg.norm(g_tau) + eps)
    lr_tau_clipped = jnp.minimum(max_lr, lr_tau_uncapped)

    lr_h_uncapped = tgt_h / (jnp.linalg.norm(g_h) + eps)
    lr_h_clipped = jnp.minimum(max_lr, lr_h_uncapped)

    lr_J_uncapped = tgt_J / (jnp.linalg.norm(g_J) + eps)
    lr_J_clipped = jnp.minimum(max_lr, lr_J_uncapped)


     # ── 4. median+MAD scale r_B ───────────────────────────────────────
    def med_mad(vec):
        med = jnp.median(jnp.abs(vec))
        mad = jnp.median(jnp.abs(vec - med))
        return med + mad

    r_tau_pre, r_h_pre, r_J_pre = med_mad(g_tau), med_mad(g_h), med_mad(g_J)

    # determine scale_factor
    if scale_by_num_train:
        if   num_train >= 20: scale_factor = NC / 8
        elif num_train >= 11: scale_factor = NC / 4
        else:                 scale_factor = NC / 2
    else:
        scale_factor = 1.0

    # apply scaling
    r_tau = r_tau_pre * scale_factor
    r_h   = r_h_pre   * scale_factor
    r_J   = r_J_pre   * scale_factor

     # ── 5. per-parameter learning rates ───────────────────────────────
    def per_param_lr(η, r, g):
        raw = η * r / (jnp.abs(g) + r + eps)
        return jnp.clip(raw, 1e-6, max_lr)

    lr_tau = per_param_lr(lr_tau_clipped, r_tau, g_tau)
    lr_h   = per_param_lr(lr_h_clipped,   r_h,   g_h)
    lr_J   = per_param_lr(lr_J_clipped,   r_J,   g_J)

    # ── 6. scatter into full lr_tree ──────────────────────────────────
    lr_tree = jnp.zeros_like(grads)
    lr_tree = lr_tree.at[mask_tau].set(lr_tau)
    lr_tree = lr_tree.at[mask_h  ].set(lr_h)
    lr_tree = lr_tree.at[mask_J  ].set(lr_J)

   
    if debug:

        

        print("\n── get_groupwise_lr_trees DEBUG ──")
        print(f"max_lr                = {max_lr:.3e}")

        # print(f"scale_by_num_train    = {scale_by_num_train} (scale_factor={scale_factor:.3e})")
        print(f"per_block_target_upd  = {per_block_target_update}")
        print(f'‖θ‖/√M (global RMS)   = {global_rms_params:.3e}')
        # print target per group
        print(f"Target updates (τ, h, J) = {tgt_tau:.3e}, {tgt_h:.2e}, {tgt_J:.2e}")
        if scale_by_num_train:
            # print r values before and after scaling
            print("r values (pre-scale → post-scale):")
            print(f"  τ: {r_tau_pre:.2e} → {r_tau:.2e}")
            print(f"  h: {r_h_pre:.2e} → {r_h:.2e}")
            print(f"  J: {r_J_pre:.2e} → {r_J:.2e}")
        # print block-wise η_max ceilings for each block
        print("η_max ceilings (uncapped → clipped):")
        print(f"  τ: {lr_tau_uncapped:.2e} → {lr_tau_clipped:.2e}")
        print(f"  h: {lr_h_uncapped:.2e} → {lr_h_clipped:.2e}")
        print(f"  J: {lr_J_uncapped:.2e} → {lr_J_clipped:.2e}")
        # print global η_max for reference
        print(f"Global η_max (uncapped → clipped) = {eta_all:.2e} → {clipped_eta_all:2e}")
        # print final per-group learning-rate statistics
        print("Final lrs by block η_τ, η_:")
        print(f"  τ-group: min={jnp.min(lr_tau):.2e}, max={jnp.max(lr_tau):.2e}, mean={jnp.mean(lr_tau):.2e}")
        print(f"  h-group: min={jnp.min(lr_h):.2e},   max={jnp.max(lr_h):.2e},   mean={jnp.mean(lr_h):.2e}")
        print(f"  J-group: min={jnp.min(lr_J):.2e},   max={jnp.max(lr_J):.2e},   mean={jnp.mean(lr_J):.2e}")
        print("────────────────────────────────────────\n")
        # print(f"lr_tree: {lr_tree}")

    # ── 8. return ────────────────────────────────────────────────────
    return lr_tree, mask_tau, mask_h, mask_J


def get_optimizer(
    case: int,
    opt_lr,
    time_steps: int, 
    num_epochs: int = None,
    # Adam hyper-parameters
    b1: float = 0.99,
    b2: float = 0.999,
    eps: float = 1e-8,
    # ReduceLROnPlateau hyper-parameters
    factor: float = 0.9,
    patience: int = 50,
    rtol: float = 1e-4,
    atol: float = 0.0,
    cooldown: int = 0,
    accumulation_size: int = 1,
    cost_threshold=1e-3,
    grad_var_tol=1e-6,
    min_scale: float = 0.0,
    # fraction of epochs to wait before enabling plateau logic
    warmup_steps:int = 100,
    warmup_fraction: float = 0.5,
):

    # print(f"get_optimizer() for case_num = {case}. opt_lr: {opt_lr} ")
    def wrap(base_optimizer, passes_value: bool, passes_step: bool):
        def init_fn(params):
            return base_optimizer.init(params)

        def update_fn(updates, state, params=None, *, value=None, step=None, **extra):
            kwargs = {}
            # print(f"state: {state}")
            if passes_value:
                kwargs["value"] = value
            if passes_step:
                kwargs["step"] = step
            # print(f"kwargs: {kwargs}")
            return base_optimizer.update(updates, state, params=params, **kwargs)

        return optax_base.GradientTransformationExtraArgs(init_fn, update_fn)
    def make_chain(component):
        masks = {"t": [True, False, False],
                 "h": [False, True, False],
                 "J": [False, False, True]}[component]
        return lambda gt: optax.masked(gt, dict(zip("thJ", masks)))


    if case == 1:
        desc = "masked per group Adam"
        base_opt = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.masked(optax.adam(opt_lr["t"], b1=b1, b2=b2, eps=eps), {"t": True,  "h": False, "J": False}),
            optax.masked(optax.adam(opt_lr["h"], b1=b1, b2=b2, eps=eps), {"t": False, "h": True,  "J": False}),
            optax.masked(optax.adam(opt_lr["J"], b1=b1, b2=b2, eps=eps), {"t": False, "h": False, "J": True}),
        )
        # does NOT need the loss value
        return desc, wrap(base_opt, passes_value=False, passes_step=False)

    elif case == 2:
        desc = "masked per group Adam + delayed ReduceLROnPlateau"
        # how many update‐calls to skip plateau reductions
        if not warmup_steps:
            warmup_steps = int(num_epochs * warmup_fraction)
        print(f"warmup steps: {warmup_steps}")

        def one_group(rate):
            return optax.chain(
                optax.inject_hyperparams(optax.adam)(
                    learning_rate=rate, b1=b1, b2=b2, eps=eps
                ),
                reduce_on_plateau(
                    factor=factor,
                    patience=patience,
                    rtol=rtol,
                    atol=atol,
                    cooldown=cooldown,
                    accumulation_size=accumulation_size,
                    min_scale=min_scale,
                    warmup_steps=warmup_steps,
                ),
            )

        mask_t = {"t": True,  "h": False, "J": False}
        mask_h = {"t": False, "h": True,  "J": False}
        mask_J = {"t": False, "h": False, "J": True}

        base_opt = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.masked(one_group(opt_lr["t"]), mask_t),
            optax.masked(one_group(opt_lr["h"]), mask_h),
            optax.masked(one_group(opt_lr["J"]), mask_J),
        )
        return desc, wrap(base_opt, passes_value=True, passes_step=True)
    elif case == 3:
        desc = "group Adam + cost-threshold shrink"
        sched = reduce_on_cost_threshold(cost_threshold, factor, min_scale)
        def g(rate): return optax.chain(adam(rate), sched)
        base = optax.chain(
            optax.clip_by_global_norm(1.),
            make_chain("t")(g(opt_lr["t"])),
            make_chain("h")(g(opt_lr["h"])),
            make_chain("J")(g(opt_lr["J"])),
        )
        return desc, wrap(base, passes_value=True, passes_step=False)


    elif case == 4:
        desc = "group Adam + grad-variance shrink"
        sched = reduce_on_grad_variance(factor, patience, 50,
                                        grad_var_tol, min_scale)
        def g(rate): return optax.chain(adam(rate), sched)
        base = optax.chain(
            optax.clip_by_global_norm(1.),
            make_chain("t")(g(opt_lr["t"])),
            make_chain("h")(g(opt_lr["h"])),
            make_chain("J")(g(opt_lr["J"])),
        )
        return desc, wrap(base, passes_value=False, passes_step=True)
    # ------------------------------------------------------------------
    # case == 5  →  slice-wise masked Adam  +  per-param scaling
    # ------------------------------------------------------------------
    elif case == 5:
        """
        case 5 – slice-wise independent Adam (τᵗ ∪ Jᵗ) + per-param LR scaling.
        Expects:
            opt_lr = (lr_tree, assignment_mask) from get_groupwise_lr_trees_slicewise
              lr_tree         : flat [M] array
              assignment_mask : flat [M] ints in {0,…,T}  (T labels the h-block)
        """
        desc = "slice-wise Adam (multi_transform) + per-param LR"
        lr_tree, assignment_mask = opt_lr      # unpack
        T          = time_steps
        lr_pytree  = {                         # for the final element-wise scaling
            "t": lr_tree[:T],
            "h": lr_tree[T:T+3],
            "J": lr_tree[T+3:],
        }

        # 1) build a label-pytree matching the param structure
        #
        #    t-leaf  : [0, 1, …, T−1]
        #    h-leaf  : [T, T, T]
        #    J-leaf  : labels copied from assignment_mask[T+3:]
        #
        t_labels = jnp.arange(T, dtype=jnp.int32)
        h_labels = jnp.full((3,), T, dtype=jnp.int32)
        j_labels = assignment_mask[T+3:].astype(jnp.int32)
        label_pytree = {"t": t_labels, "h": h_labels, "J": j_labels}

        # 2) optimiser dict   {label:int → Adam(1.0)}
        opt_dict = {gid: optax.adam(learning_rate=1.0, b1=b1, b2=b2, eps=eps)
                    for gid in range(T + 1)}       # 0..T

        # 3) final element-wise LR multiplier (unchanged helper)
        def per_param_lr(tree_lr):
            def init_fn(_): return ()
            def upd_fn(upd, state, params=None):
                scaled = jax.tree_util.tree_map(lambda g, lr: g * lr, upd, tree_lr)
                return scaled, state
            return optax.GradientTransformation(init_fn, upd_fn)

        base_opt = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.multi_transform(opt_dict, label_pytree),
            per_param_lr(lr_pytree)
        )
        return desc, wrap(base_opt, passes_value=False, passes_step=False)
    elif case == 6:
        """
        case 6 ─ slice-wise median/MAD LR  →  masked Adam
        --------------------------------------------------
        opt_lr == (lr_tree, assignment_mask)

          lr_tree         : flat [M] array
          assignment_mask : flat [M] ints in {0,1,…,T}, where T labels the h-block
        """
       
        desc = "slice-wise (median/MAD) Adam + per-param scaling"
        # ----------------------------------------------------------------------
        # element-wise learning-rate: multiply each update by lr_tree_pytree
        # ----------------------------------------------------------------------
        def per_param_lr(lr_pytree):
            """Return a GradientTransformation that scales updates by lr_pytree."""
            def init_fn(_):
                return ()                         # stateless

            # NOTE: must accept the unused `params` positional arg!
            def update_fn(updates, state, params=None):
                scaled = jax.tree_util.tree_map(
                    lambda g, lr: g * lr, updates, lr_pytree
                )
                return scaled, state

            return optax.GradientTransformation(init_fn, update_fn)

        lr_tree, _ = opt_lr                    # assignment_mask is no longer needed
        # reshape the flat lr_tree into the same pytree structure as `params`
        lr_pytree = {
            "t": lr_tree[:time_steps],                       # shape [T]
            "h": lr_tree[time_steps:time_steps + 3],         # shape [3]
            "J": lr_tree[time_steps + 3:]                    # shape [T * num_J]
        }

        base_opt = optax.chain(
            optax.clip_by_global_norm(1.0),                  # safety first
            optax.adam(learning_rate=1.0, b1=b1, b2=b2, eps=eps),
            per_param_lr(lr_pytree)                          # ← element-wise LR
        )
        return desc, wrap(base_opt, passes_value=False, passes_step=False)

        
    # default fallback
    desc = "per-param Adam"
    base_opt = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.inject_hyperparams(optax.adam)(
            learning_rate=opt_lr, b1=b1, b2=b2, eps=eps
        ),
    )
    return desc, wrap(base_opt, passes_value=False)


class Sim_QuantumReservoir:
    def __init__(self, params, N_ctrl, N_reserv, num_J, time_steps=1,bath=False,num_bath = 0):
        self.bath = bath
        self.num_bath = num_bath
        self.N_ctrl = N_ctrl
        self.N_reserv = N_reserv
        self.reserv_qubits = qml.wires.Wires(list(range(N_ctrl, N_reserv+N_ctrl)))
        self.ctrl_qubits = qml.wires.Wires(list(range(N_ctrl)))

        if bath:
            self.bath_qubits = qml.wires.Wires(list(range(N_reserv+N_ctrl, N_reserv+N_ctrl+num_bath)))
            self.bath_interactions = params['bath']
            self.N = N_ctrl + N_reserv + num_bath
            self.dev = qml.device("default.qubit", wires = [*self.ctrl_qubits, *self.reserv_qubits,*self.bath_qubits]) 
        else:
            self.N = N_ctrl + N_reserv
            self.dev = qml.device("default.qubit", wires = [*self.ctrl_qubits, *self.reserv_qubits]) 

        self.qubits = qml.wires.Wires(list(range(self.N)))

        self.k_coefficient = params['K_coef']
        self.steps = time_steps

        self.num_J = num_J
        self.params = params
        self.current_index = 0

   
    def get_all_wires(self):
        return self.qubits            
    def get_dev(self):
        return self.dev

    def get_ctrl_wires(self):
        return self.ctrl_qubits

    def get_reserv_wires(self):
        return self.reserv_qubits   
    def get_wires(self):
        return self.qubits   

    def get_XY_coupling(self, i, j):
        '''Return the XY coupling between qubits i and j with a coefficient function.'''
        return ((qml.PauliX(wires=i) @ qml.PauliX(wires=j)) + (qml.PauliY(wires=i) @ qml.PauliY(wires=j)))
    
    def get_X_res(self):
        '''Return the XY coupling between qubits i and j with a coefficient function.'''
        return (qml.PauliX(wires=r) for r in [*self.reserv_qubits])

   
    def get_total_hamiltonian_components(self):
        def J_func(p, t):
            ''' Function to return time-dependent J parameter '''
            return p

        def hx_func(p,t):

            ''' Function to return constant hx parameter '''
            return p
        def hz_func(p,t):

            ''' Function to return constant hz parameter '''
            return p
        def hy_func(p,t):

            ''' Function to return constant hy parameter '''
            return p
        coefficients = []
        operators = []
        # Add hx terms for reservoir qubits
        
        # Add XY coupling terms
        for i,qubit_a in enumerate(self.ctrl_qubits):
            for j,qubit_b in  enumerate(self.reserv_qubits):
        # for i,qubit_a in enumerate(self.ctrl_qubits):
        #     for j,qubit_b in  enumerate(self.reserv_qubits):
                idx = j * self.N_reserv + (i  - self.N_ctrl)
                # Lambda function to capture the current parameter and time
                coefficients.append(J_func)
                
                new_operator = self.get_XY_coupling(qubit_a,qubit_b)
                
                operators.append(new_operator)
                
        
        coefficients.append(hx_func)
        new_operator = sum((qml.PauliX(wires=r) for r in self.reserv_qubits))      
        operators.append(new_operator)

        coefficients.append(hy_func)
        new_operator = sum((qml.PauliY(wires=r) for r in self.reserv_qubits))      
        operators.append(new_operator)

        coefficients.append(hz_func)
        new_operator = sum((qml.PauliZ(wires=r) for r in self.reserv_qubits))      
        operators.append(new_operator)

        H_dynamic = qml.dot(coefficients,operators)
        #print(f"H_dynamic: {H_dynamic}")
        ''' Construct the non-parametrized part of the Hamiltonian '''
        static_coefficients = []
        static_operators = []

        
        for qubit_a in range(len(self.reserv_qubits)):
            
            for qubit_b in range(len(self.reserv_qubits)):
                
                if qubit_a != qubit_b:
                    
                    interaction_coeff = self.k_coefficient[qubit_a, qubit_b]
                   
                    static_coefficients.append(interaction_coeff)
                    new_operator = self.get_XY_coupling(self.reserv_qubits[qubit_a], self.reserv_qubits[qubit_b])
                    static_operators.append(new_operator)

        # tbd add bath

        #print(static_coefficients, static_operators)
        if self.N_reserv == 1 and self.bath == False:
            total_H = H_dynamic
        
        else:
            H_static = qml.dot(static_coefficients, static_operators)
            total_H = H_dynamic+H_static
        ##sum(coeff * op for coeff, op in zip(static_coefficients, static_operators))
        
        
        

        return total_H
    


def run_test(params, 
    num_epochs, 
    N_reserv, 
    N_ctrl, 
    time_steps,
    N_train,
    folder,
    gate,gate_name,
    bath,
    num_bath,
    init_params_dict, 
    dataset_key,
    case_num,
    PATIENCE,
    ACCUMULATION_SIZE, 
    RTOL = 1e-4,
    ATOL=1e-7,
    COOLDOWN=0,
    FACTOR = 0.9,
    MIN_SCALE = 0.01,
    scale_by_num_train=True,
    per_block_target_update=False,
    ):
    float32=''
    opt_lr = None
    a_marked = None
    preopt_results = None
    selected_indices, min_var_indices,replacement_indices = [],[],[]
    num_states_to_replace = N_train // 5

    num_J = N_ctrl*N_reserv
    folder_gate = folder + str(num_bath) + '/'+gate_name + '/reservoirs_' + str(N_reserv) + '/trotter_step_' + str(time_steps) +'/' + 'bath_'+str(bath)+'/'
    # folder_gate = folder + str(num_bath) + '/'+gate_name + '/reservoirs_' + 'sample_.5pi' + '/trotter_step_' + str(time_steps) +'/' + 'bath_'+str(bath)+'/'
    Path(folder_gate).mkdir(parents=True, exist_ok=True)
    temp_list = list(Path(folder_gate).glob('*'))
    files_in_folder = []
    for f in temp_list:
        temp_f = f.name.split('/')[-1]
        
        if not temp_f.startswith('.'):
            files_in_folder.append(temp_f)
    
    k = 1
   
    if len(files_in_folder) >= k:
        print('Already Done. Skipping: '+folder_gate)
        print('\n')
        return

    # get PQC
    sim_qr = Sim_QuantumReservoir(init_params_dict, N_ctrl, N_reserv, N_reserv * N_ctrl,time_steps,bath,num_bath)
    

    init_params = params

    filename = os.path.join(folder_gate, f'data_run_{len(files_in_folder)}.pickle')


    input_states, target_states = generate_dataset(gate, N_ctrl,training_size= N_train, key= dataset_key, new_set=False)
    # print(f"training state #1: {input_states[0]}")
    # preopt_results, input_states, target_states,second_A,second_b,f_A,f_b,opt_lr,selected_indices = optimize_traingset(gate,N_ctrl, N_reserv,time_steps, params, init_params_dict, N_train,num_datasets=5,key=dataset_key)
    ts1 = input_states[0]


    test_dataset_key = jax.random.split(dataset_key)[1]
    test_in, test_targ = generate_dataset(gate, N_ctrl,training_size= 2000, key=test_dataset_key, new_set=True)
    


    parameterized_ham = sim_qr.get_total_hamiltonian_components()

    
    ctrl_wires = sim_qr.get_ctrl_wires()
    reserv_wires = sim_qr.get_reserv_wires()
    qnode_dev = sim_qr.get_dev()
    all_wires = sim_qr.get_all_wires()
     

    @qml.qnode(qnode_dev, interface="jax")
    def circuit(params,state_input):
        
        taus = params[:time_steps]

        qml.StatePrep(state_input, wires=[*ctrl_wires])
        

        for idx, tau in enumerate(taus):
           
            hx_array = jnp.array([params[time_steps]])  # Convert hx to a 1D array
            hy_array = jnp.array([params[time_steps + 1]])  # Convert hy to a 1D array
            hz_array = jnp.array([params[time_steps + 2]])  # Convert hz to a 1D array
            J_values = params[time_steps + 3 + idx * num_J : time_steps + 3 + (idx + 1) * num_J]

            # Concatenate hx_array with J_values
            current_step = jnp.concatenate([J_values,hx_array,hy_array,hz_array])
            
            qml.evolve(parameterized_ham)(current_step, t=tau)
            
        return qml.density_matrix(wires=[*ctrl_wires])
    specs_func = qml.specs(circuit)
    specs = specs_func(params,input_states[0])
    circuit_depth = specs['resources'].depth
    num_gates = specs['resources'].num_gates

    # print(f"spcs: {specs_func(params,input_states[0])}")
    
    jit_circuit = jax.jit(circuit)
    vcircuit = jax.vmap(jit_circuit, in_axes=(None, 0))
    def batched_cost_helper(params, X, y):
        # Process the batch of states
        batched_output_states = vcircuit(params, X)
        
        # Compute fidelity for each pair in the batch and then average
        fidelities = jax.vmap(qml.math.fidelity)(batched_output_states, y)
        fidelities = jnp.clip(fidelities, 0.0, 1.0)
        average_fidelity = jnp.mean(fidelities)
       
        return 1 - average_fidelity  # Minimizing infidelity
    @jit
    def cost_func(params,input_states, target_states):
        params = jnp.asarray(params, dtype=jnp.float64)
        X = jnp.asarray(input_states, dtype=jnp.complex128)
        y = jnp.asarray(target_states, dtype=jnp.complex128)
        # Process the batch of states
        loss = batched_cost_helper(params, X, y)
        loss = jnp.maximum(loss, 0.0)  # Apply the cutoff to avoid negative costs

        return loss
    
    

    
    min_raw_lr = 0.
    # Initial training to determine appropriate learning rate
    if opt_lr == None:
        print(f"Assessing optimal init lr vector for network with N_c (ctrl qubits) = {N_ctrl} and N_r (reservoir qubits) = {N_reserv}, T (time steps) = {time_steps} -->  M = (N_r X N_c X T) + T + 3 = {len(params)} trainable parameters, L (number of training states) = {N_train}")
        s = time.time()
        
        init_loss, init_grads = jax.value_and_grad(cost_func)(params, input_states, target_states)
        e = time.time()
        dt = e - s
        

        flat_grads = jnp.ravel(init_grads)
        
        # opt_lr_tree, mask_tau, mask_h, mask_J = get_groupwise_lr_lars(
        #     params, flat_grads,N_train ,NC=N_ctrl, max_lr=max_lr, time_steps=time_steps, debug=True
        # )
        cv0 = coef_variation_by_block(case_num, flat_grads,
                              time_steps, N_ctrl, N_reserv)
        cv_str = ", ".join(f"{k}: {v:.4f}" for k, v in cv0.items())
        print("\nInitial coefficient of variation (CV_B) by block at step 0: ", cv_str)


        
        if case_num in (5, 6):
      
            print(f"Getting init lr stats for case #{case_num}: ")
            lr_tree, assignment_mask = get_groupwise_lr_trees_slicewise(
                params           = params,       
                grads            = flat_grads,
                num_train        = N_train,
                NC               = N_ctrl,
                time_steps       = time_steps,
                N_ctrl           = N_ctrl,
                N_reserv         = N_reserv,
                max_lr           = 0.2,
                debug            = True,
                target_update    = None,
                eps              = 1e-12,
                scale_by_num_train = scale_by_num_train,
                per_block_target_update=per_block_target_update
            )
            opt_lr = (lr_tree, assignment_mask)
           
        else:
            print(f"Getting init lr stats for case #{case_num}: ")
          
            lr_tree, mask_tau, mask_h, mask_J = get_groupwise_lr_trees(
                params         = params,          # your flat parameter vector
                grads          = flat_grads,     # magnitude of gradients
                num_train      = N_train,        # number of Haar‐random states
                NC             = N_ctrl,         # number of control qubits
                max_lr         = 0.2,            # your chosen clip‐upper‐bound
                time_steps     = time_steps,     # T
                debug          = True,
                scale_by_num_train = scale_by_num_train,
                target_update  = None,
                eps            = 1e-12,
                per_block_target_update=per_block_target_update
            )
            opt_lr = {
                "t": lr_tree[:time_steps],
                "h": lr_tree[time_steps:time_steps + 3],
                "J": lr_tree[time_steps + 3:]
            }
      
        
        lr_groups_init = lr_group_dict(case_num, lr_tree, time_steps, N_ctrl, N_reserv)
        
        cost = init_loss
    
 
   
    debug = False
    

    if debug:
        # Compute group‐wise subsets
        lr_tau = lr_tree[mask_tau]
        lr_h   = lr_tree[mask_h]
        lr_J   = lr_tree[mask_J]

        # Print mean / min / max for each group
        print("\n=== group-wise learning‐rate stats ===")
        print(f" t‐group: mean={float(jnp.mean(lr_tau)):.3e}, "
            f"min={float(jnp.min(lr_tau)):.3e}, max={float(jnp.max(lr_tau)):.3e}")
        print(f" h‐group: mean={float(jnp.mean(lr_h)):.3e}, "
            f"min={float(jnp.min(lr_h)):.3e}, max={float(jnp.max(lr_h)):.3e}")
        print(f" J‐group: mean={float(jnp.mean(lr_J)):.3e}, "
            f"min={float(jnp.min(lr_J)):.3e}, max={float(jnp.max(lr_J)):.3e}")

       
        print("\n=== per‐time‐step learning‐rates ===")
        reserv_qubits = sim_qr.reserv_qubits.tolist()  # List of reservoir qubit indices
        ctrl_qubits = sim_qr.ctrl_qubits.tolist()      # List of control qubit indices
        print(f'global h-vec lrs: hx={lr_tree[time_steps]:.2e} hy={lr_tree[time_steps+1]:.2e} hz={lr_tree[time_steps+2]:.2e} ')
        for t in range(time_steps):
            tau_lr = float(lr_tree[t])
            # J‐block for step t lives at indices [time_steps+3 + t*num_J : time_steps+3 + (t+1)*num_J]
            start = time_steps + 3 + t * num_J
            end   = start + num_J
            J_block = lr_tree[start:end]
            avg_J   = float(jnp.mean(J_block))
            # Print each J element in the specified order using the qubit indices
            J_elements = []
            # for i, r in enumerate(reserv_qubits):
            for j, c in enumerate(ctrl_qubits):
                for i, r in enumerate(reserv_qubits):
                    # J_index = f"J_{{{r},{c}}}={J_block[i * len(ctrl_qubits) + j]:.2e}"  # Access the element by index
                    J_index = f"J({r},{c})={J_block[i * len(ctrl_qubits) + j]:.2e}"  # Access the element by index
                    J_elements.append(J_index)
            
            print(f" step {t:2d}: t_lr={tau_lr:.2e},  avg(J_lr)={avg_J:.2e},  " + ", ".join(J_elements))
    
   
    
    opt_descr, opt = get_optimizer(
        case=case_num,
        opt_lr     = opt_lr,       # either a dict or (lr_tree,assignment_mask)
        time_steps = time_steps,   # ensure get_optimizer signature includes time_steps
        num_epochs = num_epochs,
        factor=FACTOR,
        patience=PATIENCE,
        rtol=RTOL,
        atol=ATOL,
        cooldown=COOLDOWN,
        accumulation_size=ACCUMULATION_SIZE,
        min_scale=MIN_SCALE,
        warmup_steps = 400,
        # warmup_fraction = .15,
        )
   
    
    if case_num >0:
        params = {
            "t": params[:time_steps],
            "h": params[time_steps:time_steps + 3],
            "J": params[time_steps + 3:]
        }
    
    # Define the optimization update function
    @jit
    def update(params, opt_state, X, y, step):
        # 1) Compute loss and gradients
        if case_num == 0:
            # params is a flat vector
            loss, grads = jax.value_and_grad(cost_func)(params, X, y)
            grads_pytree = grads
        else:
            # params is a pytree of three pieces
            flat = jnp.concatenate([params["t"], params["h"], params["J"]])
            loss, grads_flat = jax.value_and_grad(cost_func)(flat, X, y)
            grads_pytree = {
                "t": grads_flat[:time_steps],
                "h": grads_flat[time_steps:time_steps+3],
                "J": grads_flat[time_steps+3:]
            }
            grads = grads_flat

        # 2) Pass both the loss (`value`) and the iteration (`step`) into opt.update
        updates, new_opt_state = opt.update(
            grads_pytree,
            opt_state,
            params=params,
            value=loss,
            step=step
        )

        # 3) Apply the updates
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss, grads
   
    print("________________________________________________________________________________")
    print(f"Starting optimization for {gate_name}(epochs: {num_epochs}) T = {time_steps}, N_r = {N_reserv}, N_bath = {num_bath}...\n")
    print("________________________________________________________________________________")
    # print(f"Initial Loss: {init_loss:.4f}, initial_gradients: {np.mean(np.abs(init_grads))}. Time: {dt:.2e}")
    # print("sample rates:", lr_tree[:5].tolist())
    # print(f"per-param learning-rate tree: mean={mean:.3e}, var={var:.3e}")


    costs = []
    param_per_epoch,grads_per_epoch = [],[]
 
    
    opt_state = opt.init(params)
    # for idx,st in enumerate(opt_state):
    #     print(f"{idx}: ", st)

    threshold_counts,acceleration,consecutive_improvement_count,consecutive_threshold_limit,rocs = 0,0.0,0,4,[]

    backup_params = init_params

    s = time.time()
    full_s =s
    training_state_metrics = {}
    a_condition_set, add_more,false_improvement,improvement,prev_cost, second_prev_cost = False,False,False,True,float('inf'), float('inf')
    a_threshold =  0.0
    stored_epoch = None
    threshold_cond1, threshold_cond2 = [],[]
   
    
    backup_cost_check,backup_cost,min_cost,backup_epoch = cost,cost,float('inf'),0   
    num_reductions, lr_scale = 0, 1.0
    plateau_count_history, avg_value_history, best_value_history, cooldown_count_history, scale_history, lrs_t_history, lrs_h_history, lrs_J_history, lr_hist = [[] for _ in range(9)]
    cv_B_per_epoch, mom_per_epoch = {},{}
    epoch = 0
    while epoch < num_epochs or improvement:
        if case_num in [2,3]:
            # unpack the three MaskedStates
            _, mask_t_state, mask_h_state, mask_J_state = opt_state
            plateau_t = mask_t_state.inner_state[1]
            plateau_h = mask_h_state.inner_state[1]
            plateau_J = mask_J_state.inner_state[1]
            plateau_states = {
                't': plateau_t,
                'h': plateau_h,
                'J': plateau_J,
            }
        params, opt_state, cost, grad = update(
            params,
            opt_state,
            input_states,
            target_states,
            epoch
        )
        if 'grad_groups_history' not in locals():
            grad_groups_history = []
        grad_groups_history.append(
            {k: v.tolist() for k, v in
            grad_group_dict(case_num, grad, time_steps, N_ctrl, N_reserv).items()}
        )
        
        flat_grads = jnp.ravel(grad)
        cv_B = coef_variation_by_block(case_num, flat_grads,
                            time_steps, N_ctrl, N_reserv)
        cv_B_per_epoch[epoch] = cv_B
        mom = log_moments(opt_state, case_num, time_steps,epoch)
        mom_per_epoch[epoch] = mom

        if case_num in [2,3]:
            # store each ReduceLROnPlateau metric per epoch (assuming same plateau_count across groups)
            plateau_count_history.append(int(plateau_t.plateau_count))
            avg_value_history.append(float(plateau_t.avg_value))
            best_value_history.append(float(plateau_t.best_value))
            cooldown_count_history.append(int(plateau_t.cooldown_count))
            # store per-group scales
            t_scale = float(plateau_t.scale)
            h_scale = float(plateau_h.scale)
            J_scale = float(plateau_J.scale)
            scale_history.append(t_scale)
            # scale_history.append((t_scale, h_scale, J_scale))
            assert t_scale == J_scale == h_scale
            
            # store per-group LR vectors
            lrs_t_history.append([float(x) for x in (opt_lr['t'] * t_scale).tolist()])
            lrs_h_history.append([float(x) for x in (opt_lr['h'] * h_scale).tolist()])
            lrs_J_history.append([float(x) for x in (opt_lr['J'] * J_scale).tolist()])
            all_lrs = jnp.concatenate([opt_lr["t"], opt_lr["h"], opt_lr["J"]], dtype=jnp.float64)
            lr_hist.append([float(x) for x in (all_lrs * J_scale).tolist()])
            
            if t_scale < lr_scale:
                lr_scale = t_scale
                num_reductions += 1
                print( 
                        f" -> Reduced scale from {scale_history[-2]} to {scale_history[-1]} at epoch {epoch}:\n"
                        f"  - pc: {plateau_count_history[-2]} (new: {plateau_count_history[-1]})"
                        f"  - new best: {float(best_value_history[-1]):.3e}, diff: {best_value_history[-2]-best_value_history[-1]:.3e},\n"
                        # f"  - best:        {float(best):.2e},\n"
                        # f"  - threshold:   {float(best * (1 - RTOL) + RTOL):.2e}"
                        # f"  -> new LR stats: min={lr_min:.2e}, max={lr_max:.2e}, "
                        # f"mean={lr_mean:.2e}, var={lr_var:.2e}"
                    )
                # for group, ps in plateau_states.items():
                
                #     lrs     = opt_lr[group] * lr_scale
                #     lr_min  = float(jnp.min(lrs))
                #     lr_max  = float(jnp.max(lrs))
                #     lr_mean = float(jnp.mean(lrs))
                #     lr_var  = float(jnp.var(lrs))

                    # print( 
                    #     f"{group}: Reduced scale to {scale} at epoch {epoch}:\n"
                    #     # f"  - current_avg: {float(current_avgs[group]):.2e},\n"
                    #     # f"  - best:        {float(best):.2e},\n"
                    #     f"  - threshold:   {float(best * (1 - RTOL) + RTOL):.2e}"
                    #     f"  -> new LR stats: min={lr_min:.2e}, max={lr_max:.2e}, "
                    #     f"mean={lr_mean:.2e}, var={lr_var:.2e}"
                    # )
                    
            

        
        if epoch > 1:
            var_grad = jnp.var(grad,ddof=1)
            mean_grad = jnp.mean(jnp.abs(grad))
            if epoch >5:
                threshold_cond1.append(np.abs(mean_grad))
                threshold_cond2.append(var_grad)
            if epoch == 15:
                initial_meangrad = jnp.mean(np.array(threshold_cond1))
                initial_vargrad = jnp.mean(np.array(threshold_cond2))
                cond1  = initial_meangrad * 1e-1
                # print(f"    - setting cond1: initial mean(grad) {initial_meangrad:2e}, threshold: {cond1:2e}")
                cond2 = initial_vargrad * 1e-2
                # print(f"    - setting cond2: initial var(grad) {initial_vargrad:2e}, threshold: {cond2:2e}")
            
            acceleration = get_rate_of_improvement(cost,prev_cost,second_prev_cost)
            if epoch >= 25 and not a_condition_set and acceleration < 0.0:
                average_roc = jnp.mean(np.array(rocs[10:]))
                a_marked = jnp.abs(average_roc)
                a_threshold = max(a_marked * 1e-3, 1e-7)
              
                a_condition_set = True
            rocs.append(acceleration)

        # Store parameters and cost for analysis
        param_per_epoch.append(params)
        costs.append(cost)
        grads_per_epoch.append(grad)
        # Logging
        max_abs_grad = jnp.max(jnp.abs(grad))
        if epoch == 0 or (epoch + 1) % 250 == 0:
            var_grad = jnp.var(grad,ddof=1)
            mean_grad = jnp.mean(jnp.abs(grad))
            e = time.time()
            cv_str = ", ".join(f"{k}: {v:.4f}" for k, v in cv_B.items())
            
            # print(f"mom: {mom}")
            mom_str = ", ".join(
                f"{k}: ({m_val:.3e}, {v_val:.3e})"
                for k, (m_val, v_val) in mom.items()
            )
            
            epoch_time = e - s
           
            if cost < 1e-3:
                print(f'Epoch {epoch + 1} --- cost: {cost:.3e}, '
                    # f"best: {last_best:.4e}, avg: {last_avg:.4e}, "
                    # f'plateau_count: {last_pc}, '
                    f'CV_B: {{{cv_str}}}, '
                    f'‖m‖,‖v‖ per block: {mom_str}, '
                    f'[t: {epoch_time:.1f}s]'
                    )
            else:
                print(f'Epoch {epoch + 1} --- cost: {cost:.4f}, '
                # print(f'Epoch {epoch + 1} --- cost: {cost:.4f}, best={best_val:.4f}, avg: {current_avg:.4f}, lr={learning_rates[-1]:.4f} [{plateau_state.scale:.3f}], '
                    # f"best: {last_best:.4e}, avg: {last_avg:.4e}, "
                    # f'plateau_count: {last_pc}, '
                    f'CV_B: {{{cv_str}}}, '
                    f'‖m‖,‖v‖ per block: {mom_str}, '
                    f'[t: {epoch_time:.1f}s]'
                    )
           
        
            s = time.time()

        if cost < prev_cost:
            
            improvement = True
            consecutive_improvement_count += 1
            # current_cost_check = cost
            if case_num > 0:
                params_flat = jnp.concatenate([params["t"], params["h"], params["J"]])
                current_cost_check = cost_func(params_flat, input_states, target_states)
            else:
                current_cost_check = cost_func(params, input_states, target_states)
            # if np.abs(backup_cost_check-backup_cost) > 1e-6:
            #     print(f"Back up cost different then its check. diff: {backup_cost_check-backup_cost:.3e}\nbackup params: {backup_params}")
            if current_cost_check < backup_cost:
                # print(f"Epoch {epoch}: Valid improvement found. Updating backup params: {backup_cost:.2e} > {current_cost_check:.2e}")
                backup_cost = current_cost_check
                backup_params = params
                false_improvement = False
                backup_epoch = epoch
            if false_improvement:
                print(f"Epoch {epoch}: False improvement detected, backup params not updated. Difference: {current_cost_check- backup_cost:.2e}")
                false_improvement = True
                 
        else:
            # print(f"    backup_cost: {backup_cost:.6f}")
            improvement = False  # Stop if no improvement
            consecutive_improvement_count = 0  # Reset the improvement count if no improvement

        # Termination check
        if threshold_counts >= consecutive_threshold_limit:
            print(f"Terminating optimization: cost {cost} is below the threshold {cost_threshold} for {consecutive_threshold_limit} consecutive epochs without improvement.")
            break
        # Check if there is improvement
        second_prev_cost = prev_cost  # Move previous cost back one step
        prev_cost = cost  # Update previous cost with the current cost

        
        # Apply tau parameter constraint (must be > 0.0)
        if case_num > 0:
            tau_params = params['t']
            for i in range(time_steps):
  
                if tau_params[i] < 0:
                    tau_params = tau_params.at[i].set(np.abs(tau_params[i]))
                    params.update({'t': tau_params})
        else: 
            for i in range(time_steps):
               
                if params[i] < 0:
                    params = params.at[i].set(np.abs(params[i]))
                   
        var_condition= jnp.var(grad,ddof=1) < 1e-14
        gradient_condition= max(jnp.abs(grad)) < 1e-8
        epoch_cond = epoch >= 2*num_epochs
        # plateau_state = opt_state[-1]
        if gradient_condition or var_condition or epoch_cond:
            if epoch_cond:
                print(f"Epoch greater than max. Ending opt at epoch: {epoch}")
            if var_condition:
                print(f"Variance of the gradients below threshold [{np.var(grad,ddof=1):.1e}], thresh:  1e-10. Ending opt at epoch: {epoch}")
            if gradient_condition:
                print(f"Magnitude of maximum gradient is less than threshold [{max(jnp.abs(grad)):.1e}]. Ending opt at epoch: {epoch}")

            break
        epoch += 1  # Increment epoch count


    if backup_cost < cost and not epoch < num_epochs and backup_epoch < epoch - 25:
        print(f"backup cost (epoch: {backup_epoch}) is better with: {backup_cost:.2e} <  {cost:.2e}: {backup_cost < cost}")
        # print(f"recomputed cost (i.e. cost_func(backup_params,input_states, target_states)): {cost_func(backup_params,input_states, target_states)}")
        # print(f"cost_func(params, input_states,target_states): {cost_func(params, input_states,target_states)}")
        # print(f"final_test(backup_params,test_in, test_targ): {final_test(backup_params,test_in, test_targ)}")
        # print(f"final_test(params,test_in, test_targ): {final_test(params,test_in, test_targ)}")
        params = backup_params

    full_e = time.time()

    epoch_time = full_e - full_s
    print(f"Time optimizing: {epoch_time}, total number or lr reductions: {num_reductions}")
    if case_num == 0:
        def final_test(params,test_in,test_targ):
            params = jnp.asarray(params, dtype=jnp.float64)
            X = jnp.asarray(test_in, dtype=jnp.complex128)
            y = jnp.asarray(test_targ, dtype=jnp.complex128)
            batched_output_states = vcircuit(params, X)

            fidelities = jax.vmap(qml.math.fidelity)(batched_output_states, y)
            # fidelities = jnp.clip(fidelities, 0.0, 1.0)

            return fidelities
    if case_num >0:
        def final_test(params,test_in,test_targ):
            params_flat = jnp.concatenate([params["t"], params["h"], params["J"]], dtype=jnp.float64)
            X = jnp.asarray(test_in, dtype=jnp.complex128)
            y = jnp.asarray(test_targ, dtype=jnp.complex128)
            batched_output_states = vcircuit(params_flat, X)
            fidelities = jax.vmap(qml.math.fidelity)(batched_output_states, y)
            # fidelities = jnp.clip(fidelities, 0.0, 1.0)

            return fidelities
       
    testing_results = final_test(params,test_in, test_targ)
    
    f64 = np.array(testing_results, dtype=np.float64)
    infids = 1.0 - f64
   
    avg_fidelity = np.mean(f64)
    if 1.-avg_fidelity <1e-4:
        print(f'Avg Fidelity: {avg_fidelity:.8e}, Err: {float(np.log10(infids).mean()):.5f}')
    else: 
        print(f'Avg Fidelity: {avg_fidelity:.5f}')
     
  

    print("\nAverage Final Fidelity: ", f64.mean())
    data = {'Gate':base64.b64encode(pickle.dumps(gate)).decode('utf-8'),
            'opt_description': opt_descr,
            'case_num':case_num,
            'specs':specs,
            'epochs': num_epochs,
           
            'rocs':rocs,
            'min_var_indices':min_var_indices,
            'replacement_indices':replacement_indices,
            'a_marked': a_marked,
            'stored_epoch': stored_epoch,
            'backup_epoch': backup_epoch,
            'preopt_results':preopt_results,
            'selected_indices':selected_indices,
            'trotter_step': time_steps,
            'controls': N_ctrl, 
            'reservoirs': N_reserv,
            'N_train': N_train,
            'init_params_dict': init_params_dict,
            'init_params': init_params,
            "avg_fidelity":   float(f64.mean()),
            "avg_infidelity": float(infids.mean()),
            "avg_log_error":  float(np.log10(infids).mean()),
            "testing_results": f64.tolist(),
            "infidelities":    infids.tolist(),
            'costs': costs,
            'params_per_epoch':param_per_epoch,
            'training_states': input_states,
            'opt_params': params,
            'opt_lr': opt_lr,
            'cv_B_per_epoch':cv_B_per_epoch,
            'mom_per_epoch':mom_per_epoch,
            'group_opt_lr_tree':lr_tree,
            'init_lr_groups': {k: v.tolist() for k, v in lr_groups_init.items()},
            
            'grad_groups_history': grad_groups_history,   # list-of-dicts across epochs
            'grads_per_epoch':grads_per_epoch,
            'lrs_t':           lrs_t_history,
            'lrs_h':           lrs_h_history,
            'lrs_J':           lrs_J_history,
            'lrs_all': lr_hist,
            'plateau_count_history':plateau_count_history,
            'scale_history':scale_history,
            'avg_value_history':avg_value_history,
            'best_value_history':best_value_history,
            'cooldown_count_history':cooldown_count_history,
            'training_state_metrics':training_state_metrics,
            'depth':circuit_depth,
            'total_time':epoch_time,
            'init_grads':init_grads

                
                
            }
     
    now = datetime.now()
    print(f"Saving results to {filename}. \nDate/time: ", now.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"******************************************************************************************************************************\n\n")
    df = pd.DataFrame([data])
    while os.path.exists(filename):
        name, ext = filename.rsplit('.', 1)
        filename = f"{name}_.{ext}"

    with open(filename, 'wb') as f:
        pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)


 

if __name__ == '__main__':


    

    
    # run below 
    N_ctrl = 2
   
   

    # trots = [1,15,20,25,30,35,40]
    
    trots = [4,6,8,10,12,16,20,24,28]
    
    # trots = [8,12,20,24,28]
    trots = [1,2,3,4,5]
    trots = [8,10,16,20,24]
    
    # trots = [4,8,10,12,14,16,18,20,24,28]
    

    # res = [1, 2, 3]
    if N_ctrl == 1:
        trots = [1,2,3,4,5,6,7,8]
        res = [1,2]
    elif N_ctrl == 2:
        trots = [1,4,8,10,12,14,16,18,20,22,24]
        trots = [6,26,28]
        res = [1,2]
    elif N_ctrl == 3:
        trots = [25,28,30,32,35,38,40,45]
        res = [1]

    # trots = [15,20,25,30,35,40,45,50]
    # trots = [10,18,19,20,22]
    # trots = [10,14,20,24,28]
    # trots = [24,28]
    # trots = [28,33,50]

    
    # res = [1]
  

    
    




    num_epochs = 1500
    N_train = 20
    add=0


     # optimization protocols, leave only one uncommented (the selected protocol)
    #  (see `get_optimizer` description for more details)
    # case_num = 0 # "per-param Adam"
    case_num = 1 # "masked per group adam"

    # case_num = 2 # "masked per group Adam + delayed ReduceLROnPlateau"
    # case_num = 3 #"group Adam + cost-threshold shrink"
    # case_num = 5 #  slice-wise independent per‐group Adam
    # case_num = 6 # slice-wise  + (Masked) per‐group Adam
    patience= 5
    accum_size = 5
    rtol = 1e-4
    atol=1e-7
    scale_by_num_train=False
    per_block_target_update=True
    t_bound_key = 'pi'
    t_bound_dict = {'pi': np.pi,
    '1':1,
    '2pi':2*np.pi,
    }
    t_bound = t_bound_dict[t_bound_key]
    # folder = f'./analog_results_trainable_global/trainsize_{N_train+add}_epoch{num_epochs}_per_param_opt_.1k/'
    if case_num in [2,3]:
        folder = f'./analog_results_trainable_global/trainsize_{N_train+add}_epoch{num_epochs}/case_{case_num}/PATIENCE{patience}_ACCUMULATION{accum_size}/ATOL_1e-7/'
        # folder = f'./analog_results_trainable_global/trainsize_{N_train+add}_epoch{num_epochs}/case_{case_num}/PATIENCE{patience}_ACCUMULATION{accum_size}/'
    else:
        folder = f'./analog_results/trainsize_{N_train+add}_epoch{num_epochs}/case_{case_num}/tgt_granular_{per_block_target_update}/t={t_bound_key}/scale={scale_by_num_train}/'
        # folder = f'./analog_results_trainable_global/trainsize_{N_train+add}_epoch{num_epochs}/case_{case_num}/'
    # folder = f'./analog_results_trainable_global/trainsize_{N_train}_epoch{num_epochs}_gradientclip_beta0.999/'

    gates_random = []
    baths = [False]
    num_baths = [0]


    for i in range(20):
        U = random_unitary(2**N_ctrl, i).to_matrix()
        #pprint(Matrix(np.array(U)))
        g = partial(qml.QubitUnitary, U=U)
        g.num_wires = N_ctrl
        g.name = f'U{N_ctrl}_'+str(i)
        gates_random.append(g)

  
    for gate_idx,gate in enumerate(gates_random):



        for time_steps in trots:
          
            
            
            
            for N_reserv in res:
                # if N_reserv == 2 and time_steps in [5,7]:
                #     continue
                
                N =N_ctrl+N_reserv
                
                #folder = f'./param_initialization/Nc{N_ctrl}_Nr{N_reserv}_dt{time_steps}/fixed_params4/test7/'
                for num_bath,bath in zip(num_baths,baths):
                    params_key_seed = gate_idx*121 * N_reserv + 12345 * time_steps *N_reserv
                    params_key = jax.random.PRNGKey(params_key_seed)
                    dataset_seed = N_ctrl * gate_idx + gate_idx**2 + N_ctrl
                    dataset_key = jax.random.PRNGKey(dataset_seed)
                    main_params = jax.random.uniform(params_key, shape=(3 + (N_ctrl * N_reserv) * time_steps,), minval=-np.pi, maxval=np.pi)
                    
      
                    params_key, params_subkey1, params_subkey2 = jax.random.split(params_key, 3)
                    
                    
                    time_step_params = jax.random.uniform(params_key, shape=(time_steps,), minval=0, maxval=t_bound)
                    K_half = jax.random.uniform(params_subkey1, (N, N))
                    K = (K_half + K_half.T) / 2  # making the matrix symmetric
                    K = 2. * K - 1.
                    init_params_dict = {'K_coef': jnp.asarray(K)}
                 
                    # Combine the two parts
                    params = jnp.concatenate([time_step_params, main_params])

                    run_test(
                        params,
                        num_epochs,
                        N_reserv,
                        N_ctrl,
                        time_steps,
                        N_train,
                        folder,
                        gate,
                        gate.name,
                        bath,
                        num_bath,
                        case_num=case_num,
                        init_params_dict=init_params_dict,
                        dataset_key=dataset_key,
                        PATIENCE=patience,
                        ACCUMULATION_SIZE=accum_size,
                        scale_by_num_train=scale_by_num_train,
                        per_block_target_update=per_block_target_update
                    )