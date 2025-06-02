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
    then scatter back into a full-length vector lr_tree of shape [D].

    We follow a “LARS” recipe:
      (1) Count total params: D = T + 3 + T*(N_ctrl*N_reserv).
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
      - lr_tree: array of shape [D], listing the per‐param learning rate
      - assignment_mask: an integer array of shape [D], giving a “group ID” per index:
            0…(T−1)  → slice 0…slice T−1,
           T         → the “h” group
           (no other IDs)
        (Use `assignment_mask[i]` to see which slice/h‐group index i belongs to.)
    """
    D = grads.shape[0]
    num_J = N_ctrl * N_reserv

    # 1) Compute total # params and global base η₀
    N_params = D
    eta0 = eta_base * (N_train / L_ref) * jnp.sqrt(P_ref / float(N_params))
    # note: if eta0 > max_lr, we'll clip later at per‐group level

    # 2) Build an “assignment mask” of length D:
    #    indices 0..(time_steps-1)       → each τ_t (we’ll give it group ID t)
    #    indices time_steps..time_steps+2 → the 3‐vector h (we assign ID = time_steps)
    #    indices time_steps+3..D-1       → the J‐blocks, sliced in chunks of size num_J
    idx = jnp.arange(D)
    mask_tau    = idx < time_steps
    mask_h      = (idx >= time_steps) & (idx < time_steps + 3)
    mask_J_flat = idx >= (time_steps + 3)

    # build an integer array “group” of shape [D]:
    #   group[i] = t in [0..T-1] if i is in τ‐block for slice t
    #   group[i] = T      if i is in h‐block
    #   group[i] = t      if i is in J‐block for slice t (we must compute t from index)
    #
    # Indices for J‐blocks run from (time_steps+3) up to (time_steps+3 + T*num_J -1).
    def assign_group(i):
        # i: a scalar index in [0..D-1]
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
    group_ids    = jax.vmap(assign_group)(idx)  # shape [D], each entry in [0..T].
    assignment_mask = group_ids  # we’ll return this so downstream code knows the grouping.

    # 3) For each group in 0..T (where group=T means “h”), collect all grads/params belonging to that group,
    #    compute ‖w_group‖₂, ‖g_group‖₂, then η_group = η₀ * (‖w‖/ (‖g‖ + ε)), clipped.
    def compute_group_lr(gid):
        # gather indices where assignment_mask == gid
        idxs = jnp.where(group_ids == gid, size=D)[0]
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
        print(f" total params  D = {D},  base η₀ = {eta0:.3e}")
        for gid in range(time_steps):
            print(f"  slice {gid:2d}: η_slice = {float(group_lr_tree[gid]):.3e}")
        print(f"  h‐group: η_h = {float(group_lr_tree[time_steps]):.3e}")
        print(f" lr_tree stats:  min={float(jnp.min(lr_tree)):.3e},  max={float(jnp.max(lr_tree)):.3e},  mean={float(jnp.mean(lr_tree)):.3e}")
        print(f"------------------------------\n")

    return lr_tree, assignment_mask


from optax._src import base as optax_base
from reduce_on_plateau import reduce_on_plateau
from typing import NamedTuple
def _l2(tree):
    return jnp.sqrt(sum(jnp.sum(jnp.square(x))
                        for x in jax.tree_util.tree_leaves(tree)))
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
        g_norm = _l2(updates)
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
def _slice_masks(time_steps: int, N_ctrl: int, N_reserv: int):
    """
    Return helper closures for slice-wise grouping.

        elem_mask(t)  -> Bool[D] mask for slice t (τᵗ ∪ Jᵗ)
        h_mask()      -> Bool[D] mask for the global h-vector
    """
    num_J = N_ctrl * N_reserv
    T     = time_steps

    def elem_mask(t, D):
        idx      = jnp.arange(D)
        tau_m    = idx == t
        start_J  = T + 3 + t * num_J
        end_J    = start_J + num_J
        J_m      = (idx >= start_J) & (idx < end_J)
        return tau_m | J_m          # τₜ  ∪  J^{(t)}

    def h_mask(D):
        idx = jnp.arange(D)
        return (idx >= T) & (idx < T + 3)

    return elem_mask, h_mask


def lr_group_dict(case_num: int,
                  lr_tree: jnp.ndarray,
                  time_steps: int,
                  N_ctrl: int,
                  N_reserv: int):
    """
    Return a *dict of arrays* with per-group learning-rates, ready to drop
    into your `data` payload.

      case 0-5 : keys = {"t","h","J"}
      case 6-7 : keys = {"h", "slice_0", …, "slice_{T-1}"}
    """
    if case_num <= 5:                    # τ / h / J grouping
        return {
            "t": lr_tree[:time_steps],
            "h": lr_tree[time_steps:time_steps + 3],
            "J": lr_tree[time_steps + 3:],
        }

    # slice-wise grouping
    D                 = lr_tree.shape[0]
    elem_mask, h_mask = _slice_masks(time_steps, N_ctrl, N_reserv)
    out               = {"h": lr_tree[h_mask(D)]}
    for t in range(time_steps):
        out[f"slice_{t}"] = lr_tree[elem_mask(t, D)]
    return out


def grad_group_dict(case_num: int,
                    grads_flat: jnp.ndarray,
                    time_steps: int,
                    N_ctrl: int,
                    N_reserv: int):
    """
    Same interface as `lr_group_dict` but for *gradients*.
    Call this **inside your training loop** each epoch *before* the optimiser
    is applied so you see true pre-update gradients.
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
    D                 = grads_flat.size
    elem_mask, h_mask = _slice_masks(time_steps, N_ctrl, N_reserv)
    out               = {"h": grads_flat[h_mask(D)]}
    for t in range(time_steps):
        out[f"slice_{t}"] = grads_flat[elem_mask(t, D)]
    return out
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
    

def get_groupwise_lr_trees(
    params: jnp.ndarray,
    grads: jnp.ndarray,
    num_train,
    NC,
    time_steps: int,
    max_lr: float = 0.2,
    debug: bool = False,
    scale_by_num_train = True,
    target_update: float = 0.05,
    eps: float = 1e-12,
) -> jnp.ndarray:
    """
    Args:
    grads:        [D,] gradient magnitudes at the initial params
    max_lr:       scalar upper bound on any lr
    time_steps:   T
    debug:        if True, prints medians, MADs, r’s and a few sample lrs
    
    Returns:
     lr_tree of shape [D,], where each index
                  has its own lr_i taken from its group.
    """
    grad_magnitudes = jax.tree_util.tree_map(lambda g: jnp.abs(g) + 1e-12, grads)
    D = grad_magnitudes.shape[0]
    idx = jnp.arange(D)
    
    max_grad = float(jnp.max(grads))
    # choose a “target” maximum update size, e.g. 10% of a typical parameter
    # target_update =0.05 
    target_update =  jnp.mean(np.abs(params)) * 0.05
    # then set max_lr so that max_lr * max_grad ≈ target_update
    eta_max = target_update / (max_grad + 1e-12)

    print(f"target_update: {target_update:.5f}, eta_max: {eta_max:.5f}")
    
    if eta_max>max_lr:
        eta_max = max_lr
        print(f" - eta_max > {max_lr}... Updating bound: {eta_max}")
    
    elif eta_max < max_lr*0.1:
        eta_max=max_lr*0.1 
        print(f" - eta_max < {max_lr*0.1}... Updating bound: {eta_max}")
    
    # 1) masks
    mask_tau = idx < time_steps
    mask_h   = (idx >= time_steps) & (idx < time_steps + 3)
    mask_J   = idx >= time_steps + 3

    # 2) extract groups
    g_tau = grad_magnitudes[mask_tau]
    g_h   = grad_magnitudes[mask_h]
    g_J   = grad_magnitudes[mask_J]
    def get_base_lr(grad_norm,target_update,eps = 1e-12):
        return jnp.where(grad_norm > 0.0,
                        target_update / (grad_norm + eps),
                        1e-3)
    # 3) medians & MADs
    def med_mad(x):
        med = jnp.median(x)
        mad = jnp.median(jnp.abs(x - med))
        return med, mad
    
    median_all, mad_all = med_mad(grad_magnitudes)

    grad_norm_all = jnp.linalg.norm(grad_magnitudes)

    g_tau_norm, g_h_norm, g_J_norm = jnp.linalg.norm(g_tau) +eps, jnp.linalg.norm(g_h)+eps, jnp.linalg.norm(g_J)+eps
    sum_norms = g_tau_norm+g_h_norm+g_J_norm
    tau_base_lr, h_base_lr, J_base_lr = get_base_lr(g_tau_norm,target_update), get_base_lr(g_h_norm,target_update),get_base_lr(g_J_norm,target_update) 

    # print(f" All: med = {median_all:.3e}, mad={mad_all:.3e}, grad norm={grad_norm_all:.3e}. Upper bound set to {eta_max:.4f}")
    if debug:
        med_all,mad_all = med_mad(grad_magnitudes)
        
        print(f"\n--- global‐norm anchor ---")
        print(f"  grad_norm_all = {grad_norm_all:.3e}, sum(norms)={sum_norms:.3e}, Upper bound set to {eta_max:.4f}")
        # print(f"  (target_update = {target_update:.3e})")
        print(f"  median(|g|) = {med_all:.3e},  MAD(|g|) = {mad_all:.3e}")
        print(f" alphas:      τ={float(g_tau_norm/sum_norms):.3e}, h={float(g_h_norm/sum_norms):.3e}, J={float(g_J_norm/sum_norms):.3e}")
        print(f" base lrs:      τ={float(tau_base_lr):.3e}, h={float(h_base_lr):.3e}, J={float(J_base_lr):.3e}")

        print("-----------------------------\n")
   
    med_tau, mad_tau = med_mad(g_tau)
    med_h,   mad_h   = med_mad(g_h)
    med_J,   mad_J   = med_mad(g_J)
    if scale_by_num_train:
        if num_train >= 20:
            factor = NC/8
            
        elif num_train <= 15 and num_train > 10:
            factor = NC/4
        elif num_train <= 10:
            factor = NC/2
    else:
        factor = 1.0

    r_tau = (med_tau + mad_tau) * factor
    r_h   = (med_h   + mad_h)* factor
    r_J   = (med_J   + mad_J)* factor


    # per‐group rule
    lr_tau = r_tau * eta_max / (g_tau + r_tau + 1e-12)
    lr_h   = r_h   * eta_max / (g_h   + r_h   + 1e-12)
    lr_J   = r_J   * eta_max / (g_J   + r_J   + 1e-12)


    lr_tree = jnp.zeros_like(grad_magnitudes)
    lr_tree = lr_tree.at[mask_tau].set(lr_tau)
    lr_tree = lr_tree.at[mask_h].set(lr_h)
    lr_tree = lr_tree.at[mask_J].set(lr_J)
    
    
    if debug:
        # report final per‐group stats
        print(f"\n--- groupwise‐LR debug (per‐param values) ---")
        print(f" t‐group: mean={float(jnp.mean(lr_tau)):.3e},  min={float(jnp.min(lr_tau)):.3e},  max={float(jnp.max(lr_tau)):.3e}")
        print(f" h‐group: mean={float(jnp.mean(lr_h)):.3e},    min={float(jnp.min(lr_h)):.3e},    max={float(jnp.max(lr_h)):.3e}")
        print(f" J‐group: mean={float(jnp.mean(lr_J)):.3e},    min={float(jnp.min(lr_J)):.3e},    max={float(jnp.max(lr_J)):.3e}")
        print("-------------------------------------------\n")

    # Assert that no elements in lr_tree are less than min_lr
    min_lr = 1e-6
    # Assert that no elements in lr_tree are less than min_lr
    below_min_lr_tau = jnp.where(mask_tau)[0][lr_tree[mask_tau] < min_lr]
    below_min_lr_h = jnp.where(mask_h)[0][lr_tree[mask_h] < min_lr]
    below_min_lr_J = jnp.where(mask_J)[0][lr_tree[mask_J] < min_lr]

    assert jnp.all(lr_tree >= min_lr), (
        f"Assertion failed: Some learning rates are less than {min_lr:.3e}. "
        f"Indices and values in mask_tau: {list(zip(below_min_lr_tau.tolist(), (lr_tree[mask_tau][lr_tree[mask_tau] < min_lr]).tolist()))}, "
        f"Indices and values in mask_h: {list(zip(below_min_lr_h.tolist(), (lr_tree[mask_h][lr_tree[mask_h] < min_lr]).tolist()))}, "
        f"Indices and values in mask_J: {list(zip(below_min_lr_J.tolist(), (lr_tree[mask_J][lr_tree[mask_J] < min_lr]).tolist()))}"
    )

  
    return lr_tree, mask_tau, mask_h, mask_J

def _rms(vec: jnp.ndarray) -> float:
    return float(jnp.linalg.norm(vec) / jnp.sqrt(vec.size + 1e-12))

"""

1. Contiguous Block Learning Rate Allocation: This name emphasizes the method of dividing the parameter vector into contiguous blocks (time durations, global field, and couplings) and assigning learning rates based on the characteristics of each block's gradients.
1. Contiguous Block Learning Rate Allocation: This approach divides the parameter vector into three contiguous blocks—time durations ($\vec{\tau}$), global field ($\vec{h}$), and couplings ($\vec{J}$)—and assigns learning rates based on the characteristics of each block's gradients.

2. Parameter Type-Based Learning Rate Scaling: This method groups parameters by their type into three distinct blocks: the time durations ($\vec{\tau}$), the global field vector ($\vec{h}$), and the coupling parameters ($\vec{J}$). Each block's learning rate is derived from the median and median absolute deviation (MAD) of the gradients within that block, allowing for tailored updates based on parameter type.

3. Temporal Slice-Based Learning Rate Assignment: This approach organizes parameters into time-slice groups, where each time step $t$ includes the corresponding time duration ($\tau_t$), the coupling parameters for that time step (${J_{kl}^{(t)}}$), and the constant global field vector ($\hat{h} = (h^x, h^y, h^z)$). Learning rates are computed for each slice based on the gradients associated with that specific time step, enabling dynamic adjustments that reflect the temporal structure of the optimization problem.
"""

"""
    Compute a per-parameter learning rate "tree" as follows:
        1) Split parameters into three contiguous blocks:
           - indices [0 : time_steps)           --> the "τ" (time‐durations) block
           - indices [time_steps : time_steps+3) --> the "h" (global field) block
           - indices [time_steps+3 : D)          --> the "J" (couplings) block
         Let grad_magnitudes = |grads| + 1e-12, so we never divide by zero exactly.

        2) Take the *global* 2-norm of all gradients:  grad_norm_all = ||grads||₂ and compute a
            global ceiling \eta_max so that no parameter‐update in the first-step
            update is larger than 'target_update' (in norm‐sense):
                eta_max = target_update / (grad_norm_all + eps)
        3) For each block $B\!\in\!\{\tau,h,J\}$ compute
                \[
                    r_B = \bigl(\mathrm{median}|g_B| + \mathrm{MAD}|g_B|\bigr)
                        \times \texttt{scale\_fac}.
                \]
                If \texttt{scale\_by\_num\_train} is \textsc{True} the scale factor is
                \[
                    \texttt{scale\_fac}=
                    \begin{cases}
                    N_C/8 & N_{\text{train}}\!\ge 20,\\[2pt]
                    N_C/4 & 11\!\le N_{\text{train}}\!\le 15,\\[2pt]
                    N_C/2 & N_{\text{train}}\!\le 10,
                    \end{cases}
                \]
                otherwise $\texttt{scale\_fac}=1$.
        4) Finally, assign each parameter i in block 'B' it's individual learning rate:
                    \[
                    \eta_i =
                    \eta_{\max}\;
                    \frac{r_B}{|g_i|+r_B+\varepsilon},
                    \qquad
                    \eta_i\in[10^{-6},\; \texttt{max\_lr}].
                \]
            The 2-norm normalises the effect of a single outlier gradient (which would otherwise shrink every \eta_i),
            yielding a larger usable \eta_max even when the gradient vector is sparse. 

        5) Return the full vector lr_tree of shape [D], plus the three masks
         (mask_tau, mask_h, mask_J) so you know which indices belonged to which group.
      
    """

def get_groupwise_lr_trees_new(
    params: jnp.ndarray,
    grads: jnp.ndarray,
    num_train: int,
    NC: int,
    max_lr: float = 0.2,
    time_steps: int = 10,
    debug: bool = False,
    scale_by_num_train: bool = False,
    target_update: float = 0.05,
    eps: float = 1e-12,
    factor: float = 1.0,
    per_block_target_update: bool = False,   # ← NEW
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute a per-parameter learning rate "tree" as follows:

      1) Take the *global* 2-norm of all gradients:  grad_norm_all = ||grads||₂.
         Define a single base_lr = target_update / (grad_norm_all + eps).
         This ensures that on step 1, no parameter‐update (in norm‐sense) is larger
         than "target_update".

      2) Split parameters into three contiguous blocks:
           - indices [0 : time_steps)           --> the "τ" (time‐durations) block
           - indices [time_steps : time_steps+3) --> the "h" (global field) block
           - indices [time_steps+3 : D)          --> the "J" (couplings) block
         Let grad_magnitudes = |grads| + 1e-12, so we never divide by zero exactly.

      3) Within each block ("τ", "h", "J") compute median and MAD of that block’s |grad|’s:
           r_group = (median(|g_group|) + MAD(|g_group|)) * (scale_by_num_train ? factor : 1).
         Here “factor” = {NC/8 if num_train>=20; NC/4 if 11<= num_train<=15; NC/2 if num_train<=10}.

      4) Finally, for each parameter i in block 'group', set
           lr_i = base_lr * [ r_group / (|g_i| + r_group + eps ) ].
         (Then clip lr_i into [1e-6, max_lr].)

      5) Return the full vector lr_tree of shape [D], plus the three masks
         (mask_tau, mask_h, mask_J) so you know which indices belonged to which group.

    Returns:
      lr_tree:      an array of shape [D], one learning‐rate per parameter
      mask_tau:     boolean mask for indices in the τ‐block
      mask_h:       boolean mask for indices in the h‐block
      mask_J:       boolean mask for indices in the J‐block
    """
    if per_block_target_update:
        tgt_tau = 0.75 * _rms(params[:time_steps])
        tgt_h   = 0.75 * _rms(params[time_steps:time_steps+3])
        tgt_J   = 0.75 * _rms(params[time_steps+3:])
    else:
        tgt_tau = tgt_h = tgt_J = float(target_update)
    grad_magnitudes = jax.tree_util.tree_map(lambda g: jnp.abs(g) + 1e-12, grads)
    D = grad_magnitudes.shape[0]
    idx = jnp.arange(D)

    
    mask_tau = idx < time_steps
    mask_h   = (idx >= time_steps) & (idx < time_steps + 3)
    mask_J   = idx >= time_steps + 3
    # extract group subsets
    g_tau, g_h, g_J = grad_magnitudes[mask_tau],grad_magnitudes[mask_h],grad_magnitudes[mask_J]
   
        
    # -----------------------------------------------------------------------
    # choose η (max lr) for each block
    if per_block_target_update:
        # one ceiling per block, based on that block’s gradient norm
        eta_tau = jnp.minimum(max_lr,
                            tgt_tau / (jnp.linalg.norm(g_tau) + eps))
        eta_h   = jnp.minimum(max_lr,
                            tgt_h   / (jnp.linalg.norm(g_h)   + eps))
        eta_J   = jnp.minimum(max_lr,
                            tgt_J   / (jnp.linalg.norm(g_J)   + eps))
    else:
        # single ceiling for all blocks (old behaviour)
        grad_norm_all = jnp.linalg.norm(grad_magnitudes)
        eta_max = jnp.minimum(max_lr,
                            target_update / (grad_norm_all + eps))
        eta_tau = eta_h = eta_J = eta_max
    # -----------------------------------------------------------------------


    

    # -----------------------------------------------------------------------
    def med_mad(x):
        med = jnp.median(x)
        mad = jnp.median(jnp.abs(x - med))
        return med, mad

    med_tau, mad_tau = med_mad(g_tau)
    med_h,   mad_h   = med_mad(g_h)
    med_J,   mad_J   = med_mad(g_J)

    r_tau = med_tau + mad_tau
    r_h   = med_h   + mad_h
    r_J   = med_J   + mad_J
    if debug:
        print(f"--- group statistics before LR‐scaling ---")
        print(f"  τ‐block:  median = {float(med_tau):.3e},  MAD = {float(mad_tau):.3e},  r_tau = {float(r_tau):.3e}")
        print(f"  h‐block:  median = {float(med_h):.3e},   MAD = {float(mad_h):.3e},   r_h   = {float(r_h):.3e}")
        print(f"  J‐block:  median = {float(med_J):.3e},   MAD = {float(mad_J):.3e},   r_J   = {float(r_J):.3e}")
        print("-------------------------------------------")
    # optional extra scaling (old “factor” heuristic)
    if scale_by_num_train:
        if   num_train >= 20: fac = NC / 8
        elif num_train >= 11: fac = NC / 4
        else:                 fac = NC / 2
        r_tau *= fac; r_h *= fac; r_J *= fac

    lr_tau = eta_tau * (r_tau / (g_tau + r_tau + eps))
    lr_h   = eta_h * (r_h   / (g_h   + r_h   + eps))
    lr_J   = eta_J * (r_J   / (g_J   + r_J   + eps))
    # learning-rate vectors
    lr_tau = jnp.clip(lr_tau, 1e-6, max_lr)
    lr_h   = jnp.clip(lr_h,   1e-6, max_lr)
    lr_J   = jnp.clip(lr_J,   1e-6, max_lr)

    lr_tree = jnp.zeros_like(grad_magnitudes)
    lr_tree = lr_tree.at[mask_tau].set(lr_tau)
    lr_tree = lr_tree.at[mask_h  ].set(lr_h  )
    lr_tree = lr_tree.at[mask_J  ].set(lr_J  )

    if debug:
        print("\n--- τ/h/J ceilings ---")
        print(f"  η_tau={float(eta_tau):.3e}, η_h={float(eta_h):.3e}, η_J={float(eta_J):.3e}")
        print(f" t‐group: mean={float(jnp.mean(lr_tau)):.3e},  min={float(jnp.min(lr_tau)):.3e},  max={float(jnp.max(lr_tau)):.3e}")
        print(f" h‐group: mean={float(jnp.mean(lr_h)):.3e},    min={float(jnp.min(lr_h)):.3e},    max={float(jnp.max(lr_h)):.3e}")
        print(f" J‐group: mean={float(jnp.mean(lr_J)):.3e},    min={float(jnp.min(lr_J)):.3e},    max={float(jnp.max(lr_J)):.3e}")
        print("-------------------------------------------")

    return lr_tree, mask_tau, mask_h, mask_J

def get_groupwise_lr_trees_slicewise(
    params:        jnp.ndarray,   # flat vector of length D = T + 3 + T*(N_ctrl*N_reserv)
    grads:         jnp.ndarray,   # flat gradient vector of length D
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
    per_block_target_update: bool = False,) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Like get_groupwise_lr_trees, but split into (T) slice‐groups + 1 h‐group:
       • slice t = { index t (tauₜ) } ∪ { indices for J^{(t)} } 
       • h‐group = indices [time_steps .. time_steps+2]
    
    Returns:
      lr_tree:          shape [D], one per-param learning rate
      assignment_mask:  shape [D], ints ∈ [0..T] telling which group each index belongs to:
                          0..(T−1) → slice 0..slice T−1,
                          T         → the “h” group.
    """
    D = grads.shape[0]
    idx = jnp.arange(D)
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

    assignment_mask = jax.vmap(_assign)(idx)          # [D] int32
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

# ────────────────────────────────────────────────────────────────────
# 1) coefficient-of-variation per block, works for both case-sets
# ────────────────────────────────────────────────────────────────────
def coef_variation_by_block(case_num, grads_flat,
                            time_steps, N_ctrl, N_reserv):
    """Return {block: CV} with block ∈ {"t","h","J"} or {"h","slice_0",…}."""
    blocks = grad_group_dict(case_num, grads_flat,
                             time_steps, N_ctrl, N_reserv)
    out = {}
    for name, vec in blocks.items():
        vec = jnp.asarray(vec)
        mu  = jnp.mean(jnp.abs(vec))
        sig = jnp.std(jnp.abs(vec))
        out[name] = float(sig / (mu + 1e-12))
    return out

from optax._src.transform import ScaleByAdamState  # already in optax
# ----------------------------------------------------------------------
try:
    # newest optax
    from optax._src.masking import MaskedNode
except ImportError:
    # for older optax versions
    from optax.transforms._masking import MaskedNode  # type: ignore

def _pytree_l2(tree):
    """Return sqrt(Σ‖leaf‖²) ignoring MaskedNode & scalars."""
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
    """Return the first ScaleByAdamState found inside a nested structure."""
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
    Return {block: (‖m‖₂, ‖v‖₂)}.
    Supports:
        • case 1-5 : three masked branches (t, h, J)
        • case 6-7 : one global Adam
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
    """
    Return an Optax optimizer (and a description) based on `case`.
    
    case 0:"per-param Adam" -> Adam with per-parameter initial learning rates (not grouped by type)
    case 1: 'masked per group adam' -> parameters grouped by type and updated by type via adam 
    case 2: "masked per group adam & ReduceLROnPlateau" -> case 1 but we scale down lrs when a barren plateau is detected
    case 3: grouped Adam + delayed ReduceLROnPlateau
    4  + cost-threshold shrink
      5  + low-grad-variance shrink
    If case==7: we expect opt_lr = (lr_tree, assignment_mask).  
    Here:
      - lr_tree is a flat [D]-vector of per-parameter learning rates
      - assignment_mask is an int array of shape [D], where each entry in [0..T] 
        labels which slice (0..T-1) or h-block (T) that index belongs to.
      - D = T + 3 + T * (N_ctrl * N_reserv).
    default: case 0
    """
    
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
            print(f"kwargs: {kwargs}")
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
    # NEW  :  case == 5  →  slice-wise masked Adam  +  per-param scaling
    # ------------------------------------------------------------------
    elif case == 5:
        """
        case 5  ─  slice-wise independent Adam
        --------------------------------------
        * Same LR vector (lr_tree) you built for case 6.
        * Each slice t and the h-vector get their own Adam moment buffers.
        * Final element-wise scaling keeps per-parameter rates intact.
        Expected opt_lr input:
            opt_lr == (lr_tree, assignment_mask)
                lr_tree         : flat [D] array
                assignment_mask : flat [D] ints in {0,…,T} (T = h-group)
        """
        desc = "slice-wise masked Adam + per-param LR"

        # ── unpack what run_test passed in ────────────────────────────
        lr_tree, assignment_mask = opt_lr      # tuple from get_groupwise_lr_trees_slicewise
        T        = time_steps
        num_J    = params["J"].shape[0] // T   # need this for mask construction
        D        = lr_tree.size

        # ------------------------------------------------------------------
        # 1) build pytree masks  {"t":Bool[T], "h":Bool[3], "J":Bool[T*num_J]}
        #    one such mask per group-id  (0..T-1 slices,  T = h-group)
        # ------------------------------------------------------------------
        masks_by_gid = {}
        for gid in range(T + 1):
            flat = (assignment_mask == gid)          # [D] bool
            mask_t = flat[:T]
            mask_h = flat[T:T+3]
            mask_J = flat[T+3:]
            masks_by_gid[gid] = {"t": mask_t, "h": mask_h, "J": mask_J}

        # ------------------------------------------------------------------
        # 2) element-wise learning-rate scaling, re-using helper from case 6
        # ------------------------------------------------------------------
        def per_param_lr(lr_pytree):
            def init_fn(_):               # stateless
                return ()
            def upd_fn(updates, state, params=None):
                scaled = jax.tree_util.tree_map(lambda g, lr: g * lr,
                                                updates, lr_pytree)
                return scaled, state
            return optax.GradientTransformation(init_fn, upd_fn)

        lr_pytree = {
            "t": lr_tree[:T],
            "h": lr_tree[T:T+3],
            "J": lr_tree[T+3:]
        }

        # ------------------------------------------------------------------
        # 3) one Adam per slice (LR = 1.0, moments independent) + per-param LR
        # ------------------------------------------------------------------
        per_group_adams = []
        for gid in range(T + 1):
            adam_gid   = optax.adam(learning_rate=1.0, b1=b1, b2=b2, eps=eps)
            masked     = optax.masked(adam_gid, masks_by_gid[gid])
            per_group_adams.append(masked)

        base_opt = optax.chain(
            optax.clip_by_global_norm(1.0),   # safety
            *per_group_adams,
            per_param_lr(lr_pytree)           # keep the original lr_tree
        )
        return desc, wrap(base_opt, passes_value=False, passes_step=False)
    elif case == 6:
        """
        case 6 ─ slice-wise median/MAD LR  →  masked Adam
        --------------------------------------------------
        opt_lr == (lr_tree, assignment_mask)

          lr_tree         : flat [D] array
          assignment_mask : flat [D] ints in {0,…,T}, where T labels the h-block
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

        
    elif case == 7:
        """
        case 7: slice‐wise LARS + masked Adam

        We expect opt_lr to be a tuple (lr_tree, assignment_mask) where:
          - lr_tree is a flat [D]-array of per-parameter LARS rates
          - assignment_mask is a flat [D]-array of ints in {0,…,T}, telling which slice (0..T−1)
            or “h” block (T) each index belongs to.

        Our parameter pytree is {"t": shape=[T],  "h": shape=[3],  "J": shape=[T * num_J]}.
        We build (T + 1) separate Adam optimizers, each masked to update only that slice or h-block.
        """

        desc = "slice‐wise LARS + masked Adam"

        # Unpack the tuple passed in as opt_lr:
        lr_tree, assignment_mask = opt_lr   # lr_tree: [D], assignment_mask: [D], ints ∈ [0..T]

        # Recover the number of slices T and num_J from params:
        T = time_steps
        # Number of J-couplings per slice = (total_length_of_params – T – 3) / T
        #   because params["J"].shape[0] == T * (N_ctrl * N_reserv)
        num_J = params["J"].shape[0] // T

        # We will construct one mask pytree {"t": Bool[T], "h": Bool[3], "J": Bool[T*num_J]}
        # for each group ID ∈ {0,1,…,T} where ID=T means "h"-block, ID=t means slice t.

        masks_by_group: dict[int, dict] = {}
        D = lr_tree.shape[0]
        # assignment_mask[i] ∈ [0..T] tells which group index i belongs to.

        for gid in range(T + 1):
            # Create a flat boolean mask of length D
            flat_mask = (assignment_mask == gid)  # shape [D], dtype=bool

            # Split into the three leaves of our pytree:
            #   indices 0..(T-1) in flat correspond to params["t"]
            #   indices T..(T+2) correspond to params["h"]
            #   indices (T+3)..(T+3 + T*num_J - 1) correspond to params["J"]
            mask_t = flat_mask[0 : T]                                # shape [T]
            mask_h = flat_mask[T : T + 3]                            # shape [3]
            mask_J = flat_mask[T + 3 : T + 3 + T * num_J]  # shape [T * num_J]

            masks_by_group[gid] = {"t": mask_t, "h": mask_h, "J": mask_J}

        # Helper to build an Adam optimizer (scalar LR) for group gid
        def make_group_adam(gid: int):
            # Find one representative index in lr_tree for this gid (they are all equal within group)
            rep_index = int(jnp.where(assignment_mask == gid, size=D)[0][0])
            lr_g = float(lr_tree[rep_index])
            return optax.inject_hyperparams(optax.adam)(
                learning_rate=lr_g, b1=b1, b2=b2, eps=eps
            )

        per_group_chains = []
        for gid in range(T + 1):
            adam_for_gid = make_group_adam(gid)
            mask_tree    = masks_by_group[gid]   # pytree {"t":Bool[T], "h":Bool[3], "J":Bool[T*num_J]}
            masked_adam  = optax.masked(adam_for_gid, mask_tree)
            per_group_chains.append(masked_adam)

        # Prepend a global clip_by_global_norm(1.0)
        base_opt = optax.chain(optax.clip_by_global_norm(1.0), *per_group_chains)

        # For pure Adam we do NOT need `value` or `step` inside update
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
        s = time.time()
        
        init_loss, init_grads = jax.value_and_grad(cost_func)(params, input_states, target_states)
        e = time.time()
        dt = e - s
        

        flat_grads = jnp.ravel(init_grads)
        
        # opt_lr_tree, mask_tau, mask_h, mask_J = get_groupwise_lr_lars(
        #     params, flat_grads,N_train ,NC=N_ctrl, max_lr=max_lr, time_steps=time_steps, debug=True
        # )
        D = params.size
        rms = jnp.linalg.norm(params) / jnp.sqrt(D)
        target_update = 0.75 * jnp.linalg.norm(params) / jnp.sqrt(D)
        
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
                target_update    = target_update,
                eps              = 1e-12,
                scale_by_num_train = scale_by_num_train,
                per_block_target_update=per_block_target_update
            )
            opt_lr = (lr_tree, assignment_mask)
           
        else:
          
            lr_tree, mask_tau, mask_h, mask_J = get_groupwise_lr_trees_new(
                params         = params,          # your flat parameter vector
                grads          = flat_grads,     # magnitude of gradients
                num_train      = N_train,        # number of Haar‐random states
                NC             = N_ctrl,         # number of control qubits
                max_lr         = 0.2,            # your chosen clip‐upper‐bound
                time_steps     = time_steps,     # T
                debug          = True,
                scale_by_num_train = scale_by_num_train,
                target_update  = target_update,
                eps            = 1e-12,
                per_block_target_update=per_block_target_update
            )
            opt_lr = {
                "t": lr_tree[:time_steps],
                "h": lr_tree[time_steps:time_steps + 3],
                "J": lr_tree[time_steps + 3:]
            }
      
        
        lr_groups_init = lr_group_dict(case_num, lr_tree, time_steps, N_ctrl, N_reserv)
        cv0 = coef_variation_by_block(case_num, flat_grads,
                              time_steps, N_ctrl, N_reserv)
        cv_str = ", ".join(f"{k}: {v:.4f}" for k, v in cv0.items())
        print("\nInitial coefficient of variation (CV_B) by block at step 0: ", cv_str)
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
    print(f"Saving results to {filename}. Date/time: ", now.strftime("%Y-%m-%d %H:%M:%S"))
    df = pd.DataFrame([data])
    while os.path.exists(filename):
        name, ext = filename.rsplit('.', 1)
        filename = f"{name}_.{ext}"

    with open(filename, 'wb') as f:
        pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)


 

if __name__ == '__main__':


    

    
    # run below 
    N_ctrl = 1
   
   

    # trots = [1,15,20,25,30,35,40]
    # trots = [25,28,30,32,35,38,40,45]
    trots = [4,6,8,10,12,16,20,24,28]
    trots = [2,4,6]
    # trots = [8,12,20,24,28]
    # trots = [4,8,10,12,14,16,18,20,24,28]

    # res = [1, 2, 3]
    # trots = [10,12,14,16,20]
    # trots = [10,14,20,24,28]
    # trots = [24,28]
    # trots = [6]

    res = [1]
  

    
    




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
    scale_by_num_train=True
    per_block_target_update=False
    
    # folder = f'./analog_results_trainable_global/trainsize_{N_train+add}_epoch{num_epochs}_per_param_opt_.1k/'
    if case_num in [2,3]:
        folder = f'./analog_results_trainable_global/trainsize_{N_train+add}_epoch{num_epochs}/case_{case_num}/PATIENCE{patience}_ACCUMULATION{accum_size}/ATOL_1e-7/'
        # folder = f'./analog_results_trainable_global/trainsize_{N_train+add}_epoch{num_epochs}/case_{case_num}/PATIENCE{patience}_ACCUMULATION{accum_size}/'
    else:
        folder = f'./analog_results_trainable_global/trainsize_{N_train+add}_epoch{num_epochs}/case_{case_num}_new/tgt_granular_False/t=pi/scale={scale_by_num_train}/'
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

        if not gate_idx in [0,1]:
            continue


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
                    
                    
                    time_step_params = jax.random.uniform(params_key, shape=(time_steps,), minval=0, maxval=np.pi)
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
