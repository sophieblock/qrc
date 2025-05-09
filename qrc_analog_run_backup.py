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
diable_jit = True
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
        
        for i,qubit_a in enumerate(self.reserv_qubits):
            for j,qubit_b in  enumerate(self.ctrl_qubits):
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
    

def get_rate_of_improvement(cost, prev_cost,second_prev_cost):
    
    prev_improvement = prev_cost - second_prev_cost
    current_improvement = cost - prev_cost
    acceleration = prev_improvement - current_improvement

    return acceleration

def get_optimizer(case: int,
                  opt_lr,
                  b1=0.99,
                  b2=0.999,
                  eps=1e-8,
                  num_epochs: int = None,
                  patience: int = 50,
                  decay_rate: float = 0.9,
                  rtol: float = 1e-4,
                  factor: float = 0.9,
                  min_scale: float = 0.1,
                  warmup_steps: int = None):
    """
    Return an Optax optimizer (and a description) based on `case`.
    
    case 0: plain Adam
    case 1: Adam with fixed lr
    case 2: SGDR schedule
    case 3: ReduceLROnPlateau
    case 4: warmup_cosine_decay_schedule
    default: per-param Adam chain (as in your current)
    """
    desc = ""
    if case == 0:
        desc = "per-param Adam"
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate=opt_lr, b1=b1, b2=b2, eps=eps),
        )

    elif case == 1:
        desc = "Adam with fixed lr"
        schedule = optax.constant_schedule(opt_lr)
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.inject_hyperparams(optax.adam)(learning_rate=schedule)
        )

    elif case == 2:
        desc = "SGDR schedule"
        assert num_epochs is not None, "num_epochs needed for SGDR"
        total_steps = num_epochs // 2
        warmup = num_epochs // 2
        schedule = optax.sgdr_schedule([
            {"init_value": opt_lr * 0.85, "peak_value": opt_lr, "decay_steps": total_steps, "warmup_steps": warmup, "end_value": opt_lr * 0.5},
            {"init_value": opt_lr / 5,  "peak_value": opt_lr/2, "decay_steps": total_steps, "warmup_steps": warmup, "end_value": opt_lr * 1e-1},
            {"init_value": opt_lr * 0.85, "peak_value": opt_lr, "decay_steps": total_steps, "warmup_steps": warmup, "end_value": opt_lr * 0.5},
        ])
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.inject_hyperparams(optax.adam)(learning_rate=schedule)
        )

    elif case == 3:
        desc = "ReduceLROnPlateau"
        adam = optax.adam(learning_rate=opt_lr, b1=0.9, b2=0.99, eps=1e-8)
        reduce_on_plateau = optax.contrib.reduce_on_plateau(
            factor=factor,
            patience=patience,
            rtol=rtol,
            min_scale=min_scale,
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            adam,
            reduce_on_plateau
        )

    elif case == 4:
        desc = "warmup_cosine_decay_schedule"
        assert warmup_steps is not None and num_epochs is not None, "need warmup_steps & num_epochs"
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=opt_lr * 0.9,
            peak_value=opt_lr * 1.2,
            warmup_steps=warmup_steps,
            decay_steps=num_epochs,
            end_value=opt_lr * 0.85,
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.inject_hyperparams(optax.adam)(learning_rate=schedule)
        )

    else:
        desc = "per-param Adam"
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.inject_hyperparams(optax.adam)(
                learning_rate=opt_lr, b1=0.99, b2=0.999, eps=1e-8
            ),
        )

    return optimizer, desc

def get_base_learning_rate(grads, scale_factor=.1, min_lr=1e-5, max_lr=0.2):
    """Estimate a more practical initial learning rate based on the gradient norms."""
    grad_norm = jnp.linalg.norm(grads)
    
    initial_lr = jnp.where(grad_norm > 0, scale_factor / grad_norm, 0.1)
    
    clipped_lr = jnp.clip(initial_lr, min_lr, max_lr)
    print(f"grad_norm: {grad_norm}, initial base lr: {initial_lr:.5f}, clipped: {clipped_lr:.5f}")
    return initial_lr, clipped_lr,grad_norm



def get_initial_lr_per_param(grads, num_train, base_step=0.01,raw_lr=None, min_lr=1e-4, max_lr=0.25,debug=True,):
     # print(f"grads: {grads}")
    D = grads.shape[0]
    idx = jnp.arange(D)

    mask_tau = idx < time_steps
    mask_h   = (idx >= time_steps) & (idx < time_steps + 3)
    mask_J   = idx >= time_steps + 3
    grad_magnitudes = jax.tree_util.tree_map(lambda g: jnp.abs(g) + 1e-12, grads)
    global_norm = jnp.linalg.norm(grad_magnitudes)
    # a,b,c=get_base_learning_rate(grads)
    # print(get_initial_lr_per_param_original(grads,max_lr=a))
    N_params = grad_magnitudes.shape[0]
    median_grad = jnp.quantile(grad_magnitudes, 0.5)  # For debugging
    MAD = jnp.median(jnp.abs(grad_magnitudes - median_grad))
    
    
    norm_factor = global_norm / jnp.sqrt(N_params)
    print(f"global_norm: {global_norm:.5f}, norm factor= {norm_factor:.5f}")
    normalized_abs = grad_magnitudes / (norm_factor + 1e-8)
    median_norm = jnp.quantile(normalized_abs, 0.5)
   
    MAD_norm = jnp.quantile(jnp.abs(normalized_abs - median_norm), 0.5)
  
    #     r = (MAD+median_grad) /2
    if num_train >= 20:
        r = (MAD+median_grad)*(1/4)
        
    elif num_train <= 15 and num_train > 10:
        r = (MAD+median_grad)*(1/2)
    elif num_train <= 10:
        r = MAD+median_grad
       
    # r = 0.1* (MAD+median_grad)/2

    # print(f"grad_magnitudes: {grad_magnitudes}")
    lr_tree2 = jax.tree_util.tree_map(lambda g: max_lr * (r/ (g + r )), grad_magnitudes)
    min2 = float(jnp.min(lr_tree2))
    max2 = float(jnp.max(lr_tree2))
    med2 = float(jnp.median(lr_tree2))
    lr_tree = jax.tree_util.tree_map(lambda g: base_step / g, grad_magnitudes)
    # print(f"og: {lr_tree}")
    # print(f"lr_tree2 (min={min2:.2e}, max={max2:.3e}, med2={med2:.3e}): {lr_tree2}")

    lr_tree = jax.tree_util.tree_map(lambda lr: jnp.clip(lr, med2/2, max_lr), lr_tree2)
    if debug:
        print("=== normal learning‐rate stats ===")
        print(f"Median: {median_grad:.3e}")
        print(f"MAD: {MAD:.3e}")
        print(f"MAD+Med: {MAD+median_grad:.3e}, r: {r:.3e}")
        print(f"Final lr_tree: min = {float(jnp.min(lr_tree)):.2e}, max = {float(jnp.max(lr_tree)):.2e}, med={float(jnp.median(lr_tree))}, mean = {float(jnp.mean(lr_tree)):.2e}, var = {float(jnp.var(lr_tree)):.3e}")

        lr_tau = lr_tree[mask_tau]
        lr_h   = lr_tree[mask_h]
        lr_J   = lr_tree[mask_J]

        # Print mean / min / max for each group
        
        print(f" t‐group: mean={float(jnp.mean(lr_tau)):.3e}, "
            f"min={float(jnp.min(lr_tau)):.3e}, max={float(jnp.max(lr_tau)):.3e}")
        print(f" h‐group: mean={float(jnp.mean(lr_h)):.3e}, "
            f"min={float(jnp.min(lr_h)):.3e}, max={float(jnp.max(lr_h)):.3e}")
        print(f" J‐group: mean={float(jnp.mean(lr_J)):.3e}, "
            f"min={float(jnp.min(lr_J)):.3e}, max={float(jnp.max(lr_J)):.3e}")


        # print(lr_tree)
    return lr_tree
def get_groupwise_lr_trees(
    grads: jnp.ndarray,
    num_train,
    max_lr: float,
    time_steps: int,
    debug: bool = False,
) -> jnp.ndarray:
    """
    grads:        [D,] gradient magnitudes at the initial params
    max_lr:       scalar upper bound on any lr
    time_steps:   T
    debug:        if True, prints medians, MADs, r’s and a few sample lrs
    returns:      lr_tree of shape [D,], where each index
                  has its own lr_i taken from its group.
    """
    grad_magnitudes = jax.tree_util.tree_map(lambda g: jnp.abs(g) + 1e-12, grads)
    D = grad_magnitudes.shape[0]
    idx = jnp.arange(D)

    # 1) masks
    mask_tau = idx < time_steps
    mask_h   = (idx >= time_steps) & (idx < time_steps + 3)
    mask_J   = idx >= time_steps + 3

    # 2) extract groups
    g_tau = grad_magnitudes[mask_tau]
    g_h   = grad_magnitudes[mask_h]
    g_J   = grad_magnitudes[mask_J]

    # 3) medians & MADs
    def med_mad(x):
        med = jnp.median(x)
        mad = jnp.median(jnp.abs(x - med))
        return med, mad

    med_tau, mad_tau = med_mad(g_tau)
    med_h,   mad_h   = med_mad(g_h)
    med_J,   mad_J   = med_mad(g_J)

    # 4) r = med + mad
    #     r = (MAD+median_grad) /2
    if num_train >= 20:
        factor = 1/4
        
    elif num_train <= 15 and num_train > 10:
        factor = 1/2
    elif num_train <= 10:
        factor = 1

    r_tau = (med_tau + mad_tau) * factor
    r_h   = (med_h   + mad_h)* factor
    r_J   = (med_J   + mad_J)* factor

    # 5) per‐group rule
    lr_tau = r_tau * max_lr / (g_tau + r_tau + 1e-12)
    lr_h   = r_h   * max_lr / (g_h   + r_h   + 1e-12)
    lr_J   = r_J   * max_lr / (g_J   + r_J   + 1e-12)

    # 6) scatter back
    lr_tree = jnp.zeros_like(grad_magnitudes)
    lr_tree = lr_tree.at[mask_tau].set(lr_tau)
    lr_tree = lr_tree.at[mask_h].set(lr_h)
    lr_tree = lr_tree.at[mask_J].set(lr_J)

    if debug:
        # pull a few scalars out for printing
        print(f"\n--- groupwise‐LR debug ---")
        print(f" group sizes:  τ={g_tau.shape[0]},  h={g_h.shape[0]},  J={g_J.shape[0]}")
        print(f" medians:      τ={float(med_tau):.3e}, h={float(med_h):.3e}, J={float(med_J):.3e}")
        print(f" MADs:         τ={float(mad_tau):.3e}, h={float(mad_h):.3e}, J={float(mad_J):.3e}")
        print(f" r = med+mad:  τ={float(r_tau):.3e}, h={float(r_h):.3e}, J={float(r_J):.3e}")
        # show a handful of per‐param picks
        ex = min(5, D)
        print(f" sample grads: {grad_magnitudes[:ex].tolist()}")
        print(f" sample lrs:   {lr_tree[:ex].tolist()}")
        print(f" variances:    τ={float(jnp.var(lr_tau)):.3e}, h={float(jnp.var(lr_h)):.3e}, J={float(jnp.var(lr_J)):.3e}")
        print(f"---------------------------\n")
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

    # assert jnp.all(lr_tree >= min_lr), (
    #     f"Assertion failed: Some learning rates are less than {min_lr:.1e}. "
    #     f"Indices in mask_tau: {jnp.where(mask_tau)[0][lr_tree[mask_tau] < min_lr]}, "
    #     f"Indices in mask_h: {jnp.where(mask_h)[0][lr_tree[mask_h] < min_lr]}, "
    #     f"Indices in mask_J: {jnp.where(mask_J)[0][lr_tree[mask_J] < min_lr]}"
    # ) 
    return lr_tree, mask_tau, mask_h, mask_J

def make_groupwise_plateau_optimizer(
    opt_lr: dict[str, jnp.ndarray],
    *,
    b1=0.9, b2=0.999, eps=1e-8,
    factor=0.9, patience=50, rtol=1e-4, min_scale=0.1,
):
    def one_group(rate):
        return optax.chain(
            optax.inject_hyperparams(optax.adam)(
                learning_rate=rate, b1=b1, b2=b2, eps=eps
            ),
            optax.contrib.reduce_on_plateau(
                factor=factor, patience=patience,
                rtol=rtol, min_scale=min_scale,
            ),
        )

    mask_t = {"t": True,  "h": False, "J": False}
    mask_h = {"t": False, "h": True,  "J": False}
    mask_J = {"t": False, "h": False, "J": True}

    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.masked(one_group(opt_lr["t"]), mask_t),
        optax.masked(one_group(opt_lr["h"]), mask_h),
        optax.masked(one_group(opt_lr["J"]), mask_J),
    )
def run_test(params, num_epochs, N_reserv, N_ctrl, time_steps,N_train,folder,gate,gate_name,bath,num_bath,init_params_dict, dataset_key):
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
        raw_lr,clipped_lr,grad_norm = get_base_learning_rate(init_grads)
        flat_grads = jnp.ravel(init_grads)
        # after you’ve computed your flat_grads …
        max_grad = float(jnp.max(flat_grads))
        # choose a “target” maximum update size, e.g. 10% of a typical parameter
        target_update = 0.05
        # then set max_lr so that max_lr * max_grad ≈ target_update
        max_lr = target_update / (max_grad + 1e-12)
        print(f"raw_lr: {raw_lr:.3e}, max_lr: {max_lr:.3e}")
        if max_lr>0.2:
            max_lr = 0.2
        elif max_lr < 0.01:
            max_lr=0.01
        
        opt_lr_tree, mask_tau, mask_h, mask_J = get_groupwise_lr_trees(
            flat_grads,N_train, max_lr=max_lr, time_steps=time_steps, debug=True
        )
        # print(f"opt_lr_tree: {opt_lr_tree}")

        
        opt_lr = {
            "t": opt_lr_tree[:time_steps],
            "h": opt_lr_tree[time_steps:time_steps + 3],
            "J": opt_lr_tree[time_steps + 3:]
        }
        print(f"opt_lr: {opt_lr}")



       
        # print(f"Adjusted initial learning rate: {opt_lr:.2e}. Grad_norm: {1/grad_norm},Grad_norm: {grad_norm:.2e}")
        cost = init_loss
    
 
    # opt = optax.chain(
    #     optax.clip_by_global_norm(1.0),            # Clip gradients globally
    #     scale_by_param_lr(opt_lr),             # Apply per-parameter scaling
    #     optax.adam(learning_rate=1.0, b1=0.99, b2=0.999, eps=1e-7)  # Use a "neutral" global LR
    # )
    # opt = optax.adam(learning_rate=opt_lr, b1=0.99, b2=0.999, eps=1e-7)
    debug = True
    

    if debug:
        # Compute group‐wise subsets
        lr_tau = opt_lr_tree[mask_tau]
        lr_h   = opt_lr_tree[mask_h]
        lr_J   = opt_lr_tree[mask_J]

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
        print(f'global h-vec lrs: hx={opt_lr_tree[time_steps]:.2e} hy={opt_lr_tree[time_steps+1]:.2e} hz={opt_lr_tree[time_steps+2]:.2e} ')
        for t in range(time_steps):
            tau_lr = float(opt_lr_tree[t])
            # J‐block for step t lives at indices [time_steps+3 + t*num_J : time_steps+3 + (t+1)*num_J]
            start = time_steps + 3 + t * num_J
            end   = start + num_J
            J_block = opt_lr_tree[start:end]
            avg_J   = float(jnp.mean(J_block))
            # Print each J element in the specified order using the qubit indices
            J_elements = []
            for i, r in enumerate(reserv_qubits):
                for j, c in enumerate(ctrl_qubits):
                    # J_index = f"J_{{{r},{c}}}={J_block[i * len(ctrl_qubits) + j]:.2e}"  # Access the element by index
                    J_index = f"J({r},{c})={J_block[i * len(ctrl_qubits) + j]:.2e}"  # Access the element by index
                    J_elements.append(J_index)
            
            print(f" step {t:2d}: t_lr={tau_lr:.2e},  avg(J_lr)={avg_J:.2e},  " + ", ".join(J_elements))

    # opt_descr = 'per param'
    # opt = optax.chain(
    #     optax.clip_by_global_norm(1.0),
    #     # optax.inject_hyperparams(optax.adam)(learning_rate=opt_lr),
    #     # optax.inject_hyperparams(optax.adam)(learning_rate=opt_lr, b1=0.99, b2=0.999, eps=1e-8),
    #     optax.inject_hyperparams(optax.adam)(learning_rate=opt_lr, b1=0.99, b2=0.999, eps=1e-8),
    #     )

    # opt_descr = 'masked per group adam'
    # # Assuming params is a flat vector and time_steps is defined
    

    # opt = optax.chain(
    #     optax.clip_by_global_norm(1.0),
    #     optax.masked(optax.adam(opt_lr["t"]), mask={"t": True, "h": False, "J": False}),
    #     optax.masked(optax.adam(opt_lr["h"]), mask={"t": False, "h": True, "J": False}),
    #     optax.masked(optax.adam(opt_lr["J"]), mask={"t": False, "h": False, "J": True})
    # )

    opt_descr = 'masked per group adam'
    PATIENCE = 50
    COOLDOWN = 0
    FACTOR = 0.9
    RTOL = 1e-4
    # ACCUMULATION_SIZE = 10
    MIN_SCALE = 0.1
    new_scales = {'t': 1.0, 'h': 1.0, 'J': 1.0}

    opt = make_groupwise_plateau_optimizer(opt_lr, factor=FACTOR, patience=PATIENCE, rtol=RTOL, min_scale=MIN_SCALE,)

    params = {
        "t": params[:time_steps],
        "h": params[time_steps:time_steps + 3],
        "J": params[time_steps + 3:]
    }
    # Define the optimization update function
    if opt_descr == 'per param':
        var = np.var(opt_lr)
        mean = np.mean(opt_lr)
        @jit
        def update(params, opt_state, input_states, target_states):
            
            """Update all parameters including tau."""
            # params = jnp.asarray(params, dtype=jnp.float64)
            # input_states = jnp.asarray(input_states, dtype=jnp.complex128)
            # target_states = jnp.asarray(target_states, dtype=jnp.complex128)
            loss, grads = jax.value_and_grad(cost_func)(params, input_states, target_states)
            if not isinstance(opt_state[-1], optax.contrib.ReduceLROnPlateauState):
                updates, opt_state = opt.update(grads, opt_state, params)
            else:
                updates, opt_state = opt.update(grads, opt_state, params=params, value=loss)
            new_params = optax.apply_updates(params, updates)
            # Ensure outputs are float64
            loss = jnp.asarray(loss, dtype=jnp.float64)
            grads = jnp.asarray(grads, dtype=jnp.float64)
            return new_params, opt_state, loss, grads
    elif opt_descr == 'masked per group adam':
        var = float(jnp.var(opt_lr_tree))
        mean = float(jnp.mean(opt_lr_tree))
        @jit
        def update(params, opt_state, X, y):
            params_flat = jnp.concatenate([params["t"], params["h"], params["J"]])
            loss, grads_flat = jax.value_and_grad(cost_func)(params_flat, X, y)

            grads_pytree = {
                "t": grads_flat[:time_steps],
                "h": grads_flat[time_steps:time_steps + 3],
                "J": grads_flat[time_steps + 3:]
            }

            updates, opt_state = opt.update(grads_pytree, opt_state, params, value=loss)
            new_params = optax.apply_updates(params, updates)

            return new_params, opt_state, loss, grads_flat
    # print(f"variance per param: {np.var(opt_lr)} \nlrs: \n{opt_lr}\n")

    print("________________________________________________________________________________")
    print(f"Starting optimization for {gate_name}(epochs: {num_epochs}) T = {time_steps}, N_r = {N_reserv}, N_bath = {num_bath}...\n")
    print(f"Initial Loss: {init_loss:.4f}, initial_gradients: {np.mean(np.abs(init_grads))}. Time: {dt:.2e}")
    print("sample rates:", opt_lr_tree[:5].tolist())
    print(f"per-param learning-rate tree: mean={mean:.3e}, var={var:.3e}")


    costs = []
    param_per_epoch,grads_per_epoch = [],[]
   # print(f"Params: {params}")
    
    opt_state = opt.init(params)
    # print(f"initial opt_state: {opt_state}")

    # Define the gradient function outside the loop
    #cost_and_grad = jax.value_and_grad(partial(cost_func, time_steps=time_steps, N_reserv=N_reserv, N_ctrl=N_ctrl))
    prev_cost, second_prev_cost = float('inf'), float('inf')  # Initialize with infinity
    threshold_counts = 0
    acceleration = 0.0
    rocs = []
    consecutive_improvement_count = 0
    
    consecutive_threshold_limit = 4  # Number of consecutive epochs below threshold without improvement needed to stop
    backup_params = init_params
    improvement = True
    backup_cost,min_cost =init_loss,float('inf')   
    freeze_tau = False
    epoch = 0
    s = time.time()
    full_s =s
    training_state_metrics = {}
    add_more = False
    a_condition_set = False
    a_threshold =  0.0
    stored_epoch = None
    threshold_cond1, threshold_cond2 = [],[]
    false_improvement = False
    backup_epoch=0
    scale_reduction_epochs,learning_rates = [],[]  # Track epochs where scale is reduced
    scales_per_epoch = []  # Store scale values per epoch
    new_scale = 1.0  # Initial scale value
    backup_cost_check = cost
    while epoch < num_epochs or improvement:
        if opt_descr in ['masked per group adam']:
            
            _, mask_t_state, mask_h_state, mask_J_state = opt_state
            # print(f"mask_t_state: {mask_t_state}\nmask_t_state.inner_state[1]: {mask_t_state.inner_state[1]}")
            plateau_states = {
                't': mask_t_state.inner_state[1],
                'h': mask_h_state.inner_state[1],
                'J': mask_J_state.inner_state[1],
            }
            # capture averages before update
            current_avgs = {g: ps.avg_value for g, ps in plateau_states.items()}

        params, opt_state, cost, grad = update(params, opt_state, input_states, target_states)
        if opt_descr in ['masked per group adam']:
            for group, ps in plateau_states.items():
                scale     = ps.scale
                best      = ps.best_value
                count     = ps.plateau_count
                avg_val   = ps.avg_value

                # compute adjusted per-group learning rate
                adj_lr = opt_lr[group] * scale
                learning_rates.append({group: float(jnp.mean(adj_lr))})
                scales_per_epoch.append((group, float(scale)))

                # “should have triggered” check
                if avg_val >= best * (1 - RTOL) + RTOL and count >= PATIENCE:
                    print(f"{group}: ReduceLROnPlateau *should* have triggered at Epoch {epoch1}!")

                # detect actual reduction
                if scale < new_scales[group]:
                    lrs     = opt_lr[group] * scale
                    lr_min  = float(jnp.min(lrs))
                    lr_max  = float(jnp.max(lrs))
                    lr_mean = float(jnp.mean(lrs))
                    lr_var  = float(jnp.var(lrs))

                    print(
                        f"{group}: Reduced scale to {scale} at epoch {epoch}:\n"
                        # f"  - current_avg: {float(current_avgs[group]):.2e},\n"
                        # f"  - best:        {float(best):.2e},\n"
                        f"  - threshold:   {float(best * (1 - RTOL) + RTOL):.2e}"
                        f"  -> new LR stats: min={lr_min:.2e}, max={lr_max:.2e}, "
                        f"mean={lr_mean:.2e}, var={lr_var:.2e}"
                    )
                    scale_reduction_epochs.append((group, epoch+1))
                    new_scales[group] = scale
            

        else:
            learning_rates.append('fixed')
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
                # a_threshold = a_marked*1e-3 if a_marked*1e-3 > 9e-7 else a_marked*1e-2
                
                # print(f"acceration: {a_marked:.2e}, marked: {a_threshold:.2e}")
                # if N_ctrl == 3:
                # # if True:
                #     a_threshold *= 10
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
            # avg_value = plateau_state.avg_value
            # has_improved = jnp.where(
            #     avg_value < (1 - plateau_state.rtol) * plateau_state.best_value - plateau_state.atol, 1, 0
            # )
            # new_best_value = jnp.where(has_improved, avg_value, plateau_state.best_value)
            # curr_plateau_count = jnp.where(
            #     has_improved, 0, numerics.safe_increment(plateau_state.plateau_count)
            # )
            epoch_time = e - s
            
            # learning_rate = opt_state[1].hyperparams['learning_rate']
            if cost < 1e-3:
                print(f'Epoch {epoch + 1} --- cost: {cost:.3e}, '

                    f'[t: {epoch_time:.1f}s]'
                    )
            else:
                print(f'Epoch {epoch + 1} --- cost: {cost:.5f}, '
                # print(f'Epoch {epoch + 1} --- cost: {cost:.4f}, best={best_val:.4f}, avg: {current_avg:.4f}, lr={learning_rates[-1]:.4f} [{plateau_state.scale:.3f}], '
                    #   f' count: {plateau_state.plateau_count} '
        
                    # f'Var(grad): {var_grad:.1e}, '
                    # f'GradNorm: {np.linalg.norm(grad):.1e}, '
                    #  f'Mean(grad): {mean_grad:.1e}, '
                    f'[t: {epoch_time:.1f}s]'
                    )
            # print(f" opt_state: {opt_state}")
            # print(f"    --- Learning Rate: {learning_rate}")
        
            s = time.time()

        if cost < prev_cost:
            
            improvement = True
            consecutive_improvement_count += 1
            # current_cost_check = cost
            if opt_descr == 'masked per group adam':
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
        if opt_descr == 'masked per group adam':
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
    print(f"Time optimizing: {epoch_time}")
    if opt_descr == 'per param':
        def final_test(params,test_in,test_targ):
            params = jnp.asarray(params, dtype=jnp.float64)
            X = jnp.asarray(test_in, dtype=jnp.complex128)
            y = jnp.asarray(test_targ, dtype=jnp.complex128)
            batched_output_states = vcircuit(params, X)

            fidelities = jax.vmap(qml.math.fidelity)(batched_output_states, y)
            # fidelities = jnp.clip(fidelities, 0.0, 1.0)

            return fidelities
    elif opt_descr == 'masked per group adam':
        def final_test(params,test_in,test_targ):
            params_flat = jnp.concatenate([params["t"], params["h"], params["J"]], dtype=jnp.float64)
            X = jnp.asarray(test_in, dtype=jnp.complex128)
            y = jnp.asarray(test_targ, dtype=jnp.complex128)
            batched_output_states = vcircuit(params_flat, X)
            fidelities = jax.vmap(qml.math.fidelity)(batched_output_states, y)
            # fidelities = jnp.clip(fidelities, 0.0, 1.0)

            return fidelities
            """
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
        # """
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
            'specs':specs,
                'epochs': num_epochs,
                'lrs': learning_rates,
                'scales_per_epoch': scales_per_epoch,  # Add scales per epoch
                'scale_reduction_epochs': scale_reduction_epochs,  # Add epochs of scale reduction
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
                'group_opt_lr_tree':opt_lr_tree,
                'lr_tau':opt_lr_tree[mask_tau],
                'lr_h':opt_lr_tree[mask_h],
                'lr_J':opt_lr_tree[mask_J],
                'grads_per_epoch':grads_per_epoch,
                'bath': bath,
                'num_bath':num_bath,
                'partial_rho_qfim':True,
                # 'infidelities':infidelities,
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
   
   

    trots = [1,15,20,25,30,35,40]
    trots = [2]
    # trots = [4,5,6,7,8,10,16,20]
    # trots = [1,2,3,4,5,6]

    # res = [1, 2, 3]
    res = [1]
  

    
    




    num_epochs = 1500
    N_train = 10
    add=0
  
    
    # folder = f'./analog_results_trainable_global/trainsize_{N_train+add}_epoch{num_epochs}_per_param_opt_.1k/'
    folder = f'./analog_results_trainable_global/trainsize_{N_train+add}_epoch{num_epochs}_per_param_opt_group_lr/'
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

        # if not gate_idx in [3]:
        #     continue


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

                    run_test(params, num_epochs, N_reserv, N_ctrl, time_steps,N_train,folder,gate,gate.name,bath,num_bath,init_params_dict = init_params_dict,dataset_key = dataset_key)