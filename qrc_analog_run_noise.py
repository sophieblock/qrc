import pennylane as qml
import os

import pickle
import re
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from qiskit import *
from jax import numpy as np
import sympy
import matplotlib.pyplot as plt
import base64
from jax import numpy as jnp
import pickle

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
import numpy
import pennylane as qml


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

def get_init_params(N_ctrl, N_reserv, time_steps, bath, num_bath, key):

    N = N_reserv + N_ctrl
    
    K_half = jax.random.uniform(key, (N, N))
    K = (K_half + K_half.T) / 2  # making the matrix symmetric
    K = 2. * K - 1.

    if bath:
        # bath_array = 0.01 * jax.random.normal(key, (num_bath, N_ctrl + N_reserv))
        return {
            

            'K_coef': jnp.asarray(K),
            'key':key
        }
    return {

            'K_coef': jnp.asarray(K)
        }

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


def array(a,dtype):
    return np.asarray(a, dtype=np.float32)
def Array(a,dtype):
    return np.asarray(a, dtype=np.float32)
@jit
def get_initial_learning_rate(grads, scale_factor=0.1, min_lr=1e-3, max_lr=0.2):
    """Estimate a more practical initial learning rate based on the gradient norms."""
    grad_norm = jnp.linalg.norm(grads)
    initial_lr = jnp.where(grad_norm > 0, scale_factor / grad_norm, 0.1)
    initial_lr = jnp.clip(initial_lr, min_lr, max_lr)
    return initial_lr, grad_norm

def compute_initial_learning_rate(gradients, scale_factor=0.01, min_lr=1e-3, max_lr = 0.2):
    """
    Compute an initial learning rate based on the norm of gradients.
    """
    # Compute the norm of the gradients
    
    norm_grad = jnp.linalg.norm(gradients)
    mean_norm_grad = jnp.mean(norm_grad)
    initial_lr = scale_factor / (mean_norm_grad + 1e-8)  # Adding a small value to prevent division by zero

    initial_lr = jnp.clip(initial_lr, min_lr, max_lr)
    return initial_lr


def normalize_gradients(gradients):
    norm = jnp.linalg.norm(gradients, axis=-1, keepdims=True) + 1e-12  # Add small epsilon to avoid division by zero
    return gradients / norm

def calculate_gradient_stats(gradients,abs_grad=True):
    if abs_grad:
        gradients = jnp.abs(gradients)
    mean_grad = jnp.mean(gradients, axis=0)
    mean_grad_squared = jnp.mean(gradients ** 2, axis=0)
    var_grad = mean_grad_squared - mean_grad ** 2
    grad_norm = jnp.linalg.norm(mean_grad)
    return mean_grad, var_grad, grad_norm

def calculate_unbiased_stats(gradients, abs_grad=True):
    """Calculate the mean and unbiased variance of the gradients for each state."""
    if abs_grad:
        gradients = jnp.abs(gradients)
    mean_grad = jnp.mean(gradients, axis=-1)  # Mean across parameters for each state
    mean_grad_squared = jnp.mean(gradients ** 2, axis=-1)  # Mean squared gradients across parameters
    
    # Use ddof=1 for unbiased variance
    var_grad = jnp.var(gradients, axis=-1, ddof=1)  # Unbiased variance per state
    
    grad_norm = jnp.linalg.norm(gradients, axis=-1)  # Norm per state

    return mean_grad, var_grad, grad_norm
def get_rate_of_improvement(cost, prev_cost,second_prev_cost):
    
    prev_improvement = prev_cost - second_prev_cost
    current_improvement = cost - prev_cost
    acceleration = prev_improvement - current_improvement

    return acceleration
def get_initial_lr_per_param(grads, base_step=0.01, min_lr=1e-4, max_lr=0.25,debug=True):
     # print(f"grads: {grads}")
    
    grad_magnitudes = jax.tree_util.tree_map(lambda g: jnp.abs(g) + 1e-12, grads)
    global_norm = jnp.linalg.norm(grad_magnitudes)
    N_params = grad_magnitudes.shape[0]
    median_grad = jnp.quantile(grad_magnitudes, 0.5)  # For debugging
    MAD = jnp.median(jnp.abs(grad_magnitudes - median_grad))
    
    
    norm_factor = global_norm / jnp.sqrt(N_params)
    # print(f"global_norm: {global_norm:.5f}, norm factor= {norm_factor:.5f}")
    normalized_abs = grad_magnitudes / (norm_factor + 1e-8)
    median_norm = jnp.quantile(normalized_abs, 0.5)
   
    MAD_norm = jnp.quantile(jnp.abs(normalized_abs - median_norm), 0.5)
    r = MAD_norm+median_norm
    # r = 0.1* (MAD+median_grad)/2

    # print(f"grad_magnitudes: {grad_magnitudes}")
    lr_tree2 = jax.tree_util.tree_map(lambda g:  0.1 * (r/ (g + r )), grad_magnitudes)
    lr_tree = jax.tree_util.tree_map(lambda g: base_step / g, grad_magnitudes)
    # print(f"og: {lr_tree}")
    # print(f"lr_tree2: {lr_tree2}")
    lr_tree = jax.tree_util.tree_map(lambda lr: jnp.clip(lr, min_lr, max_lr), lr_tree2)
    if debug:
        print(f"Median: {median_grad:.3e}, Median norm: {median_norm:.3e}")
        print(f"MAD: {MAD:.3e}, MAD_norm: {MAD_norm:.3e}")
        print(f"MAD+Med: {MAD+median_grad:.3e}, MAD+Med norm: {MAD_norm+median_norm:.3e}")
        print(f"Final lr_tree: min = {float(jnp.min(lr_tree)):.2e}, max = {float(jnp.max(lr_tree)):.2e}, mean = {float(jnp.mean(lr_tree)):.2e}, var = {float(jnp.var(lr_tree)):.3e}")
        print(lr_tree)
    return lr_tree
def generate_physical_bath_matrix(
    num_bath: int,
    key: jax.random.PRNGKey,
    scale: float = 1.0,
    spectral_exponent: float = 1.0,
    positions: jnp.ndarray = None,
    alpha_dist: float = 1.0,
    freq_range: tuple = (0.2, 1.0),
):
    r"""
    Build a matrix \Lambda of shape (num_bath, num_bath) with:
    - Diagonal elements = local fields \Omega_i (drawn from a distribution).
    - Off-diagonal elements = coupling g_{ij} (distance- or frequency-based).
    
    Parameters
    ----------
    num_bath : int
        Number of bath qubits.
    key : jax.random.PRNGKey
        PRNG key for reproducibility.
    scale : float
        Global scale factor for all elements of \Lambda.
    spectral_exponent : float
        If positions=None, we sample local frequencies from freq^( - spectral_exponent ).
    positions : jnp.ndarray or None
        If provided, shape must be (num_bath, D). Used to compute distances.
        If None, we do not use geometry-based couplings.
    alpha_dist : float
        Exponential decay factor for distance-based couplings, if positions is given.
    freq_range : (float, float)
        Range of local bath frequencies if we do a direct uniform sample (for diagonal).
    
    Returns
    -------
    bath_matrix : jnp.ndarray
        (num_bath, num_bath) matrix with physically motivated local fields on diagonal
        and pairwise couplings off-diagonal. No zeroing of diagonal.
    """
    # -----------------------------------------------------
    # Step 1: sample local fields (the diagonal) from a power law or uniform distribution
    # -----------------------------------------------------
    subkey_fields, subkey_couplings = jax.random.split(key, 2)

    # Example A: Uniform in freq_range, then maybe transform if you want
    # local_freqs in [freq_min, freq_max]
    rand_vals = jax.random.uniform(subkey_fields, shape=(num_bath,))
    local_freqs = freq_range[0] + (freq_range[1] - freq_range[0]) * rand_vals

    # If you want them "Ohmic" or "sub-Ohmic", you can do freq^(-spectral_exponent):
    #     local_freqs = local_freqs ** (-spectral_exponent)
    # Or you can do something else.  We'll skip that for now:
    # local_freqs = local_freqs ** (-spectral_exponent)

    # Diagonal = these local frequencies (times scale).
    diag_terms = local_freqs

    # -----------------------------------------------------
    # Step 2: define how to compute off-diagonal couplings
    # -----------------------------------------------------
    bath_matrix = jnp.zeros((num_bath, num_bath))

    # We'll fill in the diagonal now:
    bath_matrix += jnp.diag(diag_terms)

    # If positions is provided: use distance-based decay, e.g. exp(-alpha_dist * dist_ij)
    # plus a factor from local_freqs if you want that correlation
    if positions is not None:
        # positions shape: (num_bath, D)
        diffs = positions[:, None, :] - positions[None, :, :]
        distances = jnp.sqrt(jnp.sum(diffs**2, axis=-1))  # shape: (num_bath, num_bath)

        # For i != j: g_{ij} = scale * [some function of local_freqs[i], local_freqs[j]] * e^{-alpha_dist * r_ij}
        # Example: g_{ij} = sqrt(omega_i * omega_j) * e^{-alpha_dist * r_ij}
        # or maybe just e^{-alpha_dist * r_ij}.  Up to you.
        # We'll do sqrt(...) version to reflect some spectral correlation.
        g_offdiag = (
            jnp.sqrt(local_freqs[:, None] * local_freqs[None, :]) * jnp.exp(-alpha_dist * distances)
        )

        # The diagonal of distances is zero, so that might produce large self-couplings.
        # We'll preserve the diagonal from diag_terms, so we can fill off-diagonal only.
        # We do "where = i!=j" to avoid double counting. We'll keep it symmetrical below.
        i_idx, j_idx = jnp.triu_indices(num_bath, k=1)  # upper triangle
        bath_matrix = bath_matrix.at[i_idx, j_idx].set(
            scale * g_offdiag[i_idx, j_idx]
        )
        bath_matrix = bath_matrix.at[j_idx, i_idx].set(
            scale * g_offdiag[j_idx, i_idx]
        )

    else:
        # No geometry -> define a purely "spectral" correlation approach
        # e.g. off-diagonal g_{ij} = scale * sqrt(freq_i * freq_j)
        # or freq^( - spectral_exponent ), etc.

        # Typically you might do something like:
        g_offdiag = jnp.sqrt(local_freqs[:, None] * local_freqs[None, :])

        # Symmetrically fill in:
        i_idx, j_idx = jnp.triu_indices(num_bath, k=1)
        bath_matrix = bath_matrix.at[i_idx, j_idx].set(g_offdiag[i_idx, j_idx])
        bath_matrix = bath_matrix.at[j_idx, i_idx].set(g_offdiag[j_idx, i_idx])

        # Then multiply by a scale factor if you like
        bath_matrix = bath_matrix * scale

    return bath_matrix
def generate_system_bath_couplings(
    num_bath: int,
    system_wires,            # list or array of system qubit indices
    key: jax.random.PRNGKey,
    scale: float = 0.05,
    bath_factor: float = 1.0,
    positions_sys: jnp.ndarray = None,   # shape: (num_system, D)
    positions_bath: jnp.ndarray = None,  # shape: (num_bath, D)
    alpha_dist: float = 1.0
):
    r"""
    Build a matrix gamma_matrix of shape (num_bath, num_system) with entries gamma[b, k],
    where:
      b = index of bath qubit
      k = index of system qubit (or reservoir qubit) in 'system_wires'

    Options:
    - If positions_sys and positions_bath are given, use distance-based couplings.
    - Otherwise, draw random normal or uniform couplings.

    Returns
    -------
    gamma_matrix : jnp.ndarray
        (num_bath, num_system) array of system–bath coupling strengths.
    """
    num_system = len(system_wires)

    if (positions_sys is not None) and (positions_bath is not None):
        # Distance-based approach:
        #   positions_sys: (num_system, D)
        #   positions_bath: (num_bath, D)
        # We'll build gamma[b, k] = scale * exp(-alpha_dist * distance_{b,k})
        # or possibly multiply by a random factor, up to you.
        sys_pos = jnp.array(positions_sys)
        bath_pos = jnp.array(positions_bath)

        # shape: (num_bath, num_system, D)
        diffs = bath_pos[:, None, :] - sys_pos[None, :, :]
        # shape: (num_bath, num_system)
        distances = jnp.sqrt(jnp.sum(diffs**2, axis=-1))

        # Exponential decay with distance, plus optional "bath_factor"
        gamma_matrix = scale * bath_factor * jnp.exp(-alpha_dist * distances)

    else:
        # No geometry -> random approach
        # Example: Normal(mean=scale, std=bath_factor*scale)
        subkey = jax.random.fold_in(key, 999)
        random_norm = jax.random.normal(subkey, shape=(num_bath, num_system))
        gamma_matrix = scale + bath_factor * scale * random_norm
        # If you want only positive couplings, apply jnp.abs(...) or clamp, etc.

    return gamma_matrix
class Sim_QuantumReservoir:
    def __init__(self, params, N_ctrl, N_reserv, num_J, time_steps=1, bath=False, num_bath=0, bath_factor=1.0,
                 gamma_scale=0.05, lambda_scale=0.01, PRNG_key=jax.random.PRNGKey(0),time_dependent_bath=False, positions_sys=None,
                 positions_bath=None,
                 spectral_exponent=1.0):
        self.bath = bath
        self.gamma_scale = gamma_scale  # Scale for system-bath coupling
        self.lambda_scale = lambda_scale  # Scale for bath-bath coupling
        self.bath_factor = bath_factor
        self.num_bath = num_bath
        self.time_dependent_bath = time_dependent_bath
        self.spectral_exponent = spectral_exponent
        self.PRNG_key = PRNG_key
        self.positions_bath = positions_bath
        self.positions_sys = positions_sys
        self.N_ctrl = N_ctrl
        self.N_reserv = N_reserv
        self.reserv_qubits = qml.wires.Wires(list(range(N_ctrl, N_reserv+N_ctrl)))
        self.ctrl_qubits = qml.wires.Wires(list(range(N_ctrl)))

        # Initialize bath-related properties
        if bath:
            self.bath_qubits = qml.wires.Wires(
                list(range(N_reserv + N_ctrl, N_reserv + N_ctrl + num_bath))
            )
            self.network_wires = qml.wires.Wires([*self.ctrl_qubits,*self.reserv_qubits])
            self.all_wires = qml.wires.Wires([*self.ctrl_qubits,*self.reserv_qubits,*self.bath_qubits])
   
            self.N = N_ctrl + N_reserv + num_bath
            self.dev = qml.device("default.qubit", wires=self.all_wires)
            

            # Generate local bath couplings (gamma_local)
            local_sigma = bath_factor * np.abs(gamma_scale)
            self.gamma_local = jax.random.normal(
                key=jax.random.fold_in(self.PRNG_key, 0), shape=(num_bath,)
            ) * local_sigma 
            self.gamma_matrix = generate_system_bath_couplings(
                num_bath=num_bath,
                system_wires=self.network_wires,   # all qubits in the "system" side
                key=self.PRNG_key,
                scale=gamma_scale,
                bath_factor=bath_factor,
                positions_sys=self.positions_sys,    # If you want geometry-based, pass arrays
                positions_bath=self.positions_bath,   # or define self.positions_bath
                alpha_dist=1.0
            )
            # Generate bath-bath interaction matrix
            self.bath_bath_interactions = generate_physical_bath_matrix(
                num_bath = num_bath,
                key      = self.PRNG_key,
                scale    = lambda_scale,
                spectral_exponent = self.spectral_exponent,
                positions = self.positions_bath,        # can be None or an array
                alpha_dist = 1.0,            # or whatever exponential factor
                freq_range = (0.2, 1.0),     # or any range you want
            )

        else:
            self.N = N_ctrl + N_reserv
            self.gamma_local =  None
            self.gamma_matrix =None
            self.bath_bath_interactions =  None
            self.dev = qml.device("default.qubit", wires = [*self.ctrl_qubits, *self.reserv_qubits]) 
        #print(qml.wires.Wires(list(range(self.N))))
        #print( [*self.ctrl_qubits, *self.reserv_qubits])
        self.qubits = qml.wires.Wires(list(range(self.N)))
        # device on which the circuit is executed

        #self.z_bias = params['hz']
        #self.y_bias = params['hy']
        self.k_coefficient = params['K_coef']
        self.steps = time_steps

        #print(params['bath'])
        self.num_J = num_J
        self.params = params
        self.current_index = 0
        #self.interactions_fixed = interactions
    
        

   
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
   
    def get_ZZ_coupling(self, i, j):
        return qml.PauliZ(wires=i) @ qml.PauliZ(wires=j)
    def get_ZX_coupling(self, bath_qubit):
        '''Return the ZX coupling between bath qubit and each qubit in the system.'''
        operators = []
        for qubit in self.network_wires:
            operators.append(qml.PauliZ(wires=qubit) @ qml.PauliX(wires=bath_qubit))
        return sum(operators)
    
    def get_X_res(self):
        '''Return the XY coupling between qubits i and j with a coefficient function.'''
        return (qml.PauliX(wires=r) for r in [*self.reserv_qubits])

    def modulated_coupling(self, bath_idx, time_step, tau, delta_phi):
        """
        Modulate the coupling strength of the bath qubits at a given time step.

        Args:
            bath_idx (int): Index of the bath qubit.
            time_step (int): Current time step.
            tau (float): Duration of the time step.
            delta_phi (float): Phase offset for modulation.

        Returns:
            float: Modulated coupling strength for the bath qubit.
        """
        # Oscillatory modulation based on time
        modulation = jnp.cos(time_step * delta_phi[bath_idx] * tau)
        return self.gamma_local[bath_idx] * modulation
    


    
    def get_total_hamiltonian_components(self):
        coefficients = []
        operators = []

        
        # Add h_x, h_y, and h_z terms for the reservoir qubits
        coefficients.append(qml.pulse.constant)  # h_x
        operators.append(sum(qml.PauliX(wires=r) for r in self.reserv_qubits))

        coefficients.append(qml.pulse.constant)  # h_y
        operators.append(sum(qml.PauliY(wires=r) for r in self.reserv_qubits))

        coefficients.append(qml.pulse.constant)  # h_z
        operators.append(sum(qml.PauliZ(wires=r) for r in self.reserv_qubits))
        
        # Add XY coupling terms for each control-reservoir pair
        for i, qubit_a in enumerate(self.reserv_qubits):
            for j, qubit_b in enumerate(self.ctrl_qubits):
                coefficients.append(qml.pulse.constant)  # Use constant for J coefficients
                operators.append(self.get_XY_coupling(qubit_a, qubit_b))  # Add XY coupling operator

        
        # Construct the dynamic Hamiltonian
        H_dynamic = qml.dot(coefficients, operators)

        # Construct the static Hamiltonian
        static_coefficients = [
            self.k_coefficient[qa, qb]
            for qa in range(len(self.reserv_qubits))
            for qb in range(len(self.reserv_qubits))
            if qa != qb
        ]
        static_operators = [
            self.get_XY_coupling(self.reserv_qubits[qa], self.reserv_qubits[qb])
            for qa in range(len(self.reserv_qubits))
            for qb in range(len(self.reserv_qubits))
            if qa != qb
        ]
        if self.N_reserv == 1:
            total_H = H_dynamic
        
        else:
            H_static = qml.dot(static_coefficients, static_operators)
            total_H = H_dynamic+H_static
        
        return total_H
    def get_H_bath_new(self):
        """Construct the bath part of the Hamiltonian, including:
        1) System-bath couplings (either time-dependent or not).
        2) Diagonal local fields from bath_bath_interactions[i, i].
        3) Off-diagonal pairwise ZZ couplings from bath_bath_interactions[i, j] with i < j.
        """
        coefficients = []
        operators = []

        # ----------------------------------------------------------------
        # 1) System–bath couplings: gamma_local * (Zsys ⊗ Xbath)
        # ----------------------------------------------------------------
        if self.time_dependent_bath:
            # Time-dependent couplings
            for b_idx, bath_qubit in enumerate(self.bath_qubits):
                for k_idx, sys_qubit in enumerate(self.network_wires):
                    def gamma_fn(t, b=b_idx, k=k_idx):
                        # example: gamma_matrix[b, k] * cos(2π * t / steps)
                        return self.gamma_matrix[b, k] * jnp.cos(2.0 * jnp.pi * t / self.steps)

                    coefficients.append(gamma_fn)
                    operators.append(
                        qml.PauliZ(wires=sys_qubit) @ qml.PauliX(wires=bath_qubit)
                )
        else:
            # Time-independent couplings
            for b_idx, bath_qubit in enumerate(self.bath_qubits):
                for k_idx, sys_qubit in enumerate(self.network_wires):
                    gamma_val = self.gamma_matrix[b_idx, k_idx]
                    coefficients.append(gamma_val)
                    operators.append(
                        qml.PauliZ(wires=sys_qubit) @ qml.PauliX(wires=bath_qubit)
                    )

        # ----------------------------------------------------------------
        # 2) Local field terms (diagonal of bath_bath_interactions)
        #    Each bath qubit gets Ω_i * Z_i
        # ----------------------------------------------------------------
        for i, bath_qubit_i in enumerate(self.bath_qubits):
            Omega_i = self.bath_bath_interactions[i, i]
            if Omega_i != 0.0:
                coefficients.append(Omega_i)
                operators.append(qml.PauliZ(wires=bath_qubit_i))

        # ----------------------------------------------------------------
        # 3) Pairwise ZZ couplings (off-diagonal)
        #    For i < j, g_{ij} * (Z_i ⊗ Z_j)
        # ----------------------------------------------------------------
        for i, bath_qubit_i in enumerate(self.bath_qubits):
            for j, bath_qubit_j in enumerate(self.bath_qubits):
                if j > i:  # or: if bath_qubit_j > bath_qubit_i
                    g_ij = self.bath_bath_interactions[i, j]
                    # Only add if non-zero
                    if g_ij != 0.0:
                        coefficients.append(g_ij)
                        operators.append(
                            self.get_ZZ_coupling(bath_qubit_i, bath_qubit_j)
                        )
                        print(
                            f"Adding BB interactions for q{bath_qubit_i} and q{bath_qubit_j} "
                            f"with coeff={g_ij:.3f} (indices {(i,j)})."
                        )

        # Construct the total bath Hamiltonian
        H_bath = qml.dot(coefficients, operators)
        return H_bath
    def get_H_bath(self):
        """Construct the bath part of the Hamiltonian, including:
        1) System-bath couplings (either time-dependent or not).
        2) Diagonal local fields from bath_bath_interactions[i, i].
        3) Off-diagonal pairwise ZZ couplings from bath_bath_interactions[i, j] with i < j.
        """
        coefficients = []
        operators = []

        # ----------------------------------------------------------------
        # 1) System–bath couplings: gamma_local * (Zsys ⊗ Xbath)
        # ----------------------------------------------------------------
        if self.time_dependent_bath:
            # Time-dependent (modulated) couplings
            for idx, bath_qubit in enumerate(self.bath_qubits):
                def bath_sys_coupling_fn(t):
                    # example: gamma_local[idx] * cos(2π * t / total_steps)
                    return self.gamma_local[idx] * jnp.cos(2.0 * jnp.pi * t / self.steps)

                coefficients.append(bath_sys_coupling_fn)
                operators.append(self.get_ZX_coupling(bath_qubit))
        else:
            # Time-independent couplings
            for idx, bath_qubit in enumerate(self.bath_qubits):
                gamma_val = self.gamma_local[idx]
                coefficients.append(gamma_val)
                operators.append(self.get_ZX_coupling(bath_qubit))

        # ----------------------------------------------------------------
        # 2) Local field terms (diagonal of bath_bath_interactions)
        #    Each bath qubit gets Ω_i * Z_i
        # ----------------------------------------------------------------
        for i, bath_qubit_i in enumerate(self.bath_qubits):
            Omega_i = self.bath_bath_interactions[i, i]
            if Omega_i != 0.0:
                coefficients.append(Omega_i)
                operators.append(qml.PauliZ(wires=bath_qubit_i))

        # ----------------------------------------------------------------
        # 3) Pairwise ZZ couplings (off-diagonal)
        #    For i < j, g_{ij} * (Z_i ⊗ Z_j)
        # ----------------------------------------------------------------
        for i, bath_qubit_i in enumerate(self.bath_qubits):
            for j, bath_qubit_j in enumerate(self.bath_qubits):
                if j > i:  # or: if bath_qubit_j > bath_qubit_i
                    g_ij = self.bath_bath_interactions[i, j]
                    # Only add if non-zero
                    if g_ij != 0.0:
                        coefficients.append(g_ij)
                        operators.append(
                            self.get_ZZ_coupling(bath_qubit_i, bath_qubit_j)
                        )
                        print(
                            f"Adding BB interactions for q{bath_qubit_i} and q{bath_qubit_j} "
                            f"with coeff={g_ij:.3f} (indices {(i,j)})."
                        )

        # Construct the total bath Hamiltonian
        H_bath = qml.dot(coefficients, operators)
        return H_bath
    def describe_bath_interactions_analog(self, analytical=True):
        """
        Summarize the system-bath and bath-bath interactions in an 'analog' style.

        Args:
            analytical (bool): If True, return symbolic variable names. If False, return numeric values.
        
        Returns:
            str: A multi-line string describing the bath interactions.
        """
        if not self.bath or self.num_bath == 0:
            return "No bath qubits present. No system–bath or bath–bath interactions."

        interaction_terms = []

        # -------------------------
        # 1) System–Bath Interactions
        # gamma_matrix[b, s] => \sigma^z_{s} \otimes \sigma^x_{b}
        # s runs over self.network_wires, b runs over self.bath_qubits
        # -------------------------
        if self.gamma_matrix is not None:
            nb, nsys = self.gamma_matrix.shape
            for b_idx in range(nb):
                bath_q = self.bath_qubits[b_idx]
                for s_idx in range(nsys):
                    if analytical:
                        coupling_str = f"γ_{{b={b_idx},s={s_idx}}}"
                    else:
                        val = float(self.gamma_matrix[b_idx, s_idx])
                        coupling_str = f"{val:.2e}"
                    interaction_terms.append(
                        f"{coupling_str} * [Z({s_idx}) @ X({bath_q})]"
                    )
        else:
            interaction_terms.append("No system–bath couplings (gamma_matrix is None).")

        # -------------------------
        # 2) Bath–Bath Interactions
        # bath_bath_interactions[i, j] => \sigma^z_i \otimes \sigma^z_j
        # i, j run over self.bath_qubits
        # -------------------------
        if (self.bath_bath_interactions is not None and
            isinstance(self.bath_bath_interactions, jnp.ndarray)):

            for i in range(self.num_bath):
                for j in range(i+1, self.num_bath):
                    val = self.bath_bath_interactions[i, j]
                    if abs(val) < 1e-14:
                        # skip near-zero couplings
                        continue
                    if analytical:
                        coeff_str = f"Λ_{{{i},{j}}}"
                    else:
                        coeff_str = f"{float(val):.2e}"
                    bq_i = self.bath_qubits[i]
                    bq_j = self.bath_qubits[j]
                    interaction_terms.append(
                        f"{coeff_str} * [Z({bq_i}) @ Z({bq_j})]"
                    )
        else:
            interaction_terms.append("No bath–bath couplings (bath_bath_interactions is None or empty).")

        if len(interaction_terms) == 0:
            return "No bath interactions present."

        # Format
        return "Bath Interactions (Analog):\n  " + "\n  ".join(interaction_terms)
    def describe_analog_hamiltonian(self, x_coeff, y_coeff, z_coeff, J_coeffs, analytical=True):
        """
        Provide a high-level description of the entire analog Hamiltonian, including:
        - Local fields on reservoir qubits
        - Reservoir–reservoir couplings (K_coef)
        - System–bath couplings (gamma_matrix)
        - Bath–bath interactions (bath_bath_interactions)
        - Control–reservoir XY couplings (J_coeffs)
        
        This parallels the digital version but references the analog variables
        in self.
        
        Args:
            x_coeff: numeric or symbolic X-field coefficient for reservoir qubits
            y_coeff: numeric or symbolic Y-field coefficient for reservoir qubits
            z_coeff: numeric or symbolic Z-field coefficient for reservoir qubits
            J_coeffs: array of shape (num_J,) with control–reservoir XY couplings
            analytical (bool): If True, return symbolic strings; if False, numeric.

        Returns:
            str: multi-line string describing the Hamiltonian terms.
        """
        h_terms = []

        # 1) Local fields on reservoir qubits: x_coeff, y_coeff, z_coeff
        for r in self.reserv_qubits:
            if analytical:
                h_terms.append(f"hx * X({r})")
                h_terms.append(f"hy * Y({r})")
                h_terms.append(f"hz * Z({r})")
            else:
                h_terms.append(f"{x_coeff} * X({r})")
                h_terms.append(f"{y_coeff} * Y({r})")
                h_terms.append(f"{z_coeff} * Z({r})")

        # 2) Reservoir–Reservoir couplings: self.k_coefficient => XY couplings
        # K_{a,b} * (X(a) X(b) + Y(a) Y(b))
        if self.N_reserv > 1:
            for i in range(self.N_reserv):
                for j in range(i+1, self.N_reserv):
                    k_val = self.k_coefficient[i, j]
                    q_i = self.reserv_qubits[i]
                    q_j = self.reserv_qubits[j]
                    if analytical:
                        k_str = f"K_{{r={i},{j}}}"
                    else:
                        k_str = f"{float(k_val):.2e}"
                    h_terms.append(
                        f"{k_str} * [X({q_i})@X({q_j}) + Y({q_i})@Y({q_j})]"
                    )

        # 3) Control–Reservoir couplings: J_coeffs => for each reservoir qubit + each control qubit
        # J_{(a, c)} * (X(a) X(c) + Y(a) Y(c))
        idx = 0
        for i, rsv_q in enumerate(self.reserv_qubits):
            for c_q in self.ctrl_qubits:
                val = J_coeffs[idx]
                idx += 1
                if analytical:
                    j_str = f"J_{{r={i},c={c_q}}}"
                else:
                    j_str = f"{float(val):.2e}"
                h_terms.append(
                    f"{j_str} * [X({rsv_q})@X({c_q}) + Y({rsv_q})@Y({c_q})]"
                )

        # 4) System–Bath couplings: self.gamma_matrix => Z(s) X(b)
        #    s in [self.ctrl_qubits + self.reserv_qubits], b in self.bath_qubits
        if self.bath and self.gamma_matrix is not None:
            nb, nsys = self.gamma_matrix.shape
            for b_idx in range(nb):
                bq = self.bath_qubits[b_idx]
                for s_idx in range(nsys):
                    val = self.gamma_matrix[b_idx, s_idx]
                    if analytical:
                        gamma_str = f"γ_{{b={b_idx},s={s_idx}}}"
                    else:
                        gamma_str = f"{float(val):.2e}"
                    h_terms.append(
                        f"{gamma_str} * [Z({s_idx})@X({bq})]"
                    )

        # 5) Bath–Bath interactions: self.bath_bath_interactions => ZZ
        if self.bath and self.bath_bath_interactions is not None:
            if isinstance(self.bath_bath_interactions, jnp.ndarray) and self.num_bath > 1:
                for i in range(self.num_bath):
                    for j in range(i+1, self.num_bath):
                        val = self.bath_bath_interactions[i, j]
                        bq_i = self.bath_qubits[i]
                        bq_j = self.bath_qubits[j]
                        if abs(val) < 1e-14:
                            continue
                        if analytical:
                            lambd_str = f"Λ_{{b={i},{j}}}"
                        else:
                            lambd_str = f"{float(val):.2e}"
                        h_terms.append(
                            f"{lambd_str} * [Z({bq_i})@Z({bq_j})]"
                        )

        if len(h_terms) == 0:
            return "H_analog: ( no terms )"
        joined_terms = " + \n    ".join(h_terms)
        return f"H_analog: (\n    {joined_terms}\n)"

  

def run_test(params, num_epochs, N_reserv, N_ctrl, time_steps,N_train,folder,
             gate,gate_name,init_params_dict,dataset_key, bath=False,num_bath = 0,gamma_scale=0.05,
             lambda_scale=0.01, bath_factor= 0.1,time_dependent_bath=False,positions_sys=None,positions_bath=None,spectral_exponent=1.0):
    float32=''
    opt_lr = None
    preopt_results = None
    selected_indices, min_var_indices,replacement_indices = [],[],[]
    num_states_to_replace = N_train // 5

    num_J = N_ctrl*N_reserv
    folder_gate = folder +'/'+gate_name + '/reservoirs_' + str(N_reserv) + '/trotter_step_' + str(time_steps) +'/'
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
    sim_qr = Sim_QuantumReservoir(params=init_params_dict, 
                                  N_ctrl=N_ctrl, 
                                  N_reserv=N_reserv, 
                                  num_J=N_reserv * N_ctrl,
                                  time_steps=time_steps,
                                  bath=bath,
                                  num_bath=num_bath,
                                  bath_factor=bath_factor,
                                  gamma_scale=gamma_scale,
                                  lambda_scale=lambda_scale,
                                  PRNG_key=dataset_key,
                                  time_dependent_bath=time_dependent_bath,
                                  positions_sys=positions_sys,
                                  positions_bath=positions_bath,
                                  spectral_exponent=spectral_exponent
                                  )
    

    init_params = params

    filename = os.path.join(folder_gate, f'data_run_{len(files_in_folder)}.pickle')

    # opt_a,opt_b = generate_dataset(gate, N_ctrl, N_train + 2000, key= random_key) 
    input_states, target_states = generate_dataset(gate, N_ctrl,training_size= N_train, key= dataset_key, new_set=False)
    # print(f"training state #1: {input_states[0]}")
    # preopt_results, input_states, target_states,second_A,second_b,f_A,f_b,opt_lr,selected_indices = optimize_traingset(gate,N_ctrl, N_reserv,time_steps, params, init_params_dict, N_train,num_datasets=5,key=dataset_key)
    ts1 = input_states[0]


    test_dataset_key = jax.random.split(dataset_key)[1]
    test_in, test_targ = generate_dataset(gate, N_ctrl,training_size= 2000, key=test_dataset_key, new_set=False)
    


    parameterized_ham = sim_qr.get_total_hamiltonian_components()
    if sim_qr.bath:
        # H_bath = sim_qr.get_H_bath()
        H_bath = sim_qr.get_H_bath_new()
        # print(f"H_bath: {H_bath}")
        # total_H = parameterized_ham+H_bath
        total_H = H_bath+parameterized_ham
    else:
        total_H = parameterized_ham


    ctrl_wires = sim_qr.get_ctrl_wires()
    reserv_wires = sim_qr.get_reserv_wires()
    qnode_dev = sim_qr.get_dev()
    all_wires = sim_qr.get_all_wires()
    # print(f"H: {total_H}")
   

    @qml.qnode(qnode_dev, interface="jax")
    def circuit(params, state_input):
        # print("DEBUG: params.shape =", np.array(params).shape)
        
        taus = params[:time_steps]
        # print("DEBUG: type(taus) =", type(taus))

        qml.StatePrep(state_input, wires=[*ctrl_wires])
        

        for idx, tau in enumerate(taus):
           
            hx_array = np.array([params[time_steps]])  # Convert hx to a 1D array
            hy_array = np.array([params[time_steps + 1]])  # Convert hy to a 1D array
            hz_array = np.array([params[time_steps + 2]])  # Convert hz to a 1D array
            J_values = params[time_steps + 3 + idx * num_J : time_steps + 3 + (idx + 1) * num_J]

            # Concatenate hx_array with J_values
            current_step = np.concatenate([J_values,hx_array,hy_array,hz_array])
            
            qml.evolve(total_H)(current_step, t=tau)
            
        return qml.density_matrix(wires=[*ctrl_wires])
    
    specs_func = qml.specs(circuit)
    specs = specs_func(params,input_states[0])
    circuit_depth = specs['resources'].depth
    num_gates = specs['resources'].num_gates

    # print(f"spcs: {specs_func(params,input_states[0])}")
    # print(f"Running circuit without JIT:{circuit(params, input_states[0])}")
    # out = circuit(params, input_states[0])
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
        # params = jnp.reshape(params, (-1,))
        X = jnp.asarray(input_states, dtype=jnp.complex128)
        y = jnp.asarray(target_states, dtype=jnp.complex128)
        # Process the batch of states
        loss = batched_cost_helper(params, X, y)
        loss = jnp.maximum(loss, 0.0)  # Apply the cutoff to avoid negative costs

        return loss
   

    

    def final_test(params,test_in,test_targ):
        params = jnp.asarray(params, dtype=jnp.float64)
        X = jnp.asarray(test_in, dtype=jnp.complex128)
        y = jnp.asarray(test_targ, dtype=jnp.complex128)
        batched_output_states = vcircuit(params, X)

        fidelities = jax.vmap(qml.math.fidelity)(batched_output_states, y)
        fidelities = jnp.clip(fidelities, 0.0, 1.0)

        return fidelities
    @jit
    def update(params, opt_state, input_states, target_states, value):
        """Update all parameters including tau."""
        
        loss, grads = jax.value_and_grad(cost_func)(params, input_states, target_states)
        updates, opt_state = opt.update(grads, opt_state, params=params, value=value)
        new_params = optax.apply_updates(params, updates)
        # Ensure outputs are float64
        loss = jnp.asarray(loss, dtype=jnp.float64)
        grads = jnp.asarray(grads, dtype=jnp.float64)
        return new_params, opt_state, loss, grads

        
        
    #     return new_params, opt_state, loss, grads

    
    if opt_lr == None:
        # s = time.time()
        
        init_loss, init_grads = jax.value_and_grad(cost_func)(params, input_states, target_states)
        # e = time.time()
        # dt = e - s
        # raw_lr,clipped_lr,grad_norm = get_base_learning_rate(init_grads)
        opt_lr = get_initial_lr_per_param(
            init_grads,
            base_step=0.01,
            max_lr=0.2,
            debug=False

        )
       
        # print(f"Adjusted initial learning rate: {opt_lr:.2e}. Grad_norm: {1/grad_norm},Grad_norm: {grad_norm:.2e}")
        cost = init_loss

    opt_descr = 'per param'
    
    print("________________________________________________________________________________")
    print(f"\nStarting optimization for {gate_name} with optimal lr {jnp.mean(opt_lr):.4e} time_steps = {time_steps}, N_r = {N_reserv}, N_bath = {num_bath}...\n")
    bath_stats = {}
    bath_stats['gamma_matrix'] = sim_qr.gamma_matrix
    bath_stats['bath_bath_int'] = sim_qr.bath_bath_interactions
    bath_stats['positions_sys'] = sim_qr.positions_sys
    bath_stats['positions_bath'] = sim_qr.positions_bath
    if sim_qr.bath_bath_interactions is not None:
        print(f"Gamma scale: {gamma_scale}")
        print(f"Lambda scale: {lambda_scale}")
        # print(f"Bath Factor (general bath scalar): {bath_factor}")
        # print(f"B-B Interactions {sim_qr.bath_qubits}: {sim_qr.bath_bath_interactions}")
    idx = 0
    hx_array = np.array([params[time_steps]])  # Convert hx to a 1D array
    hy_array = np.array([params[time_steps + 1]])  # Convert hy to a 1D array
    hz_array = np.array([params[time_steps + 2]])  # Convert hz to a 1D array
    J_values = params[time_steps + 3 + idx * num_J : time_steps + 3 + (idx + 1) * num_J]
    hamiltonian_str = sim_qr.describe_analog_hamiltonian(hx_array, hy_array, hz_array, J_values,analytical=False)
    interaction_description = sim_qr.describe_bath_interactions_analog(analytical=False)
    print(interaction_description)

   

    # print("H: ",total_H)
    specs = specs_func(params,input_states[0])
    circuit_depth = specs['resources'].depth
    gate_count = specs['resources'].num_gates
    # print(f"Depth: {circuit_depth}, gates: {gate_count}, Number of trainable parameters: {len(params)}")

    
    # print(f"initial loss: {init_loss}")
    
    
   

    opt = optax.chain(
        optax.clip_by_global_norm(1.0),  # Clip gradients to prevent explosions
        optax.adam(learning_rate=opt_lr, b1=0.99, b2=0.999, eps=1e-7)  # Slightly more aggressive Adam
    )
   
    



    

   

    

    costs = []
    param_per_epoch = []
    # print(f"Params: {params}")
    opt_state = opt.init(params)

    time_log_filename = os.path.join(folder_gate, f'times_log_{diable_jit}.txt')

    # Define the gradient function outside the loop
    #cost_and_grad = jax.value_and_grad(partial(cost_func, time_steps=time_steps, N_reserv=N_reserv, N_ctrl=N_ctrl))
    prev_cost = float('inf')  # Initialize with infinity
    consecutive_improvement_count = 0
    cost_threshold = 1e-5
    threshold_counts = 0
    consecutive_threshold_limit = 4  # Number of consecutive epochs below threshold without improvement needed to stop
    backup_params = init_params
    improvement = True
    backup_cost = float('inf')  
    epoch = 0
    a_condition_set,replace_states = False,False
    a_threshold =  0.0
    s = time.time()
    full_s =s
    training_state_metrics = {}
    grads_per_epoch = []
    threshold_cond1, threshold_cond2 = [],[]
    prev_cost, second_prev_cost = float('inf'), float('inf')
    acceleration = 0.0
    rocs = []
    stored_epoch = 1
    scale_reduction_epochs,learning_rates = [],[]  # Track epochs where scale is reduced
    cond1,cond2,a_threshold = -float('inf'), -float('inf'), -float('inf') 
    add_more=False
    num_states_to_replace = N_train // 4
    while epoch < num_epochs or improvement:
        params, opt_state, cost, grad = update(params, opt_state, input_states, target_states,value=cost)
        learning_rates.append('fixed')
        if epoch > 1:
            var_grad = np.var(grad,ddof=1)
            mean_grad = np.mean(jnp.abs(grad))
            if epoch >5:
                threshold_cond1.append(np.abs(mean_grad))
                threshold_cond2.append(var_grad)
            if epoch == 15:
                initial_meangrad = np.mean(np.array(threshold_cond1))
                initial_vargrad = np.mean(np.array(threshold_cond2))
                cond1  = initial_meangrad * 1e-1
                # print(f"    - setting cond1: initial mean(grad) {initial_meangrad:2e}, threshold: {cond1:2e}")
                cond2 = initial_vargrad * 1e-2
                # print(f"    - setting cond2: initial var(grad) {initial_vargrad:2e}, threshold: {cond2:2e}")
            
            acceleration = get_rate_of_improvement(cost,prev_cost,second_prev_cost)
            if epoch >= 25 and not a_condition_set and acceleration < 0.0:
                average_roc = np.mean(np.array(rocs[10:]))
                a_marked = np.abs(average_roc)
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
        if epoch == 0 or (epoch + 1) % 200 == 0:
            var_grad = np.var(grad,ddof=1)
            mean_grad = np.mean(jnp.abs(grad))
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
            current_cost_check = cost_func(params, input_states,target_states)
            backup_cost_check = cost_func(backup_params, input_states,target_states)
            if np.abs(backup_cost_check-backup_cost) > 1e-6 and epoch > 2:
                print(f"Back up cost different then its check. diff: {backup_cost_check-backup_cost:.3e}\nbackup params: {backup_params}")
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
        for i in range(time_steps):
            if params[i] < 0:
                params = params.at[i].set(np.abs(params[i]))
        var_condition= np.var(grad,ddof=1) < 1e-14
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
    # filename = os.path.join(folder_gate, f'{min_var_indices}.pickle')
    testing_results = final_test(params,test_in, test_targ)
    avg_fidelity = jnp.mean(testing_results)
    infidelities = 1.00000000000000-testing_results
    avg_infidelity = np.mean(infidelities)
    
    print(f"\nAverage Final Fidelity: {avg_fidelity:.5f}")
    
    data = {'Gate':base64.b64encode(pickle.dumps(gate)).decode('utf-8'),
            'opt_description': opt_descr,
            'specs':specs,
                'epochs': num_epochs,
                'lrs': learning_rates,
                'bath_stats':bath_stats,
                'gamma_scale':gamma_scale,
                'lambda_scale':lambda_scale,
                'bath_factor':bath_factor,
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
                'testing_results': testing_results,
                'avg_fidelity': avg_fidelity,
                'costs': costs,
                'params_per_epoch':param_per_epoch,
                'training_states': input_states,
                'opt_params': params,
                'opt_lr': opt_lr,
                'grads_per_epoch':grads_per_epoch,
                'bath': bath,
                'num_bath':num_bath,
                'partial_rho_qfim':True,
                'infidelities':infidelities,
                'training_state_metrics':training_state_metrics,
                'depth':circuit_depth,
                'total_time':epoch_time,
                'init_grads':init_grads

                
                
            }
    print(f"Saving results to {filename}\n\n")
    df = pd.DataFrame([data])
    while os.path.exists(filename):
        name, ext = filename.rsplit('.', 1)
        filename = f"{name}_.{ext}"

    with open(filename, 'wb') as f:
        pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)




if __name__ == '__main__':


    

    


    # run below 
    N_ctrl = 2
    
    # trots = [1,2,3,4,5]
    trots = [2,4,6,8,10]
    trots = [4,6]
    res = [1]

    # trots = [1,4,6,8,10,12,14]

    

    #res = [N_reserv]
    
    num_epochs = 1500
    N_train = 20
    
 
    # gamma_scale_values = [0.05, 0.1, 0.2]  # System-bath coupling scale
    # lambda_scale_values = [0.01, 0.05, 0.1]  # Bath-bath coupling scale
    gamma_scale_values = [0.01]  # System-bath coupling scale
    lambda_scale_values = [0.5]  # Bath-bath coupling scale
    bath_factor = 1.0
    time_dependent_bath = False
    if time_dependent_bath:
        base_folder_template = (f'./analog_results_trainable_baths/trainsize_{N_train}_epoch{num_epochs}_time_dependent_per_param_opt/bath_factor_{{bath_factor}}/'
                   'gamma_{gamma_scale}/lambda_{lambda_scale}/'
        )
    else:
        base_folder_template = (f'./analog_results_trainable_baths_realistic/trainsize_{N_train}_epoch{num_epochs}_per_param_opt/bath_factor_{{bath_factor}}/'
                    'gamma_{gamma_scale}/lambda_{lambda_scale}/'
            )
   

    gates_random = []
    
    # baths = [False,True]
    # num_baths = [0,1]
    baths = [True]
    num_baths = [1]
    baths = [True]
    num_baths = [2]
    # baths = [False,True,True]
    # num_baths = [0,1,2]
    for i in range(20):
        U = random_unitary(2**N_ctrl, i).to_matrix()
        #pprint(Matrix(np.array(U)))
        g = partial(qml.QubitUnitary, U=U)
        g.num_wires = N_ctrl
        g.name = f'U{N_ctrl}_'+str(i)
        gates_random.append(g)

   

    for gamma_scale in gamma_scale_values:
            for lambda_scale in lambda_scale_values:
                base_folder = base_folder_template.format(
                    bath_factor=bath_factor, gamma_scale=gamma_scale, lambda_scale=lambda_scale
                )
                for gate_idx,gate in enumerate(gates_random):

                    # if not gate_idx in [0]:
                    #     continue
                    # if gate_idx != 0:
                    #     continue
                

                    for time_steps in trots:

                        
                        
                        
                        for N_reserv in res:
                            
                            N =N_ctrl+N_reserv
                            
                            #folder = f'./param_initialization/Nc{N_ctrl}_Nr{N_reserv}_dt{time_steps}/fixed_params4/test7/'
                            for num_bath,bath in zip(num_baths,baths):
                                folder = os.path.join(base_folder, f"{num_bath}_num_baths/")
                                params_key_seed = gate_idx*121 * N_reserv + 12345 * time_steps *N_reserv
                                params_key = jax.random.PRNGKey(params_key_seed)
                                dataset_seed = N_ctrl * gate_idx + gate_idx**2 + N_ctrl
                                dataset_key = jax.random.PRNGKey(dataset_seed)
                                main_params = jax.random.uniform(params_key, shape=(3 + (N_ctrl * N_reserv) * time_steps,), minval=-np.pi, maxval=np.pi)
                                # print(f"main_params: {main_params}")
                                params_key, params_subkey1, params_subkey2 = jax.random.split(params_key, 3)
                                
                                
                                time_step_params = jax.random.uniform(params_key, shape=(time_steps,), minval=0, maxval=np.pi)
                                K_half = jax.random.uniform(params_subkey1, (N, N))
                                K = (K_half + K_half.T) / 2  # making the matrix symmetric
                                K = 2. * K - 1.
                                init_params_dict = {'K_coef': jnp.asarray(K)}
                            


                                # Combine the two parts
                                params = jnp.concatenate([time_step_params, main_params])
                                # params = jnp.asarray([0.4033546149730682, 1.4487122297286987, 2.3020467758178711, 2.9035964012145996, 0.9584765434265137, 1.7428307533264160, -1.3020169734954834, -0.8775904774665833, 2.4736261367797852, -0.4999605417251587, -0.8375297188758850, 1.7014273405075073, -0.8763229846954346, -3.1250307559967041, 1.1915868520736694, -0.4640290737152100, -1.0656110048294067, -2.6777451038360596, -2.7820897102355957, -2.3751690387725830, 0.1393062919378281])
                                # print(f"time_step_params: {time_step_params}")
                                # print(params)

                                run_test(
                                        params=params, 
                                        num_epochs=num_epochs, 
                                        N_reserv=N_reserv, 
                                        N_ctrl=N_ctrl, 
                                        time_steps=time_steps, 
                                        N_train=N_train, 
                                        folder=folder, 
                                        gate=gate, 
                                        gate_name=gate.name, 
                                        init_params_dict=init_params_dict,
                                        dataset_key = dataset_key,
                                        bath=bath,
                                        num_bath = num_bath,
                                        gamma_scale=gamma_scale, 
                                        lambda_scale=lambda_scale, 
                                        bath_factor=bath_factor, 
                                        time_dependent_bath=time_dependent_bath,
                                        positions_sys=None,
                                        positions_bath=None,
                                        spectral_exponent=1.0
                                        
                                    )

                                