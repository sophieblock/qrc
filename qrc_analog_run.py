import pennylane as qml
import os

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
import ast
from optax.tree_utils import tree_get
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



def extend_global_key_pool(key, new_size):
    """
    Extend the global key pool with new keys using a given JAX random key.
    """
    global GLOBAL_KEY_POOL
    # Generate additional keys
    new_keys = jax.random.split(key, num=new_size)
    GLOBAL_KEY_POOL.extend(new_keys)

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
        bath_array = 0.01 * jax.random.normal(key, (num_bath, N_ctrl + N_reserv))
        return {
            

            'K_coef': jnp.asarray(K),
            'bath':bath_array
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

def estimate_reductions_and_factor_with_gradients(
    total_epochs, accumulation_size, patience, cooldown, initial_lr, min_scale, init_gradients_array, rtol, atol
):
    """
    Estimate the number of reductions and compute the ideal factor for reduce_on_plateau
    when initial gradients are given as an array.

    Args:
        total_epochs (int): Total number of training epochs.
        accumulation_size (int): Number of updates before evaluating improvement.
        patience (int): Number of evaluations to wait for improvement before reducing LR.
        cooldown (int): Cooldown period after a reduction.
        initial_lr (float): Initial learning rate (η0).
        min_scale (float): Minimum learning rate (η_min).
        init_gradients_array (array-like): Array of initial gradient magnitudes for each parameter.
        rtol (float): Relative tolerance for improvement.
        atol (float): Absolute tolerance for improvement.

    Returns:
        tuple: Estimated number of reductions (n) and ideal factor (f).
    """
    # Compute the mean gradient magnitude across all parameters
    mean_gradients = np.mean(np.abs(init_gradients_array))

    # Calculate number of evaluations during training
    num_evaluations = total_epochs / accumulation_size

    # Estimate the effective number of plateaus (potential reductions)
    num_plateaus = num_evaluations / (patience + cooldown)
    print(f"num_plateaus: {num_plateaus}")
    # Estimate patience failure rate based on improvement likelihood
    improvement_likelihood = mean_gradients / initial_lr
    print(f"improvement_likelihood: {improvement_likelihood}")
    tolerance_threshold = rtol * mean_gradients + atol
    print(f"tolerance_threshold: {tolerance_threshold}")
     # Patience failure rate: Likelihood of not improving
    patience_failure_rate = max(1 - min(improvement_likelihood / tolerance_threshold, 1), 0)
    print(f"patience_failure_rate: {patience_failure_rate}")

    # Estimate reductions based on plateaus and failure rate
    num_reductions = num_plateaus * patience_failure_rate


    # Compute the ideal factor if reductions > 0
    if num_reductions > 0:
        ideal_factor = (min_scale / initial_lr) ** (1 / num_reductions)
    else:
        ideal_factor = 1.0  # No reductions expected

    return num_reductions, ideal_factor



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
def calculate_iqr(data, x,y):
    """
    Calculate the Interquartile Range (IQR) of the input data.
    """
    iqr = np.percentile(data, x) - np.percentile(data, y)

   
    return iqr

def compute_initial_learning_rate(gradients, scale_factor=0.1, min_lr=1e-3, max_lr = 0.2):
    """
    Compute an initial learning rate based on the norm of gradients.
    """
    # Compute the norm of the gradients
    
    norm_grad = jnp.linalg.norm(gradients)
    min_abs_grad = jnp.min(jnp.abs(gradients))
    #mean_norm_grad = jnp.mean(norm_grad)
    initial_lr = scale_factor / (norm_grad + 1e-8)  # Adding a small value to prevent division by zero
    print(norm_grad, initial_lr, initial_lr / (min_abs_grad * 10))
    #initial_lr =initial_lr / (min_abs_grad * 10)
    initial_lr = jnp.clip(initial_lr, min_lr, max_lr)
    return initial_lr

def apply_adaptive_penalty(var_grad, iqr_var_grad,norm_var_grad_var, num_qubits, num_time_steps, num_parameters, weight_mean_grad, weight_aggregated_var, all_variances, all_iqrs,all_var_grad_vars):
    """
    Applies adaptive penalties to the variance and IQR based on system characteristics and number of parameters.
    Logs details about penalties applied.
    """
    # Get the percentiles for variance and IQR for adaptive penalty
    # Convert lists to NumPy arrays
    all_variances = np.array(all_variances)
    all_iqrs = np.array(all_iqrs)
    all_var_grad_vars = np.array(all_var_grad_vars)  
   
    # print(f"all_variances.shape: {all_variances.shape}")
    var_percentile_90 = np.percentile(all_variances, 90)  # 90th percentile of variance
    iqr_percentile_10 = np.percentile(all_iqrs, 10)  # 10th percentile for IQR
    var_grad_var_percentile_90 = np.percentile(all_var_grad_vars, 90)  # 90th percentile of var_grad variance
    var_grad_var_percentile_10 = np.percentile(all_var_grad_vars, 50)  # 90th percentile of var_grad variance
    
    # print(f"var_percentile_90: {var_percentile_90:5e}, iqr_percentile_10: {iqr_percentile_10:5e}, var_grad_var_percentile_90: {var_grad_var_percentile_90:5e}, var_grad_var_percentile_10: {var_grad_var_percentile_10:5e}")
    # Normalize based on system size (qubits * time_steps * num_parameters)
    # norm_var_grad = var_grad / (num_qubits * num_time_steps * num_parameters)
    # norm_iqr_var_grad = iqr_var_grad / (num_qubits * num_time_steps * num_parameters)
    norm_var_grad = var_grad 
    norm_iqr_var_grad = iqr_var_grad 
   
    # Initialize penalty factor
    penalty_factor = 1.0
    # print(f"\nEvaluating penalties for dataset with norm_var_grad: {norm_var_grad:.5e}, norm_iqr_var_grad: {norm_iqr_var_grad:.5e}, norm_var_grad_var: {norm_var_grad_var:.5e}")

    # Only penalize if the variance is extremely high and the IQR is also large (indicating instability)
    if norm_var_grad > var_percentile_90 and norm_iqr_var_grad > iqr_percentile_10:
        penalty_factor *= 0.7  # Penalize unstable datasets (reduce to 70%)
        # print(f"Penalized for high variance and high instability. Penalty factor reduced to {penalty_factor:.2f}")
    
    # Reward stability (low IQR, less than 10th percentile)
    if norm_iqr_var_grad < iqr_percentile_10:
        penalty_factor *= 1.3  # Reward more stable datasets
        # print(f"Rewarded for stability (low IQR less than 10th percentile). Penalty factor increased to {penalty_factor:.2f}")
    # Reward stability (low IQR, less than 10th percentile)
    if norm_var_grad_var < var_grad_var_percentile_10:
        penalty_factor *= 1.3  # Reward more stable datasets
        # print(f"Rewarded for stability (low var(var_grad) less than 20th percentile). Penalty factor increased to {penalty_factor:.2f}")
    
    # Penalize large variance of var_grad (instability)
    if norm_var_grad_var > var_grad_var_percentile_90:
        penalty_factor *= 0.5  # Penalize unstable datasets with large var_grad variance
        # print(f"Penalized for large var_grad variance (greater than 95th percentile). Penalty factor reduced to {penalty_factor:.2f}")

    # Calculate the weighted sum with penalties applied
    weighted_sum = penalty_factor * norm_var_grad 
    
    # print(f"Final weighted sum with penalties applied: {weighted_sum:.5e} norm_var_grad: {norm_var_grad:.5e}")
    return weighted_sum


def optimize_traingset(gate, N_ctrl, N_reserv, time_steps, params, init_params_dict, N_train, num_datasets, key):
    datasets = []
    print(f"{gate.name}, dt= {time_steps}: Pre-processing {num_datasets} training sets for selection...")
    all_A, all_b = [], []
    for i in range(num_datasets):
        key, subkey = jax.random.split(key)  # Split the key for each dataset generation
        A, b = generate_dataset(gate, N_ctrl, N_train + 2000, subkey)  # Generate dataset with the subkey
        all_A.append(A)
        all_b.append(b)
    all_A = jnp.stack(all_A)
    all_b = jnp.stack(all_b)
    
    num_J = N_reserv * N_ctrl
    
    sim_qr = Sim_QuantumReservoir(init_params_dict, N_ctrl, N_reserv, num_J, time_steps, bath, num_bath)
    parameterized_ham = sim_qr.get_total_hamiltonian_components()
    ctrl_wires = sim_qr.get_ctrl_wires()
    reserv_wires = sim_qr.get_reserv_wires()
    qnode_dev = sim_qr.get_dev()
    all_wires = sim_qr.get_all_wires()
    @jit
    @qml.qnode(qnode_dev, interface="jax")
    def circuit(params, state_input):
        taus = params[:time_steps]
        qml.StatePrep(state_input, wires=[*ctrl_wires])
        for idx, tau in enumerate(taus):
            hx_array = np.array([params[time_steps]])  # Convert hx to a 1D array
            hy_array = np.array([params[time_steps + 1]])  # Convert hy to a 1D array
            hz_array = np.array([params[time_steps + 2]])  # Convert hz to a 1D array
            J_values = params[time_steps + 3 + idx * num_J : time_steps + 3 + (idx + 1) * num_J]
            current_step = np.concatenate([J_values, hx_array, hy_array, hz_array])
            qml.evolve(parameterized_ham)(current_step, t=tau)
        return qml.density_matrix(wires=[*ctrl_wires])

    @jit
    def cost_func(params, input_state, target_state):
        output_state = circuit(params, input_state)
        fidelity = qml.math.fidelity(output_state, target_state)
        return 1 - fidelity  # Minimizing infidelity

    @jit
    def collect_gradients(params, input_states, target_states):
        grad_fn = jax.grad(cost_func, argnums=0)
        gradients = jax.vmap(grad_fn, in_axes=(None, 0, 0))(params, input_states, target_states)
        return gradients
    
    def calculate_unbiased_stats_global(gradients, abs_grad=True):
        """Calculate the mean and unbiased variance of the gradients across all states."""
        if abs_grad:
            gradients = jnp.abs(gradients)
        
        mean_grad = jnp.mean(gradients, axis=0)  # Mean across all gradients
        mean_grad_squared = jnp.mean(gradients ** 2, axis=0)  # Mean squared gradients across all states
        
        # Use ddof=1 for unbiased variance
        var_grad = jnp.var(gradients, axis=0, ddof=1)  # Unbiased variance across all states
        
        grad_norm = jnp.linalg.norm(mean_grad)  # Norm of the mean gradient
        
        return mean_grad, var_grad, grad_norm
    batched_collect_gradients = jax.vmap(collect_gradients, in_axes=(None, 0, 0))
    
    all_gradients = batched_collect_gradients(params, all_A[:, :N_train], all_b[:, :N_train])
    print(f'all_gradients shape: {all_gradients.shape}')
    # print(f'batched_collect_gradients shape: {batched_collect_gradients.shape}')
    # Normalize gradients before sending to the statistics function
    def normalize_gradients(gradients):
        norm = jnp.linalg.norm(gradients, axis=-1, keepdims=True) + 1e-8  # Add small epsilon to avoid division by zero
        return gradients / norm


     # Collect all variances and IQRs for percentile-based comparison
    # Lists to store metrics for normalization purposes
    var_grad_means,var_var_grads,var_grad_means_normalized,normalized_var_grad_norms = [], [],[], []
    grad_norms_normalized,grad_norms = [], []
    min_var_grad_means,max_var_grad_means = [],[]
    mean_normalized_var_grads = []
    min_var_grad_norm = np.inf
    max_var_grad_norm = 0.0
    normalized_variance_gradients = []
    mean_gradients_normalized,mean_gradients = [],[]
    min_gradvars,max_gradvars = [], []
    min_mean_grads,max_mean_grads = [], []
    # First, collect all var_grad_means and grad_norms for normalization
    for i in range(num_datasets):

        normalized_gradients_per_state = normalize_gradients(all_gradients[i])
            # Calculate stats for all training states
       

        mean_grad, var_grad, grad_norm = calculate_unbiased_stats_global(all_gradients[i])
        mean_grad_norm, var_grad_norm, grad_norm_norm = calculate_unbiased_stats_global(normalized_gradients_per_state)
        # print(gradnorm.shape, vargrad.shape)
        # print(f"A{i}: gradnorm: {gradnorm.mean():.2e}, grad_norm: {grad_norm:.2e}, vargrad ({vargrad.shape}): {vargrad.mean():.2e}, var_grad ({var_grad.shape}):  {var_grad.mean():.2e}")
        # print(f"    - grad_norm_norm: {grad_norm_norm:.2e}, var_grad_norm ({var_grad_norm.shape}): {var_grad_norm.mean():.2e}")
        mean_gradients.append(mean_grad)

        # _,_, normalized_var_grad = calculate_correct_normalized_var_grad(all_gradients[i])
        var_grad_means.append(var_grad.mean())
        min_vargrad = min(var_grad)
        max_vargrad = max(var_grad)
        min_gradvars.append(min_vargrad)
        max_gradvars.append(max_vargrad)
        min_mean_grads.append(min(np.abs(mean_grad)))
        max_mean_grads.append(max(np.abs(mean_grad)))
        
        var_var_grads.append(jnp.var(var_grad,ddof=1))

        grad_norms.append(grad_norm)
        

        var_grad_means_normalized.append(var_grad_norm.mean())
        normalized_var_grad_norms.append(np.float64(jnp.var(var_grad_norm,ddof=1))*100)
        # normalized_var_grad_norms.append(np.float64(variance_of_variance_gradients_normalized)*100)
        mean_gradients_normalized.append(mean_grad_norm)
        # normalized_variance_gradients.append(var_grad)
        grad_norms_normalized.append(grad_norm_norm)
        # print(f'set A{i}: normalized_var_grad: {normalized_var_grad.mean():.2e}, normalized_var_grad2: {normalized_var_grad2.mean():.2e}')
    

    
    min_var_var,max_var_var = min(var_var_grads), max(var_var_grads)
    min_var_grad = min(var_grad_means)


    print(f"Variance of the gradients: [min: {min(min_gradvars):.2e}), max: {max(max_gradvars):.2e}]")
    print(f"Mean gradients: [min: {min(min_mean_grads):.2e}), max: {max(max_mean_grads):.2e}]")
    def normalize_metric(value, min_val, max_val, epsilon=1e-6, upper_bound=0.999):
        if max_val > min_val:
            # Adjust to avoid exactly 0.0 and 1.0, and ensure upper bound doesn't reach 1.0
            normalized_value = (value - min_val ) / (max_val - min_val + 2 )
            return min(normalized_value, upper_bound)
        else:
            return 0.5  # Neutral value if min and max are the same

    # Store the gradient stats for all datasets
    results = {}

    # Variables to track the best datasets for three purposes
    best_for_initial_training_idx = None
    best_for_replacement_idx = None
    best_for_fine_tuning_idx = None
    best_initial_score = jnp.inf
    best_replacement_score = jnp.inf  # We want lower variability here
    best_fine_tuning_score = jnp.inf  # Small, but more precise gradient shifts



    # Initialize tracking for scores and results
    results = {}


    alpha = 0.1
    w1 = 0.8  # Weight for normalized variance of the gradient
    w2 = .2# Weight for normalized gradient norm
    w3 = 0.5 # Weight for normalized variance of the variance of the gradient

    beta =0.5
    min_norm_var_grad = min(normalized_var_grad_norms)
    max_norm_var_grad = max(normalized_var_grad_norms)
    print(f"min: {min_norm_var_grad:.2e}, max: {max_norm_var_grad:.2e}")
    min_mean = min(min_mean_grads)
    max_mean = max(max_mean_grads)
    min_var_grad = min(min_gradvars)
    max_var_grad = max(max_gradvars)
    max_scale = max(max_mean,max_var_grad)
    min_scale = min(min_mean,min_var_grad)
    print(f"Scaled: [min: {min_scale:.2e}), max: {max_scale:.2e}]")
    # First, compute all the gradient statistics and scores for each dataset
    for i in range(num_datasets):
        # normalized_variance_gradient = normalized_variance_gradients[i]


        mean_variance_of_gradient, mean_variance_of_gradient_normalized = var_grad_means[i],var_grad_means_normalized[i]  # Mean of the variance of the gradients (gradients no normalized)
        variance_of_variance_gradients,variance_of_variance_gradients_normalized = var_var_grads[i], normalized_var_grad_norms[i] # Variance of the variance of the gradients (gradients no normalized)
        mean_gradient, mean_gradient_normalized = mean_gradients[i], mean_gradients_normalized[i]
        grad_norm, grad_norm_normalized = grad_norms[i], grad_norms_normalized[i]
        # norm_var_grad_mean = normalized_var_grad.mean()  # Mean of normalized gradient variance across params
       
        # variance_of_variance_normalized_gradient = jnp.var(normalized_variance_gradient) # Varia
        # mean_of_variance_normalized_gradient = jnp.abs(normalized_variance_gradient).mean()

        average_of_mean_gradients_abs = np.abs(mean_gradient).mean()
        average_of_mean_gradients_normalized_abs = np.abs(mean_gradient_normalized).mean()
        variance_of_mean_gradients = jnp.var(mean_gradient,ddof=1)
        variance_of_mean_gradients_normalized = jnp.var(mean_gradient_normalized)
        min_grad = min(np.abs(mean_gradient))
        max_grad = max(np.abs(mean_gradient))

        variance_of_variance_gradients_normalized_scaled = normalize_metric(variance_of_variance_gradients_normalized, min_norm_var_grad,max_norm_var_grad)
        normalized_mean_variance_of_gradient = normalize_metric(mean_variance_of_gradient, min_scale,max_scale)
        normalized_mean_gradient_score = normalize_metric(average_of_mean_gradients_abs,  min_scale,max_scale)
        # normalized_mean_variance_of_gradient_normalized = normalize_metric(mean_variance_of_gradient_normalized, min_var_grad, max_var_grad)
        normalized_var_var_grads = normalize_metric(variance_of_variance_gradients,   min_scale,max_scale)
        normalized_var_mean_gradient = normalize_metric(variance_of_mean_gradients, min_scale,max_scale)


        initial_score = (
            w1 * (normalized_mean_variance_of_gradient)  # Reward high gradient variance
            +w2 * (normalized_mean_gradient_score)   # Reward high gradient norm
            # +w2*(average_of_mean_gradients_normalized_abs)
            -w3* (variance_of_variance_gradients_normalized_scaled)  # Penalize high variance of variance of gradients
             -beta * (normalized_var_mean_gradient)  # Penalize high variance of mean gradients
        )
        other_score = (
            w1 * (mean_variance_of_gradient)  # Reward high gradient variance
            +w2 * (average_of_mean_gradients_abs)   # Reward high gradient norm
            # +w2*(average_of_mean_gradients_normalized_abs)
            -w3* (variance_of_variance_gradients)  # Penalize high variance of variance of gradients
             -w3 * (variance_of_mean_gradients)  # Penalize high variance of mean gradients
        )
        
        # Replacement score favors stable gradients with moderate variance
        replacement_score = (
            mean_variance_of_gradient * np.exp(-beta * (grad_norm_normalized - 1) ** 2) 
        )
       
        # initial_Score =  normalized_mean_variance_of_gradient * np.exp(-alpha * (normalized_grad_norm_score - 1) ** 2)
       
        # replacement_score = (
        #     normalized_mean_variance_of_gradient * np.exp(-beta * (normalized_grad_norm_score - 1) ** 2) 
        # )


        fine_tuning_score = 0.5 * mean_variance_of_gradient_normalized + 0.5 * mean_variance_of_gradient_normalized
        # if i == 2:
        #     initial_score = 1.248400e-01+.5


        results[f"dataset_{i}"] = {
            "Mean(Var Grad)": mean_variance_of_gradient,
            "Var(Var Grad)": variance_of_variance_gradients,
            "Mean(Mean Grad)": average_of_mean_gradients_abs,
            "Var(Mean Grad)": variance_of_mean_gradients,
            "Min Gradient": min_grad,
            "Max Gradient": max_grad,
            "min_var_grad_means": min_gradvars[i],
            "max_var_grad_means": max_gradvars[i],
            "Gradient Norm": grad_norm,
            "Norm(Gradient Norm)":grad_norm_normalized,
            "Mean(Var Grad) [normalized]": mean_variance_of_gradient_normalized,  # Mean of normalized gradient variance
            "Var(Var Grad) [normalized]": variance_of_variance_gradients_normalized,    # Variance of normalized gradient variance
            "Mean(Mean Grad) [normalized]": average_of_mean_gradients_normalized_abs,  # Mean of normalized gradient variance
            "Var(Mean Grad) [normalized]": variance_of_mean_gradients_normalized,    # Variance of normalized gradient variance
            "Initial Score": initial_score,
            "Replacement Score": replacement_score,
            "Fine-Tuning Score": fine_tuning_score,
            "dataset": (all_A[i], all_b[i])  # Store dataset A and b
        }

        # # Print the detailed summary statistics for each dataset
        # print(f"(A{i}, b{i}):")
        # print(f"    Raw Var(Grad): {mean_variance_of_gradient:.2e}, Normalized Var(Grad): {mean_variance_of_gradient_normalized:.2e}")
        # # print(f"    Raw Mean(Grad):  {average_of_mean_gradients_abs:.2e}, Normalized Mean(Grad): {normalized_mean_gradient_score:.2e}")
        # print(f"    Var(Grad): {w1*normalized_mean_variance_of_gradient:.2e}, Mean(Grad): {w2*normalized_mean_gradient_score:.2e} ->  Var(Grad)+Mean(Grad) = {w1*normalized_mean_variance_of_gradient+w2*normalized_mean_gradient_score:.2e} [scaled]")
        # print(f"    Var(Var) scaled: {variance_of_variance_gradients_normalized_scaled:.2e}, Var(Mean(Grad)): {normalized_var_mean_gradient:.2e} -> {-w3*variance_of_variance_gradients_normalized_scaled+ -beta*normalized_var_mean_gradient:.2e}")
        # # print(f"    Var(Var) not scaled: {variance_of_variance_gradients_normalized:.2e}")
        # print(f"    Initial Score: {initial_score:.2e}, Replacement Score: {replacement_score:.2e}, Fine-Tuning Score: {fine_tuning_score:.2e}\n")

    

    # Now select the best datasets for initial training, replacement, and fine-tuning
    sorted_by_initial = sorted(results.items(), key=lambda x: x[1]["Initial Score"], reverse=True)
    sorted_by_replacement = sorted(results.items(), key=lambda x: x[1]["Replacement Score"], reverse=True)
    sorted_by_fine_tuning = sorted(results.items(), key=lambda x: x[1]["Fine-Tuning Score"])

    # Best for initial training
    # Use the numeric index directly when accessing arrays
    best_for_initial_training = sorted_by_initial[0]
    best_for_replacement = sorted_by_replacement[1] if sorted_by_replacement[0][0] == best_for_initial_training[0] else sorted_by_replacement[0]
    best_for_fine_tuning = sorted_by_fine_tuning[0]

    # Extract the indices and scores for printing
    best_for_initial_training_idx =  int(best_for_initial_training[0].split('_')[1])
    best_initial_score = best_for_initial_training[1]["Initial Score"]

    best_for_replacement_idx = int(best_for_replacement[0].split('_')[1])
    best_replacement_score = best_for_replacement[1]["Replacement Score"]

    best_for_fine_tuning_idx =  int(best_for_fine_tuning[0].split('_')[1])
    best_fine_tuning_score = best_for_fine_tuning[1]["Fine-Tuning Score"]
    # normalized_gradients_per_state = normalize_gradients(all_gradients[i])
    best_gradients =  all_gradients[best_for_initial_training_idx]
    # best_gradients =  normalize_gradients(all_gradients[best_for_initial_training_idx])
    initial_grad_norm = results[f"dataset_{best_for_initial_training_idx}"]["Gradient Norm"]
    initial_lr = compute_initial_learning_rate(best_gradients)

    print(f"Best Dataset for Initial Training: A{best_for_initial_training_idx}, with score: {best_initial_score:.4e}")
    print(f"Best Dataset for Replacement: A{best_for_replacement_idx}, with score: {best_replacement_score:.4e}")
    print(f"Best Dataset for Fine-Tuning: A{best_for_fine_tuning_idx}, with score: {best_fine_tuning_score:.4e}")
    print(f"Initial Gradient Norm: {initial_grad_norm:2e}")
    # Extract the actual data
    best_initial_A = best_for_initial_training[1]["dataset"][0]
    best_initial_b = best_for_initial_training[1]["dataset"][1]
    best_replacement_A = best_for_replacement[1]["dataset"][0]
    best_replacement_b = best_for_replacement[1]["dataset"][1]
    best_fine_tuning_A = best_for_fine_tuning[1]["dataset"][0]
    best_fine_tuning_b = best_for_fine_tuning[1]["dataset"][1]
    return results, best_initial_A, best_initial_b, best_replacement_A, best_replacement_b, best_fine_tuning_A, best_fine_tuning_b, initial_lr, (best_for_initial_training_idx,best_for_replacement_idx,best_for_fine_tuning_idx)


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

def calculate_gradient_stats_per_state(gradients,abs_grad=True):
    
    """Calculate the mean and variance of the gradients for each state."""
    if abs_grad:
        gradients = jnp.abs(gradients)
    mean_grad = jnp.mean(gradients, axis=-1)  # Mean across parameters for each state
    mean_grad_squared = jnp.mean(gradients ** 2, axis=-1)  # Mean squared gradients across parameters
    var_grad = mean_grad_squared - mean_grad ** 2  # Variance per state
    grad_norm = jnp.linalg.norm(gradients, axis=-1)  # Norm per state

    return mean_grad, var_grad, grad_norm

def calculate_unbiased_variance(gradients, abs_grad=True):
    if abs_grad:
        gradients = jnp.abs(gradients)
    mean_grad = jnp.mean(gradients, axis=-1)  # Mean across parameters for each state
    sum_sq_diff = jnp.sum((gradients - mean_grad[..., None]) ** 2, axis=-1)  # Sum of squared differences
    var_grad_unbiased = sum_sq_diff / (gradients.shape[-1] - 1)  # Use Bessel's correction
    grad_norm = jnp.linalg.norm(gradients, axis=-1)

    return mean_grad, var_grad_unbiased, grad_norm


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
def scale_by_param_lr(param_lr):
    """
    Custom gradient transformation that multiplies each gradient element
    by a per-parameter learning rate, given as a pytree `param_lr` that
    mirrors the structure of the parameters.
    
    Args:
      param_lr: pytree of per-parameter scaling factors.
      
    Returns:
      An (init_fn, update_fn) tuple for use in an optax chain.
    """
    def init_fn(params):
        # No state is needed for this simple scaling.
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        # Multiply each update by its corresponding per-parameter learning rate.
        scaled_updates = jax.tree_util.tree_map(lambda g, lr: g * lr, updates, param_lr)
        return scaled_updates, state

    return optax.GradientTransformation(init_fn, update_fn)
def get_initial_lr_per_param(grads, base_step=0.01, min_lr=1e-3, max_lr=0.2):
    # print(f"grads: {grads}")
    grad_magnitudes = jax.tree_util.tree_map(lambda g: jnp.abs(g) + 1e-12, grads)
    # print(f"grad_magnitudes: {grad_magnitudes}")
    lr_tree = jax.tree_util.tree_map(lambda g: base_step / g, grad_magnitudes)
    # print(f"lr_tree: {lr_tree}")
    lr_tree = jax.tree_util.tree_map(lambda lr: jnp.clip(lr, min_lr, max_lr), lr_tree)
    return lr_tree
@jit
def approximate_largest_hessian_eig(params, cost_func, input_states, target_states, shift=1e-3):
    """
    Approximate the largest diagonal entry of the Hessian by parameter-shift 
    second derivatives, then use it as a crude bound on the largest eigenvalue.

    Arguments:
      params: 1D array of trainable parameters
      cost_func: function(params, input_states, target_states) -> scalar cost
      input_states, target_states: training data for cost_func
      shift: finite-difference shift for second derivative (parameter shift)
    
    Returns:
      max_diag: a float that bounds the largest Hessian eigenvalue from above
    """
    # Evaluate the unshifted cost once for efficiency
    cost_center = cost_func(params, input_states, target_states)
    
    diag_vals = []
    # We loop over each parameter to compute second derivative wrt param[i]
    for i in range(len(params)):
        # param[i] + shift
        params_plus  = params.at[i].add( shift )
        # param[i] - shift
        params_minus = params.at[i].add(-shift)

        # Evaluate cost at plus/minus shifts
        cost_plus  = cost_func(params_plus,  input_states, target_states)
        cost_minus = cost_func(params_minus, input_states, target_states)

        # Approximate second derivative w.r.t. param i
        # [f(x+δ) - 2 f(x) + f(x-δ)] / δ^2
        second_deriv = (cost_plus - 2.0*cost_center + cost_minus) / (shift**2)

        diag_vals.append(second_deriv)
    
    # The maximum absolute diagonal entry is a rough upper bound on λ_max
    diag_vals_arr = jnp.array(diag_vals)
    max_diag = jnp.max(jnp.abs(diag_vals_arr))
    return max_diag

@jit
def get_initial_lr_hessian(params, cost_func, input_states, target_states, grads,
                           shift=1e-3, min_lr=1e-3, max_lr=0.2):
    """
    Use Hessian-based approach for the initial learning rate:
       lr ~ 1 / lambda_max
    where lambda_max is approximated by the largest diagonal entry of the Hessian.
    
    Arguments:
      params:       1D array of trainable parameters
      cost_func:    function(params, X, Y) -> scalar cost
      input_states, target_states: training data
      grads:        the gradient vector (∂C/∂θ)
      shift:        finite-difference shift for approximate_largest_hessian_eig
      min_lr, max_lr: clip bounds for the returned learning rate
    
    Returns:
      lr_approx:    the recommended initial learning rate
      grad_norm:    the norm of grads (for logging/debugging)
    """
    # Bound on the largest Hessian eigenvalue from diagonal approximations
    hess_bound = approximate_largest_hessian_eig(params, cost_func, input_states, target_states, shift)
    grad_norm = jnp.linalg.norm(grads)

    # If hess_bound is too small, fallback to e.g. 0.1
    lr_approx = jnp.where(hess_bound > 1e-12, 1.0 / hess_bound, 0.1)
    lr_approx = jnp.clip(lr_approx, min_lr, max_lr)
    return lr_approx, grad_norm

# def get_initial_lr_per_param(grads, base_step=0.01, min_lr=1e-4, max_lr=0.2):
#     """
#     Provide a *vector* of learning rates, one for each parameter,
#     inversely proportional to the param's gradient magnitude.
#     """
#     grad_magnitudes = jnp.abs(grads) + 1e-12
#     print(f"grad_magnitudes: {grad_magnitudes}")
#     lr_vector = base_step / grad_magnitudes
#     print(f"lr_vector: {lr_vector}")
#     lr_vector = jnp.clip(lr_vector, min_lr, max_lr)
#     return lr_vector
def run_test(params, num_epochs, N_reserv, N_ctrl, time_steps,N_train,folder,gate,gate_name,bath,num_bath,init_params_dict, dataset_key):
    float32=''
    opt_lr = None
    a_marked = None
    preopt_results = None
    selected_indices, min_var_indices,replacement_indices = [],[],[]
    num_states_to_replace = N_train // 5

    num_J = N_ctrl*N_reserv
    folder_gate = folder + str(num_bath) + '/'+gate_name + '/reservoirs_' + str(N_reserv) + '/trotter_step_' + str(time_steps) +'/' + 'bath_'+str(bath)+'/'
    Path(folder_gate).mkdir(parents=True, exist_ok=True)
    temp_list = list(Path(folder_gate).glob('*'))
    files_in_folder = []
    for f in temp_list:
        temp_f = f.name.split('/')[-1]
        
        if not temp_f.startswith('.'):
            files_in_folder.append(temp_f)
    
    k = 3
   
    if len(files_in_folder) >= k:
        print('Already Done. Skipping: '+folder_gate)
        print('\n')
        return

    # get PQC
    sim_qr = Sim_QuantumReservoir(init_params_dict, N_ctrl, N_reserv, N_reserv * N_ctrl,time_steps,bath,num_bath)
    

    init_params = params

    filename = os.path.join(folder_gate, f'data_run_{len(files_in_folder)}.pickle')

    # opt_a,opt_b = generate_dataset(gate, N_ctrl, N_train + 2000, key= random_key) 
   

    input_states, target_states = generate_dataset(gate, N_ctrl,training_size= N_train, key= dataset_key, new_set=False)



    test_dataset_key = jax.random.split(dataset_key)[1]
    test_in, test_targ = generate_dataset(gate, N_ctrl,training_size= 2000, key=test_dataset_key, new_set=False)
    

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
           
            hx_array = np.array([params[time_steps]])  # Convert hx to a 1D array
            hy_array = np.array([params[time_steps + 1]])  # Convert hy to a 1D array
            hz_array = np.array([params[time_steps + 2]])  # Convert hz to a 1D array
            J_values = params[time_steps + 3 + idx * num_J : time_steps + 3 + (idx + 1) * num_J]

            # Concatenate hx_array with J_values
            current_step = np.concatenate([J_values,hx_array,hy_array,hz_array])
            
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
   
    def final_test(params,test_in,test_targ):
        params = jnp.asarray(params, dtype=jnp.float64)
        X = jnp.asarray(test_in, dtype=jnp.complex128)
        y = jnp.asarray(test_targ, dtype=jnp.complex128)
        batched_output_states = vcircuit(params, X)

        fidelities = jax.vmap(qml.math.fidelity)(batched_output_states, y)
        fidelities = jnp.clip(fidelities, 0.0, 1.0)

        return fidelities
 
    


    # Initial training to determine appropriate learning rate
    if opt_lr == None:
        s = time.time()
        init_loss, init_grads = jax.value_and_grad(cost_func)(params, input_states, target_states)
        e = time.time()
        dt = e - s
        # print(f"initial fidelity: {init_loss:.4f}, initial_gradients: {np.mean(np.abs(init_grads))}. Time: {dt:.2e}")
        # opt_lr,grad_norm = get_initial_learning_rate(init_grads)
       
        # opt_lr, grad_norm = get_initial_lr_hessian(
        #     params, 
        #     cost_func, 
        #     input_states, 
        #     target_states, 
        #     init_grads,
        #     shift=1e-3,
        #     min_lr=1e-3,
        #     max_lr=0.2
        # )
        opt_lr = get_initial_lr_per_param(init_grads)
        cost = init_loss
    else:
        s = time.time()
        init_loss, init_grads = jax.value_and_grad(cost_func)(params, input_states, target_states)
        e = time.time()
        dt = e - s
        opt_lr,grad_norm = get_initial_learning_rate(init_grads)
       
        cost = init_loss
        # print(f"initial fidelity: {init_loss:.4f}, init opt: {maybe_lr}. Time: {dt:.2e}")
        

    opt_descr = 'per param'
    # opt = optax.chain(
    #     optax.clip_by_global_norm(1.0),            # Clip gradients globally
    #     scale_by_param_lr(opt_lr),             # Apply per-parameter scaling
    #     optax.adam(learning_rate=1.0, b1=0.99, b2=0.999, eps=1e-7)  # Use a "neutral" global LR
    # )
    # opt = optax.adam(learning_rate=opt_lr, b1=0.99, b2=0.999, eps=1e-7)
    learning_rate_schedule = optax.constant_schedule(opt_lr)
    opt = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.inject_hyperparams(optax.adam)(learning_rate=opt_lr, b1=0.99, b2=0.999, eps=1e-7),
        )
    # print(learning_rate_schedule)
    """
    case #0
    # """
    # opt_descr = 'case 0'
    # learning_rate_schedule = optax.constant_schedule(opt_lr)
    # opt = optax.chain(
    #     optax.clip_by_global_norm(1.0),
    #     optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate_schedule),
    #     )
    """
    case #1
    """
    # opt_descr = 'case 1'
    # learning_rate_schedule = optax.constant_schedule(opt_lr)
    # opt = optax.chain(
    #     optax.clip_by_global_norm(1.0),
    #     optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate_schedule, b1=0.99, b2=0.999, eps=1e-7),
    #     )
    """
    case #2
    """
    # opt_descr = 'case 2'
    # warmup_steps = num_epochs//2
    # total_steps = num_epochs//2
    # schedule = optax.sgdr_schedule([
    #     {"init_value": opt_lr*0.85, "peak_value": opt_lr, "decay_steps": total_steps, "warmup_steps": warmup_steps,"end_value": opt_lr*0.5},
    #     {"init_value": opt_lr/5, "peak_value": opt_lr/2, "decay_steps": total_steps, "warmup_steps": warmup_steps,"end_value": opt_lr *1e-1},
    #      {"init_value": opt_lr*0.85, "peak_value": opt_lr, "decay_steps": total_steps, "warmup_steps": warmup_steps,"end_value": opt_lr*0.5},
    # ])


    # opt = optax.chain(
    #     optax.clip_by_global_norm(1.0),
    #     optax.inject_hyperparams(optax.adam)(learning_rate=schedule),
    # )


    """
    case #3
    """
    # opt_descr = 'case 3'
    # lr_min = opt_lr*0.1
    # t = 5
    
    # factor = (lr_min/opt_lr)**(1/t)
    # PATIENCE = 50
    # COOLDOWN = 0
    # FACTOR = 0.9
    # RTOL = 1e-4
    # ACCUMULATION_SIZE = 10
    # MIN_SCALE = 0.1
    # # Create the Adam optimizer
    # adam = optax.adam(learning_rate= opt_lr)

    # # Add the reduce_on_plateau transformation
    # reduce_on_plateau = optax.contrib.reduce_on_plateau(
    #     factor=FACTOR,
    #     patience=PATIENCE,
    #     rtol=RTOL,
    #     cooldown=COOLDOWN,
    #     accumulation_size=ACCUMULATION_SIZE,
    #     min_scale=MIN_SCALE,
    # )
    # opt = optax.chain(
    #     optax.clip_by_global_norm(1.0),
    #     adam, 
    #     reduce_on_plateau)
   
    """
    case #4
    """
    # opt_descr = 'case 4'
   
    
    # schedule = optax.warmup_cosine_decay_schedule(
    #     init_value=opt_lr*0.9,
    #     peak_value=opt_lr*1.2,
    #     warmup_steps=100,
    #     decay_steps=num_epochs,
    #     end_value=opt_lr*0.85,
    # )


    # opt = optax.chain(
    #     optax.clip_by_global_norm(1.0),
    #     optax.inject_hyperparams(optax.adam)(learning_rate=schedule, b1=0.99, b2=0.999, eps=1e-7),
    # )


    # Define the optimization update function
    @jit
    def update(params, opt_state, input_states, target_states, value):
        """Update all parameters including tau."""
        # params = jnp.asarray(params, dtype=jnp.float64)
        # input_states = jnp.asarray(input_states, dtype=jnp.complex128)
        # target_states = jnp.asarray(target_states, dtype=jnp.complex128)
        loss, grads = jax.value_and_grad(cost_func)(params, input_states, target_states)
        if not isinstance(opt_state[-1], optax.contrib.ReduceLROnPlateauState):
            updates, opt_state = opt.update(grads, opt_state, params)
        else:
            updates, opt_state = opt.update(grads, opt_state, params=params, value=value)
        new_params = optax.apply_updates(params, updates)
        # Ensure outputs are float64
        loss = jnp.asarray(loss, dtype=jnp.float64)
        grads = jnp.asarray(grads, dtype=jnp.float64)
        return new_params, opt_state, loss, grads

    opt_state = opt.init(params)
    print(f"per param lrs: \n{opt_lr}\n")
    # print(f"initial opt_state: {opt_state}")
    print("________________________________________________________________________________")
    print(f"Starting optimization for {gate_name}(epochs: {num_epochs}) with avg optimal lr ({np.mean(opt_lr)}), time_steps = {time_steps}, N_r = {N_reserv}, N_bath = {num_bath}...\n")
    print(f"Initial Loss: {init_loss:.4f}, initial_gradients: {np.mean(np.abs(init_grads))}. Time: {dt:.2e}")
    print(f"Depth: {circuit_depth}, Num Gates: {num_gates} ")
    print("Number of trainable parameters: ", len(params))


    costs = []
    param_per_epoch,grads_per_epoch = [],[]
   # print(f"Params: {params}")
    
    

    # Define the gradient function outside the loop
    #cost_and_grad = jax.value_and_grad(partial(cost_func, time_steps=time_steps, N_reserv=N_reserv, N_ctrl=N_ctrl))
    prev_cost, second_prev_cost = float('inf'), float('inf')  # Initialize with infinity
    threshold_counts = 0
    acceleration = 0.0
    rocs = []
    consecutive_improvement_count = 0
    cost_threshold = 1e-5
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
    while epoch < num_epochs or improvement:
        if opt_descr in ['case 3']:
            plateau_state = opt_state[-1]
            scale,best_val,plateau_count,cooldown_count,*_ = plateau_state

            current_avg = plateau_state.avg_value
        params, opt_state, cost, grad = update(params, opt_state, input_states, target_states,value=cost)
        if opt_descr in ['case 3']:
            plateau_state = opt_state[-1]
            plateau_scale = plateau_state.scale
            adjusted_lr = opt_lr * plateau_scale
            learning_rates.append(adjusted_lr)
            scales_per_epoch.append(plateau_scale)
            
            # Check if plateau should have triggered
            if (
                plateau_state.avg_value >= plateau_state.best_value * (1 - RTOL) + RTOL
                and plateau_state.plateau_count >= PATIENCE
            ):
                print(f"ReduceLROnPlateau *should* have triggered at Epoch {epoch + 1}!")
            # Verify if scale is reducing
            if plateau_state.scale < new_scale:
                print(f"Reduced scale at epoch {epoch}: \n - current_avg: {current_avg:.2e},\n - best: {plateau_state.best_value:.2e},\n - res: {plateau_state.best_value * (1 - RTOL) + RTOL:.2e}")
                # print(f"Reduced scale at epoch {epoch}: {plateau_state.scale:.5f}. \n - current_avg: {current_avg:.2e},\n - best: {plateau_state.best_value:.2e},\n - res: {plateau_state.best_value * (1 - RTOL) + RTOL:.2e}")
                scale_reduction_epochs.append(epoch + 1)
                new_scale = plateau_state.scale
        # elif 'learning_rate' in opt_state[1].hyperparams:
        #     plateau_scale = 1.0
        #     learning_rate = opt_state[1].hyperparams['learning_rate']
        #     learning_rates.append(learning_rate)
        else:
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
        if epoch == 0 or (epoch + 1) % 100 == 0:
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
         
            print(f'Epoch {epoch + 1} --- cost: {cost:.5f}, '
            # print(f'Epoch {epoch + 1} --- cost: {cost:.4f}, best={best_val:.4f}, avg: {current_avg:.4f}, lr={learning_rates[-1]:.4f} [{plateau_state.scale:.3f}], '
                #   f' count: {plateau_state.plateau_count} '
                #   f'has improved: {(1 - RTOL) * plateau_state.best_value}'
                # f'Var(grad): {var_grad:.1e}, '
                # f'GradNorm: {np.linalg.norm(grad):.1e}, '
                 f'Mean(grad): {mean_grad:.1e}, '
                f'[t: {epoch_time:.1f}s]'
                )
            # if optax.tree_utils.tree_get(opt_state, 'learning_rate'):
            #     print(f"opt_state: {optax.tree_utils.tree_get(opt_state, 'learning_rate')}")
            # else:
            #     print(f'opt_state: {opt_state}')
            print(f" opt_state: {opt_state}")
            # print(f"    --- Learning Rate: {learning_rate}")
        
            s = time.time()

        if cost < prev_cost:
            
            improvement = True
            consecutive_improvement_count += 1
            # current_cost_check = cost
            current_cost_check = cost_func(params, input_states,target_states)
            backup_cost_check = cost_func(backup_params, input_states,target_states)
            if np.abs(backup_cost_check-backup_cost) > 1e-6:
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
       
        # plateau_state = opt_state[-1]
        if max(np.abs(grad)) < 1e-10 or np.var(grad,ddof=1) < 1e-8 or epoch >= 2*num_epochs:
            print(f"max(np.abs(grad)) < 1e-10 or np.var(grad,ddof=1) < 1e-8 and plateau_state.scale <= MIN_SCALE. breaking at epoch {epoch}....")
            # plateau_state = opt_state[-1]
            # print(f"ReduceLROnPlateau hyps. avg: {plateau_state.avg_value:.2e}, best: {plateau_state.best_value:.2e}, new_scale: {plateau_state.scale:.5f}!")
           
            break
        epoch += 1  # Increment epoch count


    if backup_cost < cost and not epoch < num_epochs:
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


    print("\nAverage Final Fidelity: ", avg_fidelity)
    # print(f"X[0]: {input_states[0]}")
    
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
    N_ctrl = 3
   
   
    # trots = [16,18,20,22,24]
    trots =[1,4,8,10,12,14,16,18,20,22,24,26]

    res = [1]
    trots = [35]
    # trots = [2]

    
    




    num_epochs = 1500
    N_train = 20
    add=0
   
    
    # folder = f'./analog_results_trainable_global/trainsize_{N_train+add}_epoch{num_epochs}_cond2/'
    folder = f'./analog_results_trainable_global/trainsize_{N_train+add}_epoch{num_epochs}_per_param_opt/'
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

        if not gate_idx in [8,16]:
            continue
        # if gate_idx != 0:
        #     continue
       

        for time_steps in trots:

            
            
            
            for N_reserv in res:
                
                N =N_ctrl+N_reserv
                
                #folder = f'./param_initialization/Nc{N_ctrl}_Nr{N_reserv}_dt{time_steps}/fixed_params4/test7/'
                for num_bath,bath in zip(num_baths,baths):
                    params_key_seed = gate_idx*121 * N_reserv + 12345 * time_steps *N_reserv
                    params_key = jax.random.PRNGKey(params_key_seed)
                    dataset_seed = N_ctrl * gate_idx + gate_idx**2 + N_ctrl
                    dataset_key = jax.random.PRNGKey(dataset_seed)
                    main_params = jax.random.uniform(params_key, shape=(3 + (N_ctrl * N_reserv) * time_steps,), minval=-np.pi, maxval=np.pi)
                    # print(f"main_params: {main_params}")
                    params_key, params_subkey1, params_subkey2 = jax.random.split(params_key, 3)
                    
                    
                    time_step_params = jax.random.uniform(params_key, shape=(time_steps,), minval=0, maxval=np.pi)
                    init_params_dict = get_init_params(N_ctrl, N_reserv, time_steps,bath,num_bath,params_subkey1)
                    


                    # Combine the two parts
                    params = jnp.concatenate([time_step_params, main_params])
                    # params = jnp.asarray([0.708180844783782959,2.8630681037902832,0.51971900463104248,0.102997027337551117,0.5567360520362854,2.78287434577941895,1.57961368560791016,0.866679072380065918,2.72562146186828613,0.56838381290435791,2.61631464958190918,2.09357690811157227,0.705745697021484375,0.871583163738250732,2.9413750171661377,2.5572206974029541,2.25589108467102051,2.58423185348510742,1.24116122722625732,0.675378262996673584,0.259453773498535156,2.53845119476318359,0.298816800117492676,2.51644587516784668,1.16178691387176514,0.302116096019744873,2.62891721725463867,2.2046971321105957,0.181666463613510132,1.66521167755126953,2.10095763206481934,1.95381510257720947,3.03793001174926758,0.753958046436309814,1.9740145206451416,0.783146083354949951,2.53042101860046387,0.746370375156402588,1.573944091796875,2.19305944442749023,-0.740846395492553711,-2.14908385276794434,1.15825963020324707,2.84470033645629883,-0.328924059867858887,-2.83768844604492188,-2.00651001930236816,0.28257480263710022,-1.08436489105224609,-3.05580830574035645,2.97736454010009766,0.304987728595733643,-2.75177407264709473,-2.78570270538330078,2.48746943473815918,-0.341458261013031006,-0.112607285380363464,1.61129927635192871,-0.136841148138046265,1.4562525749206543,-1.40975785255432129,-0.945134997367858887,1.85663366317749023,2.19307374954223633,-2.54177999496459961,2.79172015190124512,-1.02528917789459229,-0.60877680778503418,1.29766404628753662,-3.01436758041381836,1.32913053035736084,-1.3869251012802124,1.69927799701690674,1.17338061332702637,-0.473877638578414917,0.376701444387435913,2.51183605194091797,-1.071022629737854,2.87483382225036621,-1.1927940845489502,0.132605060935020447,-1.08920586109161377,-0.648234248161315918,-0.0223977342247962952,1.29679763317108154,-1.91868412494659424,2.22426080703735352,3.03661894798278809,2.08379387855529785,0.718560516834259033,-3.10053229331970215,-2.64503049850463867,2.25072979927062988,0.159820646047592163,-1.74804234504699707,1.21237015724182129,0.27378547191619873,-0.649013161659240723,-2.77437114715576172,-1.71469390392303467,1.94135427474975586,2.75203847885131836,-0.521626830101013184,-1.28643155097961426,-1.64881479740142822,2.16837811470031738,1.83844888210296631,-1.03077304363250732,1.9857337474822998,-0.648720085620880127,-0.783919751644134521,1.98524951934814453,1.53545141220092773,2.59280824661254883,-2.53451418876647949,-2.15049290657043457,2.42705225944519043,1.40798842906951904,2.10079097747802734,-1.85509192943572998,-1.95991683006286621,-1.87856805324554443,-1.42525207996368408,-0.673454761505126953,2.00832796096801758,0.610223114490509033,-1.60458683967590332,-1.15768992900848389,-1.11240410804748535,2.19812273979187012,-2.39813089370727539,0.556749582290649414,1.8967738151550293,-2.61383056640625,-1.36014199256896973,2.61524200439453125,2.01110935211181641,2.09496569633483887,-1.81603610515594482,-2.25705409049987793,2.87667560577392578,2.22872662544250488,-3.13997626304626465,2.44711756706237793,2.15184450149536133,-1.68608808517456055,2.93016624450683594,-0.802432894706726074,2.14380383491516113,1.47287189960479736,-2.0552980899810791,2.41168117523193359,2.96615505218505859,1.45948731899261475,0.800140798091888428,-1.50944530963897705,-1.66607260704040527,-2.8509671688079834,-2.47937941551208496,3.12403154373168945,3.05443429946899414,-1.66225099563598633,1.30399894714355469])
                    # print(f"time_step_params: {time_step_params}")



                    run_test(params, num_epochs, N_reserv, N_ctrl, time_steps,N_train,folder,gate,gate.name,bath,num_bath,init_params_dict = init_params_dict,dataset_key = dataset_key)
