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


class Sim_QuantumReservoir:
    def __init__(self, params, N_ctrl, N_reserv, num_J, time_steps=1,bath=False,num_bath = 0,bath_factor = 1.0):
        self.bath = bath
        self.bath_factor = bath_factor
        self.num_bath = num_bath
        self.N_ctrl = N_ctrl
        self.N_reserv = N_reserv
        self.reserv_qubits = qml.wires.Wires(list(range(N_ctrl, N_reserv+N_ctrl)))
        self.ctrl_qubits = qml.wires.Wires(list(range(N_ctrl)))

        if bath:
            self.bath_qubits = qml.wires.Wires(list(range(N_reserv+N_ctrl, N_reserv+N_ctrl+num_bath)))
            self.network_wires = Wires([*self.ctrl_qubits,*self.reserv_qubits])
           # self.bath_interactions = params['bath']
            self.N = N_ctrl + N_reserv + num_bath
            self.dev = qml.device("default.qubit", wires = [*self.ctrl_qubits, *self.reserv_qubits,*self.bath_qubits]) 
            # Initialize random seed for reproducibility
            self.key = params['key']

            # Generate central coupling values for system-bath interactions
            gamma_central = bath_factor * (jax.random.uniform(self.key, (num_bath,)) - 0.5) * 2
            initial_sigma = 0.01 * np.abs(gamma_central)
            
            # Initialize initial couplings with variability
            self.initial_bath_couplings = numpy.random.normal(gamma_central, initial_sigma)
            #print(f"initial_couplings: {self.initial_couplings.shape},{self.initial_couplings}")
            

            # Initialize bath-bath interactions
            # Initialize symmetric bath-bath interactions
            bath_bath_interactions = jax.random.normal(self.key, shape=(num_bath, num_bath))
            self.bath_bath_interactions = (bath_bath_interactions + bath_bath_interactions.T) / 2
            #print(f"bb-int: {self.bath_bath_interactions.shape},{self.bath_bath_interactions}")

        else:
            self.N = N_ctrl + N_reserv
            self.initial_bath_couplings =  None
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
        '''Return the XY coupling between qubits i and j with a coefficient function.'''
        return ((qml.PauliZ(wires=i) @ qml.PauliZ(wires=j)) + (qml.PauliZ(wires=i) @ qml.PauliZ(wires=j)))
    
    def get_ZX_coupling(self, bath_qubit):
        '''Return the ZX coupling between bath qubit and each qubit in the system.'''
        operators = []
        for qubit in self.network_wires:
            operators.append(qml.PauliZ(wires=qubit) @ qml.PauliX(wires=bath_qubit))
        return sum(operators)
    
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
        ''' Construct the time-independent part of the Hamiltonian '''
        static_coefficients = []
        static_operators = []

        
        for qubit_a in range(len(self.reserv_qubits)):
            #print(f"qubit_a: {qubit_a}")
            for qubit_b in range(len(self.reserv_qubits)):
                #print(f"qubit_b: {qubit_b}")
                if qubit_a != qubit_b:
                    #print("K_coeff: ",self.k_coefficient)
                    interaction_coeff = self.k_coefficient[qubit_a, qubit_b]
                    #print("interaction_coeff: ", interaction_coeff)
                    static_coefficients.append(interaction_coeff)
                    new_operator = self.get_XY_coupling(self.reserv_qubits[qubit_a], self.reserv_qubits[qubit_b])
                    static_operators.append(new_operator)

        if self.bath:
            #for idx, (qubit, bath_qubit) in enumerate([(q, b) for q in self.network_wires for b in self.bath_qubits]):
            for idx,bath_qubit in enumerate(self.bath_qubits):
                coupling_strength = self.initial_bath_couplings[idx]
                static_coefficients.append(coupling_strength)
                new_operator = self.get_ZX_coupling(bath_qubit)
                static_operators.append(new_operator)
            
            # Add symmetric bath-bath interactions
            for i,bath_qubit_i in enumerate(self.bath_qubits):
                for j,bath_qubit_j in enumerate(self.bath_qubits):
                    if bath_qubit_i != bath_qubit_j and bath_qubit_i<bath_qubit_j:
                        static_coefficients.append(self.bath_bath_interactions[i, j])
                        new_operator = self.get_ZZ_coupling(bath_qubit_i, bath_qubit_j)
                        static_operators.append(new_operator)
        
        if self.N_reserv == 1 and not self.bath:
            total_H = H_dynamic
        
        else:
            H_static = qml.dot(static_coefficients, static_operators)
            
            total_H = H_dynamic + H_static
    
        return total_H
        

    def get_bath_hamiltonian(self, tau, coupling_params):
        bath_coefficients = []
        bath_operators = []

        def bath_coupling_func(coupling_strength, t):
            return coupling_strength * jax.numpy.sin(t)

        #for idx, (qubit, bath_qubit) in enumerate([(q, b) for q in self.network_wires for b in self.bath_qubits]):
        for idx,bath_qubit in enumerate(self.bath_qubits):
            coupling_strength = coupling_params[idx]
            bath_coefficients.append(lambda p, t, c=coupling_strength: bath_coupling_func(c, t))
            new_operator = self.get_ZX_coupling(bath_qubit)
            bath_operators.append(new_operator)
        
        # Add symmetric bath-bath interactions
        for i,bath_qubit_i in enumerate(self.bath_qubits):
            for j,bath_qubit_j in enumerate(self.bath_qubits):
                if bath_qubit_i != bath_qubit_j:
                    bath_coefficients.append(self.bath_bath_interactions[i, j])
                    new_operator = self.get_ZZ_coupling(bath_qubit_i, bath_qubit_j)
                    bath_operators.append(new_operator)
        H_bath = qml.dot(bath_coefficients, bath_operators)

        return H_bath
    
    def get_bath_hamiltonian_noperturb(self, tau):
        bath_coefficients = []
        bath_operators = []

        def bath_coupling_func(p, t):
            return p * np.sin(tau * t)  # Example time-dependent coupling

        for qubit in self.network_wires:
            for bath_qubit in self.bath_qubits:
                bath_coefficients.append(lambda p, t: 0.01 * bath_coupling_func(p, t))  # Directly scale by 0.01
                new_operator = qml.PauliZ(wires=qubit) @ qml.PauliX(wires=bath_qubit)
                bath_operators.append(new_operator)

        H_bath = qml.dot(bath_coefficients, bath_operators)

        return H_bath
  

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

def optimize_traingset(gate,N_ctrl, N_reserv,time_steps, params, init_params_dict, N_train,num_datasets, key):
    datasets = []
    print(f"Pre-processing a batch of {num_datasets} training sets for selection... ")
    all_A, all_b = [],[]
    for i in range(num_datasets):
        key, subkey = jax.random.split(key)  # Split the key for each dataset generation
        A,b = generate_dataset(gate, N_ctrl, N_train + 2000, subkey)  # Generate dataset with the subkey
        all_A.append(A)
        all_b.append(b)
    all_A = jnp.stack(all_A)
    all_b = jnp.stack(all_b)
    # Convert datasets list into two arrays for inputs and targets
    
    num_J = N_reserv *N_ctrl
    
    sim_qr = Sim_QuantumReservoir(init_params_dict, N_ctrl, N_reserv, num_J,time_steps,bath,num_bath)
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

    batched_collect_gradients = jax.vmap(collect_gradients, in_axes=(None, 0, 0))

    all_gradients = batched_collect_gradients(params, all_A[:, :N_train], all_b[:, :N_train])
    
    
    def calculate_gradient_stats(gradients):
        mean_grad = jnp.mean(gradients, axis=0)
        mean_grad_squared = jnp.mean(gradients**2, axis=0)
        var_grad = mean_grad_squared - mean_grad**2
        return mean_grad, var_grad

    best_dataset_idx = None
    max_var_grad_sum = -jnp.inf
    worst_dataset_idx = None
    min_var_grad_sum = jnp.inf

    # Calculate and print gradient statistics for each dataset
    for i in range(num_datasets):
        mean_grad, var_grad = calculate_gradient_stats(all_gradients[i])
        var_grad_sum = var_grad.sum()
        mean_grad_sum = mean_grad.sum()
        min_grad = min(var_grad)
        
        print(f"(A{i+1}, b{i+1}):")
        print(f"var_grad: {var_grad}")
        print(f"Variance Gradient sum: {var_grad_sum}, mean_grad_sum: {mean_grad_sum}, minimum grad: {min_grad}\n")
        if var_grad_sum > max_var_grad_sum:
            second_best_idx = best_dataset_idx

            max_var_grad_sum = var_grad_sum
            
            best_dataset_idx = i
        if var_grad_sum < min_var_grad_sum:
            min_var_grad_sum = var_grad_sum
            worst_dataset_idx = i

    print(f"Selected Dataset: A{best_dataset_idx + 1}, b{best_dataset_idx + 1} with Variance Sum: {max_var_grad_sum}")
    
    best_A = all_A[best_dataset_idx]
    best_b = all_b[best_dataset_idx]
    worst_A = all_A[second_best_idx]
    worst_b = all_b[second_best_idx]
    best_gradients = all_gradients[best_dataset_idx]
    initial_lr = compute_initial_learning_rate(best_gradients)
    print(f"Initial Learning Rate: {initial_lr}")
    assert best_dataset_idx != second_best_idx
    return best_A, best_b,worst_A,worst_b,initial_lr

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

def run_test(params, num_epochs, N_reserv, N_ctrl, time_steps,N_train,folder,gate,gate_name,bath,num_bath,init_params_dict, dataset_key,bath_factor):
    float32=''
    opt_lr = None
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
    
    k = 1
   
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
    print(f"training state #1: {input_states[0]}")


    test_dataset_key = jax.random.split(dataset_key)[1]
    test_in, test_targ = generate_dataset(gate, N_ctrl,training_size= 2000, key=test_dataset_key, new_set=False)
    

    second_A, second_b = generate_dataset(gate, N_ctrl,training_size= 500, key= dataset_key, new_set=True) 
    assert not any(np.allclose(x1, x2) for x1 in input_states for x2 in second_A), "Duplicate states found!"

    
   
    parameterized_ham = sim_qr.get_total_hamiltonian_components()
    print("H: ",parameterized_ham)
    
    
    print("Number of trainable parameters: ", len(params))


    ctrl_wires = sim_qr.get_ctrl_wires()
    reserv_wires = sim_qr.get_reserv_wires()
    qnode_dev = sim_qr.get_dev()
    all_wires = sim_qr.get_all_wires()

   

    @qml.qnode(qnode_dev, interface="jax")
    def circuit(params,state_input):
        
        taus = params[:time_steps]
        #print(f"taus: {taus}")
        qml.StatePrep(state_input, wires=[*ctrl_wires])
        if sim_qr.bath:
            for bath_qubit in sim_qr.bath_qubits:
                qml.Hadamard(bath_qubit)
        #print(f"coupling_params: {coupling_params}")
        for idx, tau in enumerate(taus):
           
            hx_array = np.array([params[time_steps]])  # Convert hx to a 1D array
            hy_array = np.array([params[time_steps + 1]])  # Convert hy to a 1D array
            hz_array = np.array([params[time_steps + 2]])  # Convert hz to a 1D array
            J_values = params[time_steps + 3 + idx * num_J : time_steps + 3 + (idx + 1) * num_J]
 
            # Concatenate hx_array with J_values and coupling_params
            current_step = np.concatenate([J_values, hx_array, hy_array, hz_array])
            
            
           
            #print(f"H at time step {idx}: {total_H}")
            qml.evolve(parameterized_ham)(current_step, t=tau)
            
            
        return qml.density_matrix(wires=[*ctrl_wires])
    
    
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
    @jit
    def update(params, opt_state, input_states, target_states):
        params = jnp.asarray(params, dtype=jnp.float64)
        input_states = jnp.asarray(input_states, dtype=jnp.complex128)
        target_states = jnp.asarray(target_states, dtype=jnp.complex128)
        loss, grads = jax.value_and_grad(cost_func)(params, input_states, target_states)
        updates, opt_state = opt.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        # Ensure outputs are float64
        loss = jnp.asarray(loss, dtype=jnp.float64)
        grads = jnp.asarray(grads, dtype=jnp.float64)
        return new_params, opt_state, loss, grads


    
    if opt_lr == None:
        s = time.time()
        init_loss, init_grads = jax.value_and_grad(cost_func)(params, input_states, target_states)
        e = time.time()
        dt = e - s
        print(f"initial fidelity: {init_loss}, initial_gradients: {init_grads}. Time: {dt}")
        opt_lr,grad_norm = get_initial_learning_rate(init_grads)
        print(f"Adjusted initial learning rate: {opt_lr}. Grad_norm: {1/grad_norm},Grad_norm: {grad_norm}")
        
    
    


    print("________________________________________________________________________________")
    print(f"Starting optimization for {gate_name} with optimal lr {opt_lr} time_steps = {time_steps}, N_r = {N_reserv}, N_bath = {num_bath}...\n")

    opt = optax.chain(
        optax.clip_by_global_norm(1.0),  # Clip gradients to prevent explosions
        optax.adam(learning_rate=opt_lr, b1=0.99, b2=0.999, eps=1e-7)  # Slightly more aggressive Adam
    )
   
    
    print("Number of trainable parameters: ", len(params))


    

   

    

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
    consecutive_threshold_limit = 4  # Number of consecutive epochs below threshold without improvement needed to stop
    backup_params = None
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
    cond1,cond2,a_threshold = -float('inf'), -float('inf'), -float('inf') 
    add_more=False
    num_states_to_replace = N_train // 4
    while epoch < num_epochs or improvement:
        params, opt_state, cost,grad = update(params, opt_state, input_states, target_states)
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
                print(f"    - setting cond1: initial mean(grad) {initial_meangrad:2e}, threshold: {cond1:2e}")
                cond2 = initial_vargrad * 1e-2
                print(f"    - setting cond2: initial var(grad) {initial_vargrad:2e}, threshold: {cond2:2e}")
            
            acceleration = get_rate_of_improvement(cost,prev_cost,second_prev_cost)
            if epoch >= 25 and not a_condition_set and acceleration < 0.0:
                average_roc = np.mean(np.array(rocs[10:]))
                a_marked = np.abs(average_roc)
                a_threshold = max(a_marked * 1e-3, 1e-7)
                # a_threshold = a_marked*1e-3 if a_marked*1e-3 > 9e-7 else a_marked*1e-2
                
                print(f"acceration: {a_marked:.2e}, marked: {a_threshold:.2e}")
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
        
        if epoch == 0 or (epoch + 1) % 100 == 0:
            var_grad = np.var(grad,ddof=1)
            mean_grad = np.mean(jnp.abs(grad))
            e = time.time()
            epoch_time = e - s
            
            print(f'Epoch {epoch + 1} --- cost: {cost:.5f}, '
                  f'a: {acceleration:.2e} '
                f'Var(grad): {var_grad:.1e}, '
                f'Mean(grad): {mean_grad:.1e}, '
                f'[t: {epoch_time:.1f}s]')
            s = time.time()
        
        if cost < prev_cost:
            
            improvement = True
            consecutive_improvement_count += 1
            current_cost_check = cost
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


        
        # Apply tau parameter constraint (must be > 0.0)
        for i in range(time_steps):
            if params[i] < 0:
                params = params.at[i].set(np.abs(params[i]))
                

        second_prev_cost = prev_cost  # Move previous cost back one step
        prev_cost = cost 
        if np.abs(max(grad)) < 1e-14 or np.var(grad,ddof=1) < 1e-10:
            print(f"max(grad)<1e-14. breaking....")
            break
        epoch += 1 
        cond3 = np.var(grad,ddof=1) < cond2
        cond4 = np.abs(acceleration) < a_threshold
        cond5 = not improvement
        cond6 = np.mean(np.abs(grad)) < cond1 
        # if epoch > 2 and add_more:
        if (epoch >= 100 and cond6
            and cond3 and add_more and epoch <= 0.9 * num_epochs and (cond5 and cond4)):
            print(f"Improvement: {improvement}")
            if cond3:
                print(f"cond3 True -> np.var(grad,ddof=1) < {cond2:.2e}")
            if cond4:
                print(f"cond4 True -> np.abs(acceleration) < {a_threshold:.2e}")
            if cond6:
                print(f"cond6 True -> np.mean(np.abs(grad)) < {cond1:.2e}")
            grad_circuit = grad
            stored_epoch = epoch
            mean_grad = jnp.mean(np.abs(grad_circuit))
            var_grad = jnp.var(grad_circuit,ddof=1)
           

           
            # Concatenate the new states (add_a, add_b) with the existing input_states and target_states
            # Add new states (instead of replacing existing states)
            # print(f"***Adding {num_states_to_replace} new states at epoch {epoch}***")
            costs_per_state,gradients_per_state = collect_gradients(params, input_states=input_states, target_states=target_states)
            new_costs_per_state,gradients_new_states = collect_gradients(params, input_states=second_A,target_states=second_b)
            # costs_per_state,gradients_per_state = collect_gradients(params, input_states=input_states, target_states=target_states)
            
            print(f"Epoch {epoch}:  cost: {cost:.5f}")
            print(f"***flat landscape*** roc: {acceleration:.2e} mean(grad): {mean_grad:.2e}, Var(Grad): {var_grad:.2e}***")
            # print(f"og shape: {gradients_per_state.shape}, costs_per_state.shape: {costs_per_state.shape}")
            # print(f"new shape: {gradients_new_states.shape}")
            meangrad_unbiased, vargrad_unbiased, grad_norm_unbiased = calculate_unbiased_stats(gradients_per_state)
            
            # Calculate stats for all training states
            meangrad_new, vargrad_new, grad_new = calculate_unbiased_stats(gradients_new_states)
            
            # meangrad_norm, vargrad_norm, grad_norm_norm = calculate_gradient_stats_per_state(normalized_gradients_per_state)
            sorted_vargrad_indices = np.argsort(vargrad_new)[::-1]  # Sort descending by variance
            sorted_meangrad_indices = np.argsort(meangrad_new)[::-1]  # Sort descending by mean gradient
            
            
            # total_length = len(sorted_vargrad_indices)  
            # even_indices = np.linspace(0, total_length - 1, 1000, dtype=int)

            # sampled_vargrad_indices = sorted_vargrad_indices[even_indices]
            
            # sampled_meangrad_indices = sorted_meangrad_indices[even_indices]
            
            

            max_var_indices_new_states = sorted_vargrad_indices[:num_states_to_replace]
            max_meangrad_indices_new_states = sorted_meangrad_indices[:num_states_to_replace]


            # print(f"max_var_indices_new_states: {max_var_indices_new_states}")
            # print(f"max_meangrad_indices_new_states: {max_meangrad_indices_new_states}")
            # Select the states from `second_A` and `second_B` based on `max_var_indices_new_states`
            add_a = np.asarray(second_A[max_var_indices_new_states])
            add_b = np.asarray(second_b[max_var_indices_new_states])


            print(f"\nNew replacement states: ")
            print(f"    Indices selected on max var: {max_var_indices_new_states}")
            print(f"    - Costs: ({new_costs_per_state.min():.1e},{new_costs_per_state.max():.1e}): {[f's{i}: {new_costs_per_state[i]:.1e}' for i in max_var_indices_new_states]}")
            print(f"    - Var(Grad) ({vargrad_new.min():.1e},{vargrad_new.max():.1e}): {[f's{i}: {vargrad_new[i]:.1e}' for i in max_var_indices_new_states]}")
            print(f"    - Mean(Grad) ({meangrad_new.min():.1e},{meangrad_new.max():.1e}): {[f's{i}: {meangrad_new[i]:.1e}' for i in max_var_indices_new_states]}")
            
            # print(f"    Indices selected on max mean: {max_meangrad_indices_new_states}")
            # print(f"    - Mean(Grad) ({meangrad_new.min():.1e},{meangrad_new.max():.1e}): {[f's{i}: {meangrad_new[i]:.1e}' for i in max_meangrad_indices_new_states]}")
            # normalized_grads_variance_new = jnp.var(normalized_gradients_per_state, axis=tuple(range(1, normalized_gradients_per_state.ndim)))
        
            
            
    
            sorted_gradnormns = np.argsort(grad_norm_unbiased)[::-1]  # Sort descending by mean gradient
            # print(f"sorted_gradnormns: {sorted_gradnormns}")
            
            for idx in sorted_gradnormns:
                
                # print(f"s{idx} [cost {costs_per_state[idx]}] - v: ({vargrad_unbiased[idx]:.1e}), n: ({grad_norm_unbiased[idx]:.1e}), g: {meangrad_unbiased[idx]:.1e}")
                training_state_metrics[int(idx)] = {
                    'cost': costs_per_state[idx],
                    'Var(Grad)': vargrad_unbiased[idx],
                    'Mean(Grad)': meangrad_unbiased[idx],
                    'Norm(Grad)': grad_norm_unbiased[idx],  # This is now calculated per state
                    
                }
                # Single-line output per state
                
                # print(f"{idx}: Var(Grad): ({vargrad[idx]:.1e},{vargrad_norm[idx]:.1e}) , Mean(Grad): ({meangrad[idx]:.1e},{meangrad_norm[idx]:.1e}), Var(NormGrad): {normalized_grads_variance[idx]:.1e}")




            input_states = np.concatenate([input_states, add_a], axis=0)
            target_states = np.concatenate([target_states, add_b], axis=0)
            print(f"New number of training states: {len(input_states)}")
            add_more = False
        

    if backup_cost < cost and not epoch < num_epochs and backup_epoch > stored_epoch:
        print(f"backup cost (epoch: {backup_epoch}) is better with: {backup_cost:.2e} <  {cost:.2e}: {backup_cost < cost}")
        params = backup_params
    
    full_e = time.time()

    epoch_time = full_e - full_s
    print(f"Time optimizing: {epoch_time}")
    testing_results = final_test(params,test_in, test_targ)
    avg_fidelity = jnp.mean(testing_results)
    infidelities = 1.00000000000000-testing_results
    avg_infidelity = np.mean(infidelities)
    
    print(f"\nAverage Final Fidelity: {avg_fidelity:.5f}")
    
    data = {'Gate':base64.b64encode(pickle.dumps(gate)).decode('utf-8'),
                'epochs': num_epochs,
                'rocs':rocs,
                'trotter_step': time_steps,
                'grads_per_epoch':grads_per_epoch,
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
                'sim_qr.initial_couplings':sim_qr.initial_bath_couplings,
                'bb-int':sim_qr.bath_bath_interactions,
                'bath': bath,
                'num_bath':num_bath,
                'training_state_metrics':training_state_metrics,
                'stored_epoch': stored_epoch,
        
    
            }
    print(f"Saving results to {filename}")
    df = pd.DataFrame([data])
    while os.path.exists(filename):
        name, ext = filename.rsplit('.', 1)
        filename = f"{name}_.{ext}"

    with open(filename, 'wb') as f:
        pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)




if __name__ == '__main__':


    

    


    
    
    
    #folder = './results_jax_baths_global_h/'
    # Example usage

    
    # run below 
    N_ctrl = 2
    
    # trots = [1,2,3,4,5]
    res = [1]

    trots = [1,4,6,8,10,12,14]

    
    trots = [6]
    #res = [N_reserv]
    
    num_epochs = 1200
    N_train = 20
    
    # base_folder = f'./analog_results_trainable_global/noise_opt_cost/'
    bath_factor = 0.1
    base_folder = f'./analog_results_trainable_global/trainsize_{N_train}_epoch{num_epochs}_bath_factor_{bath_factor}/'
    #folder = f'./analog_results_trainable_global/trainsize_{N_train}_optimize_trainset/'

    gates_random = []
    
    baths = [True]
    num_baths = [2]
    for i in range(10):
        U = random_unitary(2**N_ctrl, i).to_matrix()
        #pprint(Matrix(np.array(U)))
        g = partial(qml.QubitUnitary, U=U)
        g.num_wires = N_ctrl
        g.name = f'U{N_ctrl}_'+str(i)
        gates_random.append(g)

    
    #gates_known = [qml.CNOT,qml.CY]
    #for g in gates_known:
    #    g.name = g(wires=list(range(g.num_wires))).name

    
    for gate_idx,gate in enumerate(gates_random):

        if not gate_idx in [0]:
            continue
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
                    init_params_dict = get_init_params(N_ctrl, N_reserv, time_steps,bath,num_bath,params_subkey1)
                    


                    # Combine the two parts
                    params = jnp.concatenate([time_step_params, main_params])
                    # params = jnp.asarray([0.4033546149730682, 1.4487122297286987, 2.3020467758178711, 2.9035964012145996, 0.9584765434265137, 1.7428307533264160, -1.3020169734954834, -0.8775904774665833, 2.4736261367797852, -0.4999605417251587, -0.8375297188758850, 1.7014273405075073, -0.8763229846954346, -3.1250307559967041, 1.1915868520736694, -0.4640290737152100, -1.0656110048294067, -2.6777451038360596, -2.7820897102355957, -2.3751690387725830, 0.1393062919378281])
                    print(f"time_step_params: {time_step_params}")



                    run_test(params, num_epochs, N_reserv, N_ctrl, time_steps,N_train,folder,gate,gate.name,bath,num_bath,init_params_dict = init_params_dict,dataset_key = dataset_key,bath_factor=bath_factor)

                        # run_test(params, init_params_dict,num_epochs, N_reserv, N_ctrl, time_steps,N_train,folder,gate,gate.name,bath,num_bath,random_key = params_subkey2,bath_factor=bath_factor)
