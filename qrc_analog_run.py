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
from qutip import *
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
from pennylane.typing import TensorLike
from pennylane.ops import functions
from pennylane.ops import Evolution
from parametrized_hamiltonian import ParametrizedHamiltonian
from parametrized_ham_pytree import ParametrizedHamiltonianPytree
from hard_ham import HardwareHamiltonian
from evolution2 import Evolution
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
    qml.QubitStateVector(input_state, wires=[*qubits])
    gate(wires=qubits)
    return qml.density_matrix(wires=[*qubits])

def generate_dataset(gate, n_qubits, training_size, key):
    '''
    Generate the dataset of input and output states according to the gate provided.
    Uses a seed for reproducibility.
    '''
    

    # Generate random state vectors
    X = []
    
    # Split the key into subkeys for the full training size
    keys = jax.random.split(key, num=training_size)
    
    # Loop through the subkeys and generate the dataset
    for i, subkey in enumerate(keys):
        subkey = jax.random.fold_in(subkey, i)  # Fold in the index to guarantee uniqueness
        seed_value = int(jax.random.randint(subkey, (1,), 0, 2**32 - 1)[0])  # Get a scalar seed
        
        # Use the seed to generate the random state vector
        state_vec = random_statevector(2**n_qubits, seed=seed_value).data
        X.append(np.asarray(state_vec, dtype=jnp.complex128))
    
    
    X = np.stack(X)
    qubits = Wires(list(range(n_qubits)))
    dev_data = qml.device('default.qubit', wires=qubits)
    circuit = qml.QNode(quantum_fun, device=dev_data, interface='jax')
    
    # Execute the circuit for each input state
    # Execute the circuit for each input state and ensure Hermiticity
    # y = []
    # for i in range(training_size):
    #     target_state = np.array(circuit(gate, X[i], qubits), dtype=jnp.complex128)
        
    #     # Ensure the target state is Hermitian: A -> (A + A†) / 2
    #     target_state_hermitian = qml.Hermitian(target_state,wires=qubits)
    #     # print(type(target_state),type(target_state_hermitian))
        
    #     y.append(target_state_hermitian)
        

    y = [np.array(circuit(gate, X[i], qubits), dtype=jnp.complex128) for i in range(training_size)]
    y = np.stack(y)
    return X, y
def get_init_params(N_ctrl, N_reserv, time_steps, bath, num_bath, key):
    N = N_reserv + N_ctrl

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
    

def hyperparameter_trainingset_optimization_batch(gate, num_epochs, N_reserv, N_ctrl, N_train, time_steps, folder, bath, num_bath, init_params_dict, sim_qr, params, lr, num_datasets,key):
    

    datasets = []
    for i in range(num_datasets):
        key, subkey = jax.random.split(key)  # Split the key for each dataset generation
        dataset = generate_dataset(gate, N_ctrl, N_train + 1990, subkey)  # Generate dataset with the subkey
        datasets.append(dataset)

    # Convert datasets list into two arrays for inputs and targets
    all_a, all_b = zip(*datasets)
    all_a = jnp.array(all_a)
    all_b = jnp.array(all_b)
    #print("all_a[0]: ",all_a[0])
    # Define a vmap version of the test function over the first axis (different datasets)
    vrun_hyperparam_test = jax.vmap(partial(run_hyperparam_trainingset_test,
                                            lr=lr,
                                            num_epochs=num_epochs,
                                            N_reserv=N_reserv,
                                            N_ctrl=N_ctrl,
                                            time_steps=time_steps,
                                            folder=folder,
                                            batch_size=N_train,
                                            gate=gate,
                                            bath=bath,
                                            num_bath=num_bath,
                                            init_params_dict=init_params_dict,
                                            sim_qr=sim_qr,
                                            params=params), in_axes=(0, 0))
    
    # Run the tests in parallel over all datasets
    performances = vrun_hyperparam_test(all_a, all_b)
    #print("performances: ",performances)
    # Analyze the results to find the best dataset
    best_performance_index = jnp.argmax(performances)
    #print("idx: ",best_performance_index)
    #print("performances[idx] ",performances[best_performance_index])
    best_dataset = datasets[best_performance_index]
    opt_a,opt_b = best_dataset
    return opt_a,opt_b, best_performance_index

def run_hyperparam_trainingset_test(a,b,lr,num_epochs, N_reserv, N_ctrl, time_steps, folder, batch_size, gate, bath,num_bath,init_params_dict,sim_qr,params):
    opt = optax.adam(learning_rate=lr)
    input_states, target_states = np.asarray(a[:batch_size]), np.asarray(b[:batch_size])
    test_in, test_targ = a[batch_size:], b[batch_size:]
    num_J = N_ctrl*N_reserv
    #key = jax.random.PRNGKey(0)
    
     # get hamiltonian circuit variables
    parameterized_ham = sim_qr.get_total_hamiltonian_components()
    #print("\nParam ham: ", parameterized_ham)
    ctrl_wires = sim_qr.get_ctrl_wires()
    reserv_wires = sim_qr.get_reserv_wires()

    qnode_dev = sim_qr.get_dev()
    
    costs = []
    
    opt_state = opt.init(params)
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
    vcircuit = jax.vmap(circuit, in_axes=(None, 0))
    def batched_cost_helper(params, input_states, target_states):
        # Process the batch of states
        batched_output_states = vcircuit(params, input_states)
        
        # Compute fidelity for each pair in the batch and then average
        fidelities = jax.vmap(qml.math.fidelity)(batched_output_states, target_states)
        average_fidelity = np.sum(fidelities)/len(fidelities)
        
        return 1 - average_fidelity  # Minimizing infidelity

    @partial(jit, static_argnums=(1, 2, 3))
    def cost_func(params, time_steps, N_reserv, N_ctrl, input_states, target_states):
        return batched_cost_helper(params, input_states, target_states)
   

    def final_test(params, time_steps, N_reserv, N_ctrl, test_in, test_targ):
        num_J = N_ctrl*N_reserv
    
  
        
        fidelitity_tot = 0.
        
        count = 0 
        
        fidelities = []
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
        jit_circuit = jax.jit(circuit)
        for state_input, state_target in zip(test_in, test_targ):
    
            
            
            
            count+=1
            
            rho_traced = jit_circuit(params,state_input)
            
    
    
            
            fidelity = qml.math.fidelity(rho_traced,state_target)
            fidelities.append(fidelity)
            fidelitity_tot += fidelity
       
    
    
        return fidelities
    for epoch in range(500):
        
        cost, grad_circuit = jax.value_and_grad(cost_func)(params, time_steps, N_reserv, N_ctrl, input_states, target_states)
        
        # Update parameters using the optimizer
        updates, opt_state = opt.update(grad_circuit, opt_state)
        params = optax.apply_updates(params, updates)
        params = params.at[:time_steps].set(jax.numpy.where(params[:time_steps] < 0, jax.numpy.abs(params[:time_steps]), params[:time_steps]))

        
        
        
    #testing_results = final_test(params, time_steps, N_reserv, N_ctrl, test_in, test_targ)
    
    #total_tests = len(testing_results)
    #avg_fidelity = np.sum(np.asarray(testing_results))/total_tests
    print(f"Resulting fidelity for learning rate {lr}: {1-cost}")
    return 1-cost

def array(a,dtype):
    return np.asarray(a, dtype=np.float32)
def Array(a,dtype):
    return np.asarray(a, dtype=np.float32)
@jit
def get_initial_learning_rate(grads, scale_factor=0.01, min_lr=1e-4, max_lr=0.5):
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

def compute_initial_learning_rate(gradients, scale_factor=0.1, min_lr=1e-3, max_lr = 0.1):
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
    
    def calculate_gradient_stats(gradients):
        mean_grad = jnp.mean(gradients, axis=0)
        mean_grad_squared = jnp.mean(gradients ** 2, axis=0)
        var_grad = mean_grad_squared - mean_grad ** 2
        # print(f"var_grad.shape {var_grad.shape}")
        grad_norm = jnp.linalg.norm(mean_grad)

        return mean_grad, var_grad, grad_norm
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
def run_test(params, num_epochs, N_reserv, N_ctrl, time_steps,N_train,folder,gate,gate_name,bath,num_bath,init_params_dict, random_key):
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
    #print(list(Path(folder_gate).glob('*')))
    # if time_steps == 16:
    #     k = 2
    # else: 
    #     k = 1
    if len(files_in_folder) >= k:
        print('Already Done. Skipping: '+folder_gate)
        print('\n')
        return

    
    # get PQC
    sim_qr = Sim_QuantumReservoir(init_params_dict, N_ctrl, N_reserv, N_reserv * N_ctrl,time_steps,bath,num_bath)
    
    
    # Get optimal hyperparameter (learning rate)
    
    init_params = params
    # print(f"init_params: {init_params}")

    # opt_a,opt_b,worst_a,worst_b,opt_lr = optimize_traingset(gate,N_ctrl, N_reserv,time_steps, params, init_params_dict, N_train,10,key=random_key)
    # preopt_results, opt_a, opt_b,second_A,second_b,finetuning_A,finetuning_b,opt_lr, selected_indices = optimize_traingset(gate,N_ctrl, N_reserv,time_steps, params,init_params_dict, N_train,10,key=random_key)
    filename = os.path.join(folder_gate, f'data_run_{len(files_in_folder)}.pickle')
    # best_dataset_idx = selected_indices[0]
    # filename = os.path.join(folder_gate, f'A{best_dataset_idx}_to_A{selected_indices[1]}.pickle')
    # filename = os.path.join(folder_gate, f'A{best_dataset_idx}.pickle')
    
    opt_a,opt_b = generate_dataset(gate, N_ctrl, N_train + 2000, key= random_key) 
    # second_A,second_b = np.asarray(second_A[:200]), np.asarray(second_b[:200])

    # init_meangrad = preopt_results[f'dataset_{best_dataset_idx}']['Mean(Mean Grad)']
    # init_vargrad = preopt_results[f'dataset_{best_dataset_idx}']['Mean(Var Grad)']
    # cond1 = init_meangrad*1e-1
    # print(f"init_meangrad: {init_meangrad:2e}, threshold: {cond1:2e}")
    # cond2 = init_vargrad*1e-2


    input_states, target_states = np.asarray(opt_a[:N_train]), np.asarray(opt_b[:N_train])

    test_in, test_targ = opt_a[N_train:], opt_b[N_train:]
    

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
    @jit
    def cost_per_state(params, input_state, target_state):
        output_state = jit_circuit(params, input_state)
        fidelity = qml.math.fidelity(output_state, target_state)
        return 1 - fidelity  # Minimizing infidelity

    
    def collect_gradients(params, input_states, target_states):
        grad_fn = jax.grad(cost_per_state, argnums=0)
        gradients = jax.vmap(grad_fn, in_axes=(None, 0, 0))(params, input_states, target_states)
        return gradients
    

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
        print(f"initial fidelity: {init_loss}, initial_gradients: {np.mean(np.abs(init_grads))}. Time: {dt}")
        opt_lr,grad_norm = get_initial_learning_rate(init_grads)
        print(f"Adjusted initial learning rate: {opt_lr}. Grad_norm: {1/grad_norm},Grad_norm: {grad_norm}")
        """
        #opt_lr = 0.01
        """

    


    print("________________________________________________________________________________")
    print(f"Starting optimization for {gate_name}(epochs: {num_epochs}) with optimal lr {opt_lr} time_steps = {time_steps}, N_r = {N_reserv}, N_bath = {num_bath}...\n")

    #opt = optax.novograd(learning_rate=opt_lr)
    opt = optax.adam(learning_rate=opt_lr)
    
    # opt = optax.chain(
    #     optax.clip_by_global_norm(1.0),  # Clip gradients to prevent explosions
    #     optax.adam(learning_rate=opt_lr, b1=0.9, b2=0.99, eps=1e-8)  # Slightly more aggressive Adam
    # )


    # Define the optimization update function
    @jit
    def update(params, opt_state, input_states, target_states):
        """Update all parameters including tau."""
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

    print("Number of trainable parameters: ", len(params))


    costs = []
    param_per_epoch,grads_per_epoch = [],[]
   # print(f"Params: {params}")
    opt_state = opt.init(params)

    # Define the gradient function outside the loop
    #cost_and_grad = jax.value_and_grad(partial(cost_func, time_steps=time_steps, N_reserv=N_reserv, N_ctrl=N_ctrl))
    prev_cost, second_prev_cost = float('inf'), float('inf')  # Initialize with infinity
    threshold_counts = 0
    acceleration = 0.0
    rocs = []
    consecutive_improvement_count = 0
    cost_threshold = 1e-5
    consecutive_threshold_limit = 4  # Number of consecutive epochs below threshold without improvement needed to stop
    backup_params = None
    improvement = True
    backup_cost,min_cost = float('inf'),float('inf')   
    freeze_tau = False
    epoch = 0
    s = time.time()
    full_s =s
    training_state_metrics = {}
    add_more = True
    a_condition_set = False
    a_threshold =  0.0
    stored_epoch = None
    threshold_cond1, threshold_cond2 = [],[]
    false_improvement = False
    backup_epoch=0
    while epoch < num_epochs or improvement:

        params, opt_state, cost, grad = update(params, opt_state, input_states, target_states)
        
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
        max_abs_grad = jnp.max(jnp.abs(grad))
        if epoch == 0 or (epoch + 1) % 100 == 0:
            var_grad = np.var(grad,ddof=1)
            mean_grad = np.mean(jnp.abs(grad))
            e = time.time()
            epoch_time = e - s
            normalized_var_grad = var_grad /  np.mean(grad**2) 
            #print(f"step {epoch+1}, cost {cost}, time: {epoch_time}s")
            # print(f"step {epoch+1}, cost {cost:4e}. Max gradient:  {max(grad):3e}, var(grad): {np.var(grad):3e} [time: {epoch_time:3e}s]")
            print(f'Epoch {epoch + 1} cost: {cost:.5f}, '
                  f'a: {acceleration:.2e}'
                f'Var(grad): {var_grad:.1e}, '
                f'Mean(grad): {mean_grad:.1e}, '
                f'[t: {epoch_time:.1f}s]')
        
            s = time.time()

        if cost < prev_cost:
            
            improvement = True
            consecutive_improvement_count += 1
            current_cost_check = cost_func(params, input_states, target_states)
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
        # cond1=cond2 = 1e-3
        
        # if (epoch >= 50 and np.mean(np.abs(grad)) < cond1 
        #     and np.var(grad,ddof=1) < cond2 and add_more and epoch <= 0.9 * num_epochs and (not improvement or np.abs(acceleration) < a_threshold)):
        
        #     grad_circuit = grad
        #     stored_epoch = epoch
        #     mean_grad = jnp.mean(np.abs(grad_circuit))
        #     var_grad = jnp.var(grad_circuit,ddof=1)
        #     # Normalized gradient variance
        #     normalized_var_grad_abs = var_grad / (mean_grad ** 2) if mean_grad != 0 else float('inf')
        #     normalized_var_grad = var_grad / (jnp.mean(grad_circuit) ** 2) if jnp.mean(grad_circuit) != 0 else float('inf')

        #     # print(f"params: {type(params)}, {params.dtype}")
        #     # print(f"params: {params}")
            
        #     if N_ctrl < 4:
        #         gradients_per_state = collect_gradients(params, input_states=input_states, target_states=target_states)
        #         gradients_new_states = collect_gradients(params, input_states=second_A,target_states=second_b)
        #         normalized_gradients_per_state = normalize_gradients(gradients_per_state)
        #         # Calculate unbiased stats for comparison
        #         meangrad_unbiased, vargrad_unbiased, grad_norm_unbiased = calculate_unbiased_stats(gradients_per_state)
        #         meangrad_norm, vargrad_norm, grad_norm_norm = calculate_unbiased_stats(normalized_gradients_per_state)
                
        #         # Calculate stats for all training states
        #         meangrad2, vargrad2, grad_norm2 = calculate_unbiased_stats(normalize_gradients(gradients_new_states))
        #         # meangrad_norm, vargrad_norm, grad_norm_norm = calculate_gradient_stats_per_state(normalized_gradients_per_state)
        #         sorted_vargrad_indices = np.argsort(vargrad2)[::-1]  # Sort descending by variance
        #         sorted_meangrad_indices = np.argsort(meangrad2)[::-1]  # Sort descending by mean gradient
                
                
        #         total_length = len(sorted_vargrad_indices)  
        #         even_indices = np.linspace(0, total_length - 1, 1000, dtype=int)

        #         sampled_vargrad_indices = sorted_vargrad_indices[even_indices]
        #         # print(f"sampled_var indices: {sampled_vargrad_indices}")
        #         sampled_meangrad_indices = sorted_meangrad_indices[even_indices]
                
                


        #         max_var_indices_new_states = sampled_vargrad_indices[:num_states_to_replace]
        #         max_meangrad_indices_new_states = sampled_meangrad_indices[:num_states_to_replace]


        #         print(f"max_var_indices_new_states: {max_var_indices_new_states}")
        #         print(f"max_meangrad_indices_new_states: {max_meangrad_indices_new_states}")
        #         # Select the states from `second_A` and `second_B` based on `max_var_indices_new_states`
        #         add_a = np.asarray(second_A[max_var_indices_new_states])
        #         add_b = np.asarray(second_b[max_var_indices_new_states])
        #         # normalized_grads_variance_new = jnp.var(normalized_gradients_per_state, axis=tuple(range(1, normalized_gradients_per_state.ndim)))
         
               
                
      

        #         print(f"Epoch {epoch}:  cost: {cost:.5f}")
        #         print(f"***flat landscape warning at epoch {epoch} w/ roc: {acceleration:.2e} mean(grad): {np.mean(np.abs(grad)):.2e}, Var(Grad): {np.var(grad,ddof=1):.2e}***")

        #         for idx in range(len(input_states)):
        #             training_state_metrics[idx] = {
        #                 'Var(Grad)': vargrad_unbiased[idx],
        #                 'Mean(Grad)': meangrad_unbiased[idx],
        #                 'Norm(Grad)': grad_norm_unbiased[idx],  # This is now calculated per state
        #                 'Var(Grad)_norm': vargrad_norm[idx],
        #                 'Mean(Grad)_norm': meangrad_norm[idx],
        #                 'Norm(Grad)_norm': grad_norm_norm[idx]  # This is also per state now
        #             }
        #             # Single-line output per state
        #             # print(f"{idx} - ({vargrad_unbiased[idx]:.1e}), ({grad_norm_unbiased[idx]:.1e}), c: {meangrad_unbiased[idx]:.1e}")
        #             # print(f"{idx}: Var(Grad): ({vargrad[idx]:.1e},{vargrad_norm[idx]:.1e}) , Mean(Grad): ({meangrad[idx]:.1e},{meangrad_norm[idx]:.1e}), Var(NormGrad): {normalized_grads_variance[idx]:.1e}")


        #         min_var_indices = np.argsort(vargrad_unbiased)[:num_states_to_replace]
        #         print(f"    - indices selected on min variance: {min_var_indices}")
        #         # min_varnorm_indices = np.argsort(vargrad_norm)[:num_states_to_replace]
        #         # print(f"    - indices selected on minimum variance normgrad: {min_varnorm_indices}")

        #         min_gradnorm_indices = np.argsort(grad_norm_unbiased)[:num_states_to_replace]
        #         # print(f"    - indices selected on min gradient norm: {min_gradnorm_indices}")
        #         min_mean_indices = np.argsort(meangrad_unbiased)[:num_states_to_replace]
        #         print(f"    - indices selected on min mean gradients: {min_mean_indices}")
        #         replacement_indices = min_var_indices
        #         print(f"Selected states indices for replacement: {replacement_indices}")
                
                

        #         # Log selected states based on calculated stats
        #         print(f"\nVar(Grad) - Min: {vargrad_unbiased.min():.2e}, Max: {vargrad_unbiased.max():.2e}")
        #         # print(f"    - states: {[f'{val:.2e}' for val in vargrad[min_var_indices]]}")
        #         print(f"    - states: {[f's{i}: {vargrad_unbiased[i]:.1e}' for i in min_var_indices]}")
        #         # print(f"    - states: {[f's{i}: {vargrad_unbiased[i]:.1e}' for i in min_gradnorm_indices]}")
        #         # print(f"    - states: {[f's{i}: {vargrad_unbiased[i]:.1e}' for i in min_mean_indices]}")
        #         # print(f"    - states: {[f'({idx}, {val:.2e})' for idx,val in zip(min_varnorm_indices,vargrad[min_varnorm_indices])]}")
        #         print(f"\nMean(Grad) - Min: {meangrad_unbiased.min():.1e}, Max: {meangrad_unbiased.max():.2e}")
        #         print(f"    - states: {[f's{i}: {meangrad_unbiased[i]:.1e}' for i in min_var_indices]}")
        #         # print(f"    - states: {[f's{i}: {meangrad_unbiased[i]:.1e}' for i in min_gradnorm_indices]}")
        #         # print(f"    - states: {[f's{i}: {meangrad_unbiased[i]:.1e}' for i in min_mean_indices]}")
                
        #         # print(f"\nNorm(Grad) - Min: {vargrad_norm.min():.2e}, Max: {vargrad_norm.max():.2e}")
        #         # print(f"    - states: {[f'{val:.2e}' for val in vargrad_norm[min_var_indices]]}")
        #         # print(f"    - states: {[f'{val:.2e}' for val in vargrad_norm[min_varnorm_indices]]}")
        #         print(f"\nNew replacement states: ")
        #         print(f"    Indices selected on max var: {max_var_indices_new_states}")
        #         print(f"    - Var(Grad) ({vargrad2.min():.1e},{vargrad2.max():.1e}): {[f's{i}: {vargrad2[i]:.1e}' for i in max_var_indices_new_states]}")
        #         print(f"    Indices selected on mac mean: {max_meangrad_indices_new_states}")
        #         print(f"    - Mean(Grad) ({meangrad2.min():.1e},{meangrad2.max():.1e}): {[f's{i}: {meangrad2[i]:.1e}' for i in max_meangrad_indices_new_states]}")
        #         # print(f"    Indices selected on gradnorm: {max_gradnorm_indices_new_states}")
        #         # print(f"    - GradNorm ({grad_norm2.min():.1e},{grad_norm2.max():.1e}): {[f's{i}: {grad_norm2[i]:.1e}' for i in max_gradnorm_indices_new_states]}")
        #         # Replace the states with the smallest variances with the new states
        #         print(f"Replacing {num_states_to_replace} states with new states at epoch {epoch}")
        #         for idx in replacement_indices:
        #             input_states = input_states.at[idx].set(add_a[replacement_indices == idx].squeeze(axis=0))
        #             target_states = target_states.at[idx].set(add_b[replacement_indices == idx].squeeze(axis=0))
        #     else:
        #         # Concatenate the new states (add_a, add_b) with the existing input_states and target_states
        #         # Add new states (instead of replacing existing states)
        #         print(f"***Adding {num_states_to_replace} new states at epoch {epoch}***")

        #         input_states = np.concatenate([input_states, add_a], axis=0)
        #         target_states = np.concatenate([target_states, add_b], axis=0)
            
        #     add_more = False

        if np.abs(max(grad)) < 1e-14:
            print(f"max(grad)<1e-14. breaking....")
            break
        epoch += 1  # Increment epoch count


    if backup_cost < cost:
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
    
    data = {'Gate':base64.b64encode(pickle.dumps(gate)).decode('utf-8'),
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

                
                
            }
    now = datetime.now()
    print(f"Saving results to {filename}. Date/time: ", now.strftime("%Y-%m-%d %H:%M:%S"))
    df = pd.DataFrame([data])
    while os.path.exists(filename):
        name, ext = filename.rsplit('.', 1)
        filename = f"{name}_.{ext}"

    with open(filename, 'wb') as f:
        pickle.dump(df, f)


 

if __name__ == '__main__':


    

    


    
    
    
    #folder = './results_jax_baths_global_h/'
    # Example usage

    
    # run below 
    N_ctrl = 2
    # trots = [1,6,8,10,12,14,17,18,19,20,21,22]
    # trots = [6,7,8,9,10,11,12,13,14,15,16]
    # trots = [1,2,3,4,5,6,7,8]
    # trots = [2, 4, 6, 8, 10,12,14,16,18,20,22,24,26,28,30]
    # trots = [15,18,21,24,27,30,33,36,39,40]
    # trots = [7,8,9,10,11,12,13,14,15,16]
    # trots = [9, 12, 15, 18,20,21,24,28,32,36,40,44,48,52]

   
    # trots = [1,2,3,4,5,6,7,8]

    
    # res = [1,2]
    # trots = [2,12]
    res = [1,2,3]
    trots = [1, 8, 10, 12, 14, 16, 18, 20, 22, 26, 28]
    # trots = [1,6,8,10,12,14,16,18,20,22,24,26,28,30]
    
    




    num_epochs = 1500
    N_train = 10
    add=0
    # if N_ctrl ==4:
    #     add = 5_optimized_by_cost3
    
    # folder = f'./analog_results_trainable_global/trainsize_{N_train+add}/'
    folder = f'./analog_results_trainable_global/trainsize_{N_train}_optimized_by_cost3/'

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

    
    #gates_known = [qml.CNOT,qml.CY]
    #for g in gates_known:
    #    g.name = g(wires=list(range(g.num_wires))).name

    

    for gate_idx,gate in enumerate(gates_random):
        
        # key,subkey = jax.random.split(key)
        if True:

            # if not gate_idx in [0,1]:
            #    continue
            # if gate_idx <= 1:
            #     continue

            for time_steps in trots:

                
                
                
                for N_reserv in res:
                    
                    N =N_ctrl+N_reserv
                    
                    #folder = f'./param_initialization/Nc{N_ctrl}_Nr{N_reserv}_dt{time_steps}/fixed_params4/test7/'
                    for num_bath,bath in zip(num_baths,baths):
                        params_key_seed = gate_idx*121 * N_reserv + 12345 * time_steps *N_reserv
                        params_key = jax.random.PRNGKey(params_key_seed)
                        main_params = jax.random.uniform(params_key, shape=(3 + (N_ctrl * N_reserv) * time_steps,), minval=-np.pi, maxval=np.pi)
                        # print(f"main_params: {main_params}")
                        params_key, params_subkey1, params_subkey2 = jax.random.split(params_key, 3)
                        
                        
                        time_step_params = jax.random.uniform(params_key, shape=(time_steps,), minval=0, maxval=np.pi)
                        init_params_dict = get_init_params(N_ctrl, N_reserv, time_steps,bath,num_bath,params_subkey1)
                        
    

                        # Combine the two parts
                        params = jnp.concatenate([time_step_params, main_params])
                        # params = jnp.asarray([0.4033546149730682, 1.4487122297286987, 2.3020467758178711, 2.9035964012145996, 0.9584765434265137, 1.7428307533264160, -1.3020169734954834, -0.8775904774665833, 2.4736261367797852, -0.4999605417251587, -0.8375297188758850, 1.7014273405075073, -0.8763229846954346, -3.1250307559967041, 1.1915868520736694, -0.4640290737152100, -1.0656110048294067, -2.6777451038360596, -2.7820897102355957, -2.3751690387725830, 0.1393062919378281])




                        run_test(params, num_epochs, N_reserv, N_ctrl, time_steps,N_train,folder,gate,gate.name,bath,num_bath,init_params_dict = init_params_dict,random_key = params_subkey2)
