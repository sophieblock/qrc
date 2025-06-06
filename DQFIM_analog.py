import pennylane as qml
import os
import pickle
import re
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from qiskit import *
from datetime import datetime
from jax import numpy as jnp
import sympy
import matplotlib.pyplot as plt
import base64
import pickle
from qutip import *

 # Using pennylane's wrapped numpy
from sympy import symbols, MatrixSymbol, lambdify, Matrix, pprint
import jax
import numpy as np
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
from pennylane.math import vn_entropy
import matplotlib.cm as cm
from functools import partial
from pennylane import numpy as pnp
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

#from pennylane.pulse import ParametrizedEvolution, ParametrizedHamiltonian,HardwareHamiltonian
from jax.experimental.ode import odeint
from pennylane.devices.qubit.apply_operation import _evolve_state_vector_under_parametrized_evolution,apply_parametrized_evolution
has_jax = True
diable_jit = False
config.update('jax_disable_jit', diable_jit)
#config.parse_flags_with_absl()
config.update("jax_enable_x64", True)
os.environ['JAX_TRACEBACK_FILTERING'] = 'off'

def is_non_empty_file(fpath):  
    """ Check if file is non-empty """
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0
def create_ghz_state(num_qubits):
    """Creates a GHZ state for a given number of qubits."""
    state = jnp.zeros((2**num_qubits,), dtype=jnp.complex128)
    state = state.at[0].set(1.0 / jnp.sqrt(2))
    state = state.at[-1].set(1.0 / jnp.sqrt(2))
    return state


def get_init_params(N_ctrl, N_reserv, time_steps, bath, num_bath,key, E_0=1.0,K_0 = 1.0):
    N = N_reserv + N_ctrl
   
   
    K_half = jax.random.normal(key = key,shape= (N, N)) 
    K = (K_half + K_half.T) / 2  # making the matrix symmetric
    K = 2. * K - 1.
    K = K * K_0/2
    #print(f"K: {K}")
    if bath:
        bath_array = 0.01 * jax.random.normal(subkey, (num_bath, N_ctrl + N_reserv))
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
    state = jnp.zeros(2**num_qubits)

    if base_state == 'basis_state':
        state = state.at[0].set(1)

    elif base_state == 'GHZ_state':
        state = state.at[0].set(1 / jnp.sqrt(2))
        state = state.at[-1].set(1 / jnp.sqrt(2))

    return state


class Sim_QuantumReservoir:
    def __init__(self, params, N_ctrl, N_reserv, num_J,time_steps=1,bath=False,num_bath = 0):
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

        #print(params['bath'])
        self.num_J = num_J
        self.params = params
        self.current_index = 0

   
                
    def get_dev(self):
        return self.dev

    def get_ctrl_wires(self):
        return self.ctrl_qubits

    def get_reserv_wires(self):
        return self.reserv_qubits   
    def get_all_wires(self):
        return self.qubits   
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




def generate_batch_params(params_key, tests, time_steps, N_ctrl, N_reserv,sample_range):
    param_batch = []
    for test_num in tests:
        
        test_key = jax.random.fold_in(params_key, test_num)

        
        remaining_params = jax.random.truncated_normal(test_key,
                                             shape=(3 + (N_ctrl * N_reserv) * time_steps,), 
                                             lower=-sample_range, 
                                             upper=sample_range)
        
        # remaining_params = jax.random.uniform(test_key,shape=(3 + (N_ctrl * N_reserv) * time_steps,), minval=-sample_range, maxval=sample_range)
        _, params_subkey1 = jax.random.split(test_key, 2)
        time_step_params = np.array([1.]*time_steps)
        # time_step_params = jax.random.uniform(params_subkey1, shape=(time_steps,), minval=0, maxval=1)
        params = jnp.concatenate([time_step_params, remaining_params])
        param_batch.append(params)
    return jax.numpy.stack(param_batch)


def vectorize(matrix):
    """Vectorize a matrix by stacking its columns."""
    return jnp.ravel(matrix, order='F')
def quantum_fun(gate, input_state, qubits):
    '''
    Apply the gate to the input state and return the output state.
    '''
    qml.StatePrep(input_state, wires=[*qubits])
    gate(wires=qubits)
    # return qml.density_matrix(wires=[*qubits])
    return qml.state()

def generate_dataset(N_ctrl,N_reserv, training_size, key):
    '''
    Generate the dataset of input and output states according to the gate provided.
    Uses a seed for reproducibility.
    '''
    

    # Generate random state vectors
    X = []
    keys = jax.random.split(key, num=training_size)
    for i, subkey in enumerate(keys):
        subkey = jax.random.fold_in(subkey, i)  # Fold in the index to guarantee uniqueness
        seed_value = int(jax.random.randint(subkey, (1,), 0, 2**32 - 1)[0])  # Get a scalar seed
        
        # Use the seed to generate the random state vector
        state_vec = random_statevector(2**N_ctrl, seed=seed_value).data
        X.append(np.asarray(state_vec, dtype=jnp.complex128))
    
    
    L = np.stack(X)
    
    
    return L

def main():
    def is_normalized_density_matrix(density_matrix):
        trace_value = jnp.trace(density_matrix)
        return jnp.isclose(trace_value, 1.0)
    N_ctrl = 2
    baths = [False]
    num_bath = 0
    number_of_fixed_param_tests_base = 5
    number_trainable_params_tests = 25
    num_input_states = 20
    base_state = f'L_{num_input_states}'
    
    
    # trots = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    
    trots = [1,2,3,4,5,6,7]
    # trots = [10]
    reservoirs = [1,2]


    
   
    delta_x = 1.49011612e-08
    #Kfactors = [0.01,0.1,1.,10]
    kFactor = 1
    K_0 = kFactor
    # folder = f'./QFIM_traced_final_results/gate_model_DQFIM/Nc_{N_ctrl}/{base_state}/{kFactor}xK/'
    #folder = f'./QFIM_traced_trainable_global/analog_model/Nc_{N_ctrl}/{base_state}/{Kfactor}xK/'

    batch = True
    # sample_range = np.pi/2
    # sample_range_label = '.5pi'
    sample_range = np.pi
    sample_range_label = '2pi_1tau'

    folder = f'./DQFIM_results/analog/Nc_{N_ctrl}/sample_{sample_range_label}/{kFactor}xK/'
    for time_steps in trots:
        for N_reserv in reservoirs:
            
            if N_reserv == 1:
                number_of_fixed_param_tests = 1
            else:
                number_of_fixed_param_tests = number_of_fixed_param_tests_base
            for bath in baths:
                print("________________________________________________________________________________")
                print(f"N_ctrl: {N_ctrl}, N_R: {N_reserv}, time_steps: {time_steps}")
                
                folder_gate = os.path.join(folder, f'Nr_{N_reserv}', f'trotter_step_{time_steps}',f'L_{num_input_states}')
                print("\nfolder_gate: ",folder_gate)
                Path(folder_gate).mkdir(parents=True, exist_ok=True)
                N = N_reserv+N_ctrl
                data_filename = os.path.join(folder_gate, 'data.pickle')
                num_J = N_reserv*N_ctrl
                
                base_key_seed = 123* N_reserv + 12345 * time_steps *N_reserv
                params_key = jax.random.PRNGKey(base_key_seed)
                params_subkey0, params_subkey1 = jax.random.split(params_key, 2)
                    
                L = generate_dataset(N_ctrl,N_reserv, num_input_states,params_subkey1)
                # print(data_filename)
                all_tests_data = {}

                if is_non_empty_file(data_filename):
                    with open(data_filename, 'rb') as f:
                        all_tests_data = pickle.load(f)
                else:
                    all_tests_data = {}
                
                number_fixed_tests_completed = len(all_tests_data.keys())
                print("number_fixed_tests_completed: ", number_fixed_tests_completed)
                for fixed_param_num in range(0, number_of_fixed_param_tests):
                    
                    new_tests_count = 0  # Counter for new tests
                    fixed_param_key = f'fixed_params{fixed_param_num}'
                    all_tests_data.setdefault(fixed_param_key, {})

                    # if fixed_param_key != 'fixed_params0':
                    #     print(f"Resetting {fixed_param_key}")
                    #     all_tests_data[fixed_param_key] = {}
                    
                    number_trainable_tests_completed = len(all_tests_data[fixed_param_key].keys())
                    tests_to_run = number_trainable_params_tests - number_trainable_tests_completed
                    print(f"{fixed_param_key}... Completed: {number_trainable_tests_completed}")

                    # print(f"Number of tests to run for {fixed_param_key}... Completed: {number_trainable_tests_completed}, set = {number_trainable_params_tests} ")
                    if tests_to_run > 0:
                        print(f"{tests_to_run} test(s) to run...")
                    else:
                        print("Moving on!")
                    if tests_to_run > 0:
                        base_key_seed = fixed_param_num*123 + fixed_param_num*1000 * time_steps

                        params_key = jax.random.PRNGKey(base_key_seed)
                        K_half =  jax.random.normal(params_key, shape=(N,N))*kFactor
                        K = (K_half + K_half.T) / 2  # making the matrix symmetric
                        K = 2. * K - 1.
                        K_coeffs = K * K_0/2
                        fixed_params = {'K_coef':K_coeffs}
                        #test_state = create_initial_state(N_ctrl,base_state)
                        sim_qr = Sim_QuantumReservoir(fixed_params, N_ctrl, N_reserv, num_J, time_steps=time_steps)
    
                        # get hamiltonian circuit variables
                        parameterized_ham = sim_qr.get_total_hamiltonian_components()
                   
                        ctrl_wires = sim_qr.get_ctrl_wires()
                      
                        qnode_dev = sim_qr.get_dev()
      
                        @jax.jit
                        @qml.qnode(qnode_dev, interface="jax")
                        def _circuit(params, input_state):
                            
                            taus = params[:time_steps]

                            qml.StatePrep(input_state, wires=[*ctrl_wires])
                            

                            for idx, tau in enumerate(taus):
                            
                                hx_array = jax.numpy.array([params[time_steps]])  # Convert hx to a 1D array
                                hy_array = jax.numpy.array([params[time_steps + 1]])  # Convert hy to a 1D array
                                hz_array = jax.numpy.array([params[time_steps + 2]])  # Convert hz to a 1D array
                                J_values = params[time_steps + 3 + idx * num_J : time_steps + 3 + (idx + 1) * num_J]

                                # Concatenate hx_array with J_values
                                current_step = jax.numpy.concatenate([J_values,hx_array,hy_array,hz_array])

                                qml.evolve(parameterized_ham)(current_step, t=tau)
                            
                            
                            return qml.density_matrix(wires=[*ctrl_wires])
                        jit_circuit = jax.jit(_circuit)
                        
                        def get_partial_grads(params, input_states,jit_circuit, delta_x=1.49011612e-08):
                            """
                            Computes the averaged gradients of the PQC output density matrix 
                            with respect to each parameter for all training states.
                            
                            input_states: A batch of training states (|\psi_l\rangle).
                            """
                            
                            all_res = []
                            

                            def single_param_grad(params,idx,input_state):


                                # Shift parameter up
                                shifted_params_plus = params.at[idx].set(params[idx] + delta_x)
                                shifted_plus_circuit = jit_circuit(shifted_params_plus, input_state)

                                shifted_params_minus = params.at[idx].set(params[idx] - delta_x)
                                shifted_minus_circuit = jit_circuit(shifted_params_minus, input_state)

                                
                                # Calculate the gradient as the symmetric difference
                                grad = (shifted_plus_circuit - shifted_minus_circuit) / (2 * delta_x)
                                
                                return grad


                            # Initialize a variable to store the sum of the gradients
                            for idx in range(len(params)):
                                grad_sum = jnp.zeros_like(jit_circuit(params, input_states[0]))  # Initialize to zero matrix
                                
                                # Loop over all training states to compute and sum the gradients
                                for input_state in input_states:

                                    density_matrix_grad = single_param_grad(params,idx, input_state)
                                    grad_sum += density_matrix_grad
                                    
                                    
                                # Average the gradient over all the training states
                                avg_grad = grad_sum / len(input_states)
                                all_res.append(avg_grad)
                            
                            return jnp.asarray(all_res)
                        

                        

    
                        def get_density_matrix_sum(params, input_states, jit_circuit):
                            """
                            Computes the sum of density matrices after applying the PQC on all training states using a pre-jitted circuit.
                            
                            input_states: A batch of training states (|\psi_l\rangle).
                            jit_circuit: A pre-jitted version of the quantum circuit.
                            """
                            
                            # Initialize a variable to store the sum of the density matrices
                            density_matrix_sum = jnp.zeros_like(jit_circuit(params, input_states[0]))
                            entropies = []
                            # Loop over each input state and sum the density matrices
                            for input_state in input_states:
                                out = jit_circuit(params, input_state)
                                entropy = vn_entropy(out, indices=[*ctrl_wires])
                                entropies.append(entropy)
                                density_matrix_sum += out
                            # print(f"type density_matrix_sum: {type(density_matrix_sum)}, {type(L)}")
                            # Return the averaged density matrix (Π_L)
                            return jnp.array(entropies), density_matrix_sum / num_input_states

                        
                        # @jax.jit
                        def compute_qfim_eigen_decomposition(params):
                            density_matrix_grads = get_partial_grads(params, L, jit_circuit)
                            entropies,Pi_L = get_density_matrix_sum(params, L, jit_circuit)

                            # density_matrix_grads, rho = get_unitary_grad(params)
                            Pi_L = Pi_L / jnp.trace(Pi_L)
                            
                            # Eigenvalue decomposition
                            eigvals, eigvecs = jnp.linalg.eigh(Pi_L)
                            
                            n_params = len(density_matrix_grads)
                            QFIM = jnp.zeros((n_params, n_params), dtype=jnp.complex128)
                            
                            for a in range(n_params):
                                vec_grad_a = density_matrix_grads[a]
                                
                                for b in range(n_params):
                                    vec_grad_b = density_matrix_grads[b]
                                    
                                    sum_terms = 0
                                    for i in range(len(eigvals)):
                                        for j in range(len(eigvals)):
                                            de = eigvals[i] + eigvals[j]
                                            
                                            valid = jnp.where(de > 1e-12, 1.0, 0.0)  # 1.0 if true, 0.0 if false
                                            num1 = jnp.dot(jnp.conj(eigvecs[:, i]), jnp.dot(vec_grad_a, eigvecs[:, j]))
                                            num2 = jnp.dot(jnp.conj(eigvecs[:, j]), jnp.dot(vec_grad_b, eigvecs[:, i]))
                                            
                                            term = (num1 * num2) / (de + 1e-12)  # Adding small value to avoid division by zero
                                            sum_terms += 2.0 * jnp.real(term)
                                            
                                    
                                    QFIM = QFIM.at[a, b].set(sum_terms)
                            eigvals, eigvecs = jnp.linalg.eigh(QFIM)
                            return eigvals, eigvecs,QFIM, entropies,Pi_L

                        
                        test_indices = np.arange(number_trainable_tests_completed,number_trainable_params_tests)
                        param_batch = generate_batch_params(params_subkey0,test_indices, time_steps, N_ctrl=N_ctrl, N_reserv=N_reserv,sample_range=sample_range)

                    
                        if batch:
                            # gradients = collect_gradients(param_batch)
                            s = time.time()
                            batch_results = jax.vmap(compute_qfim_eigen_decomposition)(param_batch)
                            batch_results = jax.tree_util.tree_map(jax.block_until_ready,batch_results)
                            # batch_results = jax.vmap(get_qfi_eigvals, in_axes=(0,))(param_batch)
                            e = time.time()
                            print(f'Time for param set: {e-s}')
                            now = datetime.now()

                            # Format and print the date and time
                            print("Current date and time:", now.strftime("%Y-%m-%d %H:%M:%S"))

                            # Restructuring results
                            restructured_results = [
                                (batch_results[0][i], batch_results[1][i], batch_results[2][i], batch_results[3][i], batch_results[4][i])
                                for i in range(len(param_batch))  # or range(tests_to_run)
                            ]
                            
                            
                            for test_num, (eigvals, eigvecs, qfim, entropies,Pi_L) in enumerate(restructured_results):
                                test_key = f'test{test_num + number_trainable_tests_completed}'
                                params = param_batch[test_num]
                                if jnp.any(jnp.isnan(qfim)) or jnp.any(jnp.isinf(qfim)):
                                    raise ValueError("dqfim contains NaNs or infinite values")
                                # Check if any eigenvalue is less than 0
                                error_threshold = -1e-11
                                if jnp.any(eigvals < error_threshold):
                                    # Print the params and fixed params if negative eigenvalues are found
                                    print(f"Negative eigenvalue detected for test {test_key}\neigvals: {eigvals}")
                                    param_str = ', '.join([str(p) for p in params])
                                    fixed_param_str = ', '.join([f'N_ctrl: {N_ctrl}', f'N_reserv: {N_reserv}', 
                                                                f'time_steps: {time_steps}', f'Kfactor: {Kfactor}', 
                                                                f'num_bath: {num_bath}'])
                                    print(f"Trainable params: {param_str}")
                                    print(f"Fixed params: {fixed_params}")
                                    raise ValueError("qfim contains negative eigenvalues")
                                # original_circuit = jit_circuit(params)
                                # original_circuit = original_circuit / np.trace(original_circuit)
                                # entropy = vn_entropy(original_circuit, indices=[*ctrl_wires])
                                
                                all_tests_data[fixed_param_key][test_key] = {
                                    'L': L,
                                    "Pi_L":Pi_L,
                                    'entropies': entropies,
                                    'fixed_params': fixed_params,
                                    'K_coeffs': K_coeffs,
                                    'qfim_eigvals': eigvals,
                                    'qfim_eigvecs': eigvecs,
                                    'trainable_params': params,
                                    
                                    'qfim': qfim,
                                    
                                    'n_reserv': N_reserv,
                                    'n_ctrl': N_ctrl,
                                    'time_steps': time_steps,
                                    'num_bath': num_bath,
                                    'sample_range': sample_range
                                }
                                
                                new_tests_count += 1
                        #e = time.time()
                        #print(f'Time for getting entropy for new param: {e-s}')
                        #print(f"Added {new_tests_count} new tests for {fixed_param_key}")
                            
                            
                        else:
                            # Non-batch processing, run each test one by one
                            print("Running tests one by one...")
                            for test_num, params in enumerate(param_batch):
                                print(f"Running test {test_num + 1} of {len(param_batch)}")
                                s = time.time()
                                eigvals, eigvecs, qfim,entropies,Pi_L = compute_qfim_eigen_decomposition(params)
                                e = time.time()
                                print(f'Time for getting data: {e-s}')
                                test_key = f'test{test_num + number_trainable_tests_completed}'
                                if jnp.any(jnp.isnan(qfim)) or jnp.any(jnp.isinf(qfim)):
                                    raise ValueError("QFIM contains NaNs or infinite values")
                                error_threshold = -1e-12
                                if jnp.any(eigvals < error_threshold):
                                    print(f"Negative eigenvalue detected for test {test_key}, eigvals: {eigvals}")
                                    raise ValueError("QFIM contains negative eigenvalues")
                                all_tests_data[fixed_param_key][test_key] = {
                                    'L': L,
                                    "Pi_L":Pi_L,
                                    'entropies': entropies,
                                    'fixed_params': fixed_params,
                                    'K_coeffs': K_coeffs,
                                    'qfim_eigvals': eigvals,
                                    'qfim_eigvecs': eigvecs,
                                    'trainable_params': params,
                                    
                                    'qfim': qfim,
                                    
                                    'n_reserv': N_reserv,
                                    'n_ctrl': N_ctrl,
                                    'time_steps': time_steps,
                                    'num_bath': num_bath,
                                    'sample_range': sample_range
                                }
                                
                                new_tests_count += 1
                            # with open(data_filename, 'wb') as f:
                            #     pickle.dump(all_tests_data, f)
                        # print(f"Updated and added {new_tests_count} tests")
                        with open(data_filename, 'wb') as f:
                                pickle.dump(all_tests_data, f)
                        print(f"Updated and added {new_tests_count} tests")
                    else:
                        print(f"{number_trainable_params_tests} completed ({tests_to_run} tests to run) for {fixed_param_key}. Continue..")    

if __name__ == '__main__':
    main()