import pennylane as qml
import os
import pickle
import re
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from qiskit import *

from jax import numpy as jnp
import sympy
import matplotlib.pyplot as plt
import base64
import pickle
from qutip import *
from qutip.qip.operations import cnot,rz,rx,ry,snot
from qutip.qip.circuit import QubitCircuit
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

def is_non_empty_file(fpath):  
    """ Check if file is non-empty """
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0
def create_ghz_state(num_qubits):
    """Creates a GHZ state for a given number of qubits."""
    state = jnp.zeros((2**num_qubits,), dtype=jnp.complex128)
    state = state.at[0].set(1.0 / jnp.sqrt(2))
    state = state.at[-1].set(1.0 / jnp.sqrt(2))
    return state

def get_init_params(N_ctrl, N_reserv, time_steps, bath, num_bath,K_factor, base_key):
    N = N_reserv + N_ctrl
    # Adjust the key based on time_steps and fixed_param_num
    key = jax.random.PRNGKey((base_key + time_steps) * 123456789 % (2**32))  # Example combination
    key, subkey = jax.random.split(key)
   
    K_half = jax.random.normal(key = subkey,shape= (N, N)) 
    K = (K_half + K_half.T) / 2  # making the matrix symmetric
    K = 2. * K - 1.
    K *= K_factor
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
def quantum_fun(gate, input_state, qubits):
    '''
    Apply the gate to the input state and return the output state.
    '''
    qml.QubitStateVector(input_state, wires=[*qubits])
    gate(wires=qubits)
    return qml.density_matrix(wires=[*qubits])

def generate_dataset(gate, n_qubits, training_size):
    '''
    Generate the dataset of input and output states according to the gate provided.
    '''
    X = np.stack([np.asarray(random_statevector(2**n_qubits, i)) for i in range(training_size)])
    qubits = Wires(list(range(n_qubits)))
    dev_data = qml.device('default.qubit', wires = [*qubits])
    circuit = qml.QNode(quantum_fun, device=dev_data, interface = 'jax')
    y = np.stack([np.asarray(circuit(gate, X[i], qubits)) for i in range(training_size)])
    
    return X, y


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

        # if non-markovian noise present, add interactions
        if self.bath:
            
            for bath_qubit_idx, bath_qubit in enumerate(self.bath_qubits):
                # reservoir-bath interactions
                for res_qubit_idx, res_qubit in enumerate(self.reserv_qubits):
                    interaction_coeff = self.bath_interactions[bath_qubit_idx, res_qubit_idx]
                    static_coefficients.append(interaction_coeff)
                    new_operator = self.get_XY_coupling(bath_qubit, res_qubit)
                    static_operators.append(new_operator)


                    static_coefficients.append(interaction_coeff)
                    new_operator = self.get_XY_coupling(res_qubit,bath_qubit)
                    static_operators.append(new_operator)

                # ctrl-bath interactions
                for ctrl_qubit_idx, ctrl_qubit in enumerate(self.ctrl_qubits):
                    
                    interaction_coeff = self.bath_interactions[bath_qubit_idx, self.N_reserv + ctrl_qubit_idx]
                    static_coefficients.append(interaction_coeff)
                    new_operator = self.get_XY_coupling(bath_qubit, ctrl_qubit)
                    static_operators.append(new_operator)


                    #static_coefficients.append(interaction_coeff)
                    #new_operator = self.get_XY_coupling(self.ctrl_qubits[qubit_b],self.bath_qubits[qubit_a])
                    #static_operators.append(new_operator)




        #print(static_coefficients, static_operators)
        if self.N_reserv == 1:
            total_H = H_dynamic
        
        else:
            H_static = qml.dot(static_coefficients, static_operators)
            total_H = H_dynamic+H_static
        ##sum(coeff * op for coeff, op in zip(static_coefficients, static_operators))
        
        
        

        return total_H




def generate_batch_params(tests, time_steps, N_ctrl, N_reserv, sample_range):
    param_batch = []
    
    # Base seed for reproducibility
    base_seed = 1987 * N_reserv + 123 * time_steps * N_ctrl
    params_key = jax.random.PRNGKey(base_seed)

    for test_num in tests:
        # Fold in the test number for reproducibility (similar to digital)
        test_key = jax.random.fold_in(params_key, test_num)

        # Split key for separate generation of time and other parameters
        test_key, params_subkey1, params_subkey2 = jax.random.split(test_key, 3)

        # Time parameters: Use uniform distribution with non-negative values (0 to 1)
        time_step_params = jax.random.uniform(params_subkey1, shape=(time_steps,), minval=0, maxval=sample_range)

        # Remaining parameters: Generate uniformly in range [-sample_range, sample_range]
        remaining_params = jax.random.uniform(params_subkey2, shape=(3 + (N_ctrl * N_reserv) * time_steps,), minval=-sample_range, maxval=sample_range)

        # Combine the time parameters and the remaining parameters
        params = jnp.concatenate([time_step_params, remaining_params])
        
        # Append to the batch
        param_batch.append(params)

    # Stack the batch into a single array for easy use
    return jax.numpy.stack(param_batch)


def vectorize(matrix):
    """Vectorize a matrix by stacking its columns."""
    return jnp.ravel(matrix, order='F')
def quantum_fun(gate, input_state, qubits):
    '''
    Apply the gate to the input state and return the output state.
    '''
    qml.QubitStateVector(input_state, wires=[*qubits])
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
    number_of_fixed_param_tests = 1
    number_trainable_params_tests = 25 

    base_state = 'GHZ_state'
    
    
    # trots = [1,2,3,4,5,6,7,8,9,10]
    
    # trots = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
    # trots = [8,10,12]
    # trots = [1,2,3,4,4,5,6,7,8,9,10,12,16,20,24]
    # trots = [1]
    trots = np.arange(1,20,1)
    reservoirs = [1]
   
    
   
    delta_x = 1.49011612e-08
    #Kfactors = [0.01,0.1,1.,10]
    Kfactors = [1.0]
    #folder = f'./QFIM_traced_trainable_global/analog_model/Nc_{N_ctrl}/{base_state}/{Kfactor}xK/'
    key = jax.random.PRNGKey(103450*N_ctrl)
    batch = True
    sample_range = 2*np.pi
    sample_range_label = '2pi'
    num_input_states = 50
    for Kfactor in Kfactors:
        for time_steps in trots:
            for N_reserv in reservoirs:
                
                
                for bath in baths:
                    print(f"N_ctrl: {N_ctrl}, N_R: {N_reserv}, time_steps: {time_steps}, num_bath: {num_bath}")
                    folder = f'./QFIM_traced_final_results/analog_model_DQFIM/Nc_{N_ctrl}/{base_state}/'
                    folder_gate = os.path.join(folder, f'Nr_{N_reserv}', f'trotter_step_{time_steps}',f'L_{num_input_states}')
                    print("\nfolder_gate: ",folder_gate)
                    Path(folder_gate).mkdir(parents=True, exist_ok=True)
                    N = N_reserv+N_ctrl
                    num_J = N_reserv*N_ctrl
                    if sample_range:
                        data_filename = os.path.join(folder_gate, f'data_{sample_range_label}.pickle')
                    else:
                        data_filename = os.path.join(folder_gate, 'data.pickle')
                    
                    
                    L = generate_dataset(N_ctrl,N_reserv, num_input_states,key)
                    print(data_filename)
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
                        
                        number_trainable_tests_completed = len(all_tests_data[fixed_param_key].keys())
                        tests_to_run = number_trainable_params_tests - number_trainable_tests_completed
                        print(f"Number of tests to run for {fixed_param_key}: ", tests_to_run)
                        if tests_to_run > 0:
                            
                            base_key = fixed_param_num *fixed_param_num+ 1000 * time_steps*fixed_param_num
                            fixed_params = get_init_params(N_ctrl, N_reserv, time_steps, bath, num_bath, Kfactor,base_key)
                            print(f"fixed: {fixed_params}")
                            #test_state = create_initial_state(N_ctrl,base_state)
                            sim_qr = Sim_QuantumReservoir(fixed_params, N_ctrl, N_reserv, num_J, time_steps=time_steps)
        
                            # get hamiltonian circuit variables
                            parameterized_ham = sim_qr.get_total_hamiltonian_components()
                            #print(f"hamiltonian: {parameterized_ham}")
                            ctrl_wires = sim_qr.get_ctrl_wires()
                            reserv_wires = sim_qr.get_reserv_wires()
                            all_wires = sim_qr.get_all_wires()
                            qnode_dev = sim_qr.get_dev()
                            
                            # test_state = create_initial_state(N_ctrl,base_state)

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
                                return eigvals, eigvecs,QFIM, entropies
    
                            
                            # Generate a batch of parameters
                            param_batch = generate_batch_params(np.arange(number_trainable_tests_completed,number_trainable_params_tests), time_steps, N_ctrl, N_reserv, sample_range)
                            if batch:
                                # gradients = collect_gradients(param_batch)
                                s = time.time()
                                batch_results = jax.vmap(compute_qfim_eigen_decomposition)(param_batch)
                                # batch_results = jax.vmap(get_qfi_eigvals, in_axes=(0,))(param_batch)
                                e = time.time()
                                print(f'Time for param set: {e-s}')

                                # Restructuring results
                                restructured_results = [(batch_results[0][i], batch_results[1][i], batch_results[2][i], batch_results[3][i]) for i in range(number_trainable_params_tests)]

                                
                                
                            # s = time.time()
                                for test_num, (eigvals, eigvecs, dqfim, entropies) in enumerate(restructured_results):
                                    test_key = f'test{test_num + number_trainable_tests_completed}'
                                    params = param_batch[test_num]
                                    if jnp.any(jnp.isnan(dqfim)) or jnp.any(jnp.isinf(dqfim)):
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
                                        'L':L,
                                        'fixed_params': fixed_params,
                                        'qfim_eigvals': eigvals,
                                        'qfim_eigvecs': eigvecs,
                                        'trainable_params': params,
                                        
                                        'entropies': entropies,
                                        'dqfim': dqfim,
                                        
                                        'n_reserv': N_reserv,
                                        'n_ctrl': N_ctrl,
                                        'time_steps': time_steps,
                                        'num_bath': num_bath,
                                        'sample_range': sample_range_label
                                    }
                                    
                                    new_tests_count += 1
                            #e = time.time()
                            #print(f'Time for getting entropy for new param: {e-s}')
                            #print(f"Added {new_tests_count} new tests for {fixed_param_key}")

                                with open(data_filename, 'wb') as f:
                                    pickle.dump(all_tests_data, f)
                            else:
                                # Non-batch processing, run each test one by one
                                print("Running tests one by one...")
                                for test_num, params in enumerate(param_batch):
                                    print(f"Running test {test_num + 1} of {len(param_batch)}")
                                    s = time.time()
                                    eigvals, eigvecs, dqfim,entropies = compute_qfim_eigen_decomposition(params)
                                    e = time.time()
                                    print(f'Time for getting data: {e-s}')
                                    test_key = f'test{test_num + number_trainable_tests_completed}'
                                    if jnp.any(jnp.isnan(dqfim)) or jnp.any(jnp.isinf(dqfim)):
                                        raise ValueError("QFIM contains NaNs or infinite values")
                                    error_threshold = -1e-12
                                    if jnp.any(eigvals < error_threshold):
                                        print(f"Negative eigenvalue detected for test {test_key}, eigvals: {eigvals}")
                                        raise ValueError("QFIM contains negative eigenvalues")
                                    all_tests_data[fixed_param_key][test_key] = {
                                        'L':L,
                                        'fixed_params': fixed_params,
                                        'qfim_eigvals': eigvals,
                                        'qfim_eigvecs': eigvecs,
                                        'trainable_params': params,
                                        
                                        'entropies': entropies,
                                        'dqfim': dqfim,
                                        
                                        'n_reserv': N_reserv,
                                        'n_ctrl': N_ctrl,
                                        'time_steps': time_steps,
                                        'num_bath': num_bath
                                    }
                                    
                                    new_tests_count += 1
                                    with open(data_filename, 'wb') as f:
                                        pickle.dump(all_tests_data, f)
                            #print(f"Updated and added {tests_to_run} tests")
                            
if __name__ == '__main__':
    main()