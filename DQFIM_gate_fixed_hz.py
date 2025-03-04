import pennylane as qml
import os
import pickle
from pennylane.math import vn_entropy
import re
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from qiskit import *
from jax import numpy as np
from jax import numpy as jnp
import sympy
import matplotlib.pyplot as plt
import base64
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


# Function to calculate variance
def calculate_gradient_variance(gradients):
    grad_matrix = jnp.array(gradients)
    mean_grad = jnp.mean(grad_matrix, axis=0)
    var_grad = jnp.mean((grad_matrix - mean_grad) ** 2, axis=0)
    return var_grad
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
def quantum_fun(gate, input_state, qubits):
    '''
    Apply the gate to the input state and return the output state.
    '''
    qml.StatePrep(input_state, wires=[*qubits])
    gate(wires=qubits)
    return qml.density_matrix(wires=[*qubits])

# def generate_dataset(gate, n_qubits, training_size):
#     '''
#     Generate the dataset of input and output states according to the gate provided.
#     '''
#     X = np.stack([np.asarray(random_statevector(2**n_qubits, i)) for i in range(training_size)])
#     qubits = Wires(list(range(n_qubits)))
#     dev_data = qml.device('default.qubit', wires = [*qubits])
#     circuit = qml.QNode(quantum_fun, device=dev_data, interface = 'jax')
#     y = np.stack([np.asarray(circuit(gate, X[i], qubits)) for i in range(training_size)])
    
#     return X, y



# class QuantumReservoirGate:

#     def __init__(self, n_rsv_qubits, n_ctrl_qubits, K_coeffs,trotter_steps=1, static=False, bath_params=False,num_bath = 0):
#         self.static = static
#         self.bath_params = bath_params
#         self.num_bath = num_bath
#         self.ctrl_qubits = Wires(list(range(n_ctrl_qubits))) # wires of the reservoir qubits (i.e. number of qubits in the reservoir)
#         self.rsv_qubits = Wires(list(range(n_ctrl_qubits, n_rsv_qubits+n_ctrl_qubits))) # wires of the control qubits (i.e. number of qubits in the control)
    
#         self.all_wires = Wires([*self.ctrl_qubits,*self.rsv_qubits])
#         self.dev = qml.device("default.qubit", wires =self.all_wires)

#         self.trotter_steps = trotter_steps

#         self.K_coeffs = K_coeffs  # parameter of the reservoir (XY_coupling)
#         # print(K_coeffs)
        

        
#     def set_gate_reservoir(self):
#         """ RANDOMLY, FIXED PARAMS """
        
#         for i, rsv_qubit_i in enumerate(self.rsv_qubits):
#             for j, rsv_qubit_j in enumerate(self.rsv_qubits):
                
#                 if i != j and i < j:
#                     k = self.K_coeffs[i, j]
                    
                    
#                     #print(f"{i},{j}/ {rsv_qubit_i},{rsv_qubit_j} -> k: {k} ")
#                     #print(f"RESERVOIR wires: {[rsv_qubit_i, rsv_qubit_j]}")
#                     qml.IsingXY(k, wires=[rsv_qubit_i, rsv_qubit_j])
    
#     def set_gate_params(self, x_coeff,z_coeff,y_coeff, J_coeffs):
#         """ TRAINABLE PARAMS """
#         for r in self.rsv_qubits:
#             qml.RX(x_coeff, wires=r)
#             qml.RZ(z_coeff, wires=r)
#             qml.RY(y_coeff, wires=r)
#         for i,qubit_a in enumerate(self.rsv_qubits):
#             for j,qubit_b in enumerate(self.ctrl_qubits):
#                 #print(f"CONTROL wires: {[self.ctrl_qubits[j],self.rsv_qubits[i]]}")
#                 qml.IsingXY(J_coeffs[i * len(self.ctrl_qubits) + j], wires=[qubit_a, qubit_b])


class QuantumNetwork:

    def __init__(self, n_rsv_qubits, n_ctrl_qubits, K_coeffs,hZ, trotter_steps=1, static=False, bath_params=None):
        self.static = static
        self.ctrl_qubits = Wires(list(range(n_ctrl_qubits))) # wires of the reservoir qubits (i.e. number of qubits in the reservoir)
        self.rsv_qubits = Wires(list(range(n_ctrl_qubits, n_rsv_qubits+n_ctrl_qubits))) # wires of the control qubits (i.e. number of qubits in the control)
        self.all_wires = [*self.ctrl_qubits,*self.rsv_qubits]

        self.dev = qml.device("default.qubit", wires =self.all_wires) 
        self.trotter_steps = trotter_steps
        self.z_coeffs = hZ  # parameters of the reservoir (Z)
        #self.y_coeffs = y_coeffs  # parameter of the reservoir (XY_coupling)
        self.K_coeffs = K_coeffs  # parameter of the reservoir (XY_coupling)
       # print(K_coeffs)
        self.bath_params = bath_params
    def set_gate_reservoir(self):
        for i,r in enumerate(self.rsv_qubits):
            qml.RZ(self.z_coeffs[i], wires = r)
        
        for i, rsv_qubit_i in enumerate(self.rsv_qubits):
            for j, rsv_qubit_j in enumerate(self.rsv_qubits):
                
                if i != j and i < j:
                    k = self.K_coeffs[i, j]
                    
                    qml.IsingXY(k, wires=[rsv_qubit_i, rsv_qubit_j])
    
    def set_gate_params(self, x_coeff,y_coeff, J_coeffs):
        for r in self.rsv_qubits:
            qml.RX(x_coeff, wires=r)   
            qml.RY(y_coeff, wires=r)

        for i,qubit_a in enumerate(self.rsv_qubits):
            for j,qubit_b in enumerate(self.ctrl_qubits):
                #print(f"CONTROL wires: {[self.ctrl_qubits[j],self.rsv_qubits[i]]}")
                qml.IsingXY(J_coeffs[i * len(self.ctrl_qubits) + j], wires=[qubit_a, qubit_b])
    # def build_ham(self,x,y,z,J):
    #     terms = []
    #     coefs = []
    #     for i, rsv_qubit_i in enumerate(self.rsv_qubits):
    #         for j, rsv_qubit_j in enumerate(self.rsv_qubits):
                
    #             if i != j and i < j:
    #                 k = self.K_coeffs[i, j]
                    
    #                 qml.IsingXY(k, wires=[rsv_qubit_i, rsv_qubit_j])

def generate_circuit(qrc, trotter_steps):
    """Create a quantum circuit based on qrc and its trotter steps."""
    @qml.qnode(qrc.dev, interface='jax')
    def circuit(params, input_state):
        x_coeff, y_coeff = params[:2]
        J_coeffs = params[2:]
        qml.StatePrep(input_state, wires=qrc.ctrl_qubits)
        for _ in range(trotter_steps):
            qrc.set_gate_reservoir()
            qrc.set_gate_params(x_coeff, y_coeff, J_coeffs)
        return qml.density_matrix(wires=qrc.ctrl_qubits)
    return circuit
def draw_circuits(param_batch, input_state, qrc, trotter_steps):
    """Draw the circuits for all parameter sets in the batch."""
    for test_num, params in enumerate(param_batch):
        print(f"Drawing circuit for parameter set {test_num + 1} of {len(param_batch)}:")
        circuit = generate_circuit(qrc, trotter_steps)
        print(qml.draw(circuit)(params, input_state))

def generate_batch_params(params_key, tests, trotter_steps, N_ctrl, N_reserv, sample_range):
    param_batch = []
    
    # Base seed for reproducibility
    # base_seed = 1987 * N_reserv + 123 * trotter_steps
    # params_key = jax.random.PRNGKey(base_seed)
    
    # Iterate over the test numbers
    for test_num in tests:
        # Fold in the test number to get a unique seed for each test
        test_key = jax.random.fold_in(params_key, test_num)
        
        # Generate the parameters using normal distribution (or any other distribution)
        params = jax.random.uniform(test_key, shape=(2 + (N_ctrl * N_reserv) * trotter_steps,), minval= -sample_range,maxval = sample_range)
        
        
        # Append parameters to the batch
        param_batch.append(params)
    
    # Stack the batch into a single array for easy use
    return jax.numpy.stack(param_batch)



def vectorize(matrix):
    """Vectorize a matrix by stacking its columns."""
    return jnp.ravel(matrix, order='F')
def generate_dataset(N_ctrl,N_reserv, training_size, key):
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
        state_vec = random_statevector(2**N_ctrl, seed=seed_value).data
        X.append(np.asarray(state_vec, dtype=jnp.complex128))
    
    
    # X = np.stack(X)
    # qubits = Wires(list(range(N_ctrl)))
    # dev_data = qml.device('default.qubit', wires=qubits)
    # circuit = qml.QNode(quantum_fun, device=dev_data, interface='jax')
    
    L = np.stack(X)
    
    
    return L


def main():
    N_ctrl = 1

    baths = [False]
    num_bath = 0
    number_of_fixed_param_tests = 20
    number_trainable_params_tests = 50
    static = False
    
    
    entangle = False
    delta_x=1.49011612e-08

    
    reservoirs = [1,2,3,4,5,6,7]
    # reservoirs = [2,3,4]
    # trots =[1, 10, 15,16,17,18,19, 20, 22, 25,27, 30,35]
    # trots = [10,12]
    # trots = np.arange(1,30,1)
    # trots = [1,2,3,4,5,6,7,8,9,10,11,12]
    trots = [1]

    #trots = [1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
    # trots = [20,22,25,27,30,32,35,37,40]
    # trots = [1,4,6, 8, 10, 12, 14, 16, 18, 20, 22, 24,26,28]
    # trots = [12]
    batch=True
   
    kFactor = 1.
    sample_range_label = "pi"
    sample_range = np.pi
    num_input_states = 50
    base_state = f'L_{num_input_states}'
    folder = f'./QFIM_traced_final_results/gate_model_hZ_DQFIM/Nc_{N_ctrl}/{base_state}/{kFactor}xK/2pi_E_0/'
    for trotter_steps in trots:
        for n_rsv_qubits in reservoirs:
            N_reserv = n_rsv_qubits
        

       # ./QFIM_traced_trainable_global/gate_model/Nc_2/basis_state/Nr_1/trotter_step_32/data.pickle

            #if n_rsv_qubits >3 and trotter_steps % 2== 0:
                #continue
            for bath in baths:
                K_0 = kFactor
                print("________________________________________________________________________________")
                print(f"N_ctrl = {N_ctrl}; N_R: {n_rsv_qubits}, trotter_steps: {trotter_steps}")
                folder_gate = os.path.join(folder, f'Nr_{N_reserv}', f'trotter_step_{trotter_steps}',f'L_{num_input_states}')
                
                print("\nfolder_gate: ",folder_gate)
                Path(folder_gate).mkdir(parents=True, exist_ok=True)
                N = n_rsv_qubits+N_ctrl
                num_J = n_rsv_qubits*N_ctrl
                # data_filename = os.path.join(folder_gate, f'data2.pickle')
                data_filename = os.path.join(folder_gate, f'data_{sample_range_label}_range.pickle')
                base_key_seed = 123* n_rsv_qubits + 12345 * trotter_steps *n_rsv_qubits
                params_key = jax.random.PRNGKey(base_key_seed)
                params_subkey0, params_subkey1 = jax.random.split(params_key, 2)
                            
                        
                
                L = generate_dataset(N_ctrl,N_reserv,num_input_states, params_subkey0)
                print(data_filename)
                all_tests_data = {}
                #print("data_filename: ",data_filename)
                if is_non_empty_file(data_filename):
                    with open(data_filename, 'rb') as f:
                        all_tests_data = pickle.load(f)
                else:
                    all_tests_data = {}
                number_fixed_tests_completed = len(all_tests_data.keys())
                
                for fixed_param_num in range(0,number_of_fixed_param_tests):
                    fixed_param_seed = base_key_seed * fixed_param_num + base_key_seed *1234
                    fixed_PRNGKey = jax.random.PRNGKey(fixed_param_seed)
                    fized_params_key1, fized_params_key2 = jax.random.split(fixed_PRNGKey, 2)
                    new_tests_count = 0
                    fixed_param_key = f'fixed_params{fixed_param_num}'
                    all_tests_data.setdefault(fixed_param_key, {})
                    
                    number_trainable_tests_completed = len(all_tests_data[fixed_param_key].keys())
                    tests_to_run = number_trainable_params_tests - number_trainable_tests_completed
                    
                    K_half =   jax.random.uniform(fized_params_key1, shape=(N,N), minval=-kFactor, maxval=kFactor)
                    K = (K_half + K_half.T) / 2  # making the matrix symmetric
                    K = 2. * K - 1.
                    K_coeffs = K * K_0/2
                    E_0 = np.pi*2
                    hZ = jax.random.uniform(fized_params_key2, (N_reserv,), minval = -0.5,maxval=0.5)
                    hZ = E_0*hZ
                    # print(f"hZ: {hZ}, K_coeffs: {K_coeffs}")
                    qrc = QuantumNetwork(n_rsv_qubits=n_rsv_qubits, n_ctrl_qubits=N_ctrl, K_coeffs=K_coeffs,hZ=hZ,trotter_steps=trotter_steps, static=static)
                    ctrl_qubits = qrc.ctrl_qubits # wires of the reservoir qubits (i.e. number of qubits in the reservoir)
                    rsv_qubits = qrc.rsv_qubits# wires of the control qubits (i.e. number of qubits in the control)
                    all_wires = qrc.all_wires
                    dev =qrc.dev
                    if tests_to_run > 0:
                        print(f"Number of tests to run for {fixed_param_key}: ", tests_to_run)
                        
                        
                        



                        @jax.jit
                        @qml.qnode(dev,interface='jax',diff_method="backprop")
                        def circuit(params,input_state):
                            x_coeff = params[0]
                            y_coeff = params[1]
                            J_coeffs = params[2:]

                    
                            qml.StatePrep(input_state, wires=[*qrc.ctrl_qubits])
                            
                            for i in range(trotter_steps):
                                qrc.set_gate_reservoir()

                                step = len(rsv_qubits)*len(ctrl_qubits)
                                qrc.set_gate_params(x_coeff,y_coeff, J_coeffs[i*step:(i+1)*step])
                            return qml.density_matrix(wires=[*ctrl_qubits])

                        
                        
                        jit_circuit = jax.jit(circuit)
                        

                        def get_density_matrix_sum(params, input_states, jit_circuit):
                            """
                            Computes the sum of density matrices after applying the PQC on all training states using a pre-jitted circuit.
                            
                            input_states: A batch of training states .
                            jit_circuit: A pre-jitted version of the quantum circuit.
                            """
                            
                            # Initialize a variable to store the sum of the density matrices
                            density_matrix_sum = jnp.zeros_like(jit_circuit(params, input_states[0]))
                            entropies = []
                            # network_entropies = []
                            # Loop over each input state and sum the density matrices
                            for input_state in input_states:
                                out = jit_circuit(params, input_state)
                                
                                
                                entropy = vn_entropy(out, indices=[*qrc.ctrl_qubits])
                                entropies.append(entropy)
                                

                                # network_entropies.append(vn_entanglement_entropy(out_full, indices0 = [sim_qr.ctrl_qubits], indices1=[sim_qr.rsv_qubits]))
                                density_matrix_sum += jit_circuit(params, input_state)
                            
                            # Return the averaged density matrix (Π_L)
                            return jnp.array(entropies), density_matrix_sum / len(input_states)
                        # Function to compute the gradient of the circuit output with respect to each parameter separately
                        def get_partial_grads(params, input_states,jit_circuit, delta_x=1.49011612e-08):
                            """
                            Computes the averaged gradients of the PQC output density matrix 
                            with respect to each parameter for all training states using the parameter-shift rule.
                            
                            input_states: A batch of training states (|\psi_l\rangle).
                            delta_x: The shift for the parameter-shift rule.
                            """
                            
                            all_res = []

                            
                            def shift_circuit(params, idx, input_state):
                                # Shift parameter up
                                shifted_params_plus = params.at[idx].set(params[idx] + delta_x)
                                # Shift parameter down
                                shifted_params_minus = params.at[idx].set(params[idx] - delta_x)

                                # Evaluate the circuit with the shifted parameters
                                shifted_plus_circuit = jit_circuit(shifted_params_plus, input_state)
                                shifted_minus_circuit = jit_circuit(shifted_params_minus, input_state)
                                
                                # Calculate the gradient using the parameter-shift rule
                                # grad = (shifted_plus_circuit - shifted_minus_circuit) / (delta_x)
                                grad = (shifted_plus_circuit - shifted_minus_circuit) / (2 * delta_x)
                                
                                return grad

                            # Initialize a variable to store the sum of the gradients for each parameter
                            for idx in range(len(params)):
                                grad_sum = jnp.zeros_like(jit_circuit(params, input_states[0]))  # Initialize to zero matrix
                                
                                # Loop over all training states to compute and sum the gradients
                                for input_state in input_states:
                                    # Compute the gradient for this training state using the shift rule
                                    grad = shift_circuit(params, idx, input_state)
                                    grad_sum += grad
                                
                                # Average the gradient over all the training states
                                avg_grad = grad_sum / len(input_states)
                                all_res.append(avg_grad)
                            
                            return jnp.asarray(all_res)
                                                
                        def compute_qfim_eigval_decomp(params):
                            
                            density_matrix_grads = get_partial_grads(params, L, jit_circuit)
                            entropies,Pi_L = get_density_matrix_sum(params, L, jit_circuit)
                            # print(f"Pi_L: {Pi_L}")
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
                            # print(f"\nfinal eigvals: {eigvals}\n")
                            return eigvals, eigvecs, QFIM, entropies
                        param_batch = generate_batch_params(params_subkey1,np.arange(number_trainable_tests_completed,number_trainable_params_tests), trotter_steps, N_ctrl, n_rsv_qubits,sample_range)

                        
                        if batch:

                            s = time.time()
                            batch_results = jax.vmap(compute_qfim_eigval_decomp)(param_batch)
                            # batch_results = jax.vmap(get_qfi_eigvals, in_axes=(0,))(param_batch)
                            e = time.time()
                            print(f'Time for param set: {e-s}')

                            # Restructuring results
                            restructured_results = [(batch_results[0][i], batch_results[1][i], batch_results[2][i], batch_results[3][i]) for i in range(number_trainable_params_tests)]

                            # Storing data
                            for test_num, (eigvals, eigvecs, qfim, entropies) in enumerate(restructured_results):
                                test_key = f'test{test_num + number_trainable_tests_completed}'
                                if jnp.any(jnp.isnan(qfim)) or jnp.any(jnp.isinf(qfim)):
                                    raise ValueError("qfim contains NaNs or infinite values")
                                error_threshold = -1e-12
                                if jnp.any(eigvals < error_threshold):
                                    # Print the params and fixed params if negative eigenvalues are found
                                    print(f"Negative eigenvalue detected for test {test_key}\neigvals: {eigvals}")
                                    param_str = ', '.join([str(p) for p in params])
                                    fixed_param_str = ', '.join([f'N_ctrl: {N_ctrl}', f'N_reserv: {n_rsv_qubits}', 
                                                                f'trots: {trotter_steps}', f'Kfactor: {kFactor}',  f'hZ: {hZ}', 
                                                                f'num_bath: {num_bath}'])
                                    print(f"Trainable params: {param_str}")
                                    # print(f"Fixed params: {fixed_params}")
                                    raise ValueError("qfim contains negative eigenvalues")
                                params = param_batch[test_num]
                                # original_circuit = jit_circuit(params)
                                # original_circuit = original_circuit / np.trace(original_circuit)
                                # entropy = vn_entropy(original_circuit, indices=[*ctrl_qubits])
                                all_tests_data[fixed_param_key][test_key] = {
                                    'L': L,
                                    'entropies': entropies,
                                    'K_coeffs': K_coeffs,
                                    'hZ': hZ,
                                    'E_0':E_0,
                                    'K_0':K_0,
                                    'qfim_eigvals': eigvals,
                                    'qfim_eigvecs': eigvecs,
                                    'trainable_params': params,
                                    'n_reserv': n_rsv_qubits,
                                    'qfim': qfim,
                                    'n_ctrl': N_ctrl,
                                    'n_reserv': n_rsv_qubits,
                                    'time_steps': trotter_steps,
                                    'num_bath': num_bath
                                }
                                new_tests_count += 1
                        else:
                            # Non-batch processing, run each test one by one
                            # print("Running tests one by one...")
                            # draw_circuits(param_batch, L[0],qrc,trotter_steps)
                            for test_num, params in enumerate(param_batch):
                                # print(f"Running test {test_num + 1} of {len(param_batch)}")
                                eigvals, eigvecs, qfim, entropies = compute_qfim_eigval_decomp(params)
                                test_key = f'test{test_num + number_trainable_tests_completed}'
                                
                                if jnp.any(jnp.isnan(qfim)) or jnp.any(jnp.isinf(qfim)):
                                    raise ValueError("QFIM contains NaNs or infinite values")
                                error_threshold = -1e-12
                                if jnp.any(eigvals < error_threshold):
                                    print(f"Negative eigenvalue detected for test {test_key}, eigvals: {eigvals}")
                                    raise ValueError("QFIM contains negative eigenvalues")
                                print(f"Tr(eigvals): {sum(eigvals):.2e}, Var: {jnp.var(eigvals):.2e}\n")
                                # Store results individually
                                all_tests_data[fixed_param_key][test_key] = {
                                    'L': L,
                                    'entropies': entropies,
                                    'K_coeffs': K_coeffs,
                                    'hZ': hZ,
                                    'E_0':E_0,
                                    'K_0':K_0,
                                    'qfim_eigvals': eigvals,
                                    'qfim_eigvecs': eigvecs,
                                    'trainable_params': params,
                                    'n_reserv': n_rsv_qubits,
                                    'qfim': qfim,
                                    'n_ctrl': N_ctrl,
                                    'n_reserv': n_rsv_qubits,
                                    'time_steps': trotter_steps,
                                
                                    'num_bath': num_bath
                                }
                                new_tests_count += 1
                        with open(data_filename, 'wb') as f:
                            pickle.dump(all_tests_data, f)
                        
                        #print(f"Updated and added {new_tests_count} tests")
                    #else:
                        #print(f"{number_trainable_params_tests} completed ({tests_to_run} tests to run) for {fixed_param_key}. Continue..")    

if __name__ == '__main__':
    main()