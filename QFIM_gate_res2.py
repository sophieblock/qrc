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
# from qutip import *
# from qutip.qip.operations import cnot,rz,rx,ry,snot
# from qutip.qip.circuit import QubitCircuit
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



class QuantumReservoirGate:

    def __init__(self, n_rsv_qubits, n_ctrl_qubits, K_coeffs,trotter_steps=1, static=False, bath_params=False,num_bath = 0):
        self.static = static
        self.bath_params = bath_params
        self.num_bath = num_bath
        self.ctrl_qubits = Wires(list(range(n_ctrl_qubits))) # wires of the reservoir qubits (i.e. number of qubits in the reservoir)
        self.rsv_qubits = Wires(list(range(n_ctrl_qubits, n_rsv_qubits+n_ctrl_qubits))) # wires of the control qubits (i.e. number of qubits in the control)
    
        self.all_wires = Wires([*self.ctrl_qubits,*self.rsv_qubits])
        self.dev = qml.device("default.qubit", wires =self.all_wires)

        self.trotter_steps = trotter_steps

        self.K_coeffs = K_coeffs  # parameter of the reservoir (XY_coupling)
        # print(K_coeffs)
        

        
    def set_gate_reservoir(self,x_coeff,z_coeff,y_coeff):
        
        for r in self.rsv_qubits:
            qml.RX(x_coeff, wires=r)
            qml.RZ(z_coeff, wires=r)
            qml.RY(y_coeff, wires=r)
        for i, rsv_qubit_i in enumerate(self.rsv_qubits):
            for j, rsv_qubit_j in enumerate(self.rsv_qubits):
                
                if i != j and i < j:
                    k = self.K_coeffs[i, j]
                    
                    
                    #print(f"{i},{j}/ {rsv_qubit_i},{rsv_qubit_j} -> k: {k} ")
                    #print(f"RESERVOIR wires: {[rsv_qubit_i, rsv_qubit_j]}")
                    qml.IsingXY(k, wires=[rsv_qubit_i, rsv_qubit_j])
    def set_gate_params(self, J_coeffs):
        
        for i,qubit_a in enumerate(self.rsv_qubits):
            for j,qubit_b in enumerate(self.ctrl_qubits):
                #print(f"CONTROL wires: {[self.ctrl_qubits[j],self.rsv_qubits[i]]}")
                qml.IsingXY(J_coeffs[i * len(self.ctrl_qubits) + j], wires=[qubit_a, qubit_b])
    

def generate_batch_params(params_key, tests, trotter_steps, N_ctrl, N_reserv,sample_range):
    param_batch = []
    for test_num in tests:
        
        test_key = jax.random.fold_in(params_key, test_num)

        
       
        
        # params = jax.random.uniform(test_key,shape=(3 + (N_ctrl * N_reserv) * trotter_steps,), minval=0, maxval=sample_range)
        # params = jax.random.uniform(test_key,shape=(3 + (N_ctrl * N_reserv) * trotter_steps,), minval=-sample_range, maxval=sample_range)
        params = jax.random.truncated_normal(
            test_key,
            shape=(3 + (N_ctrl * N_reserv) * trotter_steps,),
            lower=-sample_range,
            upper=sample_range
        )

        param_batch.append(params)
    return jax.numpy.stack(param_batch)


def vectorize(matrix):
    """Vectorize a matrix by stacking its columns."""
    return jnp.ravel(matrix, order='F')

def main():
    N_ctrl = 1

    baths = [False]
    num_bath = 0
    number_of_fixed_param_tests_base = 1
    number_trainable_params_tests = 200
    static = False
    base_state = 'GHZ_state'
    # base_state = 'basis_state'

    delta_x=1.49011612e-08

    
    reservoirs = [1,2]
    # reservoirs=[1,2]
    
    # trots = [1,5,10,15,20,25,30,35,40,45,50]
    trots = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
    # trots = [1,4,8,12,13,14,15,16,17,18,19,20,24,28,30,32,36,40]
    trots = [23,24,25]
    # trots = trots[::-1]
    trots = [1,2,3,4,5,6,7,8,9,10]
    # trots = [12]

    batch=True
    
    kFactor = 1
    K_0 = kFactor
    threshold=1e-12
    # sample_range = 2*np.pi
    # sample_range_label = 'positive_pi'
    sample_range = np.pi
    sample_range_label = 'normal_2pi'
    if base_state == 'basis_state':
        folder = f'./QFIM_results/gate_basis/Nc_{N_ctrl}/sample_{sample_range_label}/{kFactor}bbxK/'
    else:
        folder = f'./QFIM_results/gate/Nc_{N_ctrl}/sample_{sample_range_label}/{kFactor}bbxK/'
    for trotter_steps in trots:
        
        for n_rsv_qubits in reservoirs:
            if n_rsv_qubits == 1 or n_rsv_qubits == 2:
                number_of_fixed_param_tests = 1
            else:
                number_of_fixed_param_tests = number_of_fixed_param_tests_base


            for bath in baths:
                print("________________________________________________________________________________")
                print(f"N_ctrl = {N_ctrl}; N_R: {n_rsv_qubits}, trotter_steps: {trotter_steps}")
                folder_gate = os.path.join(folder, f'Nr_{n_rsv_qubits}', f'trotter_step_{trotter_steps}')
                print("\nfolder_gate: ",folder_gate)
                Path(folder_gate).mkdir(parents=True, exist_ok=True)
                N = n_rsv_qubits+N_ctrl
                num_J = n_rsv_qubits*N_ctrl
                data_filename = os.path.join(folder_gate, 'data.pickle')


                base_key_seed = 123* n_rsv_qubits + 12345 * trotter_steps *n_rsv_qubits
                params_key = jax.random.PRNGKey(base_key_seed)
                params_subkey0, params_subkey1 = jax.random.split(params_key, 2)
                    
                all_tests_data = {}
                #print("data_filename: ",data_filename)
                if is_non_empty_file(data_filename):
                    with open(data_filename, 'rb') as f:
                        all_tests_data = pickle.load(f)
                else:
                    all_tests_data = {}
                number_fixed_tests_completed = len(all_tests_data.keys())
                # If N_R==1 or 2, we only ever want one fixed_params0 entry
                if n_rsv_qubits < 3 and number_fixed_tests_completed > 1:
                    print(f"N_C = {N_ctrl}, N_R = {n_rsv_qubits} → pruning {number_fixed_tests_completed-1} extra fixed_params")
                    # Keep only the first batch
                    fixed_params0_dict = all_tests_data['fixed_params0']
                    all_tests_data = {'fixed_params0': fixed_params0_dict}
                    number_fixed_tests_completed = 1

                    # Overwrite the pickle so on disk it's also pruned
                    with open(data_filename, 'wb') as f:
                        pickle.dump(all_tests_data, f)
                    print("  → data.pickle updated to contain only fixed_params0")
                for fixed_param_num in range(0,number_of_fixed_param_tests):
                    
                

                    new_tests_count = 0
                    fixed_param_key = f'fixed_params{fixed_param_num}'
                    all_tests_data.setdefault(fixed_param_key, {})
                    
                    number_trainable_tests_completed = len(all_tests_data[fixed_param_key].keys())
                    tests_to_run = number_trainable_params_tests - number_trainable_tests_completed
                    
                    if tests_to_run > 0:
                        print(f"Number of tests to run for {fixed_param_key}: ", tests_to_run, f" Completed: {number_trainable_tests_completed}")
                        
                        base_key_seed = fixed_param_num*123 + fixed_param_num*1000 * trotter_steps
                        params_key = jax.random.PRNGKey(base_key_seed)
                            
                        
                            
                        K_half =  jax.random.normal(params_key, shape=(N,N))
                        K = (K_half + K_half.T) / 2  # making the matrix symmetric
                        K = 2. * K - 1.
                        K_coeffs = K * K_0/2
                        qrc = QuantumReservoirGate(n_rsv_qubits=n_rsv_qubits, n_ctrl_qubits=N_ctrl, K_coeffs=K_coeffs,trotter_steps=trotter_steps, static=static)
                        ctrl_qubits = qrc.ctrl_qubits # wires of the reservoir qubits (i.e. number of qubits in the reservoir)
                        rsv_qubits = qrc.rsv_qubits# wires of the control qubits (i.e. number of qubits in the control)
                        all_wires = qrc.all_wires
                        dev =qrc.dev


                        test_state = create_initial_state(N_ctrl,base_state)
                        
                        
                        @qml.qnode(dev,interface='jax')
                        def circuit(params):
                            x_coeff = params[0]
                            z_coeff = params[1]
                            y_coeff = params[2]
                            J_coeffs = params[3:]

                            #print(J_coeffs)
                     
                            qml.StatePrep(test_state, wires=[*qrc.ctrl_qubits])
                            qrc.set_gate_reservoir(x_coeff,z_coeff,y_coeff)
                            for i in range(trotter_steps):
                                # qrc.set_gate_reservoir(x_coeff,z_coeff,y_coeff)
                                step = len(rsv_qubits)*len(ctrl_qubits)
                                qrc.set_gate_params(J_coeffs[i*step:(i+1)*step])
                                qrc.set_gate_reservoir(x_coeff,z_coeff,y_coeff)
                            return qml.density_matrix(wires=[*ctrl_qubits])

                        

                        jit_circuit = jax.jit(circuit)
                        def single_param_grad(idx, params,original_circuit, delta_x):
                            # Shift parameter up
                            shifted_params_plus = params.at[idx].set(params[idx] + delta_x)
                            shifted_plus_circuit = jit_circuit(shifted_params_plus)
                            shifted_plus_circuit = shifted_plus_circuit/np.trace(shifted_plus_circuit)
                            
                            # Calculate the gradient as the symmetric difference
                            grad = (shifted_plus_circuit - original_circuit) / (delta_x)
                            
                            return grad

                        
                        def get_unitary_grad(params):
                            n_params = len(params)

                            # Compute the original circuit state
                            original_circuit = jit_circuit(params)
                            original_circuit = original_circuit/np.trace(original_circuit)
                            # Vectorize the single_param_grad function over all parameters
                            grad_fn_vmap = jax.vmap(single_param_grad, in_axes=(0, None, None,None))
                            grad_list = grad_fn_vmap(jax.numpy.arange(n_params), params,original_circuit, delta_x)

                            return grad_list, original_circuit
                        def compute_qfim_eigval_decomp(params):
                            
                            density_matrix_grads, rho = get_unitary_grad(params)
                            rho = rho /jnp.trace(rho)
                           

                            # Eigenvalue decomposition
                            eigvals, eigvecs = jnp.linalg.eigh(rho)
                            n_params = len(density_matrix_grads)
                            eye = jnp.eye(rho.shape[0])
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
                            entropy = vn_entropy(rho, indices=[*ctrl_qubits])
                            return eigvals, eigvecs, QFIM,entropy
                       

                        test_indices = np.arange(number_trainable_tests_completed,number_trainable_params_tests)
                        param_batch = generate_batch_params(params_subkey0,test_indices, trotter_steps, N_ctrl, n_rsv_qubits,sample_range)

     
                        
                        if batch:

                            # s = time.time()

                            batch_results = jax.vmap(compute_qfim_eigval_decomp)(param_batch)
                            
                            # batch_results = jax.vmap(get_qfi_eigvals, in_axes=(0,))(param_batch)
                            # e = time.time()
                            # print(f'Time for param set: {e-s}')

                            # Restructuring results
                            restructured_results = [(batch_results[0][i], batch_results[1][i], batch_results[2][i], batch_results[3][i]) for i in range(len(param_batch))]

                            # Storing data
                            for test_num, (eigvals, eigvecs, qfim,entropy) in enumerate(restructured_results):
                                test_key = f'test{test_num + number_trainable_tests_completed}'
                                if jnp.any(jnp.isnan(qfim)) or jnp.any(jnp.isinf(qfim)):
                                    raise ValueError("qfim contains NaNs or infinite values")
                                error_threshold = -1e-10
                                if jnp.any(eigvals < error_threshold):
                                    # Print the params and fixed params if negative eigenvalues are found
                                    print(f"Negative eigenvalue detected for test {test_key}\neigvals: {eigvals}")
                                    param_str = ', '.join([str(p) for p in params])
                                    fixed_param_str = ', '.join([f'N_ctrl: {N_ctrl}', f'N_reserv: {n_rsv_qubits}', 
                                                                f'trots: {trotter_steps}', f'Kfactor: {kFactor}', 
                                                                f'num_bath: {num_bath}'])
                                    print(f"Trainable params: {param_str}")
                                    # print(f"Fixed params: {fixed_params}")
                                    raise ValueError("qfim contains negative eigenvalues")
                                params = param_batch[test_num]
                                eigvals = np.where(eigvals<0,0.0,eigvals)
                                
                                
                                all_tests_data[fixed_param_key][test_key] = {
                                    'base_state': base_state,
                                    'entropy': entropy,
                                    'K_coeffs': K_coeffs,
                                    'qfim_eigvals': eigvals,
                                    'qfim_eigvecs': eigvecs,
                                    'trainable_params': params,
                                    'n_reserv': n_rsv_qubits,
                                    'qfim': qfim,
                                    'n_ctrl': N_ctrl,
                                    'time_steps': trotter_steps,
                                    'sample_range':sample_range,
                                
                                    'num_bath': num_bath,
                                    'base_state':base_state
                                }
                                new_tests_count += 1
                        else:
                            # Non-batch processing, run each test one by one
                            print("Running tests one by one...")
                            for test_num, params in enumerate(param_batch):
                                # print(f"Running test {test_num + 1} of {len(param_batch)}")
                                eigvals, eigvecs, qfim,entropy = compute_qfim_eigval_decomp(params)
                                test_key = f'test{test_num+number_trainable_tests_completed}'
                                nonzero_eigvals = eigvals[eigvals > threshold]
                                if len(nonzero_eigvals) > 17:
                                    print(f"FP: {fixed_param_num} {test_key} reached rank: {nonzero_eigvals}")
                                if jnp.any(jnp.isnan(qfim)) or jnp.any(jnp.isinf(qfim)):
                                    raise ValueError("QFIM contains NaNs or infinite values")
                                
                                if jnp.any(jnp.isnan(qfim)) or jnp.any(jnp.isinf(qfim)):
                                    raise ValueError("QFIM contains NaNs or infinite values")
                                error_threshold = -1e-10
                                if jnp.any(eigvals < error_threshold):
                                    print(f"Negative eigenvalue detected for test {test_key}, eigvals: {eigvals}")
                                    raise ValueError("QFIM contains negative eigenvalues")
                                eigvals = np.where(eigvals<0,0.0,eigvals)
                                # Store results individually
                                all_tests_data[fixed_param_key][test_key] = {
                                    'base_state': base_state,
                                    'entropy': entropy,
                                    'K_coeffs': K_coeffs,
                                    'qfim_eigvals': eigvals,
                                    'qfim_eigvecs': eigvecs,
                                    'trainable_params': params,
                                    'n_reserv': n_rsv_qubits,
                                    'qfim': qfim,
                                    'n_ctrl': N_ctrl,
                                    'time_steps': trotter_steps,
                                    'sample_range':sample_range,
                                    'num_bath': num_bath,
                                    'base_state':base_state
                                }
                                new_tests_count += 1
                                
                        with open(data_filename, 'wb') as f:
                            pickle.dump(all_tests_data, f)
                        
                        print(f"Updated and added {new_tests_count} tests")
                    else:
                        print(f"{number_trainable_params_tests} completed ({tests_to_run} tests to run) for {fixed_param_key}. Continue..")    
        # number_trainable_params_tests += 50
if __name__ == '__main__':
    main()