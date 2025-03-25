import pennylane as qml
from pennylane.math import vn_entropy
import os
import pickle
import re
import pickle
from pathlib import Path

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from qiskit import *
from jax import numpy as np
from scipy import stats
import sympy
import matplotlib.pyplot as plt
import base64
from jax import numpy as jnp
import pickle

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

#from pennylane.pulse import ParametrizedEvolution, ParametrizedHamiltonian,HardwareHamiltonian
from jax.experimental.ode import odeint
from pennylane.devices.qubit.apply_operation import _evolve_state_vector_under_parametrized_evolution,apply_parametrized_evolution
has_jax = True
diable_jit = False
config.update('jax_disable_jit', diable_jit)
#config.parse_flags_with_absl()
config.update("jax_enable_x64", True)
os.environ['JAX_TRACEBACK_FILTERING'] = 'off'

def compute_initial_learning_rate(gradients, scale_factor=0.1, min_lr=1e-3, max_lr = 0.1):
    """
    Compute an initial learning rate based on the norm of gradients.
    """
    # Compute the norm of the gradients
    
    norm_grad = jnp.linalg.norm(gradients)
    min_abs_grad = jnp.min(jnp.abs(gradients))
    #mean_norm_grad = jnp.mean(norm_grad)
    initial_lr = scale_factor / (norm_grad + 1e-10)  # Adding a small value to prevent division by zero
    print(norm_grad, initial_lr, initial_lr / (min_abs_grad * 10))
    #initial_lr =initial_lr / (min_abs_grad * 10)
    initial_lr = jnp.clip(initial_lr, min_lr, max_lr)
    return initial_lr

def get_qfim_stats_basis_state(file_path,fixed_param_dict_key, trainable_param_dict_key):
    file_path = Path(file_path) / f'data.pickle'
    # print(file_path)
    with open(file_path, 'rb') as f:
        all_tests_data = pickle.load(f)
    results = all_tests_data[fixed_param_dict_key][trainable_param_dict_key]

    assert 'qfim_eigvals' in results
    # print(results.keys())
    eigvals = results.get('qfim_eigvals', None)
    nonzero = eigvals[eigvals > threshold]
    return {
        'qfim':results.get('qfim', None),
        'qfim_eigvals':eigvals,
        'raw_trace':np.sum(eigvals),
        'raw_var_nonzero':  np.var(nonzero),
        'entropy':results.get('entropy', None),
        'trainable_params':results.get('trainable_params', None),
    }

def get_qfim_eigvals(file_path, fixed_param_dict_key, trainable_param_dict_key):
    """
    Load data from a pickle file and return QFIM eigenvalues for the given fixed and trainable parameter dictionary keys.

    Parameters:
    - file_path: str or Path, the path to the pickle file.
    - fixed_param_dict_key: str, the key for the fixed parameters dictionary.
    - trainable_param_dict_key: str, the key for the trainable parameters dictionary.

    Returns:
    - qfim_eigvals: list of QFIM eigenvalues.
    """
    file_path = Path(file_path) / f'data.pickle'
    # Ensure file_path is a Path object
    file_path = Path(file_path) if not isinstance(file_path, Path) else file_path
    
    if not file_path.exists():
        print(f"File {file_path} does not exist.")
        return None

    with open(file_path, 'rb') as f:
        df = pickle.load(f)
    
    # Initialize variables
    qfim_eigvals = None
    test_data = df[fixed_param_dict_key]
    results = test_data[trainable_param_dict_key]

    if 'qfim_eigvals' in results:
        # print(results.keys())
        qfim_eigvals = results['qfim_eigvals']
        qfim = results['qfim']
        K_coeffs = results['K_coeffs']
        fixed_params = {'K_coef':K_coeffs}
        params = results['trainable_params']
        entropy = results['entropy']

        return qfim_eigvals,fixed_params,params,qfim, entropy
    
    print("QFIM eigenvalues not found for the given parameter keys.")
    return None,None,None, None



def get_dqfim_stats(file_path,fixed_param_dict_key, trainable_param_dict_key, num_L = 1,threshold=1e-10):
    file_path = Path(file_path) / f'L_{num_L}/data.pickle'
    # print(file_path)
    with open(file_path, 'rb') as f:
        all_tests_data = pickle.load(f)
    results = all_tests_data[fixed_param_dict_key][trainable_param_dict_key]

    assert 'qfim_eigvals' in results
    # print(results.keys())
    eigvals = results.get('qfim_eigvals', None)
    nonzero = eigvals[eigvals > threshold]
    return {
        'dqfim_eigvals':results.get('qfim_eigvals', None),
        'dqfim':results.get('qfim', None),
        'L':results.get('L',None),
        'raw_trace':np.sum(eigvals),
        'raw_var_nonzero':  np.var(nonzero),
        'entropies':results.get('entropies', None),
        'trainable_params':results.get('trainable_params', None),
    }

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






def get_initial_learning_rate(grads, scale_factor=0.01, min_lr=1e-4, max_lr=0.2):
    """Estimate a more practical initial learning rate based on the gradient norms."""
    grad_norm = jnp.linalg.norm(grads)
    
    initial_lr = jnp.where(grad_norm > 0, scale_factor / grad_norm, 0.1)
    
    clipped_lr = jnp.clip(initial_lr, min_lr, max_lr)
    print(f"initial base lr: {initial_lr:.5f}, clipped: {clipped_lr:.5f}")
    return initial_lr, clipped_lr
def get_initial_lr_per_param(grads, base_step=0.001, min_lr=1e-5, max_lr=0.2):
    # print(f"grads: {grads}")
    grad_magnitudes = jax.tree_util.tree_map(lambda g: jnp.abs(g) + 1e-12, grads)
    # print(f"grad_magnitudes: {grad_magnitudes}")
    lr_tree = jax.tree_util.tree_map(lambda g: base_step / g, grad_magnitudes)
    # print(f"lr_tree: {lr_tree}")
    lr_tree = jax.tree_util.tree_map(lambda lr: jnp.clip(lr, min_lr, max_lr), lr_tree)
    return lr_tree
def get_initial_learning_rate_DQFIM(params, qrc, X, gate, init_grads, scale_factor=0.1, min_lr=9e-5, max_lr=0.1, target_DQFIM=False, compute_both=False, threshold = 1.e-10):
    """
    Compute an initial learning rate based on the quantum natural gradient using the DQFIM.
    
    If compute_both is True, then both the training (input) and target DQFIM stats are computed 
    using the same compiled circuit. Otherwise, the function computes the DQFIM stats for either the 
    training states (if target_DQFIM is False) or the target states (if target_DQFIM is True).
    """
    parameterized_ham = qrc.get_total_hamiltonian_components() # wires of the reservoir qubits (i.e. number of qubits in the reservoir)
    
    grad_norm = jnp.linalg.norm(init_grads)
    # print(f"grad_norm: {grad_norm} max_grad: {max(np.abs(init_grads))}")

    ctrl_wires = qrc.get_ctrl_wires()
    qnode_dev = qrc.get_dev()
    time_steps = qrc.steps
    num_J = qrc.num_J

    dev_data = qml.device('default.qubit', wires=ctrl_wires)
    def get_target_state(gate, input_state, qubits):
        '''
        Apply the gate to the input state and return the output state.
        '''
        qml.StatePrep(input_state, wires=[*qubits])
        gate(wires=qubits)
        return qml.state()
    target_generator = qml.QNode(get_target_state, device=dev_data, interface='jax')
    
    # Helper to select states:
    def get_L(use_target):
        if use_target:
            return np.stack([np.asarray(target_generator(gate, x, ctrl_wires)) for x in X])
        else:
            return X
   

    @jax.jit
    @qml.qnode(qnode_dev, interface="jax")
    def circuit(params, input_state):
        
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
    jit_circuit = jax.jit(circuit)
        
    def get_density_matrix_sum(params, states):
        """
        Computes the sum of density matrices after applying the PQC on all training states using a pre-jitted circuit.
        
        input_states: A batch of training states (|\psi_l\rangle).
        jit_circuit: A pre-jitted version of the quantum circuit.
        """
        
        # Initialize a variable to store the sum of the density matrices
        density_matrix_sum = jnp.zeros_like(jit_circuit(params, states[0]))
        entropies = []
        # network_entropies = []
        # Loop over each input state and sum the density matrices
        for input_state in states:
            out = jit_circuit(params, input_state)
            entropy = vn_entropy(out, indices=[*qrc.ctrl_qubits])
            entropies.append(entropy)
            

            # network_entropies.append(vn_entanglement_entropy(out_full, indices0 = [sim_qr.ctrl_qubits], indices1=[sim_qr.rsv_qubits]))
            density_matrix_sum += out
        
        # Return the averaged density matrix (Π_L)
        return jnp.array(entropies), density_matrix_sum / len(states)
    # Function to compute the gradient of the circuit output with respect to each parameter separately
    def get_partial_grads(params, states, delta_x=1.49011612e-08):
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
            grad_sum = jnp.zeros_like(jit_circuit(params, states[0]))  # Initialize to zero matrix
            
            # Loop over all training states to compute and sum the gradients
            for input_state in states:
                # Compute the gradient for this training state using the shift rule
                grad = shift_circuit(params, idx, input_state)
                grad_sum += grad
            
            # Average the gradient over all the training states
            avg_grad = grad_sum / len(states)
            all_res.append(avg_grad)
        
        return jnp.asarray(all_res)
                            
    def compute_qfim_eigval_decomp(L_local):
        density_matrix_grads = get_partial_grads(params, L_local)
        entropies,Pi_L = get_density_matrix_sum(params, L_local)

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
        trace_dqfim = jnp.trace(QFIM)
        print(f"\nNew Tr(DQFIM): {trace_dqfim}")
        nonzero_eigvals = eigvals[eigvals > threshold]
        variance_nonzero = np.var(nonzero_eigvals)
        print(f"New Var(DQFIM): {variance_nonzero}")
        return QFIM, trace_dqfim, eigvals, eigvecs, entropies
        # return eigvals, eigvecs, QFIM, entropies,density_matrix_grads,Pi_L,trace_dqfim
    if compute_both:
        # Compute both training (input) and target DQFIM stats using the same compiled circuit.
        L_train = get_L(False)
        L_target = get_L(True)
        QFIM_train, trace_train, eigvals_train, eigvecs_train, entropies_train = compute_qfim_eigval_decomp(L_train)
        QFIM_target, trace_target, eigvals_target, eigvecs_target, entropies_target = compute_qfim_eigval_decomp(L_target)
        # Here, we choose the training DQFIM to compute the learning rate (you may change this if desired)
        initial_lr_train = scale_factor / (jnp.real(trace_train) * grad_norm + 1e-12)
        dqfim_inv_eigvals_train = jnp.where(eigvals_train > 1e-12, 1.0 / eigvals_train, 0.0)
        dqfim_inv_train = jnp.dot(eigvecs_train, jnp.diag(dqfim_inv_eigvals_train)).dot(jnp.conj(eigvecs_train.T))
        dqfim_grad_norm_train = jnp.dot(jnp.conj(init_grads), jnp.dot(dqfim_inv_train, init_grads))
        initial_lr_train = scale_factor / (jnp.real(dqfim_grad_norm_train) + 1e-12)
        initial_lr_train = jnp.clip(initial_lr_train, min_lr, max_lr)
        return initial_lr_train, {
            "DQFIM_train": QFIM_train,
            "trace_train": trace_train,
            "eigvals_train": eigvals_train,
            "entropies_train": entropies_train,},{
            "DQFIM_target": QFIM_target,
            "trace_target": trace_target,
            "eigvals_target": eigvals_target,
            "entropies_target": entropies_target
        }
    else:
        # Compute only one set of DQFIM stats based on the target_DQFIM flag.
        L_local = get_L(target_DQFIM)
        if target_DQFIM:
            dqfim_type = 'target'
        else:
            dqfim_type = 'train'
        QFIM, trace_dqfim, eigvals, eigvecs, entropies = compute_qfim_eigval_decomp(L_local)
        initial_lr = scale_factor / (jnp.real(trace_dqfim) * grad_norm + 1e-12)
        dqfim_inv_eigvals = jnp.where(eigvals > 1e-12, 1.0 / eigvals, 0.0)
        dqfim_inv = jnp.dot(eigvecs, jnp.diag(dqfim_inv_eigvals)).dot(jnp.conj(eigvecs.T))
        dqfim_grad_norm = jnp.dot(jnp.conj(init_grads), jnp.dot(dqfim_inv, init_grads))
        initial_lr = scale_factor / (jnp.real(dqfim_grad_norm) + 1e-12)
        initial_lr = jnp.clip(initial_lr, min_lr, max_lr)
        return initial_lr, {f"eigvals_{dqfim_type}": eigvals,  f"DQFIM_{dqfim_type}": QFIM, f"entropies_{dqfim_type}": entropies, f"trace_{dqfim_type}": trace_dqfim}
    

    

def quantum_fun(gate, input_state, qubits):
    '''
    Apply the gate to the input state and return the output state.
    '''
    qml.StatePrep(input_state, wires=[*qubits])
    gate(wires=qubits)
    return qml.density_matrix(wires=[*qubits])






def generate_dataset(gate, n_qubits, size, key, L=None):
    """
    Generate a dataset of input and output states for the given gate.
    
    Parameters:
      gate: The quantum gate to apply.
      n_qubits: Number of qubits in the system.
      size: Number of states to generate.
      key: JAX random key for reproducibility.
      L: (Optional) Pre-selected states (used for training).
      
    Returns:
      Tuple (X, y) where X are the input states and y are the output states.
    """
    if L is not None:
        # Use the provided states for the first 'size' training samples.
        X = np.asarray(L[:size])
        print(f"Using pre-selected states for training. Number of training states: {X.shape[0]}")
    else:
        X = []
        for _ in range(size):
            key, subkey = jax.random.split(key)
            seed_value = int(jax.random.randint(subkey, (1,), 0, 2**32 - 1)[0])
            state_vec = random_statevector(2**n_qubits, seed=seed_value).data
            X.append(np.asarray(state_vec))
        X = np.stack(X)
    qubits = Wires(list(range(n_qubits)))
    dev_data = qml.device("default.qubit", wires=qubits)
    circuit = qml.QNode(quantum_fun, device=dev_data, interface="jax")
    y = np.stack([np.asarray(circuit(gate, X[i], qubits)) for i in range(size)])
    return X, y

def run_test(params,num_epochs, N_reserv, N_ctrl, time_steps,N_train,N_test,folder,gate,gate_name,init_params_dict,filename,dataset_key, L = [], use_L=False):
    opt_lr = None
    num_J = N_ctrl*N_reserv
    init_params = params
    
    
    # opt_a,opt_b,worst_a,worst_b,opt_lr = optimize_traingset(gate,N_ctrl, N_reserv,time_steps, params, init_params_dict, N_train,10,key)
    #key,subkey = jax.random.split(states_key)
    #add_a,add_b =  np.asarray(worst_a[:N_train]), np.asarray(worst_b[:N_train])
    if use_L:
        train_X, train_y = generate_dataset(gate, N_ctrl, N_train, dataset_key, L)
    else:
        train_X, train_y = generate_dataset(gate, N_ctrl, N_train, dataset_key)
        
    # Generate testing dataset with a new key for independence.
    test_dataset_key = jax.random.split(dataset_key)[1]
    test_X, test_y = generate_dataset(gate, N_ctrl, N_test, test_dataset_key)

    input_states, target_states = np.asarray(train_X), np.asarray(train_y)
    # print(f"Training dataset shapes: input_states: {input_states.shape}, target_states: {target_states.shape}")
    # print(f"Testing dataset shapes: test_X: {test_X.shape}, test_y: {test_y.shape}")
    
    if use_L and len(L) > 0:
        assert np.array_equal(input_states, np.asarray(L)[:N_train]), (
            f"Training set not set correctly. input_states[0]: {input_states[0]}, L[0]: {L[0]}"
        )
    sim_qr = Sim_QuantumReservoir(init_params_dict, N_ctrl, N_reserv, N_reserv * N_ctrl,time_steps)
    
    
    parameterized_ham = sim_qr.get_total_hamiltonian_components()



    print("Number of trainable parameters: ", len(params))

    ctrl_wires = sim_qr.get_ctrl_wires()
   
    qnode_dev = sim_qr.get_dev()



    costs = []
    param_per_epoch = []
    

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
   
    def final_test(params,test_X,test_y):
        params = jnp.asarray(params, dtype=jnp.float64)
        X = jnp.asarray(test_X, dtype=jnp.complex128)
        y = jnp.asarray(test_y, dtype=jnp.complex128)
        batched_output_states = vcircuit(params, X)

        fidelities = jax.vmap(qml.math.fidelity)(batched_output_states, y)
        fidelities = jnp.clip(fidelities, 0.0, 1.0)

        return fidelities
    # Initial training to determine appropriate learning rate
    if opt_lr is None:
        s = time.time()
        init_loss, init_grads = jax.value_and_grad(cost_func)(params, input_states, target_states)
        e = time.time()
        dt = e - s
        raw_lr, clipped_lr = get_initial_learning_rate(init_grads)
        opt_lr = get_initial_lr_per_param(init_grads, max_lr=raw_lr)
        cost = init_loss

        if not use_L:
            # Compute DQFIM stats for the training space (using the input states)
            dqfim_initial_lr_train, dqfim_dict_train = get_initial_learning_rate_DQFIM(
                params=params,
                qrc=sim_qr,
                X=train_X,
                gate=gate,
                init_grads=init_grads,
                target_DQFIM=False
            )
            # Note: In the non-compute_both branch, the function returns a dict with key "DQFIM_target"
            DQFIM = dqfim_dict_train.get('DQFIM_target', None)
            tr_dqfim_train = jnp.trace(DQFIM)
            evals, _ = jnp.linalg.eigh(DQFIM)
            nonzero = evals[evals > threshold]
            print(f"DQFIM state space --- lr: {dqfim_initial_lr_train:.5f}\n - Tr(DQFIM) = {tr_dqfim_train:.3f}\n - var(DQFIM eigs) = {np.var(nonzero):.3f}")

        # Always compute DQFIM stats for the target space (using the target states)
        dqfim_initial_lr_targ, dqfim_dict_target = get_initial_learning_rate_DQFIM(
            params=params,
            qrc=sim_qr,
            X=train_X,
            gate=gate,
            init_grads=init_grads,
            target_DQFIM=True
        )
        DQFIM_target = dqfim_dict_target.get('DQFIM_target', None)
        tr_dqfim_targ = jnp.trace(DQFIM_target)
        evals, _ = jnp.linalg.eigh(DQFIM_target)
        nonzero = evals[evals > threshold]
        print(f"target space DQFIM --- lr: {dqfim_initial_lr_targ:.5f}\n - Tr(tDQFIM) = {tr_dqfim_targ:.3f}\n - var(tDQFIM eigs) = {np.var(nonzero):.3f}")
    else:
        s = time.time()
        init_loss, init_grads = jax.value_and_grad(cost_func)(params, input_states, target_states)
        e = time.time()
        dt = e - s
        opt_lr,grad_norm = get_initial_learning_rate(init_grads)
        cost = init_loss
        # print(f"initial fidelity: {init_loss:.4f}, init opt: {maybe_lr}. Time: {dt:.2e}")
        
    opt_descr = 'per param'

    
    learning_rate_schedule = optax.constant_schedule(opt_lr)
    opt = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.inject_hyperparams(optax.adam)(learning_rate=opt_lr, b1=0.99, b2=0.999, eps=1e-7),
        )
    


   
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
    print(f"per param lrs: \n{opt_lr}\n")

    print("________________________________________________________________________________")
    print(f"Starting optimization for {gate_name}(epochs: {num_epochs}) with optimal lr {np.mean(opt_lr)}, time_steps = {time_steps}, N_r = {N_reserv}...\n")
    print(f"Initial Loss: {init_loss:.4f}, initial_gradients: {np.mean(np.abs(init_grads))}. Time: {dt:.2e}")
    print(f"Depth: {circuit_depth}, Num Gates: {num_gates} ")
    print("Number of trainable parameters: ", len(params))
    costs = []
    param_per_epoch,grads_per_epoch = [],[]
   # print(f"Params: {params}")
    
    opt_state = opt.init(params)
    prev_cost, second_prev_cost = float('inf'), float('inf')  # Initialize with infinity
    threshold_counts = 0

    consecutive_improvement_count = 0
    cost_threshold = 1e-5
    consecutive_threshold_limit = 4  # Number of consecutive epochs below threshold without improvement needed to stop
    backup_params = init_params
    improvement = True
    backup_cost,min_cost =init_loss,float('inf')   
    freeze_tau = False
    epoch = 0
    s = time.time()

    false_improvement = False
    backup_epoch=0
    fullstr = time.time()

    while epoch < num_epochs or improvement:
        #print(params, type(params))
        params, opt_state, cost, grad = update(params, opt_state, input_states, target_states,value=cost)
        
        # Store parameters and cost for analysis
        param_per_epoch.append(params)
        costs.append(cost)
        grads_per_epoch.append(grad)
        # Logging
        max_abs_grad = jnp.max(jnp.abs(grad))
        if epoch == 0 or (epoch + 1) % 250 == 0:
            var_grad = np.var(grad,ddof=1)
            mean_grad = np.mean(jnp.abs(grad))
            e = time.time()
            epoch_time = e - s
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



    if backup_cost < cost and not epoch < num_epochs and backup_epoch < epoch - 100:
        print(f"backup cost (epoch: {backup_epoch}) is better with: {backup_cost:.2e} <  {cost:.2e}: {backup_cost < cost}")
        # print(f"recomputed cost (i.e. cost_func(backup_params,input_states, target_states)): {cost_func(backup_params,input_states, target_states)}")
        # print(f"cost_func(params, input_states,target_states): {cost_func(params, input_states,target_states)}")
        # print(f"final_test(backup_params,test_in, test_targ): {final_test(backup_params,test_in, test_targ)}")
        # print(f"final_test(params,test_in, test_targ): {final_test(params,test_in, test_targ)}")
        params = backup_params
    fullend = time.time()
    print(f"time optimizing: {fullend-fullstr}")
    df = pd.DataFrame()
    
    # print(f"Testing opt params against {test_size} new random states...")

    testing_results = final_test(params,test_X, test_y)
    avg_fidelity = jnp.mean(testing_results)
    infidelities = 1.00000000000000-testing_results
    print("\nAverage Final Fidelity: ", avg_fidelity)
    
    data = {'Gate':base64.b64encode(pickle.dumps(gate)).decode('utf-8'),
            'opt_description': opt_descr,
            'specs':specs,
            
                'epochs': num_epochs,
                'trotter_step': time_steps,
                'time_steps': time_steps,
                'controls': N_ctrl, 
                'reservoirs': N_reserv,
                'N_train': N_train,
                'fixed_params': init_params_dict,
                'init_params': init_params,
                'testing_results': np.array(testing_results).tolist(),
                'infidelities': np.array(infidelities).tolist(),
                'avg_fidelity': avg_fidelity,
                'costs': costs,
                'params_per_epoch':param_per_epoch,
                'training_states': input_states,
               
                'opt_params': params,
                'opt_lr': opt_lr,
                
               
                'grads_per_epoch':grads_per_epoch,
                'init_grads':init_grads,
                'training_states': train_X,

            }
    data["target DQFIM stats"] = dqfim_dict_target
    if not use_L:
                
        data['DQFIM_stats_local'] = dqfim_dict_train 
    return data
    


def convert_to_float(value):
    """Convert NumPy arrays or other numeric types to float or list of floats."""
    if isinstance(value, np.ndarray):
        # Convert NumPy array to list of floats
        return [float(f"{x:.16}") for x in value]
    elif isinstance(value, (float, int, np.float32, np.float64, np.int32, np.int64)):
        # Convert single numeric value to float
        return float(f"{value:.16}")
    else:
        # Return the value as-is if it's not a recognized numeric type
        return value



if __name__ == '__main__':
    float32=''

  
    gates_random = []
    
    
    N_ctrl = 1
    N_r = 1
    num_J = N_ctrl*N_r
    time_steps = 2
    
    for i in range(20):
        U = random_unitary(2**N_ctrl, i).to_matrix()

        g = partial(qml.QubitUnitary, U=U)
        g.num_wires = N_ctrl
        g.name = f'U{N_ctrl}_'+str(i)
        gates_random.append(g)

    trainable_param_keys = ['test3', 'test6', 'test8', 'test9']
    all_gates = gates_random


    num_epochs = 500
    N_train = 10
    N_test = 2000

    base_state = 'GHZ_state'

    state = 'GHZ'
    Kfactor = '1.0'
    K_0 = '1'
    sample_range_label = 'normal_.5pi'
    num_L = 20
    L = []
    delta_x = 1.49011612e-08
    threshold = 1.e-10
    use_L = False
    fixed_param_name='fixed_params0'
    if use_L:
        folder = f'./param_initialization_final/analog_results_use_L_{use_L}/Nc_{N_ctrl}/epochs_{num_epochs}'
    else:
        folder = f'./param_initialization_final/analog_results/Nc_{N_ctrl}/epochs_{num_epochs}'
    for gate_idx,gate in enumerate(all_gates):
        for test_key in trainable_param_keys:
        # for fixed_param_name, test_key in zip(fixed_param_keys,trainable_param_keys):
            # print(fixed_param_name,test_key)
        
            folder_gate = os.path.join(
                folder,
                f"reservoirs_{N_r}",
                f"trotter_{time_steps}",
                f"trainsize_{N_train}",
                f"sample_{sample_range_label}",
                fixed_param_name,
                f"{test_key}",
                gate.name
            )
            
            Path(folder_gate).mkdir(parents=True, exist_ok=True)
            temp_list = list(Path(folder_gate).glob('*'))
            files_in_folder = []
            for f in temp_list:
                temp_f = f.name.split('/')[-1]
                
                if not temp_f.startswith('.'):
                    files_in_folder.append(temp_f)
            tests_completed = len(files_in_folder)
            if tests_completed >= 1:
                print('Already Done. Skipping: '+folder_gate)
                print('\n')
                continue
         
            
            print("________________________________________________________________________________")
            dataset_seed =  gate_idx*time_steps*N_r + gate_idx**2 + N_ctrl
            dataset_key = jax.random.PRNGKey(dataset_seed)

            filename = os.path.join(folder_gate, f'data_run_{len(files_in_folder)}.pickle')
            
                    
            N =N_ctrl+N_r
            qfim_base_path2 = f'/Users/sophieblock/QRCCapstone/parameter_analysis_directory/QFIM_results_basis_state/analog/Nc_{N_ctrl}/sample_{sample_range_label}/{K_0}xK/Nr_{N_r}/trotter_step_{time_steps}'

            qfim_basis_init_state_stats = get_qfim_stats_basis_state(qfim_base_path2,fixed_param_name, test_key)
            dqfim_file_path =f'/Users/sophieblock/QRCCapstone/parameter_analysis_directory/QFIM_global_results/analog_model_DQFIM/Nc_{N_ctrl}/sample_{sample_range_label}/{K_0}xK/Nr_{N_r}/trotter_step_{time_steps}/'

            dqfim_stats_dict = get_dqfim_stats(dqfim_file_path, fixed_param_dict_key=fixed_param_name, trainable_param_dict_key=test_key, num_L = num_L)

            qfim_base_path = f'/Users/sophieblock/QRCCapstone/parameter_analysis_directory/QFIM_results/analog/Nc_{N_ctrl}/sample_{sample_range_label}/{K_0}xK'
            qfim_file_path = Path(qfim_base_path) / f'Nr_{N_r}' / f'trotter_step_{time_steps}/' 
            

            # print(qfim_file_path)
            # print(f"{test_key} params (range: (-{sample_range_label}, {sample_range_label}))")
            # print(f"{fixed_param_name}")
            eigvals, fixed_params_dict,params,qfim,entropy = get_qfim_eigvals(qfim_file_path, fixed_param_name, test_key)

            # eigvals, fixed_params_dict, params, qfim, entropies = get_qfim_eigvals(qfim_file_path, fixed_param_name, test_key)
            
            # print(f"{test_key}")
            # print(f"{fixed_param_name}: {fixed_params_dict}")
            
            
            trace_qfim = np.sum(eigvals)
            nonzero = eigvals[eigvals > threshold]
            var_qfim = np.var(nonzero)
            
            print(f"Tr(Q) (GHZ): {trace_qfim:.2f}, Tr(Q) (basis): {qfim_basis_init_state_stats['raw_trace']:.2f}, DQFIM Tr: {dqfim_stats_dict['raw_trace']:.2f}")
            print(f"Var(evs) (GHZ): {var_qfim:.5f}, Var(evs) (basis): {qfim_basis_init_state_stats['raw_var_nonzero']:.5f}, DQFIM Var: {dqfim_stats_dict['raw_var_nonzero']:.5f}")

            if use_L:
                input_states = dqfim_stats_dict['L']
                data = run_test(params=params,num_epochs=num_epochs, N_reserv=N_r,N_ctrl= N_ctrl, time_steps=time_steps,N_train=N_train,N_test=N_test,folder=folder,gate=gate,gate_name=gate.name,init_params_dict=fixed_params_dict,filename=filename,dataset_key= dataset_key, L = input_states, use_L=True)
            else:
                data = run_test(params=params,num_epochs=num_epochs, N_reserv=N_r,N_ctrl= N_ctrl, time_steps=time_steps,N_train=N_train,N_test=N_test,folder=folder,gate=gate,gate_name=gate.name,init_params_dict=fixed_params_dict,filename=filename,dataset_key= dataset_key,  use_L=False)
            data['test_key'] = test_key
            data['QFIM Results'] = {"qfim_eigvals":eigvals,
                                    "trainable_params": params,
                                    "qfim": qfim,
                                    "entropy":entropy,
            'variance':var_qfim,
            'trace':trace_qfim,

                }
            data[f'DQFIM_stats_{num_L}_L_states'] = dqfim_stats_dict
            if use_L:
                # same results
                data['DQFIM_stats_local'] = dqfim_stats_dict 
            
            data['QFIM_basis_state'] = qfim_basis_init_state_stats
            
            df = pd.DataFrame([data])
            while os.path.exists(filename):
                name, ext = filename.rsplit('.', 1)
                filename = f"{name}_.{ext}"

            with open(filename, 'wb') as f:
                pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved to path: {filename}")

