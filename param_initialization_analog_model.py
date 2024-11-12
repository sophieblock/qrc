import pennylane as qml
from pennylane.math import vn_entropy
import os
import pickle
import re
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
from qutip import *
from qutip.qip.operations import cnot,rz,rx,ry,snot
from qutip.qip.circuit import QubitCircuit
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
from QFIM import get_qfim_data_analog
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


def get_target_state(gate, input_state, qubits):
    '''
    Apply the gate to the input state and return the output state.
    '''
    qml.QubitStateVector(input_state, wires=[*qubits])
    gate(wires=qubits)
    return qml.state()





def generate_dataset(gate, n_qubits, training_size,testing_size, key, L=[]):
    '''
    Generate the dataset of input and output states according to the gate provided.
    Uses a seed for reproducibility.
    '''
    
    if len(L) == 0:
        # Generate random state vectors
        X = []
        for _ in range(training_size + testing_size):
            key, subkey = jax.random.split(key)  # Split the key to update it for each iteration
            # Extract the scalar seed value explicitly to avoid deprecation warning
            seed_value = int(jax.random.randint(subkey, (1,), 0, 2**32 - 1)[0])
            # Use the extracted scalar seed value
            state_vec = random_statevector(2**n_qubits, seed=seed_value).data
            X.append(np.asarray(state_vec))
    else:
        L = np.asarray(L) 
        print(f"Using pre-selected states for training")
        X = [state for i,state in enumerate(L) if i < training_size]
        print(f"len(x): {len(X)}")
        for _ in range(testing_size):
            key, subkey = jax.random.split(key)  # Split the key to update it for each iteration
            # Extract the scalar seed value explicitly to avoid deprecation warning
            seed_value = int(jax.random.randint(subkey, (1,), 0, 2**32 - 1)[0])
            # Use the extracted scalar seed value
            state_vec = random_statevector(2**n_qubits, seed=seed_value).data
            X.append(np.asarray(state_vec))

    # Convert lists to np.ndarrays before concatenating
    # X = np.asarray(X)  # Convert list X to an ndarray
    #  # Convert list L to an ndarray

    # # Concatenate the arrays
    # X = np.concatenate([L, X], axis=0)
    X = np.stack(X)
    qubits = Wires(list(range(n_qubits)))
    dev_data = qml.device('default.qubit', wires=qubits)
    circuit = qml.QNode(quantum_fun, device=dev_data, interface='jax')
    
    # Execute the circuit for each input state
    y = np.stack([np.asarray(circuit(gate, X[i], qubits)) for i in range(training_size + testing_size)])
    
    return X, y
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

def get_init_params(N_ctrl, N_reserv, time_steps, bath, num_bath, base_key):
    N = N_reserv + N_ctrl
    # Adjust the key based on time_steps and fixed_param_num
    key = jax.random.PRNGKey((base_key) * 123456789 % (2**32))  # Example combination
    
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

def run_hyperparam_test(lr,num_epochs, N_reserv, N_ctrl, time_steps, folder, batch_size,gate,a,b,init_params_dict,sim_qr,params):
    opt = optax.adam(learning_rate=lr)
    input_states, target_states = a,b
    assert len(input_states) == batch_size
    num_J = N_ctrl*N_reserv
    key = jax.random.PRNGKey(0)
    #params = jax.random.normal(key, shape=(1+time_steps+(N_ctrl * N_reserv)*time_steps,))

    #for i in range(time_steps):
        #params = params.at[i].set(np.abs(params[i]))

    

    
   # sim_qr = Sim_QuantumReservoir(init_params_dict, N_ctrl, N_reserv, N_reserv * N_ctrl, input_states, target_states,time_steps,bath,num_bath)
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
        print(len(params))
        # Process the batch of states
        batched_output_states = vcircuit(params, input_states)
        
        # Compute fidelity for each pair in the batch and then average
        fidelities = jax.vmap(qml.math.fidelity)(batched_output_states, target_states)
        average_fidelity = np.sum(fidelities)/len(fidelities)
        
        return 1 - average_fidelity  # Minimizing infidelity

    @partial(jit, static_argnums=(1, 2, 3))
    def cost_func(params, time_steps, N_reserv, N_ctrl, input_states, target_states):
        return batched_cost_helper(params, input_states, target_states)
   

    
    for epoch in range(num_epochs):
        
        cost, grad_circuit = jax.value_and_grad(cost_func)(params, time_steps, N_reserv, N_ctrl, input_states, target_states)
        
        # Update parameters using the optimizer
        updates, opt_state = opt.update(grad_circuit, opt_state)
        params = optax.apply_updates(params, updates)
        params = params.at[:time_steps].set(jax.numpy.where(params[:time_steps] < 0, jax.numpy.abs(params[:time_steps]), params[:time_steps]))

        

    print(f"Resulting fidelity for learning rate {lr}: {1-cost}")
    return 1- cost


def hyperparameter_optimization_batch(gate, num_epochs, N_reserv, N_ctrl, N_train, time_steps,folder,init_params_dict,a,b,sim_qr,params):
    randomize = False
    if N_ctrl == 1:
        if N_reserv > 2:
            return 0.1, randomize
            learning_rates = np.array([0.1,0.2,0.3])


        else:
            learning_rates = np.array([0.01,0.05, 0.1,0.2])
        #learning_rates = np.array([0.25,0.2, 0.15,0.1,0.05,0.01])
        
    elif N_ctrl == 2:
        learning_rates = np.array([0.1,0.05])
    else:
        learning_rates = np.array([0.2,0.1,0.05,0.01])
    
    # Partially apply the fixed parameters to your test function
    partial_hyperparam_test = partial(run_hyperparam_test, 
                                      num_epochs=num_epochs, 
                                      N_reserv=N_reserv, 
                                      N_ctrl=N_ctrl, 
                                      time_steps=time_steps, 
                                      folder=folder, 
                                      batch_size=N_train, 
                                      gate=gate, 
                                      
                                      a=a, 
                                      b=b, 
                                      init_params_dict=init_params_dict,
                                      sim_qr=sim_qr,
                                      params=params)

    vrun_hyperparam_test = jax.vmap(partial_hyperparam_test, in_axes=0)

    # Run the tests in parallel
    performances = vrun_hyperparam_test(learning_rates)

    # Analyze the results to find the best learning rate
    best_performance_index = jax.numpy.argmax(performances)
    best_lr = learning_rates[best_performance_index]
    
    return best_lr, randomize

def array(a,dtype):
    return np.asarray(a, dtype=np.float32)
def Array(a,dtype):
    return np.asarray(a, dtype=np.float32)

# @jit
def get_initial_learning_rate(grads, scale_factor=0.1, min_lr=5e-3, max_lr=0.05):
    """Estimate a more practical initial learning rate based on the gradient norms."""
    grad_norm = jnp.linalg.norm(grads)
    max_grad = max(np.abs(grads))
    print(f"max_grad: {max_grad:4e}, inv: {1/max_grad:4e}, grad norm: {grad_norm:4e}")
    initial_lr = jnp.where(grad_norm > 0, scale_factor / grad_norm, 0.1)
    initial_lr = jnp.clip(initial_lr, min_lr, max_lr)
    return initial_lr, grad_norm

def get_initial_learning_rate_DQFIM(params,qrc,X,y,gate,init_grads, scale_factor=0.1, min_lr=9e-4, max_lr = 0.1):
    """
    Compute an initial learning rate based on the norm of gradients.
    """
    parameterized_ham = qrc.get_total_hamiltonian_components() # wires of the reservoir qubits (i.e. number of qubits in the reservoir)
    rsv_qubits = qrc.get_reserv_wires() # wires of the control qubits (i.e. number of qubits in the control)
    grad_norm = jnp.linalg.norm(init_grads)
    print(f"grad_norm: {grad_norm} max_grad: {max(np.abs(init_grads))}")

    ctrl_wires = qrc.get_ctrl_wires()
    qnode_dev = qrc.get_dev()

    dev_data = qml.device('default.qubit', wires=ctrl_wires)
    target_generator = qml.QNode(get_target_state, device=dev_data, interface='jax')
    
    # Execute the circuit for each input state
    L = np.stack([np.asarray(target_generator(gate, x, ctrl_wires)) for x in X])
    # print(f"X[0].shape: {X[0].shape} L[0].shape: {L[0].shape}, ctrl_wires: {ctrl_wires}")


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
        
    def get_density_matrix_sum(params, states, jit_circuit):
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
    def get_partial_grads(params, states,jit_circuit, delta_x=1.49011612e-08):
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
                            
    def compute_qfim_eigval_decomp(params, L):
        density_matrix_grads = get_partial_grads(params, L, jit_circuit)
        entropies,Pi_L = get_density_matrix_sum(params, L, jit_circuit)

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
        return eigvals, eigvecs, QFIM, entropies,density_matrix_grads,Pi_L,trace_dqfim
    # Eigenvalue decomposition of the DQFIM
    dqfim_eigvals,dqfim_eigvecs, DQFIM, entropies,density_matrix_grads,Pi_L,trace_dqfim =compute_qfim_eigval_decomp(params,L)
    initial_lr = scale_factor / (jnp.real(trace_dqfim) * grad_norm + 1e-12)
    print(f"Initial (Tr(dqfim*grad_norm)^-1) lr: {initial_lr:.6f}")

    
    # threshold = 1e-12  # Small threshold to avoid division by zero
    nonzero_eigvals = dqfim_eigvals[dqfim_eigvals > 1e-12 ]
    # print(f"dqfim_eigvecs shape: {dqfim_eigvecs.shape}")
    # Invert the eigenvalues (pseudo-inverse)
    dqfim_inv_eigvals = jnp.where(dqfim_eigvals > 1e-12, 1.0 / dqfim_eigvals, 0.0)
    # print(f"dqfim_inv_eigvals shape: {dqfim_inv_eigvals.shape}")    
    

    # Reconstruct the pseudo-inverse of the DQFIM
    dqfim_inv = jnp.dot(dqfim_eigvecs, jnp.diag(dqfim_inv_eigvals)).dot(jnp.conj(dqfim_eigvecs.T))

    # Calculate the gradient norm using the DQFIM pseudo-inverse (natural gradient)
    dqfim_grad_norm = jnp.dot(jnp.conj(init_grads), jnp.dot(dqfim_inv, init_grads))

    initial_lr = scale_factor / (jnp.real(dqfim_grad_norm) + 1e-12)
    print(f"Initial learning rate based on quantum natural gradient: {initial_lr:.6f}")


    



    initial_lr = jnp.clip(initial_lr, min_lr, max_lr)
    

    
    return initial_lr, {"dqfim_eigvals": dqfim_eigvals,"dqfim_eigvecs": dqfim_eigvecs, "DQFIM": DQFIM,"entropies": entropies}

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
        penalty_factor *= 0.8  # Penalize unstable datasets (reduce to 70%)
        # print(f"Penalized for high variance and high instability. Penalty factor reduced to {penalty_factor:.2f}")
    
    # Reward stability (low IQR, less than 10th percentile)
    if norm_iqr_var_grad < iqr_percentile_10:
        penalty_factor *= 1.2  # Reward more stable datasets
        # print(f"Rewarded for stability (low IQR less than 10th percentile). Penalty factor increased to {penalty_factor:.2f}")
    # Reward stability (low IQR, less than 10th percentile)
    if norm_var_grad_var < var_grad_var_percentile_10:
        penalty_factor *= 1.2  # Reward more stable datasets
        # print(f"Rewarded for stability (low var(var_grad) less than 20th percentile). Penalty factor increased to {penalty_factor:.2f}")
    
    # Penalize large variance of var_grad (instability)
    if norm_var_grad_var > var_grad_var_percentile_90:
        penalty_factor *= 0.8  # Penalize unstable datasets with large var_grad variance
        # print(f"Penalized for large var_grad variance (greater than 95th percentile). Penalty factor reduced to {penalty_factor:.2f}")

    # Calculate the weighted sum with penalties applied
    weighted_sum = penalty_factor * norm_var_grad 
    
    # print(f"Final weighted sum with penalties applied: {weighted_sum:.5e} norm_var_grad: {norm_var_grad:.5e}")
    return weighted_sum
def optimize_traingset(gate,N_ctrl, N_reserv,time_steps, params, init_params_dict, N_train,num_datasets, key):
    datasets = []
    print(f"Pre-processing a batch of {num_datasets} training sets for selection... ")
    all_A, all_b = [],[]
    for i in range(num_datasets):
        key, subkey = jax.random.split(key)  # Split the key for each dataset generation
        A,b = generate_dataset(gate, N_ctrl, N_train, 2000, subkey)  # Generate dataset with the subkey
        all_A.append(A)
        all_b.append(b)
    all_A = jnp.stack(all_A)
    all_b = jnp.stack(all_b)
    # Convert datasets list into two arrays for inputs and targets
    
    num_J = N_reserv *N_ctrl
    
    sim_qr = Sim_QuantumReservoir(init_params_dict, N_ctrl, N_reserv, num_J,time_steps)
    parameterized_ham = sim_qr.get_total_hamiltonian_components()


    ctrl_wires = sim_qr.get_ctrl_wires()
    reserv_wires = sim_qr.get_reserv_wires()
    qnode_dev = sim_qr.get_dev()
    all_wires = sim_qr.get_all_wires()

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
        mean_grad_squared = jnp.mean(gradients**2, axis=0)
        var_grad = mean_grad_squared - mean_grad**2
        return mean_grad, var_grad
    batched_collect_gradients = jax.vmap(collect_gradients, in_axes=(None, 0, 0))

    all_gradients = batched_collect_gradients(params, all_A[:, :N_train], all_b[:, :N_train])

    all_variances = []
    all_iqrs = []
    all_var_grad_vars = []
    for i in range(num_datasets):
        _, var_grad = calculate_gradient_stats(all_gradients[i])
        all_variances.append(var_grad.mean())
        all_iqrs.append(stats.iqr(var_grad,rng=(30,70)))
        all_var_grad_vars.append(np.var(var_grad)) 


    best_dataset_idx = None
    max_var_grad_sum = -jnp.inf
    second_best_idx = None
    min_var_grad_sum = jnp.inf
# Define the weight factors for `mean_grad.mean()` and `aggregated_var`
    weight_mean_grad = 0.4  # You can adjust this weight
    weight_aggregated_var = 0.6  # You can adjust this weight
    max_weighted_sum = -jnp.inf

    min_weighted_sum = jnp.inf
    # Calculate and print gradient statistics for each dataset
    for i in range(num_datasets):
        mean_grad, var_grad = calculate_gradient_stats(all_gradients[i])
        mean_grad_var =jnp.var(mean_grad)
        # Calculate the weighted average of `mean_grad.mean()` and `aggregated_var`
        aggregated_var = var_grad.mean()
        
        
        min_grad = jnp.min(jnp.abs(mean_grad))
        max_grad = jnp.max(jnp.abs(mean_grad))
        min_var = jnp.min(var_grad)
        max_var = jnp.max(var_grad)
        # Calculate IQR of the variances
       
        iqr_var_grad_25_75 = stats.iqr(var_grad, rng=(20,80))
        iqr_var_grad_30_70 = stats.iqr(var_grad, rng=(30,70))
        num_parameters = len(var_grad)
        
        # weighted_sum = apply_adaptive_penalty(aggregated_var, iqr_var_grad_25_75, N_ctrl, time_steps, weight_mean_grad, weight_aggregated_var, all_variances, all_iqrs)
        # weighted_sum = apply_adaptive_penalty(aggregated_var, iqr_var_grad_25_75,np.var(var_grad), N_ctrl+N_reserv, time_steps, num_parameters, weight_mean_grad, weight_aggregated_var, all_variances, all_iqrs,all_var_grad_vars)
        weighted_sum = aggregated_var

        print(f"A{i+1}, b{i+1}): Variance of gradients = {aggregated_var:5e}; IQR of Variance (20,80) = {iqr_var_grad_25_75:5e};  IQR of Variance (30,70) = {iqr_var_grad_30_70:5e}\n")
        # print(f"        - min/max varience: {min_var:5e}, {max_var:3e}")
        # print(f"        - var(var_grad): {np.var(var_grad):5e}")
        # print(f"Weighted sum: {weighted_sum:5e}")
        # print(f"IQRs of var_grad: (10,90): {stats.iqr(var_grad,rng=(10,90)):5e}, (20,80): {stats.iqr(var_grad,rng=(20,80)):5e}):")
        
        # Update the best and second-best datasets based on the weighted sum
        if weighted_sum > max_weighted_sum:
            if best_dataset_idx is not None:
                second_best_idx = best_dataset_idx
            max_weighted_sum = weighted_sum
            best_dataset_idx = i
        elif best_dataset_idx is not None and (second_best_idx is None or weighted_sum > calculate_gradient_stats(all_gradients[second_best_idx])[1].mean()):
            second_best_idx = i

        if weighted_sum < min_weighted_sum:
            min_weighted_sum = weighted_sum
    
    print(f"Selected Dataset: A{best_dataset_idx + 1}, b{best_dataset_idx + 1} with Variance Sum: {max_weighted_sum}\nsecond best:  A{second_best_idx + 1} ")
    best_A = all_A[best_dataset_idx]
    best_b = all_b[best_dataset_idx]
    
   
    if second_best_idx is not None:
        worst_A = all_A[second_best_idx]
        worst_b = all_b[second_best_idx]
    else:
        raise ValueError("Second best dataset index was not correctly identified.")
    best_gradients = all_gradients[best_dataset_idx]
    initial_lr = compute_initial_learning_rate(best_gradients)
    print(f"Initial Learning Rate: {initial_lr}")
    assert best_dataset_idx != second_best_idx
    return best_A, best_b, worst_A, worst_b, initial_lr


def run_test(num_epochs, N_reserv, N_ctrl, time_steps,N_train,N_test,folder,gate,gate_name,init_params_dict,params,filename,based_subkey, L = []):
    opt_lr = None
    num_J = N_ctrl*N_reserv
    init_params = params
    
    states_key = jax.random.PRNGKey(based_subkey)
    key,subkey = jax.random.split(states_key)
    # opt_a,opt_b,worst_a,worst_b,opt_lr = optimize_traingset(gate,N_ctrl, N_reserv,time_steps, params, init_params_dict, N_train,10,key)
    #key,subkey = jax.random.split(states_key)
    #add_a,add_b =  np.asarray(worst_a[:N_train]), np.asarray(worst_b[:N_train])
    if len(L) > 0:
        opt_a,opt_b = generate_dataset(gate, N_ctrl, N_train, 2000, subkey, L) 
    else:
        opt_a,opt_b = generate_dataset(gate, N_ctrl, N_train, 2000, subkey) 
    input_states, target_states = np.asarray(opt_a[:N_train]), np.asarray(opt_b[:N_train])
    print(f"opt_a.shape: {opt_a.shape}; train_in shape: {input_states.shape}")
    # print(f"opt_b.shape: {opt_b.shape}; train_in shape: {target_states.shape}")
    test_in, test_targ = opt_a[N_train:], opt_b[N_train:]
    # print(f"test_in.shape: {test_in.shape}; test_targ shape: {test_targ.shape}")
    if len(L) > 0:
        assert np.array_equal(input_states, L[:N_train]), f"Training set not set correctly. input_states[0]: {input_states[0]}, L[0]: {L[0]}"
    sim_qr = Sim_QuantumReservoir(init_params_dict, N_ctrl, N_reserv, N_reserv * N_ctrl,time_steps)
    
    
    parameterized_ham = sim_qr.get_total_hamiltonian_components()



    print("Number of trainable parameters: ", len(params))

    ctrl_wires = sim_qr.get_ctrl_wires()
    reserv_wires = sim_qr.get_reserv_wires()
    qnode_dev = sim_qr.get_dev()
    all_wires = sim_qr.get_all_wires()


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
        
        batched_output_states = vcircuit(params, test_in)
        fidelities = jax.vmap(qml.math.fidelity)(batched_output_states, test_targ)        
    
        return fidelities

    if opt_lr == None:
        # s = time.time()
        init_loss, init_grads = jax.value_and_grad(cost_func)(params, time_steps, N_reserv, N_ctrl, input_states, target_states)
        # e = time.time()
        # dt = e - s
        # print(f"initial fidelity: {init_loss}, initial_gradients: {init_grads}. Time: {dt}")
        opt_lr, dqfim_data = get_initial_learning_rate_DQFIM(params=params,qrc=sim_qr, X = input_states,y=target_states, gate=gate,init_grads=init_grads)
        print(f"Adjusted initial learning rate: {opt_lr:.4f}")
    
    # if opt_lr == None:
    #     s = time.time()
    #     init_loss, init_grads = jax.value_and_grad(cost_func)(params, time_steps, N_reserv, N_ctrl, input_states, target_states)
    #     e = time.time()
    #     dt = e - s
    #     # print(f"initial fidelity: {init_loss}, initial_gradients: {init_grads}. Time: {dt}")
    #     opt_lr,grad_norm = get_initial_learning_rate(init_grads,scale_factor=0.01)
    #     print(f"Adjusted initial learning rate: {opt_lr:0.4e}. Grad_norm 1: {1/grad_norm:0.4e},Grad_norm 0.1: {0.1/grad_norm:0.4e}, Grad_norm 0.01: {0.01/grad_norm:0.4e}")




    print("________________________________________________________________________________")
    print(f"Starting optimization for {gate_name} with {len(input_states)} training states, lr {opt_lr:5e} time_steps = {time_steps}, N_r = {N_reserv}...\n")

    opt = optax.adam(learning_rate=opt_lr)
    @jit
    def update(params, opt_state, input_states, target_states):
        loss, grads = jax.value_and_grad(cost_func)(params, time_steps, N_reserv, N_ctrl, input_states, target_states)
        updates, new_opt_state = opt.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss, grads
    prev_cost = float('inf')  # Initialize with infinity
    threshold_counts = 0
    cost_threshold = 1e-5
    cost,grads_per_epoch = [],[]
    param_per_epoch = []
    print("Number of trainable parameters: ", len(params))
    opt_state = opt.init(params)
    consecutive_threshold_limit = 4  # Number of consecutive epochs below threshold without improvement needed to stop
    prev_cost = float('inf')  # Initialize with infinity
    threshold_counts = 0
    conv_tol = 1e-08
    consecutive_threshold_limit = 4  # Number of consecutive epochs below threshold without improvement needed to stop
    backup_params = None
    improvement = True
    
    backup_cost = float('inf')  
    epoch = 0
    s = time.time()
    full_s =s
    add_more = False
    while epoch < num_epochs or improvement:
        #print(params, type(params))
        
        #cost, grad_circuit = jax.value_and_grad(partial(cost_func, time_steps=time_steps, N_reserv=N_reserv, N_ctrl=N_ctrl, input_states=input_states,target_states= target_states))(params)
        params, opt_state, cost,grad = update(params, opt_state, input_states, target_states)
        
        # Store parameters and cost for analysis
        param_per_epoch.append(params)
        costs.append(cost)
        grads_per_epoch.append(grad)
        # Logging
        if epoch == 0 or (epoch + 1) % 100 == 0:
            e = time.time()
            epoch_time = e - s
            var_grad = np.var(grad)
            mean_grad = np.mean(jnp.abs(grad))
            normalized_var_grad = var_grad /  np.mean(grad**2) 
            #print(params)
            print(f'{epoch + 1}: {cost:.5f}. '
                f'mean(grad): {mean_grad:.1e}, '
                f'Var(grad): {var_grad:.1e}, '
                # f'Mean(grad): {mean_grad:.1e}, '
                f'Var(norm(grad)): {normalized_var_grad:.2e}  [t: {epoch_time:.2f}s]')
        
            # print(f"Epoch {epoch+1}, cost {cost}, gradient: {round(max(grad),5)} time: {round(epoch_time,2)}s")
            #print(f"step {epoch+1}, cost {cost}")
            s = time.time()
        # Check if there is improvement
        if cost < prev_cost:
            
            improvement = True
            if cost < backup_cost:

                backup_cost = cost
                backup_params = params
        else:
            # print(f"    backup_cost: {backup_cost:.6f}")
            improvement = False  # Stop if no improvement

        for i in range(time_steps):
            if params[i] < 0:
                params = params.at[i].set(np.abs(params[i]))
        # Termination check
        if prev_cost <= conv_tol: 
            print(f"Terminating optimization: cost {cost} is below the threshold {conv_tol}")
            break

        if np.abs(max(grad)) < 1e-14 or epoch >= 5000:
            break
            # if epoch > num_epochs/2:
            #     print(f"abs(max(grad) < 1e-14. Breaking...")
            #     break
            # else:
            #     new_key = jax.random.PRNGKey(epoch)
            #     new_in,new_targ = generate_dataset(gate, N_ctrl, 1, new_key)
            #     print(f"grad low {np.var(grad)}, adding state(s) to the training")
            #     input_states = jnp.concatenate([input_states, new_in], axis=0)
            #     target_states = jnp.concatenate([target_states, new_targ], axis=0)
        

        #print(params)
        prev_cost = cost
        epoch += 1  # Increment epoch count

    if backup_cost < cost:
        print(f"backup cost is better: {backup_cost:.6f} <  {cost:.6f}: {backup_cost < cost}")
        params = backup_params
    full_e = time.time()
    epoch_time = full_e - full_s
    print(f"Time optimizing: {epoch_time}")

    testing_results = final_test(params, time_steps, N_reserv, N_ctrl, test_in, test_targ)
    #for input_state,target_state in zip(test_in, test_targ):
    total_tests = len(testing_results)
    avg_fidelity = np.sum(np.asarray(testing_results))/total_tests
    print("\nAverage Final Fidelity: ", avg_fidelity)
    
    data = {'Gate':base64.b64encode(pickle.dumps(gate)).decode('utf-8'),
            'DQFIM_target_states':dqfim_data,
                'epochs': num_epochs,
                'trotter_step': time_steps,
                'controls': N_ctrl, 
                'reservoirs': N_reserv,
                'N_train': N_train,
                'fixed_params': init_params_dict,
                'init_params': init_params,
                'testing_results': testing_results,
                'avg_fidelity': avg_fidelity,
                'costs': costs,
                'params_per_epoch':param_per_epoch,
                'training_states': input_states,
                'target_states':target_states,
                'opt_params': params,
                'opt_lr': opt_lr,
                
                'grads_per_epoch':grads_per_epoch,
    
            }
    return data
    
import pickle
from pathlib import Path


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
    # Ensure file_path is a Path object
    file_path = Path(file_path) if not isinstance(file_path, Path) else file_path
    
    if not file_path.exists():
        print(f"File {file_path} does not exist.")
        return None

    with open(file_path, 'rb') as f:
        df = pickle.load(f)
    
    # Initialize variables
    qfim_eigvals = None
    
    for fixed_params_dict in df.keys():
        if fixed_params_dict == fixed_param_dict_key:
            for trainable_params_dict in df[fixed_params_dict].keys():
                if trainable_params_dict == trainable_param_dict_key:
                    results = df[fixed_params_dict][trainable_params_dict]
                    if 'qfim_eigvals' in results:
                        qfim_eigvals = results['qfim_eigvals']
                        qfim = results['dqfim']
                        fixed_params = results['fixed_params']
                        fixed_params = results['fixed_params']
                        params = results['trainable_params']
                        entropies = results['entropies']

                        return qfim_eigvals,fixed_params,params,qfim, results['L'],entropies
    
    print("QFIM eigenvalues not found for the given parameter keys.")
    return None,None,None, None, None




if __name__ == '__main__':
    float32=''

  
    num_epochs = 1000
    N_train = 20
    N_test = 2000

    gates_random = []
    
    
    N_ctrl = 1
    N_r = 1
    num_J = N_ctrl*N_r
    time_steps = 1
    folder = f'./param_initialization_final/analog_results/Nc_{N_ctrl}/'
    for i in range(37):
        U = random_unitary(2**N_ctrl, i).to_matrix()

        g = partial(qml.QubitUnitary, U=U)
        g.num_wires = N_ctrl
        g.name = f'U{N_ctrl}_'+str(i)
        gates_random.append(g)

    fp_idx = 0
   
    fixed_param_keys = [f'fixed_params{fp_idx}',f'fixed_params{fp_idx}',f'fixed_params{fp_idx}',f'fixed_params{fp_idx}',f'fixed_params{fp_idx}',f'fixed_params{fp_idx}',f'fixed_params{fp_idx}',f'fixed_params{fp_idx}',f'fixed_params{fp_idx}',f'fixed_params{fp_idx}',f'fixed_params{fp_idx}',f'fixed_params{fp_idx}',f'fixed_params{fp_idx}',f'fixed_params{fp_idx}',f'fixed_params{fp_idx}',f'fixed_params{fp_idx}',f'fixed_params{fp_idx}',f'fixed_params{fp_idx}',f'fixed_params{fp_idx}',f'fixed_params{fp_idx}',f'fixed_params{fp_idx}',f'fixed_params{fp_idx}']
    # trainable_param_keys = ['test53','test42','test177', 'test78','test38']
    # trainable_param_keys = ['test147','test153','test22','test23', 'test45','test133']
    # trainable_param_keys = ['test22','test50','test44','test2','test104', 'test88','test146','test138', 'test63'] 
    # trainable_param_keys = ['test44','test43','test62','test69','test109','test145','test123']
    # trainable_param_keys = ['test108','test0','test15','test186','test123', 'test49'] # N_ctrl = 1, N_r = 1, trot = 1

    # trainable_param_keys = ['test76','test169','test61','test178','test107'] # N_ctrl = 2, N_r = 1, trot = 8
    # trainable_param_keys = ['test33','test89','test80', 'test60','test20'] # lowest
    # trainable_param_keys = ['test1','test6','test73', 'test15','test57'] # highest
    
    trainable_param_keys = ['test6', 'test41','test6', 'test57','test1','test53','test66', 'test96','test43','test33','test89','test80', 'test60','test20', 'test15']
    trainable_param_keys =  ['test41', 'test57','test1','test53','test96','test43','test33','test89', 'test60','test20', 'test15']
    trainable_param_keys = ['test65','test34','test40', 'test36','test32']
    all_gates = gates_random
    base_state = 'GHZ_state'

    state = 'GHZ'
    Kfactor = '1.0'
    num_L = 50
    delta_x = 1.49011612e-08
    threshold = 1.e-14
    get_states = True
    for gate_idx,gate in enumerate(all_gates):
        for fixed_param_name, test_key in zip(fixed_param_keys,trainable_param_keys):
            # print(fixed_param_name,test_key)
        
            
            folder_gate = folder + '/reservoirs_' + str(N_r) + '/trotter_step_' + str(time_steps) +  '/' +f'{N_train}_training_states_random/'+ str(fixed_param_name)+f'/2pi/'+ test_key+'/' + gate.name + '/'

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
            set_key = jax.random.PRNGKey(gate_idx)
            
            
            print("________________________________________________________________________________")
            

            filename = os.path.join(folder_gate, f'data_run_{len(files_in_folder)}.pickle')
            
                    
            N =N_ctrl+N_r
            
            qfim_base_path = f'/Users/sophieblock/QRCCapstone/QFIM_traced_final_results/analog_model_DQFIM/Nc_{N_ctrl}/{state}_state/'


            qfim_file_path = Path(qfim_base_path) / f'Nr_{N_r}' / f'trotter_step_{time_steps}/L_{num_L}' / 'data_2pi.pickle'
            print(qfim_file_path)
            eigvals, fixed_params_dict, params, qfim, L, entropies = get_qfim_eigvals(qfim_file_path, fixed_param_name, test_key)
            L = []
            print(f"{test_key}")
            # print(f"{fixed_param_name}: {fixed_params_dict}")
            
            
            trace_qfim = np.sum(eigvals)
            var_qfim = np.var(eigvals)
            print(f"QFIM trace: {trace_qfim:.6f}")
            print(f"QFIM var: {var_qfim:.6f}")
            
            data = run_test(num_epochs=num_epochs, N_reserv=N_r,N_ctrl= N_ctrl, time_steps=time_steps,N_train=N_train,N_test=N_test,folder=folder,gate=gate,gate_name=gate.name,init_params_dict=fixed_params_dict,params=params,filename=filename,based_subkey= gate_idx*time_steps*N_r, L = L)
            
            data['QFIM Results'] = {"qfim_eigvals":eigvals,
            'qfim':qfim,
            'trace':trace_qfim,
            'entropies': entropies,
                }
            
            df = pd.DataFrame([data])
            while os.path.exists(filename):
                name, ext = filename.rsplit('.', 1)
                filename = f"{name}_.{ext}"

            with open(filename, 'wb') as f:
                pickle.dump(df, f)
            print(f"Saved to path: {filename}")

