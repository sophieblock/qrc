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
    for _ in range(training_size):
        key, subkey = jax.random.split(key)  # Split the key to update it for each iteration
        # Extract the scalar seed value explicitly to avoid deprecation warning
        seed_value = int(jax.random.randint(subkey, (1,), 0, 2**32 - 1)[0])
        # Use the extracted scalar seed value
        state_vec = random_statevector(2**n_qubits, seed=seed_value).data
        X.append(np.asarray(state_vec))
    
    X = np.stack(X)
    qubits = Wires(list(range(n_qubits)))
    dev_data = qml.device('default.qubit', wires=qubits)
    circuit = qml.QNode(quantum_fun, device=dev_data, interface='jax')
    
    # Execute the circuit for each input state
    y = np.stack([np.asarray(circuit(gate, X[i], qubits)) for i in range(training_size)])
    
    return X, y
def get_init_params(N_ctrl, N_reserv, time_steps, bath, num_bath, base_key):
    N = N_reserv + N_ctrl
    # Adjust the key based on time_steps and fixed_param_num
    key = jax.random.PRNGKey((base_key) * 123456789 % (2**32))  # Example combination
    
    N = N_reserv + N_ctrl
    

    
    K_half = jax.random.uniform(key, (N, N))
    K = (K_half + K_half.T) / 2  # making the matrix symmetric
    K = 2. * K - 1.
    
    key, subkey = jax.random.split(key)
    if bath:
        #bath_array = 0.01 * jax.random.normal(key, (num_bath, N_ctrl + N_reserv))
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
        # Add system-bath interactions
        def bath_coupling_func(p, t):
            return p * np.sin(t)  # Example time-dependent coupling

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
    

def run_hyperparam_test(lr,num_epochs, N_reserv, N_ctrl, time_steps, folder, batch_size, gate, bath,num_bath,a,b,init_params_dict,sim_qr,params):
    opt = optax.adam(learning_rate=lr)
    input_states, target_states = np.asarray(a[:batch_size]), np.asarray(b[:batch_size])
    
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

        
        
        
    #testing_results = final_test(params, time_steps, N_reserv, N_ctrl, test_in, test_targ)
    
    #total_tests = len(testing_results)
    #avg_fidelity = np.sum(np.asarray(testing_results))/total_tests
    print(f"Resulting fidelity for learning rate {lr}: {1-cost}")
    return 1- cost


def hyperparameter_optimization_batch(gate, num_epochs, N_reserv, N_ctrl, N_train, time_steps,folder,bath,num_bath,init_params_dict,a,b,sim_qr,params):
    randomize = False
    if N_ctrl == 1:
        if N_reserv > 2:
            return 0.1, randomize
            learning_rates = np.array([0.1,0.2,0.3])


        else:
            learning_rates = np.array([0.01,0.05, 0.1,0.2])
        #learning_rates = np.array([0.25,0.2, 0.15,0.1,0.05,0.01])
        
    elif N_ctrl == 2:
        learning_rates = np.array([0.2,0.1,0.05,0.01])
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
                                      bath=bath, 
                                      num_bath =num_bath,
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

def compute_initial_learning_rate(gradients, scale_factor=0.01, min_lr=1e-3, max_lr = 0.2):
    """
    Compute an initial learning rate based on the norm of gradients.
    """
    # Compute the norm of the gradients
    
    norm_grad = jnp.linalg.norm(gradients)
    mean_norm_grad = jnp.mean(norm_grad)
    initial_lr = scale_factor / (mean_norm_grad + 1e-8)  # Adding a small value to prevent division by zero
    print(norm_grad, initial_lr)
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
    """
    vcircuit = jax.vmap(circuit, in_axes=(None, 0))
    @jit
    def cost_func(params, input_states, target_states):
        def batched_cost_helper(params, input_states, target_states):
            # Process the batch of states
            batched_output_states = vcircuit(params, input_states)
            # Compute fidelity for each pair in the batch and then average
            fidelities = jax.vmap(qml.math.fidelity)(batched_output_states, target_states)
            average_fidelity = jnp.mean(fidelities)
            return 1 - average_fidelity  # Minimizing infidelity

        return batched_cost_helper(params, input_states, target_states)
    
    
    @jit
    def collect_gradients(params, A, b):
        grad_fn = jax.grad(cost_func, argnums=0)
        return grad_fn(params, A, b)
    """
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


def run_test(num_epochs, N_reserv, N_ctrl, time_steps,N_train,folder,gate,gate_name,bath,num_bath,key,based_subkey,bath_factor):
    float32=''
    
    num_J = N_ctrl*N_reserv
    folder_gate = folder + f"bath_coupling_order_{bath_factor}/"+ str(num_bath) + '/'+gate_name + '/reservoirs_' + str(N_reserv) + '/trotter_step_' + str(time_steps) +'/'
    Path(folder_gate).mkdir(parents=True, exist_ok=True)
    temp_list = list(Path(folder_gate).glob('*'))
    files_in_folder = []
    for f in temp_list:
        temp_f = f.name.split('/')[-1]
        
        if not temp_f.startswith('.'):
            files_in_folder.append(temp_f)
    
    print(files_in_folder)
    #print(list(Path(folder_gate).glob('*')))
    if len(files_in_folder) >= 1:
        print('Already Done. Skipping: '+folder_gate)
        print('\n')
        return
    filename = os.path.join(folder_gate, f'data_run_{len(files_in_folder)}.pickle')

    # print(filename)
    
    init_params_dict = get_init_params(N_ctrl, N_reserv, time_steps,bath,num_bath,based_subkey)
    
    params_key, params_subkey1, params_subkey2 = jax.random.split(key, 3)
    time_step_params = jax.random.uniform(params_subkey1, shape=(time_steps,), minval=0, maxval=1.0)
    remaining_params = jax.random.uniform(params_subkey2, shape=(3 + (N_ctrl * N_reserv) * time_steps,), minval=-np.pi, maxval=np.pi)

    # Combine the two parts
    params = jnp.concatenate([time_step_params, remaining_params])



    # get PQC
    sim_qr = Sim_QuantumReservoir(init_params_dict, N_ctrl, N_reserv, N_reserv * N_ctrl,time_steps,bath,num_bath,bath_factor)
    
    
    # Get optimal hyperparameter (learning rate)
    
    init_params = params

    states_key = jax.random.PRNGKey(based_subkey)
    key,subkey = jax.random.split(states_key)
    
    
    #set_key = jax.random.PRNGKey(0)
    opt_a,opt_b,worst_a,worst_b,opt_lr = optimize_traingset(gate,N_ctrl, N_reserv,time_steps, params, init_params_dict, N_train,5,key)
    
    input_states, target_states = np.asarray(opt_a[:N_train]), np.asarray(opt_b[:N_train])
    test_in, test_targ = opt_a[N_train:], opt_b[N_train:]
    
    
   
    parameterized_ham = sim_qr.get_total_hamiltonian_components()
    print("H: ",parameterized_ham)
    
    
    #s = time.time()
    #opt_lr,randomize = hyperparameter_optimization_batch(gate, 100, N_reserv, N_ctrl, N_train, time_steps,folder,bath,num_bath,init_params_dict,opt_a, opt_b,sim_qr,params)
    #e = time.time()
    #print("opt_lr: ",opt_lr," time: ", e-s)
    
    
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
        for bath_qubit in sim_qr.bath_qubits:
            qml.Hadamard(bath_qubit)
        #print(f"coupling_params: {coupling_params}")
        for idx, tau in enumerate(taus):
           
            hx_array = np.array([params[time_steps]])  # Convert hx to a 1D array
            hy_array = np.array([params[time_steps + 1]])  # Convert hy to a 1D array
            hz_array = np.array([params[time_steps + 2]])  # Convert hz to a 1D array
            J_values = params[time_steps + 3 + idx * num_J : time_steps + 3 + (idx + 1) * num_J]
            """
            print(f"hx: {hx_array}")
            print(f"hy: {hy_array}")
            print(f"hz: {hz_array}")
            print(f"J_values: {J_values}")
            """
            # Concatenate hx_array with J_values and coupling_params
            current_step = np.concatenate([J_values, hx_array, hy_array, hz_array])
            
            
            total_H = parameterized_ham
            #print(f"H at time step {idx}: {total_H}")
            qml.evolve(total_H)(current_step, t=tau)
            
            
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
            
            for bath_qubit in sim_qr.bath_qubits:
                qml.Hadamard(bath_qubit)
            for idx, tau in enumerate(taus):
               
                hx_array = np.array([params[time_steps]])  # Convert hx to a 1D array
                hy_array = np.array([params[time_steps + 1]])  # Convert hy to a 1D array
                hz_array = np.array([params[time_steps + 2]])  # Convert hz to a 1D array
                J_values = params[time_steps + 3 + idx * num_J : time_steps + 3 + (idx + 1) * num_J]
                current_step = np.concatenate([J_values, hx_array, hy_array, hz_array])
            
                
                total_H = parameterized_ham 
                #print(f"H at time step {idx}: {total_H}")
                qml.evolve(total_H)(current_step, t=tau)
                # Concatenate hx_array with J_values
                
                
                
            return qml.density_matrix(wires=[*ctrl_wires])
        jit_circuit = jax.jit(circuit)
        
        for state_input, state_target in zip(test_in, test_targ):
    
            
            
            
            count+=1
            
            rho_traced = jit_circuit(params,state_input)
            
    
    
            
            fidelity = qml.math.fidelity(rho_traced,state_target)
            fidelities.append(fidelity)
            fidelitity_tot += fidelity
       
    
    
        return fidelities
    if opt_lr == None:
        s = time.time()
        init_loss, init_grads = jax.value_and_grad(cost_func)(params, time_steps, N_reserv, N_ctrl, input_states, target_states)
        e = time.time()
        dt = e - s
        print(f"initial fidelity: {init_loss}, initial_gradients: {init_grads}. Time: {dt}")
        opt_lr,grad_norm = get_initial_learning_rate(init_grads)
        print(f"Adjusted initial learning rate: {opt_lr}. Grad_norm: {1/grad_norm},Grad_norm: {grad_norm}")
        """
        #opt_lr = 0.01
        """

    
    


    print("________________________________________________________________________________")
    print(f"Starting optimization for {gate_name} with optimal lr {opt_lr} time_steps = {time_steps}, N_r = {N_reserv}, N_bath = {num_bath}...\n")

    #opt = optax.novograd(learning_rate=opt_lr)
    opt = optax.adam(learning_rate=opt_lr)
    #opt = optax.chain( optax.clip_by_global_norm(1.0), optax.novograd(learning_rate=opt_lr, b1=0.9, b2=0.1, eps=1e-6))
    
    @jit
    def update(params, opt_state, input_states, target_states):
        loss, grads = jax.value_and_grad(cost_func)(params, time_steps, N_reserv, N_ctrl, input_states, target_states)
        updates, new_opt_state = opt.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss, grads
    
    print("Number of trainable parameters: ", len(params))


    

   

    

    costs = []
    param_per_epoch = []
    print(f"Params: {params}")
    opt_state = opt.init(params)

    time_log_filename = os.path.join(folder_gate, f'times_log_{diable_jit}.txt')

    # Define the gradient function outside the loop
    #cost_and_grad = jax.value_and_grad(partial(cost_func, time_steps=time_steps, N_reserv=N_reserv, N_ctrl=N_ctrl))
    prev_cost = float('inf')  # Initialize with infinity
    threshold_counts = 0
    cost_threshold = 1e-5
    consecutive_threshold_limit = 4  # Number of consecutive epochs below threshold without improvement needed to stop
    backup_params = None
    improvement = True
    backup_cost = float('inf')  
    epoch = 0
    s = time.time()
    full_s =s
    grads_per_epoch = []
    while epoch < num_epochs or improvement:
        params, opt_state, cost,grad = update(params, opt_state, input_states, target_states)

        # Store parameters and cost for analysis
        param_per_epoch.append(params)
        costs.append(cost)
        grads_per_epoch.append(grad)
        
        # Logging
        
        if epoch == 0 or (epoch + 1) % 100 == 0:
            e = time.time()
            epoch_time = e - s
            print(f"step {epoch+1}, cost {cost}, time: {epoch_time}s")
            #print(f"step {epoch+1}, cost {cost},max grad: {max(grad)}, time: {epoch_time}s")
            s = time.time()
        
        # Termination check
        if threshold_counts >= consecutive_threshold_limit:
            print(f"Terminating optimization: cost {cost} is below the threshold {cost_threshold} for {consecutive_threshold_limit} consecutive epochs without improvement.")
            break
        # Check if there is improvement
        if cost < prev_cost:
            prev_cost = cost  # Update previous cost to current cost
            improvement = True
        else:
            improvement = False  # Stop if no improvement

        # Write the time to the file
        #with open(time_log_filename, 'a') as f:
            #f.write(f"Epoch {epoch+1}: {epoch_time} seconds\n")
        
        # Apply tau parameter constraint (must be > 0.0)
        for i in range(time_steps):
            if params[i] < 0:
                params = params.at[i].set(np.abs(params[i]))
                
       
        prev_cost = cost
        if np.abs(max(grad)) < 1e-14:
            break
        epoch += 1 


    
    
    testing_results = final_test(params, time_steps, N_reserv, N_ctrl, test_in, test_targ)
    
    total_tests = len(testing_results)
    avg_fidelity = np.sum(np.asarray(testing_results))/total_tests
    print("\nAverage Final Fidelity: ", avg_fidelity)
    
    data = {'Gate':base64.b64encode(pickle.dumps(gate)).decode('utf-8'),
                'epochs': num_epochs,
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
        
               

                
                
            }
    print(f"Saving results to {filename}")
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
    N_ctrl = 1
    
    trots = [1,2,3,4,5,6,7,8,9,10]
    res = [1,2,3]

    #trots = [9,10]

    
    #trots = [time_steps]
    #res = [N_reserv]
    
    num_epochs = 1000
    N_train = 10
    base_folder = f'./analog_results_trainable_global/noise__opt_cost/'
    bath_factor = 0.1
    #folder = f'./analog_results_trainable_global/trainsize_{N_train}_optimize_trainset/'

    gates_random = []
    baths = [True,True]
    num_baths = [1,2]
    key = jax.random.PRNGKey(10)

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
        key,subkey = jax.random.split(key)
        if True:

            for time_steps in trots:
                key,subkey = jax.random.split(subkey)
                
                
                
                for N_reserv in res:
                    
                    N =N_ctrl+N_reserv
                    
                    #folder = f'./param_initialization/Nc{N_ctrl}_Nr{N_reserv}_dt{time_steps}/fixed_params4/test7/'
                    for num_bath,bath in zip(num_baths,baths):
                        folder = os.path.join(base_folder, f"{num_bath}_num_baths/trainsize_{N_train}/")
                        params_key_seed = gate_idx*1000 * N_reserv + 10569 * time_steps* N_reserv 

                        params_key = jax.random.PRNGKey(params_key_seed)
                        params_key, params_subkey = jax.random.split(params_key)
                        base_key, based_subkey = jax.random.split(params_key)
                        


                        run_test(num_epochs, N_reserv, N_ctrl, time_steps,N_train,folder,gate,gate.name,bath,num_bath,params_subkey,params_key_seed,bath_factor)
