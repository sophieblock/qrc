import pennylane as qml
import os
from pennylane.pulse import constant, pwc
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
    

    def get_total_hamiltonian_components_new(self):
        coefficients = []
        operators = []

        
        # Add h_x, h_y, and h_z terms for the reservoir qubits
        coefficients.append(qml.pulse.constant)  # h_x
        operators.append(sum(qml.PauliX(wires=r) for r in self.reserv_qubits))

        coefficients.append(qml.pulse.constant)  # h_y
        operators.append(sum(qml.PauliY(wires=r) for r in self.reserv_qubits))

        coefficients.append(qml.pulse.constant)  # h_z
        operators.append(sum(qml.PauliZ(wires=r) for r in self.reserv_qubits))
        
        # Add XY coupling terms for each control-reservoir pair
        for i, qubit_a in enumerate(self.reserv_qubits):
            for j, qubit_b in enumerate(self.ctrl_qubits):
                coefficients.append(qml.pulse.constant)  # Use constant for J coefficients
                operators.append(self.get_XY_coupling(qubit_a, qubit_b))  # Add XY coupling operator

        
        # Construct the dynamic Hamiltonian
        H_dynamic = qml.dot(coefficients, operators)

        # Construct the static Hamiltonian
        static_coefficients = [
            self.k_coefficient[qa, qb]
            for qa in range(len(self.reserv_qubits))
            for qb in range(len(self.reserv_qubits))
            if qa != qb
        ]
        static_operators = [
            self.get_XY_coupling(self.reserv_qubits[qa], self.reserv_qubits[qb])
            for qa in range(len(self.reserv_qubits))
            for qb in range(len(self.reserv_qubits))
            if qa != qb
        ]
        if self.N_reserv == 1 and self.bath == False:
            total_H = H_dynamic
        
        else:
            H_static = qml.dot(static_coefficients, static_operators)
            total_H = H_dynamic+H_static
        
        return total_H
    
    def get_total_hamiltonian_components_const_global(self):
        coefficients = []
        operators = []

        
        # Add h_x, h_y, and h_z terms for the reservoir qubits
        coefficients.append(qml.pulse.constant)  # h_x
        operators.append(sum(qml.PauliX(wires=r) for r in self.reserv_qubits))

        coefficients.append(qml.pulse.constant)  # h_y
        operators.append(sum(qml.PauliY(wires=r) for r in self.reserv_qubits))

        coefficients.append(qml.pulse.constant)  # h_z
        operators.append(sum(qml.PauliZ(wires=r) for r in self.reserv_qubits))
        
        # Add XY coupling terms for each control-reservoir pair
        for i, qubit_a in enumerate(self.reserv_qubits):
            for j, qubit_b in enumerate(self.ctrl_qubits):
                coefficients.append(qml.pulse.constant)  # Use constant for J coefficients
                operators.append(self.get_XY_coupling(qubit_a, qubit_b))  # Add XY coupling operator

        
        # Construct the dynamic Hamiltonian
        H_dynamic = qml.dot(coefficients, operators)

        # Construct the static Hamiltonian
        static_coefficients = [
            self.k_coefficient[qa, qb]
            for qa in range(len(self.reserv_qubits))
            for qb in range(len(self.reserv_qubits))
            if qa != qb
        ]
        static_operators = [
            self.get_XY_coupling(self.reserv_qubits[qa], self.reserv_qubits[qb])
            for qa in range(len(self.reserv_qubits))
            for qb in range(len(self.reserv_qubits))
            if qa != qb
        ]
        if self.N_reserv == 1 and self.bath == False:
            total_H = H_dynamic
        
        else:
            H_static = qml.dot(static_coefficients, static_operators)
            total_H = H_dynamic+H_static
        
        return total_H
    
    def get_static_components(self):
        coefficients=[]
        operators = []
        
        # Add h_x, h_y, and h_z terms for the reservoir qubits
        coefficients.append(qml.pulse.constant)  # h_x
        operators.append(sum(qml.PauliX(wires=r) for r in self.reserv_qubits))

        coefficients.append(qml.pulse.constant)  # h_y
        operators.append(sum(qml.PauliY(wires=r) for r in self.reserv_qubits))

        coefficients.append(qml.pulse.constant)  # h_z
        operators.append(sum(qml.PauliZ(wires=r) for r in self.reserv_qubits))
        return qml.dot(coefficients, operators)
    
    def get_static_ops(self):
        # coefficients=[]
        operators = []
        

        operators.append(sum(qml.PauliX(wires=r) for r in self.reserv_qubits))


        operators.append(sum(qml.PauliY(wires=r) for r in self.reserv_qubits))


        operators.append(sum(qml.PauliZ(wires=r) for r in self.reserv_qubits))
        return operators

    def get_total_hamiltonian_components_no_global(self):
        coefficients = []
        operators = []

        # Add XY coupling terms for each control-reservoir pair
        for i, qubit_a in enumerate(self.reserv_qubits):
            for j, qubit_b in enumerate(self.ctrl_qubits):
                coefficients.append(qml.pulse.constant)  # Use constant for J coefficients
                operators.append(self.get_XY_coupling(qubit_a, qubit_b))  # Add XY coupling operator

        # Construct the dynamic Hamiltonian (without global fields)
        H_dynamic = qml.dot(coefficients, operators)

        # Construct the static Hamiltonian
        static_coefficients = [
            self.k_coefficient[qa, qb]
            for qa in range(len(self.reserv_qubits))
            for qb in range(len(self.reserv_qubits))
            if qa != qb
        ]
        static_operators = [
            self.get_XY_coupling(self.reserv_qubits[qa], self.reserv_qubits[qb])
            for qa in range(len(self.reserv_qubits))
            for qb in range(len(self.reserv_qubits))
            if qa != qb
        ]
        if self.N_reserv == 1 and not self.bath:
            total_H = H_dynamic
        else:
            H_static = qml.dot(static_coefficients, static_operators)
            total_H = H_dynamic + H_static

        return total_H
    

   
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
        coefficients.append(hx_func)
        new_operator = sum((qml.PauliX(wires=r) for r in self.reserv_qubits))      
        operators.append(new_operator)

        coefficients.append(hy_func)
        new_operator = sum((qml.PauliY(wires=r) for r in self.reserv_qubits))      
        operators.append(new_operator)

        coefficients.append(hz_func)
        new_operator = sum((qml.PauliZ(wires=r) for r in self.reserv_qubits))      
        operators.append(new_operator)

        # Add XY coupling terms
        
        for i,qubit_a in enumerate(self.reserv_qubits):
            for j,qubit_b in  enumerate(self.ctrl_qubits):
                idx = j * self.N_reserv + (i  - self.N_ctrl)
                # Lambda function to capture the current parameter and time
                coefficients.append(J_func)
                
                new_operator = self.get_XY_coupling(qubit_a,qubit_b)
                
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


def get_rate_of_improvement(cost, prev_cost,second_prev_cost):
    
    prev_improvement = prev_cost - second_prev_cost
    current_improvement = cost - prev_cost
    acceleration = prev_improvement - current_improvement

    return acceleration
def run_test(params, num_epochs, N_reserv, N_ctrl, time_steps,N_train,folder,gate,gate_name,bath,num_bath,init_params_dict, dataset_key):
    float32=''
    opt_lr = 0.1
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
    
    k = 8
   
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
    # print(f"training state #1: {input_states[0]}")
    print(f"params: {params}")


    test_dataset_key = jax.random.split(dataset_key)[1]
    test_in, test_targ = generate_dataset(gate, N_ctrl,training_size= 2000, key=test_dataset_key, new_set=False)
    

    # parameterized_ham = sim_qr.get_total_hamiltonian_components_no_global()
    parameterized_ham = sim_qr.get_total_hamiltonian_components_new()
    # res_ham = sim_qr.get_static_components()
    print(f"Res-Ctrl Ham: {parameterized_ham}")
    # print(f"Res Ham: {res_ham}")
    
    ctrl_wires = sim_qr.get_ctrl_wires()
    reserv_wires = sim_qr.get_reserv_wires()
    qnode_dev = sim_qr.get_dev()
    all_wires = sim_qr.get_all_wires()
    res_ops = sim_qr.get_static_ops()
    # print(f"res_ops: {res_ops}")

    # @qml.qnode(qnode_dev, interface="jax")
    # def circuit(params, state_input):
    #     #circuit1
    #     taus = params[:time_steps]
    #     qml.StatePrep(state_input, wires=[*ctrl_wires])

    #     # cumulative_time = 0
    #     res_ham = qml.Hamiltonian(params[time_steps:time_steps+3],res_ops)
    #     # U_res =qml.ops.Evolution(res_ham,)
    #     # print(f"U_res: {res_ham}")
    #     for idx, tau in enumerate(taus):
    #         qml.evolve(res_ham,tau)
    #         # qml.ops.Evolution(res_ham,tau)
    #         # qml.ops.Evolution(res_ham,cumulative_time+tau)
    #         J_values = params[time_steps + 3 + idx * num_J : time_steps + 3 + (idx + 1) * num_J]
    #         # print(J_values.shape)
            
    #         # Apply dynamic Hamiltonian using qml.evolve
    #         current_step = np.concatenate([J_values])
    #         # print(current_step.shape)
    #         # qml.evolve(parameterized_ham)(current_step, t=cumulative_time + tau)
    #         qml.evolve(parameterized_ham)(current_step, t=[cumulative_time,cumulative_time + tau])
    #         cumulative_time += tau

    #     return qml.density_matrix(wires=[*ctrl_wires])
    @qml.qnode(qnode_dev, interface="jax")
    def circuit(params,state_input):
            # circuit2
        taus = params[:time_steps]

        qml.StatePrep(state_input, wires=[*ctrl_wires])
        hx_array = np.array([params[time_steps]])  # Convert hx to a 1D array
        hy_array = np.array([params[time_steps + 1]])  # Convert hy to a 1D array
        hz_array = np.array([params[time_steps + 2]])  # Convert hz to a 1D array
        cumulative_time = 0
        for idx, tau in enumerate(taus):
           
            
            J_values = params[time_steps + 3 + idx * num_J : time_steps + 3 + (idx + 1) * num_J]

            # Concatenate hx_array with J_values
            current_step = np.concatenate([hx_array,hy_array,hz_array,J_values])
            # qml.evolve(parameterized_ham)(current_step, t=[cumulative_time,cumulative_time + tau],atol=1e-6,rtol=1e-6)
            qml.evolve(parameterized_ham)(current_step, t=tau,atol=1e-7,rtol=1e-6)
            cumulative_time += tau
            
        return qml.density_matrix(wires=[*ctrl_wires])
    specs_func = qml.specs(circuit)
    specs = specs_func(params,input_states[0])
    # print(f"specs: {specs}")
    # print(specs['resources'])
    circuit_depth = specs['resources'].depth
    gate_count = specs['resources'].num_gates
    print(f"Depth: {circuit_depth}, gates: {gate_count}")
    # print(f'Specs resources: {specs["resources"]}')
    output = circuit(params,input_states[0])
    # print(output)
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
    def update(params, opt_state, input_states, target_states, value):
        """Update all parameters including tau."""
       
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
        print(f"initial fidelity: {init_loss:.4f}, initial_gradients: {np.mean(np.abs(init_grads))}. Time: {dt:.2e}")
        opt_lr,grad_norm = get_initial_learning_rate(init_grads)
        print(f"Adjusted initial learning rate: {opt_lr:.2e}. Grad_norm: {1/grad_norm},Grad_norm: {grad_norm:.2e}")
        cost = init_loss
    else:
        s = time.time()
        init_loss, init_grads = jax.value_and_grad(cost_func)(params, input_states, target_states)
        e = time.time()
        cost = init_loss
        epoch_time = e-s

    


    print("________________________________________________________________________________")
    print(f"Starting optimization for {gate_name}(epochs: {num_epochs}) with optimal lr {opt_lr} time_steps = {time_steps}, N_r = {N_reserv}, N_bath = {num_bath}...\n")

    """
    case #1
    """
    opt_descr = 'case 1'
    learning_rate_schedule = optax.constant_schedule(opt_lr)
    opt = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate_schedule, b1=0.99, b2=0.999, eps=1e-7),
        )
    
    
    
    costs = []
    param_per_epoch,grads_per_epoch = [],[]
   # print(f"Params: {params}")
    
    opt_state = opt.init(params)
    # print(f"initial opt_state: {opt_state}")

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
   
    epoch = 0
    s = time.time()
    full_s =s
    
    a_condition_set = False
    a_threshold =  0.0

    threshold_cond1, threshold_cond2 = [],[]
    false_improvement = False
    backup_epoch=0
    learning_rates = []  # Track epochs where scale is reduced
    time_per_epoch = [epoch_time]
    while epoch < num_epochs or improvement:

        params, opt_state, cost, grad = update(params, opt_state, input_states, target_states,value=cost)
        if 'learning_rate' in opt_state[1].hyperparams:
            plateau_scale = 1.0
            learning_rate = opt_state[1].hyperparams['learning_rate']
            learning_rates.append(learning_rate)
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
        e = time.time()
        epoch_time = e - s
        time_per_epoch.append(epoch_time)
        s = time.time()
        if epoch == 0 or (epoch + 1) % 100 == 0:
            var_grad = np.var(grad,ddof=1)
            mean_grad = np.mean(jnp.abs(grad))
            
            # learning_rate = opt_state[1].hyperparams['learning_rate']
         
            # print(f'Epoch {epoch + 1} --- cost: {cost:.5f}, lr: {learning_rates[-1]:.2e}, scale: {plateau_scale}'
            print(f'Epoch {epoch + 1} --- cost: {cost:.5f}, '
                #   f'grad: {grad}'
                 f'Mean(grad): {mean_grad:.1e}, '
                f'[t: {epoch_time:.1f}s]')

        
            

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
       

        if np.abs(max(grad)) < 1e-14 or np.var(grad,ddof=1) < 1e-10:
            print(f"max(grad)<1e-14. breaking....")
            break
        epoch += 1  # Increment epoch count


    if backup_cost < cost and not epoch < num_epochs:
        print(f"backup cost (epoch: {backup_epoch}) is better with: {backup_cost:.2e} <  {cost:.2e}: {backup_cost < cost}")

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
            'opt_description': opt_descr,
                'epochs': num_epochs,
                'lrs': learning_rates,
                'time_per_epoch':time_per_epoch,
                'rocs':rocs,
                'min_var_indices':min_var_indices,
                'replacement_indices':replacement_indices,
                'a_marked': a_marked,
       
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
    N_ctrl = 2
   
   

    res = [1]
    trots = [6]

    
    




    num_epochs = 500
    N_train = 10
    add=0
   
    
    folder = f'./analog_results_trainable_global/trainsize_{N_train+add}_epoch{num_epochs}_pwc/'
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

        if not gate_idx in [1]:
            continue

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
                    # params = jnp.asarray([0.4033546149730682, 1.4487122297286987, 2.3020467758178711, 2.9035964012145996, 0.9584765434265137, 1.7428307533264160, -1.3020169734954834, -0.8775904774665833, 2.4736261367797852, -0.4999605417251587, -0.8375297188758850, 1.7014273405075073, -0.8763229846954346, -3.1250307559967041, 1.1915868520736694, -0.4640290737152100, -1.0656110048294067, -2.6777451038360596, -2.7820897102355957, -2.3751690387725830, 0.1393062919378281])
                    print(f"time_step_params: {time_step_params}")



                    run_test(params, num_epochs, N_reserv, N_ctrl, time_steps,N_train,folder,gate,gate.name,bath,num_bath,init_params_dict = init_params_dict,dataset_key = dataset_key)
