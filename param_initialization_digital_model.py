import pennylane as qml
from jax import numpy as np
from qiskit.circuit.library import *
from qiskit import *
from qiskit.quantum_info import *
import matplotlib.pyplot as plt
from scipy import stats
from pennylane.math import vn_entropy
from pennylane.wires import Wires
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
import pandas as pd
import pickle
import jax
from jax import jit, config
import jax.numpy as jnp
from jax import jit, value_and_grad
import time
import jax.numpy as jnp
from jax import jit, value_and_grad, vmap
import optax
import base64
import os

#os.environ['OPENBLAS_NUM_THREADS'] = '1'
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


def get_target_state(gate, input_state, qubits):
    '''
    Apply the gate to the input state and return the output state.
    '''
    qml.StatePrep(input_state, wires=[*qubits])
    gate(wires=qubits)
    return qml.state()

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


class QuantumReservoirGate:

    def __init__(self, n_rsv_qubits, n_ctrl_qubits, K_coeffs,trotter_steps=1, static=False, bath_params=None):
        self.static = static
        self.ctrl_qubits = Wires(list(range(n_ctrl_qubits))) # wires of the reservoir qubits (i.e. number of qubits in the reservoir)
        self.rsv_qubits = Wires(list(range(n_ctrl_qubits, n_rsv_qubits+n_ctrl_qubits))) # wires of the control qubits (i.e. number of qubits in the control)
        self.all_wires = [*self.ctrl_qubits,*self.rsv_qubits]

        self.dev = qml.device("default.qubit", wires =self.all_wires) 
        self.trotter_steps = trotter_steps

        self.K_coeffs = K_coeffs  # parameter of the reservoir (XY_coupling)
       # print(K_coeffs)
        self.bath_params = bath_params

    def set_gate_reservoir(self):
        
        
        for i, rsv_qubit_i in enumerate(self.rsv_qubits):
            for j, rsv_qubit_j in enumerate(self.rsv_qubits):
                
                if i != j and i < j:
                    k = self.K_coeffs[i, j]
                    
                    
                    #print(f"{i},{j}/ {rsv_qubit_i},{rsv_qubit_j} -> k: {k} ")
                    #print(f"RESERVOIR wires: {[rsv_qubit_i, rsv_qubit_j]}")
                    qml.IsingXY(k, wires=[rsv_qubit_i, rsv_qubit_j])
    
    def set_gate_params(self, x_coeff,z_coeff,y_coeff, J_coeffs):
        for r in self.rsv_qubits:
            qml.RX(x_coeff, wires=r)
            qml.RZ(z_coeff, wires=r)
            qml.RY(y_coeff, wires=r)
        for i,qubit_a in enumerate(self.rsv_qubits):
            for j,qubit_b in enumerate(self.ctrl_qubits):
                #print(f"CONTROL wires: {[self.ctrl_qubits[j],self.rsv_qubits[i]]}")
                qml.IsingXY(J_coeffs[i * len(self.ctrl_qubits) + j], wires=[qubit_a, qubit_b])


# @jit
def get_initial_learning_rate(grads, scale_factor=0.01, min_lr=1e-3, max_lr = 0.075):
    """Estimate a more practical initial learning rate based on the gradient norms."""
    grad_norm = jnp.linalg.norm(grads)
    print(f"grad_norm: {grad_norm} max_grad: {max(np.abs(grads))}")
    
    initial_lr = jnp.where(grad_norm > 0, scale_factor / grad_norm, 0.01)
    print(f"initial_lr: {scale_factor / grad_norm}")
    initial_lr = jnp.clip(initial_lr, min_lr, max_lr)
    return initial_lr, grad_norm

def get_data_quantum_natural_gradient(params, dqfim, density_matrix_grads):
    """
    Computes the Quantum Natural Gradient based on the DQFIM.
    
    params: Current parameters of the circuit.
    dqfim: The DQFIM matrix.
    density_matrix_grads: The gradient of the density matrix with respect to the parameters.
    
    Returns:
    natural_gradient: The quantum natural gradient with respect to the parameters.
    """
    # Invert the DQFIM
    dqfim_inv = jnp.linalg.pinv(dqfim)  # Using pseudo-inverse to avoid issues with ill-conditioning
    
    # Compute the classical gradient
    classical_grads = jnp.array([jnp.trace(density_matrix_grads[i]) for i in range(len(params))])
    
    # Compute the natural gradient by multiplying the inverse of DQFIM with the classical gradient
    natural_gradient = jnp.dot(dqfim_inv, classical_grads)
    
    return natural_gradient
def get_initial_learning_rate_DQFIM(params,qrc,X,gate,init_grads, scale_factor=0.1, min_lr=9e-5, max_lr = 1.0):
    """
    Compute an initial learning rate based on the norm of gradients.
    """
    ctrl_qubits = qrc.ctrl_qubits # wires of the reservoir qubits (i.e. number of qubits in the reservoir)
    rsv_qubits = qrc.rsv_qubits# wires of the control qubits (i.e. number of qubits in the control)
    grad_norm = jnp.linalg.norm(init_grads)
    print(f"grad_norm: {grad_norm} max_grad: {max(np.abs(init_grads))}")
    dev =qrc.dev

    
    dev_data = qml.device('default.qubit', wires=ctrl_qubits)
    circuit = qml.QNode(get_target_state, device=dev_data, interface='jax')
    
    # Execute the circuit for each input state
    L = np.stack([np.asarray(circuit(gate, x, ctrl_qubits)) for x in X])
    

    @jax.jit
    @qml.qnode(dev,interface='jax',diff_method="backprop")
    def circuit(params,input_state):
        x_coeff = params[0]
        z_coeff = params[1]
        y_coeff = params[2]
        J_coeffs = params[3:]

        #print(J_coeffs)
        #qml.StatePrep(test_state, wires=[*ctrl_qubits])
        qml.StatePrep(input_state, wires=[*qrc.ctrl_qubits])
        
        for i in range(trotter_steps):
            qrc.set_gate_reservoir()
            if static or trotter_steps==1:
                # print(f"J_coeffs: {J_coeffs}")
                qrc.set_gate_params(x_coeff,z_coeff,y_coeff, J_coeffs)
            else:
                step = len(rsv_qubits)*len(ctrl_qubits)
                qrc.set_gate_params(x_coeff,z_coeff,y_coeff,  J_coeffs[i*step:(i+1)*step])
        return qml.density_matrix(wires=[*ctrl_qubits])
    jit_circuit = jax.jit(circuit)
        
    def get_density_matrix_sum(params, input_states, jit_circuit):
        """
        Computes the sum of density matrices after applying the PQC on all training states using a pre-jitted circuit.
        
        input_states: A batch of training states (|\psi_l\rangle).
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
            density_matrix_sum += out
        
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
    dqfim_eigvals,dqfim_eigvecs, DQFIM, entropies,density_matrix_grads,Pi_L,trace_dqfim =compute_qfim_eigval_decomp(params)
    
    initial_lr = scale_factor / (jnp.real(trace_dqfim) * grad_norm + 1e-12)
    print(f"Initial (Tr(dqfim*grad_norm)^-1) lr: {initial_lr}")


    initial_lr = jnp.clip(initial_lr, min_lr, max_lr)
    
    # print(f"lambda_max: {lambda_max}")
    # # print(f"Initial learning rate: {initial_lr}")
    
    return initial_lr, {"dqfim_eigvals": dqfim_eigvals,"dqfim_eigvecs": dqfim_eigvecs, "DQFIM": DQFIM,"entropies": entropies}
def calculate_gradient_stats(gradients):
    mean_grad = jnp.mean(gradients, axis=0)
    mean_grad_squared = jnp.mean(gradients ** 2, axis=0)
    var_grad = mean_grad_squared - mean_grad ** 2
    grad_norm = jnp.linalg.norm(mean_grad)
    return mean_grad, var_grad, grad_norm
def get_initial_lr_per_param(grads, base_step=0.001, min_lr=1e-3, max_lr=0.2):
    # print(f"grads: {grads}")
    grad_magnitudes = jax.tree_util.tree_map(lambda g: jnp.abs(g) + 1e-12, grads)
    # print(f"grad_magnitudes: {grad_magnitudes}")
    lr_tree = jax.tree_util.tree_map(lambda g: base_step / g, grad_magnitudes)
    # print(f"lr_tree: {lr_tree}")
    lr_tree = jax.tree_util.tree_map(lambda lr: jnp.clip(lr, min_lr, max_lr), lr_tree)
    return lr_tree

def run_experiment(params, steps, n_rsv_qubits, n_ctrl_qubits,  K_coeffs, trotter_steps, static, dataset_key, gate, gate_name, folder, test_size, training_size,opt_lr,L,qfim):
    bath = False
    dqfim_data = None
    init_params = params
    X, y =generate_dataset(gate, N_ctrl,training_size= training_size, key= dataset_key, new_set=False)

    test_dataset_key = jax.random.split(dataset_key)[1]
    test_X, test_y = generate_dataset(gate, N_ctrl,training_size= 2000, key=test_dataset_key, new_set=False)
   
    
    print(f"N_r = {n_rsv_qubits}, N_ctrl = {n_ctrl_qubits}")
    
    #print(f"K_coeffs {type(K_coeffs)}: {K_coeffs}")
    # print(f"X shape: {X.shape}")
    # print(f"test_X shape: {test_X.shape}")
    qrc = QuantumReservoirGate(n_rsv_qubits=n_rsv_qubits, n_ctrl_qubits=n_ctrl_qubits, K_coeffs=K_coeffs,trotter_steps=trotter_steps, static=static)
    @qml.qnode(qrc.dev, interface="jax",diff_method="backprop")
    def circuit(params, input_state):
        x_coeff = params[0]
        z_coeff = params[1]
        y_coeff = params[2]
        J_coeffs = params[3:]
        qml.StatePrep(input_state, wires=[*qrc.ctrl_qubits])
        for i in range(trotter_steps):
            
            qrc.set_gate_reservoir()
            
            if qrc.static or trotter_steps==1:
                qrc.set_gate_params(x_coeff,z_coeff,y_coeff, J_coeffs)
            else:
                step = len(qrc.rsv_qubits)*len(qrc.ctrl_qubits)
                qrc.set_gate_params(x_coeff,z_coeff,y_coeff,  J_coeffs[i*step:(i+1)*step])
            
        return qml.density_matrix(wires=[*qrc.ctrl_qubits])

        
    vcircuit = jax.vmap(circuit, in_axes=(None, 0))
    def batched_cost_helper(params, X, y):
        # Process the batch of states
        batched_output_states = vcircuit(params, X)
        
        # Compute fidelity for each pair in the batch and then average
        fidelities = jax.vmap(qml.math.fidelity)(batched_output_states, y)
        # average_fidelity = np.sum(fidelities)/len(fidelities)
        fidelities = jnp.clip(fidelities, 0.0, 1.0)
        average_fidelity = jnp.mean(fidelities)
        # print(f"batched_cost_helper - average_fidelity dtype: {average_fidelity.dtype}")
        return 1 - average_fidelity  # Minimizing infidelity
        
    @partial(jit, static_argnums=(3, 4, 5, 6))
    def cost_func(params,X, y, n_rsv_qubits, n_ctrl_qubits, trotter_steps, static):
        params = jnp.asarray(params, dtype=jnp.float64)
        X = jnp.asarray(X, dtype=jnp.complex128)
        y = jnp.asarray(y, dtype=jnp.complex128)
        loss = batched_cost_helper(params, X, y)
        # print(f"cost_func - loss dtype: {loss.dtype}")
        loss = jnp.maximum(loss, 0.0)  # Apply the cutoff to avoid negative costs

        return loss
    
   
    
    def final_costs(params, X, y, n_rsv_qubits=None, n_ctrl_qubits=None, trotter_steps=None, static=None):
        
        batched_output_states = vcircuit(params, X)
        
        fidelities = jax.vmap(qml.math.fidelity)(batched_output_states, y)
        fidelities = jnp.clip(fidelities, 0.0, 1.0)
        return fidelities

    if opt_lr == None:
        s = time.time()
        init_loss, init_grads = jax.value_and_grad(cost_func)(params, X, y, n_rsv_qubits, n_ctrl_qubits, trotter_steps, static)
        e = time.time()
        dt = e - s
        # print(f"Initial gradients dtype: {init_grads.dtype}, Initial loss dtype: {init_loss.dtype}")
        # print(f"initial fidelity: {init_loss}, initial_gradients: {np.mean(np.abs(init_grads))}. Time: {dt}")
        # print(f"initial fidelity: {init_loss}, initial_gradients: {init_grads}. Time: {dt}")
        # opt_lr, raw_lr = get_initial_learning_rate(init_grads)
        opt_lr = get_initial_lr_per_param(init_grads)
        # print(f"Adjusted initial learning rate: {opt_lr:.4f}.")
        cost = init_loss
    opt_descr = 'per param'
    learning_rate_schedule = optax.constant_schedule(opt_lr)
    opt = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.inject_hyperparams(optax.adam)(learning_rate=opt_lr, b1=0.99, b2=0.999, eps=1e-7),
        )
    print(f"per param lrs: \n{opt_lr}\n")
    print("________________________________________________________________________________")
    print(f"Starting optimization for {gate_name} with {len(X)} training states,  trots = {trotter_steps}, N_r = {n_rsv_qubits}  with avg optimal lr ({np.mean(opt_lr)}) time_steps = {trotter_steps}, N_r = {n_rsv_qubits}...\n")
    print(f"Initial Loss: {init_loss:.4f}, initial_gradients: {np.mean(np.abs(init_grads))}. Time: {dt:.2e}")
    

    cost_threshold= 1e-05
    conv_tol = 1e-08
    prev_cost = float('inf')  # Initialize with infinity
    threshold_counts = 0
  
    backup_cost = float('inf')  

    cost_res = 1
    costs,grads_per_epoch = [],[]
    epoch = 0
    improvement = True
    opt_state = opt.init(params)
    @jit
    def update(params, opt_state, X, y, value):
        # Ensure inputs are float64
        params = jnp.asarray(params, dtype=jnp.float64)
        X = jnp.asarray(X, dtype=jnp.complex128)
        y = jnp.asarray(y, dtype=jnp.complex128)
        
        loss, grads = jax.value_and_grad(cost_func)(params, X, y, n_rsv_qubits, n_ctrl_qubits, trotter_steps, static)
        if not isinstance(opt_state[-1], optax.contrib.ReduceLROnPlateauState):
            updates, opt_state = opt.update(grads, opt_state, params)
        else:
            updates, opt_state = opt.update(grads, opt_state, params=params, value=value)
        new_params = optax.apply_updates(params, updates)
        
        # Ensure outputs are float64
        loss = jnp.asarray(loss, dtype=jnp.float64)
        grads = jnp.asarray(grads, dtype=jnp.float64)
        
    
        return new_params, opt_state, loss, grads
    

    fullstr = time.time()
    improvement_count = 0
    a_threshold, acceleration =  0.0, 0.0
    threshold_cond1, threshold_cond2 = [],[]
    a_condition_set = False
    a_threshold =  0.0
    stored_epoch = None
    false_improvement = False
    # Introduce tracking for barren plateaus

    scale_reduction_epochs = []  # Track epochs where scale is reduced
    scales_per_epoch = []  # Store scale values per epoch
    new_scale = 1.0  # Initial scale value
    learning_rates = []
    # print(f"params: {type(params)}, {params.dtype}")
    num_reductions = 0
    new_scale = 1.0
    while epoch  < steps or improvement:
        params, opt_state, cost,grad_circuit = update(params, opt_state, X, y,value=cost)
        grad =grad_circuit
        costs.append(cost)
        grads_per_epoch.append(grad_circuit)

        if epoch == 0 or (epoch + 1) % 200 == 0:
            
            var_grad = np.var(grad,ddof=1)
            mean_grad = np.mean(jnp.abs(grad))
            max_grad = max(jnp.abs(grad))
            e = time.time()
            epoch_time = e - s
            print(f'Epoch {epoch + 1} --- cost: {cost:.5f}, '
                #   f'a: {acceleration:.2e} '
                # f'Var(grad): {var_grad:.1e}, '
                # f"opt_state[1]['learning_rate']= {opt_state[1].hyperparams['learning_rate']}",
                # f'GradNorm: {np.linalg.norm(grad):.1e}, '
                 f'Mean(grad): {mean_grad:.1e}, '
                 f'Max(grad): {max_grad:.1e}, '
                f'[t: {epoch_time:.1f}s]')
        # Check if there is improvement
        if cost < prev_cost:
            prev_cost = cost  # Update previous cost to current cost
            # improvement = True

            current_cost_check = cost_func(params, X, y,n_rsv_qubits, n_ctrl_qubits, trotter_steps, static)
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
        var_condition= np.var(grad_circuit,ddof=1) < 1e-14
        gradient_condition= max(jnp.abs(grad)) < 1e-8
        cost_confition = cost < 1e-10
        
        if var_condition or gradient_condition or epoch >=2*steps or cost_confition:
            if epoch >=2*steps:
                print(f"Epoch greater than max. Ending opt at epoch: {epoch}")
            if var_condition:
                print(f"Variance of the gradients below threshold [{np.var(grad_circuit,ddof=1):.1e}], thresh:  1e-10. Ending opt at epoch: {epoch}")
            if cost_confition:
                print(f"Cost below threshold [{cost:.1e}]. Ending opt at epoch: {epoch}")
            if gradient_condition:
                print(f"Magnitude of maximum gradient is less than threshold [{max_grad:.1e}]. Ending opt at epoch: {epoch}")

            break
        # Check if there is improvement
        second_prev_cost = prev_cost  # Move previous cost back one step
        prev_cost = cost  # Update previous cost with the current cost

     


        epoch += 1
    if backup_cost < cost:
        print(f"backup cost (epoch: {backup_epoch}) is better with: {backup_cost:.2e} <  {cost:.2e}: {backup_cost < cost}")
        # print(f"recomputed cost (i.e. cost_func(backup_params,input_states, target_states)): {cost_func(backup_params,input_states, target_states)}")
        # print(f"cost_func(params, input_states,target_states): {cost_func(params, input_states,target_states)}")
        # print(f"final_test(backup_params,test_in, test_targ): {final_test(backup_params,test_in, test_targ)}")
        # print(f"final_test(params,test_in, test_targ): {final_test(params,test_in, test_targ)}")
        params = backup_params
    fullend = time.time()
    print(f"time optimizing: {fullend-fullstr} improvement count: {improvement_count}")
    df = pd.DataFrame()
    
    print(f"Testing opt params against {test_size} new random states...")

    
    fidelities =  final_costs(params, X=test_X, y=test_y, n_rsv_qubits=n_rsv_qubits, n_ctrl_qubits=n_ctrl_qubits, trotter_steps=trotter_steps, static=static)
    # fidelities =  final_fidelities(params, X=test_X, y=test_y,V=gate, n_rsv_qubits=n_rsv_qubits, n_ctrl_qubits=n_ctrl_qubits, trotter_steps=trotter_steps, static=static)
    #print('Prev fidelity: ', np.mean(tempt_infidelies))
    infidelities = 1.00000000000000-fidelities
    # print(f"infidelities dtype: {np.array(infidelities).tolist()[0].dtype}")
    avg_infidelity = np.mean(infidelities)
    # print(f"avg_infidelity dtype: {avg_infidelity.dtype}")
    avg_fidelity = np.mean(fidelities)
    print(f'Avg Fidelity: {avg_fidelity:.5f}')
   # infidelities_backup =  final_costs(backup_params, X=test_X, y=test_y, n_rsv_qubits=n_rsv_qubits, n_ctrl_qubits=n_ctrl_qubits, trotter_steps=trotter_steps, static=static)
  #  print('Backup params infidelity: ', np.mean(infidelities_backup))
    x_coeff = params[0]
    z_coeff = params[1]
    y_coeff = params[2]
    J_coeffs = params[3:]
    
    data = {
        'Gate': base64.b64encode(pickle.dumps(gate)).decode('utf-8'),
        'opt_description': opt_descr,
        'epochs': steps,
        # 'selected_indices': selected_indices,
        'lrs': learning_rates,
        'scales_per_epoch': scales_per_epoch,  # Add scales per epoch
        'scale_reduction_epochs': scale_reduction_epochs,  # Add epochs of scale reduction
        'init_params': init_params,
        # 'preopt_results': preopt_results,
        'grads_per_epoch': grads_per_epoch,
        'opt_lr': opt_lr,
        'trotter_step': trotter_steps,
        'controls': n_ctrl_qubits,
        'reservoirs': n_rsv_qubits,
        'x_coeff': np.array(x_coeff).item(),  # Convert JAX array to Python float
        'J_coeffs': np.array(J_coeffs).tolist(),  # Convert to list of Python floats
        'y_coeff': np.array(y_coeff).item(),
        'z_coeff': np.array(z_coeff).item(),
        'K_coeffs': np.array(K_coeffs).tolist(),

        'costs': np.array(costs).tolist(),
        'backup_cost': np.array(backup_cost).item(),
        'backup_params': np.array(backup_params).tolist(),
        'avg_fidelity': np.array(avg_fidelity).item(),
        'avg_infidelity': np.array(avg_infidelity).item(),
        'test_results': np.array(infidelities).tolist(),
        'testing_results': np.array(fidelities).tolist(),
        'training_size': training_size,
        'X': np.array(X).tolist(),
        'y': np.array(y).tolist(),
        'bath': bath,
        'static': static,
    }
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
                        fixed_params = results['K_coeffs']
                        try:
                            jacobian = results['jacobian']
                            det = results['det']
                        except KeyError:
                            jacobian = None
                            det = None
                        # entropy = results['entropy']
                        qfim = results['qfim']
                        params = results['trainable_params']
                        entropy = results['entropy']
                       
                        return qfim_eigvals,fixed_params,params,qfim,entropy
    
    print("QFIM eigenvalues not found for the given parameter keys.")
    return None,None,None, None,None

  
if __name__=='__main__':

# --------------------------------------------------------------------------------------------
    
    

    

    steps = 1500
    training_size_list = [10]
    noise_central = 0.1
    noise_range = 0.002
    test_size = 2000
# --------------------------------------------------------------------------------------------
    

    

    #folder = './results_gate_model_dec_26_full_001/' #save the results in this folder
    
    
    
    static_options = [False]
    bath_qubit = [False]
    training_size = training_size_list[0]
    parameters = []
    
    N_r = 1
    trotter_steps = 8
  
    bath = False
    static = False
    N_ctrl = 2

    folder = f'./param_initialization_final/digital_results/Nc_{N_ctrl}/'
    gates_random = []
    for i in range(30):
        U = random_unitary(2**N_ctrl, i).to_matrix()
        g = partial(qml.QubitUnitary, U=U)
        g.num_wires = N_ctrl
        g.name = f'U{N_ctrl}_'+str(i)
        gates_random.append(g)
    
    gates =   gates_random 

 
    fp_idx = 0
   
    trainable_param_keys = [f'test{i}' for i in range(100)]
    # trainable_param_keys = ['test6','test82','test79','test66','test13','test4'] trot = 10
    # trainable_param_keys = ['test79', 'test47','test46', 'test2','test32','test27','test45','test14','test84']
    # print(len(fixed_param_keys), len(trainable_param_keys))
    fixed_param_name='fixed_params0'
    all_gates = gates_random
    base_state = 'GHZ_state'
    state = 'GHZ'
    sample_range_label = 'pi'
    
    K_0 = '1'
    opt_lr = None
    delta_x = 1.49011612e-08
    threshold = 1.e-14
    all_gates = gates_random
    for gate_idx,gate in enumerate(all_gates):
        
        for test_key in trainable_param_keys:
            
            folder_gate = os.path.join(
                folder,
                f"reservoirs_{N_r}",
                f"trotter_{trotter_steps}",
                f"trainsize_{training_size}",
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
            params_key_seed = gate_idx*121 * N_r + 12345 * trotter_steps *N_r
            print(f"params_key_seed: {params_key_seed}")
            params_key = jax.random.PRNGKey(params_key_seed)
            dataset_seed = N_ctrl * gate_idx + gate_idx**2 + N_ctrl
            dataset_key = jax.random.PRNGKey(dataset_seed)

            
            print("________________________________________________________________________________")
            filename = os.path.join(folder_gate, f'data_run_{len(files_in_folder)}.pickle')
            # filename = os.path.join(folder_gate, f'{test_key}.pickle')
            
            N = N_ctrl + N_r
            
            qfim_base_path = f'/Users/sophieblock/QRCCapstone/parameter_analysis_directory/QFIM_results/gate/Nc_{N_ctrl}/sample_{sample_range_label}/{K_0}xK'
            # qfim_file_path = Path(qfim_base_path) / f'Nr_{N_r}' / f'trotter_step_{trotter_steps}' /  f'data2.pickle'

            qfim_file_path = Path(qfim_base_path) / f'Nr_{N_r}' / f'trotter_step_{trotter_steps}' /  f'data.pickle'
            print(qfim_file_path)
            print(f"{test_key} params (range: (-{sample_range_label}, {sample_range_label}))")
            print(f"{fixed_param_name}")

            eigvals, K_coeffs,params,qfim,entropy = get_qfim_eigvals(qfim_file_path, fixed_param_name, test_key)
            
            trace_qfim = np.sum(eigvals)
            var_qfim = np.var(eigvals)
            print(f"QFIM trace: {trace_qfim}")
            print(f"QFIM var: {var_qfim}")


            data = run_experiment(params = params,steps =  steps,n_rsv_qubits= N_r,n_ctrl_qubits= N_ctrl,K_coeffs= K_coeffs,trotter_steps= trotter_steps,static= static, dataset_key=dataset_key, gate=gate,gate_name= gate.name,folder= folder,test_size= test_size,training_size= training_size,opt_lr=opt_lr, L=None,qfim=qfim)
            data['QFIM Results'] = {"qfim_eigvals":eigvals,
                                    "trainable_params": params,
                                    "qfim": qfim,
                                    "entropy":entropy,
            'variance':var_qfim,
            'trace':trace_qfim,

                }
            
            df = pd.DataFrame([data])
            while os.path.exists(filename):
                name, ext = filename.rsplit('.', 1)
                filename = f"{name}_.{ext}"

            with open(filename, 'wb') as f:
                pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved to path: {filename}")

            