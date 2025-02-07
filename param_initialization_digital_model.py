import pennylane as qml
from jax import numpy as np
from qiskit.circuit.library import *
from qiskit import *
from qiskit.quantum_info import *
import matplotlib.pyplot as plt
import time
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

def generate_dataset(gate, n_qubits, training_size,testing_size, key, L=[]):
    '''
    Generate the dataset of input and output states according to the gate provided.
    Uses a seed for reproducibility.
    '''
    
    if len(L) == 0:
        X = []
        # Split the key into subkeys for the full training size
        keys = jax.random.split(key, num=training_size+testing_size)
        
        # Loop through the subkeys and generate the dataset
        for i, subkey in enumerate(keys):
            subkey = jax.random.fold_in(subkey, i)  # Fold in the index to guarantee uniqueness
            seed_value = int(jax.random.randint(subkey, (1,), 0, 2**32 - 1)[0])  # Get a scalar seed
            
            # Use the seed to generate the random state vector
            state_vec = random_statevector(2**n_qubits, seed=seed_value).data
            X.append(np.asarray(state_vec, dtype=jnp.complex128))
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

# def compute_initial_learning_rate(gradients, scale_factor=0.01, min_lr=1e-3, max_lr = 0.01):
#     """
#     Compute an initial learning rate based on the norm of gradients.
#     """
#     # Compute the norm of the gradients
    
#     norm_grad = jnp.linalg.norm(gradients)
    
#     initial_lr = scale_factor / (norm_grad + 1e-8)  # Adding a small value to prevent division by zero
#     print(norm_grad, initial_lr)
#     initial_lr = jnp.clip(initial_lr, min_lr, max_lr)
#     return initial_lr
# def get_initial_learning_rate(dqfim, scale_factor=0.1, min_lr=1e-3, max_lr = 0.1):
#     """
#     Compute an initial learning rate based on the norm of gradients.
#     """
    


#     # Eigenvalue decomposition of the DQFIM
#     dqfim_eigvals = jnp.linalg.eigvalsh(dqfim)
    
#     # Use the largest eigenvalue to scale the learning rate
#     lambda_max = jnp.max(dqfim_eigvals)
    
#     # Initial learning rate scaled by the largest eigenvalue of the DQFIM
#     initial_lr = scale_factor / lambda_max
    
#     # Ensure the learning rate is within specified bounds
#     initial_lr = jnp.clip(initial_lr, min_lr, max_lr)
    
#     print(f"lambda_max: {lambda_max}")
#     print(f"Initial learning rate: {initial_lr}")
    
#     return initial_lr
# def get_initial_learning_rate(dqfim, scale_factor=0.1, min_lr=1e-3, max_lr = 0.1):
#     """
#     Compute an initial learning rate based on the norm of gradients.
#     """
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


    # grad_norm = jnp.linalg.norm(density_matrix_grads)
    # initial_lr = scale_factor / (grad_norm + 1e-12)  # Adding small epsilon to avoid division by zero
    # print(f"Initial (grad_norm^-1) lr: {initial_lr}")
    # combined_scaling_factor = grad_norm * trace_dqfim

    # # Initial learning rate adjusted by both the gradient norm and QFIM trace
    # initial_lr = scale_factor / (combined_scaling_factor + 1e-12)  # Small epsilon to avoid division by zero
    # print(f"Initial (combined) lr: {initial_lr}")
    
    # # Use the largest eigenvalue to scale the learning rate
    # lambda_max = jnp.max(dqfim_eigvals)
    # print(f"Initial (grad_norm^-1) lr: {initial_lr}")
    # combined_scaling_factor = grad_norm * lambda_max

    # # Initial learning rate adjusted by both the gradient norm and QFIM trace
    # initial_lr = scale_factor / (combined_scaling_factor + 1e-12)  # Small epsilon to avoid division by zero
    # print(f"Initial (lambda_max*gradnorm)^-1) lr: {initial_lr}")
    
    
    # # Initial learning rate scaled by the largest eigenvalue of the DQFIM
    # initial_lr = scale_factor / lambda_max
    # print(f"Initial (preclippd) learning rate: {initial_lr}")
    # # Ensure the learning rate is within specified bounds
    initial_lr = jnp.clip(initial_lr, min_lr, max_lr)
    
    # print(f"lambda_max: {lambda_max}")
    # # print(f"Initial learning rate: {initial_lr}")
    
    return initial_lr, {"dqfim_eigvals": dqfim_eigvals,"dqfim_eigvecs": dqfim_eigvecs, "DQFIM": DQFIM,"entropies": entropies}
def optimize_traingset(gate,N_ctrl, N_r,trotter_steps, params, K_coeffs, N_train,num_datasets, key):
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
    
   
    
    sim_qr = QuantumReservoirGate(N_r, N_ctrl, K_coeffs,trotter_steps)
    @qml.qnode(sim_qr.dev, interface='jax')
    def circuit(params, input_state):
        x_coeff = params[0]
        z_coeff = params[1]
        y_coeff = params[2]
        J_coeffs = params[3:]
        qml.StatePrep(input_state, wires=[*sim_qr.ctrl_qubits])
        for i in range(trotter_steps):
            sim_qr.set_gate_reservoir()
            if sim_qr.static or trotter_steps == 1:
                sim_qr.set_gate_params(x_coeff, z_coeff, y_coeff, J_coeffs)
            else:
                step = len(sim_qr.rsv_qubits) * len(sim_qr.ctrl_qubits)
                sim_qr.set_gate_params(x_coeff, z_coeff, y_coeff, J_coeffs[i * step:(i + 1) * step])
        return qml.density_matrix(wires=[*sim_qr.ctrl_qubits])
    
    

   
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
    

    

    batched_collect_gradients = vmap(collect_gradients, in_axes=(None, 0, 0))

    all_gradients = batched_collect_gradients(params, all_A[:, :N_train], all_b[:, :N_train])
    
    #print("all_gradients shape:", all_gradients.shape)
    def calculate_gradient_stats(gradients):
        mean_grad = jnp.mean(gradients, axis=0)
        mean_grad_squared = jnp.mean(gradients**2, axis=0)
        var_grad = mean_grad_squared - mean_grad**2
        return mean_grad, var_grad

    best_dataset_idx = None
    max_var_grad_sum = -jnp.inf
    second_best_idx = None
    min_var_grad_sum = jnp.inf

    # Calculate and print gradient statistics for each dataset
    for i in range(num_datasets):
        mean_grad, var_grad = calculate_gradient_stats(all_gradients[i])
        aggregated_var = var_grad.mean()
        mean_grad_sum = mean_grad.sum()
        min_grad = min(var_grad)
        iqr_var_grad_25_75 = stats.iqr(var_grad, rng=(20,80))
        iqr_var_grad_30_70 = stats.iqr(var_grad, rng=(30,70))

        print(f"A{i+1}, b{i+1}): Variance of gradients = {aggregated_var:5e}; IQR of Variance (20,80) = {iqr_var_grad_25_75:5e};  IQR of Variance (30,70) = {iqr_var_grad_30_70:5e}\n")
        # print(f"(A{i+1}, b{i+1}):")

        # print(f"Variance Gradient sum: {var_grad_sum}, mean_grad_sum: {mean_grad_sum}, minimum grad: {min_grad}\n")
        if aggregated_var > max_var_grad_sum:
            if best_dataset_idx is not None:
                second_best_idx = best_dataset_idx
            max_var_grad_sum = aggregated_var
            best_dataset_idx = i
        elif best_dataset_idx is not None and (second_best_idx is None or aggregated_var > calculate_gradient_stats(all_gradients[second_best_idx])[1].mean()):
            second_best_idx = i
        if aggregated_var < min_var_grad_sum:
            min_var_grad_sum = aggregated_var

    print(f"Selected Dataset: A{best_dataset_idx + 1}, b{best_dataset_idx + 1} with Variance Sum: {max_var_grad_sum}")
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
def get_rate_of_improvement(cost, prev_cost,second_prev_cost):
    
    prev_improvement = prev_cost - second_prev_cost
    current_improvement = cost - prev_cost
    acceleration = prev_improvement - current_improvement

    return acceleration
def run_experiment(params, steps, n_rsv_qubits, n_ctrl_qubits,  K_coeffs, trotter_steps, static, gate, gate_name, test_size, training_size,opt_lr= None,L = [], key=0):
    bath = False
    dqfim_data = None
    
    
    if len(L) > 0:
        A, b = generate_dataset(gate, gate.num_wires, training_size, test_size,key=key, L=L)
    else:
        A, b = generate_dataset(gate, gate.num_wires, training_size, test_size,key=key)
    init_params = params
    X, y = A[:training_size], b[:training_size]
    test_X, test_y = A[training_size:], b[training_size:]
    if len(L) > 0:
        assert np.array_equal(X, L[:training_size]), f"Training set not set correctly. input_states[0]: {X[0]}, L[0]: {L[0]}"
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
            
            if trotter_steps==1:
                qrc.set_gate_params(x_coeff=x_coeff,z_coeff=z_coeff,y_coeff=y_coeff, J_coeffs=J_coeffs)
            else:
                step = len(qrc.rsv_qubits)*len(qrc.ctrl_qubits)
                qrc.set_gate_params(x_coeff=x_coeff,z_coeff=z_coeff,y_coeff=y_coeff,J_coeffs=J_coeffs[i*step:(i+1)*step])

        return qml.density_matrix(wires=[*qrc.ctrl_qubits])
        
    vcircuit = jax.vmap(circuit, in_axes=(None, 0))
    def batched_cost_helper(params, input_states, target_states):
        # Process the batch of states
        batched_output_states = vcircuit(params, input_states)
        
        # Compute fidelity for each pair in the batch and then average
        fidelities = jax.vmap(qml.math.fidelity)(batched_output_states, target_states)
        fidelities = jnp.clip(fidelities, 0.0, 1.0)
        average_fidelity = np.sum(fidelities)/len(fidelities)
        
        return 1 - average_fidelity  # Minimizing infidelity
        
    @partial(jit, static_argnums=(3, 4, 5, 6))
    def cost_func(params,X, y, n_rsv_qubits, n_ctrl_qubits, trotter_steps, static):
        
        loss = batched_cost_helper(params, X, y)
        # print(f"cost_func - loss dtype: {loss.dtype}")
        loss = jnp.maximum(loss, 0.0)  # Apply the cutoff to avoid negative costs
    
        
        return loss
    
   
    
    def final_costs(params, X, y, n_rsv_qubits=None, n_ctrl_qubits=None, trotter_steps=None, static=None):
        
        batched_output_states = vcircuit(params, X)
        
        fidelities = jax.vmap(qml.math.fidelity)(batched_output_states, y)
        fidelities = jnp.clip(fidelities, 0.0, 1.0)
        return fidelities
    # if opt_lr == None:
    #     # s = time.time()
    #     # init_loss, init_grads = jax.value_and_grad(cost_func)(params, X, y, n_rsv_qubits, n_ctrl_qubits, trotter_steps, static)
    #     # e = time.time()
    #     # dt = e - s
    #     # print(f"initial fidelity: {init_loss}, initial_gradients: {init_grads}. Time: {dt}")
    #     opt_lr = get_initial_learning_rate(dqfim)
    #     print(f"Adjusted initial learning rate: {opt_lr:5e}")
    if opt_lr == None:
        # s = time.time()
        init_loss, init_grads = jax.value_and_grad(cost_func)(params, X, y, n_rsv_qubits, n_ctrl_qubits, trotter_steps, static)
        # e = time.time()
        # dt = e - s
        # print(f"initial fidelity: {init_loss}, initial_gradients: {init_grads}. Time: {dt}")
        opt_lr, dqfim_data = get_initial_learning_rate_DQFIM(params=params,qrc=qrc, X = X, gate=gate,init_grads=init_grads)
        print(f"Adjusted initial learning rate: {opt_lr:.3e}")
    # if opt_lr == None:
    #     s = time.time()
    #     init_loss, init_grads = jax.value_and_grad(cost_func)(params, X, y, n_rsv_qubits, n_ctrl_qubits, trotter_steps, static)
    #     e = time.time()
    #     dt = e - s
    #     # print(f"initial fidelity: {init_loss}, initial_gradients: {init_grads}. Time: {dt}")
    #     opt_lr, grad_norm = get_initial_learning_rate(init_grads)
    #     print(f"Adjusted initial learning rate: {opt_lr}. Grad_norm: {1 / grad_norm}")

    print("________________________________________________________________________________")
    print(f"Starting optimization for {gate_name} with {len(X)} training states, lr {opt_lr} time_steps = {trotter_steps}, N_r = {n_rsv_qubits}...\n")

    opt = optax.adam(learning_rate=0.1)

    # opt = optax.chain(
    #     optax.clip_by_global_norm(1.0),  # Clip gradients to prevent explosions
    #     optax.adam(learning_rate=opt_lr, eps=1e-8)  # Slightly more aggressive Adam
    # )

    cost_threshold= 1e-05
    conv_tol = 1e-08
    prev_cost = float('inf')  # Initialize with infinity
    threshold_counts = 0
    consecutive_improvement_count = 0
    consecutive_threshold_limit = 4
    backup_params = None
    backup_cost = float('inf')  
    threshold_cond1, threshold_cond2 = [],[]
    cost_res = 1
    a_condition_set = False
    a_threshold =  0.0
    stored_epoch = None
    costs,grads_per_epoch,rocs = [],[],[]
    epoch = 0
    improvement = True
    opt_state = opt.init(params)
    @jax.jit
    def update(params, opt_state, X, y):
        loss, grads = jax.value_and_grad(cost_func)(params, X, y, n_rsv_qubits, n_ctrl_qubits,trotter_steps, static)
        updates, opt_state = opt.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state, loss,grads
    fullstr = time.time()
    while epoch  < steps or improvement:

        params, opt_state, cost,grad_circuit = update(params, opt_state, X, y)
        if epoch > 1:
            var_grad = np.var(grad_circuit,ddof=1)
            mean_grad = np.mean(jnp.abs(grad_circuit))
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

        costs.append(cost)
        grads_per_epoch.append(grad_circuit)

        if (epoch + 1) % 100 == 0 or epoch==0:
            #cost_res = cost(params, n_rsv_qubits=n_rsv_qubits, n_ctrl_qubits=n_ctrl_qubits, z_coeffs=z_coeffs, K_coeffs=K_coeffs, trotter_steps=trotter_steps, static=static, X=X, y=y)
            print(f'Cost after step {epoch + 1}: {cost}. Max gradient: {max(grad_circuit)}')
        
        
        
        # Check if there is improvement
        if cost < prev_cost:
            improvement = True
            consecutive_improvement_count += 1
            current_cost_check = cost_func(params, X, y, n_rsv_qubits, n_ctrl_qubits,trotter_steps, static)
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
            improvement = False  # Stop if no improvement
            consecutive_improvement_count = 0  # Reset the improvement count if no improvement
        second_prev_cost = prev_cost  # Move previous cost back one step
        prev_cost = cost  # Update previous cost with the current cost
        if prev_cost <= conv_tol: 
            break
        if np.abs(max(grad_circuit)) < 1e-14:
            break
        prev_cost = cost
        epoch += 1
    if backup_cost < cost:
        print(f"*backup cost (epoch: {backup_epoch}) is better with: {backup_cost:.2e} <  {cost:.2e}: {backup_cost < cost}")
        params = backup_params
    fullend = time.time()
    print(f"time optimizing: {fullend-fullstr}")
    testing_results = final_costs(params, X=test_X, y=test_y, n_rsv_qubits=n_rsv_qubits, n_ctrl_qubits=n_ctrl_qubits, trotter_steps=trotter_steps, static=static)
    avg_fidelity = jnp.mean(testing_results)
    infidelities = 1.00000000000000-testing_results
    avg_infidelity = np.mean(infidelities)

    print(f"\nAverage Final Fidelity: {avg_fidelity:.2e}")

    x_coeff = params[0]
    z_coeff = params[1]
    y_coeff = params[2]
    J_coeffs = params[3:]
    
    data = {'Gate':base64.b64encode(pickle.dumps(gate)).decode('utf-8'),
                'DQFIM_target_states':dqfim_data,
                'epochs': epoch,
                'trotter_step': trotter_steps,
                'backup_epoch': backup_epoch,
                'controls': n_ctrl_qubits, 
                'reservoirs': n_rsv_qubits,
                'x_coeff': x_coeff,
                'z_coeff': z_coeff,
                'y_coeff': y_coeff,
                'J_coeffs': J_coeffs,
                'K_coeffs': K_coeffs,
                'opt_lr':opt_lr,
                'training_states':X,
                'costs': costs,
                'backup_cost':backup_cost,
                'init_params':init_params,
                'grads_per_epoch':grads_per_epoch,
                'avg_infidelity':avg_infidelity,
                'avg_fidelity':avg_fidelity,
                'infidelities':infidelities,
                'testing_results':testing_results,
                'training_size': training_size,
                'noise_central': noise_central,
                'noise_range': noise_range,
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
    print(f"file_path: {file_path}")
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
                        entropies = results['entropies']
                       
                        return qfim_eigvals,fixed_params,params,qfim, results['L'],entropies
    
    print("QFIM eigenvalues not found for the given parameter keys.")
    return None,None,None, None,None

  
if __name__=='__main__':

# --------------------------------------------------------------------------------------------
    
    

    

    steps = 1000
    training_size_list = [10]
    noise_central = 0.1
    noise_range = 0.002
    test_size = 2000
# --------------------------------------------------------------------------------------------
    

    

    #folder = './results_gate_model_dec_26_full_001/' #save the results in this folder
    
    
    
    static_options = [False]
    bath_qubit = [False]
    training_size = 10
    parameters = []
    
    N_r = 1
    trotter_steps = 10
  
    bath = False
    static = False
    N_ctrl = 2

    folder = f'./param_initialization_final/digital_results/Nc_{N_ctrl}/'
    gates_random = []
    for i in range(50):
        U = random_unitary(2**N_ctrl, i).to_matrix()
        g = partial(qml.QubitUnitary, U=U)
        g.num_wires = N_ctrl
        g.name = f'U{N_ctrl}_'+str(i)
        gates_random.append(g)
    
    gates =   gates_random 

 
    fp_idx = 0
   
    fixed_param_keys = [f'fixed_params{fp_idx}',f'fixed_params{fp_idx}',f'fixed_params{fp_idx}',f'fixed_params{fp_idx}',f'fixed_params{fp_idx}',f'fixed_params{fp_idx}',f'fixed_params{fp_idx}',f'fixed_params{fp_idx}',f'fixed_params{fp_idx}',f'fixed_params{fp_idx}',f'fixed_params{fp_idx}',f'fixed_params{fp_idx}',f'fixed_params{fp_idx}']
    trainable_param_keys = ['test45','test33','test20','test43','test1','test13', 'test9', 'test19','test3', 'test39', 'test35']
    # trainable_param_keys = ['test20', 'test60', 'test181', 'test142', 'test138', 'test95']
    # trainable_param_keys = ['test110','test5', 'test103', 'test20', 'test22', 'test7', 'test52', 'test116', 'test13', 'test113']
    # trainable_param_keys =['test1', 'test141','test133','test130','test190','test82','test141', 'test165']
    # trainable_param_keys = ['test66','test40','test75', 'test86', 'test44'] # 0.1 range
    # trainable_param_keys = ['test28', 'test15'] # pi range
    # trainable_param_keys = ['test5', 'test107' ,'test127','test44']# pi normal range
    # trainable_param_keys = ['test137','test118', 'test10'] # pi*.3 normal range

    # list_of_trainable_sets = {"1": ['test18','test89','test172','test73','test35','test196','test139','test15', 'test118','test82','test146','test170' ],
    #                         #   "pi_normal": ['test5', 'test107' ], 
    #                         #   "0.1": ['test75', 'test86' , 'test44'],
    #                         #   "pi_normal.5": ['test154','test142']
    #                           }
    
    list_of_trainable_sets = {"pi": trainable_param_keys,
                             }
    # sample_range_label = "pi_normal.3"
    # sample_range = np.pi
    # print(len(fixed_param_keys), len(trainable_param_keys))
    all_gates = gates_random
    base_state = 'GHZ_state'
    opt_lr = None
    delta_x = 1.49011612e-08
    threshold = 1.e-14
    num_L = 100
    all_gates = gates_random
    for gate_idx,gate in enumerate(all_gates):
        for fixed_param_name, (sample_range_label, trainable_param_keys) in zip(fixed_param_keys,list_of_trainable_sets.items()):
            for test_key in trainable_param_keys:
                test_key_dict = f'{sample_range_label}/{test_key}'
                # folder_gate = folder + '/reservoirs_' + str(N_r) + '/trotter_step_' + str(trotter_steps) + f'/{base_state}'+  '/' + f"/{training_size}_training_states_random_DQFIM/" + str(fixed_param_name)+  '/'+ test_key+f'_{sample_range_label}/' + gate.name + '/'
                folder_gate = folder + '/reservoirs_' + str(N_r) + '/trotter_step_' + str(trotter_steps) + f'/{base_state}'+  '/' + f"/{training_size}_training_states_DQFIM/" + str(fixed_param_name)+  '/'+ test_key_dict+f'/' + gate.name + '/'
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
                # if i != 4:
                #     continue

                set_key = jax.random.PRNGKey(gate_idx*12345)
                
                
                print("________________________________________________________________________________")
                filename = os.path.join(folder_gate, f'{test_key}.pickle')
                # filename = os.path.join(folder_gate, f'{test_key}.pickle')
                
                N = N_ctrl + N_r
                state = 'GHZ'
                Kfactor = '1xK'
                qfim_base_path = f'/Users/so714f/Documents/offline/qrc/QFIM_traced_final_results/gate_model_DQFIM/Nc_{N_ctrl}/L_{num_L}/{Kfactor}/'
                # qfim_file_path = Path(qfim_base_path) / f'Nr_{N_r}' / f'trotter_step_{trotter_steps}' /  f'data2.pickle'

                qfim_file_path = Path(qfim_base_path) / f'Nr_{N_r}' / f'trotter_step_{trotter_steps}/L_{num_L}' /  f'data_{sample_range_label}_range.pickle'
                print(qfim_file_path)
                print(f"{test_key} params (range: (-{sample_range_label}, {sample_range_label}))")
                

                eigvals, K_coeffs,params,qfim,L,entropies = get_qfim_eigvals(qfim_file_path, fixed_param_name, test_key)
                L = []
                print(f"{test_key}")
                # print(f"eigvals: {eigvals}")
                # print(f"params: {params}")
                
            # A, b, opt_lr,first_grad = optimize_traingset(gate,N_ctrl, N_r,trotter_steps, params, K_coeffs, training_size,5,set_key)
                # A, b,worst_a,worst_b,opt_lr = optimize_traingset(gate,N_ctrl, N_r,trotter_steps, params, K_coeffs, training_size,20,set_key)
                #print(f"{test_key} params: {params}")
                #print(f"{fixed_param_name}: {K_coeffs}")
                
                #A, b,opt_lr,first_grad = optimize_traingset(gate,n_ctrl_qubits, n_rsv_qubits,trotter_steps, params, K_coeffs, training_size,5,params_subkey)

                trace_qfim = np.sum(eigvals)
                var_qfim = np.var(eigvals)
                print(f"QFIM trace: {trace_qfim}")
                print(f"QFIM var: {var_qfim}")
                based_subkey= gate_idx*trotter_steps*N_r,

                data = run_experiment(params = params,steps =  steps,n_rsv_qubits= N_r,n_ctrl_qubits= N_ctrl,K_coeffs= K_coeffs,trotter_steps= trotter_steps,static= static,gate=gate,gate_name= gate.name,test_size= test_size,training_size= training_size,opt_lr=opt_lr, L=L,key=set_key)
                data['QFIM Results'] = {"qfim_eigvals":eigvals,
                                        "trainable_params": params,
                                        "qfim": qfim,
                                        "entropies":entropies,
                'variance':var_qfim,
                'trace':trace_qfim,

                    }
                
                df = pd.DataFrame([data])
                while os.path.exists(filename):
                    name, ext = filename.rsplit('.', 1)
                    filename = f"{name}_.{ext}"

                with open(filename, 'wb') as f:
                    pickle.dump(df, f)
                print(f"Saved to path: {filename}")

                

                
