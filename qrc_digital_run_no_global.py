import pennylane as qml

from qiskit.circuit.library import *
from qiskit import *
from qiskit.quantum_info import *
import matplotlib.pyplot as plt
import numpy
from pennylane.wires import Wires
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
import pandas as pd
import pickle
import base64
from itertools import product
import optax
import numpy
import os
import jax
import time
from jax import config
from jax import numpy as np
import jax.numpy as jnp
from jax import jit, value_and_grad, vmap
import pennylane.numpy as pnp
#os.environ['OPENBLAS_NUM_THREADS'] = '1'
has_jax = True
diable_jit = False
config.update('jax_disable_jit', diable_jit)
#config.parse_flags_with_absl()
config.update("jax_enable_x64", True)
os.environ['JAX_TRACEBACK_FILTERING'] = 'off'


def oracle(wires, omega):
    """Apply the oracle that flips the phase of the omega state."""
    qml.FlipSign(omega, wires=wires)

def grover_diffusion(wires):
    """Apply the Grover diffusion operator."""
    qml.templates.GroverOperator(wires=wires)

def grover_iteration(wires, omega):
    """Perform one iteration of Grover's algorithm (oracle + diffusion)."""
    oracle(wires, omega)
    grover_diffusion(wires)

def grover_fun(input_state, wires, iterations, omega):
    """Apply Grover's algorithm to the input state."""
    qml.StatePrep(input_state, wires=wires)
    
    for _ in range(iterations):
        grover_iteration(wires, omega)
    
    return qml.density_matrix(wires=wires)

def generate_grover_dataset(omega,  n_qubits, training_size, key, iterations=1):
    """Generate dataset for Grover's algorithm with a specific marked state (omega) and output states as targets."""
    X = []
    y = []
    keys = jax.random.split(key, num=training_size)
    for _ in range(training_size):
        subkey = jax.random.fold_in(subkey, i) 
        # Generate a random initial state for each training sample
        seed_value = int(jax.random.randint(subkey, (1,), 0, 2**32 - 1)[0])  # Get a scalar seed
        
        initial_state = random_statevector(2**n_qubits, seed=seed_value).data

        # Define the qubit wires
        wires = range(n_qubits)
        dev = qml.device('default.qubit', wires=wires)
        
        # Construct the Grover's algorithm circuit using PennyLane
        grover_circuit = qml.QNode(grover_fun, device=dev, interface='jax')
        
        # Apply Grover's algorithm and get the output state
        output_state = grover_circuit(initial_state, wires, iterations, omega)
        
        # Save the initial state and corresponding full output state
        X.append(np.asarray(initial_state))
        y.append(np.asarray(output_state))  # Use the full output state as the target
    
    return np.stack(X), np.stack(y)

def generate_omegas(n_ctrl_qubits):
    """Generate all possible omega values based on the number of control qubits."""
    return [list(omega) for omega in product([0, 1], repeat=n_ctrl_qubits)]


def quantum_fun(gate, input_state, qubits):
    '''
    Apply the gate to the input state and return the output state.
    '''
    qml.StatePrep(input_state, wires=[*qubits])
    gate(wires=qubits)
    return qml.density_matrix(wires=[*qubits])

def generate_dataset(gate, n_ctrl_qubits,n_rsv_qubits, training_size, key):
    '''
    Generate the dataset of input and output states according to the gate provided.
    Uses a seed for reproducibility.
    '''
    
    X = []
    
    # Split the key into subkeys for the full training size
    keys = jax.random.split(key, num=training_size)
    
    # Loop through the subkeys and generate the dataset
    for i, subkey in enumerate(keys):
        subkey = jax.random.fold_in(subkey, i)  # Fold in the index to guarantee uniqueness
        seed_value = int(jax.random.randint(subkey, (1,), 0, 2**32 - 1)[0])  # Get a scalar seed
        
        # Use the seed to generate the random state vector
        state_vec = random_statevector(2**n_ctrl_qubits, seed=seed_value).data
        X.append(np.asarray(state_vec, dtype=jnp.complex128))
    
    
    X = np.stack(X)
    qubits = Wires(list(range(n_ctrl_qubits)))
    dev_data = qml.device('default.qubit', wires=qubits)
    circuit = qml.QNode(quantum_fun, device=dev_data, interface='jax')
    

    y = [np.array(circuit(gate, X[i], qubits), dtype=jnp.complex128) for i in range(training_size)]
    y = np.stack(y)
    return X, y

def generate_low_entangled_dataset(gate, n_qubits, training_size, key):
    '''
    Generate a dataset of input low-entangled states and output states according to the gate provided.
    Uses a seed for reproducibility.
    '''
    
    X = []  # To store the generated low-entangled states
    keys = jax.random.split(key, num=training_size)
    
    for i, subkey in enumerate(keys):
        subkey = jax.random.fold_in(subkey, i)
        
        # Use random state initialization for each training instance
        random_angles = jax.random.uniform(subkey, shape=(n_qubits,), minval=0, maxval=2 * np.pi)
        
        # Create a low-entangled state with some initial randomness
        qubits = Wires(list(range(n_qubits)))
        dev = qml.device('default.qubit', wires=qubits)
        
        @qml.qnode(dev, interface='jax')
        def low_entangled_circuit():
            # Initialize qubits in random superpositions
            for q in range(n_qubits):
                qml.RY(random_angles[q], wires=q)
            
            # Apply entangling gates (low level of entanglement)
            for q in range(n_qubits // 2):
                qml.CNOT(wires=[q, (q + 1) % n_qubits])
            return qml.state()

        # Generate the state vector
        state_vec = low_entangled_circuit()
        X.append(np.asarray(state_vec, dtype=jnp.complex128))
    
    X = np.stack(X)
    
    # Apply the provided gate to generate output states
    dev_data = qml.device('default.qubit', wires=qubits)
    circuit = qml.QNode(quantum_fun, device=dev_data, interface='jax')
    
    y = [np.array(circuit(gate, X[i], qubits), dtype=jnp.complex128) for i in range(training_size)]
    y = np.stack(y)
    
    return X, y


"""
n the context of VQCs, the Mean(Var Grad) represents how spread out the gradient values are across the parameter space. A non-vanishing variance of gradients is generally seen as positive, as it prevents the barren plateau problem, which is characterized by vanishing gradients. The paper on efficient estimation of trainability for VQCs argues that having a non-vanishing variance ensures fluctuati
"""


class QuantumReservoirGate:

    def __init__(self, n_rsv_qubits, n_ctrl_qubits, K_coeffs,trotter_steps=1, static=False, bath_params=False,num_bath = 0):
        self.static = static
        self.bath_params = bath_params
        self.num_bath = num_bath
        self.ctrl_qubits = Wires(list(range(n_ctrl_qubits))) # wires of the reservoir qubits (i.e. number of qubits in the reservoir)
        self.rsv_qubits = Wires(list(range(n_ctrl_qubits, n_rsv_qubits+n_ctrl_qubits))) # wires of the control qubits (i.e. number of qubits in the control)
        if self.bath_params != False:
            self.network_wires = Wires([*self.ctrl_qubits,*self.rsv_qubits])

            self.bath_wires = Wires(list(range(n_rsv_qubits+n_ctrl_qubits, n_rsv_qubits+n_ctrl_qubits+self.num_bath)))
            self.all_wires = Wires([*self.ctrl_qubits,*self.rsv_qubits,*self.bath_wires])
            self.dev = qml.device("default.qubit", wires =self.all_wires) 
        else:
            self.all_wires = Wires([*self.ctrl_qubits,*self.rsv_qubits])
            self.dev = qml.device("default.qubit", wires =self.all_wires) 

        self.trotter_steps = trotter_steps

        self.K_coeffs = K_coeffs  # parameter of the reservoir (XY_coupling)
        # print(K_coeffs)
        

        
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
    def circuit(self, params, input_state=None):

        @qml.qnode(self.dev,interface="jax",diff_method="backprop")
        def _circuit(self, params, input_state):
            x_coeff = params[0]
            z_coeff = params[1]
            y_coeff = params[2]
            J_coeffs = params[3:]
            qml.StatePrep(input_state, wires=[*self.ctrl_qubits])
            for i in range(trotter_steps):
                
                self.set_gate_reservoir()
                
                if self.static or trotter_steps==1:
                    self.set_gate_params(x_coeff,z_coeff,y_coeff, J_coeffs)
                else:
                    step = len(self.rsv_qubits)*len(self.ctrl_qubits)
                    self.set_gate_params(x_coeff,z_coeff,y_coeff,  J_coeffs[i*step:(i+1)*step])
                
            return qml.density_matrix(wires=[*self.ctrl_qubits])
        return _circuit(params, params=params, input_state=input_state)


import jax.numpy as jnp
@jit
def get_initial_learning_rate(grads, scale_factor=0.01, min_lr=1e-3, max_lr = 0.2):
    """Estimate a more practical initial learning rate based on the gradient norms."""
    grad_norm = jnp.linalg.norm(grads)
    initial_lr = jnp.where(grad_norm > 0, scale_factor / grad_norm, 0.1)
    clipped_lr = jnp.clip(initial_lr, min_lr, max_lr)
    return clipped_lr, initial_lr

def compute_initial_learning_rate(gradients, scale_factor=0.1, min_lr=1e-3, max_lr = 0.3):
    """
    Compute an initial learning rate based on the norm of gradients.
    """
    # Compute the norm of the gradients
    
    norm_grad = jnp.linalg.norm(gradients)
    
    initial_lr = scale_factor / (norm_grad + 1e-8)  # Adding a small value to prevent division by zero
    # print(norm_grad, initial_lr)
    initial_lr = jnp.clip(initial_lr, min_lr, max_lr)
    return initial_lr

def calculate_gradient_stats(gradients):
    mean_grad = jnp.mean(gradients, axis=0)
    mean_grad_squared = jnp.mean(gradients ** 2, axis=0)
    var_grad = mean_grad_squared - mean_grad ** 2
    grad_norm = jnp.linalg.norm(mean_grad)
    return mean_grad, var_grad, grad_norm
def get_rate_of_improvement(cost, prev_cost,second_prev_cost):
    
    prev_improvement = prev_cost - second_prev_cost
    current_improvement = cost - prev_cost
    acceleration = prev_improvement - current_improvement

    return acceleration
def optimize_traingset(gate, N_ctrl, N_r, trotter_steps, params, K_coeffs, N_train, num_datasets, key, data_type = "haar"):
    datasets = []
    print(f"Pre-processing a batch of {num_datasets} training sets for selection... ")
    all_A, all_b = [], []
    for i in range(num_datasets):
        key, subkey = jax.random.split(key)  # Split the key for each dataset generation
        if data_type == "le":
            A, b = generate_low_entangled_dataset(gate, N_ctrl, N_r, N_train + 2000, subkey)

        else:
            A, b = generate_dataset(gate, N_ctrl, N_r, N_train + 2000, subkey)  # Generate dataset with the subkey
        all_A.append(A)
        all_b.append(b)
    all_A = jnp.stack(all_A)
    all_b = jnp.stack(all_b)

    # Instantiate the quantum reservoir system
    sim_qr = QuantumReservoirGate(N_r, N_ctrl, K_coeffs, trotter_steps)
    
    # Define the quantum circuit using PennyLane
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
    
    # Define the cost function
    @jit
    def cost_func(params, input_state, target_state):
        output_state = circuit(params, input_state)
        fidelity = qml.math.fidelity(output_state, target_state)
        return 1 - fidelity  # Minimizing infidelity

    # Compute gradients
    @jit
    def collect_gradients(params, input_states, target_states):
        grad_fn = jax.grad(cost_func, argnums=0)
        gradients = jax.vmap(grad_fn, in_axes=(None, 0, 0))(params, input_states, target_states)
        return gradients

    batched_collect_gradients = vmap(collect_gradients, in_axes=(None, 0, 0))
    all_gradients = batched_collect_gradients(params, all_A[:, :N_train], all_b[:, :N_train])
    # Normalize gradients before sending to the statistics function
    def normalize_gradients(gradients):
        norm = jnp.linalg.norm(gradients, axis=-1, keepdims=True) + 1e-8  # Add small epsilon to avoid division by zero
        return gradients / norm
    
    # Corrected: Direct calculation of normalized gradient variance
    def calculate_correct_normalized_var_grad(gradients):
        mean_grad = jnp.mean(gradients, axis=0)
        var_grad = jnp.var(gradients, axis=0)
        
        # Calculate the normalized gradient variance
        normalized_var_grad = var_grad / (jnp.mean(gradients ** 2, axis=0) + 1e-8)  # Adding epsilon for stability
        return mean_grad, var_grad, normalized_var_grad


    var_grad_means,var_var_grads = [], []
    grad_norms = []
    mean_normalized_var_grads = []
    min_var_grad_norm = np.inf
    max_var_grad_norm = 0.0
    normalized_variance_gradients = []
    mean_gradients = []
    # First, collect all var_grad_means and grad_norms for normalization
    for i in range(num_datasets):

        
        normalized_gradients = normalize_gradients(all_gradients[i])
        _, normalized_variance_gradient, _ = calculate_gradient_stats(normalized_gradients)
        mean_grad, var_grad, grad_norm = calculate_gradient_stats(all_gradients[i])
        mean_gradients.append(mean_grad)

        # _,_, normalized_var_grad = calculate_correct_normalized_var_grad(all_gradients[i])
        var_grad_means.append(var_grad.mean())
        
        var_var_grads.append(jnp.var(var_grad))

        grad_norms.append(grad_norm)
        


        mean_normalized_var_grads.append(normalized_variance_gradient.mean())
        normalized_variance_gradients.append(normalized_variance_gradient)
        # print(f'set A{i}: normalized_var_grad: {normalized_var_grad.mean():.2e}, normalized_var_grad2: {normalized_var_grad2.mean():.2e}')
    
    min_norm_var_grad = min(mean_normalized_var_grads)
    max_norm_var_grad = max(mean_normalized_var_grads)
    # print(f"Variance of the normalized gradients: [min: {min_var_grad_norm}, max: {max_var_grad_norm}]")

    
    min_var_var,max_var_var = min(var_var_grads), max(var_var_grads)
    min_var_grad = min(var_grad_means)
    max_var_grad = max(var_grad_means)
    print(f"Variance of the gradients (not normalized): [min: {min_var_grad}), max: {max_var_grad}]")

    min_grad_norm = min(grad_norms)
    max_grad_norm = max(grad_norms)
    print(f"Gradient Norm: [min: {min_grad_norm}), max: {max_grad_norm}]")

    def normalize_metric(value, min_val, max_val, epsilon=1e-6, upper_bound=0.999):
        if max_val > min_val:
            # Adjust to avoid exactly 0.0 and 1.0, and ensure upper bound doesn't reach 1.0
            normalized_value = (value - min_val ) / (max_val - min_val + 2 )
            return min(normalized_value, upper_bound)
        else:
            return 0.5  # Neutral value if min and max are the same

    # Store the gradient stats for all datasets
    results = {}

    # Variables to track the best datasets for three purposes
    best_for_initial_training_idx = None
    best_for_replacement_idx = None
    best_for_fine_tuning_idx = None
    best_initial_score = jnp.inf
    best_replacement_score = jnp.inf  # We want lower variability here
    best_fine_tuning_score = jnp.inf  # Small, but more precise gradient shifts



    # Initialize tracking for scores and results
    results = {}


    alpha = 0.1
    beta =0.5
    w1 = 0.5  # Weight for normalized variance of the gradient
    w2 = 0.35  # Weight for normalized gradient norm
    w3 = 0.25  # Weight for normalized variance of the variance of the gradient

    # First, compute all the gradient statistics and scores for each dataset
    for i in range(num_datasets):
        normalized_variance_gradient = normalized_variance_gradients[i]
        # mean_grad, var_grad, grad_norm, _ = calculate_gradient_stats(all_gradients[i])
        mean_normalized_variance_gradient = mean_normalized_var_grads[i]

        mean_variance_of_gradient = var_grad_means[i] # Mean of the variance of the gradients (gradients no normalized)
        variance_of_variance_gradients = var_var_grads[i] # Variance of the variance of the gradients (gradients no normalized)
        mean_gradient = mean_gradients[i]
        grad_norm = grad_norms[i]
        # norm_var_grad_mean = normalized_var_grad.mean()  # Mean of normalized gradient variance across params
        normalized_gradient_variance_max = normalized_variance_gradient.max()    # Max of normalized gradient variance across params
        variance_of_variance_normalized_gradient = jnp.var(normalized_variance_gradient) # Varia
        mean_of_variance_normalized_gradient = jnp.abs(normalized_variance_gradient).mean()

        average_of_mean_gradients_abs = np.abs(mean_gradient).mean()
        variance_of_mean_gradients = jnp.var(mean_gradient)
        min_grad = min(np.abs(mean_gradient))

        normalized_mean_variance_of_gradient = normalize_metric(mean_variance_of_gradient, min_var_grad, max_var_grad)
        normalized_grad_norm_score = normalize_metric(grad_norm, min_grad_norm, max_grad_norm)
        # normalized_mean_variance_of_gradient_normalized = normalize_metric(mean_of_variance_normalized_gradient, min_norm_var_grad, max_norm_var_grad)
        normalized_var_var_grads = normalize_metric(variance_of_variance_gradients, min_var_var, max_var_var)

        # initial_score = normalized_mean_variance_of_gradient * np.exp(-alpha * (normalized_grad_norm_score - 1) ** 2)* normalized_var_var_grads
        initial_score = np.exp(
            -(
                w1 * (normalized_mean_variance_of_gradient - 1) ** 2 +
                w2 * (normalized_grad_norm_score - 1) ** 2 +
                w3 * (normalized_var_var_grads - 1) ** 2
            )
        )
        # initial_Score =  normalized_mean_variance_of_gradient * np.exp(-alpha * (normalized_grad_norm_score - 1) ** 2)
        replacement_score = (
            normalized_mean_variance_of_gradient * np.exp(-beta * (normalized_grad_norm_score - 1) ** 2) 
        )



        fine_tuning_score = 0.5 * normalized_mean_variance_of_gradient + 0.5 * mean_of_variance_normalized_gradient



        results[f"dataset_{i}"] = {
            "Mean(Var Grad)": mean_variance_of_gradient,
            "Var(Var Grad)": variance_of_variance_gradients,
            "Mean(Mean Grad)": average_of_mean_gradients_abs,
            "Var(Mean Grad)": variance_of_mean_gradients,
            "Min Gradient": min_grad,
            "Gradient Norm": grad_norm,
            "Mean(Var Grad) [normalized]": mean_of_variance_normalized_gradient,  # Mean of normalized gradient variance
            "Var(Var Grad) [normalized]": variance_of_variance_normalized_gradient,    # Variance of normalized gradient variance
            "Initial Score": initial_score,
            "Replacement Score": replacement_score,
            "Fine-Tuning Score": fine_tuning_score,
            "dataset": (all_A[i], all_b[i])  # Store dataset A and b
        }

        # Print the detailed summary statistics for each dataset
        # print(f"(A{i}, b{i}):")
        # print(f"    Var(Grad): {mean_variance_of_gradient:.2e}, Normalized Var(Grad): {normalized_mean_variance_of_gradient:.2e}")
        # print(f"    Gradient Norm: {grad_norm:.2e}, Normalized Grad Norm: {normalized_grad_norm_score:.2e}")
        # print(f"    Var(NormGrad): {mean_of_variance_normalized_gradient:.2e}, Normalized Var(NormGrad): {normalized_mean_variance_of_gradient_normalized:.2e}")
        # print(f"    Var(Var(Grad)): {variance_of_variance_gradients:.2e}, Normalized  Var(Var(Grad)): {normalized_var_var_grads:.2e}")
        # print(f"    Initial Score: {initial_score:.2e}, Replacement Score: {replacement_score:.2e}, Fine-Tuning Score: {fine_tuning_score:.2e}\n")

    

    # Now select the best datasets for initial training, replacement, and fine-tuning
    sorted_by_initial = sorted(results.items(), key=lambda x: x[1]["Initial Score"], reverse=True)
    sorted_by_replacement = sorted(results.items(), key=lambda x: x[1]["Replacement Score"], reverse=True)
    sorted_by_fine_tuning = sorted(results.items(), key=lambda x: x[1]["Fine-Tuning Score"])

    # Best for initial training
    # Use the numeric index directly when accessing arrays
    best_for_initial_training = sorted_by_initial[0]
    best_for_replacement = sorted_by_replacement[1] if sorted_by_replacement[0][0] == best_for_initial_training[0] else sorted_by_replacement[0]
    best_for_fine_tuning = sorted_by_fine_tuning[0]

    # Extract the indices and scores for printing
    best_for_initial_training_idx =  int(best_for_initial_training[0].split('_')[1])
    best_initial_score = best_for_initial_training[1]["Initial Score"]

    best_for_replacement_idx = int(best_for_replacement[0].split('_')[1])
    best_replacement_score = best_for_replacement[1]["Replacement Score"]

    best_for_fine_tuning_idx =  int(best_for_fine_tuning[0].split('_')[1])
    best_fine_tuning_score = best_for_fine_tuning[1]["Fine-Tuning Score"]

    best_gradients = all_gradients[best_for_initial_training_idx]
    initial_grad_norm = results[f"dataset_{best_for_initial_training_idx}"]["Gradient Norm"]
    initial_lr = compute_initial_learning_rate(best_gradients)

    print(f"Best Dataset for Initial Training: A{best_for_initial_training_idx}, with score: {best_initial_score:.4e}")
    print(f"Best Dataset for Replacement: A{best_for_replacement_idx}, with score: {best_replacement_score:.4e}")
    print(f"Best Dataset for Fine-Tuning: A{best_for_fine_tuning_idx}, with score: {best_fine_tuning_score:.4e}")
    print(f"Initial Gradient Norm: {initial_grad_norm:2e}")
    # Extract the actual data
    best_initial_A = best_for_initial_training[1]["dataset"][0]
    best_initial_b = best_for_initial_training[1]["dataset"][1]
    best_replacement_A = best_for_replacement[1]["dataset"][0]
    best_replacement_b = best_for_replacement[1]["dataset"][1]
    best_fine_tuning_A = best_for_fine_tuning[1]["dataset"][0]
    best_fine_tuning_b = best_for_fine_tuning[1]["dataset"][1]
    return results, best_initial_A, best_initial_b, best_replacement_A, best_replacement_b, best_fine_tuning_A, best_fine_tuning_b, initial_lr, (best_for_initial_training_idx,best_for_replacement_idx,best_for_fine_tuning_idx)

def calculate_gradient_stats(gradients):
    mean_grad = jnp.mean(gradients, axis=0)
    mean_grad_squared = jnp.mean(gradients ** 2, axis=0)
    var_grad = mean_grad_squared - mean_grad ** 2
    grad_norm = jnp.linalg.norm(mean_grad)
    return mean_grad, var_grad, grad_norm
def run_experiment(params, bath_params, steps, n_rsv_qubits, n_ctrl_qubits, K_coeffs, trotter_steps, static, gate, gate_name, folder, test_size, training_size, noise_central, opt_lr,random_key):
    num_states_to_replace = training_size // 5
    selected_indices, preopt_results = {},{}
    bath = False
    init_params = params
    # folder_gate = folder + gate_name + '/reservoirs_' + str(n_rsv_qubits) + '/trotter_step_' + str(trotter_steps) +  '/' + '/bath_' + str(bath) + '/'+ "testing_preopt" + '/'
    folder_gate = folder + gate_name + '/reservoirs_' + str(n_rsv_qubits) + '/trotter_step_' + str(trotter_steps) +  '/' + '/bath_' + str(bath) + '/'
    Path(folder_gate).mkdir(parents=True, exist_ok=True)
    temp_list = list(Path(folder_gate).glob('*'))
    files_in_folder = []
    for f in temp_list:
        temp_f = f.name.split('/')[-1]
        
        if not temp_f.startswith('.'):
            files_in_folder.append(temp_f)
    k = 1
    if len(files_in_folder) >= k:
        print('Already Done. Skipping: '+folder_gate)
        print('\n')
        return
    A, b = generate_dataset(gate, N_ctrl, n_rsv_qubits, training_size + test_size, random_key)  # Generate dataset with the subkey
    # A, b = generate_grover_dataset(omega, n_ctrl_qubits, training_size+test_size,random_key)
    # A, b = generate_low_entangled_dataset(gate, n_ctrl_qubits, training_size+test_size,random_key)
    # preopt_results, A, b,second_A,second_b,finetuning_A,finetuning_b,opt_lr, selected_indices = optimize_traingset(gate,n_ctrl_qubits, n_rsv_qubits,trotter_steps, params, K_coeffs, training_size,10,random_key)
    filename = os.path.join(folder_gate, f'data_run_{len(files_in_folder)}.pickle')
    # best_dataset_idx = selected_indices[0]
    # filename = os.path.join(folder_gate, f'trainingset_A{best_dataset_idx}_rep_A{selected_indices[1]}.pickle')
    #print(f"params: {params}")
    # omega = gate
    # init_meangrad = preopt_results[f'dataset_{best_dataset_idx}']['Mean(Mean Grad)']
    # init_vargrad = preopt_results[f'dataset_{best_dataset_idx}']['Mean(Var Grad)']
    # cond1 = init_meangrad*1e-1
    # cond2 = init_vargrad*1e-2
    # print(f"init_meangrad: {init_meangrad:2e}, threshold: {cond1:2e}")
    # print(f"init_vargrad: {init_vargrad:2e}, threshold: {cond2:2e}")
    # add_a, add_b = np.asarray(second_A[:num_states_to_replace]), np.asarray(second_b[:num_states_to_replace])
    X, y = A[:training_size], b[:training_size]
    _, subkey = jax.random.split(random_key)
    test_X, test_y = generate_dataset(gate, N_ctrl, n_rsv_qubits, test_size, subkey) 

    qrc = QuantumReservoirGate(n_rsv_qubits=n_rsv_qubits, n_ctrl_qubits=n_ctrl_qubits,  K_coeffs=K_coeffs, static=static)
    ctrl_wires = qrc.ctrl_qubits
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


    
    # @qml.qnode(qrc.dev, interface="jax", diff_method="backprop")
    # def circuit(params, input_state, target_state):
        
    #     x_coeff = params[0]
    #     z_coeff = params[1]
    #     y_coeff = params[2]
    #     J_coeffs = params[3:]
        
    #     # Prepare the input state
    #     qml.StatePrep(input_state, wires=[*qrc.ctrl_qubits])
        
    #     # Apply gates in the trotterization
    #     for i in range(trotter_steps):
    #         qrc.set_gate_reservoir()
    #         if qrc.static or trotter_steps == 1:
    #             qrc.set_gate_params(x_coeff, z_coeff, y_coeff, J_coeffs)
    #         else:
    #             step = len(qrc.rsv_qubits) * len(qrc.ctrl_qubits)
    #             qrc.set_gate_params(x_coeff, z_coeff, y_coeff, J_coeffs[i*step:(i+1)*step])
        
    #     # Return the expectation value of the Hermitian operator corresponding to the target state
    #     return qml.expval(qml.Hermitian(target_state, wires=[*qrc.ctrl_qubits]))

    # vcircuit = jax.vmap(circuit, in_axes=(None, 0, 0))
    # @partial(jit, static_argnums=(3, 4, 5, 6))
    # def cost_func(params,X, y, n_rsv_qubits, n_ctrl_qubits, trotter_steps, static):
    #     params = jnp.asarray(params, dtype=jnp.float64)
    #     X = jnp.asarray(X, dtype=jnp.complex128)
    #     y = jnp.asarray(y, dtype=jnp.complex128)
    #     loss = jnp.mean(vcircuit(params, X, y))
    #     # print(f"cost_func - loss dtype: {loss.dtype}")
    #     return loss
    # @jit
    # def collect_gradients(params, input_states, target_states):
    #     def cost_func_grad(params, input_state, target_state):
    #         # Use the vcircuit to calculate the expectation values
    #         output_expval = vcircuit(params, input_state, target_state)
    #         return 1 - output_expval  # Minimizing infidelity


    #     grad_fn = jax.grad(cost_func_grad, argnums=0)
    #     gradients = jax.vmap(grad_fn, in_axes=(None, 0, 0))(params, input_states, target_states)
    #     return gradients
    # def final_costs(params, X, y, n_rsv_qubits=None, n_ctrl_qubits=None, trotter_steps=None, static=None):
        batched_expvals = vcircuit(params, X, y)
        
        print(f"final_costs - batched_expvals dtype: {batched_expvals.dtype}")
        return batched_expvals
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
    
    @jit
    def collect_gradients(params, input_states, target_states):
        def cost_func_grad(params, input_state, target_state):
            output_state = circuit(params, input_state)
            fidelity = qml.math.fidelity(output_state, target_state)
            return 1 - fidelity  # Minimizing infidelity

        grad_fn = jax.grad(cost_func_grad, argnums=0)
        gradients = jax.vmap(grad_fn, in_axes=(None, 0, 0))(params, input_states, target_states)
        return gradients


    def final_costs(params, X, y, n_rsv_qubits=None, n_ctrl_qubits=None, trotter_steps=None, static=None):
        
        batched_output_states = vcircuit(params, X)
        print(f"y dtype: {y[0].dtype}")
        
        fidelities = jax.vmap(qml.math.fidelity)(batched_output_states, y)
        fidelities = jnp.clip(fidelities, 0.0, 1.0)
        # print(f"final_costs - fidelities dtype: {fidelities.dtype}")
        return fidelities
    
    

    # opt_lr = 0.01
    if opt_lr == None:
        s = time.time()
        init_loss, init_grads = jax.value_and_grad(cost_func)(params, X, y, n_rsv_qubits, n_ctrl_qubits, trotter_steps, static)
        e = time.time()
        dt = e - s
        # print(f"Initial gradients dtype: {init_grads.dtype}, Initial loss dtype: {init_loss.dtype}")
        print(f"initial fidelity: {init_loss}")
        # print(f"initial fidelity: {init_loss}, initial_gradients: {init_grads}. Time: {dt}")
        opt_lr, raw_lr = get_initial_learning_rate(init_grads)
        print(f"Adjusted initial learning rate: {opt_lr:.4f}.prev: {raw_lr:.5f}")

    print("________________________________________________________________________________")
    print(f"Starting optimization for {gate_name}  trots = {trotter_steps}, N_r = {n_rsv_qubits} with optimal lr {opt_lr:.4f} time_steps = {trotter_steps}, N_r = {n_rsv_qubits}...\n")

    #opt = optax.adam(learning_rate=0.1)
    # opt = optax.chain(
    #     optax.clip_by_global_norm(1.0),  # Clip gradients to prevent explosions
    #     optax.adam(learning_rate=opt_lr, b1=0.9, b2=0.999, eps=1e-8)  # Slightly more aggressive Adam
    # )
    opt = optax.adam(learning_rate=opt_lr)

    cost_threshold= 1e-05
    conv_tol = 1e-08
    prev_cost = float('inf')  # Initialize with infinity
    threshold_counts = 0
    consecutive_threshold_limit = 4
    backup_params = None
    backup_cost = float('inf')  

    cost_res = 1
    costs = []
    grads_per_epoch = []
    epoch = 0
    improvement = True
    opt_state = opt.init(params)
    @jit
    def update(params, opt_state, X, y):
        # Ensure inputs are float64
        params = jnp.asarray(params, dtype=jnp.float64)
        X = jnp.asarray(X, dtype=jnp.complex128)
        y = jnp.asarray(y, dtype=jnp.complex128)
        
        loss, grads = jax.value_and_grad(cost_func)(params, X, y, n_rsv_qubits, n_ctrl_qubits, trotter_steps, static)
        updates, opt_state = opt.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        # Ensure outputs are float64
        loss = jnp.asarray(loss, dtype=jnp.float64)
        grads = jnp.asarray(grads, dtype=jnp.float64)
        
    
        return new_params, opt_state, loss, grads
    fullstr = time.time()
    add_more = True
    improvement_count = 0
    a_threshold =  0.0
    stored_epoch = None
    false_improvement = False
    # print(f"params: {type(params)}, {params.dtype}")
    while epoch  < steps or improvement:
        #cost, grad_circuit  = jax.value_and_grad(cost_func)(params, X, y, n_rsv_qubits, n_ctrl_qubits,trotter_steps, static)
        #updates, opt_state = opt.update(grad_circuit, opt_state)
        #params = optax.apply_updates(params, updates)
        params, opt_state, cost,grad_circuit = update(params, opt_state, X, y)

        costs.append(cost)
        grads_per_epoch.append(grad_circuit)

        if (epoch + 1) % 100 == 0 or epoch==0:

            var_grad = np.var(grad_circuit)
            if epoch == 0:
                initial_vargrad = var_grad
                cond2 = initial_vargrad * 1e-2
                print(f"    - setting cond2: initial var(grad) {initial_vargrad:2e}, cond2 threshold: {cond2:2e}")
            mean_grad = np.mean(grad_circuit)  # No absolute value for normalized gradient variance
            
            # Normalized gradient variance
            normalized_var_grad = var_grad /  np.mean(grad_circuit**2) 
            # Print with only two decimal places in scientific notation
            print(f'Epoch {epoch + 1} cost: {cost:.5f}, '
                 
                f'Var(grad): {var_grad:.1e}, '
                f'Mean(grad): {mean_grad:.1e}')
            # print(f'Cost after step {epoch + 1}: {cost:.2e}. '
            #     f'Var(grad): {var_grad:.2e}, '
            #     f'Mean(grad): {mean_grad:.2e}, '
            #     f'Normalized Var(grad): {normalized_var_grad:.2e}')
            


        
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


        # if (epoch >= 50 and np.mean(np.abs(grad_circuit)) < cond1 
        #     and np.var(grad_circuit) < cond2 and add_more and epoch <= 0.9 * steps):

        #     mean_grad = jnp.mean(np.abs(grad_circuit))
        #     var_grad = jnp.var(grad_circuit)
        #     # Normalized gradient variance
        #     normalized_var_grad_abs = var_grad / (mean_grad ** 2) if mean_grad != 0 else float('inf')
        #     normalized_var_grad = var_grad / (jnp.mean(grad_circuit) ** 2) if jnp.mean(grad_circuit) != 0 else float('inf')

        #     # print(f"params: {type(params)}, {params.dtype}")
        #     # print(f"params: {params}")
        #     gradients_per_state = collect_gradients(params, X, y)
        #     meangrad, vargrad, grad_norm = calculate_gradient_stats(gradients_per_state)
        #     normalized_var_grad_L = vargrad / (meangrad ** 2)
        #     # Compute summary statistics for the normalized variance
        #     norm_var_grad_mean = normalized_var_grad_L.mean()  # Mean of normalized gradient variance across params
        #     norm_var_grad_max = normalized_var_grad_L.max()    # Max of normalized gradient variance across params
        #     norm_var_grad_var = jnp.var(normalized_var_grad_L) # Variance of normalized gradient variance across params
            
        #     grads_variance = jnp.var(gradients_per_state, axis=tuple(range(1, gradients_per_state.ndim)))  # Calculate variance across the parameters for each state
        #     # Compute the statistics and print them nicely
        #     var_grad_mean = vargrad.mean()
        #     var_var_grad = jnp.var(vargrad)
        #     min_grad = min(np.abs(grad_circuit))
        #     max_grad = max(np.abs(grad_circuit))

        #     print(f"Epoch {epoch}:")
        #     print(f"***flat landscape warning at epoch {epoch}. Mean(grad): {mean_grad:3e}, Var(Grad): {var_grad:3e}, Norm(vargrad): {normalized_var_grad:3e}, abs(mean): {normalized_var_grad_abs:3e}")
        #     print(f"    ∑_l Mean(Var Grad): {var_grad_mean:.2e})")
        #     print(f"    ∑_l Var(Var Grad): {var_var_grad:.2e}")
        #     print(f"    ∑_l Mean(Mean Grad): {jnp.mean(jnp.abs(meangrad)):.2e}")
        #     print(f"    ∑_l Var(Mean Grad): {jnp.var(jnp.abs(meangrad)):.2e}")
        #     print(f"    ∑_l Norm(Var Grad) (mean): {norm_var_grad_mean:.2e}")
        #     print(f"    ∑_l Norm(Var Grad) (normalized_var_grad): {normalized_var_grad:.2e}")
        #     print(f"    ∑_l Norm(Var Grad) (var): {norm_var_grad_var:.2e}")
        #     print(f"    Max Grad: {max_grad:.2e}")
        #     print(f"    Min Grad: {min_grad:.2e}\n")
        #     # Identify the indices of the states with the smallest gradient variances
        #     min_var_indices = np.argsort(grads_variance)[:num_states_to_replace]
        #     # Replace the states with the smallest variances with the new states
        #     for idx in min_var_indices:
        #         X = X.at[idx].set(add_a[min_var_indices == idx].squeeze(axis=0))
        #         y = y.at[idx].set(add_b[min_var_indices == idx].squeeze(axis=0))
            
        #     add_more = False
        
        # if np.abs(max(grad_circuit)) < 1e-14:
        #     print(f"")
        #     break
        if cost < 1e-6:
            print(f"Cost below threshold [{cost:.1e}]. Ending opt at epoch: {epoch}")
            break
        if np.var(grad_circuit) < 1e-10 and epoch >=1000:
            print(f"Variance of the gradients below threshold [{np.var(grad_circuit):.1e}]. Ending opt at epoch: {epoch}")
            break
        prev_cost = cost


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

    #test_X, test_y = generate_dataset(gate, n_ctrl_qubits,n_rsv_qubits, test_size,random_key)
    
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
        'epochs': epoch,
        'selected_indices': selected_indices,
        'init_params': init_params,
        'preopt_results': preopt_results,
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
        'bath_params': bath_params,
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
    df = pd.DataFrame([data])
    while os.path.exists(filename):
        name, ext = filename.rsplit('.', 1)
        filename = f"{name}_.{ext}"

    # print(f"Before pickling: {fidelities[0]}, type: {type(fidelities[0])}, dtype: {fidelities[0].dtype}")

    with open(filename, 'wb') as f:
        pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(filename)
    # with open(filename, 'rb') as f:
    #     loaded_df = pickle.load(f)

    # # # After pickling
    # value_after = loaded_df['testing_results'][0][0]
    # print(f"After pickling: {value_after}, type: {type(value_after)}")

    # # Check precision manually
    # if isinstance(value_after, float):
    #     print(f"After pickling dtype: {np.array(value_after).dtype}")
    # elif hasattr(value_after, "dtype"):  # In case it remains an array-like
    #     print(f"After pickling dtype: {value_after.dtype}")
    # else:
    #     print("Unable to determine dtype after pickling.")
    # with open(filename, 'wb') as f:
    #     pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)

    # print(filename)
    #df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
    #df.to_csv(folder_gate +'data.dataframe', index=False)



if __name__=='__main__':

# --------------------------------------------------------------------------------------------
    
    

    

    steps = 1000
    training_size_list = [20]
    noise_central = 0.01
    noise_range = 0.002
    test_size = 2000
    bath = False
# --------------------------------------------------------------------------------------------

    #folder = './gate_model_results_new/' #save the results in this folder
    
    
    
    # trotter_step_list = [1,2,3,4,5,6,7]
    # trotter_step_list = [20,21,23,23,24,25,26,27,28,29,30,31,32,33,34]
    trotter_step_list = [1, 8, 10, 12, 14, 16,18,20,22]
    trotter_step_list = trotter_step_list[::-1]
    # trotter_step_list = [22,24,26]
    # trotter_step_list = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45]
    # trotter_step_list = [1,3,6,9,10,11,12,13,14,15,16,17,18,19,20,22,24,28,29]
    # trotter_step_list = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56]
    
    # rsv_qubits_list = [2,3,4,5,6]
    rsv_qubits_list = [1,2,3]
    # trotter_step_list = [1]
    static_options = [False]
    N_ctrl = 2
    # omegas = generate_omegas(N_ctrl)
    # folder = f'./digital_results_trainable_global/trainsize_{training_size_list[0]}_same_epoch{steps}/'
    # folder = f'./digital_results_trainable_global/trainsize_{training_size_list[0]}_optimized_by_cost3/'
    folder = f'./digital_results_trainable_global/trainsize_{training_size_list[0]}_no_hypertraining_costcut_1e-6/'
    gates_random = []
    for i in range(20):
        U = random_unitary(2**N_ctrl, i).to_matrix()
        g = partial(qml.QubitUnitary, U=U)
        g.num_wires = N_ctrl
        g.name = f'U{N_ctrl}_'+str(i)
        gates_random.append(g)
    

    # gates = omegas
    gates = gates_random
    random = True
    parameters = []
    opt_lr = None
    
    for training_size in training_size_list:
        for static in static_options:
            for gate_idx,gate in enumerate(gates):
                # if gate_idx not in [12]:
                #     continue
                for trotter_steps in trotter_step_list:
                    for n_rsv_qubits in rsv_qubits_list:
                    
                        

                            
                        N = N_ctrl +n_rsv_qubits
                        
                        params_key_seed = gate_idx*121 * n_rsv_qubits + 12345 * trotter_steps *n_rsv_qubits
                        params_key = jax.random.PRNGKey(params_key_seed)

                        params = jax.random.uniform(params_key, shape=(3 + (N_ctrl * n_rsv_qubits) * trotter_steps,), minval=-np.pi, maxval=np.pi, dtype=jnp.float64)
                       
                        _, params_subkey1, params_subkey2 = jax.random.split(params_key, 3)
                        
                        
                        
                        n_ctrl_qubits = N_ctrl
                        
                        K_0 = 1.0
                        # print(gate)
                        K_half = jax.random.uniform(params_subkey1, (N, N))
                        K = (K_half + K_half.T) / 2  # making the matrix symmetric
                        K = 2. * K - 1.  # Uniform in [-1, 1]
                        K_coeffs = K * K_0 / 2  # Scale to [-K_0/2, K_0/2]
                        

                        bath_params = None
                        

                        if random:
                            label = gate.name
                        else:
                            label = gate
                        run_experiment(params, bath_params, steps, n_rsv_qubits, n_ctrl_qubits, K_coeffs, trotter_steps, static, gate, f'{label}', folder, test_size, training_size, noise_central, opt_lr,random_key=params_subkey2)