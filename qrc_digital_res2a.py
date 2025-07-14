import warnings

# Suppress PennyLane‐vs‐JAX compatibility warnings
warnings.filterwarnings(
    "ignore",
    # message=".*PennyLane is (not yet compatible|currently not compatible) with JAX versions > 0.4.28.*",
    category=RuntimeWarning,
)
import pennylane as qml

from qiskit.circuit.library import *
from qiskit import *
from qiskit.quantum_info import *
from numpy import float32
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
import numpy as np
import jax.numpy as jnp
from jax import jit, value_and_grad, vmap
import pennylane.numpy as pnp
from optax import tree_utils as otu
from optax import contrib
from optax.contrib import reduce_on_plateau

from optax._src import base as optax_base
from reduce_on_plateau import reduce_on_plateau
from typing import NamedTuple

#os.environ['OPENBLAS_NUM_THREADS'] = '1'
has_jax = True
diable_jit = False
config.update('jax_disable_jit', diable_jit)
#config.parse_flags_with_absl()
config.update("jax_enable_x64", True)
os.environ['JAX_TRACEBACK_FILTERING'] = 'off'
# Show on which platform JAX is running.
print("JAX running on", jax.devices()[0].platform.upper())


def _slice_masks(T: int, N_ctrl: int, N_reserv: int):
    """
    Generate boolean-mask helper lambdas for **slice-wise grouping**.

    Returns
    -------
    elem_mask : Callable[[int, int], jax.Array]
        ``elem_mask(t, M)`` → length-*M* boolean mask selecting the
        :math:`τ_t` scalar **plus** all couplings
        :math:`\{J_{kℓ}^{(t)}\}` belonging to Trotter slice *t*.
    h_mask : Callable[[int], jax.Array]
        ``h_mask(M)`` → mask of the three global field parameters
        :math:`(h^x,h^y,h^z)`.
    """
    num_J = N_ctrl * N_reserv
   

    def elem_mask(t, M):
        idx      = jnp.arange(M)
        tau_m    = idx == t
        start_J  = T + 3 + t * num_J
        end_J    = start_J + num_J
        J_m      = (idx >= start_J) & (idx < end_J)
        return tau_m | J_m          # τₜ  ∪  J^{(t)}

    def h_mask(M):
        idx = jnp.arange(M)
        return (idx >= T) & (idx < T + 3)

    return elem_mask, h_mask

def lr_group_dict(case_num: int,
                  lr_tree: jnp.ndarray,
                  T: int,
                  N_ctrl: int,
                  N_reserv: int):
    """
    Re-shape a *flat* learning-rate vector into a dict keyed by parameter block.

    
    Returns
    -------
    dict[str, jax.Array]
        Each entry is a *view* (no copy) into ``lr_tree`` covering exactly the
        indices of that block.
    """
    M                 = lr_tree.shape[0]
    if case_num in [1]:                  
        mask_h = np.arange(M) < 3
    
        mask_J   = np.arange(M) >=  3
        return {
            "h": lr_tree[mask_h],
            "J": lr_tree[mask_J],
        }

    # slice-wise grouping
    
    elem_mask, h_mask = _slice_masks(T, N_ctrl, N_reserv)
    out               = {"h": lr_tree[h_mask(M)]}
    for t in range(T):
        out[f"slice_{t}"] = lr_tree[elem_mask(t, M)]
    return out

def grad_group_dict(case_num: int,
                    grads_flat: jnp.ndarray,
                    T:int,
                    N_ctrl: int,
                    N_reserv: int):
    """
    Same interface as :pyfunc:`lr_group_dict` but applied to **gradients**.

    Use this inside the training loop *before* calling the optimiser so you
    capture **pre-update** gradients.

    Returns
    -------
    dict[str, jax.Array]
        Block-wise gradient subvectors, suitable for computing CV\_B or other
        diagnostics.
    """
    M                 = grads_flat.size
    if case_num <= 5:
        mask_h = np.arange(M) < 3       # concrete boolean mask
        mask_J = np.arange(M) >= 3
        return {
            "h": grads_flat[mask_h],
            "J": grads_flat[mask_J],
        }

    # slice-wise
    
    elem_mask, h_mask = _slice_masks(T, N_ctrl, N_reserv)
    out               = {"h": grads_flat[h_mask(M)]}
    for t in range(T):
        out[f"slice_{t}"] = grads_flat[elem_mask(t, M)]
    return out
def coef_variation_by_block(case_num, grads_flat,T, N_ctrl, N_reserv):
    """Return {block: CV} with block ∈ {"t","h","J"} or {"h","slice_0",…}."""
    blocks = grad_group_dict(case_num, grads_flat, T,N_ctrl, N_reserv)
    out = {}
    for name, vec in blocks.items():
        vec = jnp.asarray(vec)
        mu  = jnp.mean(jnp.abs(vec))
        sig = jnp.std(jnp.abs(vec))
        out[name] = float(sig / (mu + 1e-12))
    return out

def array(a,dtype):
    return np.asarray(a, dtype=np.float32)
def Array(a,dtype):
    return np.asarray(a, dtype=np.float32)

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


"""
n the context of VQCs, the Mean(Var Grad) represents how spread out the gradient values are across the parameter space. A non-vanishing variance of gradients is generally seen as positive, as it prevents the barren plateau problem, which is characterized by vanishing gradients. The paper on efficient estimation of trainability for VQCs argues that having a non-vanishing variance ensures fluctuati
"""


class QuantumReservoirNetwork:

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
    

  


import jax.numpy as jnp


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

# ---------------------------------------------------------------------------
#  Digital‑model learning‑rate helper  –  *no τ block* (case 1)
# ---------------------------------------------------------------------------
def get_groupwise_lr_trees(
    params: jnp.ndarray,          # flat parameter vector (shape: [3 + |J|])
    grads:  jnp.ndarray,          # flat gradient vector (same shape)
    num_train: int,               # kept for API parity – not used here
    N_C: int,                      # number of control qubits
    N_R: int,                      # number of reservoir qubits
    T: int,              # must be 0 → digital model has no τ
    *,
    max_lr: float = 0.2,
    debug: bool = False,
    scale_by_num_train: bool = False,
    target_update: float = 0.05,
    eps: float = 1e-12,
    factor: float = 1.0,        # kept for backward-compat
    per_block_target_update: bool = False,
):
    """
    Initialise a **per‑parameter learning‑rate vector** for the *digital* QRC,
    using the same case-1 trust‑ratio heuristic that the analog code applies
    to its  h / J blocks – **but here only the h and J blocks exist**.

    Blocks
    ------
    • **h**  – indices [0, 3)   : global RX / RZ / RY angles  
    • **J**  – indices [3, M)   : control‑→reservoir couplings  
      (`M = params.size = 3 + NC × NR × trotter_steps`)

    Returns
    -------
    (lr_tree, mask_tau, mask_h, mask_J)
        mask_tau is an all‑False array (kept for call‑site compatibility).
    """
    
    M   = params.size
    idx = jnp.arange(M, dtype=int)
    # ── target step magnitude Δθ*  (c · RMS(θ)) ────────────────────────
    c          = TRUST_COEFF_DEFAULT         # c ≈ 0.01
    global_rms_params  = jnp.linalg.norm(params) / jnp.sqrt(M)
    global_target      = c * global_rms_params

    # ── boolean masks ──────────────────────────────────────────────────
    mask_h   = idx < 3
    mask_J   = idx >= 3

    # split params / grads once
    p_h, p_J   = params[mask_h], params[mask_J]
    g_h, g_J   = grads[mask_h],  grads[mask_J]

    # ── 2. block-wise target_update  (Δθ*) ────────────────────────────
    
    if per_block_target_update:
        # *Should* tighten the trust ratio separately for the slow-moving h-vector and the typically noisier J-couplings - need sources...
        def rms_block(mask):
            #  per-block RMS
            vec = params[mask]
            return jnp.linalg.norm(vec) / jnp.sqrt(vec.size)
        tgt_h   = c * rms_block(mask_h)
        tgt_J   = c * rms_block(mask_J)
    else:                       
        target_update  = tgt_h = tgt_J = c * global_rms_params
    block_map = {
        
        "h": slice(0, 3),
        "J": slice(3, M),
    }
    test_report = verify_c(
        params,
        grads,
        block_map,
        c,
        per_block=per_block_target_update
    )
    tests = test_report.get('tests',{})
    messages = test_report.get('messages',{})
    metrics = test_report.get('metrics', {})
    # ── block‑wise η_max ceilings (trust ratio) ────────────────────────
    def eta_max(tgt, g):
        raw = tgt / (jnp.linalg.norm(g) + eps)
        return raw,jnp.minimum(max_lr, raw)

    eta_h_raw, eta_h_max = eta_max(tgt_h, g_h)
    eta_J_raw, eta_J_max = eta_max(tgt_J, g_J)

    # ── robust scale r_B  (median + MAD) ───────────────────────────────
    def r_scale(vec):
        if vec.size == 0:
            return 0.0
        med = jnp.median(jnp.abs(vec))
        mad = jnp.median(jnp.abs(vec - med))
        return med + mad

    r_h, r_J = r_scale(g_h), r_scale(g_J)

    # ── per‑parameter learning rates ───────────────────────────────────
    def per_param(η_max, r, g):
        raw = η_max * r / (jnp.abs(g) + r + eps)
        return jnp.clip(raw, 1e-6, max_lr)

    lr_h = per_param(eta_h_max, r_h, g_h)
    lr_J = per_param(eta_J_max, r_J, g_J)

    # ── assemble flat lr_tree ──────────────────────────────────────────
    lr_tree = jnp.zeros_like(params)
    lr_tree = lr_tree.at[mask_h].set(lr_h)
    lr_tree = lr_tree.at[mask_J].set(lr_J)

    if debug:
        print("── digital get_groupwise_lr_trees() ──")
        print(f"  c = {c:.2e}, max_lr = {max_lr:.2e}")
        
        print("\n── VERIFYING c IS WELL-CHOSEN ──")
        for test_name, passed in tests.items():
            status = "PASS" if passed else "FAIL"
            msg = messages.get(test_name, "")
            if passed:
                print(f"  {test_name}: {status}")
            else:
                print(f"  {test_name}: {status}  → {msg}")
        print("\n  Metrics:")
        for metric_name, value in metrics.items():
            print(f"    {metric_name:<10} = {value:.3e}")
        if per_block_target_update:
            print("  → Using Per-Block Target Update!")
        else:
            print("  → Using Global Target Update!")
       
        print(f'‖θ‖/√M (global RMS)   = {global_rms_params:.3e}')
        print(f"Target updates (h, J) = {tgt_h:.2e}, {tgt_J:.2e}")
        print(f"  h‑block: min={lr_h.min():.2e}, max={lr_h.max():.2e}, mean={lr_h.mean():.2e}")
        print(f"  J‑block: min={lr_J.min():.2e}, max={lr_J.max():.2e}, mean={lr_J.mean():.2e}")
        print("---------------------------------------")

    return lr_tree, mask_h, mask_J
# ---------------------------------------------------------------------------
#  Learning‑rate helpers — identical to case 1 in qrc_analog_run_B.py
# ---------------------------------------------------------------------------

TRUST_COEFF_DEFAULT = 0.01          # 1 % of weight‑norm (see Benzing et al.)

def _l2_vec(vec: jnp.ndarray) -> float:
    """Return the ℓ₂ norm as a Python float (JIT‑friendly little helper)."""
    return float(jnp.linalg.norm(vec))

def verify_c(theta: jnp.ndarray,
             grads: jnp.ndarray,
             block_map: dict[str, slice],
             c: float,
             *,
             per_block: bool = False,
             safe_lo: float = 5e-3,      # 0.005 rad  (≈0.3°)
             safe_hi: float = 2e-2,      # 0.02 rad   (≈1.1°)
             weak_floor: float = 1e-3,   # 1 mrad
             frac_limit: float = 5e-2,   # ‖Δθ‖/‖θ‖ < 5 %
             tol_global: float = .20,
             tol_block:  float = .05,
             eps: float = 1e-12):
    """
    Re-implemented verbatim from the analog script.  It safeguards the choice
    of c; see the docstring there for full details.  Returns a dict with keys
    'tests', 'messages', 'metrics'.  (Only metrics are used downstream.)
    """
    M          = theta.size
    rms0       = _l2_vec(theta) / jnp.sqrt(M)
    dtheta     = c * rms0
    g_norms    = {k: _l2_vec(grads[v]) for k, v in block_map.items()}
    max_g      = max(g_norms.values()) + eps
    eta_global = dtheta / max_g

    shift_dom  = eta_global * max_g
    min_g      = min(g_norms.values()) + eps
    shift_w    = eta_global * min_g
    frac       = dtheta / (_l2_vec(theta) + eps)
    upd_norms  = jnp.asarray([eta_global * g for g in g_norms.values()])
    cv_up      = float(jnp.std(upd_norms) / (jnp.mean(upd_norms) + eps))
    limit      = tol_block if per_block else tol_global

    def _check(cond, label, msg):
        return (cond, "" if cond else f"{label} FAIL: {msg}")

    T1, m1 = _check(safe_lo <= shift_dom <= safe_hi,
                    "T1", f"dominant shift {shift_dom:.3e} ∉ [{safe_lo},{safe_hi}]")
    T2, m2 = _check(shift_w > weak_floor,
                    "T2", f"weakest shift {shift_w:.2e} < {weak_floor:.2e}")
    T3, m3 = _check(frac < frac_limit,
                    "T3", f"‖Δθ‖/‖θ‖ = {frac:.3f} > {frac_limit}")
    T4, m4 = _check(cv_up <= limit,
                    "T4", f"CV_B_update = {cv_up:.3f} > {limit}")

    return {
        "tests":   dict(T1=T1, T2=T2, T3=T3, T4=T4),
        "messages": dict(T1=m1, T2=m2, T3=m3, T4=m4),
        "metrics": dict(RMS0=rms0, Δθₛ=dtheta, η_glob=eta_global,
                        shift_dom=shift_dom, shift_weak=shift_w, CV_up=cv_up),
    }


def get_optimizer(
    case: int,
    opt_lr,
    T: int, 
    num_epochs: int = None,
    # Adam hyper-parameters
    b1: float = 0.99,
    b2: float = 0.999,
    eps: float = 1e-8,
    # ReduceLROnPlateau hyper-parameters
    factor: float = 0.9,
    patience: int = 50,
    rtol: float = 1e-4,
    atol: float = 0.0,
    cooldown: int = 0,
    accumulation_size: int = 1,
    cost_threshold=1e-3,
    grad_var_tol=1e-6,
    min_scale: float = 0.0,
    # fraction of epochs to wait before enabling plateau logic
    warmup_steps:int = 100,
    warmup_fraction: float = 0.5,
):

    def wrap(base_optimizer, passes_value: bool, passes_step: bool):
        def init_fn(params):
            return base_optimizer.init(params)

        def update_fn(updates, state, params=None, *, value=None, step=None, **extra):
            kwargs = {}
            # print(f"state: {state}")
            if passes_value:
                kwargs["value"] = value
            if passes_step:
                kwargs["step"] = step
            # print(f"kwargs: {kwargs}")
            return base_optimizer.update(updates, state, params=params, **kwargs)

        return optax_base.GradientTransformationExtraArgs(init_fn, update_fn)
    def make_chain(component):
        masks = {"t": [True, False, False],
                 "h": [False, True, False],
                 "J": [False, False, True]}[component]
        return lambda gt: optax.masked(gt, dict(zip("thJ", masks)))

    if case == 1:
        desc = 'masked per group adam'

        base_opt = optax.chain(
            optax.clip_by_global_norm(1.0),

            optax.masked(optax.adam(opt_lr["h"],b1=b1,b2=b2,eps=eps), mask={"h": True, "J": False}),
            optax.masked(optax.adam(opt_lr["J"],b1=b1,b2=b2,eps=eps), mask={"h": False, "J": True})
        )
        return desc, wrap(base_opt, passes_value=False, passes_step=False)

    elif case == 2:
        desc = "masked per group adam & ReduceLROnPlateau"
        def make_groupwise_plateau_optimizer(
            opt_lr: dict[str, jnp.ndarray],
            *,
            b1=0.99, b2=0.999, eps=1e-8,
            factor=0.9, patience=50,  
            rtol=1e-4, atol=0.0, 
            cooldown=0, accum_size=10,
            min_scale=0.1, 
        ):
            def one_group(rate):
                return optax.chain(
                    optax.inject_hyperparams(optax.adam)(
                        learning_rate=rate, b1=b1, b2=b2, eps=eps
                    ),
                    optax.contrib.reduce_on_plateau(
                        factor=factor, patience=patience,
                        rtol=rtol, atol=atol,
                        cooldown=cooldown, accumulation_size=accum_size, 
                        min_scale=min_scale,
                    ),
                )

            
            mask_h = { "h": True,  "J": False}
            mask_J = { "h": False, "J": True}

            return optax.chain(
                optax.clip_by_global_norm(1.0),
                
                optax.masked(one_group(opt_lr["h"]), mask_h),
                optax.masked(one_group(opt_lr["J"]), mask_J),
            )
        
        optimizer = make_groupwise_plateau_optimizer(opt_lr, 
            factor=factor, 
            patience=patience, 
            rtol=rtol,
            atol=atol,
            cooldown=cooldown,
            accum_size=accumulation_size,
            min_scale=min_scale,
            )
        return desc, optimizer
    # default fallback
    desc = "per-param Adam"
    base_opt = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.inject_hyperparams(optax.adam)(
            learning_rate=opt_lr, b1=b1, b2=b2, eps=eps
        ),
    )
    return desc, wrap(base_opt, passes_value=False, passes_step=False)

def run_experiment(params, 
    bath_params, 
    num_epochs, 
    n_rsv_qubits, 
    n_ctrl_qubits, 
    K_coeffs, 
    trotter_steps, 
    static, 
    gate, 
    gate_name, 
    folder, 
    test_size, 
    training_size, 
    opt_lr,
    num_bath,
    dataset_key,
    case_num,
    PATIENCE = 3,
    ACCUMULATION_SIZE=0,
    RTOL = 1e-4,
    ATOL=1e-7,
    COOLDOWN=0,
    FACTOR = 0.9,
    MIN_SCALE = 0.01,
    scale_by_num_train=True,
    per_block_target_update=False,
    ):
    N_ctrl = n_ctrl_qubits
    N_reserv = n_rsv_qubits
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

    filename = os.path.join(folder_gate, f'data_run_{len(files_in_folder)}.pickle')
    
    X, y =generate_dataset(gate, N_ctrl,training_size= training_size, key= dataset_key, new_set=False)

    # print(f"training state #1: {X[0]}")
    test_dataset_key = jax.random.split(dataset_key)[1]
    test_X, test_y = generate_dataset(gate, N_ctrl,training_size= 2000, key=test_dataset_key, new_set=True)
  
    
    qrc = QuantumReservoirNetwork(n_rsv_qubits=n_rsv_qubits, n_ctrl_qubits=n_ctrl_qubits,  K_coeffs=K_coeffs, static=static)
    ctrl_wires = qrc.ctrl_qubits
    rsv_wires = qrc.rsv_qubits# wires of the control qubits (i.e. number of qubits in the control)
    
    @qml.qnode(qrc.dev, interface="jax",diff_method="backprop")
    def circuit(params, input_state):
        x_coeff = params[0]
        z_coeff = params[1]
        y_coeff = params[2]
        J_coeffs = params[3:]
        qml.StatePrep(input_state, wires=[*qrc.ctrl_qubits])
        qrc.set_gate_reservoir(x_coeff,z_coeff,y_coeff)
        for i in range(trotter_steps):
            
        
            step = len(rsv_wires)*len(ctrl_wires)
            qrc.set_gate_params(J_coeffs[i*step:(i+1)*step])
            qrc.set_gate_reservoir(x_coeff,z_coeff,y_coeff)
        return qml.density_matrix(wires=[*qrc.ctrl_qubits])

    jit_circuit = jax.jit(circuit)
    vcircuit = jax.vmap(jit_circuit, in_axes=(None, 0))
    def batched_cost_helper(params, X, y):
        # Process the batch of states
        batched_output_states = vcircuit(params, X)
        
        # Compute fidelity for each pair in the batch and then average
        fidelities = jax.vmap(qml.math.fidelity)(batched_output_states, y)
        fidelities = jnp.clip(fidelities, 0.0, 1.0)
        average_fidelity = jnp.mean(fidelities)
        
        return 1. - average_fidelity  # Minimizing infidelity
        
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
        params = jnp.asarray(params, dtype=jnp.float64)
        X = jnp.asarray(X, dtype=jnp.complex128)
        y = jnp.asarray(y, dtype=jnp.complex128)
        batched_output_states = vcircuit(params, X)

        
        fidelities = jax.vmap(qml.math.fidelity)(batched_output_states, y)
       
        return fidelities
    
    
    # ------------------------------------------------------------------
    #  Case-1 learning rate (analog style)  ────────────────────────────
    # ------------------------------------------------------------------
    if opt_lr == None:
        t0          = time.time()
        init_loss, init_grads = jax.value_and_grad(cost_func)(
            params, X, y, n_rsv_qubits, n_ctrl_qubits, trotter_steps, static
        )
        
        dt_init = time.time() - t0
        flat_grads = jnp.ravel(init_grads)

        cv0 = coef_variation_by_block(case_num, flat_grads,T=trotter_steps,
                               N_ctrl=n_ctrl_qubits, N_reserv=n_rsv_qubits)
        cv_str = ", ".join(f"{k}: {v:.4f}" for k, v in cv0.items())
        print("\nInitial coefficient of variation (CV_B) by block at step 0: ", cv_str)
        if case_num in (0,1):
            print(f"Getting init lr stats for case #{case_num}: ")
            lr_tree,mask_h,mask_J = get_groupwise_lr_trees(
                params        = params,
                grads         = init_grads,
                num_train     = training_size,
                N_C            = n_ctrl_qubits,
                N_R            = n_rsv_qubits,
                T    = trotter_steps,             
                max_lr        = 0.2,
                debug         = True
            )
            
         
            if case_num == 0:
                opt_lr = lr_tree
            else:
                lr_h   = lr_tree[mask_h]
                lr_J   = lr_tree[mask_J]
                opt_lr = {
                    "h": lr_h,
                    "J": lr_J
                }
        else:
            print(f"Getting init lr stats for case #{case_num}: ")
            lr_tree, assignment_mask = get_groupwise_lr_trees_slicewise(
                params        = params,
                grads         = init_grads,
                num_train     = training_size,
                N_C            = n_ctrl_qubits,
                N_R            = n_rsv_qubits,
                T    = trotter_steps,  
                max_lr        = 0.2,
                debug         = False
            )
            opt_lr = (lr_tree, assignment_mask)


        lr_groups_init = lr_group_dict(case_num, lr_tree, T=trotter_steps, N_ctrl=n_ctrl_qubits, N_reserv=n_rsv_qubits)
        

        cost = init_loss
    
    opt_descr, opt = get_optimizer(
        case=case_num,
        opt_lr     = opt_lr,       # either a dict or (lr_tree,assignment_mask)
        T = trotter_steps,  
        num_epochs = num_epochs,
        factor=FACTOR,
        patience=PATIENCE,
        rtol=RTOL,
        atol=ATOL,
        cooldown=COOLDOWN,
        accumulation_size=ACCUMULATION_SIZE,
        min_scale=MIN_SCALE,
        warmup_steps = 400,
        # warmup_fraction = .15,
        )

    if case_num in [1]:
        params = {
            'h': params[mask_h],
            'J':params[mask_J]
        }

    print("________________________________________________________________________________")
    print(f"Starting optimization for {gate_name}(epochs: {num_epochs}) T = {trotter_steps}, N_r = {N_reserv}, N_bath = {num_bath}...\n")
    print(f"    Mean LR: {np.mean(lr_tree):.5f}), "
        f"variance={np.var(lr_tree)}…"
        )
    # print(f"Initial Loss: {init_loss:.4f}, initial_gradients: {np.mean(np.abs(init_grads))}. Time: {dt:.2e}")
   
    
    opt_state = opt.init(params)

    @jit
    def update(params, opt_state, X, y, step):
        X = jnp.asarray(X, dtype=jnp.complex128)
        y = jnp.asarray(y, dtype=jnp.complex128)
        # Ensure inputs are float64
        if isinstance(params,dict):
            flat = jnp.concatenate([jnp.ravel(v) for v in params.values()]) 
            loss, grads_flat = jax.value_and_grad(cost_func)(flat, X, y, n_rsv_qubits, n_ctrl_qubits, trotter_steps, static)
        # params = jnp.asarray(params, dtype=jnp.float64)
        else:
            flat = params
            loss, grads_flat = jax.value_and_grad(cost_func)(flat, X, y, n_rsv_qubits, n_ctrl_qubits, trotter_steps, static)
        
        
        grads_pytree = grad_group_dict(case_num, grads_flat,T=trotter_steps,N_ctrl=N_ctrl, N_reserv=N_reserv)
       
        # grads_pytree = {
               
        #         "h": grads_flat[mask_h],
        #         "J": grads_flat[mask_J]
        #     }
        grad_for_opt = grads_flat if case_num == 0 else grads_pytree
        updates, new_opt_state = opt.update(
            grad_for_opt, 
            opt_state, 
            params=params, 
            value=loss,
            step=step
            )
        new_params = optax.apply_updates(params, updates)
       
        return new_params, new_opt_state, loss, grads_flat,grad_for_opt
    fullstr = time.time()

    prev_cost = float('inf')  # Initialize with infinity

    backup_params = None
    backup_cost = float('inf')  

    cost_res = 1
    costs = []

    param_per_epoch,grads_per_epoch,rocs = [],[],[]
    epoch = 0
    improvement = True

    add_more = True
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
    cv_B_per_epoch = {}
    s = time.time()
    new_scale = 1.0
    while epoch  < num_epochs or improvement:
        
        params, opt_state, cost,grads_flat,grads_pytree = update(params, opt_state, X, y,step=epoch)
        if 'grad_groups_history' not in locals():
            grad_groups_history = []
        grad_groups_history.append(grads_pytree)
        grad =grads_flat
        costs.append(cost)
        param_per_epoch.append(params)
        grads_per_epoch.append(grads_flat)
        
        cv_B = coef_variation_by_block(case_num, grads_flat,
                            trotter_steps, N_ctrl, N_reserv)
        cv_B_per_epoch[epoch] = cv_B
       
       
     
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
                # print(f"    - setting cond1: initial mean(grad) {initial_meangrad:2e}, threshold: {cond1:2e}")
                cond2 = initial_vargrad * 1e-2
                # print(f"    - setting cond2: initial var(grad) {initial_vargrad:2e}, threshold: {cond2:2e}")
            
            acceleration = get_rate_of_improvement(cost,prev_cost,second_prev_cost)
            if epoch >= 25 and not a_condition_set and acceleration < 0.0:
                average_roc = np.mean(np.array(rocs[10:]))
                a_marked = np.abs(average_roc)
                a_threshold = max(a_marked * 1e-3, 1e-7)
                # a_threshold = a_marked*1e-3 if a_marked*1e-3 > 9e-7 else a_marked*1e-2
                
                # print(f"acceration: {a_marked:.2e}, marked: {a_threshold:.2e}")
               
                a_condition_set = True
            rocs.append(acceleration)
        if epoch == 0 or (epoch + 1) % 250 == 0:
            
            var_grad = np.var(grad,ddof=1)
            mean_grad = np.mean(jnp.abs(grad))
            max_grad = max(jnp.abs(grad))
            e = time.time()
            epoch_time = e - s
            
            if cost < 1e-4:
                print(f'Epoch {epoch + 1} --- cost: {cost:.7e}, '
                f'CV_B: {{{cv_str}}}, '
                f'[t: {epoch_time:.1f}s]')
            
            else:
                print(f'Epoch {epoch + 1} --- cost: {cost:.5f}, '
                    f'CV_B: {{{cv_str}}}, '
                    f'[t: {epoch_time:.1f}s]')
                
            s = time.time()
            


        
        # Check if there is improvement
        if cost < prev_cost:
            prev_cost = cost  # Update previous cost to current cost
            # improvement = True
            if isinstance(params,dict):
                params_flat = jnp.concatenate([jnp.ravel(v) for v in params.values()]) 
                current_cost_check = cost_func(params_flat, X, y,n_rsv_qubits, n_ctrl_qubits, trotter_steps, static)
            else:
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
        var_condition= np.var(grads_flat,ddof=1) < 1e-14
        gradient_condition= max(jnp.abs(grad)) < 1e-8
        cost_confition = cost < 1e-12
        
        if var_condition or gradient_condition or epoch >=2*num_epochs or cost_confition:
            if epoch >=2*num_epochs:
                print(f"Epoch greater than max. Ending opt at epoch: {epoch}")
            if var_condition:
                print(f"Variance of the gradients below threshold [{np.var(grads_flat,ddof=1):.1e}], thresh:  1e-10. Ending opt at epoch: {epoch}")
            if cost_confition:
                print(f"Cost below threshold [{cost:.1e}]. Ending opt at epoch: {epoch}")
            if gradient_condition:
                print(f"Magnitude of maximum gradient is less than threshold [{max_grad:.1e}]. Ending opt at epoch: {epoch}")

            break
        # Check if there is improvement
        second_prev_cost = prev_cost  # Move previous cost back one step
        prev_cost = cost  # Update previous cost with the current cost

     


        epoch += 1
    # if backup_cost < cost:
    if backup_cost < cost and not epoch < num_epochs and backup_epoch < epoch - 25:
        print(f"\nbackup cost (epoch: {backup_epoch}) is better with: {backup_cost:.3e} <  {cost:.3e}: {backup_cost < cost}")

        params = backup_params
    else:
        print(f'Final cost: {cost:.3e}')
    fullend = time.time()
    print(f"time optimizing: {fullend-fullstr} improvement count: {improvement_count}")
    
    
    df = pd.DataFrame()
    
    print(f"Testing opt params against {test_size} new random states...")
    if isinstance(params,dict):
        params = jnp.concatenate([jnp.ravel(v) for v in params.values()]) 
        
    fidelities =  final_costs(params, X=test_X, y=test_y, n_rsv_qubits=n_rsv_qubits, n_ctrl_qubits=n_ctrl_qubits, trotter_steps=trotter_steps, static=static)
    f64 = np.array(fidelities, dtype=np.float64)
    mask = f64 > 1.0
    if np.any(mask):
        # amount over 1.0
        errs = f64[mask] - 1.0
        # reflect and clamp
        f64[mask] = np.clip(1.0 - errs, 0.0, 1.0)
    infids = 1.0 - f64

    avg_fidelity = np.mean(f64)
    if 1.-avg_fidelity <1e-4:
        print(f'Avg Fidelity: {avg_fidelity:.8e}, Err: {float(np.log10(infids).mean()):.5f}')
    else: 
        print(f'Avg Fidelity: {avg_fidelity:.5f}')
    # eps = jnp.finfo(fidelities.dtype).eps
    # mask = jnp.abs(fidelities - 1.0) <= eps
    # num_ones = jnp.sum(mask)
    # print(f'  # fidelities ≈ 1.0 (±{eps:.3e}) (jax): {int(num_ones)}')

    eps = np.finfo(f64.dtype).eps
    num_ones = np.sum(np.abs(f64 - 1.0) <= eps)
    print(f'  # fidelities ≈ 1.0 (±{eps:.3e}) (numpy): {num_ones}')
    assert not num_ones
   # infidelities_backup =  final_costs(backup_params, X=test_X, y=test_y, n_rsv_qubits=n_rsv_qubits, n_ctrl_qubits=n_ctrl_qubits, trotter_steps=trotter_steps, static=static)
  #  print('Backup params infidelity: ', np.mean(infidelities_backup))
    x_coeff = params[0]
    z_coeff = params[1]
    y_coeff = params[2]
    J_coeffs = params[3:]
    
    data = {
        'Gate': base64.b64encode(pickle.dumps(gate)).decode('utf-8'),
        'opt_description': opt_descr,
        'epochs': num_epochs,
        'selected_indices': selected_indices,
        'lrs': learning_rates,
        'scales_per_epoch': scales_per_epoch,  # Add scales per epoch
        'scale_reduction_epochs': scale_reduction_epochs,  # Add epochs of scale reduction
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
        "avg_fidelity":   float(f64.mean()),
        "avg_infidelity": float(infids.mean()),
        "avg_log_error":  float(np.log10(infids).mean()),
        "testing_results": f64.tolist(),
        "infidelities":    infids.tolist(),
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
  



if __name__=='__main__':

    N_ctrl = 1
   
    

    # res = [1, 2, 3]
    if N_ctrl == 1:
        trots = [1,2,3,4,5,6,7,8]
        res = [1,2]
    elif N_ctrl == 2:
        trots = [1,4,8,10,12,14,16,18,20,22,24]
        trots = [6,26,28]
        res = [1,2]
    elif N_ctrl == 3:
        trots = [25,28,30,32,35,38,40,45]
        res = [1]

    trots = [2,5]
    res = [1]
    rsv_qubits_list = res
    trotter_step_list = trots
    # omegas = generate_omegas(N_ctrl)
    # folder = f'./digital_results_trainable_global/trainsize_{training_size_list[0]}_same_epoch{num_epochs}/'
    # folder = f'./digital_results_trainable_global/trainsize_{training_size_list[0]}_optimized_by_cost3/'
    
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

    opt_lr = None
    baths = [False]
    num_baths = [0]
    training_size = 20
    case_num = 0 # "per-param Adam"
    # case_num = 1 # "masked per group adam"

    patience= 5
    accum_size = 5
    rtol = 1e-4
    atol=1e-7
    scale_by_num_train=False
    per_block_target_update=True

    K_0 = 2.
    bound_key = 'pi'
    bound_dict = {'pi': np.pi,
    '1':1,
    '2pi':2*np.pi,
    }
    p_bound = bound_dict[bound_key]
    static =False

    num_epochs = 1500
    
    noise_central = 0.01
    noise_range = 0.002
    test_size = 2000
    bath = False
    folder = f'./digital_results/trainsize_{training_size}_epoch{num_epochs}/case_{case_num}/tgt_granular_{per_block_target_update}/'
    for gate_idx,gate in enumerate(gates):
        

        for trotter_steps in trotter_step_list:
            for n_rsv_qubits in rsv_qubits_list:
            
                

                    
                N = N_ctrl +n_rsv_qubits
                # main seed DO NOT DELETE
                params_key_seed = gate_idx*121 * n_rsv_qubits + 12345 * trotter_steps *n_rsv_qubits

                # run #2 seed to verfy overparameterization
                # params_key_seed = gate_idx*987 * n_rsv_qubits + 749 * trotter_steps *n_rsv_qubits + 5
                # print(f"params_key_seed: {params_key_seed}")
                params_key = jax.random.PRNGKey(params_key_seed)
                
                # main dataset_seed seed DO NOT DELETE
                dataset_seed = N_ctrl * gate_idx + gate_idx**2 + N_ctrl
                # dataset_seed = 10
                dataset_key = jax.random.PRNGKey(dataset_seed)

                

                params = jax.random.uniform(params_key, shape=(3 + (N_ctrl * n_rsv_qubits) * trotter_steps,), 
                                            minval=-p_bound, maxval=p_bound, dtype=jnp.float64)
            
                _, params_subkey1, params_subkey2 = jax.random.split(params_key, 3)
                n_ctrl_qubits = N_ctrl
                
                
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
                run_experiment(
                    params=params, 
                    bath_params=None, 
                    num_epochs=num_epochs, 
                    n_rsv_qubits=n_rsv_qubits, 
                    n_ctrl_qubits=N_ctrl, 
                    K_coeffs=K_coeffs, 
                    trotter_steps=trotter_steps, 
                    static=static, 
                    gate=gate, 
                    gate_name=gate.name if random else str(gate), 
                    folder=folder, 
                    test_size=test_size, 
                    training_size=training_size, 
                   
                    opt_lr=None, 
                    num_bath=0,
                    dataset_key=dataset_key,
                    case_num=case_num,
                    scale_by_num_train=scale_by_num_train,
                    per_block_target_update=per_block_target_update
                )