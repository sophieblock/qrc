import pennylane as qml
import os
import pickle
import re
import seaborn as sns

import pandas as pd
import matplotlib.pyplot as plt
from qiskit import *

from jax import numpy as jnp
import sympy
import matplotlib.pyplot as plt
import base64
import pickle
from qutip import *

 # Using pennylane's wrapped numpy
from sympy import symbols, MatrixSymbol, lambdify, Matrix, pprint
import jax
import numpy as np
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

import os
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def is_valid_pickle_file(file_path):
    """Check if a pickle file is valid and re-save it after cleaning."""
    try:
        if file_path.exists() and file_path.stat().st_size > 0:
            with open(file_path, 'rb') as f:
                try:
                    data = pickle.load(f)
                    # Clean the data to ensure compatibility
                    cleaned_data = clean_array(data)
                    
                    # Re-save the cleaned data to the same file
                    with open(file_path, 'wb') as fw:
                        pickle.dump(cleaned_data, fw)
                    
                    return True
                except EOFError:
                    print(f"File {file_path} is corrupted.")
                    return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False
def clean_array(data):
    """Helper function to clean deprecated JAX arrays."""
    if isinstance(data, np.ndarray):
        return np.array(data)  # Strip any unsupported attributes
    elif isinstance(data, dict):
        return {k: clean_array(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_array(v) for v in data]
    elif hasattr(data, 'named_shape'):  # Specifically check for deprecated JAX atributes
        del data.named_shape
        return data
    else:
        return data  # Return as is if not an array or collection


def get_cached_data(base_path, N_ctrl, K_0):
    """
    Load cached_data if it exists, else return empty placeholders.
    Returns (cached_data, processed_files).
    
    Expects a cache file named: 'digital_new_QFIM_Nc_{N_ctrl}_{K_0}K.pkl'
    """
    cache_file = os.path.join(base_path, f'digital_new_QFIM_Nc_{N_ctrl}_{K_0}K.pkl')

    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            cached_data, processed_files = pickle.load(f)
        print(f"[INFO] Loaded cache for N_ctrl={N_ctrl} from disk at {cache_file}.")
        return cached_data, processed_files
    else:
        print(f"[WARN] No cache for N_ctrl={N_ctrl} at {cache_file}.")
        return {}, set()
def load_and_clean_pickle(data_file):
    """Helper function to load and clean a pickle file."""
    with open(data_file, 'rb') as f:
        df = pickle.load(f)
        return clean_array(df)
def save_cached_data(base_path, cached_data, processed_files, N_ctrl, K_0):
    """
    Save the (cached_data, processed_files) to a single pickle file.
    """
    cache_file = os.path.join(base_path, f'digital_new_QFIM_Nc_{N_ctrl}_{K_0}K.pkl')
    with open(cache_file, 'wb') as f:
        pickle.dump((cached_data, processed_files), f)
    print(f"[INFO] Cache saved for N_ctrl={N_ctrl} at {cache_file}.")

def extract_trotter_step(data_file):
    """
    For example, parse the directory name 'step_10' => 10
    """
    return int(data_file.parent.name.split('_')[-1])

def extract_Nr(data_file):
    """
    For example, parse the directory 'Nr_2' => 2
    """
    return int(data_file.parent.parent.name.split('_')[-1])


def process_data_expanded(df, threshold, by_test, Nc, N_R, trot, print_bool=False):
    """
    Create a list of dictionaries, each representing one (fixed_params_key, test_key) row.
    Returns a list of row-dicts for direct DataFrame construction.
    """
    rows = []
    for fixed_params_key, test_dict in df.items():
        for test_key, results in test_dict.items():
            qfim_eigvals = results.get('qfim_eigvals', None)
            entropy      = results.get('entropy', None)
            row = {
                "N_ctrl": Nc,
                "N_reserv": N_R,
                "Trotter_Step": trot,
                "fixed_params_key": fixed_params_key,
                "test_key": test_key,
                "qfim_eigvals": qfim_eigvals,
                "entropy": entropy,
                # Add any other fields you might need later
            }
            rows.append(row)
    return rows
def build_df_expanded_from_raw(base_path, sample_range, model_type, N_ctrls, K_str, threshold, by_test):
    """
    Build the df_expanded DataFrame solely from the raw pickle files on disk.
    This function scans the directory structure under base_path/QFIM_results and
    processes each data.pickle file using process_data_expanded. It does not use
    any cached expanded data.

    Parameters
    ----------
    base_path : str
        The base directory where the QFIM_results directories are located.
    sample_range : str
        The sample range string used in the directory structure (e.g., "pi").
    model_type : str
        The model type (e.g., "gate").
    N_ctrls : list of int
        A list of N_ctrl values to process.
    K_str : str
        The K parameter string (e.g., "1").
    threshold : float
        The threshold value passed to the processing functions.
    by_test : bool
        The by_test flag to pass to processing functions.

    Returns
    -------
    pd.DataFrame
        A DataFrame constructed by processing every data.pickle file’s raw data.
    """
    from pathlib import Path
    import os
    import pandas as pd

    all_expanded_rows = []
    
    # Iterate over each N_ctrl value
    for N_ctrl in N_ctrls:
        model_path = Path(base_path) / "QFIM_results" / model_type / f"Nc_{N_ctrl}" / f"sample_{sample_range}/{K_str}xK"
        if not model_path.exists():
            print(f"[WARN] Model path {model_path} does not exist for N_ctrl={N_ctrl}.")
            continue
        
        # Iterate over each Nr directory
        for Nr in sorted(os.listdir(model_path)):
            Nr_path = model_path / Nr
            if not Nr_path.is_dir():
                continue
            
            # Iterate over each trotter step directory
            for trotter_step_dir in sorted(os.listdir(Nr_path)):
                trotter_step_path = Nr_path / trotter_step_dir
                if not trotter_step_path.is_dir():
                    continue
                
                data_file = trotter_step_path / "data.pickle"
                if not data_file.exists():
                    continue
                # Validate the pickle file without using cached results
                if not is_valid_pickle_file(data_file):
                    continue
                
                # Load the raw pickle data
                raw_data = load_and_clean_pickle(data_file)
                # Extract trotter step and reservoir count from the directory structure
                try:
                    trotter_step_num = extract_trotter_step(data_file)
                    reservoir_count = extract_Nr(data_file)
                except Exception as e:
                    print(f"[ERROR] Could not extract parameters from {data_file}: {e}")
                    continue

                # Process raw data using your expanded function (do not use cached expanded rows)
                expanded_rows = process_data_expanded(raw_data, threshold, by_test, N_ctrl, reservoir_count, trotter_step_num)
                all_expanded_rows.extend(expanded_rows)

    return pd.DataFrame(all_expanded_rows)


from scipy.stats import median_abs_deviation
def spread_per_sample_vectorized(
    eigs_2d, method="variance", threshold=1e-12, ddof=1, scale="normal"
):
    """
    For each row in eigs_2d (one 'draw'), threshold small eigenvalues to 0,
    renormalize so they sum to 1, then compute spread of log(eigs).
    Returns a 1D array: one spread value per draw.
    """
    # Clip to zero below threshold
    clipped = np.where(eigs_2d > threshold, eigs_2d, 0.0)
    row_sums = clipped.sum(axis=1, keepdims=True)
    nonzero_mask = (row_sums[:, 0] > 0)
    clipped[nonzero_mask] /= row_sums[nonzero_mask]

    with np.errstate(divide='ignore'):
        logs = np.log(clipped, out=np.zeros_like(clipped), where=(clipped > 0))

    def compute_spread_metric(values, method, ddof, scale):
        if len(values) <= 1:
            return 0.0
        if method == "variance":
            return np.var(values, ddof=ddof)
        elif method == "mad":
            return median_abs_deviation(values, scale=scale)
        else:
            raise ValueError(f"Unknown method: {method}")

    results = []
    # '/Volumes/Block 1/untitled folder'
    for i in range(eigs_2d.shape[0]):
        row_nonzero = logs[i, clipped[i] > 0]
        spread_val = compute_spread_metric(row_nonzero, method, ddof, scale)
        results.append(spread_val)

    return np.array(results)
def spread_pooling_vectorized(
    eigs_2d, method="variance", threshold=1e-12, ddof=1, scale="normal"
):
    """
    Flatten all draws into one big array (above threshold), renormalize,
    compute the spread of log(eigs) for the entire pool.
    """
    flat = eigs_2d.ravel()
    filtered = flat[flat > threshold]
    if filtered.size <= 1:
        return 0.0
    total = filtered.sum()
    filtered /= total

    with np.errstate(divide='ignore'):
        logs = np.log(filtered)

    if method == "variance":
        return np.var(logs, ddof=ddof)
    elif method == "mad":
        return median_abs_deviation(logs, scale=scale)
    else:
        raise ValueError(f"Unknown method: {method}")

def effective_dimension_from_paper(
    fisher_eigenvalues: np.ndarray,
    n: int,
    gamma: float,
    V_theta: float
) -> float:
    """
    Compute the effective dimension d_{gamma,n}(M_theta) as in Eq. (2) of the paper:

        d_{gamma,n} = 2 * log( (1/sqrt(V_theta)) * sqrt(det(I_d + alpha * Fhat(theta))) )
                     = - log(V_theta) + log( prod_i (1 + alpha * lambda_i) )

    where alpha = gamma * n / (2 * log(n)) and the product runs over the eigenvalues lambda_i
    of the Fisher matrix.

    Parameters
    ----------
    fisher_eigenvalues : np.ndarray
        1D array of eigenvalues (nonnegative) of the Fisher matrix Fhat(theta).
    n : int
        Number of data samples, used in alpha = gamma*n / (2*log(n)).
    gamma : float
        Constant in (0, 1], controlling how strongly the Fisher enters the dimension.
    V_theta : float
        The volume of the parameter space.  (Set to 1 if unsure.)

    Returns
    -------
    d_eff : float
        The effective dimension from the paper's Eq. (2).
    """
    if n < 2 or np.log(n) <= 0:
        # Edge case: can't compute alpha if n=1 or log(n) <= 0
        return 0.0

    alpha = gamma * n / (2.0 * np.log(n))

    # Avoid negative or complex logs if alpha < 0 or fisher_eigenvalues < 0
    # but these should be nonnegative anyway (Fisher is PSD).
    log_det_term = 0.0
    for lam in fisher_eigenvalues:
        # (1 + alpha * lam)
        log_det_term += np.log1p(alpha * lam) if lam > 0 else 0.0

    # So the core formula from the paper: 
    #    d_eff = - log(V_theta) + sum_i( log(1 + alpha * lam_i) )
    d_eff = - np.log(V_theta) + log_det_term
    return d_eff
def approximate_effective_dimension(
    eigenvalues: np.ndarray,
    n: int,
    gamma: float,
    vol_param_space: float = 1.0
) -> float:
    """
    Approximate the effective dimension d_{gamma,n} by summing log(1 + alpha * lambda_i).
    This is a discrete local version of the formula from
    'The power of quantum neural networks' Section 3.2, ignoring integrals over Theta.

    Parameters
    ----------
    eigenvalues : np.ndarray
        1D array of eigenvalues from a (local) Fisher matrix.
    n : int
        The sample size, used in the factor alpha = gamma*n / (2*log(n)).
    gamma : float
        The hyperparameter in (0,1], controlling how strongly the Fisher enters the dimension.
    vol_param_space : float
        Approximate volume of the parameter space, V_theta. Often set to 1 if unknown.

    Returns
    -------
    d_eff : float
        The approximate effective dimension for these eigenvalues.
    """
    # Avoid log(0) or log negative if n <= 1
    if n <= 1:
        return 0.0

    alpha = gamma * n / (2.0 * np.log(n))

    # sum(log(1 + alpha * lambda_i)) ignoring negative/zero-lambda issues
    inside_sum = 0.0
    for lam in eigenvalues:
        if lam > 0.0:
            inside_sum += np.log(1.0 + alpha * lam)

    # The final formula is: d_eff = -log(V_theta) + sum(...)
    d_eff = -np.log(vol_param_space) + inside_sum
    return d_eff
def compute_single_draw_stats(
    eigvals,
    threshold=1e-12,
    spread_methods=("variance", "mad"),
    ddof=1,
    scale="normal",
    do_effective_dim=True,
    vol_param_space=1.0,
    gamma=0.1,
    n=100,
    V_theta=1.0,
):
    """
    Compute QFIM statistics for a SINGLE set of eigenvalues (one draw).

    For this single array `eigvals`, it computes:
      - draw_rank: number of nonzero eigenvalues (after thresholding)
      - variance_all_eigenvalues: variance computed on all eigenvalues
      - variance_nonzero_eigenvalues: variance computed on nonzero eigenvalues only
      - trace_eigenvalues: sum of eigenvalues (the trace)
      - variance_normalized_by_length: variance_all_eigenvalues divided by the total number of eigenvalues
      - trace_normalized_by_length: trace_eigenvalues divided by the total number of eigenvalues
      - variance_normalized_by_rank: variance_all_eigenvalues divided by the draw_rank
      - trace_normalized_by_rank: trace_eigenvalues divided by the draw_rank
      - effective_dimension_paper: effective dimension computed using the paper's formula
      - spread_metric_{method}: for each method in spread_methods, the spread of the log of normalized eigenvalues

    Returns a single dictionary with these metrics.
    """
    import numpy as np

    # If the row's 'qfim_eigvals' is a list with exactly one array, unwrap it.
    if isinstance(eigvals, list) and len(eigvals) == 1 and isinstance(eigvals[0], (list, np.ndarray)):
        eigvals = eigvals[0]

    arr = np.array(eigvals, dtype=float)
    # Zero out values below threshold.
    arr = np.where(arr < threshold, 0.0, arr)

    draw_rank = np.count_nonzero(arr)
    variance_all_eigenvalues = np.var(arr)
    nonzero = arr[arr > 0]
    variance_nonzero_eigenvalues = np.var(nonzero) if nonzero.size > 1 else 0.0
    trace_eigenvalues = arr.sum()

    length = len(arr) if len(arr) else 1
    variance_normalized_by_length = variance_all_eigenvalues / length
    trace_normalized_by_length = trace_eigenvalues / length
    variance_normalized_by_rank = variance_all_eigenvalues / draw_rank if draw_rank > 0 else 0.0
    trace_normalized_by_rank = trace_eigenvalues / draw_rank if draw_rank > 0 else 0.0

    # Compute spread metrics for this single draw.
    arr_2d = arr.reshape(1, -1)
    spread_metrics = {}
    for method in spread_methods:
        spread_value = spread_per_sample_vectorized(
            arr_2d, method=method, threshold=threshold, ddof=ddof, scale=scale
        )
        # For a single row, there's exactly 1 value.
        spread_metrics[f"spread_metric_{method}"] = spread_value[0] if spread_value.size > 0 else 0.0

    # Compute effective dimension for this single draw, if desired.
    effective_dimension_paper = 0.0
    if do_effective_dim and arr.size > 0:
        effective_dimension_paper = effective_dimension_from_paper(
            fisher_eigenvalues=arr, n=n, gamma=gamma, V_theta=V_theta
        )

    # Build the dictionary with descriptive keys.
    stats_dict = {
        "draw_rank": draw_rank,
        "variance_all_eigenvalues": variance_all_eigenvalues,
        "variance_nonzero_eigenvalues": variance_nonzero_eigenvalues,
        "trace_eigenvalues": trace_eigenvalues,
        "variance_normalized_by_length": variance_normalized_by_length,
        "trace_normalized_by_length": trace_normalized_by_length,
        "variance_normalized_by_rank": variance_normalized_by_rank,
        "trace_normalized_by_rank": trace_normalized_by_rank,
        "effective_dimension_paper": effective_dimension_paper,
    }
    stats_dict.update(spread_metrics)

    return stats_dict

if __name__ == "__main__":
    from qfim_store import maybe_rebuild_or_process
    base_path = "/Users/sophieblock/QRCcapstone/parameter_analysis_directory/"
    model_type = "gate"
    N_ctrls = [2,3]
    sample_range = "pi"
    K_str = "1"
    threshold = 1e-10
    by_test = False

    # 1) Possibly we do:
    #    df_all = rebuild_df_from_existing_cache(base_path, N_ctrls, K_str)
    #    if df_all.empty, we call process_and_cache_new_files
    #    or we do the combined approach:
    df_all, df_expanded = maybe_rebuild_or_process(
        base_path, sample_range, model_type, N_ctrls, K_str,
        threshold=threshold, by_test=by_test
    )

    df_sub = isolate_qfim_subset(
        df_expanded,
        N_ctrl=2,
        N_reserv=1,
        trotter_step=10
    )
    print(f"SHAPE OF SPECIFIC FUCKING CONFIGURATION: {df_sub.shape}")
    print(f"KEYS OF SPECIFIC FUCKING CONFIGURATION: {df_sub.keys()}")
    print(f"TEST FUCKING KEYS: {df_sub[''].shape}")

