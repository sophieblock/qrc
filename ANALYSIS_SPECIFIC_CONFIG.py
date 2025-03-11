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


def process_data_dqfim(df, threshold, by_test, Nc, N_R, trot, print_bool=False):
    """
    Create a list of dictionaries, each representing one (fixed_params_key, test_key) row.
    Returns a list of row-dicts for direct DataFrame construction.
    """
    rows = []
    for fixed_params_key, test_dict in df.items():
        for test_key, results in test_dict.items():
            qfim_eigvals = results.get('qfim_eigvals', None)
            qfim_mat = results.get('qfim',None)
            entropy      = results.get('entropy', None)
            row = {
                "N_ctrl": Nc,
                "N_reserv": N_R,
                "Trotter_Step": trot,
                "fixed_params_key": fixed_params_key,
                "test_key": test_key,
                "qfim_eigvals": qfim_eigvals,
                "qfim_mat":qfim_mat,
                "entropy": entropy,
                # Add any other fields you might need later
            }
            rows.append(row)
    return rows
def build_df_expanded_DQFIM(base_path, sample_range, model_type, N_ctrls, K_str,datasize, threshold, by_test):
    """
   
    """
    from pathlib import Path
    import os
    import pandas as pd

    all_expanded_rows = []
    
    # Iterate over each N_ctrl value
    for N_ctrl in N_ctrls:
        model_path = Path(base_path) / "QFIM_global_results" / f"{model_type}_model_DQFIM" / f"Nc_{N_ctrl}" / f"sample_{sample_range}/{K_str}xK"
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
                
                data_file = trotter_step_path / f"L_{datasize}/data.pickle"
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
                expanded_rows = process_data_dqfim(raw_data, threshold, by_test, N_ctrl, reservoir_count, trotter_step_num)
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

def compute_single_draw_stats(
    eigvals,
    threshold=1e-12,
    spread_methods=("variance", "mad"),
    ddof=1,
    scale="normal",
    # Approximate Abbas dimension parameters
    # We'll expose them as function arguments so you can override defaults:
    gamma=0.1,
    n=1,
    V_theta=1.0,
):
    """
    Compute QFIM statistics for a SINGLE set of eigenvalues (one 'draw').

    'n' is interpreted as the 'number of data samples' in the
    Abbas local dimension approach, i.e. alpha = gamma*n/(2 log(n)) if n>1.

    If you truly want to treat this single draw as 'just 1 data sample',
    you can set n=1. If you prefer to interpret the entire set of
    random draws as your 'sample size', you could pass that bigger number
    here. It's up to how you interpret the local dimension formula.

    Similarly, 'gamma' in (0,1] is a scaling constant in Abbas's formula,
    and 'V_theta' is the parameter-space volume factor (assumed = 1 if unknown).
    
    Categories of Metrics:
      1. ABSOLUTE SCALE: e.g., raw variance, raw trace.
      2. SHAPE of the spectrum (distribution only, ignoring total magnitude):
         e.g., normalized IPR or normalized Abbas dimension.
      3. AVERAGE PER NONZERO MODE: e.g., var / rank, trace / rank.

    Parameters
    ----------
    eigvals : list or np.ndarray
        QFIM eigenvalues for this single draw.
    threshold : float
        Threshold for zeroing out tiny eigenvalues.
    spread_methods : tuple of str
        E.g. ("variance", "mad") => compute these spread-of-log metrics on the 
        normalized (per-draw) eigenvalue distribution (like the multi-draw function).
    ddof : int
        Delta degrees of freedom for variance-based computations.
    scale : str
        Scale for 'median_abs_deviation'.
    vol_param_space : float
        Not used here but kept to match your signature if you want the option to 
        subtract log(vol_param_space) for local definitions.
    gamma : float
        Scaling factor in Abbas dimension (typ. 0 < gamma <= 1).
    n : int
        Interpreted as # data samples for the Abbas dimension formula.
    V_theta : float
        Local parameter-space volume for Abbas dimension (set 1.0 if unknown).

    Returns
    -------
    stats_dict : dict
        With keys that are grouped by the three categories above, plus 
        rank-based items and spread-of-log metrics. For example:

        {
          # Basic rank and raw measurements:
          "draw_rank": ...,
          "absolute_scale_var_all": ...,
          "absolute_scale_var_nonzero": ...,
          "absolute_scale_trace": ...,

          # Average per nonzero mode:
          "avg_per_active_mode_var_rank": ...,
          "avg_per_active_mode_trace_rank": ...,

          # Spectrum shape (normalized metrics):
          "spectrum_shape_ipr_norm": ...,
          "spectrum_shape_abbas_norm": ...,

          # Raw IPR & Abbas (still an 'absolute scale' dimension):
          "ipr_deff_raw": ...,
          "abbas_deff_raw": ...,

          # Spread-of-log:
          "spread_metric_variance": ...,
          "spread_metric_mad": ...,
          ...
        }
    """
    import numpy as np

    # 1) Convert input to array, threshold small eigenvalues
    if isinstance(eigvals, list):
        arr = np.array(eigvals, dtype=float)
    elif isinstance(eigvals, np.ndarray):
        arr = eigvals.copy()
    else:
        arr = np.array(eigvals, dtype=float)

    arr = np.where(arr < threshold, 0.0, arr)

    # 2) Basic stats
    draw_rank = np.count_nonzero(arr)
    var_all_eigenvalues = np.var(arr)  # absolute scale
    nonzero = arr[arr > 0]
    var_nonzero_eigenvalues = np.var(nonzero) if nonzero.size > 1 else 0.0
    trace_eigenvalues = arr.sum()

    # 3) Average per nonzero mode => rank-based
    if draw_rank > 0:
        var_normalized_by_rank = var_all_eigenvalues / draw_rank
        trace_normalized_by_rank = trace_eigenvalues / draw_rank
    else:
        var_normalized_by_rank = 0.0
        trace_normalized_by_rank = 0.0

    # 4) IPR-based dimension
    sum_eigs_sq = np.sum(arr**2)
    if sum_eigs_sq > 0.0:
        ipr_deff_raw = (trace_eigenvalues**2) / sum_eigs_sq
    else:
        ipr_deff_raw = 0.0

    # normalized => shape of spectrum
    if trace_eigenvalues > 0.0:
        arr_norm = arr / trace_eigenvalues
        sum_norm_sq = np.sum(arr_norm**2)
        ipr_deff_norm = 1.0 / sum_norm_sq if sum_norm_sq > 0 else 0.0
    else:
        arr_norm = None
        ipr_deff_norm = 0.0

    # 5) Abbas-based dimension
    # alpha = gamma*n / (2 * log(n)) if n>1
    if (n > 1) and (np.log(n) != 0.0):
        alpha = (gamma * n) / (2.0 * np.log(n))
    else:
        alpha = 0.0

    # (A) raw => absolute scale
    abbas_deff_raw = 0.0
    for lam in arr:
        val = 1.0 + alpha * lam
        if val <= 0.0:
            val = 1e-15
        abbas_deff_raw += np.log(val)
    # Subtract log(V_theta) if you have a real param volume
    # if V_theta != 1.0:
    #     abbas_deff_raw -= np.log(V_theta)

    # (B) normalized => shape
    abbas_deff_norm = 0.0
    if arr_norm is not None:
        for lam_norm in arr_norm:
            val = 1.0 + alpha * lam_norm
            if val <= 0.0:
                val = 1e-15
            abbas_deff_norm += np.log(val)
        # if V_theta != 1.0:
        #     abbas_deff_norm -= np.log(V_theta)

    # 6) Spread-of-log => shape
    arr_2d = arr.reshape((1, -1))
    
    spread_metrics = {}
    for method in spread_methods:
        per_draw = spread_per_sample_vectorized(
            arr_2d, method=method, threshold=threshold, ddof=ddof, scale=scale
        )
        val = per_draw[0] if per_draw.size > 0 else 0.0
        spread_metrics[f"spread_metric_{method}"] = val

    # 7) Build final dictionary
    stats_dict = {
        # Basic rank & stats
        "draw_rank": draw_rank,

        # -- Absolute scale
        "absolute_scale_var_all": var_all_eigenvalues,
        "absolute_scale_var_nonzero": var_nonzero_eigenvalues,
        "absolute_scale_trace": trace_eigenvalues,
        "ipr_deff_raw": ipr_deff_raw,
        "abbas_deff_raw": abbas_deff_raw,

        # -- Shape
        "spectrum_shape_ipr_norm": ipr_deff_norm,
        "spectrum_shape_abbas_norm": abbas_deff_norm,
        **spread_metrics,

        # -- Average per active mode
        "avg_per_active_mode_var_rank": var_normalized_by_rank,
        "avg_per_active_mode_trace_rank": trace_normalized_by_rank,
    }
    return stats_dict

def isolate_qfim_subset(df_expanded, N_ctrl=None, fixed_params_key=None, N_reserv=None, trotter_step=None):
    """
    Return a subset of df_expanded matching the provided filters.
    """
    subset = df_expanded.copy()
    if N_ctrl is not None:
        subset = subset[subset["N_ctrl"] == N_ctrl]
    if fixed_params_key is not None:
        subset = subset[subset["fixed_params_key"] == fixed_params_key]
    if N_reserv is not None:
        subset = subset[subset["N_reserv"] == N_reserv]
    if trotter_step is not None:
        subset = subset[subset["Trotter_Step"] == trotter_step]
    return subset

if __name__ == "__main__":
    base_path = "/Users/sophieblock/QRCcapstone/parameter_analysis_directory/"
    model_type = "gate"
    N_ctrls = [2,3]
    sample_range = "pi"
    K_str = "1"
    threshold = 1e-10
    by_test = False
    # Build df_expanded directly from the raw files; this completely ignores any cache.
    df_expanded = build_df_expanded_from_raw(base_path, sample_range, model_type, N_ctrls, K_str, threshold, by_test)
    print("Built df_expanded from raw data. Shape:", df_expanded.shape)

    df_sub = isolate_qfim_subset(
        df_expanded,
        N_ctrl=2,
        N_reserv=1,
        trotter_step=10
    )
