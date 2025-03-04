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

from jax import jit
import pennylane as qml
import time
#from pennylane.pulse import ParametrizedEvolution, ParametrizedHamiltonian,HardwareHamiltonian
from jax.experimental.ode import odeint

has_jax = True
diable_jit = False
config.update('jax_disable_jit', diable_jit)
#config.parse_flags_with_absl()
config.update("jax_enable_x64", True)
os.environ['JAX_TRACEBACK_FILTERING'] = 'off'
# global_cache_data = None
# global_processed_files = None

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
    
def get_keys(df):
    """
    Helper function to extract a minimal representation of the data keys.
    Returns a dict mapping each fixed_param key to the set of test keys.
    """
    return {fixed_param: set(df[fixed_param].keys()) for fixed_param in df}

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
def get_cached_data_once(base_path, N_ctrl, K_0):
    """
    Load cached_data if it exists, else return empty placeholders.
    Returns (cached_data, processed_files).
    
    Expects a cache file named: 'digital_results_QFIM_Nc_{N_ctrl}_{K_0}K.pkl'
    """
    global global_cache_data, global_processed_files
    cache_file = os.path.join(base_path, f'digital_new_QFIM_Nc_{N_ctrl}_{K_0}K.pkl')

    if global_cache_data and global_processed_files:
        print(f"Using cached data from memory for N_ctrl={N_ctrl}.")
        return global_cache_data, global_processed_files

    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            global_cache_data, global_processed_files = pickle.load(f)
        print(f"Loaded cache for N_ctrl={N_ctrl} from disk.")
        return global_cache_data, global_processed_files

    global_cache_data, global_processed_files = {}, set()
    return global_cache_data, global_processed_files

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

def process_data_combined(df, threshold, by_test, Nc, N_R, trot, print_bool=False):
    """
    Here, you do your minimal extraction:
      - qfim_eigvals
      - maybe 'entropy', 'cond_nums'
    Return a dictionary with the essential fields you want to cache.
    """
    # Example logic:
    qfim_eigval_list = []
    entropies = []
    # Suppose 'df' is a dict-of-dicts with structure df[fixed_params][test], etc.
    # You adapt to your real logic:
    key_tuple = []
    for fixed_params_dict in df.keys():
        for test_id in df[fixed_params_dict].keys():
            # If the pickle structure has 'qfim_eigvals' stored in some sub-key:
            qfim_eigvals = df[fixed_params_dict][test_id].get('qfim_eigvals', None)
            if qfim_eigvals is not None:
                qfim_eigval_list.append(qfim_eigvals)

            # Similarly for entropy
            entval = df[fixed_params_dict][test_id].get('entropy', None)
            if entval is not None:
                entropies.append(entval)
            key_tuple.append((fixed_params_dict,test_id))
    keys_snapshot = get_keys(df)
    num_test_keys = sum(len(tests) for tests in keys_snapshot.values())

# Build your minimal dictionary
    processed_data = {
        "N_ctrl": Nc,
        "N_reserv": N_R,
        "Trotter_Step": trot,
        "all_qfim_eigvals": qfim_eigval_list,
        "mean_entropy": np.mean(entropies) if entropies else np.nan,
        "num_test_keys": num_test_keys,
        "key_pair_tuple":key_tuple,
        # etc. if needed
    }
    return processed_data

def process_and_cache_new_files(base_path, K_0, sample_range, model_type, N_ctrls, threshold, by_test, 
                                check_for_new_data, cached_data, processed_files):
    """
    Example function to scan 'QFIM_results' directories, load pickles,
    compute minimal 'processed_data' for each file, store in 'cached_data'.
    Then returns a DataFrame of all processed_data for convenience.
    """
    all_data = []

    for N_ctrl in N_ctrls:
        model_path = Path(base_path) / 'QFIM_results' / model_type / f'Nc_{N_ctrl}' / f'sample_{sample_range}/{K_0}xK'

        for Nr in sorted(os.listdir(model_path)):
            Nr_path = model_path / Nr
            if not Nr_path.is_dir():
                continue

            for trotter_step_dir in sorted(os.listdir(Nr_path)):
                trotter_step_path = Nr_path / trotter_step_dir
                if not trotter_step_path.is_dir():
                    continue

                data_file = trotter_step_path / 'data.pickle'
                file_id = str(data_file)

                # If already in cache, skip or update
                if file_id in cached_data:
                    if check_for_new_data:
                        print(f"Updating cached data.file_id: {file_id}")
                        cached_data, processed_files = update_cached_data(data_file, cached_data, processed_files, N_ctrl, threshold)
                    all_data.append(cached_data[file_id]['processed_data'])
                    continue

                # Otherwise, process
                if is_valid_pickle_file(data_file):
                    print(f"[INFO] Reading new file: {data_file}")
                    raw_df = load_and_clean_pickle(data_file)
                    trotter_step_num = extract_trotter_step(data_file)
                    reservoir_count  = extract_Nr(data_file)

                    processed_data = process_data_combined(
                        raw_df, threshold, by_test, Nc=N_ctrl, N_R=reservoir_count,
                        trot=trotter_step_num, print_bool=False
                    )

                    cached_data[file_id] = {
                        "processed_data": processed_data,
                        "raw_data": raw_df  # optional if you want to keep the raw
                    }
                    processed_files.add(file_id)
                    all_data.append(processed_data)

        # After scanning all directories for this N_ctrl, save the updated cache
        save_cached_data(base_path, cached_data, processed_files, N_ctrl, K_0)

    return cached_data, processed_files, pd.DataFrame(all_data)

###############################################################################
# 4) The function to incorporate newly generated data pickles
###############################################################################
def update_cached_data(data_file, cached_data, processed_files, N_ctrl, threshold):
    file_id = str(data_file)
    df = load_and_clean_pickle(data_file)
    new_keys = get_keys(df)
    old_keys = cached_data[file_id].get("raw_keys", {})

    needs_update = False
    for fixed_param, tests in new_keys.items():
        if fixed_param not in old_keys:
            needs_update = True
            break
        else:
            if tests - old_keys[fixed_param]:
                needs_update = True
                break

    if needs_update:
        print(f"[INFO] New fixed_param/test keys detected in {file_id}. Updating processed data.")
        trotter_step_num = extract_trotter_step(data_file)
        reservoir_count  = extract_Nr(data_file)
        processed_data = process_data_combined(
            df, threshold, by_test=False, Nc=N_ctrl,
            N_R=reservoir_count, trot=trotter_step_num, print_bool=False
        )
        cached_data[file_id]["processed_data"] = processed_data
        cached_data[file_id]["raw_keys"] = new_keys

    processed_files.add(file_id)
    return needs_update  # Return True if updated, else False

def update_cache_with_new_data(
    base_path, 
    K_0, 
    sample_range, 
    model_type, 
    N_ctrl, 
    threshold, 
    by_test,
    cached_data, 
    processed_files
):
    """
    For a single N_ctrl, look for new data.pickle files in the QFIM_results/gate/... directories 
    that are NOT yet in cached_data. Then process them, store them in cached_data, 
    and return an updated DataFrame plus the updated (cached_data, processed_files).
    """
    all_new_data = []
    model_path = Path(base_path) / 'QFIM_results' / model_type / f'Nc_{N_ctrl}' / f'sample_{sample_range}/{K_0}xK'

    if not model_path.exists():
        print(f"[WARN] Directory {model_path} does not exist. No new data found.")
        return cached_data, processed_files, pd.DataFrame()

    for Nr in sorted(os.listdir(model_path)):
        Nr_path = model_path / Nr
        if not Nr_path.is_dir():
            continue

        for trotter_step_dir in sorted(os.listdir(Nr_path)):
            trotter_step_path = Nr_path / trotter_step_dir
            if not trotter_step_path.is_dir():
                continue

            data_file = trotter_step_path / 'data.pickle'
            file_id = str(data_file)

            # if file_id in cached_data:
                
            #     print(f"{file_id} found in cached data. File path: {file_id}")
            #     continue  # skip, already in cache
            if file_id in cached_data:
                print(f"[INFO] Checking cached file: {file_id}")
                updated = update_cached_data(data_file, cached_data, processed_files, N_ctrl, threshold)
                # Only append if it was actually updated
                if updated:
                    all_new_data.append(cached_data[file_id]['processed_data'])
                continue
            

            if is_valid_pickle_file(data_file):
                print(f"[INFO] Found NEW data file: {data_file}")
                raw_df = load_and_clean_pickle(data_file)
                trotter_step_num = extract_trotter_step(data_file)
                reservoir_count  = extract_Nr(data_file)

                processed_data = process_data_combined(
                    raw_df, threshold, by_test,
                    Nc=N_ctrl, N_R=reservoir_count, trot=trotter_step_num
                )
                cached_data[file_id] = {
                    "processed_data": processed_data,
                    "raw_keys": get_keys(raw_df)
                }
                processed_files.add(file_id)
                all_new_data.append(processed_data)


    if all_new_data:
        save_cached_data(base_path, cached_data, processed_files, N_ctrl, K_0)
    return cached_data, processed_files, pd.DataFrame(all_new_data)
# ----------------------------------------------------------------------------
# Rebuild df_all from existing cached_data 
# ----------------------------------------------------------------------------
def rebuild_df_from_existing_cache(base_path, N_ctrls, K_0):
    """
    Rebuild a single DataFrame from the cached_data dictionaries saved on disk.
    This function does NOT scan the original QFIM_results directories, but instead
    loads preprocessed cached data. Each row in the returned DataFrame corresponds
    to one experimental run for a given configuration (i.e. a specific combination of
    control qubits (N_ctrl), reservoir qubits (N_R), and Trotter step (T)). 
    
    The cached data includes a column (typically named 'all_qfim_eigvals') which stores
    the raw QFIM eigenvalue arrays. Each element of this column is the result of
    computing the QFIM for one set of randomly sampled trainable parameters for that
    configuration.
    
    Parameters
    ----------
    base_path : Path or str
        The base directory where cached pickle files are stored.
    N_ctrls : list of int
        List of control qubit counts (N_ctrl values) for which cached data is to be loaded.
    K_0 : str or int
        The K parameter string/index used in naming the cache files (e.g. '1', '2', etc.).
        
    Returns
    -------
    df_all : pd.DataFrame
        A DataFrame constructed by concatenating all 'processed_data' entries from
        each cached_data file for the specified N_ctrls. Each row represents one experiment run,
        with the column 'all_qfim_eigvals' holding the raw QFIM eigenvalue arrays computed from
        a single random sample of trainable parameters.
    """
    all_data = []

    for N_ctrl in N_ctrls:
        cdata, pfiles = get_cached_data(base_path, N_ctrl, K_0)
        for file_id, file_info in cdata.items():
            pdata = file_info.get('processed_data', {})
            all_data.append(pdata)

    df_all = pd.DataFrame(all_data)
    return df_all

# ----------------------------------------------------------------------------
# Possibly unify reading from cache or processing new files
# ----------------------------------------------------------------------------
def maybe_rebuild_or_process(base_path, sample_range, model_type, N_ctrls, K_0, threshold=1e-10, by_test=False):
    """
    For each specified control qubit value (N_ctrl), this function:
      1) Attempts to rebuild the DataFrame from existing cached_data,
      2) If the cache is missing or incomplete, processes new QFIM_results files to generate data,
      3) Combines the resulting partial DataFrames into one df_all.
    
    Each row in df_all corresponds to a single experiment run for a given configuration
    (i.e. a unique combination of N_ctrl, reservoir qubits, and Trotter step). In these rows,
    the column 'all_qfim_eigvals' contains the raw QFIM eigenvalue arrays computed from one sample
    of random trainable parameters.
    
    Parameters
    ----------
    base_path : Path or str
        The base directory where the cache files and/or raw QFIM_results are stored.
    sample_range : str
        The sample range label used in your file/directory naming (e.g. "pi").
    model_type : str
        The model type (e.g. "gate").
    N_ctrls : list of int
        List of control qubit counts to process.
    K_0 : str or int
        The K parameter string/index (e.g. "1").
    threshold : float, optional
        The threshold value used when processing QFIM data.
    by_test : bool, optional
        If True, process data by individual test key; otherwise, use the standard aggregation.
        
    Returns
    -------
    df_all : pd.DataFrame
        A DataFrame constructed by combining cached (or newly processed) data for each N_ctrl.
        Each row represents one experiment run (i.e. one random sample of trainable parameters) for
        the configuration, and the column 'all_qfim_eigvals' stores the corresponding raw QFIM eigenvalue arrays.
    """
    combined_frames = []
    for N_ctrl in N_ctrls:
        cached_data, processed_files = get_cached_data(base_path, N_ctrl, K_0)
        if not cached_data:
            # If no cache, let's process from scratch
            print(f"[INFO] No existing cache for N_ctrl={N_ctrl}, scanning directories.")
            cached_data, processed_files, df_partial = process_and_cache_new_files(
                base_path, K_0, sample_range, model_type, [N_ctrl],
                threshold, by_test, check_for_new_data=False,
                cached_data=cached_data, processed_files=processed_files
            )
            combined_frames.append(df_partial)
        else:
            # 3) We do have a cache; let's see if there's new data not in the cache
            cached_data, processed_files, df_new = update_cache_with_new_data(
                base_path=base_path,
                K_0=K_0,
                sample_range=sample_range,
                model_type=model_type,
                N_ctrl=N_ctrl,
                threshold=threshold,
                by_test=by_test,
                cached_data=cached_data,
                processed_files=processed_files
            )
            # df_new is newly added data. We also want the old data
            local_rows = []
            for fid, fdict in cached_data.items():
                local_rows.append(fdict["processed_data"])
            df_partial = pd.DataFrame(local_rows)
            combined_frames.append(df_partial)

    if combined_frames:
        df_all = pd.concat(combined_frames, ignore_index=True)
    else:
        df_all = pd.DataFrame()
    return df_all

###############################################################################
#  "Build QFIM DataFrame" pipeline (the code from your question)
###############################################################################
from scipy.stats import median_abs_deviation
def to_2d(evals_list):
    """
    Convert list of 1D arrays => single 2D array: shape (n_draws, d).
    """
    return np.vstack(evals_list)

def compute_spread_metric(values, method="variance", ddof=1, scale="normal"):
    """
    1D array 'values' => compute variance, mad, or median.
    """
    if len(values) <= 1:
        return 0.0
    if method == "variance":
        return np.var(values, ddof=ddof)
    elif method == "mad":
        return median_abs_deviation(values, scale=scale)
    elif method == "median":
        return np.median(values)
    else:
        raise ValueError(f"Unknown method: {method}")

def spread_per_sample_vectorized(eigs_2d, method="variance", threshold=1e-12, ddof=1, scale="normal"):
    clipped = np.where(eigs_2d>threshold, eigs_2d, 0.0)
    row_sums = clipped.sum(axis=1, keepdims=True)
    nonzero_mask = (row_sums[:,0]>0)
    clipped[nonzero_mask] = clipped[nonzero_mask]/row_sums[nonzero_mask]
    with np.errstate(divide='ignore'):
        logs = np.log(clipped, out=np.zeros_like(clipped), where=(clipped>0))

    n_draws, d = eigs_2d.shape
    results = np.zeros(n_draws)
    for i in range(n_draws):
        row_nonzero = logs[i, clipped[i]>0]
        results[i] = compute_spread_metric(row_nonzero, method=method, ddof=ddof, scale=scale)
    return results

def spread_pooling_vectorized(eigs_2d, method="variance", threshold=1e-12, ddof=1, scale="normal"):
    flat = eigs_2d.ravel()
    filtered = flat[flat>threshold]
    if filtered.size <= 1:
        return 0.0
    s = filtered.sum()
    filtered /= s
    with np.errstate(divide='ignore'):
        logs = np.log(filtered)
    return compute_spread_metric(logs, method=method, ddof=ddof, scale=scale)

def compute_spread_columns(df, eigs_2d_col="qfim_eigs_2d", threshold=1e-12, ddof=1, scale="normal", spread_method="variance"):
    arr_2d_list = df[eigs_2d_col].values
    per_sample_means = []
    per_sample_stds  = []
    pooled_vals      = []

    for arr2d in arr_2d_list:
        per_draw = spread_per_sample_vectorized(arr2d, 
                                                method=spread_method, 
                                                threshold=threshold, 
                                                ddof=ddof, 
                                                scale=scale)
        per_sample_means.append(per_draw.mean() if per_draw.size else 0.0)
        per_sample_stds.append(per_draw.std() if per_draw.size>1 else 0.0)

        pool_val = spread_pooling_vectorized(arr2d, 
                                             method=spread_method, 
                                             threshold=threshold, 
                                             ddof=ddof, 
                                             scale=scale)
        pooled_vals.append(pool_val)

    prefix = spread_method.lower()
    if prefix == "mad":
        prefix += f'_{scale}'
    df[f"spread_mean_per_sample_{prefix}"] = per_sample_means
    df[f"spread_std_per_sample_{prefix}"]  = per_sample_stds
    df[f"spread_val_pooled_{prefix}"]      = pooled_vals
    return df

def compute_all_stats(
    eigval_list,
    threshold=1e-12,
    spread_methods=("variance", "mad"),  # e.g. ["variance", "mad"]
    ddof=1,
    scale="normal",
    # Additional args for the approximate effective dimension
    do_effective_dim=True,
 
):
    """
    Compute QFIM statistics for a list of draws (eigval_list) corresponding to one experimental run.
    
    Here, each element of eigval_list represents the raw QFIM eigenvalues computed
    from a single draw (i.e. one sample of randomly generated trainable parameters) for a 
    given configuration (specified by a unique combination of N_ctrl, reservoir qubits (N_R),
    and Trotter step (T)). 
    
    For each draw, the following metrics are computed:
      - rank: Number of nonzero eigenvalues after applying the threshold.
      - var_qfim_eigvals: Variance computed on all eigenvalues.
      - var_qfim_eigvals_nonzero: Variance computed on nonzero eigenvalues only.
      - trace: Sum of eigenvalues (i.e. the trace of the QFIM).
      - var_norm_len: Variance normalized by the total number of eigenvalues.
      - trace_norm_len: Trace normalized by the total number of eigenvalues.
      - var_norm_rank: Variance normalized by the rank (number of nonzero eigenvalues).
      - trace_norm_rank: Trace normalized by the rank.
      - Spread metrics: For each method in spread_methods (e.g. "variance", "mad"), compute the spread of the log of the eigenvalues.
      - effective_dimension: TODO
    
    In addition, pooled metrics are computed across all draws:
      - Average and variance for each per-draw metric.
      - Pooled spread-of-log metrics.
      
    Returns
    -------
    metrics : dict
        A dictionary with keys that follow your naming convention. For example:
          - "QFIM_ranks": List of ranks for each draw.
          - "test_var_qfim_eigvals": List of variances (all eigenvalues) for each draw.
          - "test_tr_qfim_eigvals": List of traces for each draw.
          - "avg_test_var_qfim_eigvals": Average variance across draws.
          - "avg_test_tr_qfim_eigvals": Average trace across draws.
          - "effective_dimension": TODO
          - Spread-of-log metrics for each method (e.g. "spread_mean_per_sample_variance_normal", etc.).
    
    Additional Notes:
    -----------------
    This function assumes that eigval_list is a list of 1D arrays (or lists),
    where each array corresponds to the QFIM eigenvalues computed for a single
    random draw of trainable parameters. The computed metrics are intended to capture
    both per-draw variability and the overall (pooled) behavior of the QFIM eigenvalue distribution,
    which are later used to correlate with measures of learnability and generalization.
    """

    import numpy as np

    # 1) Basic per-draw computations
    ranks_per_draw = []
    var_all_per_draw = []
    var_nonzero_per_draw = []
    trace_per_draw = []
    var_norm_len_per_draw = []
    var_norm_rank_per_draw = []
    trace_norm_len_per_draw = []
    trace_norm_rank_per_draw = []

    for eigs in eigval_list:
        arr = np.array(eigs, dtype=float)
        arr = np.where(arr < threshold, 0.0, arr)
        rank = np.count_nonzero(arr)
        ranks_per_draw.append(rank)

        var_all = np.var(arr)
        var_all_per_draw.append(var_all)

        nonz = arr[arr > 0]
        var_non = np.var(nonz) if nonz.size > 1 else 0.0
        var_nonzero_per_draw.append(var_non)

        trace_val = arr.sum()
        trace_per_draw.append(trace_val)

        length = len(arr) if len(arr) else 1
        var_norm_len_per_draw.append(var_all / length)
        trace_norm_len_per_draw.append(trace_val / length)

        if rank > 0:
            var_norm_rank_per_draw.append(var_all / rank)
            trace_norm_rank_per_draw.append(trace_val / rank)
        else:
            var_norm_rank_per_draw.append(0.0)
            trace_norm_rank_per_draw.append(0.0)

    # 2) Aggregate across draws
    D_C = max(ranks_per_draw) if ranks_per_draw else 0
    avg_var_all = np.mean(var_all_per_draw) if var_all_per_draw else 0.0
    avg_var_nonzero = np.mean(var_nonzero_per_draw) if var_nonzero_per_draw else 0.0
    avg_trace = np.mean(trace_per_draw) if trace_per_draw else 0.0
    avg_trace_len = np.mean(trace_norm_len_per_draw) if trace_norm_len_per_draw else 0.0
    avg_trace_rank = np.mean(trace_norm_rank_per_draw) if trace_norm_rank_per_draw else 0.0
    avg_var_norm_len = np.mean(var_norm_len_per_draw) if var_norm_len_per_draw else 0.0
    avg_var_norm_rank = np.mean(var_norm_rank_per_draw) if var_norm_rank_per_draw else 0.0

    var_var_all = np.var(var_all_per_draw) if len(var_all_per_draw) > 1 else 0.0
    var_var_nonzero = np.var(var_nonzero_per_draw) if len(var_nonzero_per_draw) > 1 else 0.0

   
   
   
    # TODO: Effective Dimension from Eq. (32) in the paper (captures concentration of QFIM spectrum)

   

    # Spread-of-log metrics for each method in spread_methods
    #    We'll build arr_2d to use your existing vectorized functions.
    arr_2d = np.zeros((len(eigval_list), max(len(x) for x in eigval_list))) if eigval_list else np.zeros((0,0))
    for i, e in enumerate(eigval_list):
        tmp = np.array(e, dtype=float)
        tmp = np.where(tmp < threshold, 0.0, tmp)
        arr_2d[i, :len(tmp)] = tmp

    spread_results = {}
    from functools import partial


    for method in spread_methods:
        per_draw = spread_per_sample_vectorized(
            arr_2d, method=method, threshold=threshold, ddof=ddof, scale=scale
        )
        spread_mean = per_draw.mean() if per_draw.size else 0.0
        spread_std  = per_draw.std()  if per_draw.size > 1 else 0.0

        pooled_val = spread_pooling_vectorized(
            arr_2d, method=method, threshold=threshold, ddof=ddof, scale=scale
        )

        # Store them with keys that include the scale
        prefix = method.lower()
        # e.g. "spread_mean_per_sample_mad_normal"
        spread_results[f"spread_mean_per_sample_{prefix}_{scale}"] = spread_mean
        spread_results[f"spread_std_per_sample_{prefix}_{scale}"]  = spread_std
        spread_results[f"spread_val_pooled_{prefix}_{scale}"]      = pooled_val

    # 5) Final dictionary
    metrics = {
        # Per-draw lists
        "QFIM_ranks": ranks_per_draw,
        "test_var_qfim_eigvals": var_all_per_draw,
        "test_var_qfim_eigvals_nonzero": var_nonzero_per_draw,
        "test_tr_qfim_eigvals": trace_per_draw,
        "test_var_qfim_eigvals_normalized": var_norm_len_per_draw,
        "test_var_qfim_eigvals_normalized_by_rank": var_norm_rank_per_draw,

        # Aggregated
        "D_C": D_C,
        "avg_test_var_qfim_eigvals": avg_var_all,
        "avg_test_var_qfim_eigvals_nonzero": avg_var_nonzero,
        "avg_test_tr_qfim_eigvals": avg_trace,
        "avg_test_tr_qfim_eigvals_norm": avg_trace_len,
        "avg_test_tr_qfim_eigvals_norm_by_rank": avg_trace_rank,
        "avg_test_var_qfim_eigvals_normalized": avg_var_norm_len,
        "avg_test_var_qfim_eigvals_normalized_by_rank": avg_var_norm_rank,
        "var_test_var_qfim_eigvals": var_var_all,
        "var_test_var_qfim_eigvals_nonzero": var_var_nonzero,

     
    }

    # Merge in the spread-of-log results with the new naming
    metrics.update(spread_results)

    return metrics
def build_qfim_dataframe(df_all, threshold=1e-12):
    """
    1) Convert all_qfim_eigvals -> qfim_eigs_2d
    2) Single-pass stats => expanded columns (including spread-of-log)
    3) Optionally, extra 'compute_spread_columns' calls 
       if you want separate columns for 'median', etc.
    4) Return final df_all with everything included.
    """
    # 1) Convert for convenience
    df_all["qfim_eigs_2d"] = df_all["all_qfim_eigvals"].apply(to_2d)

    # 2) Single-pass stats
    stats_series = df_all["all_qfim_eigvals"].apply(
        lambda x: compute_all_stats(
            x, 
            threshold=threshold, 
            spread_methods=["variance", "mad"], # you can add 'median' if you like
            ddof=1, 
            scale="normal"
        )
    )
    df_stats = pd.json_normalize(stats_series)
    df_out = pd.concat([df_all.reset_index(drop=True), df_stats.reset_index(drop=True)], axis=1)

    # 3) If you still want “extra” spread columns for e.g. 'median' or other transformations,
    #    you can also call your existing compute_spread_columns(...) here:
    # df_out = compute_spread_columns(df_out, threshold=threshold, spread_method="median", scale="normal")
    # etc.

    
    # Spread-based columns: variance, mad, median
    # df_all = compute_spread_columns(df_all, threshold=threshold, spread_method="variance", scale="normal")
    # df_all = compute_spread_columns(df_all, threshold=threshold, spread_method="mad", scale="normal")
    # df_all = compute_spread_columns(df_all, threshold=threshold, spread_method="median", scale="normal")

    return df_all


if __name__ == "__main__":
    base_path = "/Users/sophieblock/QRCcapstone/parameter_analysis_directory/"
    model_type = "gate"
    N_ctrls = [2, 3]
    sample_range = "pi"
    K_str = "1"
    threshold = 1e-10
    by_test = False

    # 1) Possibly we do:
    #    df_all = rebuild_df_from_existing_cache(base_path, N_ctrls, K_str)
    #    if df_all.empty, we call process_and_cache_new_files
    #    or we do the combined approach:
    df_all = maybe_rebuild_or_process(
        base_path, sample_range, model_type, N_ctrls, K_str,
        threshold=threshold, by_test=by_test
    )

    print("[INFO] df_all shape after reading cache or scanning directories:", df_all.shape)

    # # 2) Build QFIM DataFrame with advanced metrics
    df_all = build_qfim_dataframe(df_all, threshold=1e-12)
    print("[INFO] df_all final shape:", df_all.shape)
    print(df_all.head())