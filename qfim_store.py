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

def process_data_combined(df, threshold, by_test, Nc, N_R, trot, print_bool):
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


# Build your minimal dictionary
    processed_data = {
        "N_ctrl": Nc,
        "N_reserv": N_R,
        "Trotter_Step": trot,
        "all_qfim_eigvals": qfim_eigval_list,
        "mean_entropy": np.mean(entropies) if entropies else np.nan,
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
                        # "raw_data": raw_df  # optional if you want to keep the raw
                    }
                    processed_files.add(file_id)
                    all_data.append(processed_data)

        # After scanning all directories for this N_ctrl, save the updated cache
        save_cached_data(base_path, cached_data, processed_files, N_ctrl, K_0)

    return cached_data, processed_files, pd.DataFrame(all_data)

###############################################################################
# 4) The function to incorporate newly generated data pickles
###############################################################################
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

            if file_id in cached_data:
                continue  # skip, already in cache

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
                    "raw_data": raw_df
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
    This does NOT require scanning the original QFIM_results directories.
    
    Parameters:
    -----------
    base_path : Path or str
        The base directory where cached pickles are stored.
    N_ctrls : list of int
        List of N_ctrl values you want to load from cache.
    K_0 : str or int
        The K string/index used in naming the cache files, e.g. '1', '2', etc.

    Returns:
    --------
    df_all : pd.DataFrame
        A DataFrame constructed by concatenating all 'processed_data'
        entries from each cached_data for all given N_ctrls.
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
    1) Attempt to rebuild from existing caches for each N_ctrl in N_ctrls.
    2) If a cache is missing or incomplete, process new files for that N_ctrl.
    3) Combine partial DataFrames into one df_all.
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
    df[f"spread_mean_per_sample_{prefix}"] = per_sample_means
    df[f"spread_std_per_sample_{prefix}"]  = per_sample_stds
    df[f"spread_val_pooled_{prefix}"]      = pooled_vals
    return df

def compute_all_stats(eigval_list, threshold=1e-12):
    """
    Single-pass extraction of rank, average variance, etc. 
    Just as in your original 'compute_all_stats' function.
    """
    ranks_per_draw = []
    var_qfim_per_draw = []
    var_qfim_nonzero_per_draw = []
    trace_qfim_per_draw = []
    trace_qfim_norm_by_len = []
    trace_qfim_norm_by_rank = []
    var_qfim_norm_by_len = []
    var_qfim_norm_by_rank = []

    for eigvals in eigval_list:
        arr = np.array(eigvals, dtype=float)
        arr = np.where(arr<threshold, 0.0, arr)
        rank = np.count_nonzero(arr)
        ranks_per_draw.append(rank)
        var_qfim = np.var(arr)
        var_qfim_per_draw.append(var_qfim)
        nonzeros = arr[arr>0]
        var_non = np.var(nonzeros) if len(nonzeros)>1 else 0.0
        var_qfim_nonzero_per_draw.append(var_non)
        trace_val = arr.sum()
        trace_qfim_per_draw.append(trace_val)
        length = len(arr)
        trace_qfim_norm_by_len.append(trace_val/length if length>0 else 0.0)
        var_qfim_norm_by_len.append(var_qfim/length if length>0 else 0.0)

        if rank>0:
            trace_qfim_norm_by_rank.append(trace_val/rank)
            var_qfim_norm_by_rank.append(var_qfim/rank)
        else:
            trace_qfim_norm_by_rank.append(0.0)
            var_qfim_norm_by_rank.append(0.0)

    D_C = max(ranks_per_draw) if ranks_per_draw else 0
    avg_test_var_qfim_eigvals = np.mean(var_qfim_per_draw) if var_qfim_per_draw else 0.0
    avg_test_var_qfim_eigvals_nonzero = np.mean(var_qfim_nonzero_per_draw) if var_qfim_nonzero_per_draw else 0.0
    avg_test_tr_qfim_eigvals = np.mean(trace_qfim_per_draw) if trace_qfim_per_draw else 0.0
    avg_test_tr_qfim_eigvals_norm = np.mean(trace_qfim_norm_by_len) if trace_qfim_norm_by_len else 0.0
    avg_test_tr_qfim_eigvals_norm_by_rank = np.mean(trace_qfim_norm_by_rank) if trace_qfim_norm_by_rank else 0.0
    avg_test_var_qfim_eigvals_normalized = np.mean(var_qfim_norm_by_len) if var_qfim_norm_by_len else 0.0
    avg_test_var_qfim_eigvals_normalized_by_rank = np.mean(var_qfim_norm_by_rank) if var_qfim_norm_by_rank else 0.0
    var_test_var_qfim_eigvals = np.var(var_qfim_per_draw) if len(var_qfim_per_draw)>1 else 0.0
    var_test_var_qfim_eigvals_log = np.log(var_test_var_qfim_eigvals) if var_test_var_qfim_eigvals>0 else 0.0
    var_test_var_qfim_eigvals_nonzero = np.var(var_qfim_nonzero_per_draw) if len(var_qfim_nonzero_per_draw)>1 else 0.0

    return {
        "QFIM_ranks": ranks_per_draw,
        "D_C": D_C,
        "test_var_qfim_eigvals": var_qfim_per_draw,
        "test_var_qfim_eigvals_nonzero": var_qfim_nonzero_per_draw,
        "test_tr_qfim_eigvals": trace_qfim_per_draw,
        "avg_test_var_qfim_eigvals": avg_test_var_qfim_eigvals,
        "avg_test_var_qfim_eigvals_nonzero": avg_test_var_qfim_eigvals_nonzero,
        "avg_test_tr_qfim_eigvals": avg_test_tr_qfim_eigvals,
        "var_test_var_qfim_eigvals": var_test_var_qfim_eigvals,
        "var_test_var_qfim_eigvals_log": var_test_var_qfim_eigvals_log,
        "var_test_var_qfim_eigvals_nonzero": var_test_var_qfim_eigvals_nonzero,
        "avg_test_var_qfim_eigvals_normalized": avg_test_var_qfim_eigvals_normalized,
        "avg_test_var_qfim_eigvals_normalized_by_rank": avg_test_var_qfim_eigvals_normalized_by_rank,
        "test_var_qfim_eigvals_normalized": var_qfim_norm_by_len,
        "test_var_qfim_eigvals_normalized_by_rank": var_qfim_norm_by_rank,
        "avg_test_tr_qfim_eigvals_norm": avg_test_tr_qfim_eigvals_norm,
        "avg_test_tr_qfim_eigvals_norm_by_rank": avg_test_tr_qfim_eigvals_norm_by_rank
    }
def build_qfim_dataframe(df_all, threshold=1e-12):
    """
    1) Convert all_qfim_eigvals -> qfim_eigs_2d
    2) Single-pass stats => expanded columns
    3) Spread-based metrics (variance, mad, median).
    4) Return final df_all with everything included.
    """
    # Convert
    df_all["qfim_eigs_2d"] = df_all["all_qfim_eigvals"].apply(to_2d)

    # Single-pass stats
    stats_series = df_all["all_qfim_eigvals"].apply(lambda x: compute_all_stats(x, threshold=threshold))
    df_stats = pd.json_normalize(stats_series)
    df_all = pd.concat([df_all.reset_index(drop=True), df_stats.reset_index(drop=True)], axis=1)

    # Spread-based columns: variance, mad, median
    df_all = compute_spread_columns(df_all, threshold=threshold, spread_method="variance", scale="normal")
    df_all = compute_spread_columns(df_all, threshold=threshold, spread_method="mad", scale="normal")
    df_all = compute_spread_columns(df_all, threshold=threshold, spread_method="median", scale="normal")

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

    # 2) Build QFIM DataFrame with advanced metrics
    df_all = build_qfim_dataframe(df_all, threshold=1e-12)
    print("[INFO] df_all final shape:", df_all.shape)
    print(df_all.head())