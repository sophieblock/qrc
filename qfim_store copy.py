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

def process_data_combined(df, threshold, by_test, Nc, N_R, trot, print_bool = False):
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
def process_and_cache_new_files(base_path, K_0, sample_range, model_type, N_ctrls, threshold, by_test, 
                                check_for_new_data, cached_data, processed_files):
    """
    Example function to scan 'QFIM_results' directories, load pickles,
    compute minimal 'processed_data' for each file, store in 'cached_data'.
    Then returns a DataFrame of all processed_data for convenience.
    """
    all_data_condensed = []  # (OLD) aggregated
    all_data_expanded  = []  # (NEW) one row per (fixed_params_key, test_key)


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

                # If already in cache, just append old + new data from cache
                if file_id in cached_data:
                    old_condensed = cached_data[file_id].get('processed_data', {})
                    old_expanded  = cached_data[file_id].get('expanded_rows', [])
                    if old_condensed:
                        all_data_condensed.append(old_condensed)
                    if old_expanded:
                        all_data_expanded.extend(old_expanded)
                    continue

                # Otherwise, load and process
                if is_valid_pickle_file(data_file):
                    print(f"[INFO] Reading new file: {data_file}")
                    raw_df = load_and_clean_pickle(data_file)
                    trotter_step_num = extract_trotter_step(data_file)
                    reservoir_count  = extract_Nr(data_file)

                    # (OLD) aggregated approach
                    processed_data_condensed = process_data_combined(
                        raw_df, threshold, by_test, 
                        Nc=N_ctrl, N_R=reservoir_count,
                        trot=trotter_step_num, 
                        print_bool=False
                    )

                    # (NEW) expanded approach
                    processed_rows_expanded = process_data_expanded(
                        raw_df, threshold, by_test, 
                        Nc=N_ctrl, N_R=reservoir_count,
                        trot=trotter_step_num,
                        print_bool=False
                    )

                    # Save to cache
                    cached_data[file_id] = {
                        "processed_data": processed_data_condensed,
                        "expanded_rows": processed_rows_expanded
                    }
                    processed_files.add(file_id)

                    # Append to local lists
                    all_data_condensed.append(processed_data_condensed)
                    all_data_expanded.extend(processed_rows_expanded)

        # After scanning all directories for this N_ctrl, save the updated cache
        save_cached_data(base_path, cached_data, processed_files, N_ctrl, K_0)

    # Build DataFrames
    df_condensed = pd.DataFrame(all_data_condensed)
    df_expanded  = pd.DataFrame(all_data_expanded)
    return cached_data, processed_files, df_condensed, df_expanded

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

                # Compute both outputs
                processed_data_combined = process_data_combined(
                    raw_df, threshold, by_test,
                    Nc=N_ctrl, N_R=reservoir_count, trot=trotter_step_num
                )
                processed_rows_expanded = process_data_expanded(
                    raw_df, threshold, by_test,
                    Nc=N_ctrl, N_R=reservoir_count, trot=trotter_step_num
                )
                # Save both to the cache:
                cached_data[file_id] = {
                    "processed_data": processed_data_combined,
                    "expanded_rows": processed_rows_expanded,
                    "raw_data": raw_df
                }
                processed_files.add(file_id)
                all_new_data.append(processed_data_combined)

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
    combined_frames_condensed = []
    combined_frames_expanded  = []

    for N_ctrl in N_ctrls:
        cached_data, processed_files = get_cached_data(base_path, N_ctrl, K_0)
        if not cached_data:
            # No cache => process from scratch
            print(f"[INFO] No existing cache for N_ctrl={N_ctrl}, scanning directories.")
            cached_data, processed_files, df_c, df_e = process_and_cache_new_files(
                base_path, K_0, sample_range, model_type, [N_ctrl],
                threshold, by_test, check_for_new_data=False,
                cached_data=cached_data, processed_files=processed_files
            )
            combined_frames_condensed.append(df_c)
            combined_frames_expanded.append(df_e)
        else:
            # We do have a cache => check for new data
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
            # df_new is newly added "condensed" data from the old approach.
            # For the new approach, you'd do a second call or unify logic similarly.
            # Or you can re-construct the expanded DF from the cache:
            local_condensed = []
            local_expanded  = []
            for fid, fdict in cached_data.items():
                cond = fdict.get("processed_data", {})
                exp  = fdict.get("expanded_rows", [])
                if cond:
                    local_condensed.append(cond)
                if exp:
                    local_expanded.extend(exp)
            df_c = pd.DataFrame(local_condensed)
            df_e = pd.DataFrame(local_expanded)
            combined_frames_condensed.append(df_c)
            combined_frames_expanded.append(df_e)

    # Merge everything
    df_condensed = pd.concat(combined_frames_condensed, ignore_index=True) if combined_frames_condensed else pd.DataFrame()
    df_expanded  = pd.concat(combined_frames_expanded,  ignore_index=True) if combined_frames_expanded  else pd.DataFrame()

    return df_condensed, df_expanded

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
    df_all, df_expanded = maybe_rebuild_or_process(
        base_path, sample_range, model_type, N_ctrls, K_str,
        threshold=threshold, by_test=by_test
    )

# if __name__ == "__main__":
#     base_path = "/Users/sophieblock/QRCcapstone/parameter_analysis_directory/"
#     model_type = "gate"
#     N_ctrls = [2, 3]
#     sample_range = "pi"
#     K_str = "1"
#     threshold = 1e-10
#     by_test = False

#     # 1) Possibly we do:
#     #    df_all = rebuild_df_from_existing_cache(base_path, N_ctrls, K_str)
#     #    if df_all.empty, we call process_and_cache_new_files
#     #    or we do the combined approach:
#     df_all = maybe_rebuild_or_process(
#         base_path, sample_range, model_type, N_ctrls, K_str,
#         threshold=threshold, by_test=by_test
#     )

#     print("[INFO] df_all shape after reading cache or scanning directories:", df_all.shape)
