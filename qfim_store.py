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

from jax import jit
import pennylane as qml
import time
#from pennylane.pulse import ParametrizedEvolution, ParametrizedHamiltonian,HardwareHamiltonian
from jax.experimental.ode import odeint
"""
original compute_all_stats(..) keys:

            # Aggregated
        "D_C": D_C,  # rank-based dimension
        "avg_test_var_qfim_eigvals": avg_var_all,
        "avg_test_var_qfim_eigvals_nonzero": avg_var_nonzero,
        "avg_test_tr_qfim_eigvals": avg_trace,
        "avg_test_var_qfim_eigvals_normalized_by_rank": avg_var_norm_rank,
        "avg_test_tr_qfim_eigvals_norm_by_rank": avg_trace_norm_rank,
        "var_test_var_qfim_eigvals": var_var_all,
        "var_test_var_qfim_eigvals_nonzero": var_var_nonzero,

        # --- NEW: IPR-based dimension (raw + normalized) and Abbas dimension (raw + normalized) ---
        "ipr_deffs_raw": ipr_deffs_raw,
        "ipr_deffs_norm": ipr_deffs_norm,
        "abbas_deffs_raw": abbas_deffs_raw,
        "abbas_deffs_norm": abbas_deffs_norm,

        "avg_ipr_deffs_raw": avg_ipr_raw,
        "avg_ipr_deffs_norm": avg_ipr_norm,
        "avg_abbas_deffs_raw": avg_abbas_raw,
        "avg_abbas_deffs_norm": avg_abbas_norm,
    }



"""
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

    qfim_eigval_list = []
    qfim_mats_list = []
    entropies = []
    #  'df' is a dict-of-dicts with structure df[fixed_params_key][test_key], etc.
    
    key_tuple = []
    for fixed_params_key in df.keys():
        for test_key in df[fixed_params_key].keys():
            # If the pickle structure has 'qfim_eigvals' stored in some sub-key:
            qfim_eigvals = df[fixed_params_key][test_key].get('qfim_eigvals', None)
            qfim_mat = df[fixed_params_key][test_key].get('qfim',None)
            if qfim_eigvals is not None:
                qfim_eigval_list.append(qfim_eigvals)
                qfim_mats_list.append(qfim_mat)

            # Similarly for entropy
            entval = df[fixed_params_key][test_key].get('entropy', None)
            if entval is not None:
                entropies.append(entval)
            key_tuple.append((fixed_params_key,test_key))
    keys_snapshot = get_keys(df)
    num_test_keys = sum(len(tests) for tests in keys_snapshot.values())

# Build your minimal dictionary
    processed_data = {
        "N_ctrl": Nc,
        "N_reserv": N_R,
        "Trotter_Step": trot,
        "all_qfim_eigvals": qfim_eigval_list,
        "all_full_qfim_mats":qfim_mats_list,
        "mean_entropy": np.mean(entropies) if entropies else np.nan,
        "num_test_keys": num_test_keys,
        "key_pair_tuple":key_tuple,
        # etc. if needed
    }
    return processed_data

def process_data_expanded(df, threshold, by_test, Nc, N_R, trot, print_bool=False):


    #  'df' is a dict-of-dicts with structure df[fixed_params_key][test_key], etc.
    
    rows = []
    for fixed_params_key in df.keys():
        for test_key in df[fixed_params_key].keys():
           
            qfim_eigvals = df[fixed_params_key][test_key].get('qfim_eigvals', None)
            qfim_mat = df[fixed_params_key][test_key].get('qfim',None)
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
def process_and_cache_new_files_expanded(base_path,
                                         K_0,
                                         sample_range,
                                         model_type,
                                         N_ctrls,
                                         threshold,
                                         by_test,
                                         check_for_new_data,
                                         cached_data,
                                         processed_files):
    """
    Similar logic to process_and_cache_new_files, but uses process_data_expanded
    to produce multiple rows per (N_ctrl, N_reserv, Trotter_Step).
    Then caches them in 'digital_expanded_QFIM_Nc_{N_ctrl}_{K_0}K.pkl'.
    """
    import os
    from pathlib import Path
    import pickle
    import pandas as pd

    all_data = []

    for N_ctrl in N_ctrls:
        model_path = Path(base_path) / 'QFIM_results' / model_type / f'Nc_{N_ctrl}' / f'sample_{sample_range}/{K_0}xK'

        if not model_path.exists():
            print(f"[WARN] Directory {model_path} does not exist for N_ctrl={N_ctrl}.")
            continue

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

                if is_valid_pickle_file(data_file):
                    print(f"[INFO] Reading new file: {data_file}")
                    raw_df = load_and_clean_pickle(data_file)
                    trotter_step_num = extract_trotter_step(data_file)
                    reservoir_count = extract_Nr(data_file)

                    # Expand -> multiple row dicts
                    expanded_rows = process_data_expanded(
                        df=raw_df,
                        threshold=threshold,
                        by_test=by_test,
                        Nc=N_ctrl,
                        N_R=reservoir_count,
                        trot=trotter_step_num
                    )
                    # Possibly store to cache
                    cached_data[file_id] = {
                        "processed_rows": expanded_rows,  # Instead of "processed_data"
                        "raw_data": raw_df
                    }
                    processed_files.add(file_id)
                    all_data.extend(expanded_rows)

        # After scanning all directories for this N_ctrl, save the updated cache
        # Use a new name e.g. 'digital_expanded_QFIM_Nc_{N_ctrl}_{K_0}K.pkl'
        new_cache_file = os.path.join(base_path, f'digital_expanded_QFIM_Nc_{N_ctrl}_{K_0}K.pkl')
        with open(new_cache_file, 'wb') as f:
            pickle.dump((cached_data, processed_files), f)
        print(f"[INFO] Expanded cache saved for N_ctrl={N_ctrl} at {new_cache_file}.")

    return cached_data, processed_files, pd.DataFrame(all_data)
def rebuild_df_expanded_from_cache(base_path, N_ctrls, K_0):
    """
    For each N_ctrl, load the expanded cache file 
      'digital_expanded_QFIM_Nc_{N_ctrl}_{K_0}K.pkl'
    Then combine all 'processed_rows' into one big df.
    """
    import os, pickle
    import pandas as pd

    all_data = []
    for N_ctrl in N_ctrls:
        cache_file = os.path.join(base_path, f'digital_expanded_QFIM_Nc_{N_ctrl}_{K_0}K.pkl')
        if not os.path.isfile(cache_file):
            print(f"[WARN] No expanded cache for N_ctrl={N_ctrl}. Skipping.")
            continue
        with open(cache_file, 'rb') as f:
            cached_data, processed_files = pickle.load(f)
        # each entry in cached_data is { "processed_rows": [...], "raw_data":... }
        for file_id, file_dict in cached_data.items():
            rows = file_dict.get("processed_rows", [])
            all_data.extend(rows)
    df_all = pd.DataFrame(all_data)
    return df_all
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


#######################################################
# 3) PROCESSING DQFIM: Build a DataFrame from data-based QFIM pickles
#######################################################
def process_data_dqfim(raw_dict, n_ctrl, n_reserv, trotter_steps, threshold=1e-12):
    """
    Convert a single raw dictionary for DQFIM into a list of row dicts.
    Each row might correspond to (fixed_param_key, test_key) in your file.

    Typically, the dictionary structure is:
      raw_dict[fixed_param_key][test_key] = {
          'L': array_of_input_states,   # optional
          'qfim_eigvals': ...,
          'qfim': ...,  # NxN matrix
          'entropies': ...,  # optional
          ...
      }

    Returns a list of row dicts for direct DataFrame creation.
    """
    rows = []
    for fixed_param_key, test_entries in raw_dict.items():
        for test_key, results in test_entries.items():
            qfim_eigvals = results.get('qfim_eigvals', None)
            qfim_mat = results.get('qfim', None)  # NxN
            entropies = results.get('entropies', None)
            # Possibly also store L = results.get('L', None) => the input states used

            row = {
                "N_ctrl": n_ctrl,
                "N_reserv": n_reserv,
                "Trotter_Step": trotter_steps,
                "fixed_param_key": fixed_param_key,
                "test_key": test_key,

                "qfim_eigvals": qfim_eigvals,
                "qfim_mats_list": [qfim_mat] if qfim_mat is not None else [],
                # store as a single-item list if you want to do the Qiskit dimension approach
                "entropies": entropies,
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


def build_df_dqfim_dataframe(base_path,
                             n_ctrl,
                             k_str,
                             n_reserv_list,
                             trot_list,
                             num_input_states,  # this is the # of training states used to build the DQFIM
                             threshold=1e-12):
    """
    Example function that scans for DQFIM data pickles (like 'data.pickle') in your
    'QFIM_global_results/gate_model_DQFIM/...' directory, loads them, and builds
    a DataFrame with one row per (fixed_params, test_key).

    Then we'll call compute_all_stats(..., dataset_sizes=num_input_states)
    so that if you want Qiskit dimension, it uses that as well.

    Returns
    -------
    df_out : pd.DataFrame
        Each row has QFIM data from one random draw. We then compute the usual
        rank, trace, etc. plus the potential Qiskit dimension if qfim_mats_list is present.
    """

    # or adapt if your 'analysis_specific_config' is integrated, etc.

    all_rows = []

    # Example path for DQFIM: base_path/QFIM_global_results/gate_model_DQFIM/Nc_{n_ctrl}/sample_pi/1xK...
    dqfim_root = Path(base_path) / "QFIM_global_results" / "gate_model_DQFIM" / f"Nc_{n_ctrl}" / f"sample_pi/{k_str}xK"  # adapt if needed
    for n_rsv in n_reserv_list:
        nr_folder = dqfim_root / f"Nr_{n_rsv}"
        if not nr_folder.exists():
            continue

        for trot in trot_list:
            trot_folder = nr_folder / f"trotter_step_{trot}" / f"L_{num_input_states}"
            if not trot_folder.exists():
                continue

            data_file = trot_folder / "data.pickle"
            if not data_file.exists():
                continue
            if not is_valid_pickle_file(data_file):
                continue

            raw_dict = load_and_clean_pickle(data_file)
            # parse it
            row_dicts = process_data_dqfim(raw_dict, n_ctrl, n_rsv, trot, threshold=threshold)
            all_rows.extend(row_dicts)

    df_dqfim = pd.DataFrame(all_rows)
    if df_dqfim.empty:
        print("[WARN] No DQFIM data found.")
        return df_dqfim

    # Now compute the usual QFIM-based stats using compute_all_stats
    df_out = build_qfim_dataframe_dqfim(df_dqfim, threshold=threshold, dataset_sizes=num_input_states)
    return df_out


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
    logs = np.zeros_like(filtered)
    with np.errstate(divide="ignore"):
        logs[filtered>0]=np.log(filtered[filtered>0])
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
def compute_ipr_dimension(eigenvalues, threshold=1e-10):
    """
    Compute the inverse-participation-ratio-based dimension for a
    single list of eigenvalues, following eq. (32):
        d_eff = (sum(eigvals)^2) / sum(eigvals^2)
    Eigenvalues below 'threshold' are treated as negligible.
    """
    # Filter out very small eigenvalues
    valid_eigs = [val for val in eigenvalues if val > threshold]
    
    if not valid_eigs:
        # If all eigenvalues are below threshold, return 0
        return 0.0
    
    sum_eigs = sum(valid_eigs)
    sum_eigs_sq = sum(val**2 for val in valid_eigs)
    
    # If sum_eigs_sq == 0 (which is extremely unlikely if valid_eigs is non-empty),
    # set dimension to 0 to avoid divide-by-zero
    if sum_eigs_sq == 0:
        return 0.0
    
    return (sum_eigs ** 2) / sum_eigs_sq
def compute_all_stats(
    eigval_list,
    threshold=1e-12,
    spread_methods=("variance", "mad"),  # e.g. ["variance", "mad"]
    ddof=1,
    scale="normal",
    # Additional args for the approximate effective dimension
    do_effective_dim=True,
    n = 1,
     # NEW ARGS for Qiskit-like step:
    qfim_mats_list=None,
    dataset_sizes=None,
 
):
    """
    Compute QFIM statistics for a list of draws (eigval_list) corresponding to one experimental run.
    
    Here, each element of eigval_list represents the raw QFIM eigenvalues computed
    from a single draw (i.e. one sample of randomly generated trainable parameters) for a 
    given configuration (specified by a unique combination of N_ctrl, reservoir qubits (N_R),
    and Trotter step (T)). 

    Parameters
    ----------
    eigval_list : List[List[float]] or List[np.ndarray]
        Each element of eigval_list represents the QFIM eigenvalues computed for 
        a single random draw (trainable parameters) in your experiment.
    threshold : float
        Threshold for zeroing out near-zero eigenvalues.
    spread_methods : tuple of str
        Methods for spread-of-log computations. Possible values: "variance", "mad", "median".
    ddof : int
        Delta degrees of freedom for variance-based computations.
    scale : str
        Scale for median_abs_deviation in "mad" case.
    do_effective_dim : bool
        Whether to compute IPR-based and Abbas-based effective dimension metrics.
    qfim_mats_list : List[np.ndarray], optional
        If provided, each element is the *full NxN QFIM matrix* for that draw. 
        Must be the same length as eigval_list. 
        We'll average them and do the determinant-based dimension from Qiskit's approach.
    dataset_sizes : int or list of ints, optional
        The 'n' or array of 'n' values for computing the final dimension. 
        By default, Qiskit's code calls this 'dataset_size'. 
        If None, we skip the global ED calculation.

        Notes on the Three Categories of Metrics:
      1) ABSOLUTE SCALE: e.g. raw trace and raw variance (sums of eigenvalues, etc.).
      2) SHAPE OF THE SPECTRUM: e.g. normalized IPR, normalized Abbas dimension, 
         which ignore total magnitude by normalizing eigenvalues to sum=1.
      3) AVERAGE PER NONZERO MODE: e.g. trace/rank, variance/rank.

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

      
     Returns
    -------
    metrics : dict
        Dictionary containing per-draw lists and aggregated statistics. 
        The keys are split into categories:

        [1] "absolute_scale_*" ...
        [2] "spectrum_shape_*" ...
        [3] "avg_per_active_mode_*" ...
        
        ... plus spread-of-log metrics for each method in spread_methods.
    
    Additional Notes:
    -----------------
    This function assumes that eigval_list is a list of 1D arrays (or lists),
    where each array corresponds to the QFIM eigenvalues computed for a single
    random draw of trainable parameters. The computed metrics are intended to capture
    both per-draw variability and the overall (pooled) behavior of the QFIM eigenvalue distribution,
    which are later used to correlate with measures of learnability and generalization.
    """

    import numpy as np

    # -------------------------------------------------------------------------
    # 1) Per-draw computations
    # -------------------------------------------------------------------------
    ranks_per_draw = []
    var_all_per_draw = []     # raw variance (absolute scale)
    var_nonzero_per_draw = [] # raw variance of nonzero eigenvals only
    trace_per_draw = []       # raw trace (absolute scale)

    var_norm_rank_per_draw = []   # average per nonzero mode
    trace_norm_rank_per_draw = [] # average per nonzero mode

    # Effective dimension (IPR & Abbas) - raw vs. normalized
    ipr_deffs_raw    = []
    ipr_deffs_norm   = []
    abbas_deffs_raw  = []
    abbas_deffs_norm = []
    abbas_deffs_norm_by_gc = []

    # For Abbas measure (optional local dimension approach)
    # =============== NEW #2: Abbas local dimension ===============
    # d_{n, gamma}(theta) ~ -log(V_theta) + sum_i log(1 + alpha * lambda_i)
    # We'll just set V_theta=1 => -log(1)=0, so that term vanishes.
    # Then we do sum(log(1 + alpha * lambda_i)).
    # If alpha*lambda_i < -1, that log is undefined; in practice, alpha*lambda_i >= 0 if alpha>0, lambda_i >= 0

    # n_draws = len(eigval_list)   # interpret as "number of random draws"
    n_draws = n
    # For Abbas measure (a local version)
    gamma = 1.0
    if n_draws > 1:
        alpha = (gamma * n_draws) / (2.0 * np.log(n_draws))
    else:
        alpha = 0.0
    V_theta = 1.0  # placeholder

    for eigs in eigval_list:
        arr = np.array(eigs, dtype=float)
       
        arr = np.where(arr < threshold, 0.0, arr)   # threshold small values
        # rank = #nonzero eigenvalues
        rank = np.count_nonzero(arr)
        ranks_per_draw.append(rank)

        # raw variance + trace = "absolute scale"
        var_all = np.var(arr)
        trace_val = arr.sum()

        var_all_per_draw.append(var_all)
        trace_per_draw.append(trace_val)

        # variance over nonzero eigenvals (still absolute scale, ignoring zeros)
        nonz = arr[arr > 0]
        var_non = np.var(nonz) if nonz.size > 1 else 0.0
        var_nonzero_per_draw.append(var_non)

        
        var_norm_rank_per_draw.append(var_all / rank)
        trace_norm_rank_per_draw.append(trace_val / rank)

         # --------------------------- IPR-based d_eff ---------------------------
        # raw
        sum_eigs_sq = np.sum(arr**2)
        ipr_raw = (trace_val**2) / sum_eigs_sq

        ipr_deffs_raw.append(ipr_raw)

        # normalized (i.e., shape only)
      
        arr_norm = arr / trace_val
        sum_norm_sq = np.sum(arr_norm**2)
        if sum_norm_sq > 0.0:
            ipr_norm = 1.0 / sum_norm_sq
        else:
            ipr_norm = 0.0

        ipr_deffs_norm.append(ipr_norm)

        # --------------------------- Abbas-based d_eff -------------------------
        # raw
        abbas_raw = 0.0
        for lam in arr:
            val = 1.0 + alpha * lam
            if val <= 0.0:
                val = 1e-15
            abbas_raw += np.log(val)
        # if V_theta != 1: abbas_raw -= np.log(V_theta)
        abbas_deffs_raw.append(abbas_raw)
        abbas_deffs_norm_by_gc.append(compute_ipr_dimension(eigs))

        # normalized
        # abbas_norm = 0.0
        # if trace_val > 0:
        #     for lam_norm in (arr / trace_val):
        #         val = 1.0 + alpha * lam_norm
        #         if val <= 0.0:
        #             val = 1e-15
        #         abbas_norm += np.log(val)
        # if trace_val > 0:
        #     abbas_norm = 0.0
        #     for lam_norm in (arr / trace_val):
        #         val = 1.0 + alpha * lam_norm
        #         if val <= 0.0:
        #             val = 1e-15
        #         abbas_norm += np.log(val)
        # else:
        #     abbas_norm = 0.0

        abbas_norm=0.0
        arr_norm = arr/trace_val
        for lam_norm in arr_norm:
            val = 1.0+ alpha*lam_norm
            if val<=0.0:
                val=1e-15
            abbas_norm+=np.log(val)
        abbas_deffs_norm.append(abbas_norm)
        # if V_theta != 1: abbas_norm -= np.log(V_theta)
        # abbas_deffs_norm.append(abbas_norm)


    # -------------------------------------------------------------------------
    # 2) Aggregate across draws
    # -------------------------------------------------------------------------
    D_C = max(ranks_per_draw) if ranks_per_draw else 0

    avg_var_all = float(np.mean(var_all_per_draw)) if var_all_per_draw else 0.0
    avg_trace = float(np.mean(trace_per_draw)) if trace_per_draw else 0.0
    avg_var_nonzero = float(np.mean(var_nonzero_per_draw)) if var_nonzero_per_draw else 0.0
    avg_var_norm_rank = float(np.mean(var_norm_rank_per_draw)) if var_norm_rank_per_draw else 0.0
    avg_trace_norm_rank = float(np.mean(trace_norm_rank_per_draw)) if trace_norm_rank_per_draw else 0.0

    var_var_all = float(np.var(var_all_per_draw)) if len(var_all_per_draw) > 1 else 0.0
    var_var_nonzero = float(np.var(var_nonzero_per_draw)) if len(var_nonzero_per_draw) > 1 else 0.0

    avg_ipr_raw   = float(np.mean(ipr_deffs_raw))  if ipr_deffs_raw  else 0.0
    avg_ipr_norm  = float(np.mean(ipr_deffs_norm)) if ipr_deffs_norm else 0.0
    avg_abbas_raw = float(np.mean(abbas_deffs_raw))  if abbas_deffs_raw  else 0.0
    avg_abbas_norm= float(np.mean(abbas_deffs_norm)) if abbas_deffs_norm else 0.0
    abbas_deffs_simple = float(np.mean(abbas_deffs_norm_by_gc))

    # -------------------------------------------------------------------------
    # 3) Spread-of-log metrics on the entire 2D array of eigenvalues
    # -------------------------------------------------------------------------
    if eigval_list:
        max_len = max(len(x) for x in eigval_list)
    else:
        max_len = 0
    arr_2d = np.zeros((n_draws, max_len))
    for i, e in enumerate(eigval_list):
        tmp = np.array(e, dtype=float)
        tmp = np.where(tmp < threshold, 0.0, tmp)
        arr_2d[i, :len(tmp)] = tmp

    spread_results = {}
    for method in spread_methods:
        per_draw = spread_per_sample_vectorized(
            arr_2d, method=method, threshold=threshold, ddof=ddof, scale=scale
        )
        spread_mean = float(per_draw.mean()) if per_draw.size else 0.0
        spread_std  = float(per_draw.std())  if per_draw.size > 1 else 0.0
        pooled_val  = float(spread_pooling_vectorized(
            arr_2d, method=method, threshold=threshold, ddof=ddof, scale=scale
        ))
        prefix = method.lower()
        spread_results[f"spread_mean_per_sample_{prefix}_{scale}"] = spread_mean
        spread_results[f"spread_std_per_sample_{prefix}_{scale}"]  = spread_std
        spread_results[f"spread_val_pooled_{prefix}_{scale}"]      = pooled_val

    # # -------------------------------------------------------------------------
    # # 4) Build Final Dictionary with CLEAR Category Labels
    # # -------------------------------------------------------------------------
    # metrics = {
    #     # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #     # Category [A]: Basic Info & Per-draw lists
    #     # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #     "QFIM_ranks": ranks_per_draw,
    #     "var_all_eigenvals_per_draw": var_all_per_draw,       # absolute scale
    #     "var_nonzero_eigenvals_per_draw": var_nonzero_per_draw,# absolute scale
    #     "trace_eigenvals_per_draw": trace_per_draw,           # absolute scale

    #     # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #     # Category [1]: ABSOLUTE SCALE (aggregated)
    #     # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #     "absolute_scale_avg_var_all": avg_var_all,
    #     "absolute_scale_avg_var_nonzero": avg_var_nonzero,
    #     "absolute_scale_avg_trace": avg_trace,

    #     # For those wanting the variance of the 'var_all_eigenvals_per_draw'
    #     "absolute_scale_var_of_var_all": var_var_all,
    #     "absolute_scale_var_of_var_nonzero": var_var_nonzero,

    #     # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #     # Category [2]: SHAPE OF THE SPECTRUM
    #     # (these are the normalized versions of IPR & Abbas)
    #     # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #     "spectrum_shape_ipr_deffs_norm_per_draw": ipr_deffs_norm,
    #     "spectrum_shape_avg_ipr_deffs_norm": avg_ipr_norm,
        
    #     "spectrum_shape_abbas_deffs_norm_per_draw": abbas_deffs_norm,
    #     "spectrum_shape_avg_abbas_deffs_norm": avg_abbas_norm,

    #     # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #     # Category [3]: AVERAGE PER NONZERO MODE
    #     # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #     "avg_per_active_mode_var_norm_rank_per_draw": var_norm_rank_per_draw,
    #     "avg_per_active_mode_trace_norm_rank_per_draw": trace_norm_rank_per_draw,

    #     "avg_per_active_mode_avg_var_norm_rank": avg_var_norm_rank,
    #     "avg_per_active_mode_avg_trace_norm_rank": avg_trace_norm_rank,

    #     # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #     # Rank-based dimension as a simpler measure of capacity
    #     # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #     "D_C": D_C,  # max rank observed across draws

    #     # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #     # RAW IPR & ABBAS for absolute scale dimension
    #     # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #     "ipr_deffs_raw_per_draw": ipr_deffs_raw,
    #     "avg_ipr_deffs_raw": avg_ipr_raw,

    #     "abbas_deffs_raw_per_draw": abbas_deffs_raw,
    #     "avg_abbas_deffs_raw": avg_abbas_raw,
    # }


    ###############
    # 4) Optionally: "global" dimension approach from the references
    ###############
    # (like the integral-based approach in [Abbas2020], but we do 
    #  a simpler average-Fisher approach if qfim_mats_list is provided).
    global_dim_results = {}
    if (qfim_mats_list is not None) and (dataset_sizes is not None) and len(qfim_mats_list)==n_draws:
        # We'll do a simple "empirical average fisher" -> normalized fisher -> logdet approach
        # to replicate a global dimension measure.
        fisher_stack = np.stack(qfim_mats_list, axis=0)  # shape (n_draws, N, N)
        # average them -> 'empirical fisher'
        avg_fisher = np.mean(fisher_stack, axis=0)       # shape (N,N)
        # check trace
        fisher_trace = np.trace(avg_fisher)
        n_params = avg_fisher.shape[0]
        if fisher_trace<1e-14:
            # degenerate
            normalized_fisher = np.zeros_like(avg_fisher)
        else:
            normalized_fisher = (n_params * avg_fisher)/fisher_trace
        
        # define helper to compute the effective dimension from [Abbas2020 eq ...].
        # We'll do a single or list of dataset_sizes
        if isinstance(dataset_sizes, (int,float)):
            dataset_sizes = [dataset_sizes]
        import numpy.linalg as la
        from math import log
        out_dims = []
        for ds in dataset_sizes:
            if ds<=1 or np.log(ds)<=0:
                out_dims.append(0.0)
                continue
            # build f_mod
            factor = ds/(2.0*np.pi*log(ds))
            f_mod = normalized_fisher*factor
            one_plus = np.eye(n_params)+f_mod
            sign, logdet_val = la.slogdet(one_plus)
            if sign<=0:
                # negative or zero => dimension = 0?
                out_dims.append(0.0)
                continue
            # eq dimension
            det_div = 0.5*logdet_val
            denom = log(ds/(2.0*np.pi*log(ds)))
            # no logsumexp needed, we do single average approach
            eff_dim = 2.0*det_div/denom if denom!=0 else 0.0
            out_dims.append(eff_dim)
        if len(out_dims)==1:
            global_dim_results["global_effective_dimension"] = out_dims[0]
        else:
            global_dim_results["global_effective_dimension"] = out_dims
        global_dim_results["fisher_trace"] = fisher_trace

    ###############
    # 5) Build final dictionary
    ###############
    metrics = {
        # Per-draw
        "QFIM_ranks": ranks_per_draw,
        "test_var_qfim_eigvals": var_all_per_draw,
        "test_var_qfim_eigvals_nonzero": var_nonzero_per_draw,
        "test_tr_qfim_eigvals": trace_per_draw,
        "test_var_qfim_eigvals_normalized_by_rank": var_norm_rank_per_draw,
        "test_tr_qfim_eigvals_norm_by_rank": trace_norm_rank_per_draw,

        # Summaries
        "D_C": D_C,  # rank-based dimension (max rank)
        "avg_test_var_qfim_eigvals": float(avg_var_all),
        "avg_test_var_qfim_eigvals_nonzero": float(avg_var_nonzero),
        "avg_test_tr_qfim_eigvals": float(avg_trace),
        "avg_test_var_qfim_eigvals_normalized_by_rank": float(avg_var_norm_rank),
        "avg_test_tr_qfim_eigvals_norm_by_rank": float(avg_trace_norm_rank),
        "var_test_var_qfim_eigvals": float(var_var_all),
        "var_test_var_qfim_eigvals_nonzero": float(var_var_nonzero),
        
        # IPR-based local dimension
        "ipr_deffs_raw": ipr_deffs_raw,
        "ipr_deffs_norm": ipr_deffs_norm,
        "avg_ipr_deffs_raw": float(avg_ipr_raw),
        "avg_ipr_deffs_norm": float(avg_ipr_norm),
        
        # Abbas-based local dimension
        "abbas_deffs_raw": abbas_deffs_raw,
        "abbas_deffs_norm": abbas_deffs_norm,
        "avg_abbas_deffs_raw": float(avg_abbas_raw),
        "avg_abbas_deffs_norm": float(avg_abbas_norm),
        "abbas_deffs_simple":abbas_deffs_simple,
    }

    # Add in spread-of-log results
    metrics.update(spread_results)

    # Possibly add global dimension results if computed
    if global_dim_results:
        metrics.update(global_dim_results)

    return metrics
   

def build_qfim_dataframe(df_all, threshold=1e-12, dataset_sizes=None):
    """
    1) Convert all_qfim_eigvals -> qfim_eigs_2d (convenience for some spread-of-log steps).
    2) For each row, call compute_all_stats, which calculates:
       - Rank, trace, variance, IPR-based dimension, Abbas dimension, etc. from the eigenvalues,
       - If 'qfim_mats_list' is present, also compute Qiskit-style global ED for the given 'dataset_sizes'.

    Parameters
    ----------
    df_all : pd.DataFrame
        Must have columns:
            - "all_qfim_eigvals": List of eigenvalue arrays (one set per row).
            - "qfim_mats_list" (optional): List of QFIM NxN matrices (same length as eigenvals).
              If present, we can compute global dimension from them.
    threshold : float
        For zeroing out small eigenvalues.
    dataset_sizes : int or list of int, optional
        If provided, replicate Qiskit's global ED for each dataset size in this argument.
        E.g. could be 100, or [50, 100, 200]. If None, skip Qiskit-style ED.

    Returns
    -------
    df_out : pd.DataFrame
        The original df_all with additional columns for stats, including potential global ED results
        in "qiskit_style_globalED" and "qiskit_style_avgFisherTrace".
    """
    # 1) Convert for convenience
    df_all["qfim_eigs_2d"] = df_all["all_qfim_eigvals"].apply(to_2d)

    # 2) Single-pass stats for each row
    def _per_row_stats(row):
        eigs_list = row["all_qfim_eigvals"]  # list of 1D arrays
        # optional: None if not present
        qfim_mats_list = row.get("qfim_mats_list", None)

        # call compute_all_stats with or without qfim_mats_list
        stats = compute_all_stats(
            eigval_list=eigs_list,
            threshold=threshold,
            spread_methods=["variance", "mad"],  # or add 'median'
            ddof=1,
            scale="normal",
            qfim_mats_list=qfim_mats_list,   # newly added param
            dataset_sizes=dataset_sizes      # newly added param
        )
        return stats

    # Apply row-wise
    stats_series = df_all.apply(_per_row_stats, axis=1)
    # 3) Flatten stats (dict) to columns
    df_stats = pd.json_normalize(stats_series)
    # Combine with original
    df_out = pd.concat([df_all.reset_index(drop=True), df_stats.reset_index(drop=True)], axis=1)
        # 3) If you still want “extra” spread columns for e.g. 'median' or other transformations,
    #    you can also call your existing compute_spread_columns(...) here:
    # df_out = compute_spread_columns(df_out, threshold=threshold, spread_method="median", scale="normal")
    # etc.

    
    # Spread-based columns: variance, mad, median
    # df_all = compute_spread_columns(df_all, threshold=threshold, spread_method="variance", scale="normal")
    # df_all = compute_spread_columns(df_all, threshold=threshold, spread_method="mad", scale="normal")
    # df_all = compute_spread_columns(df_all, threshold=threshold, spread_method="median", scale="normal")
    return df_out




if __name__ == "__main__":
    base_path = "/Users/sophieblock/QRCcapstone/parameter_analysis_directory/"
    model_type = "gate"
    N_ctrls = [2, 3]
    sample_range = "pi"
    K_str = "1"
    threshold = 1e-10
    by_test = False

    # 1) fetch cached data:
    #    df_all = rebuild_df_from_existing_cache(base_path, N_ctrls, K_str)
    #    if df_all.empty, we call process_and_cache_new_files
    #    or we do the combined approach:
    df_all = maybe_rebuild_or_process(
        base_path, sample_range, model_type, N_ctrls, K_str,
        threshold=threshold, by_test=by_test
    )

    print("[INFO] df_all shape after reading cache or scanning directories:", df_all.shape)

    # # 2) Build QFIM DataFrame with advanced metrics
    # df_all = build_qfim_dataframe(df_all, threshold=threshold)
    # print("[INFO] df_all final shape:", df_all.shape)
    # print(df_all.head())