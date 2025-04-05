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

has_jax = True
diable_jit = False
config.update('jax_disable_jit', diable_jit)
#config.parse_flags_with_absl()
config.update("jax_enable_x64", True)
os.environ['JAX_TRACEBACK_FILTERING'] = 'off'
# global_cache_data = None
# global_processed_files = None

###############################################################################
#  Common Helpers
###############################################################################

def clean_array(data):
    """Helper function to clean deprecated JAX arrays and nested structures."""
    if isinstance(data, np.ndarray):
        return np.array(data)  # Strip any unsupported attributes
    elif isinstance(data, dict):
        return {k: clean_array(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_array(v) for v in data]
    elif hasattr(data, 'named_shape'):  # Specifically check for deprecated JAX attributes
        del data.named_shape
        return data
    else:
        return data  # Return as-is if not an array or collection

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
                    print(f"File {file_path} is corrupted (EOFError).")
                    return False
    except Exception as e:
        print(f"An error occurred reading {file_path}: {e}")
        return False
    return False

def load_and_clean_pickle(data_file):
    """Helper function to load and clean a pickle file."""
    with open(data_file, 'rb') as f:
        df = pickle.load(f)
        return clean_array(df)

def get_keys(df):
    """
    Helper function to extract a minimal representation of the data keys.
    Returns a dict mapping each fixed_param key to the set of test keys.
    """
    return {fixed_param: set(df[fixed_param].keys()) for fixed_param in df}


###############################################################################
# 1) DQFIM-Specific Cache: get_cached_data_dqfim / save_cached_data_dqfim
###############################################################################
def get_cached_data_dqfim(base_path, model_type, N_ctrl, K_0, sample_label, num_L):
    """
    Load DQFIM cached_data if it exists, else return ( {}, set() ).

    The expected filename is: 
       '{model_type}_DQFIM_Nc_{N_ctrl}_{K_0}K_sample_{sample_label}_L{num_L}.pkl'
    """
    import os
    cache_file = os.path.join(
        base_path,
        f"{model_type}_DQFIM_Nc_{N_ctrl}_{K_0}K_sample_{sample_label}_L{num_L}.pkl"
    )

    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            cached_data, processed_files = pickle.load(f)
        print(f"[INFO] Loaded DQFIM cache for N_ctrl={N_ctrl} from {cache_file}, num files: {len(processed_files)}.")
        return cached_data, processed_files
    else:
        print(f"[WARN] No DQFIM cache for N_ctrl={N_ctrl} at {cache_file}. Returning empty.")
        return {}, set()

def save_cached_data_dqfim(base_path, model_type, cached_data, processed_files, N_ctrl, K_0, sample_label, num_L):
    """
    Save the (cached_data, processed_files) for DQFIM to 
    '{model_type}_DQFIM_Nc_{N_ctrl}_{K_0}K_sample_{sample_label}_L{num_L}.pkl'
    """
    import os
    cache_file = os.path.join(
        base_path,
        f"{model_type}_DQFIM_Nc_{N_ctrl}_{K_0}K_sample_{sample_label}_L{num_L}.pkl"
    )

    with open(cache_file, 'wb') as f:
        pickle.dump((cached_data, processed_files), f)
    print(f"[INFO] DQFIM cache saved for N_ctrl={N_ctrl} => {cache_file}.")

def process_data_combined_dqfim(raw_dict, N_ctrl, threshold=1e-12):
    """
    Similar to 'process_data_combined' in qfim_store.py.
    We gather *all* the qfim_eigvals, qfim_mats, etc. from every (fixed_param_key, test_key)
    in 'raw_dict' into one aggregated dictionary. This yields a single row per file.

    Output keys: [
       "N_ctrl", "N_reserv", "Trotter_Step",
       "all_qfim_eigvals", "all_full_qfim_mats",
       "mean_entropy", "num_test_keys", "key_pair_tuple"
    ]
    """
    qfim_eigval_list = []
    qfim_mats_list   = []
    entropies        = []
    key_tuple        = []
    N_R   = None
    trot = None

    # Summarize how many (fixed_param_key, test_key) pairs
    keys_snapshot = get_keys(raw_dict)
    num_test_keys = sum(len(tests) for tests in keys_snapshot.values())

    for fixed_param_key in raw_dict:
        for test_key in raw_dict[fixed_param_key]:
            entry = raw_dict[fixed_param_key][test_key]

            # Gather eigenvalues
            qfims_eigs = entry.get('qfim_eigvals', None)
            if qfims_eigs is not None:
                qfim_eigval_list.append(qfims_eigs)

            # Gather QFIM matrix
            qfims_mat = entry.get('qfim', None)
            if qfims_mat is not None:
                qfim_mats_list.append(qfims_mat)

            # Optionally gather entropies
            e = entry.get('entropies', None)
            if e is not None:
                entropies.append(e)

            # Record the param/test keys
            key_tuple.append((fixed_param_key, test_key))

            # Attempt to read n_reserv, time_steps
            # (some files store them per test_key).
            if "n_reserv" in entry:
                N_R = entry["n_reserv"]
            if "time_steps" in entry:
                trot = entry["time_steps"]

    # Build aggregator dictionary
    out_dict = {
        "N_ctrl": N_ctrl,
        "N_reserv": N_R,
        "Trotter_Step": trot,
        "all_qfim_eigvals": qfim_eigval_list,
        "all_full_qfim_mats": qfim_mats_list,
        "mean_entropy": float(np.mean(entropies)) if entropies else np.nan,
        "num_test_keys": num_test_keys,
        "key_pair_tuple": key_tuple,
    }
    return out_dict
###############################################################################
# 2) Processing a Single Dictionary of DQFIM Data => Row Dicts
###############################################################################
def process_data_dqfim(raw_dict, n_ctrl, n_reserv, trotter_steps, threshold=1e-12):
    """
    Convert a single raw dictionary for DQFIM into a list of row dicts.
    Typically, the structure is:
      raw_dict[fixed_param_key][test_key] = {
          'qfim_eigvals': ...,
          'qfim': ...,        # NxN matrix
          'entropies': ...,   # optional
          ...
      }

    Returns a list of row dicts. Each row corresponds to (fixed_param_key, test_key).
    """
    rows = []
    for fixed_param_key, test_entries in raw_dict.items():
        for test_key, results in test_entries.items():
            qfim_eigvals = results.get('qfim_eigvals', None)
            qfim_mat = results.get('qfim', None)
            entropies = results.get('entropies', None)

            row = {
                "N_ctrl": n_ctrl,
                "N_reserv": n_reserv,
                "Trotter_Step": trotter_steps,
                "fixed_param_key": fixed_param_key,
                "test_key": test_key,
                "qfim_eigvals": qfim_eigvals,
                "qfim_mats_list": [qfim_mat] if qfim_mat is not None else [],
                "entropies": entropies,
            }
            rows.append(row)
    return rows
def process_and_cache_new_files_dqfim(base_path,
                                      K_0,
                                      num_L,
                                      sample_range,
                                      model_type,
                                      N_ctrls,
                                      threshold,
                                      by_test,
                                      check_for_new_data,
                                      cached_data,
                                      processed_files):
    """
    For each new data.pickle file, we produce ONE aggregated dictionary 
    using process_data_combined_dqfim(...), store it in `cached_data[file_id]["processed_data"]`,
    and then build a DataFrame from all those aggregated rows.

    => ONE ROW per file => the columns are 
       "N_ctrl","N_reserv","Trotter_Step",
       "all_qfim_eigvals","all_full_qfim_mats","mean_entropy",
       "num_test_keys","key_pair_tuple"
    """
    from pathlib import Path
    import pandas as pd

    all_rows = []  # Each file => exactly one aggregator dictionary

    for N_ctrl in N_ctrls:
        model_path = Path(base_path) / "DQFIM_results" / model_type / f"Nc_{N_ctrl}" / f"sample_{sample_range}" / f"{K_0}xK"
        if not model_path.exists():
            print(f"[WARN] DQFIM directory {model_path} does not exist for N_ctrl={N_ctrl}.")
            continue

        for Nr_dir in sorted(os.listdir(model_path)):
            Nr_path = model_path / Nr_dir
            if not Nr_path.is_dir():
                continue

            for trot_dir in sorted(os.listdir(Nr_path)):
                trot_path = Nr_path / trot_dir
                if not trot_path.is_dir():
                    continue

                data_folder = trot_path / f"L_{num_L}"
                data_file = data_folder / "data.pickle"
                file_id = str(data_file)

                if not data_folder.exists():
                    continue

                if file_id in cached_data:
                    # Already processed
                    if check_for_new_data:
                        updated = update_cached_data_dqfim(
                            data_file, cached_data, processed_files, 
                            N_ctrl, threshold
                        )
                        if updated:
                            row_data = cached_data[file_id]["processed_data"]
                            all_rows.append(row_data)
                    else:
                        row_data = cached_data[file_id]["processed_data"]
                        all_rows.append(row_data)
                    continue

                # Brand-new file
                if is_valid_pickle_file(data_file):
                    print(f"[INFO] Found NEW DQFIM file: {data_file}")
                    raw_df = load_and_clean_pickle(data_file)
                    aggregated_row = process_data_combined_dqfim(raw_df, N_ctrl, threshold=threshold)
                    cached_data[file_id] = {
                        "processed_data": aggregated_row,
                        "raw_keys": get_keys(raw_df),
                    }
                    processed_files.add(file_id)
                    all_rows.append(aggregated_row)

        # After scanning for this N_ctrl, save
        save_cached_data_dqfim(base_path, model_type, cached_data, processed_files,
                               N_ctrl, K_0, sample_range, num_L)

    df_all = pd.DataFrame(all_rows)
    return cached_data, processed_files, df_all
###############################################################################
# 4) update_cached_data_dqfim: Re-check an existing file for new fixed_param/test_key
###############################################################################
def update_cached_data_dqfim(data_file, cached_data, processed_files, 
                             N_ctrl, threshold, n_reserv, trotter_steps):
    """
    Load the data_file, compare structure vs. cached_data[file_id]["raw_keys"].
    If new param/test keys appear, re-run process_data_dqfim and update.

    Returns True if updated, False if no change.
    """
    file_id = str(data_file)
    raw_df = load_and_clean_pickle(data_file)
    new_keys = get_keys(raw_df)
    old_keys = cached_data[file_id].get("raw_keys", {})

    # Check for newly added fixed_param or test_key
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
        print(f"[INFO] New DQFIM fixed_param/test keys in {file_id}. Updating processed data.")
        row_dicts = process_data_dqfim(
            raw_df,
            n_ctrl=N_ctrl,
            n_reserv=n_reserv,
            trotter_steps=trotter_steps,
            threshold=threshold
        )
        cached_data[file_id]["processed_data"] = row_dicts
        cached_data[file_id]["raw_keys"] = new_keys

    processed_files.add(file_id)
    return needs_update
def update_cache_with_new_data_dqfim(base_path, 
                                     K_0, 
                                     sample_range, 
                                     num_L,
                                     model_type, 
                                     N_ctrl, 
                                     threshold, 
                                     by_test,
                                     cached_data, 
                                     processed_files):
    """
    Scan for new or updated data files => produce aggregator rows. 
    Return a DataFrame of newly added/updated rows.
    """
    import pandas as pd
    from pathlib import Path

    newly_aggregated = []

    model_path = Path(base_path) / "DQFIM_results" / model_type / f"Nc_{N_ctrl}" / f"sample_{sample_range}" / f"{K_0}xK"
    if not model_path.exists():
        print(f"[WARN] DQFIM directory {model_path} does not exist for N_ctrl={N_ctrl}. No new data found.")
        return cached_data, processed_files, pd.DataFrame()

    for Nr_dir in sorted(os.listdir(model_path)):
        Nr_path = model_path / Nr_dir
        if not Nr_path.is_dir():
            continue

        for trot_dir in sorted(os.listdir(Nr_path)):
            trot_path = Nr_path / trot_dir
            if not trot_path.is_dir():
                continue

            data_folder = trot_path / f"L_{num_L}"
            data_file   = data_folder / "data.pickle"
            file_id     = str(data_file)

            if not data_folder.exists():
                continue

            if file_id in cached_data:
                updated = update_cached_data_dqfim(
                    data_file, 
                    cached_data, 
                    processed_files,
                    N_ctrl,
                    threshold
                )
                if updated:
                    row_data = cached_data[file_id]["processed_data"]
                    newly_aggregated.append(row_data)
            else:
                # brand-new
                if is_valid_pickle_file(data_file):
                    print(f"[INFO] Found NEW DQFIM data file: {data_file}")
                    raw_df = load_and_clean_pickle(data_file)
                    aggregated_row = process_data_combined_dqfim(raw_df, N_ctrl, threshold=threshold)
                    cached_data[file_id] = {
                        "processed_data": aggregated_row,
                        "raw_keys": get_keys(raw_df),
                    }
                    processed_files.add(file_id)
                    newly_aggregated.append(aggregated_row)

    df_new = pd.DataFrame(newly_aggregated)
    if len(newly_aggregated) > 0:
        save_cached_data_dqfim(base_path, model_type, cached_data, processed_files,
                               N_ctrl, K_0, sample_range, num_L)
    return cached_data, processed_files, df_new

###############################################################################
# 5) update_cache_with_new_data_dqfim
###############################################################################
# def update_cache_with_new_data_dqfim(base_path, 
#                                      K_0, 
#                                      sample_range, 
#                                      num_L,
#                                      model_type, 
#                                      N_ctrl, 
#                                      threshold, 
#                                      by_test,
#                                      cached_data, 
#                                      processed_files):
#     """
#     For a single N_ctrl, look for new data files not in 'cached_data'.
#     If new, process them with 'process_data_dqfim'.
#     If existing but changed, re-process. Return a DataFrame of newly updated rows.
#     """
#     import pandas as pd
#     from pathlib import Path

#     all_new_rows = []  # again, list-of-lists-of-dicts

#     model_path = Path(base_path) / "DQFIM_results" / model_type / f"Nc_{N_ctrl}" / f"sample_{sample_range}" / f"{K_0}xK"
#     if not model_path.exists():
#         print(f"[WARN] DQFIM directory {model_path} does not exist. No new data found.")
#         return cached_data, processed_files, pd.DataFrame()

#     for Nr_dir in sorted(os.listdir(model_path)):
#         Nr_path = model_path / Nr_dir
#         if not Nr_path.is_dir():
#             continue

#         try:
#             n_reserv = int(Nr_dir.split('_')[-1])
#         except:
#             n_reserv = None

#         for trot_dir in sorted(os.listdir(Nr_path)):
#             trot_path = Nr_path / trot_dir
#             if not trot_path.is_dir():
#                 continue

#             try:
#                 trotter_steps = int(trot_dir.split('_')[-1])
#             except:
#                 trotter_steps = None

#             data_file = trot_path / f"L_{num_L}/data.pickle"
#             file_id = str(data_file)

#             if file_id in cached_data:
#                 print(f"[INFO] Checking cached DQFIM file: {data_file}")
#                 updated = update_cached_data_dqfim(
#                     data_file, 
#                     cached_data, 
#                     processed_files,
#                     N_ctrl,
#                     threshold,
#                     n_reserv,
#                     trotter_steps
#                 )
#                 if updated:
#                     # If changed, re-append the new processed_data
#                     all_new_rows.append(cached_data[file_id]["processed_data"])
#                 continue

#             # brand-new file
#             if is_valid_pickle_file(data_file):
#                 print(f"[INFO] Found NEW DQFIM data file: {data_file}")
#                 raw_df = load_and_clean_pickle(data_file)
#                 row_dicts = process_data_dqfim(
#                     raw_df,
#                     n_ctrl=N_ctrl,
#                     n_reserv=n_reserv,
#                     trotter_steps=trotter_steps,
#                     threshold=threshold
#                 )
#                 cached_data[file_id] = {
#                     "processed_data": row_dicts,
#                     "raw_keys": get_keys(raw_df)
#                 }
#                 processed_files.add(file_id)
#                 all_new_rows.append(row_dicts)

#     # Flatten the newly added rows
#     flat_new = []
#     for block in all_new_rows:
#         flat_new.extend(block)

#     df_new = pd.DataFrame(flat_new)

#     # If we found new data, re-save the updated cache:
#     if len(flat_new) > 0:
#         save_cached_data_dqfim(base_path, model_type, cached_data, processed_files,
#                                N_ctrl, K_0, sample_range, num_L)

#     return cached_data, processed_files, df_new


###############################################################################
# 6) maybe_rebuild_or_process_dqfim
###############################################################################
def maybe_rebuild_or_process_dqfim(base_path, 
                                   sample_range, 
                                   model_type, 
                                   N_ctrls, 
                                   K_0, 
                                   sample_label,
                                   num_L,
                                   threshold=1e-10, 
                                   by_test=False):
    """
    1) For each N_ctrl, load existing cache if any.
    2) If none => process_and_cache_new_files_dqfim
    3) If found => update_cache_with_new_data_dqfim
    4) Combine all aggregator rows into df_all
    """
    import pandas as pd

    all_frames = []
    for N_ctrl in N_ctrls:
        cached_data, processed_files = get_cached_data_dqfim(
            base_path, model_type, N_ctrl, K_0, sample_label, num_L
        )

        if not cached_data:
            print(f"[INFO] No existing DQFIM cache for N_ctrl={N_ctrl}, scanning directories.")
            cached_data, processed_files, df_partial = process_and_cache_new_files_dqfim(
                base_path=base_path,
                K_0=K_0,
                num_L=num_L,
                sample_range=sample_range,
                model_type=model_type,
                N_ctrls=[N_ctrl],
                threshold=threshold,
                by_test=by_test,
                check_for_new_data=True,
                cached_data=cached_data,
                processed_files=processed_files
            )
            all_frames.append(df_partial)
        else:
            # We do have a cache => see if there's new data
            cached_data, processed_files, df_new = update_cache_with_new_data_dqfim(
                base_path=base_path,
                K_0=K_0,
                sample_range=sample_range,
                num_L=num_L,
                model_type=model_type,
                N_ctrl=N_ctrl,
                threshold=threshold,
                by_test=by_test,
                cached_data=cached_data,
                processed_files=processed_files
            )
            # Also load old aggregator data
            old_rows = [fdata["processed_data"] for fdata in cached_data.values()]
            df_partial = pd.DataFrame(old_rows)
            all_frames.append(df_partial)

    if all_frames:
        df_all = pd.concat(all_frames, ignore_index=True)
    else:
        df_all = pd.DataFrame()

    return df_all



###############################################################################
# Example usage
###############################################################################
if __name__ == "__main__":
    base_path = "/Users/sophieblock/QRCCapstone/parameter_analysis_directory/"
    model_type = "gate"
    num_L = 20
    N_ctrls = [2]
    sample_range = "pi"
    K_str = "1b"
    threshold = 1e-8
    by_test = False

    df_dqfim = maybe_rebuild_or_process_dqfim(
        base_path=base_path,
        sample_range=sample_range,
        model_type=model_type,
        N_ctrls=N_ctrls,
        K_0=K_str,
        sample_label=sample_range,
        num_L=num_L,
        threshold=threshold,
        by_test=by_test
    )

    print("[INFO] df_dqfim shape after reading or scanning DQFIM directories:", df_dqfim.shape)
    print(df_dqfim.head())
    print(f"df_dqfim columns: {df_dqfim.columns}")