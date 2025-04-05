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
            qfim_mat = results.get('qfim', None)  # NxN
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

###############################################################################
# 3) process_and_cache_new_files_dqfim
###############################################################################
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
    Scan 'DQFIM_results' directories for new data.pickle files not yet in 'cached_data'.
    For each new file, parse it with 'process_data_dqfim', store the results 
    in 'cached_data', and return a list of all newly processed rows in a DataFrame.

    This is analogous to 'process_and_cache_new_files' in qfim_store.py, 
    but specialized for DQFIM.
    """
    from pathlib import Path
    import pandas as pd

    all_data = []

    for N_ctrl in N_ctrls:
        model_path = Path(base_path) / "DQFIM_results" / model_type / f"Nc_{N_ctrl}" / f"sample_{sample_range}" / f"{K_0}xK"

        if not model_path.exists():
            print(f"[WARN] DQFIM directory {model_path} does not exist for N_ctrl={N_ctrl}.")
            raise ValueError

        # Walk through e.g. Nr_1 / trotter_step_10, etc.
        for Nr_dir in sorted(os.listdir(model_path)):
            Nr_path = model_path / Nr_dir
            if not Nr_path.is_dir():
                print(f"[WARN] DQFIM directory {model_path} does not exist for N_ctrl={N_ctrl}, N_R={Nr_dir}")
                continue

            # Example: 'Nr_2' => parse to get n_reserv=2 if needed:
            try:
                n_reserv = int(Nr_dir.split('_')[-1])
            except:
                n_reserv = None

            for trot_dir in sorted(os.listdir(Nr_path)):
                trot_path = Nr_path / trot_dir
                if not trot_path.is_dir():
                    print(f"[WARN] DQFIM directory {model_path} does not exist for N_ctrl={N_ctrl}, N_R={Nr_dir}, T={trot_dir}")
                    continue

                # e.g. 'step_10' => parse trotter steps
                try:
                    trotter_steps = int(trot_dir.split('_')[-1])
                except:
                    trotter_steps = None

                data_file = trot_path / f"L_{num_L}/data.pickle"
                file_id = str(data_file)

                if file_id in cached_data:
                    # Already processed
                    if check_for_new_data:
                        print(f"[INFO] Checking for new data in cached file: {file_id}")
                        updated = update_cached_data_dqfim(
                            data_file, cached_data, processed_files, 
                            N_ctrl, threshold, n_reserv, trotter_steps
                        )
                        if updated:
                            all_data.append(cached_data[file_id]["processed_data"])
                    else:
                        # No "update" check => just re-append existing data
                        all_data.append(cached_data[file_id]["processed_data"])
                    continue

                # Not in cache => process it
                if is_valid_pickle_file(data_file):
                    print(f"[INFO] Found NEW DQFIM file: {data_file}")
                    raw_df = load_and_clean_pickle(data_file)

                    row_dicts = process_data_dqfim(
                        raw_df, 
                        n_ctrl=N_ctrl, 
                        n_reserv=n_reserv, 
                        trotter_steps=trotter_steps,
                        threshold=threshold
                    )

                    # We'll store row_dicts as the "processed_data"
                    cached_data[file_id] = {
                        "processed_data": row_dicts,
                        "raw_keys": get_keys(raw_df),
                    }
                    processed_files.add(file_id)
                    all_data.append(row_dicts)

        # After scanning all directories for this N_ctrl, save the updated cache
        save_cached_data_dqfim(base_path, model_type, cached_data, processed_files, 
                               N_ctrl, K_0, sample_range, num_L=num_L)

    # Flatten all_data (since each entry might be a list of rows)
    flattened = []
    for block in all_data:
        flattened.extend(block)
    df_all = pd.DataFrame(flattened)
    return cached_data, processed_files, df_all


###############################################################################
# 4) update_cached_data_dqfim: Re-check an existing file for new fixed_param/test_key
###############################################################################
def update_cached_data_dqfim(data_file, cached_data, processed_files, 
                             N_ctrl, threshold, n_reserv, trotter_steps):
    """
    Load the data_file, compare its structure to cached_data[file_id]["raw_keys"].
    If new fixed_params or test_keys appear, re-run process_data_dqfim and update.

    Returns True if updated, False if no change.
    """
    file_id = str(data_file)
    raw_df = load_and_clean_pickle(data_file)
    new_keys = get_keys(raw_df)
    old_keys = cached_data[file_id].get("raw_keys", {})

    # Check for new fixed_param or new test_key
    needs_update = False
    for fixed_param, tests in new_keys.items():
        if fixed_param not in old_keys:
            needs_update = True
            break
        else:
            # if any new test_keys appear
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


###############################################################################
# 5) update_cache_with_new_data_dqfim
###############################################################################
def update_cache_with_new_data_dqfim(base_path, 
                                     K_0, 
                                     sample_range, 
                                     model_type, 
                                     N_ctrl, 
                                     threshold, 
                                     by_test,
                                     cached_data, 
                                     processed_files):
    """
    For a single N_ctrl, look for new data.pickle files in the 
    'DQFIM_results/{model_type}/Nc_{N_ctrl}/...' directory 
    that are NOT yet in cached_data. If new, process them.
    If existing, check for updated keys (via update_cached_data_dqfim).

    Return updated (cached_data, processed_files) plus a DataFrame
    of newly updated rows.
    """
    import pandas as pd
    from pathlib import Path

    all_new_rows = []

    model_path = Path(base_path) / "DQFIM_results" / model_type / f"Nc_{N_ctrl}" / f"sample_{sample_range}" / f"{K_0}xK"
    if not model_path.exists():
        print(f"[WARN] DQFIM directory {model_path} does not exist. No new data found.")
        return cached_data, processed_files, pd.DataFrame()

    for Nr_dir in sorted(os.listdir(model_path)):
        Nr_path = model_path / Nr_dir
        if not Nr_path.is_dir():
            continue
        try:
            n_reserv = int(Nr_dir.split('_')[-1])
        except:
            n_reserv = None

        for trot_dir in sorted(os.listdir(Nr_path)):
            trot_path = Nr_path / trot_dir
            if not trot_path.is_dir():
                continue
            try:
                trotter_steps = int(trot_dir.split('_')[-1])
            except:
                trotter_steps = None

            data_file = trot_path / "data.pickle"
            file_id = str(data_file)

            if file_id in cached_data:
                print(f"[INFO] Checking cached DQFIM file: {data_file}")
                updated = update_cached_data_dqfim(
                    data_file, 
                    cached_data, 
                    processed_files,
                    N_ctrl,
                    threshold,
                    n_reserv,
                    trotter_steps
                )
                # If updated is True => new row(s) were added/changed
                if updated:
                    all_new_rows.append(cached_data[file_id]["processed_data"])
                continue

            # If not in cache
            if is_valid_pickle_file(data_file):
                print(f"[INFO] Found NEW DQFIM data file: {data_file}")
                raw_df = load_and_clean_pickle(data_file)
                row_dicts = process_data_dqfim(
                    raw_df,
                    n_ctrl=N_ctrl,
                    n_reserv=n_reserv,
                    trotter_steps=trotter_steps,
                    threshold=threshold
                )
                cached_data[file_id] = {
                    "processed_data": row_dicts,
                    "raw_keys": get_keys(raw_df)
                }
                processed_files.add(file_id)
                all_new_rows.append(row_dicts)
    save_cached_data_dqfim(base_path, model_type, cached_data, processed_files, 
                               N_ctrl, K_0, sample_range, num_L=num_L)
    if all_new_rows:
        save_cached_data_dqfim(base_path, model_type, cached_data, processed_files, 
                               N_ctrl, K_0, sample_range)

    # Flatten
    flattened = []
    for block in all_new_rows:
        flattened.extend(block)
    df_new = pd.DataFrame(flattened)
    return cached_data, processed_files, df_new


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
    1) For each N_ctrl in N_ctrls, try loading an existing DQFIM cache
       via get_cached_data_dqfim.
    2) If no cache is found, we run process_and_cache_new_files_dqfim 
       to build it from scratch.
    3) Else, we run update_cache_with_new_data_dqfim to see if there's 
       new or changed files in the directory that are not in the cache.
    4) Combine all partial DataFrames into a single df_all.
    """
    import pandas as pd

    all_frames = []
    for N_ctrl in N_ctrls:
        cached_data, processed_files = get_cached_data_dqfim(
            base_path, model_type, N_ctrl, K_0, sample_label,num_L
        )

        if not cached_data:
            # No cache => process everything from scratch
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
            # We do have a cache => see if new data has arrived
            cached_data, processed_files, df_new = update_cache_with_new_data_dqfim(
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
            # df_new is newly added. We also want the old data
            local_rows = []
            for fid, fdict in cached_data.items():
                local_rows.extend(fdict["processed_data"])
            df_partial = pd.DataFrame(local_rows)
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
    num_L=20
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