import numpy as np
import sympy
import matplotlib.pyplot as plt
import base64
import pickle
from sympy import symbols, MatrixSymbol, lambdify, Matrix, pprint

from scipy.optimize import minimize
from matplotlib.ticker import FuncFormatter
from sympy import symbols, MatrixSymbol, lambdify
from matplotlib import cm
import random
import matplotlib.colors as mcolors
import scipy
import time
from pathlib import Path
import os
import ast
import pandas as pd
from pathlib import Path
from matplotlib.ticker import ScalarFormatter


import pennylane as qml
from functools import partial
from qiskit.circuit.library import *
from qiskit import *
from qiskit.quantum_info import *
import autograd
from pennylane.wires import Wires
import matplotlib.cm as cm
import base64
from qiskit import *
from qiskit.quantum_info import *
import os
import pickle
import re
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
global_cache_data_analog = None
global_processed_files_analog = None
global_cache_data_digital = None
global_processed_files_digital = None
import os
import pickle
import re
import numpy as np
from pathlib import Path


def is_valid_pickle_file(file_path):
    try:
        if file_path.exists() and file_path.stat().st_size > 0:
            with open(file_path, 'rb') as f:
                # Attempt to load the pickle file
                df = pickle.load(f)
                
            return True
        else:
            return False
    except EOFError:
        return False
def extract_last_number(text):
    numbers = re.findall(r'\d+', text)
    return int(numbers[-1]) if numbers else 0

def read_jax_file_digital(file_path, gate_name):
    with open(file_path, 'rb') as f:
        df = pickle.load(f)
        df = clean_array(df)
        costs = np.asarray([float(i) for i in df['costs'][0]], dtype=np.float64)
        grads_per_epoch = [np.asarray(i, dtype=np.float64) for i in df['grads_per_epoch'][0]]

        fidelity =df['avg_fidelity'][0]
        num_params = 3 + int(df['controls'][0]) * int(df['reservoirs'][0]) * int(df['trotter_step'][0])
        try:
            test_results =  np.asarray(df['fidelities'][0], dtype=np.float64)
        except KeyError:
            
            test_results =  np.asarray(df['testing_results'][0], dtype=np.float64)
        num_epochs = df['epochs'][0]
        return costs, fidelity, num_params, test_results,grads_per_epoch



def read_jax_file_analog(file_path, gate_name):
    with open(file_path, 'rb') as f:
        df = pickle.load(f)
        
    # Clean any deprecated arrays in the data
    df = clean_array(df)
    with open(file_path, 'wb') as f:
        
        pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
    costs = [float(i) for i in df['costs'][0]]
    fidelity = df['avg_fidelity'][0]
    test_results = [float(a) for a in df['testing_results'][0]]
    num_params = 3 + int(df['trotter_step'][0]) + int(df['controls'][0]) * int(df['reservoirs'][0]) * int(df['trotter_step'][0])
    grads_per_epoch = df['grads_per_epoch'][0]
    try:
        selected_indices = df['selected_indices'][0]
    except KeyError:
        selected_indices = None
    return costs, fidelity, num_params, test_results, grads_per_epoch, selected_indices

def clean_array(data):
    """Helper function to clean any deprecated JAX arrays."""
    if isinstance(data, np.ndarray):
        return np.array(data)  # Ensure the array doesn't have deprecated attributes
    elif isinstance(data, dict):
        return {k: clean_array(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_array(v) for v in data]
    else:
        return data
    
def get_cached_data_once_digital(base_path, N_ctrl = None):
    """Load cached data only once and keep it in memory for future runs for the digital model."""
    global global_cache_data_digital, global_processed_files_digital

    # If the cache has already been loaded, skip re-loading
    if global_cache_data_digital is not None and global_processed_files_digital is not None:
        return global_cache_data_digital, global_processed_files_digital

    # Load the cache from disk (only happens once)
    cache_file = os.path.join(base_path, 'cached_results.pkl')
    if os.path.exists(cache_file):
        if os.stat(cache_file).st_size == 0:
            print(f"[ERROR] Cache file {cache_file} is empty. Returning empty cache.")
            global_cache_data_digital = {}
            global_processed_files_digital = set()
            return global_cache_data_digital, global_processed_files_digital

        try:
            with open(cache_file, 'rb') as f:
                global_cache_data_digital, global_processed_files_digital = pickle.load(f)
            global_cache_data_digital = clean_array(global_cache_data_digital)

            # Filter the cache to only include files from the correct base path
            global_processed_files_digital = set(
                file for file in global_processed_files_digital if file.startswith(base_path)
            )
            return global_cache_data_digital, global_processed_files_digital
        except (EOFError, pickle.UnpicklingError) as e:
            print(f"[ERROR] Failed to load cache file {cache_file}: {e}")
            global_cache_data_digital = {}
            global_processed_files_digital = set()
            return global_cache_data_digital, global_processed_files_digital
    else:
        print(f"[INFO] Cache file {cache_file} does not exist. Creating empty cache.")
        global_cache_data_digital = {}
        global_processed_files_digital = set()
        return global_cache_data_digital, global_processed_files_digital

def get_cached_data_once_analog(base_path, N_ctrl=None):
    """Load cached data only once and keep it in memory for future runs for the analog model."""
    global global_cache_data_analog, global_processed_files_analog

    # If the cache has already been loaded, skip re-loading
    if global_cache_data_analog is not None and global_processed_files_analog is not None:
        return global_cache_data_analog, global_processed_files_analog

    # Load the cache from disk (only happens once)
    cache_file = os.path.join(base_path, 'cached_results.pkl')
    if os.path.exists(cache_file):
        if os.stat(cache_file).st_size == 0:
            print(f"[ERROR] Cache file {cache_file} is empty. Returning empty cache.")
            global_cache_data_analog = {}
            global_processed_files_analog = set()
            return global_cache_data_analog, global_processed_files_analog

        try:
            with open(cache_file, 'rb') as f:
                global_cache_data_analog, global_processed_files_analog = pickle.load(f)
            global_cache_data_analog = clean_array(global_cache_data_analog)

            # Filter the cache to only include files from the correct base path
            global_processed_files_analog = set(
                file for file in global_processed_files_analog if file.startswith(base_path)
            )
            return global_cache_data_analog, global_processed_files_analog
        except (EOFError, pickle.UnpicklingError) as e:
            print(f"[ERROR] Failed to load cache file {cache_file}: {e}")
            global_cache_data_analog = {}
            global_processed_files_analog = set()
            return global_cache_data_analog, global_processed_files_analog
    else:
        print(f"[INFO] Cache file {cache_file} does not exist. Creating empty cache.")
        global_cache_data_analog = {}
        global_processed_files_analog = set()
        return global_cache_data_analog, global_processed_files_analog
def clean_cached_data_with_missing_paths(cached_data):
    """
    Remove entries from the cached data where 'path' is None.
    """
    for N_ctrl, gates in cached_data.items():
        for gate, reservoirs in gates.items():
            for reservoir_count, trotter_steps in reservoirs.items():
                for trotter_step, data_points in trotter_steps.items():
                    valid_data_points = [dp for dp in data_points if dp.get('path') is not None]
                    cached_data[N_ctrl][gate][reservoir_count][trotter_step] = valid_data_points
    return cached_data


def retrofit_cached_data_with_path(base_path, cached_data):
    """
    Adds the 'path' key to cached data if it doesn't exist.
    """
    for N_ctrl, gates in cached_data.items():
        for gate, reservoirs in gates.items():
            for reservoir_count, trotter_steps in reservoirs.items():
                for trotter_step, data_points in trotter_steps.items():
                    for data_point in data_points:
                        if 'path' not in data_point:
                            # Attempt to reconstruct the file path
                            trotter_path = os.path.join(
                                base_path,
                                gate,
                                f"reservoirs_{reservoir_count}",
                                f"trotter_step_{trotter_step}",
                            )
                            # Example logic: Assume file names can be inferred from 'run' key
                            if 'run' in data_point:
                                file_name = f"{data_point['run']}.pickle"
                                file_path = os.path.join(trotter_path, file_name)
                                if os.path.exists(file_path):
                                    data_point['path'] = file_path
                                else:
                                    data_point['path'] = None  # Mark as unknown if file is missing
                            else:
                                data_point['path'] = None  # Mark as unknown if 'run' is missing
    return cached_data

def save_cached_data(base_path, cached_data, processed_files):
    """Save cached data and list of processed files to cache stored in the base_path."""
    cache_file = os.path.join(base_path, f'cached_results.pkl')
    with open(cache_file, 'wb') as f:
        pickle.dump((cached_data, processed_files), f)

def process_new_files_digital(base_path, gate_prefixes, reservoir_counts, trots, cached_data, processed_files, N_ctrl):
    """
    Process or re-check .pickle files in directories matching:
      base_path/gate_prefix_xxx/reservoirs_YYY/trotter_step_ZZZ/{bath_True,bath_False}/
    for the specified N_ctrl, and update the cache with the best fidelity found.

    If the folder's actual file count differs from the cached 'num_data_runs',
    we re-check all files to find the best run again. Otherwise, we only check 
    new (unprocessed) files.

    Structure of `cached_data`:
      cached_data[N_ctrl][gate][reservoir_count][trotter_step] = [
         {
           'costs': ...,
           'gate': gate,
           'fidelity': ...,
           'test_results': ...,
           'param_count': ...,
           'run': 'data_run_<i>',
           'num_data_runs': ...,
           'grads_per_epoch': ...,
           'path': 'full_file_path.pickle'
         }
      ]

    Parameters
    ----------
    base_path : str
        Base folder path for the runs.
    gate_prefixes : list[str]
        E.g., ['U2', 'U3'] for 2- or 3-qubit digital gates.
    reservoir_counts : list[int]
        Reservoir sizes to include.
    trots : list[int]
        Trotter steps to include.
    cached_data : dict
        Existing cache dictionary.
    processed_files : set
        Paths of files already processed.
    N_ctrl : int
        Control qubit dimension.

    Returns
    -------
    (cached_data, processed_files)
      The updated cache and set of processed files.
    """
    print(f"[process_new_files_digital] Processing for N_ctrl = {N_ctrl}")
    for gate_prefix in gate_prefixes:
        for folder_name in sorted(os.listdir(base_path)):
            if folder_name.startswith(gate_prefix + "_"):
                gate = folder_name

                # Ensure the top-level structure
                if N_ctrl not in cached_data:
                    cached_data[N_ctrl] = {}
                if gate not in cached_data[N_ctrl]:
                    cached_data[N_ctrl][gate] = {}

                # We'll check bath_True and bath_False/ subfolders
                for bath_status in ['bath_True', 'bath_False/']:
                    subfolder_path = os.path.join(base_path, gate)
                    if not os.path.isdir(subfolder_path):
                        continue

                    # For each "reservoirs_*"
                    for subfolder in sorted(os.listdir(subfolder_path), key=extract_last_number):
                        if 'reservoirs_' in subfolder:
                            reservoir_count = extract_last_number(subfolder)
                            if reservoir_count not in reservoir_counts:
                                continue

                            full_res_path = os.path.join(subfolder_path, subfolder)
                            if not os.path.isdir(full_res_path):
                                continue

                            # For each "trotter_step_*"
                            for trotter_folder in sorted(os.listdir(full_res_path), key=extract_last_number):
                                if 'trotter_step_' in trotter_folder:
                                    trotter_step = extract_last_number(trotter_folder)
                                    if trotter_step not in trots:
                                        continue

                                    trotter_path = os.path.join(full_res_path, trotter_folder, bath_status)
                                    if not os.path.exists(trotter_path) or not os.path.isdir(trotter_path):
                                        continue

                                    files_in_folder = [
                                        f for f in os.listdir(trotter_path)
                                        if not f.startswith('.') and f.endswith('.pickle')
                                    ]
                                    num_files_found = len(files_in_folder)
                                    print(f"[process_new_files_digital] {gate}, N_c={N_ctrl}, N_R={reservoir_count}, "
                                          f"T={trotter_step}, num files(runs)= {num_files_found}")

                                    # Ensure we have the nested dict
                                    if reservoir_count not in cached_data[N_ctrl][gate]:
                                        cached_data[N_ctrl][gate][reservoir_count] = {}
                                    if trotter_step not in cached_data[N_ctrl][gate][reservoir_count]:
                                        cached_data[N_ctrl][gate][reservoir_count][trotter_step] = []

                                    cached_list = cached_data[N_ctrl][gate][reservoir_count][trotter_step]
                                    previous_best = cached_list[0] if cached_list else None

                                    previous_best_fidelity = float('-inf')
                                    cached_num_runs = 0
                                    if previous_best:
                                        previous_best_fidelity = previous_best.get('fidelity', float('-inf'))
                                        cached_num_runs = previous_best.get('num_data_runs', 0)

                                    # Decide if we re-check all or just new
                                    recheck_all = (previous_best is None) or (cached_num_runs != num_files_found)

                                    best_fidelity_overall = previous_best_fidelity
                                    best_data_point_overall = previous_best
                                    runs_counted = 0

                                    if recheck_all:
                                        # Re-check all files in the folder
                                        for f in files_in_folder:
                                            pickle_file = os.path.normpath(os.path.join(trotter_path, f))
                                            if is_valid_pickle_file(Path(pickle_file)):
                                                costs, fidelity, num_params, test_results, grads_per_epoch = \
                                                    read_jax_file_digital(pickle_file, gate)
                                                runs_counted += 1
                                                run_name = os.path.splitext(os.path.basename(f))[0]

                                                if fidelity > best_fidelity_overall:
                                                    best_fidelity_overall = fidelity
                                                    best_data_point_overall = {
                                                        'costs': costs,
                                                        'gate': gate,
                                                        'fidelity': fidelity,
                                                        'test_results': test_results,
                                                        'param_count': num_params,
                                                        'run': run_name,
                                                        'num_data_runs': runs_counted,
                                                        'grads_per_epoch': grads_per_epoch,
                                                        'path': pickle_file
                                                    }
                                                processed_files.add(pickle_file)

                                        # After scanning all files, update cache if we found a best
                                        if best_data_point_overall:
                                            best_data_point_overall['num_data_runs'] = runs_counted
                                            cached_data[N_ctrl][gate][reservoir_count][trotter_step] = [best_data_point_overall]

                                    else:
                                        # Only check new/unprocessed files
                                        runs_counted = cached_num_runs
                                        for f in files_in_folder:
                                            pickle_file = os.path.normpath(os.path.join(trotter_path, f))
                                            if pickle_file not in processed_files and is_valid_pickle_file(Path(pickle_file)):
                                                costs, fidelity, num_params, test_results, grads_per_epoch = \
                                                    read_jax_file_digital(pickle_file, gate)
                                                runs_counted += 1
                                                run_name = os.path.splitext(os.path.basename(f))[0]

                                                if fidelity > best_fidelity_overall:
                                                    best_fidelity_overall = fidelity
                                                    best_data_point_overall = {
                                                        'costs': costs,
                                                        'gate': gate,
                                                        'fidelity': fidelity,
                                                        'test_results': test_results,
                                                        'param_count': num_params,
                                                        'run': run_name,
                                                        'num_data_runs': runs_counted,
                                                        'grads_per_epoch': grads_per_epoch,
                                                        'path': pickle_file
                                                    }
                                                processed_files.add(pickle_file)

                                        # If we found a better run among new files, update the cache
                                        if best_data_point_overall and best_data_point_overall is not previous_best:
                                            best_data_point_overall['num_data_runs'] = runs_counted
                                            cached_data[N_ctrl][gate][reservoir_count][trotter_step] = [best_data_point_overall]

    return cached_data, processed_files


def update_cache_with_new_data_digital(base_path, gate_prefixes, reservoir_counts, trots, cached_data, processed_files, N_ctrl):
    """
    Similar to process_new_files_digital, but focuses on partial or incremental updates.
    If the # of actual files in a folder != 'num_data_runs', it re-checks everything.
    Otherwise, it just checks new files. This way we always keep the best run in cache.

    Structure of `cached_data` is the same as in process_new_files_digital.
    """
    print(f"[update_cache_with_new_data_digital] Updating cache for N_ctrl = {N_ctrl}")
    for gate_prefix in gate_prefixes:
        for folder_name in sorted(os.listdir(base_path)):
            if folder_name.startswith(gate_prefix + "_"):
                gate = folder_name

                if N_ctrl not in cached_data:
                    cached_data[N_ctrl] = {}
                if gate not in cached_data[N_ctrl]:
                    cached_data[N_ctrl][gate] = {}

                for bath_status in ['bath_True', 'bath_False/']:
                    gate_path = os.path.join(base_path, gate)
                    if not os.path.isdir(gate_path):
                        continue

                    for subfolder in sorted(os.listdir(gate_path), key=extract_last_number):
                        if 'reservoirs_' in subfolder:
                            reservoir_count = extract_last_number(subfolder)
                            if reservoir_count not in reservoir_counts:
                                continue

                            reservoir_path = os.path.join(gate_path, subfolder)
                            if not os.path.isdir(reservoir_path):
                                continue

                            for trotter_folder in sorted(os.listdir(reservoir_path), key=extract_last_number):
                                if 'trotter_step_' in trotter_folder:
                                    trotter_step = extract_last_number(trotter_folder)
                                    if trotter_step not in trots:
                                        continue

                                    trotter_path = os.path.join(reservoir_path, trotter_folder, bath_status)
                                    if not os.path.isdir(trotter_path):
                                        continue

                                    files_in_folder = [
                                        f for f in os.listdir(trotter_path)
                                        if not f.startswith('.') and f.endswith('.pickle')
                                    ]
                                    num_files_found = len(files_in_folder)
                                    print(f"[update_cache_with_new_data_digital] {gate}, N_c={N_ctrl}, "
                                          f"N_R={reservoir_count}, T={trotter_step}, num files= {num_files_found}")

                                    if reservoir_count not in cached_data[N_ctrl][gate]:
                                        cached_data[N_ctrl][gate][reservoir_count] = {}
                                    if trotter_step not in cached_data[N_ctrl][gate][reservoir_count]:
                                        cached_data[N_ctrl][gate][reservoir_count][trotter_step] = []

                                    cached_list = cached_data[N_ctrl][gate][reservoir_count][trotter_step]
                                    previous_best = cached_list[0] if cached_list else None

                                    previous_best_fidelity = float('-inf')
                                    cached_num_runs = 0
                                    if previous_best:
                                        previous_best_fidelity = previous_best.get('fidelity', float('-inf'))
                                        cached_num_runs = previous_best.get('num_data_runs', 0)

                                    # Decide if we must re-check all
                                    recheck_all = (previous_best is None) or (cached_num_runs != num_files_found)

                                    best_fidelity_overall = previous_best_fidelity
                                    best_data_point_overall = previous_best
                                    runs_counted = 0

                                    if recheck_all:
                                        # Re-check all files in the folder
                                        for f in files_in_folder:
                                            pickle_file = os.path.normpath(os.path.join(trotter_path, f))
                                            if is_valid_pickle_file(Path(pickle_file)):
                                                runs_counted += 1
                                                costs, fidelity, num_params, test_results, grads_per_epoch = \
                                                    read_jax_file_digital(pickle_file, gate)

                                                run_name = os.path.splitext(os.path.basename(f))[0]
                                                if fidelity > best_fidelity_overall:
                                                    best_fidelity_overall = fidelity
                                                    best_data_point_overall = {
                                                        'costs': costs,
                                                        'gate': gate,
                                                        'fidelity': fidelity,
                                                        'test_results': test_results,
                                                        'param_count': num_params,
                                                        'run': run_name,
                                                        'num_data_runs': runs_counted,
                                                        'grads_per_epoch': grads_per_epoch,
                                                        'path': pickle_file
                                                    }
                                                processed_files.add(pickle_file)

                                        if best_data_point_overall:
                                            best_data_point_overall['num_data_runs'] = runs_counted
                                            cached_data[N_ctrl][gate][reservoir_count][trotter_step] = [best_data_point_overall]

                                    else:
                                        # Only check new/unprocessed files
                                        runs_counted = cached_num_runs
                                        for f in files_in_folder:
                                            pickle_file = os.path.normpath(os.path.join(trotter_path, f))
                                            if (pickle_file not in processed_files) and is_valid_pickle_file(Path(pickle_file)):
                                                runs_counted += 1
                                                costs, fidelity, num_params, test_results, grads_per_epoch = \
                                                    read_jax_file_digital(pickle_file, gate)
                                                run_name = os.path.splitext(os.path.basename(f))[0]

                                                if fidelity > best_fidelity_overall:
                                                    best_fidelity_overall = fidelity
                                                    best_data_point_overall = {
                                                        'costs': costs,
                                                        'gate': gate,
                                                        'fidelity': fidelity,
                                                        'test_results': test_results,
                                                        'param_count': num_params,
                                                        'run': run_name,
                                                        'num_data_runs': runs_counted,
                                                        'grads_per_epoch': grads_per_epoch,
                                                        'path': pickle_file
                                                    }
                                                processed_files.add(pickle_file)

                                        if best_data_point_overall and best_data_point_overall is not previous_best:
                                            best_data_point_overall['num_data_runs'] = runs_counted
                                            cached_data[N_ctrl][gate][reservoir_count][trotter_step] = [best_data_point_overall]

    print(f"[update_cache_with_new_data_digital] Cache updated for N_ctrl={N_ctrl}")
    return cached_data, processed_files
def process_new_files_analog(base_path, gate_prefixes, reservoir_counts, trots, cached_data, processed_files, N_ctrl):
    """
    Process or re-check .pickle files in directories matching:
      base_path/gate_prefix_xxx/reservoirs_YYY/trotter_step_ZZZ/{bath_True,bath_False/}
    for the specified N_ctrl, and update the cache with the best fidelity found.

    Structure of `cached_data`:
      cached_data[N_ctrl][gate][reservoir_count][trotter_step] = [
         {
           'costs': ...,
           'gate': gate,
           'fidelity': ...,
           'test_results': ...,
           'param_count': ...,
           'run': 'data_run_<i>',
           'num_data_runs': ...,
           'grads_per_epoch': ...,
           'path': 'full_file_path.pickle'
         }
      ]

    Parameters
    ----------
    base_path : str
        Base folder path for the runs.
    gate_prefixes : list[str]
        E.g., ['U2', 'U3'] for 2- or 3-qubit analog gates.
    reservoir_counts : list[int]
        Reservoir sizes to include.
    trots : list[int]
        Trotter steps to include.
    cached_data : dict
        Existing cache dictionary.
    processed_files : set
        Paths of files already processed.
    N_ctrl : int
        Control qubit dimension.

    Returns
    -------
    (cached_data, processed_files)
      The updated cache and set of processed files.
    """
    print(f"[process_new_files_analog] Processing for N_ctrl = {N_ctrl}")
    for gate_prefix in gate_prefixes:
        for folder_name in sorted(os.listdir(base_path)):
            if folder_name.startswith(gate_prefix + "_"):
                gate = folder_name

                # Ensure the top-level structure
                if N_ctrl not in cached_data:
                    cached_data[N_ctrl] = {}
                if gate not in cached_data[N_ctrl]:
                    cached_data[N_ctrl][gate] = {}

                # We'll check bath_True and bath_False/ subfolders
                for bath_status in ['bath_True', 'bath_False/']:
                    subfolder_path = os.path.join(base_path, gate)
                    if not os.path.isdir(subfolder_path):
                        continue

                    # For each "reservoirs_*"
                    for subfolder in sorted(os.listdir(subfolder_path), key=extract_last_number):
                        if 'reservoirs_' in subfolder:
                            reservoir_count = extract_last_number(subfolder)
                            if reservoir_count not in reservoir_counts:
                                continue

                            full_res_path = os.path.join(subfolder_path, subfolder)
                            if not os.path.isdir(full_res_path):
                                continue

                            # For each "trotter_step_*"
                            for trotter_folder in sorted(os.listdir(full_res_path), key=extract_last_number):
                                if 'trotter_step_' in trotter_folder:
                                    trotter_step = extract_last_number(trotter_folder)
                                    if trotter_step not in trots:
                                        continue

                                    trotter_path = os.path.join(full_res_path, trotter_folder, bath_status)
                                    if not os.path.exists(trotter_path) or not os.path.isdir(trotter_path):
                                        continue

                                    files_in_folder = [
                                        f for f in os.listdir(trotter_path)
                                        if not f.startswith('.') and f.endswith('.pickle')
                                    ]
                                    num_files_found = len(files_in_folder)
                                    print(f"[process_new_files_analog] {gate}, N_c={N_ctrl}, N_R={reservoir_count}, "
                                          f"T={trotter_step}, num files(runs)= {num_files_found}")

                                    # Ensure we have the nested dict
                                    if reservoir_count not in cached_data[N_ctrl][gate]:
                                        cached_data[N_ctrl][gate][reservoir_count] = {}
                                    if trotter_step not in cached_data[N_ctrl][gate][reservoir_count]:
                                        cached_data[N_ctrl][gate][reservoir_count][trotter_step] = []

                                    cached_list = cached_data[N_ctrl][gate][reservoir_count][trotter_step]
                                    previous_best = cached_list[0] if cached_list else None

                                    previous_best_fidelity = float('-inf')
                                    cached_num_runs = 0
                                    if previous_best:
                                        previous_best_fidelity = previous_best.get('fidelity', float('-inf'))
                                        cached_num_runs = previous_best.get('num_data_runs', 0)

                                    # Decide if we re-check all or just new
                                    recheck_all = (previous_best is None) or (cached_num_runs != num_files_found)

                                    best_fidelity_overall = previous_best_fidelity
                                    best_data_point_overall = previous_best
                                    runs_counted = 0

                                    if recheck_all:
                                        # Re-check all files in the folder
                                        for f in files_in_folder:
                                            pickle_file = os.path.normpath(os.path.join(trotter_path, f))
                                            if is_valid_pickle_file(Path(pickle_file)):
                                                costs, fidelity, num_params, test_results, grads_per_epoch = \
                                                    read_jax_file_analog(pickle_file, gate)
                                                runs_counted += 1
                                                run_name = os.path.splitext(os.path.basename(f))[0]

                                                if fidelity > best_fidelity_overall:
                                                    best_fidelity_overall = fidelity
                                                    best_data_point_overall = {
                                                        'costs': costs,
                                                        'gate': gate,
                                                        'fidelity': fidelity,
                                                        'test_results': test_results,
                                                        'param_count': num_params,
                                                        'run': run_name,
                                                        'num_data_runs': runs_counted,
                                                        'grads_per_epoch': grads_per_epoch,
                                                        'path': pickle_file
                                                    }
                                                processed_files.add(pickle_file)

                                        # After scanning all files, update cache if we found a best
                                        if best_data_point_overall:
                                            best_data_point_overall['num_data_runs'] = runs_counted
                                            cached_data[N_ctrl][gate][reservoir_count][trotter_step] = [best_data_point_overall]

                                    else:
                                        # Only check new/unprocessed files
                                        runs_counted = cached_num_runs
                                        for f in files_in_folder:
                                            pickle_file = os.path.normpath(os.path.join(trotter_path, f))
                                            if pickle_file not in processed_files and is_valid_pickle_file(Path(pickle_file)):
                                                costs, fidelity, num_params, test_results, grads_per_epoch = \
                                                    read_jax_file(pickle_file, gate)
                                                runs_counted += 1
                                                run_name = os.path.splitext(os.path.basename(f))[0]

                                                if fidelity > best_fidelity_overall:
                                                    best_fidelity_overall = fidelity
                                                    best_data_point_overall = {
                                                        'costs': costs,
                                                        'gate': gate,
                                                        'fidelity': fidelity,
                                                        'test_results': test_results,
                                                        'param_count': num_params,
                                                        'run': run_name,
                                                        'num_data_runs': runs_counted,
                                                        'grads_per_epoch': grads_per_epoch,
                                                        'path': pickle_file
                                                    }
                                                processed_files.add(pickle_file)

                                        # If we found a better run among new files, update the cache
                                        if best_data_point_overall and best_data_point_overall is not previous_best:
                                            best_data_point_overall['num_data_runs'] = runs_counted
                                            cached_data[N_ctrl][gate][reservoir_count][trotter_step] = [best_data_point_overall]

    return cached_data, processed_files



def update_cache_with_new_data_analog(base_path, gate_prefixes, reservoir_counts, trots, cached_data, processed_files, N_ctrl):
    """Update cache with new key/values without reprocessing already processed files."""
    print(f"Processing for N_ctrl = {N_ctrl}")
    
    for gate_prefix in gate_prefixes:
        for folder_name in sorted(os.listdir(base_path)):
            if folder_name.startswith(gate_prefix + "_"):
                gate = folder_name
                # print(f"Processing gate: {gate}")

                for bath_status in ['bath_True', 'bath_False']:
                    for subfolder in sorted(os.listdir(os.path.join(base_path, gate)), key=extract_last_number):
                        if 'reservoirs_' in subfolder:
                            reservoir_count = extract_last_number(subfolder)
                            if reservoir_count not in reservoir_counts:
                                continue
                            # print(f"Processing reservoir: {reservoir_count}")

                            for trotter_folder in sorted(os.listdir(os.path.join(base_path, gate, subfolder)), key=extract_last_number):
                                if 'trotter_step_' in trotter_folder:
                                    trotter_step = extract_last_number(trotter_folder)
                                    if trotter_step not in trots:
                                        continue
                                    # print(f"Processing trotter step: {trotter_step}")

                                    trotter_path = os.path.join(base_path, gate, subfolder, trotter_folder, bath_status)
                                    if not os.path.exists(trotter_path):
                                        continue

                                    files_in_folder = os.listdir(trotter_path)
                                    
                                    # Ensure that N_ctrl is in cached_data
                                    if N_ctrl not in cached_data:
                                        cached_data[N_ctrl] = {}

                                    # Initialize the cached_trotter_data
                                    cached_trotter_data = cached_data.get(N_ctrl, {}).get(gate, {}).get(reservoir_count, {}).get(trotter_step, [])
                                    num_data_runs = len(cached_trotter_data)

                                    for file in files_in_folder:
                                        if not file.startswith('.'):
                                            pickle_file = os.path.normpath(os.path.join(trotter_path, file))

                                            # Extract just the data_run_<i> part from the file name
                                            run = os.path.basename(pickle_file).replace('.pickle', '')

                                            # Check if the file has already been processed
                                            if pickle_file in processed_files:
                                                # print(f"Skipping file {pickle_file}, already processed.")
                                                # Update the run field in cached data if missing
                                                for cached_result in cached_trotter_data:
                                                    if 'run' not in cached_result or cached_result['run'] != run:
                                                        cached_result['run'] = run
                                                    if 'gate' not in cached_result or cached_result['gate'] != gate:
                                                        cached_result['gate'] = gate
                                                    
                                                    # # Load file and check for 'selected_indices'
                                                    # if 'selected_indices' not in cached_result:
                                                    #     print(f"Missing data in {pickle_file}, adding now...")
                                                    #     with open(pickle_file, 'rb') as f:
                                                    #         df = pickle.load(f)
                                                    #         if 'selected_indices' in df:
                                                    #             selected_indices = df['selected_indices'][0]
                                                    #             cached_result['selected_indices'] = selected_indices
                                                    #             print(f"Added selected indices {selected_indices} to test {gate}, dt: {trotter_step} for run {run} to cache")
                                                    #         else:
                                                    #             print(f"No 'selected_indices' found in file: {pickle_file}")
                                                continue

                                            # Process new file
                                            if is_valid_pickle_file(Path(pickle_file)):
                                                costs, fidelity, num_params, test_results,grads_per_epoch,selected_indices = read_jax_file(pickle_file, gate)
                                                avg_fidelity = np.mean(test_results)

                                                # Store the new data point
                                                if gate not in cached_data[N_ctrl]:
                                                    cached_data[N_ctrl][gate] = {}
                                                if reservoir_count not in cached_data[N_ctrl][gate]:
                                                    cached_data[N_ctrl][gate][reservoir_count] = {}
                                                if trotter_step not in cached_data[N_ctrl][gate][reservoir_count]:
                                                    cached_data[N_ctrl][gate][reservoir_count][trotter_step] = []

                                                # Increment the data run count since we're adding a new run
                                                num_data_runs += 1

                                                # Prepare the new data point
                                                data_point = {
                                                    'costs': costs,
                                                    'gate': gate,
                                                    'fidelity': fidelity,
                                                    'test_results': test_results,
                                                    'param_count': num_params,
                                                    'run': run,  # Store the data_run_<i> value
                                                    'num_data_runs': num_data_runs,
                                                    'grads_per_epoch':grads_per_epoch,
                                                    'path':pickle_file
                                                    # 'selected_indices':selected_indices
                                                }

                                                # Append the new data point to the cache
                                                cached_data[N_ctrl][gate][reservoir_count][trotter_step].append(data_point)

                                                # Mark the file as processed
                                                processed_files.add(pickle_file)
                                                print(f"Added new file to cache: {pickle_file}")

    print(f"Cache updated for N_ctrl={N_ctrl}")
    return cached_data, processed_files