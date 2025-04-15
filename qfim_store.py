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
metrics_info = {
    # --- Per-draw Metrics ---
    "QFIM_ranks": {
        "title": "QFIM Ranks per Draw",
        "label": r"$\mathrm{Rank}(Q)$",
        "description": (
            "The number of nonzero eigenvalues (above the threshold) for each draw. "
            "This represents the effective number of independent directions captured by the QFIM."
        )
    },
    "var_all_eigenvals_per_draw": {
        "title": "Variance of All Eigenvalues per Draw",
        "label": r"$\mathrm{Var}(\lambda)$",
        "description": (
            "The variance computed over all eigenvalues for each QFIM draw. "
            "This measures the overall dispersion of the eigenvalue distribution on an absolute scale."
        )
    },
    "var_nonzero_eigenvals_per_draw": {
        "title": "Variance of Nonzero Eigenvalues per Draw",
        "label": r"$\mathrm{Var}_{nz}(\lambda)$",
        "description": (
            "The variance computed only over the eigenvalues that exceed the threshold for each draw. "
            "This metric reflects the dispersion of the significant eigenvalues."
        )
    },
    "trace_eigenvals_per_draw": {
        "title": "Trace of Eigenvalues per Draw",
        "label": r"$\mathrm{Tr}(Q)$",
        "description": (
            "The sum of the eigenvalues for each draw, quantifying the total magnitude of the QFIM."
        )
    },
    "var_norm_rank_per_draw": {
        "title": "Variance Normalized by Rank per Draw",
        "label": r"$\frac{\mathrm{Var}(\lambda)}{\mathrm{Rank}(Q)}$",
        "description": (
            "The variance (over all eigenvalues) divided by the number of nonzero eigenvalues (i.e. the rank) for each draw. "
            "It provides an estimate of the average dispersion per active mode."
        )
    },
    "trace_norm_rank_per_draw": {
        "title": "Trace Normalized by Rank per Draw",
        "label": r"$\frac{\mathrm{Tr}(Q)}{\mathrm{Rank}(Q)}$",
        "description": (
            "The trace (sum of eigenvalues) divided by the rank for each draw. "
            "It represents the average contribution per active eigenmode."
        )
    },

    # --- Aggregated (Absolute Scale) Metrics ---
    "D_C": {
        "title": "Maximum QFIM Rank (D_C)",
        "label": r"$D_C$",
        "description": (
            "The maximum rank observed across all draws. This serves as a simple estimate of the model's "
            "maximum effective capacity."
        )
    },
    "absolute_scale_avg_var_all": {
        "title": "Average Variance (All Eigenvalues)",
        "label": r"$\langle \mathrm{Var}(\lambda) \rangle$",
        "description": (
            "The mean variance computed over all eigenvalues across draws, reflecting the typical spread "
            "of the eigenvalue distributions on an absolute scale."
        )
    },
    "absolute_scale_avg_var_nonzero": {
        "title": "Average Variance (Nonzero Eigenvalues)",
        "label": r"$\langle \mathrm{Var}_{nz}(\lambda) \rangle$",
        "description": (
            "The mean variance computed over only the nonzero eigenvalues across draws, measuring the typical "
            "dispersion among the significant eigenvalues."
        )
    },
    "absolute_scale_avg_trace": {
        "title": "Average Trace of Eigenvalues",
        "label": r"$\langle \mathrm{Tr}(Q) \rangle$",
        "description": (
            "The average sum of the eigenvalues across all draws, indicating the typical total information content "
            "in the QFIM."
        )
    },
    "absolute_scale_var_of_var_all": {
        "title": "Variance of Variance (All Eigenvalues)",
        "label": r"$\mathrm{Var}(\mathrm{Var}(\lambda))$",
        "description": (
            "The variance of the per-draw variances (over all eigenvalues), which quantifies the consistency "
            "of the eigenvalue spread across different draws."
        )
    },
    "absolute_scale_var_of_var_nonzero": {
        "title": "Variance of Variance (Nonzero Eigenvalues)",
        "label": r"$\mathrm{Var}(\mathrm{Var}_{nz}(\lambda))$",
        "description": (
            "The variance of the per-draw variances computed over nonzero eigenvalues, indicating the consistency "
            "of the spread among significant eigenvalues."
        )
    },

    # --- Local Dimension Metrics (Spectrum Shape) ---
    # IPR-based:
    "spectrum_shape_ipr_deffs_norm_per_draw": {
        "title": "Normalized IPR Dimension per Draw",
        "label": r"$\mathrm{IPR}_{\mathrm{norm}}$",
        "description": (
            "The normalized inverse participation ratio for each draw, calculated as 1/sum(norm(eigenvalues)^2) "
            "after dividing by the trace. It captures the effective number of active modes based solely on the shape "
            "of the eigenvalue spectrum."
        )
    },
    "spectrum_shape_avg_ipr_deffs_norm": {
        "title": "Average Normalized IPR Dimension",
        "label": r"$\langle \mathrm{IPR}_{\mathrm{norm}} \rangle$",
        "description": (
            "The average normalized IPR over all draws, representing an overall measure of the effective number of modes "
            "(ignoring total magnitude) in the QFIM."
        )
    },
    "ipr_deffs_raw_per_draw": {
        "title": "Raw IPR Dimension per Draw",
        "label": r"$\mathrm{IPR}_{\mathrm{raw}}$",
        "description": (
            "The raw inverse participation ratio, computed as $(\mathrm{Tr}(Q))^2 / \sum \lambda_i^2$, for each draw. "
            "This gives an absolute-scale estimate of the effective number of modes."
        )
    },
    "avg_ipr_deffs_raw": {
        "title": "Average Raw IPR Dimension",
        "label": r"$\langle \mathrm{IPR}_{\mathrm{raw}} \rangle$",
        "description": (
            "The mean of the raw IPR values over all draws, summarizing the absolute effective dimensionality of the QFIM."
        )
    },
    # Abbas-based:
    "spectrum_shape_abbas_deffs_norm_per_draw": {
        "title": "Normalized Abbas Dimension per Draw",
        "label": r"$d_{\mathrm{eff}}^{abbas}$",
        "description": (
            "The normalized Abbas effective dimension for each draw, computed by summing the logarithms of (1 + α·λ) "
            "after normalizing by the trace. It quantifies the effective dimension based on the eigenvalue spectrum's shape."
        )
    },
    "spectrum_shape_avg_abbas_deffs_norm": {
        "title": "Average Normalized Abbas Dimension",
        "label": r"$\langle d_{\mathrm{eff}}^{abbas} \rangle$",
        "description": (
            "The average of the normalized Abbas dimensions across draws, providing an overall measure of the model's "
            "effective dimensionality according to the Abbas method."
        )
    },
    "abbas_deffs_raw_per_draw": {
        "title": "Raw Abbas Dimension per Draw",
        "label": r"$d_{\mathrm{eff}}^{abbas, raw}$",
        "description": (
            "The raw Abbas dimension for each draw, computed as the sum of log(1 + α·λ) over all eigenvalues. "
            "This metric measures the effective dimension in absolute terms."
        )
    },
    "avg_abbas_deffs_raw": {
        "title": "Average Raw Abbas Dimension",
        "label": r"$\langle d_{\mathrm{eff}}^{abbas, raw} \rangle$",
        "description": (
            "The mean raw Abbas dimension across draws, summarizing the overall effective dimension in absolute scale."
        )
    },
    "abbas_deffs_simple": {
        "title": "Simple Abbas Dimension",
        "label": r"$d_{\mathrm{eff}}^{abbas, simple}$",
        "description": (
            "A simplified effective dimension computed by an alternative approach (e.g. using an IPR-based method) "
            "on the eigenvalues. This serves as an alternative estimate of the model's effective capacity."
        )
    },

    # --- Average Per Active Mode Metrics ---
    "avg_per_active_mode_var_norm_rank_per_draw": {
        "title": "Variance per Active Mode per Draw",
        "label": r"$\frac{\mathrm{Var}(\lambda)}{\mathrm{Rank}(Q)}$",
        "description": (
            "For each draw, the variance of all eigenvalues divided by the rank (i.e. number of nonzero eigenvalues), "
            "indicating the average dispersion per active mode."
        )
    },
    "avg_per_active_mode_trace_norm_rank_per_draw": {
        "title": "Trace per Active Mode per Draw",
        "label": r"$\frac{\mathrm{Tr}(Q)}{\mathrm{Rank}(Q)}$",
        "description": (
            "For each draw, the trace of the eigenvalues divided by the rank. This gives the average contribution per "
            "active eigenmode."
        )
    },
    "avg_per_active_mode_avg_var_norm_rank": {
        "title": "Average Variance per Active Mode",
        "label": r"$\langle \frac{\mathrm{Var}(\lambda)}{\mathrm{Rank}(Q)} \rangle$",
        "description": (
            "The average of the variance-per-rank values over all draws, reflecting the typical dispersion per active mode."
        )
    },
    "avg_per_active_mode_avg_trace_norm_rank": {
        "title": "Average Trace per Active Mode",
        "label": r"$\langle \frac{\mathrm{Tr}(Q)}{\mathrm{Rank}(Q)} \rangle$",
        "description": (
            "The average of the trace-per-rank values over all draws, representing the typical signal per active mode."
        )
    },

    # --- Spread-of-Log Metrics ---
    "spread_mean_per_sample_variance_normal": {
        "title": "Mean Spread-of-Log (Variance Method)",
        "label": r"$\mu_{\mathrm{spread}}^{Var}$",
        "description": (
            "The mean of the spread-of-log values computed using the variance method. This metric quantifies "
            "the average dispersion of the logarithms of the normalized eigenvalues."
        )
    },
    "spread_std_per_sample_variance_normal": {
        "title": "Standard Deviation of Spread-of-Log (Variance Method)",
        "label": r"$\sigma_{\mathrm{spread}}^{Var}$",
        "description": (
            "The standard deviation of the spread-of-log values (via the variance method) across each draw, "
            "indicating the variability in the eigenvalue dispersion."
        )
    },
    "spread_val_pooled_variance_normal": {
        "title": "Pooled Spread-of-Log (Variance Method)",
        "label": r"$S_{\mathrm{spread}}^{Var}$",
        "description": (
            "A single pooled value computed from the spread-of-log (variance method) over all draws, summarizing "
            "the overall dispersion of the normalized eigenvalue distribution."
        )
    },
    "spread_mean_per_sample_mad_normal": {
        "title": "Mean Spread-of-Log (MAD Method)",
        "label": r"$\mu_{\mathrm{spread}}^{MAD}$",
        "description": (
            "The mean of the spread-of-log values computed using the median absolute deviation (MAD) method. "
            "It provides an alternative measure of the dispersion of the normalized eigenvalue spectrum."
        )
    },
    "spread_std_per_sample_mad_normal": {
        "title": "Standard Deviation of Spread-of-Log (MAD Method)",
        "label": r"$\sigma_{\mathrm{spread}}^{MAD}$",
        "description": (
            "The standard deviation of the spread-of-log values computed via the MAD method, reflecting the variability "
            "of the eigenvalue dispersion."
        )
    },
    "spread_val_pooled_mad_normal": {
        "title": "Pooled Spread-of-Log (MAD Method)",
        "label": r"$S_{\mathrm{spread}}^{MAD}$",
        "description": (
            "The pooled spread-of-log metric computed using the MAD method, representing the overall dispersion of the "
            "normalized eigenvalue distribution."
        )
    },

    # --- Global Dimension Metrics ---
    "global_effective_dimension": {
        "title": "Global Effective Dimension",
        "label": r"$d_{\mathrm{eff}}^{global}$",
        "description": (
            "The effective dimension computed using a global method (e.g. from the averaged full QFIM matrix via the "
            "log-determinant approach). It quantifies the overall capacity of the model when considering all draws collectively."
        )
    },
    "fisher_trace": {
        "title": "Fisher Trace of the Averaged QFIM",
        "label": r"$\mathrm{Tr}(\bar{F})$",
        "description": (
            "The trace of the average (empirical) full QFIM matrix. This value is used as the normalization factor in "
            "the global effective dimension calculation and reflects the overall magnitude of the QFIM."
        )
    },
}
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


def get_cached_data(base_path,model_type, N_ctrl, K_0,sample_label):
    """
    Load cached_data if it exists, else return empty placeholders.
    Returns (cached_data, processed_files).
    
    Expects a cache file named: 'digital_new_QFIM_Nc_{N_ctrl}_{K_0}K.pkl'
    """
    cache_file = os.path.join(base_path, f'{model_type}_new_QFIM_Nc_{N_ctrl}_{K_0}K_sample_{sample_label}.pkl')

    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            cached_data, processed_files = pickle.load(f)
        print(f"[INFO] Loaded cache for N_ctrl={N_ctrl} from disk at {cache_file}, num files: {len(processed_files)}.")
        return cached_data, processed_files
    else:
        print(f"[WARN] No cache for N_ctrl={N_ctrl} at {cache_file}.")
        return {}, set()
def load_and_clean_pickle(data_file):
    """Helper function to load and clean a pickle file."""
    with open(data_file, 'rb') as f:
        df = pickle.load(f)
        return clean_array(df)
def save_cached_data(base_path,model_type, cached_data, processed_files, N_ctrl, K_0,sample_label):
    """
    Save the (cached_data, processed_files) to a single pickle file.
    """
    cache_file = os.path.join(base_path, f'{model_type}_new_QFIM_Nc_{N_ctrl}_{K_0}K_sample_{sample_label}.pkl')
    with open(cache_file, 'wb') as f:
        pickle.dump((cached_data, processed_files), f)
    print(f"[INFO] Cache saved for N_ctrl={N_ctrl} at {cache_file}.")

def extract_trotter_step(data_file):
    """
    For example, parse the directory name 'step_10' => 10
    """
    # if isinstance(data_file,str):
    #     return int(data_file.split('_')[-1])
    return int(data_file.parent.name.split('_')[-1])

def extract_Nr(data_file):
    """
    For example, parse the directory 'Nr_2' => 2
    """
    # if isinstance(data_file,str):
    #     return int(data_file.split('_')[-1])
    return int(data_file.parent.parent.name.split('_')[-1])

def process_data_combined(df, threshold, by_test, Nc, print_bool=False):
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
            N_R = int(df[fixed_params_key][test_key].get('n_reserv', None))
            trot  = int(df[fixed_params_key][test_key].get('time_steps', None))
            assert isinstance(N_R, int) and isinstance(trot, int), f'Nr: {N_R} {type(N_R)}, T: {trot} {type(trot)},'
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
                    # trotter_step_num = raw_df.get('time_steps', None)
                    # reservoir_count  = raw_df.get('n_reserv', None)
                    # assert isinstance(reservoir_count, int) and isinstance(trotter_step_num, int), f'Nr: {reservoir_count}, T: {trotter_step_num}\n{raw_df.keys()}'
                    processed_data = process_data_combined(
                        raw_df, threshold, by_test, Nc=N_ctrl, print_bool=False
                    )

                    cached_data[file_id] = {
                        "processed_data": processed_data,
                        "raw_data": raw_df  # optional if you want to keep the raw
                    }
                    processed_files.add(file_id)
                    all_data.append(processed_data)



        # After scanning all directories for this N_ctrl, save the updated cache
        save_cached_data(base_path,model_type, cached_data, processed_files, N_ctrl, K_0,sample_label=sample_range)

    return cached_data, processed_files, pd.DataFrame(all_data)

###############################################################################
# 4) The function to incorporate newly generated data pickles
###############################################################################
def update_cached_data(data_file, cached_data, processed_files, N_ctrl,threshold):
    """
    Update a cached entry for data_file if new fixed_param keys or new test keys are present.
    Instead of storing the entire raw DataFrame, we keep a snapshot of its keys ("raw_keys").
    If differences are detected, we re-load the file, re-run process_data_combined,
    and update both the processed_data and the stored keys.
    """
    file_id = str(data_file)
    df = load_and_clean_pickle(data_file)
    new_keys = get_keys(df)
    old_keys = cached_data[file_id].get("raw_keys", {})

    # Check for any new fixed_param keys or new tests for an existing fixed_param.
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
        
        processed_data = process_data_combined(
            df, threshold, by_test=False, Nc=N_ctrl,
            print_bool=False
        )
        cached_data[file_id]["processed_data"] = processed_data
        cached_data[file_id]["raw_keys"] = new_keys

    processed_files.add(file_id)
    return needs_update
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
            # print(file_id)
            # print(Nr)
            # N_R = extract_Nr(Nr)
            # T = extract_trotter_step(trotter_step_dir)

            if file_id in cached_data:
                # print(f"[INFO] Checking cached file: {file_id}")
                updated = update_cached_data(data_file, cached_data, processed_files, N_ctrl, threshold)
                # Only append if it was actually updated
                if updated:
                    all_new_data.append(cached_data[file_id]['processed_data'])
                continue

            if is_valid_pickle_file(data_file):
                print(f"[INFO] Found NEW data file: {data_file}")
                raw_df = load_and_clean_pickle(data_file)
                # trotter_step_num = raw_df.get('time_steps', None)
                # reservoir_count  = raw_df.get('n_reserv', None)
                # assert isinstance(reservoir_count, int) and isinstance(reservoir_count, int), f'Nr: {reservoir_count}, T: {trotter_step_num}'
                processed_data = process_data_combined(
                    raw_df, threshold, by_test,
                    Nc=N_ctrl
                )
                cached_data[file_id] = {
                    "processed_data": processed_data,
                    # "raw_keys": get_keys(raw_df)
                }
                processed_files.add(file_id)
                all_new_data.append(processed_data)


    if all_new_data:
        save_cached_data(base_path,model_type, cached_data, processed_files, N_ctrl, K_0,sample_label=sample_range)
    return cached_data, processed_files, pd.DataFrame(all_new_data)
# ----------------------------------------------------------------------------
# Rebuild df_all from existing cached_data 
# ----------------------------------------------------------------------------
def rebuild_df_from_existing_cache(base_path,model_type,N_ctrls, K_0,sample_range):
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
        cdata, pfiles = get_cached_data(base_path,model_type, N_ctrl, K_0,sample_range)
        for file_id, file_info in cdata.items():
            pdata = file_info.get('processed_data', {})
            all_data.append(pdata)

    df_all = pd.DataFrame(all_data)
    return df_all

# ----------------------------------------------------------------------------
# Possibly unify reading from cache or processing new files
# ----------------------------------------------------------------------------
def maybe_rebuild_or_process(base_path, sample_range, model_type, N_ctrls, K_0, sample_label,threshold=1e-10, by_test=False):
    """
    1) Attempt to rebuild from existing caches for each N_ctrl in N_ctrls.
    2) If a cache is missing or incomplete, process new files for that N_ctrl.
    3) Combine partial DataFrames into one df_all.
    """
    combined_frames = []
    for N_ctrl in N_ctrls:
        cached_data, processed_files = get_cached_data(base_path,model_type, N_ctrl, K_0,sample_label)
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
                    reservoir_count = raw_data.get('n_reserv', None)
                    trotter_step_num  = raw_data.get('time_steps', None)
                    assert isinstance(reservoir_count, int) and isinstance(trotter_step_num, int), f'Nr: {reservoir_count}, T: {trotter_step_num}'
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
def compute_all_stats_old(
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

    # For Abbas measure (optional local dimension approach)
    # =============== NEW #2: Abbas local dimension ===============
    # d_{n, gamma}(theta) ~ -log(V_theta) + sum_i log(1 + alpha * lambda_i)
    # We'll just set V_theta=1 => -log(1)=0, so that term vanishes.
    # Then we do sum(log(1 + alpha * lambda_i)).
    # If alpha*lambda_i < -1, that log is undefined; in practice, alpha*lambda_i >= 0 if alpha>0, lambda_i >= 0

    n = len(eigval_list)   # interpret as number of data samples
    gamma = 1.0
    if n > 1:
        alpha = (gamma * n) / (2.0 * np.log(n))
    else:
        alpha = 0.0
    V_theta = 1.0  # Placeholder

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

         # [3] average per nonzero mode => divide by rank if rank>0
        if rank > 0:
            var_norm_rank_per_draw.append(var_all / rank)
            trace_norm_rank_per_draw.append(trace_val / rank)
        else:
            var_norm_rank_per_draw.append(0.0)
            trace_norm_rank_per_draw.append(0.0)

         # --------------------------- IPR-based d_eff ---------------------------
        # raw
        sum_eigs_sq = np.sum(arr**2)
        if sum_eigs_sq > 0:
            ipr_raw = (trace_val**2) / sum_eigs_sq
        else:
            ipr_raw = 0.0
        ipr_deffs_raw.append(ipr_raw)

        # normalized (i.e., shape only)
        if trace_val > 0:
            arr_norm = arr / trace_val
            sum_norm_sq = np.sum(arr_norm**2)
            if sum_norm_sq > 0.0:
                ipr_norm = 1.0 / sum_norm_sq
            else:
                ipr_norm = 0.0
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

        # normalized
        abbas_norm = 0.0
        if trace_val > 0:
            for lam_norm in (arr / trace_val):
                val = 1.0 + alpha * lam_norm
                if val <= 0.0:
                    val = 1e-15
                abbas_norm += np.log(val)
        # if V_theta != 1: abbas_norm -= np.log(V_theta)
        abbas_deffs_norm.append(abbas_norm)


    # -------------------------------------------------------------------------
    # 2) Aggregate across draws
    # -------------------------------------------------------------------------
    D_C = max(ranks_per_draw) if ranks_per_draw else 0

    avg_var_all = np.mean(var_all_per_draw) if var_all_per_draw else 0.0
    avg_trace = np.mean(trace_per_draw) if trace_per_draw else 0.0
    avg_var_nonzero = np.mean(var_nonzero_per_draw) if var_nonzero_per_draw else 0.0

    avg_var_norm_rank = np.mean(var_norm_rank_per_draw) if var_norm_rank_per_draw else 0.0
    avg_trace_norm_rank = np.mean(trace_norm_rank_per_draw) if trace_norm_rank_per_draw else 0.0

    var_var_all = np.var(var_all_per_draw) if len(var_all_per_draw) > 1 else 0.0
    var_var_nonzero = np.var(var_nonzero_per_draw) if len(var_nonzero_per_draw) > 1 else 0.0

    # Summaries for the new effective dimension lists
    avg_ipr_raw   = float(np.mean(ipr_deffs_raw))  if ipr_deffs_raw  else 0.0
    avg_ipr_norm  = float(np.mean(ipr_deffs_norm)) if ipr_deffs_norm else 0.0
    avg_abbas_raw = float(np.mean(abbas_deffs_raw))  if abbas_deffs_raw  else 0.0
    avg_abbas_norm= float(np.mean(abbas_deffs_norm)) if abbas_deffs_norm else 0.0

    # -------------------------------------------------------------------------
    # 3) Spread-of-log metrics (unchanged)
    #    We interpret each row of arr_2d as one draw's eigenvalues
    #    => Then we do the 'spread_of_log' across them.
    # -------------------------------------------------------------------------
    arr_2d = np.zeros((len(eigval_list), max(len(x) for x in eigval_list))) if eigval_list else np.zeros((0,0))
    for i, e in enumerate(eigval_list):
        tmp = np.array(e, dtype=float)
        tmp = np.where(tmp < threshold, 0.0, tmp)
        arr_2d[i, :len(tmp)] = tmp

    spread_results = {}
    for method in spread_methods:
        per_draw = spread_per_sample_vectorized(
            arr_2d, method=method, threshold=threshold, ddof=ddof, scale=scale
        )
        spread_mean = per_draw.mean() if per_draw.size else 0.0
        spread_std  = per_draw.std()  if per_draw.size > 1 else 0.0
        pooled_val  = spread_pooling_vectorized(
            arr_2d, method=method, threshold=threshold, ddof=ddof, scale=scale
        )
        prefix = method.lower()
        spread_results[f"spread_mean_per_sample_{prefix}_{scale}"] = spread_mean
        spread_results[f"spread_std_per_sample_{prefix}_{scale}"]  = spread_std
        spread_results[f"spread_val_pooled_{prefix}_{scale}"]      = pooled_val


    # -------------------------------------------------------------------------
    # 4) Build Final Dictionary with CLEAR Category Labels
    # -------------------------------------------------------------------------
    metrics = {
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Category [A]: Basic Info & Per-draw lists
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        "QFIM_ranks": ranks_per_draw,
        "var_all_eigenvals_per_draw": var_all_per_draw,       # absolute scale
        "var_nonzero_eigenvals_per_draw": var_nonzero_per_draw,# absolute scale
        "trace_eigenvals_per_draw": trace_per_draw,           # absolute scale

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Category [1]: ABSOLUTE SCALE (aggregated)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        "absolute_scale_avg_var_all": avg_var_all,
        "absolute_scale_avg_var_nonzero": avg_var_nonzero,
        "absolute_scale_avg_trace": avg_trace,

        # For those wanting the variance of the 'var_all_eigenvals_per_draw'
        "absolute_scale_var_of_var_all": var_var_all,
        "absolute_scale_var_of_var_nonzero": var_var_nonzero,

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Category [2]: SHAPE OF THE SPECTRUM
        # (these are the normalized versions of IPR & Abbas)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        "spectrum_shape_ipr_deffs_norm_per_draw": ipr_deffs_norm,
        "spectrum_shape_avg_ipr_deffs_norm": avg_ipr_norm,
        
        "spectrum_shape_abbas_deffs_norm_per_draw": abbas_deffs_norm,
        "spectrum_shape_avg_abbas_deffs_norm": avg_abbas_norm,

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Category [3]: AVERAGE PER NONZERO MODE
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        "avg_per_active_mode_var_norm_rank_per_draw": var_norm_rank_per_draw,
        "avg_per_active_mode_trace_norm_rank_per_draw": trace_norm_rank_per_draw,

        "avg_per_active_mode_avg_var_norm_rank": avg_var_norm_rank,
        "avg_per_active_mode_avg_trace_norm_rank": avg_trace_norm_rank,

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Rank-based dimension as a simpler measure of capacity
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        "D_C": D_C,  # max rank observed across draws

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # RAW IPR & ABBAS for absolute scale dimension
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        "ipr_deffs_raw_per_draw": ipr_deffs_raw,
        "avg_ipr_deffs_raw": avg_ipr_raw,

        "abbas_deffs_raw_per_draw": abbas_deffs_raw,
        "avg_abbas_deffs_raw": avg_abbas_raw,
    }

    # Incorporate spread-of-log results 
    metrics.update(spread_results)

    return metrics
metrics_info = {
    # --- Per-draw Metrics ---
    "QFIM_ranks": {
        "title": "QFIM Ranks per Draw",
        "label": r"$\mathrm{Rank}(Q)$",
        "description": (
            "The number of nonzero eigenvalues (above the threshold) for each draw. "
            "This represents the effective number of independent directions captured by the QFIM."
        )
    },
    "var_all_eigenvals_per_draw": {
        "title": "Variance of All Eigenvalues per Draw",
        "label": r"$\mathrm{Var}(\lambda)$",
        "description": (
            "The variance computed over all eigenvalues for each QFIM draw. "
            "This measures the overall dispersion of the eigenvalue distribution on an absolute scale."
        )
    },
    "var_nonzero_eigenvals_per_draw": {
        "title": "Variance of Nonzero Eigenvalues per Draw",
        "label": r"$\mathrm{Var}_{nz}(\lambda)$",
        "description": (
            "The variance computed only over the eigenvalues that exceed the threshold for each draw. "
            "This metric reflects the dispersion of the significant eigenvalues."
        )
    },
    "trace_eigenvals_per_draw": {
        "title": "Trace of Eigenvalues per Draw",
        "label": r"$\mathrm{Tr}(Q)$",
        "description": (
            "The sum of the eigenvalues for each draw, quantifying the total magnitude of the QFIM."
        )
    },
    "var_norm_rank_per_draw": {
        "title": "Variance Normalized by Rank per Draw",
        "label": r"$\frac{\mathrm{Var}(\lambda)}{\mathrm{Rank}(Q)}$",
        "description": (
            "The variance (over all eigenvalues) divided by the number of nonzero eigenvalues (i.e. the rank) for each draw. "
            "It provides an estimate of the average dispersion per active mode."
        )
    },
    "trace_norm_rank_per_draw": {
        "title": "Trace Normalized by Rank per Draw",
        "label": r"$\frac{\mathrm{Tr}(Q)}{\mathrm{Rank}(Q)}$",
        "description": (
            "The trace (sum of eigenvalues) divided by the rank for each draw. "
            "It represents the average contribution per active eigenmode."
        )
    },

    # --- Aggregated (Absolute Scale) Metrics ---
    "D_C": {
        "title": "Maximum QFIM Rank (D_C)",
        "label": r"$D_C$",
        "description": (
            "The maximum rank observed across all draws. This serves as a simple estimate of the model's "
            "maximum effective capacity."
        )
    },
    "absolute_scale_avg_var_all": {
        "title": "Average Variance (All Eigenvalues)",
        "label": r"$\langle \mathrm{Var}(\lambda) \rangle$",
        "description": (
            "The mean variance computed over all eigenvalues across draws, reflecting the typical spread "
            "of the eigenvalue distributions on an absolute scale."
        )
    },
    "absolute_scale_avg_var_nonzero": {
        "title": "Average Variance (Nonzero Eigenvalues)",
        "label": r"$\langle \mathrm{Var}_{nz}(\lambda) \rangle$",
        "description": (
            "The mean variance computed over only the nonzero eigenvalues across draws, measuring the typical "
            "dispersion among the significant eigenvalues."
        )
    },
    "absolute_scale_avg_trace": {
        "title": "Average Trace of Eigenvalues",
        "label": r"$\langle \mathrm{Tr}(Q) \rangle$",
        "description": (
            "The average sum of the eigenvalues across all draws, indicating the typical total information content "
            "in the QFIM."
        )
    },
    "absolute_scale_var_of_var_all": {
        "title": "Variance of Variance (All Eigenvalues)",
        "label": r"$\mathrm{Var}(\mathrm{Var}(\lambda))$",
        "description": (
            "The variance of the per-draw variances (over all eigenvalues), which quantifies the consistency "
            "of the eigenvalue spread across different draws."
        )
    },
    "absolute_scale_var_of_var_nonzero": {
        "title": "Variance of Variance (Nonzero Eigenvalues)",
        "label": r"$\mathrm{Var}(\mathrm{Var}_{nz}(\lambda))$",
        "description": (
            "The variance of the per-draw variances computed over nonzero eigenvalues, indicating the consistency "
            "of the spread among significant eigenvalues."
        )
    },

    # --- Local Dimension Metrics (Spectrum Shape) ---
    # IPR-based:
    "spectrum_shape_ipr_deffs_norm_per_draw": {
        "title": "Normalized IPR Dimension per Draw",
        "label": r"$\mathrm{IPR}_{\mathrm{norm}}$",
        "description": (
            "The normalized inverse participation ratio for each draw, calculated as 1/sum(norm(eigenvalues)^2) "
            "after dividing by the trace. It captures the effective number of active modes based solely on the shape "
            "of the eigenvalue spectrum."
        )
    },
    "spectrum_shape_avg_ipr_deffs_norm": {
        "title": "Average Normalized IPR Dimension",
        "label": r"$\langle \mathrm{IPR}_{\mathrm{norm}} \rangle$",
        "description": (
            "The average normalized IPR over all draws, representing an overall measure of the effective number of modes "
            "(ignoring total magnitude) in the QFIM."
        )
    },
    "ipr_deffs_raw_per_draw": {
        "title": "Raw IPR Dimension per Draw",
        "label": r"$\mathrm{IPR}_{\mathrm{raw}}$",
        "description": (
            "The raw inverse participation ratio, computed as $(\mathrm{Tr}(Q))^2 / \sum \lambda_i^2$, for each draw. "
            "This gives an absolute-scale estimate of the effective number of modes."
        )
    },
    "avg_ipr_deffs_raw": {
        "title": "Average Raw IPR Dimension",
        "label": r"$\langle \mathrm{IPR}_{\mathrm{raw}} \rangle$",
        "description": (
            "The mean of the raw IPR values over all draws, summarizing the absolute effective dimensionality of the QFIM."
        )
    },
    # Abbas-based:
    "spectrum_shape_abbas_deffs_norm_per_draw": {
        "title": "Normalized Abbas Dimension per Draw",
        "label": r"$d_{\mathrm{eff}}^{abbas}$",
        "description": (
            "The normalized Abbas effective dimension for each draw, computed by summing the logarithms of (1 + α·λ) "
            "after normalizing by the trace. It quantifies the effective dimension based on the eigenvalue spectrum's shape."
        )
    },
    "spectrum_shape_avg_abbas_deffs_norm": {
        "title": "Average Normalized Abbas Dimension",
        "label": r"$\langle d_{\mathrm{eff}}^{abbas} \rangle$",
        "description": (
            "The average of the normalized Abbas dimensions across draws, providing an overall measure of the model's "
            "effective dimensionality according to the Abbas method."
        )
    },
    "abbas_deffs_raw_per_draw": {
        "title": "Raw Abbas Dimension per Draw",
        "label": r"$d_{\mathrm{eff}}^{abbas, raw}$",
        "description": (
            "The raw Abbas dimension for each draw, computed as the sum of log(1 + α·λ) over all eigenvalues. "
            "This metric measures the effective dimension in absolute terms."
        )
    },
    "avg_abbas_deffs_raw": {
        "title": "Average Raw Abbas Dimension",
        "label": r"$\langle d_{\mathrm{eff}}^{abbas, raw} \rangle$",
        "description": (
            "The mean raw Abbas dimension across draws, summarizing the overall effective dimension in absolute scale."
        )
    },
    "abbas_deffs_simple": {
        "title": "Simple Abbas Dimension",
        "label": r"$d_{\mathrm{eff}}^{abbas, simple}$",
        "description": (
            "A simplified effective dimension computed by an alternative approach (e.g. using an IPR-based method) "
            "on the eigenvalues. This serves as an alternative estimate of the model's effective capacity."
        )
    },

    # --- Average Per Active Mode Metrics ---
    "avg_per_active_mode_var_norm_rank_per_draw": {
        "title": "Variance per Active Mode per Draw",
        "label": r"$\frac{\mathrm{Var}(\lambda)}{\mathrm{Rank}(Q)}$",
        "description": (
            "For each draw, the variance of all eigenvalues divided by the rank (i.e. number of nonzero eigenvalues), "
            "indicating the average dispersion per active mode."
        )
    },
    "avg_per_active_mode_trace_norm_rank_per_draw": {
        "title": "Trace per Active Mode per Draw",
        "label": r"$\frac{\mathrm{Tr}(Q)}{\mathrm{Rank}(Q)}$",
        "description": (
            "For each draw, the trace of the eigenvalues divided by the rank. This gives the average contribution per "
            "active eigenmode."
        )
    },
    "avg_per_active_mode_avg_var_norm_rank": {
        "title": "Average Variance per Active Mode",
        "label": r"$\langle \frac{\mathrm{Var}(\lambda)}{\mathrm{Rank}(Q)} \rangle$",
        "description": (
            "The average of the variance-per-rank values over all draws, reflecting the typical dispersion per active mode."
        )
    },
    "avg_per_active_mode_avg_trace_norm_rank": {
        "title": "Average Trace per Active Mode",
        "label": r"$\langle \frac{\mathrm{Tr}(Q)}{\mathrm{Rank}(Q)} \rangle$",
        "description": (
            "The average of the trace-per-rank values over all draws, representing the typical signal per active mode."
        )
    },

    # --- Spread-of-Log Metrics ---
    "spread_mean_per_sample_variance_normal": {
        "title": "Mean Spread-of-Log (Variance Method)",
        "label": r"$\mu_{\mathrm{spread}}^{Var}$",
        "description": (
            "The mean of the spread-of-log values computed using the variance method. This metric quantifies "
            "the average dispersion of the logarithms of the normalized eigenvalues."
        )
    },
    "spread_std_per_sample_variance_normal": {
        "title": "Standard Deviation of Spread-of-Log (Variance Method)",
        "label": r"$\sigma_{\mathrm{spread}}^{Var}$",
        "description": (
            "The standard deviation of the spread-of-log values (via the variance method) across each draw, "
            "indicating the variability in the eigenvalue dispersion."
        )
    },
    "spread_val_pooled_variance_normal": {
        "title": "Pooled Spread-of-Log (Variance Method)",
        "label": r"$S_{\mathrm{spread}}^{Var}$",
        "description": (
            "A single pooled value computed from the spread-of-log (variance method) over all draws, summarizing "
            "the overall dispersion of the normalized eigenvalue distribution."
        )
    },
    "spread_mean_per_sample_mad_normal": {
        "title": "Mean Spread-of-Log (MAD Method)",
        "label": r"$\mu_{\mathrm{spread}}^{MAD}$",
        "description": (
            "The mean of the spread-of-log values computed using the median absolute deviation (MAD) method. "
            "It provides an alternative measure of the dispersion of the normalized eigenvalue spectrum."
        )
    },
    "spread_std_per_sample_mad_normal": {
        "title": "Standard Deviation of Spread-of-Log (MAD Method)",
        "label": r"$\sigma_{\mathrm{spread}}^{MAD}$",
        "description": (
            "The standard deviation of the spread-of-log values computed via the MAD method, reflecting the variability "
            "of the eigenvalue dispersion."
        )
    },
    "spread_val_pooled_mad_normal": {
        "title": "Pooled Spread-of-Log (MAD Method)",
        "label": r"$S_{\mathrm{spread}}^{MAD}$",
        "description": (
            "The pooled spread-of-log metric computed using the MAD method, representing the overall dispersion of the "
            "normalized eigenvalue distribution."
        )
    },

    # --- Global Dimension Metrics ---
    "global_effective_dimension": {
        "title": "Global Effective Dimension",
        "label": r"$d_{\mathrm{eff}}^{global}$",
        "description": (
            "The effective dimension computed using a global method (e.g. from the averaged full QFIM matrix via the "
            "log-determinant approach). It quantifies the overall capacity of the model when considering all draws collectively."
        )
    },
    "fisher_trace": {
        "title": "Fisher Trace of the Averaged QFIM",
        "label": r"$\mathrm{Tr}(\bar{F})$",
        "description": (
            "The trace of the average (empirical) full QFIM matrix. This value is used as the normalization factor in "
            "the global effective dimension calculation and reflects the overall magnitude of the QFIM."
        )
    },
}
if __name__ == "__main__":
    base_path = "/Users/so714f/Documents/offline/qrc/"
    model_type = "analog"
    N_ctrls = [2]
    sample_range = "normal_.5pi_.1t"
    K_str = "1"#  "1" "pi"
    threshold = 1e-12
    by_test = False

    # 1) Possibly we do:
    #    df_all = rebuild_df_from_existing_cache(base_path, N_ctrls, K_str)
    #    if df_all.empty, we call process_and_cache_new_files
    #    or we do the combined approach:
    df_all = maybe_rebuild_or_process(
        base_path, sample_range, model_type, N_ctrls, K_str,sample_label=sample_range,
        threshold=threshold, by_test=by_test
    )

    print("[INFO] df_all shape after reading cache or scanning directories:", df_all.shape)

    # 2) Build QFIM DataFrame with advanced metrics
    # df_all = build_qfim_dataframe(df_all, threshold=1e-12)
    # print("[INFO] df_all final shape:", df_all.shape)
    # print(df_all.head())
