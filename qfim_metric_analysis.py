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

###############################################################################
#  "Build QFIM DataFrame" pipeline (the code from your question)
###############################################################################
from scipy.stats import median_abs_deviation
def to_2d(evals_list):
    """
    Convert list of 1D arrays => single 2D array: shape (n_draws, d).
    """
    return np.vstack(evals_list)

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

def compute_all_stats(
    eigval_list,
    threshold=1e-12,
    spread_methods=("variance", "mad"),  # e.g. ["variance", "mad"]
    ddof=1,
    scale="normal",
    # Additional args for the approximate effective dimension
    do_effective_dim=True,
    vol_param_space=1.0,
    gamma=0.1,    
    n=100,
    V_theta=1.0,
):
    """
    Compute QFIM statistics for a list of draws (eigval_list),
    plus spread-of-log metrics via sample & pooled approaches,
    and an approximate 'effective dimension' from the paper.

    Returns a dict with columns that match your usual naming scheme.
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

    # 3) Flatten eigenvalues to build a single array for "effective dimension"
    #    We'll do a naive 'pooled' approach:
    all_eigs_concat = np.concatenate([
        np.where(np.array(eigs) < threshold, 0.0, eigs) for eigs in eigval_list
    ]) if eigval_list else np.array([])
    # 4) Effective Dimension from Eq. (2) in the paper
    #    d_{gamma,n} = 2 ln( (1/sqrt(V_theta)) * sqrt( det( I + alpha F ) ) )
    #                = -ln(V_theta) + sum_i( ln(1 + alpha lambda_i) )
    #    alpha = gamma * n / (2 ln(n))
    # If you want to do a "global" integral, you'd loop over param points or do MCMC.
    eff_dim_val = 0.0
    if do_effective_dim and all_eigs_concat.size > 0:
        eff_dim_val = effective_dimension_from_paper(
            fisher_eigenvalues=all_eigs_concat,
            n=n, gamma=gamma, V_theta=V_theta
        )
    effective_dim_value = 0.0
    if do_effective_dim and all_eigs_concat.size > 0:
        effective_dim_value = approximate_effective_dimension(
            all_eigs_concat, n=n, gamma=gamma, vol_param_space=vol_param_space
        )

    # 4) Spread-of-log metrics for each method in spread_methods
    #    We'll build arr_2d to use your existing vectorized functions.
    arr_2d = np.zeros((len(eigval_list), max(len(x) for x in eigval_list))) if eigval_list else np.zeros((0,0))
    for i, e in enumerate(eigval_list):
        tmp = np.array(e, dtype=float)
        tmp = np.where(tmp < threshold, 0.0, tmp)
        arr_2d[i, :len(tmp)] = tmp

    spread_results = {}
    from functools import partial

    # We'll rely on your existing spread_per_sample_vectorized & spread_pooling_vectorized:
    # e.g. spread_per_sample_vectorized(arr_2d, method="variance", threshold=1e-12, ddof=1, scale="normal")
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

        # Approx. effective dimension
        "effective_dimension": effective_dim_value,
        "effective_dimensio2": eff_dim_val,
    }

    # Merge in the spread-of-log results with the new naming
    metrics.update(spread_results)

    return metrics
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
    df_all = compute_spread_columns(df_all, threshold=threshold, spread_method="variance", scale="normal")
    df_all = compute_spread_columns(df_all, threshold=threshold, spread_method="mad", scale="normal")
    df_all = compute_spread_columns(df_all, threshold=threshold, spread_method="median", scale="normal")

    return df_out



