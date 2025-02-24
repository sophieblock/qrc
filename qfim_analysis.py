import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import median_abs_deviation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import PowerNorm
###############################################################################
# 0) General utility: compute_spread_metric
###############################################################################
def compute_spread_metric(values, method="variance", ddof=1, scale="none"):
    """
    Given 1D array 'values' (e.g. log-transformed normalized eigenvalues),
    compute a chosen 'spread' metric:
      - 'variance': np.var(values, ddof=ddof)
      - 'mad': median_abs_deviation(values, scale=scale)
      - 'median': np.median(values)
    ddof applies to variance only.
    scale applies to MAD (either "normal" or a numeric float).
    """
    if len(values) <= 1:
        return 0.0  # or np.nan, depending on preference

    if method == "variance":
        return np.var(values, ddof=ddof)
    elif method == "mad":
        return median_abs_deviation(values, scale=scale)
    elif method == "median":
        return np.median(values)
    else:
        raise ValueError(f"Unknown method '{method}'")


###############################################################################
#  "Build QFIM DataFrame" pipeline 
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

def compute_all_stats(
    eigval_list: List[np.ndarray],
    threshold: float = 1e-12,
    compute_spread: bool = True
):
    """
    Compute a concise set of QFIM metrics (rank, trace, variance, etc.) 
    for a list of eigenvalue arrays ('eigval_list'), each array being one 'draw'.
    
    Parameters
    ----------
    eigval_list : list of 1D arrays
        Each element is a numpy array of QFIM eigenvalues for a single draw.
    threshold : float, optional
        Values below this are treated as zero.
    compute_spread : bool, optional
        Whether to compute spread-of-log metrics (pooled and per-draw).
    
    Returns
    -------
    metrics : dict
        A dictionary with keys such as:
        
        - "rank_per_draw": list of ranks (one per draw)
        - "max_rank": maximum rank across draws (a.k.a. D_C)
        - "avg_rank": average rank across draws
        - "avg_var_all": mean of the variance across draws (including zeros as 0.0)
        - "avg_var_nonzero": mean of the variance across draws (only nonzero eigs)
        - "avg_trace": mean of the sum of eigvals across draws
        - "avg_trace_norm_by_len": mean( trace/length ) across draws
        - "avg_trace_norm_by_rank": mean( trace/rank ) across draws
        - "pooled_spread_log": spread (variance, MAD, etc.) of log(eigs) 
                               across *all draws combined* (if compute_spread=True)
        - "per_draw_spread_log_mean": average per-draw spread of log(eigs) 
                                      (if compute_spread=True)
        (plus any additional sub-lists or stats you decide to include)
    """
    # -------------------------------------------------------------------------
    # 1) PER-DRAW METRICS
    # -------------------------------------------------------------------------
    ranks_per_draw  = []
    var_all = []
    var_nonzero = []
    traces = []
    trace_norm_len = []
    trace_norm_rank = []

    # Preprocess each draw
    for eigs in eigval_list:
        arr = np.array(eigs, dtype=float)
        # Zero out anything under threshold
        arr = np.where(arr < threshold, 0.0, arr)

        # Rank is just count of nonzero entries
        r = np.count_nonzero(arr)
        ranks_per_draw.append(r)

        # Variance over all (with near-zeros set to 0)
        var_all.append(np.var(arr))

        # Variance over only the strictly nonzero subset
        nz = arr[arr > 0]
        var_nonzero.append(np.var(nz) if nz.size > 1 else 0.0)

        # Trace (sum of eigenvalues)
        tr = arr.sum()
        traces.append(tr)

        # Normalized traces
        length = len(arr) if len(arr) else 1
        trace_norm_len.append(tr / length)
        trace_norm_rank.append(tr / r if r > 0 else 0.0)

    # Aggregate across draws
    max_rank = max(ranks_per_draw) if ranks_per_draw else 0
    avg_rank = np.mean(ranks_per_draw) if ranks_per_draw else 0
    avg_var_all = np.mean(var_all) if var_all else 0
    avg_var_nonzero = np.mean(var_nonzero) if var_nonzero else 0
    avg_trace = np.mean(traces) if traces else 0
    avg_trace_len = np.mean(trace_norm_len) if trace_norm_len else 0
    avg_trace_rank = np.mean(trace_norm_rank) if trace_norm_rank else 0

    # -------------------------------------------------------------------------
    # 2) SPREAD-OF-LOG METRICS (OPTIONAL)
    #    We demonstrate both "per-draw" and "pooled" approaches
    # -------------------------------------------------------------------------
    pooled_spread_log = 0.0
    per_draw_spread_log = []
    
    if compute_spread and len(eigval_list) > 0:
        # Flatten all nonzero eigenvalues across all draws
        all_eigs_concat = np.concatenate([
            np.array(e)[np.array(e) > threshold] for e in eigval_list
        ])
        # Avoid edge cases
        if all_eigs_concat.size > 1:
            # "Pooled" approach: treat all draws as one big set
            logs = np.log(all_eigs_concat)
            pooled_spread_log = np.var(logs)  # or median_abs_deviation(logs), etc.

        # Per-draw approach: each draw -> log(eigs), compute var, then average
        for e in eigval_list:
            e = np.where(e < threshold, 0.0, e)
            nonz = e[e > 0]
            if len(nonz) > 1:
                per_draw_spread_log.append(np.var(np.log(nonz)))
            else:
                per_draw_spread_log.append(0.0)
        per_draw_spread_log_mean = np.mean(per_draw_spread_log) if per_draw_spread_log else 0.0
    else:
        per_draw_spread_log_mean = 0.0

    # -------------------------------------------------------------------------
    # 3) ORGANIZE & RETURN
    # -------------------------------------------------------------------------
    metrics = {
        # Raw per-draw lists
        "QFIM_ranks": ranks_per_draw,
        "var_all_per_draw": var_all,
        "var_nonzero_per_draw": var_nonzero,
        "trace_per_draw": traces,

        # Aggregated
        "max_rank": max_rank,
       
        "avg_var_all": avg_var_all,
        "avg_var_nonzero": avg_var_nonzero,
        "avg_trace": avg_trace,
        "avg_trace_norm_by_len": avg_trace_len,
        "avg_trace_norm_by_rank": avg_trace_rank,

        # Spread-of-log (optional)
        "pooled_spread_log": pooled_spread_log,            # single float
        "per_draw_spread_log_mean": per_draw_spread_log_mean,  # single float
        # The entire list of per-draw spreads
        "per_draw_spread_log": per_draw_spread_log
    }

    return metrics

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



def equation_32(eigvals, threshold=1e-12):
    """
        $$   
        d_{\mathrm{eff}} 
        \;=\; 
        \max_{\theta \,\in\, \Theta} \text{rank}\!\bigl(F(\theta)\bigr).
        $$
    If \( \text{rank}\!\bigl(F(\theta)\bigr)\) is less than \(d\), it signifies 
    that some parameters do not induce linearly independent changes in the 
    quantum state or its measurement outcomes, leading to $d_{\mathrm{eff}}<d$. 
    Consequently, the manifold of states effectively spans fewer directions 
    than the raw parameter count might suggest.
    """
    filtered_vals = [val for val in eigvals if val > threshold]
    
    if len(filtered_vals) == 0:
        # If no eigenvalues are above threshold, dimension is effectively 0
        return 0.0
    
    sum_eigs = np.sum(filtered_vals)
    sum_eigs_sq = np.sum(np.square(filtered_vals))
    
    # Avoid division by zero if sum_eigs_sq happens to be extremely small
    if sum_eigs_sq == 0:
        return 0.0
    
    return (sum_eigs ** 2) / sum_eigs_sq
