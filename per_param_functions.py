

import jax
import jax.numpy as jnp

def get_initial_lr_per_param(grad_tree, scale_factor=0.1, min_lr=1e-5, max_lr=0.25,
                             outlier_clip=0.95, fudge_scale=1.0, debug=False):
    """
    Compute initial per-parameter learning rates using robust statistics.

    Each parameter's learning rate is computed as:
        lr = (scale_factor * scale_stat) / (|grad| + fudge)
    where:
      - |grad| is the absolute value of the gradient for each parameter.
      - scale_stat is the median of all |grad| values, which serves as a robust measure
        of the typical gradient magnitude.
      - fudge is a stabilization constant computed as:
            fudge = fudge_scale * (scale_factor / max_lr) * scale_stat
        Its role is to avoid division by zero and to moderate the sensitivity of lr to small gradients.
        When |grad| is zero, lr ≈ max_lr.
    
    Tuning the Learning Rate Spread:
      - The fudge term in the denominator smooths the differences between parameters.
      - A larger fudge (i.e. setting fudge_scale > 1) reduces the effect of differences in |grad|,
        leading to a narrower (more compressed) distribution of learning rates.
      - A smaller fudge (i.e. setting fudge_scale < 1) makes the function more sensitive to differences
        in |grad|, hence widening the distribution of learning rates.

    Arguments:
      grad_tree    : A PyTree of gradients (e.g., nested dicts/lists of jax.numpy arrays).
      scale_factor : Baseline multiplier such that if |grad| ≈ scale_stat, then lr ≈ scale_factor.
      min_lr       : Minimum allowed learning rate after clamping.
      max_lr       : Maximum allowed learning rate after clamping.
      outlier_clip : Quantile (0, 1] to clip extreme |grad| values when computing robust statistics,
                     reducing the impact of outliers.
      fudge_scale  : A multiplier to adjust the fudge constant.
                     - fudge_scale > 1 narrows the lr distribution.
                     - fudge_scale < 1 widens the lr distribution.
      debug        : If True, prints detailed diagnostic information.

    Returns:
      A PyTree matching the structure of grad_tree, where each leaf contains the computed learning rate.
    """

    # --- Step 1: Flatten the gradient PyTree ---
    grad_leaves, tree_def = jax.tree_flatten(grad_tree)
    if not grad_leaves:
        raise ValueError("grad_tree is empty; no gradients provided.")

    # --- Step 2: Concatenate gradients and compute absolute values ---
    all_grads = jnp.concatenate([jnp.ravel(g) for g in grad_leaves])
    all_abs = jnp.abs(all_grads)

    # --- Step 3: Compute robust statistics ---
    # Optionally, clip extreme gradient values at the outlier_clip quantile
    if outlier_clip is not None and 0 < outlier_clip < 1:
        high_threshold = jnp.quantile(all_abs, outlier_clip)
        abs_for_stats = jnp.clip(all_abs, a_min=0, a_max=high_threshold)
    else:
        abs_for_stats = all_abs

    # Compute the median (scale_stat) which is our robust measure for typical gradient size.
    median_abs = jnp.quantile(abs_for_stats, 0.5)
    # Additionally compute the 25th and 75th percentiles and IQR for diagnostics.
    q1 = jnp.quantile(abs_for_stats, 0.25)
    q3 = jnp.quantile(abs_for_stats, 0.75)
    iqr = q3 - q1

    scale_stat = median_abs  # This is our typical gradient magnitude.

    # In case scale_stat is zero (e.g., if many gradients are zero), choose the smallest nonzero gradient.
    if scale_stat == 0:
        nonzero_abs = all_abs[all_abs > 0]
        scale_stat = jnp.min(nonzero_abs) if nonzero_abs.size > 0 else 0.0

    # --- Step 4: Compute the fudge factor ---
    # Base fudge: (scale_factor / max_lr) * scale_stat
    # Multiply by fudge_scale to give direct control over the spread.
    # A larger fudge reduces sensitivity (narrowing the lr spread), while a smaller fudge enhances differences.
    fudge = fudge_scale * (scale_factor / max_lr) * scale_stat

    # --- Step 5: Compute raw learning rates ---
    # The formula is:
    #    lr = (scale_factor * scale_stat) / (|grad| + fudge)
    # If |grad| is around scale_stat, then lr ≈ scale_factor.
    # Smaller |grad| leads to larger lr (up to max_lr), and larger |grad| leads to smaller lr.
    lr_all = scale_factor * scale_stat / (all_abs + fudge)

    # --- Step 6: Clamp learning rates ---
    lr_clipped = jnp.clip(lr_all, a_min=min_lr, a_max=max_lr)

    # --- Step 7: Restore original PyTree structure ---
    lr_leaves = []
    idx = 0
    for g in grad_leaves:
        size = g.size
        segment = lr_clipped[idx: idx + size].reshape(g.shape)
        lr_leaves.append(segment)
        idx += size
    lr_tree = jax.tree_unflatten(tree_def, lr_leaves)

    # --- Optional Debug Information ---
    if debug:
        med_val = float(median_abs)
        iqr_val = float(iqr)
        max_grad = float(jnp.max(all_abs))
        min_grad = float(jnp.min(all_abs))
        mean_grad = float(jnp.mean(all_abs))
        max_lr_val = float(jnp.max(lr_clipped))
        min_lr_val = float(jnp.min(lr_clipped))
        mean_lr_val = float(jnp.mean(lr_clipped))
        print("[get_initial_lr_per_param] Gradient stats: min |g| = {:.2e}, median |g| = {:.2e}, mean |g| = {:.2e}, max |g| = {:.2e}, IQR = {:.2e}"
              .format(min_grad, med_val, mean_grad, max_grad, iqr_val))
        print("[get_initial_lr_per_param] Using scale_stat (median) = {:.2e}, fudge = {:.2e}"
              .format(float(scale_stat), float(fudge)))
        print("[get_initial_lr_per_param] LR (raw): min = {:.2e}, max = {:.2e}".format(float(jnp.min(lr_all)), float(jnp.max(lr_all))))
        print("[get_initial_lr_per_param] LR (clipped to [{}, {}]): min = {:.2e}, mean = {:.2e}, max = {:.2e}"
              .format(min_lr, max_lr, min_lr_val, mean_lr_val, max_lr_val))
        print("Final per-parameter learning rates:", lr_tree)

    return lr_tree


import jax
import jax.numpy as jnp

def get_initial_lr_per_param(grad_tree, scale_factor=0.1, min_lr=1e-5, max_lr=0.25,
                             outlier_clip=0.95, fudge_scale=1.0, debug=False):
    """
    Compute initial per-parameter learning rates using robust statistics.

    Each parameter's learning rate is computed as:
        lr = (scale_factor * scale_stat) / (|grad| + fudge)

    where:
      - |grad| is the absolute value of each parameter's gradient.
      - scale_stat is the median of |grad| over all parameters; it serves as a robust measure
        of the "typical" gradient magnitude.
      - fudge is calculated as:
            fudge = fudge_scale * (scale_factor / max_lr) * scale_stat
        This prevents division by zero and tempers the sensitivity to small gradients,
        ensuring that when |grad| is near zero, lr is approximately max_lr.

    The 'outlier_clip' parameter is used to clip extreme values when computing the median.
      - outlier_clip = 1.0 implies no clipping (all values are used).
      - outlier_clip < 1.0 clips values above the specified quantile.
    Note: In cases where the median is unaffected by clipping, different outlier_clip values
    may result in the same learning rates.

    Arguments:
      grad_tree    : A PyTree of gradients (e.g., nested dicts/lists of jax.numpy arrays).
      scale_factor : Baseline multiplier; if |grad| ≈ scale_stat, then lr ≈ scale_factor.
      min_lr       : Minimum allowed learning rate after clamping.
      max_lr       : Maximum allowed learning rate after clamping.
      outlier_clip : Quantile (between 0 and 1] used to clip extreme |grad| values.
      fudge_scale  : Multiplier for the fudge term, controlling the narrowness of the lr distribution.
                     - fudge_scale > 1 narrows the distribution.
                     - fudge_scale < 1 widens the distribution.
      debug        : If True, prints detailed diagnostics.

    Returns:
      A PyTree matching the structure of grad_tree, with computed learning rates.
    """

    # --- Step 1: Flatten the gradient PyTree ---
    # Use jax.tree_util.tree_flatten to avoid deprecation warnings.
    grad_leaves, tree_def = jax.tree_util.tree_flatten(grad_tree)
    if not grad_leaves:
        raise ValueError("grad_tree is empty; no gradients provided.")

    # --- Step 2: Concatenate gradients and compute absolute values ---
    all_grads = jnp.concatenate([jnp.ravel(g) for g in grad_leaves])
    all_abs = jnp.abs(all_grads)

    # --- Step 3: Compute robust statistics ---
    # Optionally clip out extreme gradient values to compute robust statistics.
    if outlier_clip is not None and 0 < outlier_clip < 1:
        high_threshold = jnp.quantile(all_abs, outlier_clip)  # e.g., 95th percentile for outlier_clip=0.95
        abs_for_stats = jnp.clip(all_abs, a_min=0, a_max=high_threshold)
    else:
        abs_for_stats = all_abs

    # Compute median (scale_stat) as the robust typical gradient magnitude.
    median_abs = jnp.quantile(abs_for_stats, 0.5)
    # Compute additional diagnostics: 25th and 75th percentiles and IQR.
    q1 = jnp.quantile(abs_for_stats, 0.25)
    q3 = jnp.quantile(abs_for_stats, 0.75)
    iqr = q3 - q1

    scale_stat = median_abs  # This is our robust measure (median) of the gradients.

    # If the median is zero (e.g., many zeros), fall back to the smallest nonzero gradient.
    if scale_stat == 0:
        nonzero_abs = all_abs[all_abs > 0]
        scale_stat = jnp.min(nonzero_abs) if nonzero_abs.size > 0 else 0.0

    # --- Step 4: Compute the fudge factor ---
    # Base fudge ensures that when |grad| = 0, lr ≈ max_lr.
    # Multiplying by fudge_scale gives you direct control over the lr distribution spread.
    fudge = fudge_scale * (scale_factor / max_lr) * scale_stat

    # --- Step 5: Compute the raw learning rates ---
    lr_all = scale_factor * scale_stat / (all_abs + fudge)

    # --- Step 6: Clamp the learning rates ---
    lr_clipped = jnp.clip(lr_all, a_min=min_lr, a_max=max_lr)

    # --- Step 7: Restore the original PyTree structure ---
    lr_leaves = []
    idx = 0
    for g in grad_leaves:
        size = g.size
        segment = lr_clipped[idx: idx + size].reshape(g.shape)
        lr_leaves.append(segment)
        idx += size
    # Use jax.tree_util.tree_unflatten to reassemble the PyTree.
    lr_tree = jax.tree_util.tree_unflatten(tree_def, lr_leaves)

    # --- Optional Debug Information ---
    if debug:
        med_val = float(median_abs)
        iqr_val = float(iqr)
        max_grad = float(jnp.max(all_abs))
        min_grad = float(jnp.min(all_abs))
        mean_grad = float(jnp.mean(all_abs))
        max_lr_val = float(jnp.max(lr_clipped))
        min_lr_val = float(jnp.min(lr_clipped))
        mean_lr_val = float(jnp.mean(lr_clipped))
        print("[get_initial_lr_per_param] Gradient stats: min |g| = {:.2e}, median |g| = {:.2e}, mean |g| = {:.2e}, max |g| = {:.2e}, IQR = {:.2e}"
              .format(min_grad, med_val, mean_grad, max_grad, iqr_val))
        print("[get_initial_lr_per_param] Using scale_stat (median) = {:.2e}, fudge = {:.2e}"
              .format(float(scale_stat), float(fudge)))
        print("[get_initial_lr_per_param] LR (raw): min = {:.2e}, max = {:.2e}".format(float(jnp.min(lr_all)), float(jnp.max(lr_all))))
        print("[get_initial_lr_per_param] LR (clipped to [{}, {}]): min = {:.2e}, mean = {:.2e}, max = {:.2e}"
              .format(min_lr, max_lr, min_lr_val, mean_lr_val, max_lr_val))
        print("Final per-parameter learning rates:", lr_tree)

    return lr_tree
import jax
import jax.numpy as jnp

def get_initial_lr_per_param(grad_tree, scale_factor=0.1, min_lr=1e-5, max_lr=0.25,
                             outlier_clip=0.95, fudge_scale=1.0, debug=False):
    """
    Compute initial per-parameter learning rates from gradients using robust statistics,
    with a modification that allows controlling the spread of learning rates without shifting
    the center (i.e. the median lr remains constant).

    The basic idea is that we first compute the robust typical gradient magnitude,
      scale_stat = median(|grad|).
    Then, instead of directly using the raw |grad| in the scaling formula, we define an
    "effective gradient" as:
  
        effective_grad = (|grad| + fudge_scale * scale_stat) / (1 + fudge_scale)
    
    Note that:
      - When |grad| == scale_stat, effective_grad = scale_stat for any fudge_scale.
      - For |grad| far from scale_stat, effective_grad is pulled toward scale_stat by an
        amount determined by fudge_scale.
      - Increasing fudge_scale “narrows” the distribution by pulling values toward the median
        (reducing differences), while decreasing fudge_scale allows a wider spread.
    
    Finally, we define each learning rate as:
      
        lr = (scale_factor * scale_stat) / effective_grad

    Thus, if |grad| equals scale_stat (the typical gradient), then lr = scale_factor.
    Parameters with gradients smaller than scale_stat receive a larger lr (up to max_lr),
    and those with larger gradients receive a smaller lr.

    Arguments:
      grad_tree    : A PyTree of gradients (e.g. nested dicts/lists of jax.numpy arrays).
      scale_factor : Baseline multiplier; for a typical gradient (|grad| ≈ scale_stat),
                     the learning rate will be approximately scale_factor.
      min_lr       : Minimum allowed learning rate (after clamping).
      max_lr       : Maximum allowed learning rate (after clamping).
      outlier_clip : Quantile (in (0,1] or None) used to clip extreme |grad| values when
                     computing the median (scale_stat). (Clipping only affects the robust
                     statistic computation.)
      fudge_scale  : A multiplier controlling the spread of learning rates. Its effect is:
                     - fudge_scale = 1.0 yields a baseline distribution.
                     - fudge_scale > 1 pulls effective_grad more toward scale_stat,
                        narrowing the distribution (i.e. reducing deviations from scale_factor).
                     - fudge_scale < 1 lets |grad| vary more, widening the distribution.
                     Crucially, because effective_grad equals scale_stat for |grad| = scale_stat,
                     the baseline (median) learning rate remains fixed.
      debug        : If True, print diagnostic statistics.

    Returns:
      A PyTree with the same structure as grad_tree, where each leaf is replaced by its computed learning rate.
    """

    # --- Step 1: Flatten the gradient PyTree ---
    # Use jax.tree_util.tree_flatten to avoid deprecation warnings.
    grad_leaves, tree_def = jax.tree_util.tree_flatten(grad_tree)
    if not grad_leaves:
        raise ValueError("grad_tree is empty; no gradients provided.")

    # --- Step 2: Concatenate gradients and compute absolute values ---
    # all_grads: 1D array of all gradient values from the PyTree.
    all_grads = jnp.concatenate([jnp.ravel(g) for g in grad_leaves])
    # all_abs: absolute value of all gradients.
    all_abs = jnp.abs(all_grads)

    # --- Step 3: Compute robust statistics on gradients ---
    # Optionally clip extreme values before computing robust statistics.
    if outlier_clip is not None and 0 < outlier_clip < 1:
        high_threshold = jnp.quantile(all_abs, outlier_clip)  # e.g., the 95th percentile for outlier_clip=0.95.
        abs_for_stats = jnp.clip(all_abs, a_min=0, a_max=high_threshold)
    else:
        abs_for_stats = all_abs

    # Compute the median of |grad| (a robust measure of the typical gradient magnitude).
    median_abs = jnp.quantile(abs_for_stats, 0.5)
    # Additional diagnostic stats: 25th and 75th percentiles and IQR.
    q1 = jnp.quantile(abs_for_stats, 0.25)
    q3 = jnp.quantile(abs_for_stats, 0.75)
    iqr = q3 - q1

    # scale_stat is defined as the median of the absolute gradients.
    scale_stat = median_abs
    # In case the median is zero (e.g., if many gradients are zero), pick the smallest nonzero value.
    if scale_stat == 0:
        nonzero_abs = all_abs[all_abs > 0]
        scale_stat = jnp.min(nonzero_abs) if nonzero_abs.size > 0 else 0.0

    # --- Step 4: Define the effective gradient ---
    # effective_grad is a combination of the actual |grad| and the median (scale_stat):
    #   effective_grad = (|grad| + fudge_scale * scale_stat) / (1 + fudge_scale)
    # Note: When |grad| equals scale_stat, effective_grad equals scale_stat regardless of fudge_scale.
    # This preserves the baseline learning rate for typical gradients.
    # For |grad| much lower than scale_stat, effective_grad becomes closer to scale_stat as fudge_scale increases,
    # which narrows the spread of learning rates.
    #
    # Compute effective gradient for each element (flattened form).
    effective_grad = (all_abs + fudge_scale * scale_stat) / (1.0 + fudge_scale)

    # --- Step 5: Compute raw per-parameter learning rates ---
    # The learning rate for each parameter is computed as:
    #   lr = (scale_factor * scale_stat) / effective_grad
    # So for a parameter with |grad| = scale_stat, lr = scale_factor.
    lr_all = scale_factor * scale_stat / effective_grad

    # --- Step 6: Clamp the learning rates ---
    # Ensure lr is between min_lr and max_lr.
    lr_clipped = jnp.clip(lr_all, a_min=min_lr, a_max=max_lr)

    # --- Step 7: Restore the learning rate values to the original PyTree structure ---
    lr_leaves = []
    idx = 0
    for g in grad_leaves:
        size = g.size
        segment = lr_clipped[idx: idx + size].reshape(g.shape)
        lr_leaves.append(segment)
        idx += size
    lr_tree = jax.tree_util.tree_unflatten(tree_def, lr_leaves)

    # --- Optional Debug Information ---
    if debug:
        med_val = float(median_abs)
        iqr_val = float(iqr)
        max_grad = float(jnp.max(all_abs))
        min_grad = float(jnp.min(all_abs))
        mean_grad = float(jnp.mean(all_abs))
        max_lr_val = float(jnp.max(lr_clipped))
        min_lr_val = float(jnp.min(lr_clipped))
        mean_lr_val = float(jnp.mean(lr_clipped))
        print("[get_initial_lr_per_param] Gradient stats: min |g| = {:.2e}, median |g| = {:.2e}, mean |g| = {:.2e}, max |g| = {:.2e}, IQR = {:.2e}"
              .format(min_grad, med_val, mean_grad, max_grad, iqr_val))
        print("[get_initial_lr_per_param] Using scale_stat (median) = {:.2e}".format(float(scale_stat)))
        print("[get_initial_lr_per_param] Effective gradient sample: {:.2e}".format(float(effective_grad[0])))
        print("[get_initial_lr_per_param] LR (raw): min = {:.2e}, max = {:.2e}".format(float(jnp.min(lr_all)), float(jnp.max(lr_all))))
        print("[get_initial_lr_per_param] LR (clipped to [{}, {}]): min = {:.2e}, mean = {:.2e}, max = {:.2e}"
              .format(min_lr, max_lr, min_lr_val, mean_lr_val, max_lr_val))
        print("Final per-parameter learning rates:", lr_tree)

    return lr_tree
