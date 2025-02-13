def get_initial_learning_rate_ngd(grads, base_step=1.0, min_lr=1e-3, max_lr=0.2):
    """
    Use normalized gradient approach:
      lr = base_step / ||grads||
    so that the initial step is 'base_step' in parameter space.
    """
    grad_norm = jnp.linalg.norm(grads)
    # if grad_norm == 0, default
    lr = jnp.where(grad_norm > 0, base_step / grad_norm, 0.1)
    lr = jnp.clip(lr, min_lr, max_lr)
    return lr, grad_norm

@jit
def approximate_largest_hessian_eig(params, cost_func, shift=1e-3):
    """
    Approximate the largest diagonal entry of the Hessian
    by parameter-shift second derivatives, 
    then use it as a crude bound on the largest eigenvalue.
    """
    diag_vals = []
    for i in range(len(params)):
        # Shift param i
        params_plus = params.at[i].add( shift )
        params_minus = params.at[i].add(-shift)
        
        cost_plus = cost_func(params_plus)
        cost_minus = cost_func(params_minus)
        cost_center = cost_func(params)
        
        # approx 2nd derivative wrt param i
        second_deriv = (cost_plus - 2*cost_center + cost_minus) / (shift**2)
        diag_vals.append(second_deriv)
        
    max_diag = jnp.max(jnp.abs(jnp.array(diag_vals)))
    # This is a rough approximation of the largest eigenvalue
    return max_diag

def get_initial_lr_hessian(params, cost_func, grads, min_lr=1e-3, max_lr=0.2):
    """
    Use Hessian-based approach for the initial learning rate:
      lr ~ 1 / lambda_max
    where lambda_max is an estimate of the largest Hessian eigenvalue 
    (bounded from above by max diagonal entry).
    """
    hess_bound = approximate_largest_hessian_eig(params, cost_func)
    grad_norm = jnp.linalg.norm(grads)
    # If hess_bound is zero or extremely small, fallback
    lr_approx = jnp.where(hess_bound > 1e-12, 1.0 / hess_bound, 0.1)
    lr_approx = jnp.clip(lr_approx, min_lr, max_lr)
    return lr_approx, grad_norm
def get_initial_lr_line_search(params, cost_func, grads, 
                               test_factors=[0.05, 0.1, 0.2, 0.5],
                               direction=None):
    """
    Attempt a minimal line search for the initial step:
      - Evaluate the cost at a set of scaled steps along -grad 
      - Pick the step that yields the largest cost decrease
    """
    if direction is None:
        direction = -grads  # Negative gradient
    
    grad_norm = jnp.linalg.norm(grads)
    theta_0 = params
    best_lr = 0.0
    best_cost = cost_func(params)  # baseline cost
    
    for f in test_factors:
        trial_step = theta_0 + f * direction
        new_cost = cost_func(trial_step)
        if new_cost < best_cost:
            best_cost = new_cost
            best_lr = f
    
    # If no improvement, fallback
    if best_lr == 0.0:
        best_lr = jnp.array(0.1)
    
    return best_lr, grad_norm


def get_initial_lr_per_param(grads, base_step=1.0, min_lr=1e-4, max_lr=0.2):
    """
    Provide a *vector* of learning rates, one for each parameter,
    inversely proportional to the param's gradient magnitude.
    """
    grad_magnitudes = jnp.abs(grads) + 1e-12
    lr_vector = base_step / grad_magnitudes
    lr_vector = jnp.clip(lr_vector, min_lr, max_lr)
    return lr_vector



# combining

def get_initial_lr_backtrack(params, cost_func, grads, 
                             factor=0.5, max_iter=5, start_lr=1.0):
    """
    Start with lr = start_lr, apply it, if cost doesn't improve enough,
    multiply lr by 'factor' (<1) and retry, up to 'max_iter' times.
    """
    cost_0 = cost_func(params)
    direction = -grads
    lr = start_lr
    for _ in range(max_iter):
        new_params = params + lr * direction
        cost_new = cost_func(new_params)
        if cost_new < cost_0:
            return lr, jnp.linalg.norm(grads)
        lr *= factor  # reduce step and try again
    # fallback
    return jnp.array(0.01), jnp.linalg.norm(grads)
