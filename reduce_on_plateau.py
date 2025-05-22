from typing import NamedTuple, Optional

import chex
import jax
import jax.numpy as jnp
from optax._src import base
from optax._src import numerics
import optax.tree_utils as otu


class ReduceLROnPlateauState(NamedTuple):
  """State for the ReduceLROnPlateau callback."""

  scale: chex.Array
  best_value: chex.Array
  plateau_count: chex.Array  # shape=(), dtype=jnp.int32
  cooldown_count: chex.Array  # shape=(), dtype=jnp.int32
  count: chex.Array  # shape=(), dtype=jnp.int32
  avg_value: chex.Array


def reduce_on_plateau(
    factor: float = 0.1,
    patience: int = 10,
    rtol: float = 1e-4,
    atol: float = 0.0,
    cooldown: int = 0,
    accumulation_size: int = 1,
    min_scale: float = 0.0,
    
    warmup_steps: int = 0,
) -> base.GradientTransformationExtraArgs:
    """Reduce learning rate when a metric has stopped improving, with optional warmup.

    Args:
      factor: Factor by which to reduce the learning rate (0 < factor < 1).
      patience: Number of iterations with no improvement before reducing.
      rtol: Relative tolerance for new optimum.
      atol: Absolute tolerance for new optimum.
      cooldown: Iterations to wait after a reduction.
      accumulation_size: Number of values to aggregate before checking plateau.
      min_scale: Minimum scale factor.
      warmup_steps: Number of initial steps to skip any reduction logic.
    """
    # Parameter checking
    if factor <= 0.0 or factor >= 1.0:
        raise ValueError(f"factor must be in (0,1), got {factor}.")
    if rtol < 0.0 or atol < 0.0:
        raise ValueError(f"rtol and atol must be non-negative, got rtol={rtol}, atol={atol}.")
    if rtol == 0.0 and atol == 0.0:
        raise ValueError("At least one of rtol or atol must be positive.")

    def init_fn(params) -> ReduceLROnPlateauState:
        params_dtype = otu.tree_dtype(params, "lowest")
        return ReduceLROnPlateauState(
            # best_value=jnp.asarray(jnp.inf, dtype=jnp.float32),
            best_value=jnp.asarray(float("inf")),
            plateau_count=jnp.asarray(0, jnp.int32),
            scale=jnp.asarray(1.0, dtype=params_dtype),
            cooldown_count=jnp.asarray(0, jnp.int32),
            count=jnp.asarray(0, jnp.int32),
            avg_value=jnp.asarray(0.0),
        )

    def _update_scale(state: ReduceLROnPlateauState) -> ReduceLROnPlateauState:
        avg_value = state.avg_value
        has_improved = jnp.where(
            avg_value < (1 - rtol) * state.best_value - atol, 1, 0
        )
        new_best_value = jnp.where(has_improved, avg_value, state.best_value)
        curr_plateau_count = jnp.where(
            has_improved, 0, numerics.safe_increment(state.plateau_count)
        )
        
        
        # jax.debug.print(
        #     "update_scale: improved?={}  plateau_count={}  best_value={}  count={}",
        #     has_improved, state.plateau_count, state.best_value, state.count,
        # )
        def in_cooldown():
            new_plateau_count = jnp.asarray(0, jnp.int32)
            new_scale = state.scale
            new_cooldown_count = state.cooldown_count - 1
            return new_plateau_count, new_scale, new_cooldown_count

        def not_in_cooldown():
            new_plateau_count = jnp.where(
                curr_plateau_count == patience, 0, curr_plateau_count
            )
            new_scale = jnp.maximum(
                jnp.where(
                    curr_plateau_count == patience,
                    state.scale * factor,
                    state.scale,
                ),
                min_scale,
            )
            new_cooldown_count = jnp.where(
                curr_plateau_count == patience, cooldown, 0
            ).astype(jnp.int32)

            return new_plateau_count, new_scale, new_cooldown_count

        new_plateau_count, new_scale, new_cooldown_count = jax.lax.cond(state.cooldown_count > 0, in_cooldown, not_in_cooldown)
        new_state = ReduceLROnPlateauState(
            plateau_count=new_plateau_count,
            best_value=new_best_value,
            scale=new_scale,
            cooldown_count=new_cooldown_count,
            count=jnp.asarray(0, dtype=jnp.int32),
            avg_value=jnp.asarray(0.0),
        )
        return new_state

    def update_fn(
        updates: base.Updates,
        state: ReduceLROnPlateauState,
        params=None,
        *,
        value: float,
        step: Optional[int] = None,
        **extra_args,
    ) -> tuple[base.Params, ReduceLROnPlateauState]:
        del params, extra_args
        # count = state.count
        # Update running average and count
        # new_count = numerics.safe_increment(state.count)
        # new_avg_value = (
        #     count * state.avg_value + jnp.astype(value, state.avg_value.dtype)
        # ) / new_count
        # st = state._replace(count=new_count, avg_value=new_avg)

        # Decide whether to apply plateau logic
        def apply_plateau(s: ReduceLROnPlateauState) -> ReduceLROnPlateauState:
            # print(f"applying plateau")
            # print(f"state: {s}")
            count = state.count
            new_count = numerics.safe_increment(count)
            new_avg_value = (
                count * state.avg_value + jnp.astype(value, state.avg_value.dtype)
            ) / new_count
            new_state = state._replace(avg_value=new_avg_value, count=new_count)

            new_state = jax.lax.cond(
                new_count == accumulation_size, _update_scale, lambda x: x, new_state
            )

            return new_state

        def skip_plateau(s: ReduceLROnPlateauState) -> ReduceLROnPlateauState:
            # print(f"skipping plateau")
            # print(f"state: {s}")
            return s
        # print(f"step: {step}")
        new_state = jax.lax.cond(
            step < warmup_steps,
            skip_plateau,
            apply_plateau,
            operand=state,
        )

        # Scale updates by current scale
        scaled_updates = jax.tree_map(lambda g: new_state.scale * g, updates)
        return scaled_updates, new_state

    return base.GradientTransformationExtraArgs(init_fn, update_fn)


"""


    elif case == 3:
        desc = "masked per group Adam + delayed ReduceLROnPlateau"
        # how many epochs to skip plateau logic
        warmup_steps = int(num_epochs * warmup_fraction)
        print(f"warmup_steps: {warmup_steps}")

        def delayed_reduce_on_plateau(
            *, factor, patience, rtol, atol, cooldown, accumulation_size, min_scale, warmup_steps
        ):
            base_plateau = reduce_on_plateau(
                factor=factor,
                patience=patience,
                rtol=rtol,
                atol=atol,
                cooldown=cooldown,
                accumulation_size=accumulation_size,
                min_scale=min_scale,
            )
            def init_fn(params):
                return base_plateau.init(params)

            def update_fn(updates, state, params=None, *, value, step, **extra):
                # update the running average & count
                new_count = state.count + 1
                new_avg = (state.avg_value * state.count + value) / new_count
                state = state._replace(count=new_count, avg_value=new_avg)

                # now branch in a JAX-friendly way
                def skip_fn(args):
                    
                    return args

                def apply_plateau(args):
                    u, s = args
                    return base_plateau.update(u, s, params=params, value=value)

                return jax.lax.cond(
                    step < warmup_steps,
                    skip_fn,
                    apply_plateau,
                    operand=(updates, state),
                )

            return optax_base.GradientTransformationExtraArgs(init_fn, update_fn)
        def one_group(rate):
            return optax.chain(
                optax.inject_hyperparams(optax.adam)(
                    learning_rate=rate, b1=b1, b2=b2, eps=eps
                ),
                delayed_reduce_on_plateau(
                    factor=factor,
                    patience=patience,
                    rtol=rtol,
                    atol=atol,
                    cooldown=cooldown,
                    accumulation_size=accumulation_size,
                    min_scale=min_scale,
                    warmup_steps=warmup_steps,
                ),
            )

        base_opt = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.masked(one_group(opt_lr["t"]), {"t": True,  "h": False, "J": False}),
            optax.masked(one_group(opt_lr["h"]), {"t": False, "h": True,  "J": False}),
            optax.masked(one_group(opt_lr["J"]), {"t": False, "h": False, "J": True}),
        )
        # requires both value & step
        return desc, wrap(base_opt, passes_value=True, passes_step=True)
"""