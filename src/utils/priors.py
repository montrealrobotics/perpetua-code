from jax import Array 
from jax.typing import ArrayLike 

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

from src.utils.math_utils import clip_log_prob


@jax.jit
def log_s_exponential_single(t: ArrayLike, lambda_: float) -> Array:
    """
    Compute the log survival function of an exponential distribution for a single input.

    Args:
        t (float): Single time at which to evaluate the survival function.
        lambda_ (float): The rate parameter of the exponential distribution.

    Returns:
        Array: The log survival function evaluated at t.
    """
    log_survival = -lambda_ * t

    return clip_log_prob(log_survival)


def log_survival_exponential(t: ArrayLike, lambda_: float) -> Array:
    """
    Compute the log survival function of an exponential distribution for multiple inputs.

    Args:
        t (ArrayLike): Times at which to evaluate the survival function.
        lambda_ (float): The rate parameter of the exponential distribution.

    Returns:
        Array: The log survival function evaluated at t.
    """
    # Vectorize the computation over multiple inputs
    times = jnp.atleast_1d(t)
    return jax.vmap(log_s_exponential_single, in_axes=(0, None))(times, lambda_).squeeze()

@jax.jit
def log_s_weibull_single(t: ArrayLike, k: float, lambda_: float) -> Array:
    """
    Compute the log survival function of a Weibull distribution for a single input.

    Args:
        t (float): Single time at which to evaluate the survival function.
        k (float): The shape parameter of the Weibull distribution.
        lambda_ (float): The scale parameter of the Weibull distribution.

    Returns:
        Array: The log survival function evaluated at t.
    """
    epsilon = 1e-10
    log_t_lambda = jnp.log(t + epsilon) - jnp.log(lambda_ + epsilon)
    log_term = k * log_t_lambda

    # Clamp log_term to avoid extreme values that lead to 0 or 1 when exponentiated
    log_term = jnp.clip(log_term, -jnp.log(jnp.finfo(jnp.float32).max), jnp.log(jnp.finfo(jnp.float32).max))
    log_survival = -jnp.exp(log_term)

    return clip_log_prob(log_survival)


def log_survival_weibull(t: ArrayLike, k: float, lambda_: float) -> Array:
    """
    Compute the log survival function of a Weibull distribution for multiple inputs.

    Args:
        t (ArrayLike): Times at which to evaluate the survival function.
        k (float): The shape parameter of the Weibull distribution.
        lambda_ (float): The scale parameter of the Weibull distribution.

    Returns:
        Array: The log survival function evaluated at t.
    """
    # Vectorize the computation over multiple inputs
    times = jnp.atleast_1d(t)
    return jax.vmap(log_s_weibull_single, in_axes=(0, None, None))(times, k, lambda_).squeeze()


@jax.jit
def log_s_lognormal_single(t: ArrayLike, logmu: float, std: float) -> Array:
    """
    Compute the log survival function of a log-normal distribution for a single input.

    Args:
        t (float): Single time at which to evaluate the survival function.
        logmu (float): The mean of the underlying normal distribution (in log space).
        std (float): The standard deviation of the underlying normal distribution (in log space).

    Returns:
        Array: The log survival function evaluated at t.
    """
    # Compute the standard normal survival argument
    z = (jnp.log(t) - logmu) / std
    
    # Use the standard normal survival function
    log_survival = norm.logsf(z)
    
    return clip_log_prob(log_survival)

def log_survival_lognormal(t: ArrayLike, logmu: float, std: float) -> Array:
    """
    Compute the log survival function of a log-normal distribution for multiple inputs.

    Args:
        t (ArrayLike): Times at which to evaluate the survival function.
        logmu (float): The mean of the underlying normal distribution (in log space).
        std (float): The standard deviation of the underlying normal distribution (in log space).

    Returns:
        Array: The log survival function evaluated at t.
    """
    # Vectorize the computation over multiple inputs
    times = jnp.atleast_1d(t)
    return jax.vmap(log_s_lognormal_single, in_axes=(0, None, None))(times, logmu, std).squeeze()