from jax import Array 
from jax.typing import ArrayLike 

from functools import partial
import jax
import jax.numpy as jnp

def bool_ifelse(boolean_or_boolfloat: ArrayLike, if_true: ArrayLike, if_false: ArrayLike) -> Array:
    boolean_or_boolfloat = jnp.asarray(boolean_or_boolfloat).astype(jnp.float32)

    return boolean_or_boolfloat * if_true + (1 - boolean_or_boolfloat) * if_false

@jax.jit
def logsumexp(a: ArrayLike, b: ArrayLike) -> Array:
    """
    Computes a stable logsumexp for two arrays element-wise:
    log(exp(a) + exp(b)) in a numerically stable way.

    Args:
        a (jnp.ndarray): First array of values.
        b (jnp.ndarray): Second array of values.

    Returns:
        jnp.ndarray: The result of the stable logsumexp for each element of a and b.
    """
    # Compute the maximum value for stability
    max_val = jnp.maximum(a, b)
    
    # Compute the logsumexp using the stable formula
    return jnp.log(jnp.exp(a - max_val) + jnp.exp(b - max_val)) + max_val

@partial(jax.jit, static_argnums=(1,))
def split_key(key, num_keys):
    key, *rng = jax.random.split(key, num_keys + 1)
    rng = jnp.reshape(jnp.stack(rng), (num_keys, 2))
    return key, rng

