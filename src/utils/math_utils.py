from typing import Union, Tuple, Callable
from jax import Array
from jax.typing import ArrayLike

import functools

from jax._src.tree_util import Partial
import jax.lax
import numpy as np

from src.utils.jax_utils import bool_ifelse

import jax.numpy as jnp
from jax._src import dtypes

info = jnp.finfo(dtypes.canonicalize_dtype(jnp.float32))
posinf = info.max
neginf = info.min

LOWER_LOGSPACE_BOUND = jnp.log(1e-10)  # neginf #jnp.log(LOWER_PSPACE_BOUND)
LOWER_PSPACE_BOUND = jax.lax.exp(LOWER_LOGSPACE_BOUND)
HIGHER_PSPACE_BOUND = 0.999999
HIGHER_LOGSPACE_BOUND = jnp.log(HIGHER_PSPACE_BOUND)


def ntn_logspace(arr: ArrayLike) -> Array:
    return jnp.nan_to_num(arr, nan=LOWER_LOGSPACE_BOUND, neginf=LOWER_LOGSPACE_BOUND, posinf=HIGHER_LOGSPACE_BOUND)


def ntn_pspace(arr: ArrayLike) -> Array:
    return jnp.nan_to_num(arr, nan=LOWER_PSPACE_BOUND, neginf=LOWER_PSPACE_BOUND, posinf=HIGHER_PSPACE_BOUND)


def logdiff(logx: ArrayLike, logy: ArrayLike) -> Array:
    """
    Compute the logarithm of the difference of two numbers in a numerically stable way.

    Args:
        logx (ArrayLike): The logarithm of the first number.
        logy (ArrayLike): The logarithm of the second number.

    Returns:
        Array: The logarithm of the difference of the two numbers.

    """

    iseq = jnp.allclose(jnp.exp(logy - logx), jnp.ones_like(logx))

    ifeq = LOWER_LOGSPACE_BOUND
    ifnoteq = ntn_logspace(logx + jnp.log1p(-jnp.exp(logy - logx)))
    return bool_ifelse(iseq, ifeq, ifnoteq)


def logsum(logx: ArrayLike, logy: ArrayLike) -> Array:
    """
    Compute the logarithm of the sum of two numbers in a numerically stable way.

    Args:
        logx (ArrayLike): The logarithm of the first number.
        logy (ArrayLike): The logarithm of the second number.

    Returns:
        Array: The logarithm of the sum of the two numbers.

    """

    # fixme this only supports single-float inputs
    pair = jnp.array([logx, logy])
    logx = pair.max()
    logy = pair.min()

    iseq = jnp.allclose(jnp.exp(logy - logx), jnp.ones_like(logx))

    ifeq = LOWER_LOGSPACE_BOUND
    ifnoteq = ntn_logspace(
        logx + jnp.log1p(jnp.exp(logy - logx)),
    )

    return bool_ifelse(iseq, ifeq, ifnoteq)


def clip_log_prob(prob: ArrayLike) -> Array:
    """
    Clip the log probability to prevent numerical underflow/overflow.

    Args:
        prob (ArrayLike): The log probability to clip.

    Returns:
        Array: The clipped log probability.

    """
    return jnp.clip(prob, LOWER_LOGSPACE_BOUND, HIGHER_LOGSPACE_BOUND)


def clip_prob(prob: jnp.ndarray) -> jnp.ndarray:
    return jnp.clip(prob, LOWER_PSPACE_BOUND, HIGHER_PSPACE_BOUND)


@functools.partial(jax.jit, static_argnums=(1,))
def create_interpolated_array(points: ArrayLike, num_steps: int) -> Tuple[Array, Callable]:
    """
    This function takes an array of points and generates a new array where the values are equally spaced by the specified num_steps. 
    The new array will pass through each of the original points.

    Args:
        points (list or np.ndarray): The original array of points.
        step_size (float): The desired step size between interpolated points.

    Returns:
        np.ndarray: A new array with equally spaced values that passes through the given points.
    """

    points_shifted_left = jnp.concatenate([points[1:], jnp.atleast_1d(points[0])])

    def foreach(p1, p2):
        interpolate = jnp.linspace(p1, p2, num_steps, endpoint=False)
        return interpolate

    interpolated = jax.vmap(foreach)(points[:-1], points_shifted_left[:-1])
    interpolated = interpolated.reshape((points.shape[0] - 1) * num_steps)
    interpolated = jnp.concatenate([interpolated, points[-1][None]])

    def collect(interpolated_array):
        last_element = interpolated_array[-1][None]
        # Substract the element added at the beginning and the last element
        outshape = points.shape[0] - 2
        interpolated_array = interpolated_array[num_steps:-1].reshape(outshape, num_steps, *interpolated_array.shape[1:])
        return jnp.concatenate([interpolated_array[:, 0], last_element]).squeeze()

    return interpolated, Partial(collect)
