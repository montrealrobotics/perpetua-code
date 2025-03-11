from typing import Tuple, Dict, List

from functools import partial

from jax import Array
from jax.typing import ArrayLike

import jax.numpy as jnp
import jax
from jax.random import PRNGKey, normal
from jax.scipy.special import gamma

from jax._src.tree_util import Partial

from src.utils.filter_state import FilterState
from src.filters import MixtureEmergenceFilters, MixturePersistenceFilters
from src.utils.filter_test_utils import run_mixture_persistence_filters, run_mixture_emergence_filters
from src.utils.logger import logger


@jax.jit
def compute_log_likelihood_pf(
    T: ArrayLike, obs: ArrayLike, obs_times: ArrayLike, p_m: ArrayLike, p_f: ArrayLike
) -> Array:
    """
    Compute the log-likelihood of the observations given a specified time threshold.
    This method is used for the persistence filter.

    This function calculates the log-likelihood of a sequence of binary observations
    based on whether the observations occur before or after a given time threshold (T).
    The likelihood is computed using the probabilities of missed detection (P_M) and
    false alarm (P_F).

    Args:
        T (ArrayLike): The time threshold used to separate the observations into two groups.
        obs (ArrayLike): A sequence of binary observations (0 or 1).
        obs_times (ArrayLike): The timestamps corresponding to the observations in obs.
        p_m (ArrayLike): The probability of a missed detection (false negative).
        p_f (ArrayLike): The probability of a false alarm (false positive).

    Returns:
        Array: The total log-likelihood of the observations given the time threshold.
    """
    # Mask for times t_i <= T
    mask_m = obs_times <= T

    # Mask for times t_i > T
    mask_f = obs_times > T

    # Log-likelihood contribution from times t_i <= T
    log_likelihood_m = (1 - obs) * jnp.log(p_m) + obs * jnp.log(1 - p_m)

    # Log-likelihood contribution from times t_i > T
    log_likelihood_f = obs * jnp.log(p_f) + (1 - obs) * jnp.log(1 - p_f)

    # Combine log-likelihoods (summing over all elements)
    total_log_likelihood = jnp.sum(log_likelihood_m * mask_m + log_likelihood_f * mask_f)

    return total_log_likelihood


@jax.jit
def compute_log_likelihood_ef(
    T: ArrayLike, obs: ArrayLike, obs_times: ArrayLike, p_m: ArrayLike, p_f: ArrayLike
) -> Array:
    """
    Compute the log-likelihood of the observations given a specified time threshold.
    This method is used for the emergence filter.

    This function calculates the log-likelihood of a sequence of binary observations
    based on whether the observations occur before or after a given time threshold (T).
    The likelihood is computed using the probabilities of missed detection (P_M) and
    false alarm (P_F).

    Args:
        T (ArrayLike): The time threshold used to separate the observations into two groups.
        obs (ArrayLike): A sequence of binary observations (0 or 1).
        obs_times (ArrayLike): The timestamps corresponding to the observations in obs.
        p_m (ArrayLike): The probability of a missed detection (false negative).
        p_f (ArrayLike): The probability of a false alarm (false positive).

    Returns:
        Array: The total log-likelihood of the observations given the time threshold.
    """
    # Mask for times t_i <= T
    mask_m = obs_times > T

    # Mask for times t_i > T
    mask_f = obs_times <= T

    # Log-likelihood contribution from times t_i <= T
    log_likelihood_m = (1 - obs) * jnp.log(p_m) + obs * jnp.log(1 - p_m)

    # Log-likelihood contribution from times t_i > T
    log_likelihood_f = obs * jnp.log(p_f) + (1 - obs) * jnp.log(1 - p_f)

    # Combine log-likelihoods (summing over all elements)
    total_log_likelihood = jnp.sum(log_likelihood_m * mask_m + log_likelihood_f * mask_f)

    return total_log_likelihood


@partial(jax.jit, static_argnames=["filter"])
def compute_marginals(
    filter: str, filter_state: FilterState, obs: ArrayLike, obs_times: ArrayLike, p_m: ArrayLike, p_f: ArrayLike
) -> Tuple[float, float]:
    """
    Computes the log-evidence and log-joint evidence of the observations using a persistence or emergence Filter.
    This method is used for the (mixture) persistence filter or emergence filter.

    This function calculates the log-evidence of a sequence of binary observations
    given the times at which they were made, using either a persistence Filter or a emergence Filter.
    The log-evidence is computed by updating the filter with each observation and accumulating the
    log-evidence from the filter.

    Args:
        filter (str): The type of filter, either "persistence" or "emergence".
        filter_state (FilterState): The initial state of the filter.
        obs (ArrayLike): A sequence of binary observations (0 or 1).
        obs_times (ArrayLike): The timestamps corresponding to the observations in obs.
        p_m (ArrayLike): The probability of a missed detection (false negative).
        p_f (ArrayLike): The probability of a false alarm (false positive).

    Returns:
        Tuple[float, float]: The log-evidence and log-joint evidence of the observations given the Persistence or Emergence Filter.
    """

    def update_persistence(
        state: FilterState, obs: ArrayLike, obs_time: float, p_m: ArrayLike, p_f: ArrayLike
    ) -> FilterState:
        return MixturePersistenceFilters.update(state, obs, obs_time, p_m, p_f)

    def update_emergence(
        state: FilterState, obs: ArrayLike, obs_time: float, p_m: ArrayLike, p_f: ArrayLike
    ) -> FilterState:
        return MixtureEmergenceFilters.update(state, obs, obs_time, p_m, p_f)

    def scan_body(
        state: FilterState, data: Tuple[ArrayLike, ArrayLike], f: str, p_m: ArrayLike, p_f: ArrayLike
    ) -> FilterState:
        obs, obs_time = data
        new_state = jax.lax.cond(f == "persistence", update_persistence, update_emergence, state, obs, obs_time, p_m, p_f)
        return new_state, None

    final_state, _ = jax.lax.scan(partial(scan_body, f=filter, p_m=p_m, p_f=p_f), filter_state, (obs, obs_times))

    return final_state.log_evidence, final_state.log_joint_evidence


@partial(jax.jit, static_argnames=["filter"])
def compute_log_likelihood(
    filter: str, T: ArrayLike, obs: ArrayLike, obs_times: ArrayLike, p_m: ArrayLike, p_f: ArrayLike
) -> Array:
    """
    Compute the log-likelihood of the observations given a specified time threshold.

    This function calculates the log-likelihood of a sequence of binary observations
    based on whether the observations occur before or after a given time threshold (T).
    The likelihood is computed using the probabilities of missed detection (P_M) and
    false alarm (P_F).

    Args:
        filter (str): The type of filter, either "persistence" or "emergence".
        T (ArrayLike): The time threshold used to separate the observations into two groups.
        obs (ArrayLike): A sequence of binary observations (0 or 1).
        obs_times (ArrayLike): The timestamps corresponding to the observations in obs.
        p_m (ArrayLike): The probability of a missed detection (false negative).
        p_f (ArrayLike): The probability of a false alarm (false positive).

    Returns:
        Array: The total log-likelihood of the observations given the time threshold.
    """
    log_likelihood = jax.lax.cond(
        filter == "persistence", compute_log_likelihood_pf, compute_log_likelihood_ef, T, obs, obs_times, p_m, p_f
    )
    return log_likelihood


@partial(jax.jit, static_argnames=["filter"])
def create_filter(
    filter: str, log_survival: Partial, params: Dict[str, float], pi: ArrayLike, initialization_time: float
) -> FilterState:
    """
    Initialize the filter state based on the selected filter type.

    Args:
        filter (str): The type of filter, either "persistence" or "emergence".
        log_survival (Partial): The log survival function.
        params (Dict[str, float]): The parameters for the filter.
        pi (ArrayLike): The initial probability distribution.
        initialization_time (float): The initial time.

    Returns:
        FilterState: The initialized filter state.
    """
    state = jax.lax.cond(
        filter == "persistence",
        MixturePersistenceFilters.init,
        MixtureEmergenceFilters.init,
        log_survival,
        params,
        pi,
        initialization_time,
    )
    return state


def run_filter(
    filter: str,
    obs: ArrayLike,
    obs_times: ArrayLike,
    p_m: ArrayLike,
    p_f: ArrayLike,
    pi: ArrayLike,
    query_times: ArrayLike,
    log_survival: Partial,
    params: Dict[str, float],
) -> Array:
    """
    Run the filter based on the selected filter type.

    Args:
        filter (str): The type of filter, either "persistence" or "emergence".
        obs (ArrayLike): A sequence of binary observations (0 or 1).
        obs_times (ArrayLike): The timestamps corresponding to the observations in obs.
        p_m (ArrayLike): The probability of a missed detection (false negative).
        p_f (ArrayLike): The probability of a false alarm (false positive).
        pi (ArrayLike): The initial probability distribution.
        query_times (ArrayLike): The timestamps at which the filter should be queried.
        log_survival (Partial): The log survival function.
        params (Dict[str, float]): The parameters for the filter.

    Returns:
        Array: The predictions from the filter.
    """
    if filter == "persistence":
        predictions = run_mixture_persistence_filters(obs, obs_times, p_m, p_f, pi, query_times, log_survival, params)
    else:
        predictions = run_mixture_emergence_filters(obs, obs_times, p_m, p_f, pi, query_times, log_survival, params)

    return predictions


def run_filter_for_sequences(
    filter: str,
    y_bool: List[ArrayLike],
    observation_times: List[ArrayLike],
    query_times: List[ArrayLike],
    pi: ArrayLike,
    log_survival: Partial,
    params: Dict[str, float],
    p_m: ArrayLike,
    p_f: ArrayLike,
) -> List[Array]:
    """
    Runs the filter on each chain and returns the resulting beliefs.

    Args:
        filter (str): The type of filter, either "persistence" or "emergence".
        y_bool (List[ArrayLike]): Sequences of binary observations.
        observation_times (List[ArrayLike]): Sequences of observation timestamps.
        query_times (List[ArrayLike]): The timestamps at which the filter should be queried.
        pi (ArrayLike): The initial probability distribution.
        log_survival (Partial): The log survival function.
        params (Dict[str, float]): The parameters for the filter.
        p_m (ArrayLike): The probability of a missed detection (false negative).
        p_f (ArrayLike): The probability of a false alarm (false positive).

    Returns:
        List[Array]: A list containing the beliefs from each chain.
    """
    n_chains = len(y_bool)
    belief_list = []
    for c in range(n_chains):
        belief = run_filter(filter, y_bool[c], observation_times[c], p_m, p_f, pi, query_times[c], log_survival, params)
        belief_list.append(belief)

    return belief_list


@jax.jit
def lognormal_pdf(x: ArrayLike, mu: float, sigma: float) -> Array:
    """
    Compute the log-pdf of a log-normal distribution.

    Args:
        x (ArrayLike): The input values.
        mu (float): The mean of the log-normal distribution.
        sigma (float): The standard deviation of the log-normal distribution.

    Returns:
        Array: The log-pdf values.
    """
    return jnp.exp(-0.5 * (((jnp.log(x) - mu) / sigma) ** 2) - jnp.log(x * sigma * jnp.sqrt(2 * jnp.pi)))


@jax.jit
def perturb(params: ArrayLike, scale: float, key: PRNGKey) -> Array:
    """
    Apply perturbation to parameters.

    Args:
        params (jax.numpy.ndarray): Parameters to perturb.

    Returns:
        jax.numpy.ndarray: Perturbed parameters.
    """
    # Compute the perturbation value (a percentage of the initial value)
    current_scale = params * scale

    # Generate random noise
    key, subkey = jax.random.split(key)
    noise = normal(subkey, shape=params.shape) * current_scale

    # Prevent params from being negative
    return jnp.clip(params + noise, 1e-3), key


def print_results(params: Dict[str, int], iteration: int, loss: float, prior: str) -> None:
    """
    Print the results of the EM algorithm for the current iteration.

    Parameters:
        lambda_est (float): Estimated value of the rate parameter.
    """
    jnp.set_printoptions(precision=4)
    if prior == "exponential":
        lambda_hat = params["lambda_"]
        pi_hat = params["pi"]
        logger.info("=" * 50)
        logger.info(f"The value of λ at iteration {iteration} is: {lambda_hat}")
        logger.info(f"The value of π at iteration {iteration} is: {pi_hat}")
        logger.info(f"The mean time at iteration {iteration} is: {1 / lambda_hat}s")
        logger.info(f"The median (half-life) time at iteration {iteration} is: {jnp.log(2) / lambda_hat}s")
        logger.info(f"The variance at iteration {iteration} is: {1 / lambda_hat**2}s")
        logger.info(f"The log-evidence log(p(Y_1:N)) at iteration {iteration} is: {loss:.3f}")
        logger.info("=" * 50)
    elif prior == "lognorm":
        mu_hat = params["mu"]
        std_hat = params["std"]
        pi_hat = params["pi"]
        var_hat = std_hat**2
        logger.info("=" * 50)
        logger.info(f"The value of (μ, σ) at iteration {iteration} is: ({mu_hat}, {std_hat})")
        logger.info(f"The value of π at iteration {iteration} is: {pi_hat}")
        logger.info(f"The mean time at iteration {iteration} is: {jnp.exp(mu_hat + (var_hat / 2))}s")
        logger.info(f"The median (half-life) time at iteration {iteration} is: {jnp.exp(mu_hat)}s")
        logger.info(
            f"The variance at iteration {iteration} is: {(jnp.exp(var_hat) - 1) * jnp.exp(2 * mu_hat + var_hat)}"
        )
        logger.info(f"The log-evidence log(p(Y_1:N)) at iteration {iteration} is: {loss:.3f}")
        logger.info("=" * 50)
    elif prior == "weibull":
        k_hat = params["k"]
        lambda_hat = params["lambda"]
        pi_hat = params["pi"]
        logger.info("=" * 50)
        logger.info(f"The value of (k, λ) at iteration {iteration} is: ({k_hat}, {lambda_hat})")
        logger.info(f"The value of π at iteration {iteration} is: {pi_hat}")
        logger.info(f"The mean time at iteration {iteration} is: {lambda_hat * gamma(1 + 1 / k_hat)}s")
        logger.info(
            f"The median (half-life) time at iteration {iteration} is: {lambda_hat * (jnp.log(2)) ** (1 / k_hat)}s"
        )
        logger.info(
            f"The variane at iteration {iteration} is: {lambda_hat ** 2 * (gamma(1 + 2 / k_hat) - (gamma(1 + 1 / k_hat)) ** 2)}"
        )
        logger.info(f"The log-evidence log(p(Y_1:N)) at iteration {iteration} is: {loss}")
        logger.info("=" * 50)
    else:
        logger.error(f"Invalid prior option: {prior}")
        raise ValueError(f"Invalid prior option: {prior}")
