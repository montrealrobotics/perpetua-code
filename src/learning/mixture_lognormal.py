
from typing import List, Tuple, Union, Callable, Dict

import time
from functools import partial
from jax.scipy.special import logsumexp, erf

from jax import Array
from jax.typing import ArrayLike

import jax
from jax import random
import jax.numpy as jnp
from jax._src.tree_util import Partial
from src.utils.filter_state import FilterState

import src.utils.priors as priors
from src.learning.learning_utils import run_filter_for_sequences, compute_log_likelihood, create_filter, compute_marginals, print_results, perturb, lognormal_pdf
from src.utils.metrics import report_metrics, compute_aic, compute_bic
from src.utils.viz import Plotter
from src.utils.logger import logger


@jax.jit
def compute_log_t_expectation(mu: float, sigma: float, t_i: float, t_i_1: float) -> Array:
    """
    Computes the integral of the function prior(T) * log(T) from t_i to t_i_1 for a log-normal prior.

    Args:
        mu (float): The mean of the log-normal distribution (in log space).
        sigma (float): The standard deviation of the log-normal distribution (in log space).
        t_i (float): The lower limit of the integration.
        t_i_1 (float): The upper limit of the integration.

    Returns:
        jnp.array: The approximate value of the integral.
    """
    eps = 1e-10  # Small epsilon to avoid division by zero

    # Compute useful constants
    log_c1 = jnp.log(mu) + 0.5 * jnp.log(jnp.pi / 2)
    log_s = jnp.log(sigma)
    two_s_sq = 2 * sigma ** 2
    two_pi = 2 * jnp.pi

    # Compute integration bounds for error function
    int_lower = jnp.log(t_i + eps) - mu
    int_upper = jnp.log(t_i_1 + eps) - mu

    # Compute the exponential terms
    term1 = jnp.exp(log_s + (- (int_lower ** 2) / two_s_sq) - 0.5 * jnp.log(two_pi))
    term2 = -jnp.exp(log_s + (- (int_upper ** 2) / two_s_sq) - 0.5 * jnp.log(two_pi))

    # Compute the error function terms
    term3 = jnp.exp(log_c1 - 0.5 * jnp.log(two_pi)) * erf(int_upper / jnp.sqrt(two_s_sq))
    term4 = -jnp.exp(log_c1 - 0.5 * jnp.log(two_pi)) * erf(int_lower / jnp.sqrt(two_s_sq))

    # Stack terms along the first axis and compute their sum
    terms = jnp.stack([term1, term2, term3, term4], axis=0)
    result = jnp.sum(terms, axis=0)

    # Clip the result to avoid numerical issues
    return jnp.clip(result, 1e-10, None)

@jax.jit
def compute_log_t_mu_expectation(
    mu: ArrayLike, sigma: ArrayLike, mu_mle: ArrayLike, t_i: ArrayLike, t_i_1: ArrayLike) -> Array:
    """
    Computes the integral of the function prior(T) * [log(T) - mu]^2 from t_i to t_i_1. prior(T) is the log-normal prior.

    Args:
        mu (float): The mean of the log-normal distribution.
        sigma (float): The standard deviation of the log-normal distribution.
        mu_mle (float): The MLE of the mean of the log-normal distribution.
        t_i (float): The lower limit of the integration.
        t_i_1 (float): The upper bound to approximate -infinity (default is -1000).

    Returns:
        jnp.array: The approximate value of the integral.
    """
    eps = 1e-10  # Small epsilon to avoid division by zero

    # Compute useful constants
    c1 = sigma**2 / 2
    c2 = sigma / jnp.sqrt(jnp.pi * 2)
    c3 = mu**2 / 2
    c4 = jnp.sqrt(2 / jnp.pi) * mu * sigma
    sqrt_2 = jnp.sqrt(2)

    # Compute integration bounds for the error function
    int_lower = (jnp.log(t_i + eps) - mu) / (sqrt_2 * sigma)
    int_upper = (jnp.log(t_i_1 + eps) - mu) / (sqrt_2 * sigma)
    diff_lower = jnp.log(t_i + eps) - mu
    diff_upper = jnp.log(t_i_1 + eps) - mu

    # Compute the terms
    term3 = jnp.zeros_like(int_upper)

    # Compute -0.5 * int_upper * exp(-int_upper ** 2)
    term1 = c1 * erf(int_upper)
    # Compute 0.25 * c2 * erf(int_upper)
    term2 = -c1 * erf(int_lower)

    # Compute 0.5 * int_lower * exp(-int_lower ** 2)
    term3 = jnp.where(int_upper != jnp.inf, -c2 * jnp.exp(-int_upper ** 2) * diff_upper, term3)
    # Compute -0.25 * c2 * erf(int_lower)
    term4 = c2 * jnp.exp(-int_lower ** 2) * diff_lower

    # Stack terms along the first axis and compute their sum
    terms = jnp.stack([term1, term2, term3, term4], axis=0)
    first_term = jnp.sum(terms, axis=0) 

    # Compute the second term
    term1 = jnp.zeros_like(int_upper)
    term1 = jnp.where(int_upper != jnp.inf, -c4 * jnp.exp(-int_upper ** 2), term1)
    term2 = c4 * jnp.exp(-int_lower ** 2)
    second_term = term1 + term2

    # Compute the third term
    third_term = c3 * (erf(int_upper) - erf(int_lower))
    # compute result integral
    first_integral = first_term + second_term + third_term

    second_integral = compute_log_t_expectation(mu, sigma, t_i, t_i_1) * 2 * mu_mle

    third_integral = (mu_mle**2 / 2) * (erf(int_upper) - erf(int_lower))

    result = first_integral - second_integral + third_integral

    # Clip the result to avoid numerical issues
    return jnp.clip(result, 1e-10, None)


@partial(jax.jit, static_argnames=['filter'])
def process_sequence_mu(filter: str, 
                        obs_times: ArrayLike, 
                        obs: ArrayLike, 
                        pi: ArrayLike, 
                        mu: ArrayLike, 
                        sigma: ArrayLike,
                        state: FilterState, 
                        p_m: ArrayLike, 
                        p_f: ArrayLike):
    """
    Process a single sequence of observations and compute the expected survival times and soft assignments.

    Args:
        filter (str): The filter type to use.
        obs_times (ArrayLike): Array of observation times.
        obs (ArrayLike): Array of observations.
        pi (ArrayLike): Mixing coefficients.
        mu (ArrayLike): Mean parameters for each component.
        sigma (ArrayLike): Standard deviation parameters for each component.
        state (FilterState): The initial state of the filter.
        p_m (ArrayLike): Probability of missed detection.
        p_f (ArrayLike): Probability of false alarm.

    Returns:
        Tuple[Array, Array]: The expected survival times (nu_j) and the expected component assignments (psi_j).
    """
    # Extend the times array with 0 at the beginning and inf at the end
    # To replace inf, we use 
    extended_times = jnp.append(obs_times, jnp.inf)
    extended_times = jnp.insert(extended_times, 0, 0.0)

    ####### E Step #######
    # Compute log-likelihoods
    ll_function = jax.vmap(compute_log_likelihood, in_axes=(None, 0, None, None, None, None))
    log_likelihoods = ll_function(filter, extended_times[:-1], obs, obs_times, p_m, p_f)

    # Vmap along two dims to avoid the for loop
    log_t_all = jax.vmap(compute_log_t_expectation, (0, 0, None, None), 1)

    integral_log_t = jnp.log(log_t_all(mu, sigma, extended_times[:-1], extended_times[1:]))
    log_mu_sum = jnp.log(pi) + logsumexp(log_likelihoods[:, None] + integral_log_t, axis=0)

    # Normalize and compute results
    log_evidence, log_joint_evidence = compute_marginals(filter, state, obs, obs_times, p_m, p_f)
    nu_j = jnp.exp(log_mu_sum - log_evidence)
    ### Compute E_q[Z_kj] for all k and j ###
    psi_j = jnp.exp(log_joint_evidence - log_evidence)
    return nu_j, psi_j

@partial(jax.jit, static_argnames=['filter'])
def process_sequence_var(filter: str, 
                        obs_times: ArrayLike, 
                        obs: ArrayLike, 
                        pi: ArrayLike, 
                        mu: ArrayLike, 
                        sigma: ArrayLike,
                        mu_mle: ArrayLike,
                        state: FilterState, 
                        p_m: ArrayLike, 
                        p_f: ArrayLike):
    """
    Process a single sequence of observations and compute the expected variance.

    Args:
        filter (str): The filter type to use.
        obs_times (ArrayLike): Array of observation times.
        obs (ArrayLike): Array of observations.
        pi (ArrayLike): Mixing coefficients.
        mu (ArrayLike): Mean parameters for each component.
        sigma (ArrayLike): Standard deviation parameters for each component.
        mu_mle (ArrayLike): The MLE of the mean of the log-normal distribution.
        state (FilterState): The initial state of the filter.
        p_m (ArrayLike): Probability of missed detection.
        p_f (ArrayLike): Probability of false alarm.

    Returns:
        Array: The expected variance (kappa_j).
    """
    # Extend the times array with 0 at the beginning and inf at the end
    extended_times = jnp.append(obs_times, jnp.inf)
    extended_times = jnp.insert(extended_times, 0, 0.0)

    ####### E Step #######
    # Compute log-likelihoods
    ll_function = jax.vmap(compute_log_likelihood, in_axes=(None, 0, None, None, None, None))
    log_likelihoods = ll_function(filter, extended_times[:-1], obs, obs_times, p_m, p_f)

    log_t_mu_all = jax.vmap(compute_log_t_mu_expectation, (0, 0, 0, None, None), 1)

    integral_log_t_mu = jnp.log(log_t_mu_all(mu, sigma, mu_mle, extended_times[:-1], extended_times[1:]))
    log_var_sum = jnp.log(pi) + logsumexp(log_likelihoods[:, None] + integral_log_t_mu, axis=0)

    # Normalize and compute results
    log_evidence, _ = compute_marginals(filter, state, obs, obs_times, p_m, p_f)
    kappa_j = jnp.exp(log_var_sum - log_evidence)
    return kappa_j

def em_step(
    filter: str,
    n_sequences: int,
    n_components: int,
    observation_times: List[ArrayLike],
    y_bool: List[ArrayLike],
    state: FilterState,
    pi_hat: ArrayLike,
    mu_hat: ArrayLike,
    sigma_hat: ArrayLike,
    p_m: ArrayLike,
    p_f: ArrayLike,
) -> Union[Array, Array]:
    """
        Perform the E-step and M-step in the EM algorithm for the mixture model with log-normal prior.

        Args:
            filter (str): The filter type to use.
            n_sequences (int): Number of observation sequences.
            n_components (int): Number of components in the mixture model.
            observation_times (List[ArrayLike]): List of arrays of observation times for each sequence.
            y_bool (List[ArrayLike]): List of binary observation arrays for each sequence.
            state (FilterState): The initial state of the filter.
            pi_hat (ArrayLike): Initial mixing coefficients.
            mu_hat (ArrayLike): Initial mean parameters for each component.
            sigma_hat (ArrayLike): Initial standard deviation parameters for each component.
            p_m (ArrayLike): Probability of missed detection.
            p_f (ArrayLike): Probability of false alarm.

        Returns:
            Tuple[Array, Array, Array]: Updated mixing coefficients (pi_bar), mean parameters (mu_bar), and standard deviation parameters (sigma_bar).
        """
    
    nu = jnp.zeros((n_sequences, n_components))
    psi = jnp.zeros((n_sequences, n_components))
    kappa = jnp.zeros((n_sequences, n_components))
    for j in range(n_sequences):
        nu_j, psi_j = process_sequence_mu(filter, observation_times[j], y_bool[j], pi_hat, mu_hat, sigma_hat, state, p_m, p_f)
        nu = nu.at[j].set(nu_j)
        psi = psi.at[j].set(psi_j)

    ####### M Step for mu and pi #######
    pi_bar = jnp.sum(psi, axis=0) / jnp.sum(psi)
    mu_bar = jnp.sum(nu, axis=0) / jnp.sum(psi, axis=0)

    for j in range(n_sequences):
        kappa_j = process_sequence_var(filter, observation_times[j], y_bool[j], pi_hat, mu_hat, sigma_hat, mu_bar, state, p_m, p_f)
        kappa = kappa.at[j].set(kappa_j)

    ####### M Step #######
    sigma_bar = jnp.sqrt(jnp.sum(kappa, axis=0) / jnp.sum(psi, axis=0))

    # Clip the parameters to prevent NaN values
    return jnp.clip(pi_bar, 1e-10, 1), jnp.clip(mu_bar, 1e-10, 1e15), jnp.clip(sigma_bar, 1e-10, 1e15)


def fit(p_m: ArrayLike, 
        p_f: float, 
        mu_init: ArrayLike, 
        sigma_init: ArrayLike, 
        pi_init: ArrayLike, 
        filter: str, 
        prior: str, 
        delta: float, 
        n_iter: int, 
        threshold: float, 
        plotter: Plotter, 
        y_bool: List[ArrayLike], 
        observation_times: List[ArrayLike], 
        x_t: List[ArrayLike], 
        query_times: List[ArrayLike],
        exp_name: str) -> Tuple[Array, Array]:
    """
    Fit the mixture model with exponential prior using the EM algorithm.

    Args:
        p_m (ArrayLike): Probability of missed detection.
        p_f (float): Probability of false alarm.
        mus_init (ArrayLike): Initial rate parameters for each component.
        sigmas_init (ArrayLike): Initial mixing coefficients.
        pi_init (ArrayLike): Initial mixing coefficients.
        filter (str): The filter type to use.
        prior (str): The prior distribution type.
        delta (float): Convergence threshold for the EM algorithm.
        n_iter (int): Maximum number of iterations for the EM algorithm.
        threshold (float): Threshold for reporting metrics.
        plotter (Plotter): Plotter object for visualizing results.
        y_bool (List[ArrayLike]): List of binary observation arrays for each sequence.
        observation_times (List[ArrayLike]): List of arrays of observation times for each sequence.
        x_t (List[ArrayLike]): List of true state arrays for each sequence.
        query_times (List[ArrayLike]): List of arrays of query times for each sequence.

    Returns:
        Array: Final rate parameters (lambda_hat) and mixing coefficients (pi_hat).
    """

    # Params and initialization
    start, m = True, 1
    n_sequences, n_components = len(y_bool), len(pi_init)
    # Initialize params
    mu_hat, sigma_hat, pi_hat = mu_init, sigma_init, pi_init
    log_s = Partial(priors.log_survival_lognormal)
    params = {"logmu": mu_hat, "std": sigma_hat}
    # Compute metrics at initialization
    loss_t_1 = -jnp.inf
    init_belief = run_filter_for_sequences(filter, y_bool, observation_times, query_times, pi_hat, log_s, params, p_m, p_f)
    # Store the loss values
    losses = [loss_t_1]
    print_results({"mu": mu_hat, "std": sigma_hat, "pi": pi_hat}, m, loss_t_1, prior=prior)

    mu_best, sigma_best, pi_best, best_loss = mu_hat, sigma_hat, pi_hat, loss_t_1
    while start and m < n_iter:
        state = create_filter(filter, log_s, {"logmu": mu_hat, "std": sigma_hat}, pi_hat, initialization_time=0.0)
        pi_bar, mu_bar, sigma_bar = em_step(filter, n_sequences, n_components, observation_times, y_bool, state, pi_hat, mu_hat, sigma_hat, p_m, p_f)

        state = create_filter(filter, log_s, {"logmu": mu_bar, "std": sigma_bar}, pi_bar, initialization_time=0.0)
        loss_t = 0.0
        for c in range(n_sequences):
            log_evidence, _ = compute_marginals(filter, state, y_bool[c], observation_times[c], p_m, p_f)
            loss_t += log_evidence
        # Increase iteration and if the loss does not improve, try again without updating params
        m += 1
        # Update lambda if iteration was successful and print results
        pi_hat, mu_hat, sigma_hat = pi_bar, mu_bar, sigma_bar
        # Checkpoint the best parameters
        mu_best, sigma_best, pi_best, best_loss = (mu_hat, sigma_hat, pi_hat, loss_t) if loss_t > best_loss and not jnp.isnan(loss_t) else (mu_best, sigma_best, pi_best, best_loss)
        print_results({"mu": mu_hat, "std": sigma_hat, "pi": pi_hat}, m, loss_t, prior=prior)

        ### Stopping criteria ###
        if jnp.abs(loss_t - loss_t_1) < delta or jnp.isnan(loss_t):
            logger.info("=" * 50)
            logger.info(f"Stopping the algorithm at iteration {m}")
            logger.info("=" * 50)
            start = False

        # Assign current loss and iterate again
        losses.append(loss_t)
        loss_t_1 = loss_t

    mu_hat, sigma_hat, pi_hat = mu_best, sigma_best, pi_best
    print_results({"mu": mu_hat, "std": sigma_hat, "pi": pi_hat}, "Final", best_loss, prior=prior)
    # Compute the final belief of the filter over all chains
    final_belief = run_filter_for_sequences(filter, y_bool, observation_times, query_times, pi_hat, log_s, {"logmu": mu_hat, "std": sigma_hat}, p_m, p_f)
    # Compute the metrics wrt to mixture
    report_metrics(
        x_t,
        [bel[-1, :] for bel in init_belief],
        [bel[-1, :] for bel in final_belief],
        query_times,
        threshold,
        table_title=exp_name
    )
    plotter.plot_loss_curve(
        jnp.arange(len(losses)) + 1, -jnp.array(losses), title="Loss", f_name=f"{exp_name}_loss_curve"
    )
    return mu_hat, sigma_hat, pi_hat, best_loss

def select_and_refine_model(
    fit: Callable,
    query_times: jnp.ndarray,
    num_datapoints: float,
    params_range: range = range(2, 5),
    perturbation_scale: jnp.ndarray = jnp.array([0.2, 0.1, 0.1]),
    num_retrain_attempts: int = 5,
    key: random.PRNGKey = random.PRNGKey(42),
    prune_threshold: float = 1e-2,
) -> Dict[str, jnp.ndarray]:
    """
    Selects the optimal model based on AIC/BIC and refines parameters with perturbations.

    Args:
        fit (Callable): EM fitting function.
        query_times (jnp.ndarray): Times of sampling.
        num_datapoints (float): Number of data points.
        params_range (range): Range of number of components to evaluate.
        perturbation_scale (jnp.ndarray): Scale of perturbations for retraining.
        num_retrain_attempts (int): Number of retraining attempts.
        key (random.PRNGKey): Random key for perturbation.
        prune_threshold (float): Threshold for pruning components.

    Returns:
        Dict[str, jnp.ndarray]: Dictionary with best parameters and evidence.
    """
    best_mu, best_sigma, best_pi = None, None, None
    best_aic = jnp.inf
    best_bic = jnp.inf
    best_evidence = -jnp.inf

    # Model selection loop
    for n in params_range:
        quantile_positions = jnp.linspace(0, 1, n + 2)[1:-1]
        quantiles = jnp.quantile(query_times, quantile_positions)
        mus_init = jnp.log(quantiles)
        sigmas_init = jnp.ones_like(mus_init).astype(jnp.float32)
        pi_init = jnp.ones_like(mus_init).astype(jnp.float32) / n

        # Fit model
        mu_em, sigma_em, pi_em, final_evidence = fit(
            mu_init=mus_init, sigma_init=sigmas_init, pi_init=pi_init
        )

        # compute aic and bic
        aic = compute_aic(final_evidence, 3 * n)
        bic = compute_bic(num_datapoints, final_evidence, 3 * n)

        # Update best parameters based on AIC
        if aic < best_aic:
            best_mu, best_sigma, best_pi = mu_em, sigma_em, pi_em
            best_aic = aic
            best_bic = bic
            best_evidence = final_evidence

        logger.info("=" * 50)
        logger.info(f"components={n}, AIC={aic}, BIC={bic}, NLL={final_evidence}")
        logger.info("=" * 50)

    # If one of the resulting weight is below the threshold, prune the component from the mixture
    if jnp.any(best_pi < prune_threshold):
        mask = best_pi > prune_threshold
        best_pi, best_mu, best_sigma = best_pi[mask], best_mu[mask], best_sigma[mask]
        best_pi = best_pi / jnp.sum(best_pi)
        best_aic, best_bic, best_evidence = jnp.inf, jnp.inf, -jnp.inf
        logger.info("=" * 50)
        logger.info(f"Pruned {len(mask) - mask.sum()} components with weight below {prune_threshold}")
        logger.info("=" * 50)

    # Refinement loop
    optimal_n_components = len(best_mu)
    for _ in range(num_retrain_attempts):
        quantile_positions = jnp.linspace(0, 1, optimal_n_components + 2)[1:-1]
        quantiles = jnp.quantile(query_times, quantile_positions)

        # Perturb parameters
        mus_init, key = perturb(jnp.log(quantiles), perturbation_scale[0], key)
        sigmas_init, key = perturb(jnp.ones_like(mus_init).astype(jnp.float32), perturbation_scale[1], key)
        pi_init, key = perturb(jnp.ones_like(mus_init).astype(jnp.float32) / optimal_n_components, perturbation_scale[2], key)
        pi_init /= jnp.sum(pi_init)

        # Refit model
        mu_em, sigma_em, pi_em, final_evidence = fit(
            mu_init=mus_init, sigma_init=sigmas_init, pi_init=pi_init
        )
        # compute aic and bic
        aic = compute_aic(final_evidence, 3 * n)
        bic = compute_bic(num_datapoints, final_evidence, 3 * n)

        # Update best parameters if evidence improves
        if final_evidence > best_evidence:
            best_mu, best_sigma, best_pi = mu_em, sigma_em, pi_em
            best_aic = aic
            best_bic = bic
            best_evidence = final_evidence

    return {
        "mu": best_mu,
        "sigma": best_sigma,
        "pi": best_pi,
        "aic": best_aic,
        "bic": best_bic,
        "evidence": best_evidence,
    }
