from typing import List, Tuple, Union, Dict, Callable
import time
from functools import partial

from jax import Array
from jax.typing import ArrayLike

import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.special import logsumexp
from jax._src.tree_util import Partial
from src.utils.filter_state import FilterState

import src.utils.priors as priors
from src.learning.learning_utils import run_filter_for_sequences, compute_log_likelihood, create_filter, compute_marginals, print_results, perturb
from src.utils.metrics import compute_aic, compute_bic, report_metrics
from src.utils.math_utils import logdiff
from src.utils.viz import Plotter
from src.utils.logger import logger


@partial(jax.jit, static_argnames=['filter'])
def process_sequence(filter: str, obs_times: ArrayLike, obs: ArrayLike, pi_: ArrayLike, lambda_: ArrayLike, state: FilterState, p_m: ArrayLike, p_f: ArrayLike):
    """
    Process a single sequence of observations using the specified filter.

    Args:
        filter (str): The filter type to use.
        obs_times (ArrayLike): Array of observation times.
        obs (ArrayLike): Array of observations.
        pi_ (ArrayLike): Mixing coefficients.
        lambda_ (ArrayLike): Rate parameters for each component.
        state (FilterState): The initial state of the filter.
        p_m (ArrayLike): Probability of missed detection.
        p_f (ArrayLike): Probability of false alarm.

    Returns:
        Tuple[Array, Array]: The expected survival times (psi_j) and the expected component assignments (phi_j).
    """
    n_components = len(pi_)

    # Initialize filter (simulating Filter class with JAX functionality)
    log_sum_psi = jnp.zeros((obs_times.shape[0] + 1, n_components))

    ####### E Step #######
    # Compute log-likelihood for t_0 = 0
    ll_function = jax.vmap(compute_log_likelihood, in_axes=(None, 0, None, None, None, None))
    log_likelihoods = ll_function(filter, jnp.insert(obs_times, 0, 0.0), obs, obs_times, p_m, p_f)

    ### Compute E_q[Z_kj T_j] for all k and j ###
    # For time t_0 = 0 and t_1 = t_1
    log_term1 = jnp.log(1 / lambda_)
    log_terms = (
        -lambda_ * obs_times[:, jnp.newaxis]
        + jnp.log(obs_times[:, jnp.newaxis] + (1 / lambda_))
    )
    log_expected_p0 = logdiff(log_term1, log_terms[0])
    log_sum_psi = log_sum_psi.at[0].set(log_likelihoods[0] + log_expected_p0)

    # Compute log \sum_{i=1}^{N-1} p(Y_{1:N} | T) \int_{t_i}^{t_{i+1}} T p(T) dT
    log_expected_ps = logdiff(log_terms[:-1], log_terms[1:])
    log_sum_psi = log_sum_psi.at[1:-1].set(
        log_likelihoods[1:-1][:, jnp.newaxis] + log_expected_ps
    )

    # Compute log p(Y_{1:N} | t_N) \int_{t_N}^{t_{N+1}} T p(T) dT
    log_sum_psi = log_sum_psi.at[-1].set(log_likelihoods[-1] + log_terms[-1])

    # Compute expected survival time under p(T | Y_{1:N})
    total_log_sum_psi = logsumexp(log_sum_psi, axis=0)
    total_log_sum_psi += jnp.log(pi_)

    # Normalize and compute results
    log_evidence, log_joint_evidence = compute_marginals(filter, state, obs, obs_times, p_m, p_f)
    psi_j = jnp.exp(total_log_sum_psi - log_evidence)

    ### Compute E_q[Z_kj] for all k and j ###
    phi_j = jnp.exp(log_joint_evidence - log_evidence)
    return psi_j, phi_j

def em_step(
    filter: str,
    n_sequences: int,
    n_components: int,
    observation_times: List[ArrayLike],
    y_bool: List[ArrayLike],
    state: FilterState,
    pi_hat: ArrayLike,
    lambda_hat: ArrayLike,
    p_m: ArrayLike,
    p_f: ArrayLike,
) -> Union[Array, Array]:
    """
    Perform the E-step and M-step in the EM algorithm for the mixture model with exponential prior.

    Args:
        filter (str): The filter type to use.
        n_sequences (int): Number of observation sequences.
        n_components (int): Number of components in the mixture model.
        observation_times (List[ArrayLike]): List of arrays of observation times for each sequence.
        y_bool (List[ArrayLike]): List of binary observation arrays for each sequence.
        state (FilterState): The initial state of the filter.
        pi_hat (ArrayLike): Initial mixing coefficients.
        lambda_hat (ArrayLike): Initial rate parameters for each component.
        p_m (ArrayLike): Probability of missed detection.
        p_f (ArrayLike): Probability of false alarm.

    Returns:
        Tuple[Array, Array]: Updated mixing coefficients (pi_bar) and rate parameters (lambda_bar).
    """
    
    psi = jnp.zeros((n_sequences, n_components))
    phi = jnp.zeros((n_sequences, n_components))
    for j in range(n_sequences):
        psi_j, phi_j = process_sequence(filter, observation_times[j], y_bool[j], pi_hat, lambda_hat, state, p_m, p_f)
        psi = psi.at[j].set(psi_j)
        phi = phi.at[j].set(phi_j)

    ####### M Step #######
    pi_bar = jnp.sum(phi, axis=0) / jnp.sum(phi)
    lambda_bar = jnp.sum(phi, axis=0) / jnp.sum(psi, axis=0)

    # Clip the parameters to prevent NaN values
    return jnp.clip(pi_bar, 1e-10, 1.0), jnp.clip(lambda_bar, 1e-10, 1e15)

def fit(p_m: ArrayLike, 
        p_f: float, 
        lambda_init: ArrayLike, 
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
        lambda_init (ArrayLike): Initial rate parameters for each component.
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
        exp_name (str): Experiment name for logging and plotting.

    Returns:
        Tuple[Array, Array]: Final rate parameters (lambda_hat) and mixing coefficients (pi_hat).
    """

    # Params and initialization
    start = True
    m = 1
    n_sequences = len(y_bool)
    n_components = len(lambda_init)
    # Initialize params
    lambda_hat = lambda_init
    pi_hat = pi_init
    # Compute metrics at initialization
    log_s = Partial(priors.log_survival_exponential)
    params = {"lambda_": lambda_hat}
    loss_t_1 = -jnp.inf
    init_belief = run_filter_for_sequences(filter, y_bool, observation_times, query_times, pi_hat, log_s, params, p_m, p_f)
    # Store the loss values
    losses = [loss_t_1]
    print_results({"lambda_": lambda_hat, "pi": pi_hat}, m, loss_t_1, prior=prior)

    lambda_best, pi_best, best_loss = lambda_hat, pi_hat, loss_t_1
    while start and m < n_iter:
        state = create_filter(filter, log_s, {"lambda_": lambda_hat}, pi_hat, initialization_time=0.0)
        pi_bar, lambda_bar = em_step(filter, n_sequences, n_components, observation_times, y_bool, state, pi_hat, lambda_hat, p_m, p_f)

        state = create_filter(filter, log_s, {"lambda_": lambda_bar}, pi_bar, initialization_time=0.0)
        loss_t = 0.0
        for c in range(n_sequences):
            log_evidence, _ = compute_marginals(filter, state, y_bool[c], observation_times[c], p_m, p_f)
            loss_t += log_evidence
        # Increase iteration and if the loss does not improve, try again without updating params
        m += 1
        # Update lambda if iteration was successful and print results
        lambda_hat, pi_hat = lambda_bar, pi_bar
        lambda_best, pi_best, best_loss = (lambda_hat, pi_hat, loss_t) if loss_t > best_loss and not jnp.isnan(loss_t) else (lambda_best, pi_best, best_loss)
        print_results({"lambda_": lambda_hat, "pi": pi_hat}, m, loss_t, prior=prior)

        ### Stopping criteria ###
        if jnp.abs(loss_t - loss_t_1) < delta or jnp.isnan(loss_t):
            logger.info("=" * 50)
            logger.info(f"Stopping the algorithm at iteration {m}")
            logger.info("=" * 50)
            start = False

        # Assign current loss and iterate again
        losses.append(loss_t)
        loss_t_1 = loss_t

    lambda_hat, pi_hat = lambda_best, pi_best
    print_results({"lambda_": lambda_hat, "pi": pi_hat}, "Final", best_loss, prior=prior)
    # Compute the final belief of the filter over all chains
    final_belief = run_filter_for_sequences(filter, y_bool, observation_times, query_times, pi_hat, log_s, {"lambda_": lambda_hat}, p_m, p_f)
    # Compute the metrics wrt to mixture
    report_metrics(
        x_t,
        [bel[-1, :] for bel in init_belief],
        [bel[-1, :] for bel in final_belief],
        query_times,
        threshold,
        table_title=exp_name,
    )
    plotter.plot_loss_curve(
        jnp.arange(len(losses)) + 1, -jnp.array(losses), title="Loss", f_name=f"{exp_name}_loss_curve"
    )
    return lambda_hat, pi_hat, best_loss

def select_and_refine_model(
    fit: Callable,
    query_times: jnp.ndarray,
    num_datapoints: int,
    params_range: range = range(2, 5),
    perturbation_scale: jnp.ndarray = jnp.array([0.01, 0.1]),
    num_retrain_attempts: int = 5,
    key: random.PRNGKey = random.PRNGKey(42),
    prune_threshold: float = 1e-2,
) -> Dict[str, jnp.ndarray]:
    """
    Selects the optimal model parameters (lambda, pi) based on AIC/BIC and refines them with perturbation.

    Args:
        fit (Callable): Function to fit the model.
        query_times (jnp.ndarray): Query times used to initial model parameters.
        num_datapoints (int): Number of data points for BIC computation.
        params_range (range): Range of number of components to evaluate.
        perturbation_scale (jnp.ndarray): Scale of perturbations for retraining.
        num_retrain_attempts (int): Number of retraining attempts.
        key (random.PRNGKey): Random key.

    Returns:
        Dict[str, jnp.ndarray]: Dictionary with best parameters and evidence.
    """
    best_lambda, best_pi = None, None
    best_aic = jnp.inf
    best_bic = jnp.inf
    best_evidence = -jnp.inf

    # Model selection loop
    for n in params_range:
        quantile_positions = jnp.linspace(0, 1, n + 2)[1:-1]
        quantiles = jnp.quantile(query_times, quantile_positions)
        lambda_init = 1 / quantiles
        pi_init = jnp.ones_like(lambda_init) / n

        # Fit model
        lambda_em, pi_em, final_evidence = fit(
            lambda_init=lambda_init, pi_init=pi_init
        )

        # Compute AIC and BIC
        aic = compute_aic(final_evidence, 2 * n)
        bic = compute_bic(num_datapoints, final_evidence, 2 * n)

        # Update best parameters based on AIC
        if aic < best_aic:
            best_lambda, best_pi = lambda_em, pi_em
            best_aic = aic
            best_bic = bic
            best_evidence = final_evidence

        logger.info("=" * 50)
        logger.info(f"components={n}, AIC={aic}, BIC={bic}, NLL={final_evidence}")
        logger.info(f"λ={lambda_em}, π={pi_em}")
        logger.info("=" * 50)

    # Check if there exists a component with a low probability and prune it
    if jnp.any(best_pi < prune_threshold):
        mask = best_pi > prune_threshold
        best_pi, best_lambda = best_pi[mask], best_lambda[mask]
        best_pi = best_pi / jnp.sum(best_pi)
        best_aic, best_bic, best_evidence = jnp.inf, jnp.inf, -jnp.inf
        logger.info("=" * 50)
        logger.info(f"Pruned {len(mask) - mask.sum()} components with weight below {prune_threshold}")
        logger.info("=" * 50)

    # Refinement loop
    optimal_n_components = len(best_lambda)
    for _ in range(num_retrain_attempts):
        quantile_positions = jnp.linspace(0, 1, optimal_n_components + 2)[1:-1]
        quantiles = jnp.quantile(query_times, quantile_positions)

        # Perturb parameters
        lambda_init, key = perturb(1 / quantiles, perturbation_scale[0], key)
        pi_init, key = perturb(
            jnp.ones_like(lambda_init) / optimal_n_components, perturbation_scale[1], key
        )
        pi_init /= jnp.sum(pi_init)

        # Refit model
        lambda_em, pi_em, final_evidence = fit(
            lambda_init=lambda_init, pi_init=pi_init
        )
        # Compute AIC and BIC
        aic = compute_aic(final_evidence, 2 * n)
        bic = compute_bic(num_datapoints, final_evidence, 2 * n)

        # Update best parameters if evidence improves
        if final_evidence > best_evidence:
            best_lambda, best_pi = lambda_em, pi_em
            best_aic = aic
            best_bic = bic
            best_evidence = final_evidence

    return {
        "lambda": best_lambda,
        "pi": best_pi,
        "aic": best_aic,
        "bic": best_bic,
        "evidence": best_evidence,
    }