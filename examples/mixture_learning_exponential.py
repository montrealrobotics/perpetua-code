import numpy as np
import argparse
from functools import partial
import os

import jax.numpy as jnp
from jax import random

from src.utils.filter_test_utils import sample_observation_times, generate_observations, generate_presence_observations
import src.learning.mixture_exponential as EMMixtureExponential
from src.utils.viz import Plotter
from src.utils.logger import setup_logger


def main(args):
    ########################
    ### Begin generation ###
    ########################
    simulation_length = 5000  # Length of simulation in seconds
    sampling_period = 5  # Inter-sample period in seconds for true state trace
    x_sampling_times = np.arange(0, simulation_length, sampling_period)  # True state sampling times
    # "Standard" simulation settings
    p_m = 0.1  # Missed detection probability
    p_f = 0.1  # False alarm probability
    lambda_r_standard = 1.0 / 20  # Standard revisitation rate
    lambda_o = 30.0  # Inter-observation rate
    p_n = 1.0 / 1.0  # p_N = probability of leaving after the last reobservation; expected # observations N = 1 / p_N

    filter = args.filter
    threshold = args.threshold
    prior = "exponential"
    n_sequences = args.n_seq
    viz = args.viz
    n_components = 2
    # Get data generation process parameters
    pi = np.array([0.7, 0.3])
    lambdas = np.array([500, 4500])
    plotter = Plotter(viz=viz, exp_name=f"{filter}_{prior}_learning")
    logger = setup_logger(os.path.join(plotter.res_path, "output.log"))
    #############
    ## Logging ##
    #############
    logger.info("=" * 50)
    logger.info(f"Running Experiment...")
    logger.info(f"The number of components in the mixture model is: {n_components}")
    logger.info(f"The number of sequences is: {n_sequences}")
    logger.info(f"The mixing coefficients are: {pi}")
    logger.info(f"The lambdas are: {lambdas}")
    logger.info("=" * 50)
    # Containers for the observations and ground truth
    obs = []
    obs_times = []
    gt = []
    gt_times = []
    num_datapoints = 0
    for _ in range(n_sequences):
        # Sample a component
        c = np.random.choice(n_components, 1, p=pi).item()
        lambda_ = lambdas[c]
        # Sample time where the feature becomes persistence or alive from an uniform distribution
        cut_time = np.random.uniform(lambda_ - 10, lambda_ + 10, 1).item()
        # Generate date depending if the filter is persistence or emergence
        if filter == "persistence":
            # Compute true state-trace
            x_t = x_sampling_times <= cut_time
            # Sample observation times
            observation_times = sample_observation_times(lambda_r_standard, lambda_o, p_n, simulation_length)
            # Sample observations
            y_binary = generate_observations(cut_time, observation_times, p_m, p_f)
        elif filter == "emergence":
            # Compute true state-trace
            x_t = x_sampling_times >= cut_time
            # Sample observation times
            observation_times = sample_observation_times(lambda_r_standard, lambda_o, p_n, simulation_length)
            # Sample observations
            y_binary = generate_presence_observations(cut_time, observation_times, p_m, p_f)
        else:
            logger.error(f"Filter {filter} not recognized")
            raise ValueError(f"Filter {filter} not recognized")

        y_bool = y_binary > 0
        # Append to chain
        obs.append(jnp.array(y_bool))
        obs_times.append(jnp.array(observation_times).astype(np.float32))
        gt.append(jnp.array(x_t))
        gt_times.append(jnp.array(x_sampling_times))
        num_datapoints += len(observation_times)
    ########################
    #### end generation ####
    ########################

    # Plot a random observation sequence of observations
    c_test = np.random.randint(0, n_sequences)
    plotter.plot_observations(
        gt_times[c_test],
        gt[c_test],
        obs_times[c_test],
        obs[c_test].astype(int),
        location="center right",
        f_name=f"observation_sequence_{c_test}",
    )
    ### EM algorithm ###
    logger.info("=" * 50)
    logger.info(f"EM Algorithm for mixture filter {filter} with {prior} prior and {n_sequences} training sequences")
    logger.info("=" * 50)

    # Create a partial of the fit function (makes code cleaner)
    fit = partial(
        EMMixtureExponential.fit,
        p_m=jnp.array(p_m),
        p_f=jnp.array(p_f),
        filter=filter,
        prior=prior,
        delta=1e-6,
        n_iter=200,
        threshold=threshold,
        plotter=plotter,
        y_bool=obs,
        observation_times=obs_times,
        x_t=gt,
        query_times=gt_times,
        exp_name=f"{filter}_learning_{prior}_{n_sequences}_sequences",
    )
    # Sweep over different number of components
    result = EMMixtureExponential.select_and_refine_model(
        fit,
        x_sampling_times,
        num_datapoints,
        range(2, 4),              # Sweep over 2 to 3 components
        jnp.array([0.1, 0.1]),    # This is a percentage, perturb the initial values by 10%
        num_retrain_attempts=3,   # Retrain 3 times with the best number of components previously found
        prune_threshold=0.01,     # Prune mixtures below 1%
        key=random.PRNGKey(42),
    )
    best_lambda, best_pi, final_evidence = (result["lambda"], result["pi"], result["evidence"])

    logger.info("=" * 50)
    logger.info(f"The value of λ using closed-form EM is: {best_lambda} and π is {best_pi}")
    logger.info(f"The ground truth value of λ is: {1 / lambdas}")
    logger.info(f"The ground truth value of π is: {pi}")
    logger.info(f"The NLL is: {final_evidence}")
    logger.info("=" * 50)


if __name__ == "__main__":
    np.random.seed(3)  # Set the random seed for reproducibility
    # Argparse with different choose of prior
    parser = argparse.ArgumentParser(
        description="Run the mixture of filters learning example using an Exponential prior with either the persistence or emergence models."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold used to compute precision and recall",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default="persistence",
        help="The type of filter to use, options are: persistence, emergence",
    )
    parser.add_argument(
        "--n_seq",
        type=int,
        default=30,
        help="The number of sampled sequences",
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        default=False,
        help="Enable vizualizations",
    )
    args = parser.parse_args()
    main(args)
