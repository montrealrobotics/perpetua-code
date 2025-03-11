import numpy as np
import argparse
import os
from functools import partial

import jax.numpy as jnp
from jax import random

from src.utils.filter_test_utils import (
    sample_observation_times,
    generate_observations,
    generate_presence_observations,
)
import src.learning.mixture_lognormal as EMMixtureLogNormal
from src.utils.viz import Plotter
from src.utils.logger import setup_logger


def main(args):
    ########################
    ### Begin generation ###
    ########################
    simulation_length = 5000  # Length of simulation in seconds
    sampling_period = 5  # Inter-sample period in seconds for true state trace
    x_sampling_times = jnp.arange(0, simulation_length, sampling_period) / 60  # True state sampling times
    # "Standard" simulation settings
    p_m = 0.1  # Missed detection probability
    p_f = 0.1  # False alarm probability
    lambda_r_standard = 1.0 / 20  # Standard revisitation rate
    lambda_o = 30.0  # Inter-observation rate
    p_n = 1.0 / 1.0  # p_N = probability of leaving after the last reobservation; expected # observations N = 1 / p_N
    filter = args.filter
    threshold = args.threshold
    prior = "lognorm"
    n_sequences = args.n_seq
    viz = args.viz
    n_components = 3
    # Get data generation process parameters
    pi = jnp.array([0.2, 0.5, 0.3])
    means = jnp.log(np.array([500, 2000, 4500]) / 60)
    sigmas = jnp.array([0.01, 0.03, 0.05])

    plotter = Plotter(viz=viz, exp_name=f"{filter}_{prior}_learning")
    logger = setup_logger(os.path.join(plotter.res_path, "output.log"))
    logger.info("=" * 50)
    logger.info(f"Running Experiment...")
    logger.info(f"The number of components in the mixture model is: {n_components}")
    logger.info(f"The number of chains is: {n_sequences}")
    logger.info(f"The mixing coefficients are: {pi}")
    logger.info(f"The means are: {means}")
    logger.info(f"The stdv are: {sigmas}")
    logger.info("=" * 50)
    # Containers for the observations and ground truth
    ###############################
    #### begin data generation ####
    ###############################
    num_datapoints = 0
    obs = []
    obs_times = []
    gt = []
    gt_times = []
    for _ in range(n_sequences):
        # Sample a component
        c = np.random.choice(n_components, 1, p=pi).item()
        mu, s = means[c], sigmas[c]
        # Sample time where the feature becomes present or absent from a lognormal distribution
        cut_time = np.random.lognormal(mean=mu, sigma=s)
        # Generate date depending if the filter is persistence or emergence
        if filter == "persistence":
            # Compute true state-trace
            x_t = x_sampling_times <= cut_time
            # Sample observation times
            observation_times = sample_observation_times(lambda_r_standard, lambda_o, p_n, simulation_length) / 60
            # Sample observations
            y_binary = generate_observations(cut_time, observation_times, p_m, p_f)
        elif filter == "emergence":
            # Compute true state-trace
            x_t = x_sampling_times >= cut_time
            # Sample observation times
            observation_times = sample_observation_times(lambda_r_standard, lambda_o, p_n, simulation_length) / 60
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
    #############################
    #### end data generation ####
    #############################

    # Plot observations
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
        EMMixtureLogNormal.fit,
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
    result = EMMixtureLogNormal.select_and_refine_model(
        fit,
        x_sampling_times,
        num_datapoints,
        range(2, 5),
        jnp.array([0.1, 0.1, 0.1]),  # This is a percentage, perturb the initial values by 10%
        num_retrain_attempts=3,
        prune_threshold=0.01,        # Prune mixtures below 1%
        key=random.PRNGKey(42),
    )
    best_mu, best_sigma, best_pi, final_evidence = (
        result["mu"],
        result["sigma"],
        result["pi"],
        result["evidence"],
    )

    logger.info("=" * 50)
    logger.info(f"The value of (μ, σ) using approximate EM is: ({best_mu}, {best_sigma})")
    logger.info(f"The ground truth value of (μ, σ) is: ({means}, {sigmas})")
    logger.info(f"The value of π using approximate EM is: {best_pi}")
    logger.info(f"The ground truth value of π is: {pi}")
    logger.info(f"The NLL is: {final_evidence}")
    logger.info("=" * 50)


if __name__ == "__main__":
    np.random.seed(1)  # Set the random seed for reproducibility
    # Argparse with different choose of prior
    parser = argparse.ArgumentParser(
        description="Run the mixture of filters learning example using a Log-Normal prior with either the persistence or emergence models."
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
        default=20,
        help="Number of sequences used to train the model",
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        default=False,
        help="Enable vizualizations",
    )
    args = parser.parse_args()
    main(args)
