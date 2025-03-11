import numpy as np
import argparse
import os
import time
import jax.numpy as jnp

from src.utils import priors
from src.utils.filter_test_utils import run_mixture_emergence_filters
from jax._src.tree_util import Partial
from src.utils.filter_test_utils import generate_presence_observations
from src.utils.viz import Plotter
from src.utils.metrics import report_metrics
from src.utils.logger import setup_logger


def main(args):
    # Configure persistence filter measurement model options
    # Data generation settings
    P_M = 0.05  # Missed detection probability
    P_F = 0.05  # False alarm probability
    n_components = 3  # Number of components in the mixture
    simulation_length = 1000  # Length of simulation in seconds
    sampling_period = 1  # Inter-sample period in seconds
    emergence_times = np.array([200, 500, 750])  # Survival times, in seconds
    pi = np.array([0.5, 0.35, 0.15])  # Mixing coefficients
    prior = args.prior
    threshold = args.threshold
    viz = args.viz
    plotter = Plotter(viz=viz, exp_name="mixture_emergence_filters")
    logger = setup_logger(os.path.join(plotter.res_path, "output.log"))

    if prior == "exponential":
        # Exponential prior settings
        params = {"lambda_": 1 / emergence_times}
        log_s = Partial(priors.log_survival_exponential)
    elif prior == "lognorm":
        # Lognormal prior settings
        params = {"logmu": np.log(emergence_times), "std": np.array([0.05, 0.05, 0.025])}
        log_s = Partial(priors.log_survival_lognormal)
    elif prior == "weibull":
        # Weibull prior settings
        params = {"k": np.array([5, 20, 50]), "lambda_": emergence_times}
        log_s = Partial(priors.log_survival_weibull)
    else:
        raise ValueError(f"Invalid prior option: {prior}")

    # The set of times to query the belief of the persistence filter and emergence prior
    query_times = np.arange(0, simulation_length, sampling_period)

    # Generate samples here
    x_sampling_times = [np.arange(0, simulation_length, sampling_period) for _ in range(n_components)]
    first_times = [(75, 125), (200, 300), (300, 450)]
    second_times = [(175, 275), (400, 600), (660, 850)]
    gt_states, gt_times, obs, obs_times = [], x_sampling_times, [], []
    for c in range(n_components):
        mask = x_sampling_times[c] >= emergence_times[c]
        x_t = np.where(mask, 1, 0)
        gt_states.append(x_t)

        # First set of observations
        first_obs_times = np.arange(*first_times[c], 1)  # Sampling times for the first set of observations
        first_obs = generate_presence_observations(
            emergence_times[c], first_obs_times, P_M, P_F
        )  # Sample Bernoulli observations according to the measurement model
        first_obs_bool = (
            first_obs > 0
        )  # Convert Bernoulli observations to Boolean values for use with Persistence Filter

        # Second set of observations
        second_obs_times = np.arange(*second_times[c], 1)
        second_obs = generate_presence_observations(
            emergence_times[c], second_obs_times, P_M, P_F
        )  # Sample Bernoulli observations according to the measurement model
        second_obs_bool = second_obs > 0
        # Concatenate these arrays and plot restuls
        t = np.concatenate((first_obs_times, second_obs_times))  # Array of observation times
        y_bool = np.concatenate((first_obs_bool, second_obs_bool))  # Array of (Boolean) observations
        # Concatete observations
        obs.append(y_bool)
        obs_times.append(t)

    max_mixture_bel = []
    for c in range(n_components):
        ### Run the survival filter ###
        start_time = time.time()
        emergence_belief = (
            run_mixture_emergence_filters(
                jnp.array(obs[c]),
                jnp.array(obs_times[c].astype(np.float32)),
                jnp.array(P_M),
                jnp.array(P_F),
                jnp.array(pi),
                jnp.array(query_times),
                log_s,
                params,
            )
            .block_until_ready()
            .squeeze()
        )
        logger.info(f"Elapsed time: {time.time() - start_time:.4f} seconds")
        # Compute the true persistence of the feature based on the survival and emergence prior
        plotter.plot_mixture_filter(
            gt_times[c],
            gt_states[c],
            obs_times[c],
            obs[c],
            query_times,
            emergence_belief,
            location="center right",
            f_name=f"component_{c}_emergence_filter",
        )
        max_mixture_bel.append(emergence_belief[-1, :])

    # Report metrics
    table = report_metrics(
        gt_states,
        None,
        max_mixture_bel,
        [query_times] * n_components,
        threshold=threshold,
        initial=False,
        table_title="Results Mixture of Emergence Filters",
    )


if __name__ == "__main__":
    np.random.seed(0)  # Set the random seed for reproducibility
    # Argparse with different choose of prior
    parser = argparse.ArgumentParser(description="Run the Mixture of emergence filters with different prior options")
    parser.add_argument(
        "--prior",
        type=str,
        default="exponential",
        help="Choose the prior to use for the emergence filter, options are: exponential, lognorm, weibull",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold used to compute precision and recall",
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        default=False,
        help="Enable vizualizations",
    )
    args = parser.parse_args()

    main(args)
