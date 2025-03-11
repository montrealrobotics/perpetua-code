# Load necessary libraries
import numpy as np
import argparse
import os
import jax.numpy as jnp
import time
from jax._src.tree_util import Partial

from src.utils import priors
from src.utils.filter_test_utils import run_perpetua
from src.utils.filter_test_utils import generate_observations
from src.utils.viz import Plotter
from src.utils.metrics import report_metrics
from src.utils.logger import setup_logger


def main(args):
    # Data generation settings
    pm = 0.01  # Missed detection probability
    pf = 0.01  # False alarm probability
    simulation_length = 12500  # Length of simulation in seconds
    sampling_period = 10.0  # 5  # Inter-sample period in seconds
    survival_times = np.array([400, 700])  # Survival times, in seconds
    emergence_times = np.array([800, 1500])
    pi_persistence = np.array([0.7, 0.3])
    pi_emergence = np.array([0.6, 0.4])
    n_components = pi_persistence.size  # Number of components in the mixture
    delta_low = 0.5
    delta_high = 0.5
    prior = args.prior
    threshold = args.threshold
    viz = args.viz
    num_steps = args.num_steps
    seed = args.seed
    eps = args.eps
    np.random.seed(seed)  # Set the random seed for reproducibility
    plotter = Plotter(viz=viz, exp_name="perpetua")
    logger = setup_logger(os.path.join(plotter.res_path, "output.log"))

    if prior == "exponential":
        # Exponential prior settings
        params_persistence = {"lambda_": 1 / survival_times}
        params_emergence = {"lambda_": 1 / (emergence_times - survival_times)}
        log_s_persistence = Partial(priors.log_survival_exponential)
        log_s_emergence = Partial(priors.log_survival_exponential)
    elif prior == "lognorm":
        # Lognormal prior settings
        params_persistence = {"logmu": np.log(survival_times), "std": np.array([0.05, 0.025])}
        params_emergence = {"logmu": np.log(emergence_times - survival_times), "std": np.array([0.025, 0.05])}
        log_s_persistence = Partial(priors.log_survival_lognormal)
        log_s_emergence = Partial(priors.log_survival_lognormal)
    elif prior == "weibull":
        # Weibull prior settings
        params_persistence = {"k": np.array([45, 50]), "lambda_": survival_times}
        params_emergence = {"k": np.array([45, 50]), "lambda_": (emergence_times - survival_times)}
        log_s_persistence = Partial(priors.log_survival_weibull)
        log_s_emergence = Partial(priors.log_survival_weibull)
    else:
        raise ValueError(f"Invalid prior option: {prior}")

    # The set of times to query the belief of the persistence filter and emergence prior
    query_times = np.arange(0, simulation_length, sampling_period)

    # Generate samples here
    x_sampling_times = [np.arange(0, simulation_length, sampling_period) for _ in range(n_components)]
    cut_times = [[(1, 2400), (8000, 10400)], [(1, 4500), (7500, 10500)]]
    gt_states, gt_times, obs, obs_times = [], x_sampling_times, [], []
    for c in range(n_components):
        n_repetitions = np.ceil(
            np.maximum(simulation_length / survival_times[c], simulation_length / emergence_times[c])
        ).astype(int)
        # Generate the logical conditions dynamically
        conditions = [
            np.logical_and(
                x_sampling_times[c] > i * emergence_times[c],
                x_sampling_times[c] <= i * emergence_times[c] + survival_times[c],
            )
            for i in range(n_repetitions)
        ]
        conditions.append(x_sampling_times[c] <= survival_times[c])
        mask = np.logical_or.reduce(conditions)
        x_t = np.where(mask, 1, 0)
        gt_states.append(x_t)

        # Initialize observation lists for the current component
        obs_list, obs_times_list = [], []

        for t_init, t_final in cut_times[c]:
            n_cycles = int((t_final - t_init) // emergence_times[c])
            # Determine if the cycle starts in death or birth
            start_persistence = t_init % emergence_times[c] < survival_times[c]
            start_emergence = t_init % emergence_times[c] > survival_times[c]
            n_cycles = n_cycles + 1 if start_emergence else n_cycles

            # Process each cycle within the interval
            for i in range(n_cycles + 1):  # Includes the last interval
                interval_start = (
                    emergence_times[c] * np.floor(t_init / emergence_times[c])
                    if start_persistence
                    else t_init + i * emergence_times[c]
                )
                interval_end = (
                    min(emergence_times[c] * np.ceil(t_init / emergence_times[c]), t_final)
                    if start_emergence
                    else min(interval_start + emergence_times[c], t_final)
                )

                # Generate observation times within this interval
                t_obs = np.arange(t_init + i * emergence_times[c], interval_end, sampling_period)

                # Generate observations for the current interval
                y_obs = generate_observations(
                    0.0 if start_emergence else survival_times[c] + interval_start, t_obs, pm, pf
                )
                y_bool = y_obs > 0

                # Append observations and times
                obs_list.append(y_bool)
                obs_times_list.append(t_obs)

                # Set start death to False
                t_init = emergence_times[c] * np.floor(t_init / emergence_times[c]) if start_persistence else t_init
                t_init = (
                    min(emergence_times[c] * np.floor(t_init / emergence_times[c]), t_final)
                    if start_emergence
                    else t_init
                )
                start_persistence, start_emergence = False, False

        # Concatenate all intervals for this component
        obs.append(np.concatenate(obs_list))
        obs_times.append(np.concatenate(obs_times_list))

    obs1, obs2 = obs[0], obs[1]
    obs_times1, obs_times2 = obs_times[0], obs_times[1]
    gt_times1, gt_times2 = gt_times[0], gt_times[1]
    gt_states1, gt_states2 = gt_states[0], gt_states[1]
    temp1 = obs1[obs_times1 > 7500]
    temp2 = obs_times1[obs_times1 > 7500]
    obs1 = np.concatenate([obs1[obs_times1 <= 7500], obs2[obs_times2 > 7500]])
    obs_times1 = np.concatenate([obs_times1[obs_times1 <= 7500], obs_times2[obs_times2 > 7500]])
    obs2 = np.concatenate([obs2[obs_times2 <= 7500], temp1])
    obs_times2 = np.concatenate([obs_times2[obs_times2 <= 7500], temp2])
    temp1 = gt_states1[gt_times1 > 7500]
    temp2 = gt_times1[gt_times1 > 7500]
    gt_states1 = np.concatenate([gt_states1[gt_times1 <= 7500], gt_states2[gt_times2 > 7500]])
    gt_times1 = np.concatenate([gt_times1[gt_times1 <= 7500], gt_times2[gt_times2 > 7500]])
    gt_states2 = np.concatenate([gt_states2[gt_times2 <= 7500], temp1])
    gt_times2 = np.concatenate([gt_times2[gt_times2 <= 7500], temp2])
    obs = [obs1, obs2]
    obs_times = [obs_times1, obs_times2]
    gt_states = [gt_states1, gt_states2]
    gt_times = [gt_times1, gt_times2]

    # Containers for the results
    for c in range(n_components):
        ### Run the survival filter ###
        start_time = time.time()
        probs, states, weights = run_perpetua(
            jnp.array(obs[c]),
            jnp.array(obs_times[c].astype(np.float32)),
            pm,
            pf,
            jnp.array(pi_persistence),
            jnp.array(pi_emergence),
            jnp.array(query_times),
            log_s_persistence,
            log_s_emergence,
            params_persistence,
            params_emergence,
            delta_high,
            delta_low,
            num_steps=num_steps,
            eps=eps,
        )
        probs.block_until_ready()
        states.block_until_ready()
        weights.block_until_ready()
        logger.info(f"Elapsed time: {time.time() - start_time:.4f} seconds")
        # Pick best component
        prediction = probs[np.arange(probs.shape[0]), jnp.argmax(weights, axis=1)]
        probs = jnp.concatenate([probs, prediction[:, None]], axis=1)
        # Plot results
        plotter.plot_perpetua(
            gt_times[c],
            gt_states[c],
            obs_times[c],
            obs[c],
            query_times,
            probs.T,
            weights.T,
            location="center right",
            f_name=f"perpetua_sequence_#{c}_{prior}",
            interval=(0, gt_times[c][-1]),
        )

        # Report metrics
        table = report_metrics(
            [gt_states[c]],
            None,
            [prediction],
            [query_times],
            threshold=threshold,
            initial=False,
            table_title=f"Perpetua Results Sequence #{c}",
        )


if __name__ == "__main__":
    # Argparse with different choose of prior
    parser = argparse.ArgumentParser(description="Run Perpetua with different prior options")
    parser.add_argument(
        "--prior",
        type=str,
        default="exponential",
        help="Choose the prior to use with Perpetua, options are: exponential, lognorm, weibull",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold used to compute precision and recall",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=10,
        help="Number of steps used during simulation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.1,
        help="Proportion of particles sampled fro the prior",
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        default=False,
        help="Enable vizualizations",
    )
    args = parser.parse_args()

    main(args)
