import os
from functools import partial
import argparse

import jax
from jax import random
import jax.numpy as jnp
import numpy as np

import src.utils.data as data_utils
from src.utils.viz import Plotter
from src.utils.logger import setup_logger


def main(args):
    # Setup plotter
    plotter = Plotter(viz=args.viz, exp_name="room", fig_fmt="pdf")
    logger = setup_logger(os.path.join(plotter.res_path, "output.log"))

    # Access general configuration variables
    data_path = "data/room"
    train_file = "train_3h"
    test_file = "test_3h"
    seed = 12345
    threshold = 0.5
    n_landmarks = 8

    # Access filter configuration variables
    delta_high, delta_low = 0.95, 0.05
    p_m, p_f = 0.1, 0.1
    eps, num_steps = 0.1, 5

    # Access training configuration variables
    prior = args.prior
    n_iter = 250
    retrain_attempts = 5
    prune_threshold = 0.01
    min_components, max_components = 1, 4
    threshold = 0.5

    logger.info(f"Results will be saved in {plotter.res_path}")

    # Seed the experiment
    np.random.seed(seed)
    key = random.PRNGKey(seed)

    # Prepare data
    test_obs_dict, sequences_dics = data_utils.prepare_data(data_path, train_file, test_file, n_landmarks)

    # Method to compute individual sequences and debug
    plotter.plot_room_environment(
        test_obs_dict, f_name="room_environment_test", x_label="Time (s)", interval=(5400, 7200)
    )

    # Define templates for each prior type
    final_params = data_utils.initialize_container(prior, n_landmarks)
    fit_method, model_selector, perturbation, log_s = data_utils.get_prior_config(prior)

    test_pred_list = []
    for l in range(n_landmarks):
        for model in {"persistence", "emergence"}:
            # Compute query times and number of datapoints
            if len(sequences_dics[l][f"{model}_t"]) == 0:
                # If a sequence is empty (birth model), populate it with a prior that starts at one
                if prior == "exponential":
                    final_params[l][model] = {"lambda_": jnp.array([1.0]), "pi": jnp.array([1.0])}
                else:
                    final_params[l][model] = {
                        "logmu": jnp.array([0.0]),
                        "std": jnp.array([0.1]),
                        "pi": jnp.array([1.0]),
                    }
                logger.info("=" * 50)
                logger.info(f"No data for landmark {l} and model {model}")
                logger.info("=" * 50)
                continue
            logger.info("=" * 50)
            logger.info(f"Training landmark {l}, {model} model")
            logger.info("=" * 50)
            query_times, num_datapoints = data_utils.compute_query_times_datapoints(sequences_dics[l][f"{model}_t"])
            # Train a single feature
            fit = partial(
                fit_method,
                p_m=p_m,
                p_f=p_f,
                filter=model,
                prior=prior,
                delta=1e-6,
                n_iter=n_iter,
                threshold=threshold,
                plotter=plotter,
                y_bool=sequences_dics[l][f"{model}_obs"],
                observation_times=sequences_dics[l][f"{model}_t"],
                # We do not have ground truth at thus point so use the data as a proxy
                x_t=sequences_dics[l][f"{model}_obs"],
                query_times=sequences_dics[l][f"{model}_t"],
                exp_name=f"landmark_{l}_{model}_learning_{prior}",
            )
            # Sweep over different number of components
            key, subkey = random.split(key)
            result = model_selector(
                fit,
                query_times,
                num_datapoints,
                range(min_components, max_components),  # Sweep over 1 to 3 components
                perturbation,
                num_retrain_attempts=retrain_attempts,
                prune_threshold=prune_threshold,  # Prune mixtures below 1%
                key=subkey,
            )
            # Save parameters and log metrics
            final_params[l][model] = data_utils.store_params(prior, result)

        # Predict using the computed params
        params_persistence = final_params[l]["persistence"].copy()
        params_emergence = final_params[l]["emergence"].copy()
        params_persistence.pop("pi")
        params_emergence.pop("pi")

        # Process test predictions
        test_prediction = data_utils.process_predictions(
            test_obs_dict,
            final_params,
            l,
            p_m,
            p_f,
            log_s,
            params_persistence,
            params_emergence,
            delta_high,
            delta_low,
            num_steps,
            eps,
            prior,
            plotter,
            interval=(5400, 7200),
            phase="test",
        )
        test_pred_list.append(test_prediction)

        # Clear jax caches to prevent OOM
        jax.clear_caches()

    # Define landmark indices for all and semi-static landmarks
    all_landmarks = range(n_landmarks)

    #####################
    ## Report Metrics ###
    #####################
    # # Compute metrics for testing phase
    _ = data_utils.compute_and_report_metrics(test_obs_dict, test_pred_list, all_landmarks, threshold, "Perpetua Test")
    ### Logging ###
    # Save the final parameters and log results
    params_file = os.path.join(plotter.res_path, "params.json")
    data_utils.save_json(params_file, final_params)


if __name__ == "__main__":
    # Argparse with different choose of prior
    parser = argparse.ArgumentParser(description="Run full perpetua example in Room environment")
    parser.add_argument(
        "--prior",
        type=str,
        default="lognorm",
        help="Choose the prior to use for the persistence filter, options are: exponential and lognorm",
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        default=False,
        help="Enable vizualizations",
    )
    args = parser.parse_args()

    main(args)
