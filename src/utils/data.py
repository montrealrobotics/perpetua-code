from typing import List, Dict, Tuple, Union

import os
import json
import pickle

from jax.typing import ArrayLike
from jax import Array
import jax.numpy as jnp
import numpy as np
from jax._src.tree_util import Partial

from src.utils.cpd_utils import detect_change_points
import src.utils.priors as priors
import src.learning.mixture_lognormal as EMLogNormal
import src.learning.mixture_exponential as EMExponential
from src.utils.metrics import report_metrics
from src.utils.filter_test_utils import run_perpetua
from src.utils.logger import logger


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, jnp.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def read_json(file_path: str):
    """Reads a json file and returns a dictionary with its content."""
    with open(file_path, "r") as file:
        return json.load(file)


def save_json(file_name: str, data: Dict[str, Union[List[float], np.ndarray]]):
    """Saves a list to a json file."""
    # Create directory if it does not exist
    if not os.path.exists(os.path.dirname(file_name)):
        os.makedirs(file_name, exist_ok=True)
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, cls=NumpyEncoder, indent=4)


def read_pickle(file_path: str):
    """Reads a pickle file and returns its content."""
    with open(file_path, "rb") as file:
        return pickle.load(file)


def save_pickle(file_path: str, data):
    """Writes a dictionary to a pickle file."""
    # Create the folder if it doesn't exist
    folder_path = os.path.dirname(file_path)
    if folder_path:
        os.makedirs(folder_path, exist_ok=True)

    with open(file_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def extract_sequences(
    change_points: List[float], observations: ArrayLike, obs_times: ArrayLike
) -> Tuple[List[Array], List[Array]]:
    persistence_seq_obs, emergence_seq_obs = [], []
    persistence_seq_t, emergence_seq_t = [], []
    start = 0
    n_change_points = len(change_points)
    for i in range(0, n_change_points - 1):
        middle = change_points[i]
        end = change_points[i + 1]
        # Extract first and second block
        first_block = observations[start:middle]
        # Validate if the cycle is death or birth
        mode_first_block = (jnp.median(first_block) >= 0.5).astype(int)
        # If the mode of the first block is one, then it is a persistence sequence
        if mode_first_block == 1:
            persistence_seq_obs.append(observations[start:end])
            # Observations start from zero
            # We substract the earliest time of the current sequence for simplicity, but it could be the
            # last time of the previous sequence
            persistence_seq_t.append(obs_times[start:end] - obs_times[start:end].min())
        # Otherwise it is a emergence sequence
        else:
            emergence_seq_obs.append(observations[start:end])
            # Observations start from zero
            # We substract the earliest time of the current sequence for simplicity, but it could be the
            # last time of the previous sequence
            emergence_seq_t.append(obs_times[start:end] - obs_times[start:end].min())
        start = middle

    # If sequences are empty, it means that there is only one sequence
    if not persistence_seq_obs or not emergence_seq_obs:
        mode_block = (jnp.median(observations[0:-1]) >= 0.5).astype(int)
        if mode_block == 1:
            persistence_seq_obs.append(observations)
            persistence_seq_t.append(obs_times - obs_times.min())
        else:
            emergence_seq_obs.append(observations)
            emergence_seq_t.append(obs_times - obs_times.min())

    return persistence_seq_obs, persistence_seq_t, emergence_seq_obs, emergence_seq_t


def compute_sequences(observations_dict: Dict[str, List[np.ndarray]], min_size=5, penalty=3) -> List[np.ndarray]:
    """
    Compute the sequences of observations from the dictionary of observations.
    """
    observations_seq = {
        i: {"persistence_obs": None, "persistence_t": None, "emergence_obs": None, "emergence_t": None}
        for i in observations_dict.keys()
    }
    for id in observations_dict.keys():
        # Compute change times for observations
        logger.info(f"Computing sequence for the {id} landmark...")
        cp_times_obs = detect_change_points(
            np.array(observations_dict[id]["obs"]),
            np.array(observations_dict[id]["obs_t"]),
            min_size=min_size,
            pentalty=penalty,
        )
        # Extract the death sequences
        death_obs, death_t, birth_obs, birth_t = extract_sequences(
            cp_times_obs, observations_dict[id]["obs"], observations_dict[id]["obs_t"]
        )
        # Store the sequences
        observations_seq[id]["persistence_obs"] = death_obs
        observations_seq[id]["persistence_t"] = death_t
        observations_seq[id]["emergence_obs"] = birth_obs
        observations_seq[id]["emergence_t"] = birth_t
    return observations_seq


def compute_query_times_datapoints(observation_times: List[ArrayLike], dt: float = 0.5) -> Tuple[Array, Array]:
    """
    Obtain the query times (largest sequyence) and the number of datapoints used during training.
    """
    longest_sequence = max([obs.max() for obs in observation_times])
    num_datapoints = sum([len(obs) for obs in observation_times])
    return jnp.arange(0, longest_sequence, dt), num_datapoints


# Helper functions
def initialize_container(prior, n_landmarks):
    if prior == "exponential":
        return {i: {"persistence": None, "emergence": None} for i in range(n_landmarks)}
    elif prior == "lognorm":
        return {i: {"persistence": None, "emergence": None} for i in range(n_landmarks)}
    else:
        raise NotImplementedError(f"Prior '{prior}' is not implemented.")


def get_prior_config(prior):
    """
    Returns Fit method, Select and Refine method, percentage of perturbation, and prior function.
    """
    if prior == "exponential":
        return (
            EMExponential.fit,
            EMExponential.select_and_refine_model,
            jnp.array([0.1, 0.1]),
            Partial(priors.log_survival_exponential),
        )
    elif prior == "lognorm":
        return (
            EMLogNormal.fit,
            EMLogNormal.select_and_refine_model,
            jnp.array([0.1, 0.1, 0.1]),
            Partial(priors.log_survival_lognormal),
        )
    else:
        raise NotImplementedError(f"Prior '{prior}' is not implemented.")


def store_params(prior, result):
    """
    Helper function to store parameters learned parameters.
    """
    logger.info(f"NLL: {result['evidence']}, BIC: {result['bic']}, AIC: {result['aic']}")
    if prior == "exponential":
        lambda_, pi = result["lambda"], result["pi"]
        logger.info(f"λ: {lambda_}, π: {pi}")
        return {"lambda_": lambda_, "pi": pi}
    else:
        mu, sigma, pi = result["mu"], result["sigma"], result["pi"]
        logger.info(f"μ: {mu}, σ: {sigma}, π: {pi}")
        return {"logmu": mu, "std": sigma, "pi": pi}


def process_predictions(
    obs_dict,
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
    interval,
    phase,
):
    # Run mixture perpetua filters
    probs, _, weights = run_perpetua(
        obs_dict[l]["obs"],
        obs_dict[l]["obs_t"],
        p_m,
        p_f,
        final_params[l]["persistence"]["pi"],
        final_params[l]["emergence"]["pi"],
        obs_dict[l]["gt_t"],
        log_s,
        log_s,
        params_persistence,
        params_emergence,
        delta_high=delta_high,
        delta_low=delta_low,
        num_steps=num_steps,
        eps=eps,
    )

    # Pick best component
    prediction = probs[np.arange(probs.shape[0]), jnp.argmax(weights, axis=1)]
    probs = jnp.concatenate([probs, prediction[:, None]], axis=1)

    # Plot results for this landmark
    plotter.plot_perpetua(
        obs_dict[l]["gt_t"],
        obs_dict[l]["gt_obs"],
        obs_dict[l]["obs_t"],
        obs_dict[l]["obs"],
        obs_dict[l]["gt_t"],
        probs.T,
        weights.T,
        interval=interval,
        location="center right",
        f_name=f"perpetua_l_{l}_learning_{prior}_{phase}",
    )

    return prediction


def compute_and_report_metrics(
    obs_dict, train_predictions, indices, threshold, exp_name
):
    gt_observations = [obs_dict[l]["gt_obs"] for l in indices]
    gt_times = [obs_dict[l]["gt_t"] for l in indices]
    # Compute Metrics
    metrics = report_metrics(
        gt_observations,
        train_predictions,
        train_predictions,
        gt_times,
        threshold=threshold,
        table_title=exp_name,
        initial=False,
    )
    return metrics


def prepare_data(data_path: str, train_file: str, test_file: str, n_landmarks: int):
    """
    Process room data and compute train sequences to train model
    """

    train_obs_file = os.path.join(data_path, f"{train_file}.pkl")
    test_obs_file = os.path.join(data_path, f"{test_file}.pkl")
    sequences_file = os.path.join(data_path, f"sequences.pkl")

    logger.info("Loading data...")
    train_obs_dict = read_pickle(train_obs_file)
    test_obs_dict = read_pickle(test_obs_file)
    if os.path.exists(sequences_file):
        sequences_dics = read_pickle(sequences_file)
    else:
        sequences_dics = compute_sequences(train_obs_dict)
        # Save data so it does not have to be processed again next time
        save_pickle(sequences_file, sequences_dics)

    return test_obs_dict, sequences_dics