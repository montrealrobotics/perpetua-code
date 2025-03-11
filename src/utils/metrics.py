from typing import Tuple, List, Dict
from jax import Array
from jax.typing import ArrayLike

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from prettytable import PrettyTable
from src.utils.logger import logger


@jax.jit
def compute_l1_error(ground_truth: ArrayLike, belief: ArrayLike, query_times: ArrayLike) -> Array:
    """
    Computes the empirical L1 error between the estimator's belief state and the ground truth state.

    Args:
        ground_truth (ArrayLike): The ground truth state of the feature.
        belief (ArrayLike): The belief state of the estimator.
        query_times (ArrayLike): The times at which the belief state is queried.

    Returns:
        Array: The L1 error between the ground truth and the belief state.
    """
    return jnp.trapezoid(jnp.fabs(ground_truth - belief), query_times)

@jax.jit
def compute_mean_absolute_error(ground_truth: ArrayLike, belief: ArrayLike, query_times: ArrayLike) -> Array:
    """
    Computes the mean absolute error (MAE) between the ground truth state and the belief state of the estimator.

    Args:
        ground_truth (ArrayLike): The ground truth state of the feature.
        belief (ArrayLike): The belief state of the estimator.
        query_times (ArrayLike): The times at which the belief state is queried.

    Returns:
        Array: The mean absolute error between the ground truth and the belief state.
    """
    return compute_l1_error(ground_truth, belief, query_times) / (query_times[-1] - query_times[0])

@jax.jit
def compute_feature_absence_precision_and_recall(ground_truth_states: ArrayLike, predicted_states: ArrayLike) -> Tuple[Array, Array]:
    """
    Computes the precision and recall for feature absence classification.

    Args:
        ground_truth_states (ArrayLike): The ground truth state of the feature (1 for present, 0 for absent).
        predicted_states (ArrayLike): The predicted state of the feature (1 for present, 0 for absent).

    Returns:
        Tuple[Array, Array]: The precision and recall of the feature absence classification.
    """
    predicted_absences = jnp.logical_not(predicted_states)
    true_absences = jnp.logical_not(ground_truth_states)

    # PRECISION:
    # Compute total number of removal decisions
    num_predicted_absences = jnp.count_nonzero(predicted_absences)

    # Compute the number of removal predictions that are actually correct
    num_correct_predicted_absences = jnp.count_nonzero(true_absences[predicted_absences])

    precision = num_correct_predicted_absences.astype(jnp.float32) / num_predicted_absences

    # RECALL:
    # Compute the total number of true absences
    num_true_absences = jnp.count_nonzero(true_absences)

    # Compute the number of true absences that were correctly detected
    num_correctly_identified_true_absences = jnp.count_nonzero(predicted_absences[true_absences])

    recall = num_correctly_identified_true_absences.astype(jnp.float32) / num_true_absences

    return precision, recall

@jax.jit
def compute_balanced_accuracy(ground_truth_states: ArrayLike, predicted_states: ArrayLike) -> Array:
    """
    Compute balanced accuracy.

    Args:
        ground_truth_states (jnp.array): Ground truth binary labels (0 or 1).
        predicted_states (jnp.array): Predicted binary labels (0 or 1).

    Returns:
        jnp.float32: Balanced accuracy score.
    """
    # Compute the confusion matrix
    true_positives = jnp.count_nonzero(jnp.logical_and(ground_truth_states, predicted_states))
    false_positives = jnp.count_nonzero(jnp.logical_and(jnp.logical_not(ground_truth_states), predicted_states))
    true_negatives = jnp.count_nonzero(
        jnp.logical_and(jnp.logical_not(ground_truth_states), jnp.logical_not(predicted_states))
    )
    false_negatives = jnp.count_nonzero(jnp.logical_and(ground_truth_states, jnp.logical_not(predicted_states)))

    sensitivity = true_positives / (true_positives + false_negatives + 1e-8)  # Avoid division by zero
    specificity = true_negatives / (true_negatives + false_positives + 1e-8)

    return (sensitivity + specificity) / 2

@jax.jit
def compute_f1_score(y_true: ArrayLike, y_pred: ArrayLike) -> Array:
    """
    Compute the F1 score.

    Args:
        y_true (jnp.array): Ground truth binary labels (0 or 1).
        y_pred (jnp.array): Predicted binary labels (0 or 1).

    Returns:
        jnp.float32: F1 score.
    """
    
    true_positives = jnp.sum(jnp.logical_and(y_true, y_pred))
    false_positives = jnp.sum(jnp.logical_and(jnp.logical_not(y_true), y_pred))
    false_negatives = jnp.sum(jnp.logical_and(y_true, jnp.logical_not(y_pred)))

    precision = true_positives / (true_positives + false_positives + 1e-8)  # Avoid division by zero
    recall = true_positives / (true_positives + false_negatives + 1e-8)  # Avoid division by zero

    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)  # Avoid division by zero
    return f1


@jax.jit
def compute_feature_precision_recall_and_mcc(ground_truth_states: ArrayLike, predicted_states: ArrayLike) -> Tuple[Array, Array, Array]:
    """
    Computes the precision, recall, and Matthews correlation coefficient (MCC) for feature classification.

    Args:
        ground_truth_states (ArrayLike): The ground truth state of the feature (1 for present, 0 for absent).
        predicted_states (ArrayLike): The predicted state of the feature (1 for present, 0 for absent).

    Returns:
        Tuple[Array, Array, Array]: The precision, recall, and MCC of the feature classification.
    """
    # Compute the confusion matrix
    true_positives = jnp.count_nonzero(jnp.logical_and(ground_truth_states, predicted_states))
    false_positives = jnp.count_nonzero(jnp.logical_and(jnp.logical_not(ground_truth_states), predicted_states))
    true_negatives = jnp.count_nonzero(
        jnp.logical_and(jnp.logical_not(ground_truth_states), jnp.logical_not(predicted_states))
    )
    false_negatives = jnp.count_nonzero(jnp.logical_and(ground_truth_states, jnp.logical_not(predicted_states)))
    # Compute Precision
    precision = true_positives / (true_positives + false_positives + 1e-30)
    # Compure Recall
    recall = true_positives / (true_positives + false_negatives + 1e-30)
    # Compute MCC
    numerator = (true_positives * true_negatives) - (false_positives * false_negatives)
    # Log-scale to handle large products
    denom_terms = [
        true_positives + false_positives,
        true_positives + false_negatives,
        true_negatives + false_positives,
        true_negatives + false_negatives,
    ]
    denom_log = jnp.log(jnp.array(denom_terms))  # Logarithmic form
    denominator_log = 0.5 * jnp.sum(denom_log)
    denominator = jnp.exp(denominator_log)
    denominator = jnp.maximum(denominator, 1e-15)
    mcc = numerator / denominator
    return precision, recall, mcc

@jax.jit
def compute_feature_accuracy(ground_truth_states: ArrayLike, predicted_states: ArrayLike) -> Array:
    """
    Computes the accuracy of the feature state predictions.

    Args:
        ground_truth_states (ArrayLike): The ground truth state of the feature (1 for present, 0 for absent).
        predicted_states (ArrayLike): The predicted state of the feature (1 for present, 0 for absent).

    Returns:
        Array: The accuracy of the feature state predictions.
    """
    num_correct_predictions = jnp.count_nonzero(ground_truth_states == predicted_states)
    return (num_correct_predictions).astype(jnp.float32) / len(ground_truth_states)

@jax.jit
def compute_brier_score(ground_truth: ArrayLike, predicted_probs: ArrayLike) -> Array:
    """
    Computes the Brier score for the predicted probabilities.

    Args:
        ground_truth (ArrayLike): The ground truth state of the feature (1 for present, 0 for absent).
        predicted_probs (ArrayLike): The predicted probabilities of the feature state.

    Returns:
        Array: The Brier score of the predicted probabilities.
    """
    return jnp.mean(jnp.square(ground_truth - predicted_probs))


@partial(jax.jit, static_argnames=['n_bins'])
def expected_calibration_error(true_labels, pred_probs, n_bins=10):
    """
    Compute the Expected Calibration Error (ECE).

    Args:
        true_labels (jax.numpy.ndarray): Array of true labels (binary, shape: [n_samples]).
        pred_probs (jax.numpy.ndarray): Array of predicted probabilities (shape: [n_samples]).
        n_bins (int): Number of bins to divide probabilities into.

    Returns:
        float: Expected Calibration Error (ECE).
    """
    # Define bin edges
    bin_edges = jnp.linspace(0.0, 1.0, n_bins + 1)

    def compute_bin_error(bin_lower, bin_upper):
        # Identify samples in this bin
        in_bin = (pred_probs >= bin_lower) & (pred_probs < bin_upper)
        # in_bin_indices = jnp.where((pred_probs >= bin_lower) & (pred_probs < bin_upper))
        prop_in_bin = jnp.mean(in_bin)
        
        # Avoid empty bins
        avg_pred_prob = jnp.sum(pred_probs * in_bin) / (jnp.sum(in_bin) + 1e-10)  # Avoid division by zero
        avg_true_prob = jnp.sum(true_labels * in_bin) / (jnp.sum(in_bin) + 1e-10)
        
        # Weighted bin error
        return prop_in_bin * jnp.abs(avg_true_prob - avg_pred_prob)

    # Vectorize over bins
    bin_lowers = bin_edges[:-1]
    bin_uppers = bin_edges[1:]
    bin_errors = jax.vmap(compute_bin_error)(bin_lowers, bin_uppers)

    # Sum all bin errors to get ECE
    return jnp.sum(bin_errors)

@jax.jit
def compute_metrics(x_t: jnp.ndarray, belief: jnp.ndarray, query_times: jnp.ndarray, threshold: float) -> Dict[str, float]:
    """
    Compute the error metrics for the estimated belief.

    Args:
        x_t (jnp.ndarray): True state values.
        belief (jnp.ndarray): Estimated belief values.
        query_times (jnp.ndarray): Query times.
        threshold (float): Threshold used to compute precision and recall.

    Returns:
        Dict[str, float]: A dictionary containing all the metrics.
    """
    l1_error = compute_l1_error(x_t, belief, query_times)
    mae = compute_mean_absolute_error(x_t, belief, query_times)
    accuracy = compute_feature_accuracy(x_t, belief >= threshold)
    precision, recall, mcc = compute_feature_precision_recall_and_mcc(x_t, belief >= threshold)
    brier = compute_brier_score(x_t, belief)
    ece = expected_calibration_error(x_t, belief, n_bins=10)
    balance_accuracy = compute_balanced_accuracy(x_t, belief >= threshold)
    f1_score = compute_f1_score(x_t, belief >= threshold)
    metrics = {
        "l1_error": l1_error,
        "mae": mae,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "mcc": mcc,
        "brier": brier,
        "ece": ece,
        "balance_accuracy": balance_accuracy,
        "f1_score": f1_score
        }

    return metrics

def compute_time_weighted_average(metrics: List[Dict[str, float]], query_times: List[jnp.ndarray]) -> Dict[str, float]:
    """
    Compute the time-weighted average of the given metrics.

    Args:
        metrics (List[Dict[str, float]]): List of dictionaries containing the metrics for each chain.
        query_times (List[np.ndarray]): List of query times for each chain.

    Returns:
        Dict[str, float]: A dictionary containing the time-weighted average of the metrics.
    """
    # Compute the total time of all queries
    total_time = jnp.sum(jnp.array(np.array([q.max() for q in query_times])))
    weighted_metrics = {k: 0 for k in metrics[0].keys()}
    # Time-weighted sum for each metric
    for i in range(len(metrics)):
        time_interval = query_times[i].max()
        weighted_metrics = {k: weighted_metrics[k] + metrics[i][k] * time_interval / total_time for k in metrics[0].keys()}
    return weighted_metrics

def report_metrics(
    x_t: List[jnp.ndarray],
    init_belief: List[jnp.ndarray],
    final_belief: List[jnp.ndarray],
    query_times: List[jnp.ndarray],
    threshold: float,
    table_title: str = "Results",
    initial: bool = True,
) -> None:
    """
    Report the error metrics for the estimator.

    Args:
        x_t (List[np.ndarray]): List of true state values for each sequence.
        init_belief (List[np.ndarray]): List of initial belief values for each sequence.
        final_belief (List[np.ndarray]): List of final belief values for each sequence.
        query_times (List[np.ndarray]): List of query times for all chains.
    """
    # Compute results for each chain
    list_final_metrics = []
    n_chains = len(x_t)

    if initial:
        list_init_metrics = []

    for c in range(n_chains):
        if initial:
            list_init_metrics.append(compute_metrics(x_t[c], init_belief[c], query_times[c], threshold))
        list_final_metrics.append(compute_metrics(x_t[c], final_belief[c], query_times[c], threshold))

    if initial:
        init_metrics = compute_time_weighted_average(list_init_metrics, query_times)
    final_metrics = compute_time_weighted_average(list_final_metrics, query_times)

    # Print results table
    table = PrettyTable()
    table.title = table_title
    if initial:
        table.field_names = ["Metric", "Initial", "Final"]
    else:
        table.field_names = ["Metric", "Result"]

    if initial:
        table.add_row(["L1 Error", f"{init_metrics['l1_error']:.4f}", f"{final_metrics['l1_error']:.4f}"])
        table.add_row(["Mean Absolute Error", f"{init_metrics['mae']:.4f}", f"{final_metrics['mae']:.4f}"])
        table.add_row(["Accuracy", f"{init_metrics['accuracy']:.4f}", f"{final_metrics['accuracy']:.4f}"])
        table.add_row(["Balanced Acc", f"{init_metrics['balance_accuracy']:.4f}", f"{final_metrics['balance_accuracy']:.4f}"])
        table.add_row([f"Precision@{threshold}", f"{init_metrics['precision']:.4f}", f"{final_metrics['precision']:.4f}"])
        table.add_row([f"Recall@{threshold}", f"{init_metrics['recall']:.4f}", f"{final_metrics['recall']:.4f}"])
        table.add_row([f"F1@{threshold}", f"{init_metrics['f1_score']:.4f}", f"{final_metrics['f1_score']:.4f}"])
        table.add_row([f"MCC@{threshold}", f"{init_metrics['mcc']:.4f}", f"{final_metrics['mcc']:.4f}"])
        table.add_row(["Brier Score", f"{init_metrics['brier']:.4f}", f"{final_metrics['brier']:.4f}"])
        table.add_row(["ECE", f"{init_metrics['ece']:.4f}", f"{final_metrics['ece']:.4f}"])
    else:
        table.add_row(["L1 Error", f"{final_metrics['l1_error']:.4f}"])
        table.add_row(["Mean Absolute Error", f"{final_metrics['mae']:.4f}"])
        table.add_row(["Accuracy", f"{final_metrics['accuracy']:.4f}"])
        table.add_row(["Balanced Acc", f"{final_metrics['balance_accuracy']:.4f}"])
        table.add_row([f"Precision@{threshold}", f"{final_metrics['precision']:.4f}"])
        table.add_row([f"Recall@{threshold}", f"{final_metrics['recall']:.4f}"])
        table.add_row([f"F1@{threshold}", f"{final_metrics['f1_score']:.4f}"])
        table.add_row([f"MCC@{threshold}", f"{final_metrics['mcc']:.4f}"])
        table.add_row(["Brier Score", f"{final_metrics['brier']:.4f}"])
        table.add_row(["ECE", f"{final_metrics['ece']:.4f}"])

    logger.info(f"\n{table}\n")

    return table

@jax.jit
def compute_bic(n: int, log_likelihood: float, k: int) -> float:
    """
    Compute the Bayesian Information Criterion (BIC) for the given log-likelihood.

    Args:
        n (int): The number of observations.
        log_likelihood (float): The log-likelihood of the observations.
        k (int): The number of parameters in the model.

    Returns:
        float: The BIC value.
    """
    return -2 * log_likelihood + k * jnp.log(n)


@jax.jit
def compute_aic(log_likelihood: float, k: int) -> float:
    """
    Compute the Akaike Information Criterion (AIC) for the given log-likelihood.

    Args:
        log_likelihood (float): The log-likelihood of the observations.
        k (int): The number of parameters in the model.

    Returns:
        float: The AIC value.
    """
    return -2 * log_likelihood + 2 * k