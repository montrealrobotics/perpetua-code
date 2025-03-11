from typing import Dict
import functools
from jax import Array 
from jax.typing import ArrayLike 

from src.utils.filter_state import FilterState
from src.utils.math_utils import clip_log_prob, clip_prob
from src.utils.jax_utils import bool_ifelse, logsumexp
import jax

from jax._src.tree_util import Partial
import jax.numpy as jnp



def init(log_s: Partial, params: Dict[str, float], initialization_time: float = 0.0) -> FilterState:
    return FilterState.create(log_s, params, initialization_time)

@jax.jit
def update(filter_state: FilterState, detector_output: bool, observation_time: float, P_M: float, P_F: float) -> FilterState:
    """
    Updates the filter by incorporating a new detector output.

    Args:
        detector_output (bool):  A Boolean value output by the detector indicating whether the given feature was detected.
        observation_time (float):  The timestamp for the detection.
        P_M (float):  A floating-point value in the range [0, 1] indicating the missed detection probability
        of the feature detector.
        P_F (float):  A floating-point value in the range [0,1] indicating the false alarm probability of the feature
        detector.

    Returns:
        FilterState: The updated filter state

    """

    def lles_is_not_none(filter_state, observation_time):
        # _log_lower_evidence_sum has been previously initialized, so just update it in the usual way
        rhs = filter_state.log_likelihood + filter_state.shifted_logdF(observation_time, filter_state.last_observation_time)
        updated_log_lower_evidence_sum = logsumexp(filter_state.log_lower_evidence_sum, rhs)
        updated_log_lower_evidence_sum = updated_log_lower_evidence_sum + jnp.log(bool_ifelse(detector_output, P_F, 1 - P_F))

        return updated_log_lower_evidence_sum

    def lles_is_none(filter_state, observation_time):
        # This is the first observation we've incorporated, so initialize the lower running sum logLY here.
        exp_result = clip_prob(jnp.exp(clip_log_prob(filter_state.shifted_log_survival_function(observation_time))))
        log1_minus_ST = jnp.log1p(-exp_result)

        # Initialize Log lower evidence
        updated_log_lower_evidence_sum = jnp.log(bool_ifelse(detector_output, P_F, 1 - P_F)) + log1_minus_ST
        return updated_log_lower_evidence_sum

    updated_log_lower_evidence_sum = bool_ifelse(filter_state.log_lower_evidence_sum_is_none, lles_is_none(filter_state, observation_time), lles_is_not_none(filter_state, observation_time))
    # Update the lower sum LY
    filter_state = filter_state.replace(log_lower_evidence_sum=updated_log_lower_evidence_sum, log_lower_evidence_sum_is_none=False)

    # Update the measurement likelihood pY_tN
    log_likelihood = filter_state.log_likelihood + jnp.log(bool_ifelse(detector_output, 1.0 - P_M, P_M))
    filter_state = filter_state.replace(log_likelihood=log_likelihood)

    # Update the last observation time
    filter_state = filter_state.replace(last_observation_time=observation_time)

    # Compute the marginal (evidence) probability pY
    log_conditional_evidence = logsumexp(
        filter_state.log_lower_evidence_sum,
        filter_state.log_likelihood + filter_state.shifted_log_survival_function(filter_state.last_observation_time),
    )
    filter_state = filter_state.replace(log_conditional_evidence=log_conditional_evidence)
    return filter_state


@jax.jit
def predict(filter_state: FilterState, prediction_time: ArrayLike) -> Array:
    """
    Predicts the posterior persistence probability p(X_t = 1 | Y_{1:N}) for one or multiple times.

    Args:
        filter_state (FilterState): The current state of the filter.
        prediction_time (ArrayLike): A floating-point value or array of values in the range
        [last_observation_time, infinity) indicating the time(s) t for which to compute the posterior survival belief
        p(X_t = 1 | Y_{1:N})

    Returns:
        (Array) A 1xN array containing the posterior persistence probabilities p(X_t = 1 | Y_{1:N})
    """
    prediction_time = jnp.atleast_1d(prediction_time)
    def predict_per_timestep(filter_state, prediction_time: float):
        # Ensure prediction_time is a jnp array for consistent vectorized operations
        query_time = jnp.atleast_1d(prediction_time)

        # Compute the posterior persistence probability 
        posterior_prob = jnp.exp(
            filter_state.log_likelihood - filter_state.log_conditional_evidence + filter_state.shifted_log_survival_function(
                query_time))
        return clip_prob(jnp.atleast_1d(posterior_prob))
    
    return jax.vmap(functools.partial(predict_per_timestep, filter_state), out_axes=1)(prediction_time)
