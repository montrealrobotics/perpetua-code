from typing import Dict

from jax import Array 
from jax.typing import ArrayLike 
import jax
from jax._src.tree_util import Partial
import jax.numpy as jnp

from src.utils.math_utils import ntn_pspace
from src.utils.jax_utils import bool_ifelse

from src.utils.filter_state import FilterState
from src.filters import PersistenceFilter


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
    def no_update_state(filter_state, *args) -> FilterState:
        return filter_state

    def perform_update(filter_state, detector_output, observation_time, P_M, P_F) -> FilterState:
        return PersistenceFilter.update(filter_state, detector_output, observation_time, 1 - P_F, 1 - P_M)

    return jax.lax.cond(
        jnp.isinf(filter_state.initialization_time),  # Condition
        no_update_state,                              # Function if True
        perform_update,                               # Function if False
        filter_state, detector_output, observation_time, P_M, P_F  # Arguments passed to the functions
    )

@jax.jit
def predict(filter_state: FilterState, prediction_time: ArrayLike) -> Array:
    """
    Predicts the posterior emergence probability p(X_t = 1 | Y_{1:N}) for one or multiple times.

    Args:
        prediction_time (ArrayLike): A floating-point value or array of values in the range
        [last_observation_time, infinity), indicating the time(s) t for which to compute the posterior emergence belief
        p(X_t = 1 | Y_{1:N}).

    Returns:
        (Array) A 1D array containing the posterior emergence probability p(X_t = 1 | Y_{1:N}) for the given 
        prediction time(s)
    """

    ret = PersistenceFilter.predict(filter_state, prediction_time)
    ret = 1 - ret

    ret = ntn_pspace(ret)

    return bool_ifelse(jnp.all(filter_state.last_observation_time == jnp.inf), jnp.zeros_like(ret), ret)