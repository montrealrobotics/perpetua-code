from typing import Tuple, Dict

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from jax._src.tree_util import Partial

from src.utils.filter_state import FilterState
from src.utils.math_utils import ntn_pspace
from src.utils.jax_utils import bool_ifelse
from src.filters import EmergenceFilter, MixturePersistenceFilters


def init(log_s: Partial, params: Dict[str, ArrayLike], pi: ArrayLike, initialization_time: float = 0.0) -> FilterState:
    return FilterState.create(log_s, params, initialization_time, pi=pi)

@jax.jit
def update(mixture_state: FilterState, detector_output: bool, observation_time: float, P_M: float, P_F: float) -> FilterState:
    """
    Updates the filter by incorporating a new detector output.

    Args:
        mixture_state (FilterState): The current state of the mixture filter.
        detector_output (bool): A Boolean value output by the detector indicating whether the given feature was detected.
        observation_time (float): The timestamp for the detection.
        P_M (float): A floating-point value in the range [0, 1] indicating the missed detection probability
        of the feature detector.
        P_F (float): A floating-point value in the range [0, 1] indicating the false alarm probability of the feature
        detector.

    Returns:
        FilterState: The updated state of the mixture (emergence) filter.
    """
    return EmergenceFilter.update(mixture_state, detector_output, observation_time, P_M, P_F)

@jax.jit
def predict(mixture_state: FilterState, prediction_time: ArrayLike) -> Tuple[Array, Array]:
    """
    Predict the mixture emergence posterior probability of the mixture component with the largest posterior weight 
    p(X_t = 1 | C_k^*, Y_{1:N}), and return the probability of all other components p(X_t = 1 | C_k, Y_{1:N}) for all k.
    
    Args:
        state (FilterState): The current state of the mixture filter.
        prediction_time (ArrayLike): A floating-point value or array of values in the range 
        [last_observation_time, infinity) indicating the time(s) t for which to compute the posterior.

    Returns:
        Tuple[Array, Array]: Two JAX arrays. The first array contains the component with the larsets posterior weight
        with shape (1, prediction_time.size). The second array contains the conditional posterior probabilities 
        with shape (n_components, prediction_time.size).
    """

    mode, preds = MixturePersistenceFilters.predict(mixture_state, prediction_time)
    mode, preds = ntn_pspace(1 - mode), ntn_pspace(1 - preds)
    mode = bool_ifelse(jnp.all(mixture_state.last_observation_time == jnp.inf), jnp.zeros_like(mode), mode)
    preds = bool_ifelse(jnp.all(mixture_state.last_observation_time == jnp.inf), jnp.zeros_like(preds), preds)
    return mode, preds