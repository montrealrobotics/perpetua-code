from typing import Tuple, Dict
from jax.typing import ArrayLike
from jax import Array
import jax
from jax._src.tree_util import Partial
import jax.numpy as jnp

from src.utils.filter_state import FilterState, PerpetuaState
from src.utils.math_utils import create_interpolated_array
from src.filters import PersistenceFilter
from src.filters import EmergenceFilter


def init(
    log_s_persistence: Partial,
    log_s_emergence: Partial,
    params_persistence: Dict[str, float],
    params_emergence: Dict[str, float],
    delta_low: float = 0.05,
    delta_high: float = 0.95,
    initialization_time: float = 0.0,
    num_steps: int = 1,
) -> PerpetuaState:
    """
    Initializes the Perpetua filter state.

    Args:
        log_s_persistence (Partial): The log survival function for the persistence filter.
        log_s_emergence (Partial): The log survival function for the emergence filter.
        params_persistence (Dict[str, float]): The parameters for the persistence filter.
        params_emergence (Dict[str, float]): The parameters for the emergence filter.
        delta_low (float, optional): The threshold for the low delta. Defaults to 0.05.
        delta_high (float, optional): The threshold for the high delta. Defaults to 0.95.
        initialization_time (float, optional): The initial time for the filter. Defaults to 0.0.
        num_steps (int, optional): The number of steps for interpolation. Defaults to 1.

    Returns:
        PerpetuaState: The initialized state of the Perpetua filter.
    """

    pf_state = PersistenceFilter.init(log_s_persistence, params_persistence, initialization_time)
    ef_state = EmergenceFilter.init(log_s_emergence, params_emergence, jnp.inf)

    return PerpetuaState.create(
        ef_state,
        pf_state,
        initialization_time=initialization_time,
        delta_d=delta_high,
        delta_b=delta_low,
        num_steps=num_steps,
    )


@jax.jit
def update(
    state: PerpetuaState, detector_output: bool, observation_time: float, P_M: float, P_F: float
) -> PerpetuaState:
    """
    Updates the single Perpetua model by incorporating a new detector output.

    Args:
        filter_state (PerpetuaState): The current state of the Perpetua model.
        detector_output (bool): A Boolean value output by the detector indicating whether the given feature was detected.
        observation_time (float): The timestamp for the detection.
        P_M (float): A floating-point value in the range [0, 1] indicating the missed detection probability
                     of the feature detector.
        P_F (float): A floating-point value in the range [0, 1] indicating the false alarm probability of the feature
                     detector.

    Returns:
        PerpetuaState: The updated state of the Perpetua filter.
    """
    # Pre-allocate the extended_times array
    extended_times = jnp.array([state.last_observation_time, observation_time])
    query_times, _ = create_interpolated_array(extended_times, int(state.num_steps))

    return _update(state, query_times, detector_output, observation_time, P_M, P_F)


@jax.jit
def _update(
    state: PerpetuaState,
    query_times: ArrayLike,
    detector_output: bool,
    observation_time: float,
    P_M: float,
    P_F: float,
) -> PerpetuaState:
    """
    Updates the Perpetua filter by incorporating a new detector output.

    Args:
        state (PerpetuaState): The current state of the Perpetua model.
        query_times (ArrayLike): An array of times at which to update the filter's state.
        detector_output (bool): A Boolean value output by the detector indicating whether the given feature was detected.
        observation_time (float): The timestamp for the detection.
        P_M (float): A floating-point value in the range [0, 1] indicating the missed detection probability
                     of the feature detector.
        P_F (float): A floating-point value in the range [0, 1] indicating the false alarm probability of the feature
                     detector.

    Returns:
        PerpetuaState: The updated state of a single component Perpetua model.
    """
    # Update state of the filter
    state = _update_state(state, query_times)

    # Update persistence filter
    state = state.replace(
        df_state=PersistenceFilter.update(state.pf_state, detector_output, observation_time, P_M, P_F)
    )

    # Update emergence filter
    state = state.replace(
        bf_state=EmergenceFilter.update(state.ef_state, detector_output, observation_time, P_M, P_F)
    )
    # Store last observation time
    state = state.replace(last_observation_time=observation_time)
    return state


@jax.jit
def _update_state(state: PerpetuaState, query_times: ArrayLike) -> PerpetuaState:
    """
    Updates the state of Perpetua by creating an array of query times between the last observation time
    and the given prediction time. It then simulates the switching between states based on the predicted emergence and
    persistence posterior probabilities. The state is updated according to the thresholds defined for emergence and persistence predictions.

    Args:
        state (PerpetuaState): The current state of Perpetua.
        query_times (ArrayLike): An array of times at which to update the filter's state.

    Returns:
        PerpetuaState: The updated state of Perpetua.
    """

    def state_identity(state):
        return state

    # Create array of query times
    def scan_body(state, query_time):
        emergence_p, persistence_p = predict_belief(query_time, state.ef_state, state.pf_state)

        def set_to_emergence(state):
            return state.replace(
                current_state=0,
                ef_state=_reset_emergence_filter(state.ef_state, init_time=query_time),
            )

        def reset_perpetua(state):
            return _reset_perpetua(state, init_time=query_time).replace(current_state=1)

        select_case_1 = jnp.logical_and(state.current_state, persistence_p <= state.delta_low).all()
        select_case_3 = jnp.logical_and(jnp.logical_not(state.current_state), emergence_p >= state.delta_high).all()

        (state) = jax.lax.cond(select_case_1, set_to_emergence, state_identity, state)

        (state) = jax.lax.cond(select_case_3, reset_perpetua, state_identity, state)

        return state, query_time

    state, _ = jax.lax.scan(
        scan_body,
        state,
        query_times,
    )

    return state


def predict(state: PerpetuaState, prediction_time: ArrayLike) -> Tuple[Array, Array, Array]:
    """
    Predicts the belief of Perpetua at a given prediction time.

    Args:
        prediction_time (Union[np.ndarray, float]): The time or array of times at which to predict the filter's behavior.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]: A tuple containing:
        - The belief array over the prediction times.
        - The switching states over the prediction times.
        - The weights over the prediction times.
    """
    # Simulate the switching behavior of the filter
    prediction_time = jnp.array(prediction_time)

    return simulate_switching(state, prediction_time)


@jax.jit
def step_simulate_switching(
    state: PerpetuaState, query_time: ArrayLike
) -> Tuple[PerpetuaState, Tuple[ArrayLike, ArrayLike, ArrayLike]]:
    """
    Simulates a single step of the switching behavior of a single component Perpetua.

    Args:
        state (PerpetuaState): The current state of Perpetua.
        query_time (ArrayLike): The time at which to simulate the filter's behavior.

    Returns:
        Tuple[PerpetuaState, Tuple[ArrayLike, ArrayLike, ArrayLike]]: A tuple containing:
        - The updated state of Perpetua.
        - A tuple with the belief array, the switching state, and the weights.
    """
    pf = state.pf_state
    ef = state.ef_state

    emergence_p, persistence_p = predict_belief(query_time, state.ef_state, state.pf_state)

    def init_emergence(state, emergence_filter, persistence_filter, query_time, emergence_p, persistence_p):
        emergence_filter = _reset_emergence_filter(emergence_filter, query_time)
        emergence_p, persistence_p = predict_belief(query_time, emergence_filter, persistence_filter)
        return 0, emergence_filter, persistence_filter, emergence_p, persistence_p

    def identity(state, emergence_filter, persistence_filter, query_time, emergence_p, persistence_p):
        return (state), emergence_filter, persistence_filter, emergence_p, persistence_p

    def init_persistence(state, emergence_filter, persistence_filter, query_time, emergence_p, persistence_p):
        persistence_filter = _reset_persistence_filter(persistence_filter, query_time)
        emergence_p, persistence_p = predict_belief(query_time, emergence_filter, persistence_filter)
        return 1, emergence_filter, persistence_filter, emergence_p, persistence_p

    select_case_1 = jnp.logical_and(state.current_state, persistence_p <= state.delta_low).all()
    select_case_3 = jnp.logical_and((jnp.logical_not(state.current_state)), emergence_p >= state.delta_high).all()

    (switch_state, ef, pf, emergence_p, persistence_p) = jax.lax.cond(
        select_case_1, init_emergence, identity, state.current_state, ef, pf, query_time, emergence_p, persistence_p
    )
    (switch_state, ef, pf, emergence_p, persistence_p) = jax.lax.cond(
        select_case_3, init_persistence, identity, switch_state, ef, pf, query_time, emergence_p, persistence_p
    )

    new_state = state.replace(
        current_state=switch_state,
        pf_state=pf,
        ef_state=ef,
    )

    return new_state, (jnp.concatenate((emergence_p, persistence_p)), switch_state, new_state.weight)


@jax.jit
def simulate_switching(
    state: PerpetuaState, prediction_times: ArrayLike
) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """
    Simulates the switching behavior of Perpetua over a series of prediction times.

    Args:
        state (PerpetuaState): The current state of Perpetua.
        prediction_times (ArrayLike): An array of times at which to predict the model's behavior.

    Returns:
        Tuple[ArrayLike, ArrayLike, ArrayLike]: A tuple containing:
        - The belief array over the prediction times.
        - The switching states over the prediction times.
        - The weights over the prediction times.
    """
    prediction_times = jnp.atleast_1d(prediction_times)
    extended_times = jnp.empty(prediction_times.shape[0] + 1, dtype=prediction_times.dtype)
    extended_times = extended_times.at[0].set(state.last_observation_time)
    extended_times = extended_times.at[1:].set(prediction_times)
    query_times, collect_query_times = create_interpolated_array(extended_times, state.num_steps)

    (_, (belief, switch_states, weights)) = jax.lax.scan(step_simulate_switching, state, query_times, unroll=1)
    return collect_query_times(belief), collect_query_times(switch_states), collect_query_times(weights)


def predict_belief(prediction_time: ArrayLike, emergence_filter: FilterState, persistence_filter: FilterState) -> Tuple[ArrayLike, ArrayLike]:
    """
    Predicts the posterior probabilities of the emergence and persistence filters at a given prediction time.

    Args:
        prediction_time (ArrayLike): The time at which to predict the filter's behavior.

    Returns:
        Tuple[ArrayLike, ArrayLike]: A tuple containing:
        - The posterior probability of the emergence filter.
        - The posterior probability of the persistence filter.
    """
    # Predict with emergence filter
    emergence_pred = EmergenceFilter.predict(emergence_filter, prediction_time)
    # Predict with persistence filter
    persistence_pred = PersistenceFilter.predict(persistence_filter, prediction_time)
    return emergence_pred, persistence_pred


@jax.jit
def _reset_persistence_filter(pf_state: FilterState, init_time: float) -> FilterState:
    """
    Reset the persistence filter to its initial state at a given initialization time.

    Args:
        init_time (float): The time at which to reset the filter.

    Returns:
        FilterState: The reset state of the persistence filter.
    """
    return PersistenceFilter.init(pf_state.log_s, pf_state.params, init_time)


@jax.jit
def _reset_emergence_filter(ef_state: FilterState, init_time: float) -> FilterState:
    """
    Reset the emergence filter to its initial state at a given initialization time.

    Args:
        init_time (float): The time at which to reset the filter.

    Returns:
        FilterState: The reset state of the emergence filter.
    """
    return EmergenceFilter.init(ef_state.log_s, ef_state.params, init_time)


@jax.jit
def _reset_perpetua(state: PerpetuaState, init_time: float) -> PerpetuaState:
    """
    Reset perpetua after a full persistence/emergence cycle.

    Args:
        state (PerpetuaState): The current state of perpetua.
        init_time (float): The time at which to reset the filter.

    Returns:
        PerpetuaState: The reset state of a single component perpetua.
    """
    pf_state = _reset_persistence_filter(pf_state=state.pf_state, init_time=init_time)

    return state.replace(
        current_state=1,
        switch_counter=state.switch_counter + 1,
        pf_state=pf_state,
        last_observation_time=init_time,
    )
