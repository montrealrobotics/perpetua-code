from typing import Tuple, Dict

from jax.typing import ArrayLike
from jax import Array
import jax
from jax._src.tree_util import Partial
import jax.numpy as jnp
import functools

from src.utils.filter_state import FilterState, PerpetuaState
from src.utils.math_utils import create_interpolated_array
from src.filters import MixtureEmergenceFilters, MixturePersistenceFilters, SinglePerpetua


def init(
    log_s_persistence: Partial,
    log_s_emergence: Partial,
    params_persistence: Dict[str, ArrayLike],
    params_emergence: Dict[str, ArrayLike],
    pi_persistence: ArrayLike,
    pi_emergence: ArrayLike,
    delta_low: float = 0.05,
    delta_high: float = 0.95,
    initialization_time: float = 0.0,
    num_steps: int = 1,
    eps: float = 1e-1,
) -> PerpetuaState:
    """
    Initializes the Perpetua filter state.

    Args:
        log_s_persistence (Partial): The log survival function for the persistence filter.
        log_s_emergence (Partial): The log survival function for the emergence filter.
        params_persistence (Dict[str, ArrayLike]): Parameters for the persistence filter.
        params_emergence (Dict[str, ArrayLike]): Parameters for the emergence filter.
        pi_persistence (ArrayLike): Mixing coefficients for the persistence model.
        pi_emergence (ArrayLike): Mixing coefficients for the emergence model.
        delta_low (float, optional): The lower threshold to switch to emergence. Defaults to 0.05.
        delta_high (float, optional): The upper threshold to switch to persistence. Defaults to 0.95.
        initialization_time (float, optional): The initial time for the filter. Defaults to 0.0.
        num_steps (int, optional): The number of steps for interpolation. Defaults to 1.
        eps (float, optional): The epsilon value for weight adjustment. Defaults to 1e-1.

    Returns:
        PerpetuaState: The initialized state of the Perpetua filter.
    """

    pf_state = MixturePersistenceFilters.init(
        log_s_persistence, params_persistence, pi_persistence, initialization_time
    )
    ef_state = MixtureEmergenceFilters.init(log_s_emergence, params_emergence, pi_emergence, jnp.inf)

    return PerpetuaState.create(ef_state, pf_state, initialization_time, delta_high, delta_low, num_steps, epsilon=eps)


@jax.jit
def update(
    state: PerpetuaState, detector_output: bool, observation_time: float, P_M: float, P_F: float
) -> PerpetuaState:
    """
    Update Perpetua by incorporating a new detector output.

    Args:
        state (PerpetuaState): The current state of the Perpetua filter.
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
    p_m: float,
    p_f: float,
) -> PerpetuaState:
    """
    Updates Perpetua by incorporating a new detector output.

    Args:
        state (PerpetuaState): The current state of the Perpetua filter.
        query_times (ArrayLike): An array of times at which to update the filter's state.
        detector_output (bool): A Boolean value output by the detector indicating whether the given feature was detected.
        observation_time (float): The timestamp for the detection.
        p_m (float): A floating-point value in the range [0, 1] indicating the missed detection probability
                     of the feature detector.
        p_f (float): A floating-point value in the range [0, 1] indicating the false alarm probability of the feature
                     detector.

    Returns:
        PerpetuaState: The updated state of the Perpetua filter.
    """
    # Update state of the filter
    state = _update_state(state, query_times)

    # Update persistence filter
    state = state.replace(
        pf_state=MixturePersistenceFilters.update(state.pf_state, detector_output, observation_time, p_m, p_f)
    )

    # Update emergence filter
    state = state.replace(
        ef_state=MixtureEmergenceFilters.update(state.ef_state, detector_output, observation_time, p_m, p_f)
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
        filter_state (PerpetuaState): The current state of Perpetua.
        query_times (ArrayLike): An array of times at which to update the filter's state.

    Returns:
        PerpetuaState: The updated state of the Perpetua filter.
    """

    def state_identity(state):
        return state

    def scan_body(state, query_time, components_state):
        belief, _, weights = predict_belief(components_state, query_time)
        # Normalize weights
        weights = weights / weights.sum()
        # Pick best component of the mixture
        prediction = belief[jnp.argmax(weights)]

        def set_to_emergence(state):
            return _reset_emergence_filter(state, init_time=query_time)

        def reset_perpetua(state):
            return _reset_perpetua(state, init_time=query_time)

        select_case_1 = jnp.logical_and(state.current_state, prediction <= state.delta_low).all()
        select_case_3 = jnp.logical_and(jnp.logical_not(state.current_state), prediction >= state.delta_high).all()

        (state) = jax.lax.cond(select_case_1, set_to_emergence, state_identity, state)

        (state) = jax.lax.cond(select_case_3, reset_perpetua, state_identity, state)

        return state, query_time

    # Create n_components^2 perpetua filters
    components_state = _init_filters(state)
    state, _ = jax.lax.scan(
        functools.partial(scan_body, components_state=components_state), state, query_times
    )

    return state


@jax.jit
def predict_belief(states: PerpetuaState, query_time: ArrayLike) -> Tuple[Array, Array, Array]:
    """
    Predicts the belief of perpetua at a given query time.

    Args:
        states (PerpetuaState): Perpetua models state.
        query_time (ArrayLike): The time at which to predict the belief.

    Returns:
        Tuple[Array, Array, Array]: The predicted belief, states, and weights of the individual perpetua models.
    """

    def filter_predict(query_time, state: PerpetuaState) -> Tuple[Array, Array, Array]:
        belief, states, weights = SinglePerpetua.predict(state, query_time)
        belief = jnp.atleast_2d(belief)
        prediction = belief[jnp.arange(belief.shape[0]), states.astype(int)]
        return prediction, states, weights

    predictions, states, weights = jax.vmap(functools.partial(filter_predict, query_time))(states)
    return predictions, states, weights


@jax.jit
def predict(state: PerpetuaState, prediction_time: ArrayLike) -> Tuple[Array, Array, Array]:
    """
    Predicts the posterior probability of the emergence and persistence filters at a given prediction time.

    Args:
        prediction_time (Union[np.ndarray, float]): The time or array of times at which to predict the belief.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]: A tuple containing:
        - The belief array over the prediction times.
        - The switching states over the prediction times.
        - The weights over the prediction times.
    """
    return simulate_switching(state, prediction_time)


@jax.jit
def simulate_switching(
    state: PerpetuaState, prediction_times: ArrayLike
) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """
    Simulates the switching behavior of Perpetua over a series of prediction times.

    Args:
        perpetua_state (PerpetuaState): The current state of Perpetua.
        prediction_times (ArrayLike): An array of times at which to predict the filter's behavior.

    Returns:
        Tuple[Array, Array, Array]: A tuple containing:
        - The belief array over the prediction times.
        - The switching states over the prediction times.
        - The weights over the prediction times.
    """
    # Pre-allocate the extended_times array
    prediction_times = jnp.atleast_1d(prediction_times)
    extended_times = jnp.empty(prediction_times.shape[0] + 1, dtype=prediction_times.dtype)
    extended_times = extended_times.at[0].set(state.last_observation_time)
    extended_times = extended_times.at[1:].set(prediction_times)
    query_times, collect_query_times = create_interpolated_array(extended_times, state.num_steps)

    # Create n_components^2 perpetua filters
    components_state = _init_filters(state)
    belief, states, weights = predict_belief(components_state, query_times)
    belief = jax.vmap(collect_query_times)(belief).T
    states = jax.vmap(collect_query_times)(states).T
    weights = jax.vmap(collect_query_times)(weights).T

    weights = jnp.atleast_2d(weights)
    weights = weights / weights.sum(axis=1, keepdims=True)

    return jnp.atleast_2d(belief), jnp.atleast_2d(states), weights


@jax.jit
def _reset_persistence_filter(state: PerpetuaState, init_time: float) -> FilterState:
    """
    Reset the mixture persistence filters to its initial state at a given initialization time.

    Args:
        init_time (float): The time at which to reset the filter.

    Returns:
        FilterState: The reset state of the mixture of persistence filters.
    """
    pf_state = state.pf_state
    # Resample weights
    posterior_weights, prior_weights = jnp.exp(pf_state.log_tau), pf_state.pi_init
    weights = posterior_weights * (1 - state.eps) + prior_weights * state.eps
    # Init new persistence filter and save original pi value
    new_pf_state = MixturePersistenceFilters.init(pf_state.log_s, pf_state.params, weights, init_time)
    new_pf_state = new_pf_state.replace(pi_init=prior_weights)

    # Update filter state
    new_state = state.replace(pf_state=new_pf_state, current_state=1)
    return new_state


@jax.jit
def _reset_emergence_filter(state: PerpetuaState, init_time: float) -> FilterState:
    """
    Reset the mixture emergence filters to its initial state at a given initialization time.

    Args:
        init_time (float): The time at which to reset the filter.

    Returns:
        FilterState: The reset state of the mixture of emergence filters.
    """
    ef_state = state.ef_state
    # Resample weights
    posterior_weights, prior_weights = jnp.exp(ef_state.log_tau), ef_state.pi_init
    weights = posterior_weights * (1 - state.eps) + prior_weights * state.eps
    # Init new emergence filter and save original pi value
    new_ef_state = MixtureEmergenceFilters.init(ef_state.log_s, ef_state.params, weights, init_time)
    new_ef_state = new_ef_state.replace(pi_init=prior_weights)

    # Update filter state
    new_state = state.replace(ef_state=new_ef_state, current_state=0)
    return new_state


@jax.jit
def _reset_perpetua(perpetua_state: PerpetuaState, init_time: float) -> FilterState:
    """
    Reset perpetua after a full persistence/emergence cycle.

    Args:
        perpetua_state (PerpetuaState): The current state of perpetua.
        init_time (float): The time at which to reset the filter.

    Returns:
        PerpetuaState: The reset state of perpetua.
    """
    perpetua_state = _reset_persistence_filter(state=perpetua_state, init_time=init_time)

    return perpetua_state.replace(
        switch_counter = perpetua_state.switch_counter + 1,
        last_observation_time = init_time,
    )


@jax.jit
def _init_filters(state: PerpetuaState) -> PerpetuaState:
    """
    Initialize k^2 perpetua filters where k is the number of components in the mixture.

    Returns:
        PerpetuaState: The initialized state of the perpetua filters.
    """

    # Helper function to create a single PerpetuaState for a given (i, j)
    def create_filter(state: PerpetuaState, pair: ArrayLike) -> PerpetuaState:
        i, j = pair
        weight = jnp.exp(state.ef_state.log_tau)[i] + jnp.exp(state.pf_state.log_tau)[j]
        emergence_filter = state.ef_state[i]
        persistence_filter = state.pf_state[j]
        f = PerpetuaState.create(
            emergence_filter,
            persistence_filter,
            state.initialization_time,
            state.delta_high,
            state.delta_low,
            state.num_steps,
        )
        return f.replace(
            current_state=state.current_state, last_observation_time=state.last_observation_time, weight=weight
        )

    # Generate all (i, j) pairs
    num_df = jnp.arange(state.pf_state.num_components)
    num_bf = jnp.arange(state.ef_state.num_components)
    x_grid, y_grid = jnp.meshgrid(num_bf, num_df, indexing='ij')
    pairs = jnp.stack([x_grid.flatten(), y_grid.flatten()], axis=-1)

    # Generate mode (i, j) pair 
    # pairs = jnp.array([jnp.argmax(perpetua_state.ef_state.log_tau), jnp.argmax(perpetua_state.pf_state.log_tau)])[
    #     None, :
    # ]

    filters = jax.vmap(functools.partial(create_filter, state))(pairs)

    return filters
