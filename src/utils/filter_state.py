from typing import Union, Dict

import jax.lax
import jax.numpy as jnp
from flax import struct
from jax._src.tree_util import Partial
from jax.typing import ArrayLike

from src.utils.math_utils import logdiff


@struct.dataclass
class FilterState:
    # This is set at the start
    log_s: Partial
    params: Dict[str, Union[float, ArrayLike]]
    initialization_time: float

    # Update function modifies this
    last_observation_time: float
    log_likelihood: float = 0.0
    log_lower_evidence_sum: Union[float, ArrayLike] = 0.0  # todo jax lax cond
    log_conditional_evidence: Union[float, ArrayLike] = 0.0
    log_lower_evidence_sum_is_none: bool = True

    # used for mixtures
    pi: ArrayLike = jnp.ones(1)
    # Used only when resetting the mixture of perpetua filter
    pi_init: ArrayLike = jnp.ones(1)

    def __getitem__(self, item):
        return FilterState(
            log_s=self.log_s,
            params={key: value[item] for key, value in self.params.items()},
            initialization_time=self.initialization_time,
            last_observation_time=self.last_observation_time,
            log_likelihood=self.log_likelihood,
            log_lower_evidence_sum=self.log_lower_evidence_sum[item][None],
            log_conditional_evidence=self.log_conditional_evidence[item][None],
            log_lower_evidence_sum_is_none=self.log_lower_evidence_sum_is_none,
            pi=self.pi[item][None],
        )

    @property
    def log_survival_function(self):
        return lambda t: self.log_s(t - self.initialization_time, **self.params)

    @property
    def log_joint_evidence(self):
        return self.log_conditional_evidence + jnp.log(self.pi)

    @property
    def log_evidence(self):
        return jax.scipy.special.logsumexp(self.log_joint_evidence)

    @property
    def log_tau(self):
        return self.log_joint_evidence - self.log_evidence

    @staticmethod
    def create(log_s: Partial, params: Dict[str, float], initialization_time: float = 0.0, pi=jnp.ones(1)):
        self = FilterState(log_s, params, initialization_time, last_observation_time=initialization_time, pi=pi)

        self = self.replace(
            log_lower_evidence_sum=jnp.zeros_like(pi), log_conditional_evidence=jnp.zeros_like(pi), pi_init=pi
        )
        return self

    @property
    def num_components(self):
        return self.pi.shape[0]

    @property
    def shifted_log_survival_function(self):
        return lambda t: self.log_s(t - self.initialization_time, **self.params)

    @property
    def shifted_logdF(self):
        return lambda t1, t0: logdiff(self.shifted_log_survival_function(t0), self.shifted_log_survival_function(t1))


@struct.dataclass
class PerpetuaState:
    initialization_time: float
    last_observation_time: float

    delta_low: float
    delta_high: float
    num_steps: int = struct.field(pytree_node=False)  # Make num_steps static

    ef_state: FilterState
    pf_state: FilterState

    # Mixing parameters
    eps: float = 1e-1

    initial_state: int = 1
    current_state: bool = None  # note: set in code to be ``initial_state'' in the create func
    switch_counter: int = 1
    weight: Union[ArrayLike, float] = 1.0

    @staticmethod
    def create(
        ef_state: FilterState,
        pf_state: FilterState,
        initialization_time: float = 0.0,
        delta_high: float = 0.95,
        delta_low: float = 0.05,
        num_steps: int = 1,
        epsilon: float = 1e-1,
    ):
        ret = PerpetuaState(
            delta_low=delta_low,
            delta_high=delta_high,
            initialization_time=initialization_time,
            last_observation_time=initialization_time,
            num_steps=num_steps,
            ef_state=ef_state,
            pf_state=pf_state,
        )
        return ret.replace(current_state=ret.initial_state, eps=epsilon)
