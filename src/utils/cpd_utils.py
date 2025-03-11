from typing import List

import numpy as np
import ruptures as rpt

class TimeWeightedBernoulliCost:
    """Bernoulli likelihood cost accounting for uneven time intervals."""

    def __init__(self, signal, time_intervals, min_size: int = 5):
        self.signal = signal
        self.time_intervals = time_intervals
        self.min_size = min_size

    def error(self, start, end):
        segment = self.signal[start:end]
        intervals = self.time_intervals[start:end]
        if len(segment) == 0:
            return 0
        p = np.sum(segment * intervals) / np.sum(intervals)  # Weighted mean
        epsilon = 1e-10  
        return -np.sum(
            intervals * (segment * np.log(p + epsilon) + (1 - segment) * np.log(1 - p + epsilon))
        )

def detect_change_points(observations, times, min_size: int = 2, pentalty: float = 5.0) -> List[float]:
    # Apply the binary cost function
    custom_cost = TimeWeightedBernoulliCost(observations, times, min_size=min_size)
    # Use PELT with the custom cost
    algo = rpt.Pelt(custom_cost=custom_cost).fit(observations)
    # Detect change points
    change_points = algo.predict(pen=pentalty)  

    return change_points