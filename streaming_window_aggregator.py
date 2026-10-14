# MLOps & Real-Time Feature Engineering: Sliding time windows, Welford's streaming variance, and exponential decay

import collections
import math
import time
import json
from typing import Dict, List, Optional, Tuple

class StreamEvent:
    """Represents a single time-stamped numerical observation in an entity event stream."""
    def __init__(self, value: float, timestamp: Optional[float] = None):
        self.value = float(value)
        self.timestamp = timestamp if timestamp is not None else time.time()


class StreamingSlidingWindowAggregator:
    """
    Maintains an active sliding time-window per entity to compute real-time
    aggregations (count, sum, mean, sample variance, and exponential decay).
    """
    def __init__(self, window_seconds: float = 60.0, half_life_seconds: float = 15.0):
        self.window_seconds = window_seconds
        self.decay_lambda = math.log(2) / half_life_seconds if half_life_seconds > 0 else 0.0
        # Maps entity_id -> deque of StreamEvent
        self._buffers: Dict[str, collections.deque] = {}

    def _evict_stale_events(self, entity_id: str, current_time: float):
        """Purges events outside the sliding time horizon [current_time - window_seconds]."""
        buffer = self._buffers.get(entity_id)
        if not buffer:
            return

        horizon = current_time - self.window_seconds
        while buffer and buffer[0].timestamp < horizon:
            buffer.popleft()

        if not buffer:
            del self._buffers[entity_id]

    def record_event(self, entity_id: str, value: float, timestamp: Optional[float] = None):
        """Appends a new observation into the entity's sliding window."""
        event_time = timestamp if timestamp is not None else time.time()
        if entity_id not in self._buffers:
            self._buffers[entity_id] = collections.deque()

        self._buffers[entity_id].append(StreamEvent(value, event_time))
        self._evict_stale_events(entity_id, event_time)

    def compute_features(self, entity_id: str, query_time: Optional[float] = None) -> Dict[str, float]:
        """
        Computes real-time streaming feature snapshot for an entity:
        count, sum, mean, stddev (Welford's algorithm), and exponentially decayed sum.
        """
        now = query_time if query_time is not None else time.time()
        self._evict_stale_events(entity_id, now)

        buffer = self._buffers.get(entity_id)
        if not buffer:
            return {
                "window_count": 0,
                "window_sum": 0.0,
                "window_mean": 0.0,
                "window_std": 0.0,
                "decayed_weighted_sum": 0.0
            }

        # Compute incremental rolling statistics
        count = 0
        total_sum = 0.0
        m2 = 0.0
        mean = 0.0
        decayed_sum = 0.0

        for event in buffer:
            count += 1
            x = event.value
            total_sum += x

            # Welford's streaming algorithm for stable mean & variance
            delta = x - mean
            mean += delta / count
            delta2 = x - mean
            m2 += delta * delta2

            # Half-life decay weight: e^(-lambda * dt)
            dt = max(0.0, now - event.timestamp)
            weight = math.exp(-self.decay_lambda * dt)
            decayed_sum += x * weight

        variance = (m2 / (count - 1)) if count > 1 else 0.0
        std_dev = math.sqrt(variance)

        return {
            "window_count": count,
            "window_sum": round(total_sum, 4),
            "window_mean": round(mean, 4),
            "window_std": round(std_dev, 4),
            "decayed_weighted_sum": round(decayed_sum, 4)
        }


if __name__ == "__main__":
    print("--- MLOps Streaming Features: Sliding Window Aggregator ---\n")

    aggregator = StreamingSlidingWindowAggregator(window_seconds=10.0, half_life_seconds=3.0)
    entity = "device_tokyo_edge_9"
    base_t = 1000.0

    print("Step 1: Ingesting streaming transaction events over time:")
    # Simulating events arriving at t = 1000, 1002, 1005, 1008
    aggregator.record_event(entity, value=100.0, timestamp=base_t + 0.0)
    aggregator.record_event(entity, value=150.0, timestamp=base_t + 2.0)
    aggregator.record_event(entity, value=200.0, timestamp=base_t + 5.0)
    aggregator.record_event(entity, value=300.0, timestamp=base_t + 8.0)

    # Snapshot at t = 1008.5 (all 4 events fall within the 10-second window [998.5 -> 1008.5])
    snap1 = aggregator.compute_features(entity, query_time=base_t + 8.5)
    print(f"Snapshot at t=1008.5s (Full Window):")
    print(f"Count: {snap1['window_count']} | Mean: {snap1['window_mean']} | Std: {snap1['window_std']}")
    print(f" Decayed Weighted Sum: {snap1['decayed_weighted_sum']}")

    print("\nStep 2: Advancing time to trigger window sliding and eviction:")
    # At t = 1013.0, the window horizon is 1003.0.
    # Events at t=1000 and t=1002 must be evicted. Only t=1005 and t=1008 remain.
    snap2 = aggregator.compute_features(entity, query_time=base_t + 13.0)
    print(f"Snapshot at t=1013.0s (Window [1003.0 -> 1013.0]):")
    print(f"Active Count: {snap2['window_count']} (Evicted 2 expired points)")
    print(f"Updated Mean: {snap2['window_mean']} | Sum: {snap2['window_sum']}")
    print(f"Decayed Weighted Sum: {snap2['decayed_weighted_sum']}")

    print("\nFull Aggregated Feature Vector (Live Serving Output):")
    print(json.dumps(snap2, indent=2))