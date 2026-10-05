# MLOps & Data Engineering: Preventing training data leakage via append-only logs and AS-OF time-travel joins

import bisect
import json
import time

class FeatureRecord:
    """Represents an immutable feature value update for an entity at a given timestamp."""
    def __init__(self, timestamp, values):
        self.timestamp = timestamp
        self.values = values  # dict of feature_name -> feature_value

    def __lt__(self, other):
        return self.timestamp < (other.timestamp if isinstance(other, FeatureRecord) else other)


class PointInTimeFeatureStore:
    """
    Append-only in-memory feature store ensuring point-in-time correctness (AS-OF joins)
    to eliminate future data leakage during training dataset assembly.
    """
    def __init__(self):
        # Maps entity_id -> list of FeatureRecord sorted chronologically
        self._registry = {}

    def log_feature_update(self, entity_id, timestamp, features):
        """Appends a new feature snapshot for an entity at a specific point in time."""
        if entity_id not in self._registry:
            self._registry[entity_id] = []

        record = FeatureRecord(timestamp, features)
        # Maintain sorted order by timestamp using bisect
        idx = bisect.bisect_right(self._registry[entity_id], record)
        self._registry[entity_id].insert(idx, record)

    def get_features_as_of(self, entity_id, as_of_timestamp):
        """
        Retrieves the exact active state of features valid at or before as_of_timestamp.
        Strictly ignores any updates logged after as_of_timestamp.
        """
        records = self._registry.get(entity_id, [])
        if not records:
            return None

        # Binary search for the rightmost record with timestamp <= as_of_timestamp
        # Using a dummy FeatureRecord for comparison
        dummy = FeatureRecord(as_of_timestamp, {})
        idx = bisect.bisect_right(records, dummy) - 1

        if idx < 0:
            # All available feature records occurred strictly AFTER as_of_timestamp
            return None

        # Merge chronological records up to idx to build the full entity state
        effective_features = {}
        for r in records[:idx + 1]:
            effective_features.update(r.values)

        return effective_features

    def join_training_observations(self, observations):
        """
        Performs point-in-time AS-OF joins for a dataset of labeled observations.
        observations: list of dicts with keys: 'entity_id', 'timestamp', 'label'
        """
        enriched_dataset = []
        for obs in observations:
            entity_id = obs["entity_id"]
            obs_time = obs["timestamp"]
            features = self.get_features_as_of(entity_id, obs_time)

            row = {
                "entity_id": entity_id,
                "observation_timestamp": obs_time,
                "label": obs["label"],
                "features": features
            }
            enriched_dataset.append(row)
        return enriched_dataset


if __name__ == "__main__":
    print("--- MLOps Data Pipelines: Point-in-Time Feature Store ---\n")

    store = PointInTimeFeatureStore()
    user_id = "user_japan_88"

    # 1. Simulate feature evolution over time
    print("Step 1: Logging chronological entity feature updates:")
    store.log_feature_update(user_id, timestamp=100.0, features={"account_age_days": 10, "avg_spend": 25.0})
    store.log_feature_update(user_id, timestamp=150.0, features={"fraud_risk_score": 0.12, "avg_spend": 45.0})
    store.log_feature_update(user_id, timestamp=200.0, features={"fraud_risk_score": 0.89, "chargeback_count": 1})

    print(f"  Logged updates for '{user_id}' at t=100, t=150, and t=200.")

    # 2. Historical Observation events (e.g. Transactions evaluated for fraud)
    observations = [
        {"entity_id": user_id, "timestamp": 120.0, "label": "legitimate"},
        {"entity_id": user_id, "timestamp": 160.0, "label": "legitimate"},
        {"entity_id": user_id, "timestamp": 210.0, "label": "fraudulent"}
    ]

    # 3. Perform point-in-time join
    print("\nStep 2: Executing Point-in-Time (AS-OF) Feature Join for Training:")
    training_data = store.join_training_observations(observations)

    for item in training_data:
        print(f" Observation at t={item['observation_timestamp']} (Label: {item['label']}):")
        print(f"Resolved Features: {item['features']}")

    # 4. Verify Leakage Prevention
    print("\nStep 3: Verifying Data Leakage Prevention:")
    obs_t120_features = training_data[0]["features"]
    assert "fraud_risk_score" not in obs_t120_features, "LEAK DETECTED: Features from t=150 leaked into t=120!"
    print("Leak check passed: Observation at t=120 does not contain features created at t=150 or t=200.")

    obs_t160_features = training_data[1]["features"]
    assert obs_t160_features["avg_spend"] == 45.0, "State mismatch for t=160 snapshot."
    assert "chargeback_count" not in obs_t160_features, "LEAK DETECTED: t=200 feature leaked into t=160!"
    print("Leak check passed: Observation at t=160 strictly contains features valid at or before t=150.")

    print("\nFinal Joined Training Dataset:")
    print(json.dumps(training_data, indent=2))