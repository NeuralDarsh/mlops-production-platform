# MLOps Telemetry & Streaming Evaluation: Delayed-label reconciliation, TTL prediction stash, and rolling performance alerts

import time
import collections
import json
from typing import Any, Dict, Optional, Tuple

class PredictionRecord:
    """Stores initial inference metadata awaiting delayed ground-truth arrival."""
    def __init__(self, prediction_id: str, prediction_value: float, features: Dict[str, Any], ttl_seconds: float):
        self.prediction_id = prediction_id
        self.prediction_value = prediction_value
        self.features = features
        self.timestamp = time.time()
        self.expires_at = self.timestamp + ttl_seconds

    def is_expired(self, current_time: float) -> bool:
        return current_time > self.expires_at


class GroundTruthReconciliationWatchdog:
    """
    Reconciles delayed ground-truth labels with buffered predictions
    and monitors real-time streaming model accuracy against an SLA threshold.
    """
    def __init__(self, stash_ttl_sec: float = 5.0, evaluation_window_size: int = 10, min_accuracy_threshold: float = 0.70):
        self.stash_ttl_sec = stash_ttl_sec
        self.evaluation_window_size = evaluation_window_size
        self.min_accuracy_threshold = min_accuracy_threshold

        self._prediction_stash: Dict[str, PredictionRecord] = {}
        self._evaluation_window: collections.deque = collections.deque(maxlen=evaluation_window_size)
        self.retraining_alerts_triggered = 0

    def purge_expired_predictions(self):
        """Sweeps and removes predictions where no label arrived before the TTL window closed."""
        now = time.time()
        expired_ids = [pid for pid, rec in self._prediction_stash.items() if rec.is_expired(now)]
        for pid in expired_ids:
            del self._prediction_stash[pid]
        return len(expired_ids)

    def record_prediction(self, prediction_id: str, predicted_score: float, features: Dict[str, Any]):
        """Buffers a live prediction awaiting subsequent ground truth."""
        self.purge_expired_predictions()
        self._prediction_stash[prediction_id] = PredictionRecord(
            prediction_id=prediction_id,
            prediction_value=predicted_score,
            features=features,
            ttl_seconds=self.stash_ttl_sec
        )
        print(f"[PREDICTION STASHED] ID='{prediction_id}' | Predicted Score={predicted_score}")

    def ingest_ground_truth(self, prediction_id: str, actual_label: int) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Reconciles arriving true label with stashed prediction.
        Computes metric and evaluates degradation thresholds.
        """
        self.purge_expired_predictions()
        record = self._prediction_stash.pop(prediction_id, None)
        if not record:
            print(f"[UNMATCHED / EXPIRED LABEL] Prediction ID '{prediction_id}' not found in buffer.")
            return False, None

        predicted_binary = 1 if record.prediction_value >= 0.5 else 0
        is_correct = (predicted_binary == actual_label)
        self._evaluation_window.append({
            "prediction_id": prediction_id,
            "predicted": predicted_binary,
            "actual": actual_label,
            "is_correct": is_correct,
            "latency_to_label_sec": round(time.time() - record.timestamp, 3)
        })

        correct_count = sum(1 for item in self._evaluation_window if item["is_correct"])
        current_accuracy = correct_count / len(self._evaluation_window)
        print(f"[LABEL RECONCILED] ID='{prediction_id}' | Actual={actual_label} | Correct={is_correct} | Window Acc={current_accuracy:.1%}")

        alert_triggered = False
        if len(self._evaluation_window) >= self.evaluation_window_size:
            if current_accuracy < self.min_accuracy_threshold:
                self.retraining_alerts_triggered += 1
                alert_triggered = True
                print(f"[DEGRADATION ALERT #{self.retraining_alerts_triggered}] Accuracy ({current_accuracy:.1%}) breached SLO ({self.min_accuracy_threshold:.1%})!")

        return True, {
            "prediction_id": prediction_id,
            "current_accuracy": round(current_accuracy, 4),
            "alert_triggered": alert_triggered
        }


if __name__ == "__main__":
    print("--- MLOps Telemetry: Ground-Truth Reconciliation & Degradation Watchdog ---\n")

    watchdog = GroundTruthReconciliationWatchdog(
        stash_ttl_sec=2.0,
        evaluation_window_size=5,
        min_accuracy_threshold=0.60
    )

    print("Step 1: Ingesting streaming live predictions:")
    for i in range(1, 7):
        pred_id = f"tx_eval_{i:03d}"
        watchdog.record_prediction(pred_id, predicted_score=0.85, features={"user_id": 100 + i})

    print("\nStep 2: Arriving delayed labels matching initial predictions (Accurate stream):")
    time.sleep(0.05)
    watchdog.ingest_ground_truth("tx_eval_001", actual_label=1)
    watchdog.ingest_ground_truth("tx_eval_002", actual_label=1)
    watchdog.ingest_ground_truth("tx_eval_003", actual_label=1)

    print("\nStep 3: Incoming ground truth reveals model degradation (Triggers Watchdog):")
    watchdog.ingest_ground_truth("tx_eval_004", actual_label=0) 
    watchdog.ingest_ground_truth("tx_eval_005", actual_label=0) 
    watchdog.ingest_ground_truth("tx_eval_006", actual_label=0)  

    print("\nStep 4: Testing TTL expiration for abandoned prediction (No label arrives):")
    watchdog.record_prediction("tx_eval_abandoned", predicted_score=0.20, features={})
    print("  Sleeping 2.1s to allow prediction TTL to expire...")
    time.sleep(2.1)
    matched, _ = watchdog.ingest_ground_truth("tx_eval_abandoned", actual_label=0)
    print(f"  Attempting join after TTL: Matched = {matched}")