# MLOps Deployment & Delivery: Stochastic traffic splitting, user stickiness hashing, and automated health rollback

import hashlib
import random
import time
from typing import Any, Callable, Dict, Optional

class ModelEndpointMetrics:
    """Tracks live invocation health for a deployed model target."""
    def __init__(self, name: str):
        self.name = name
        self.total_requests = 0
        self.failed_requests = 0
        self.latencies_ms = []

    def record(self, latency_ms: float, success: bool):
        self.total_requests += 1
        if not success:
            self.failed_requests += 1
        self.latencies_ms.append(latency_ms)

    @property
    def error_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests


class CanaryDeploymentRouter:
    """
    Progressive canary deployment router supporting weighted splits,
    session stickiness, and automated health-breach aborts.
    """
    def __init__(
        self,
        baseline_fn: Callable[[Dict[str, Any]], Any],
        canary_fn: Callable[[Dict[str, Any]], Any],
        initial_canary_weight: float = 0.10,
        max_error_threshold: float = 0.05
    ):
        self.baseline_fn = baseline_fn
        self.canary_fn = canary_fn
        self.canary_weight = initial_canary_weight  # Range: [0.0, 1.0]
        self.max_error_threshold = max_error_threshold

        self.baseline_metrics = ModelEndpointMetrics("baseline_v1")
        self.canary_metrics = ModelEndpointMetrics("canary_v2")
        self.is_aborted = False

    def set_canary_weight(self, weight: float):
        """Adjusts the canary traffic allocation percentage."""
        if not (0.0 <= weight <= 1.0):
            raise ValueError("Weight must be between 0.0 and 1.0.")
        self.canary_weight = weight
        print(f" [CANARY WEIGHT ADJUSTED] Canary now receives {int(weight * 100)}% of traffic.")

    def _should_route_to_canary(self, entity_id: Optional[str] = None) -> bool:
        """Determines routing destination using deterministic hashing or uniform random sampling."""
        if self.is_aborted or self.canary_weight <= 0.0:
            return False

        if entity_id:
            # Deterministic sticky session routing via hash
            digest = hashlib.sha256(entity_id.encode("utf-8")).hexdigest()
            bucket = int(digest[:6], 16) % 100
            return (bucket / 100.0) < self.canary_weight

        # Uniform stochastic routing
        return random.random() < self.canary_weight

    def _check_health_and_guardrail(self):
        """Automated watchdog evaluating canary health against SLA thresholds."""
        if self.canary_metrics.total_requests >= 5:
            if self.canary_metrics.error_rate > self.max_error_threshold:
                self.is_aborted = True
                self.canary_weight = 0.0
                print(
                    f"\n [CANARY SLA BREACH] Error rate ({self.canary_metrics.error_rate:.1%}) "
                    f"exceeded threshold ({self.max_error_threshold:.1%})! "
                    f"Tripping auto-abort -> 100% traffic rerouted to baseline."
                )

    def route_request(self, payload: Dict[str, Any], entity_id: Optional[str] = None) -> Dict[str, Any]:
        """Routes a single inference request to either baseline or canary with SLA monitoring."""
        use_canary = self._should_route_to_canary(entity_id)
        target_name = "canary" if use_canary else "baseline"
        target_fn = self.canary_fn if use_canary else self.baseline_fn
        metrics = self.canary_metrics if use_canary else self.baseline_metrics

        t0 = time.time()
        success = True
        result = None

        try:
            result = target_fn(payload)
        except Exception as exc:
            success = False
            result = {"error": str(exc)}
        finally:
            elapsed_ms = (time.time() - t0) * 1000.0
            metrics.record(elapsed_ms, success)
            if use_canary:
                self._check_health_and_guardrail()

        return {
            "routed_target": target_name,
            "status": "success" if success else "failed",
            "latency_ms": round(elapsed_ms, 2),
            "response": result
        }


if __name__ == "__main__":
    print("--- MLOps Safe Delivery: Dynamic Canary Traffic Router ---\n")

    # 1. Mock baseline and candidate model endpoints
    def baseline_model_v1(payload: dict) -> dict:
        time.sleep(0.01)
        return {"prediction": "approved", "model_version": "v1.0.0"}

    canary_should_fail = False

    def candidate_model_v2(payload: dict) -> dict:
        time.sleep(0.01)
        if canary_should_fail and random.random() < 0.4:
            raise RuntimeError("500 Internal Error: Canary feature transformation failure")
        return {"prediction": "approved_optimized", "model_version": "v2.0.0-rc1"}

    router = CanaryDeploymentRouter(
        baseline_fn=baseline_model_v1,
        canary_fn=candidate_model_v2,
        initial_canary_weight=0.20,  # Start at 20% canary traffic
        max_error_threshold=0.15
    )

    # 2. Phase 1: Healthy Canary Evaluation (20% Split)
    print("Phase 1: Simulating 20 requests with healthy canary (20% traffic):")
    for i in range(20):
        user = f"user_{i}"
        resp = router.route_request({"feature_val": 42}, entity_id=user)
        if resp["routed_target"] == "canary":
            print(f"  [CANARY HIT] {user} -> {resp['response']['model_version']}")

    print(
        f"\n  Baseline requests: {router.baseline_metrics.total_requests} | "
        f"Canary requests: {router.canary_metrics.total_requests}"
    )

    # 3. Phase 2: Progressive Promotion to 50%
    print("\nPhase 2: Promoting Canary Weight to 50%:")
    router.set_canary_weight(0.50)

    # 4. Phase 3: Canary Regression / Outage Occurs
    print("\nPhase 3: Canary begins failing -> Watchdog trips automatic rollback:")
    canary_should_fail = True

    for i in range(25, 45):
        user = f"user_{i}"
        resp = router.route_request({"feature_val": 99}, entity_id=user)
        if router.is_aborted:
            print(f"Fast-forward: {user} automatically routed to safe baseline.")
            break

    print(f"\nFinal Canary Status: Aborted = {router.is_aborted} | Effective Weight = {router.canary_weight}")