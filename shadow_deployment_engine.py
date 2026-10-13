# MLOps Serving & Risk Mitigation: Asynchronous request mirroring, zero-impact shadow evaluation, and drift divergence tracking

import asyncio
import time
import math
import json
from typing import Any, Callable, Dict, List, Optional

class DivergenceMetrics:
    """Tracks comparative performance and prediction alignment between primary and shadow models."""
    def __init__(self):
        self.total_shadowed_requests = 0
        self.classification_mismatches = 0
        self.absolute_score_deltas: List[float] = []
        self.shadow_failures = 0

    def record(self, primary_pred: str, shadow_pred: str, primary_score: float, shadow_score: float):
        self.total_shadowed_requests += 1
        delta = abs(primary_score - shadow_score)
        self.absolute_score_deltas.append(delta)
        if primary_pred != shadow_pred:
            self.classification_mismatches += 1

    def record_shadow_failure(self):
        self.total_shadowed_requests += 1
        self.shadow_failures += 1

    def get_summary(self) -> Dict[str, Any]:
        count = len(self.absolute_score_deltas)
        mean_delta = sum(self.absolute_score_deltas) / count if count > 0 else 0.0
        mismatch_rate = (self.classification_mismatches / self.total_shadowed_requests) if self.total_shadowed_requests > 0 else 0.0
        failure_rate = (self.shadow_failures / self.total_shadowed_requests) if self.total_shadowed_requests > 0 else 0.0

        return {
            "total_shadowed": self.total_shadowed_requests,
            "classification_mismatch_rate": round(mismatch_rate, 4),
            "mean_score_divergence": round(mean_delta, 4),
            "shadow_failure_rate": round(failure_rate, 4),
            "is_promotion_safe": mismatch_rate < 0.15 and failure_rate == 0.0
        }


class ShadowDeploymentEngine:
    """
    Mirrors live traffic asynchronously to a candidate model without affecting
    primary serving latencies or client SLAs.
    """
    def __init__(
        self,
        primary_model_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
        shadow_model_fn: Callable[[Dict[str, Any]], Dict[str, Any]]
    ):
        self.primary_model_fn = primary_model_fn
        self.shadow_model_fn = shadow_model_fn
        self.metrics = DivergenceMetrics()

    async def _evaluate_shadow_async(self, payload: Dict[str, Any], primary_result: Dict[str, Any]):
        """Runs candidate model in the background and records divergence without blocking client."""
        try:
            # Shadow invocation is completely decoupled
            shadow_result = await self.shadow_model_fn(payload)
            self.metrics.record(
                primary_pred=primary_result["label"],
                shadow_pred=shadow_result["label"],
                primary_score=primary_result["confidence"],
                shadow_score=shadow_result["confidence"]
            )
        except Exception as exc:
            # Strict error containment: shadow failure must never propagate to caller
            self.metrics.record_shadow_failure()

    async def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synchronously resolves the primary model to fulfill client SLA,
        while teeing off a non-blocking background shadow task.
        """
        # 1. Primary evaluation (Synchronous path for caller)
        primary_result = await self.primary_model_fn(payload)

        # 2. Tee / Mirror payload to shadow candidate in background
        asyncio.create_task(self._evaluate_shadow_async(payload, primary_result))

        # Return primary response immediately
        return primary_result


if __name__ == "__main__":
    print("--- MLOps Safety: Shadow Deployment (Dark Launch) Engine ---\n")

    async def main():
        # Baseline Production Model (e.g., Logistic Regression / LightGBM)
        async def primary_production_v1(payload: dict) -> dict:
            await asyncio.sleep(0.01)  # 10ms execution
            score = 0.5 + (sum(payload["features"]) * 0.1)
            score = max(0.0, min(1.0, score))
            label = "fraud" if score >= 0.7 else "legitimate"
            return {"label": label, "confidence": round(score, 3), "model": "v1.0.0-prod"}

        # Candidate Shadow Model (e.g., Deep Neural Network / Transformer)
        async def candidate_shadow_v2(payload: dict) -> dict:
            await asyncio.sleep(0.03)  # 30ms execution (simulating heavier model)
            if payload.get("trigger_bug"):
                raise RuntimeError("Unexpected tensor dimension error in shadow layer!")
            # Slightly more conservative scoring
            score = 0.48 + (sum(payload["features"]) * 0.12)
            score = max(0.0, min(1.0, score))
            label = "fraud" if score >= 0.7 else "legitimate"
            return {"label": label, "confidence": round(score, 3), "model": "v2.0.0-shadow"}

        engine = ShadowDeploymentEngine(
            primary_model_fn=primary_production_v1,
            shadow_model_fn=candidate_shadow_v2
        )

        # 1. Simulate stream of production requests
        print("Step 1: Ingesting live traffic stream through shadow engine:")
        test_payloads = [
            {"request_id": "req_001", "features": [0.5, 1.2, 0.4]},
            {"request_id": "req_002", "features": [1.8, 2.0, 0.9]},
            {"request_id": "req_003", "features": [0.1, -0.2, 0.3]},
            {"request_id": "req_004", "features": [1.4, 0.8, 0.2]},
            {"request_id": "req_005", "features": [0.2, 0.1, 0.5], "trigger_bug": True},  # Shadow failure test
        ]

        for p in test_payloads:
            t0 = time.time()
            client_response = await engine.predict(p)
            latency_ms = (time.time() - t0) * 1000.0
            print(f"  [CLIENT RESPONSE] Req: {p['request_id']} | Returned: {client_response['label']} ({client_response['confidence']}) in {round(latency_ms, 2)}ms")

        # Allow background shadow tasks to complete
        await asyncio.sleep(0.1)

        # 2. Review Shadow Auditing Telemetry
        print("\nStep 2: Evaluating Shadow Model Readiness Report:")
        summary = engine.metrics.get_summary()
        print(json.dumps(summary, indent=2))

        if summary["is_promotion_safe"]:
            print("\n[VERDICT] Candidate model exhibits acceptable variance. Approved for Canary promotion.")
        else:
            print("\n[VERDICT] Candidate model rejected or needs triage before live promotion.")

    asyncio.run(main())