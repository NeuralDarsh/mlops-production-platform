# MLOps LLM Serving & Capacity Governance: Continuous token replenishment, pre-flight reservation, and EOS token refunds

import json
import threading
import time
from typing import Any, Dict, Optional, Tuple


class TokenBucketState:
    """Tracks continuous token balance and refill timestamp for a single LLM tenant."""
    def __init__(self, capacity: float, refill_rate_per_sec: float, now: float):
        self.capacity = float(capacity)
        self.refill_rate = float(refill_rate_per_sec)
        self.available_tokens = float(capacity)
        self.last_refill_ts = now


class LLMTokenBucketLimiter:
    """
    Thread-safe Token-Bucket rate limiter governing LLM inference workloads
    by prompt + completion token budgets rather than raw HTTP request counts.
    """
    def __init__(self, default_capacity: int = 1000, default_refill_per_sec: float = 200.0):
        if default_capacity <= 0 or default_refill_per_sec <= 0:
            raise ValueError("Capacity and refill rate must be strictly positive.")
        self.default_capacity = float(default_capacity)
        self.default_refill_per_sec = float(default_refill_per_sec)
        self._buckets: Dict[str, TokenBucketState] = {}
        self._lock = threading.Lock()

    def register_tenant(self, tenant_id: str, max_burst_tokens: int, tokens_per_sec: float, now: Optional[float] = None):
        """Configures a dedicated token bucket for a tenant tier."""
        if max_burst_tokens <= 0 or tokens_per_sec <= 0:
            raise ValueError("max_burst_tokens and tokens_per_sec must be strictly positive.")
        ts = now if now is not None else time.time()
        with self._lock:
            self._buckets[tenant_id] = TokenBucketState(max_burst_tokens, tokens_per_sec, ts)

    def _refill(self, bucket: TokenBucketState, now: float):
        """Replenishes tokens proportionally to elapsed time up to max capacity."""
        elapsed = max(0.0, now - bucket.last_refill_ts)
        if elapsed > 0.0:
            added = elapsed * bucket.refill_rate
            bucket.available_tokens = min(bucket.capacity, bucket.available_tokens + added)
            bucket.last_refill_ts = now

    def reserve_tokens(
        self,
        tenant_id: str,
        prompt_tokens: int,
        max_completion_tokens: int,
        now: Optional[float] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Pre-flight admission check: reserves (prompt_tokens + max_completion_tokens)
        before dispatching to GPU workers.
        """
        total_requested = prompt_tokens + max_completion_tokens
        if total_requested <= 0:
            raise ValueError("Requested token budget must be positive.")

        ts = now if now is not None else time.time()

        with self._lock:
            bucket = self._buckets.get(tenant_id)
            if bucket is None:
                bucket = TokenBucketState(self.default_capacity, self.default_refill_per_sec, ts)
                self._buckets[tenant_id] = bucket

            self._refill(bucket, ts)
            if total_requested > bucket.capacity:
                return False, {
                    "HTTP_Status": 413,
                    "error": "Payload Too Large: Requested tokens exceed tenant max burst capacity.",
                    "requested_tokens": total_requested,
                    "max_burst_capacity": int(bucket.capacity)
                }

            if bucket.available_tokens < total_requested:
                deficit = total_requested - bucket.available_tokens
                retry_after_sec = deficit / bucket.refill_rate
                return False, {
                    "HTTP_Status": 429,
                    "error": "Token Quota Exceeded",
                    "requested_tokens": total_requested,
                    "available_tokens": round(bucket.available_tokens, 1),
                    "retry_after_sec": round(retry_after_sec, 3)
                }

            bucket.available_tokens -= total_requested
            return True, {
                "HTTP_Status": 200,
                "reserved_tokens": total_requested,
                "remaining_tokens": round(bucket.available_tokens, 1)
            }

    def reconcile_actual_usage(
        self,
        tenant_id: str,
        reserved_tokens: int,
        actual_total_tokens: int,
        now: Optional[float] = None
    ) -> float:
        """
        Refunds unused completion tokens back to the bucket when generation stops early.
        Returns updated available token balance.
        """
        ts = now if now is not None else time.time()
        refund = max(0, reserved_tokens - actual_total_tokens)

        with self._lock:
            bucket = self._buckets.get(tenant_id)
            if bucket is None:
                return 0.0
            self._refill(bucket, ts)
            bucket.available_tokens = min(bucket.capacity, bucket.available_tokens + refund)
            return round(bucket.available_tokens, 1)


if __name__ == "__main__":
    print("--- MLOps LLM Governance: Token-Bucket Prompt Quota Limiter ---\n")

    t0 = 2000.0
    limiter = LLMTokenBucketLimiter()
    limiter.register_tenant("rag_service_prod", max_burst_tokens=1000, tokens_per_sec=200.0, now=t0)

    print("Step 1: Pre-flight reservation for heavy RAG prompt (800 worst-case tokens):")
    ok1, meta1 = limiter.reserve_tokens("rag_service_prod", prompt_tokens=500, max_completion_tokens=300, now=t0)
    print(f"Admitted: {ok1} | Details: {json.dumps(meta1)}")
    print("\nStep 2: Generation finished early (580 actual tokens used) -> Refunding 220 unused tokens:")
    updated_bal = limiter.reconcile_actual_usage(
        "rag_service_prod", reserved_tokens=800, actual_total_tokens=580, now=t0 + 0.1
    )
    print(f"Updated Balance after refund + 0.1s refill: {updated_bal} tokens")
    print("\nStep 3: Attempting 600-token request immediately (Exceeds 440 available):")
    ok2, meta2 = limiter.reserve_tokens("rag_service_prod", prompt_tokens=400, max_completion_tokens=200, now=t0 + 0.1)
    print(f"Admitted: {ok2} | Details: {json.dumps(meta2)}")
    wait_sec = meta2["retry_after_sec"]
    print(f"\nStep 4: Waiting {wait_sec}s per Retry-After header and retrying 600-token request:")
    ok3, meta3 = limiter.reserve_tokens(
        "rag_service_prod", prompt_tokens=400, max_completion_tokens=200, now=t0 + 0.1 + wait_sec
    )
    print(f"Admitted: {ok3} | Details: {json.dumps(meta3)}")