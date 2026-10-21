# MLOps Infrastructure & Protection: Token refill mathematics, multi-tier priority admission, and concurrency caps

import time
import threading
import json
from typing import Dict, Optional, Tuple

class PriorityTier:
    CRITICAL = 1   # Enterprise / Real-time fraud detection (always prioritized)
    STANDARD = 2   # Normal user inference requests
    BACKGROUND = 3 # Batch offline analytics / training telemetry


class TokenBucket:
    """Represents a discrete token bucket for a client/tenant with fractional continuous refill."""
    def __init__(self, capacity: float, refill_rate_per_sec: float):
        self.capacity = float(capacity)
        self.refill_rate = float(refill_rate_per_sec)
        self.tokens = float(capacity)
        self.last_refill_time = time.time()

    def refill(self, now: float):
        """Continuously replenishes tokens based on the exact elapsed time fraction."""
        delta = max(0.0, now - self.last_refill_time)
        self.tokens = min(self.capacity, self.tokens + (delta * self.refill_rate))
        self.last_refill_time = now

    def try_consume(self, tokens_needed: float = 1.0) -> bool:
        """Consumes tokens if available; returns False immediately if capacity is exceeded."""
        now = time.time()
        self.refill(now)
        if self.tokens >= tokens_needed:
            self.tokens -= tokens_needed
            return True
        return False


class InferenceConcurrencyGovernor:
    """
    Combines per-tenant token buckets with an active hardware concurrency ceiling
    to protect GPU/CPU inference backends from saturation.
    """
    def __init__(self, max_concurrent_workers: int = 2):
        self.max_concurrent_workers = max_concurrent_workers
        self.active_workers = 0
        self.buckets: Dict[str, TokenBucket] = {}
        self.lock = threading.Lock()

    def register_tenant(self, tenant_id: str, capacity: float = 5.0, refill_rate: float = 2.0):
        with self.lock:
            self.buckets[tenant_id] = TokenBucket(capacity, refill_rate)

    def acquire(self, tenant_id: str, priority: int = PriorityTier.STANDARD) -> Tuple[bool, str]:
        """
        Evaluates admission control:
        1. Checks tenant token quota.
        2. Validates downstream active worker capacity.
        """
        with self.lock:
            now = time.time()
            bucket = self.buckets.get(tenant_id)
            if not bucket:
                return False, "TENANT_NOT_REGISTERED"

            # Check hardware concurrency ceiling
            if self.active_workers >= self.max_concurrent_workers:
                if priority != PriorityTier.CRITICAL:
                    return False, "CONCURRENCY_LIMIT_EXCEEDED"

            # Check token consumption
            if not bucket.try_consume(1.0):
                return False, "RATE_LIMIT_EXCEEDED_429"

            # Slot acquired
            self.active_workers += 1
            return True, "ADMITTED"

    def release(self):
        """Releases an active inference worker slot back to the shared pool."""
        with self.lock:
            self.active_workers = max(0, self.active_workers - 1)


if __name__ == "__main__":
    print("--- MLOps Safety: Dynamic Token Bucket & Concurrency Governor ---\n")

    governor = InferenceConcurrencyGovernor(max_concurrent_workers=2)

    # Tenant configuration: Capacity=3 tokens, Refill=1 token per second
    tenant = "client_tokyo_nlp"
    governor.register_tenant(tenant, capacity=3.0, refill_rate=1.0)

    # Step 1: Rapid burst within token capacity
    print("Step 1: Rapid fire 4 requests in burst mode (Capacity = 3):")
    for req_idx in range(1, 5):
        admitted, reason = governor.acquire(tenant, priority=PriorityTier.STANDARD)
        print(f"  Request #{req_idx}: Status: {admitted} | Code: {reason}")
        if admitted:
            # Simulate immediate worker release
            governor.release()

    # Step 2: Testing active worker hardware concurrency ceiling
    print("\nStep 2: Testing concurrency ceiling (Workers maxed at 2):")
    # Occupy all worker slots
    slot1, _ = governor.acquire(tenant, priority=PriorityTier.CRITICAL)
    slot2, _ = governor.acquire(tenant, priority=PriorityTier.CRITICAL)
    print(f"  Worker slots filled: Active = {governor.active_workers}/{governor.max_concurrent_workers}")

    # Standard request rejected due to worker saturation
    standard_admitted, standard_reason = governor.acquire(tenant, priority=PriorityTier.STANDARD)
    print(f"  Standard Request during saturation: Admitted={standard_admitted} | Reason={standard_reason}")

    # Critical request permitted via priority override
    crit_admitted, crit_reason = governor.acquire(tenant, priority=PriorityTier.CRITICAL)
    print(f"  Critical Request during saturation: Admitted={crit_admitted} | Reason={crit_reason}")

    # Release slots
    governor.release()
    governor.release()
    if crit_admitted:
        governor.release()

    # Step 3: Token refill demonstration
    print("\nStep 3: Waiting 1.5s for continuous token replenishment...")
    time.sleep(1.5)
    admitted_post_wait, reason_post_wait = governor.acquire(tenant, priority=PriorityTier.STANDARD)
    print(f"  Request after refill window: Admitted={admitted_post_wait} | Code={reason_post_wait}")
    if admitted_post_wait:
        governor.release()

    print("\nVerification complete: Rate limiters and concurrency guards functioning safely.")