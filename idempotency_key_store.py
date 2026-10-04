# MLOps & Infrastructure Reliability: Enforcing exact-once execution and caching API responses with TTL pruning

import time
import threading
import hashlib
import json

class IdempotencyState:
    IN_FLIGHT = "IN_FLIGHT"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class IdempotencyRecord:
    """Stores the execution state, cached response, and expiration timestamp for a transaction key."""
    def __init__(self, key, ttl_seconds):
        self.key = key
        self.state = IdempotencyState.IN_FLIGHT
        self.response = None
        self.expires_at = time.time() + ttl_seconds

    def is_expired(self):
        return time.time() > self.expires_at


class IdempotencyKeyStore:
    """
    Thread-safe in-memory idempotency store with sliding TTL expiration
    and atomic state transitions.
    """
    def __init__(self, default_ttl_seconds=5):
        self.default_ttl = default_ttl_seconds
        self.store = {}
        self.lock = threading.Lock()

    def _generate_fingerprint(self, payload):
        """Generates a deterministic SHA-256 hash for payloads without explicit client keys."""
        normalized = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

    def acquire_or_get(self, key=None, payload=None):
        """
        Attempts to acquire an execution lock for a key.
        Returns:
            (status: str, response: Any)
            status can be: 'ACQUIRED', 'ALREADY_COMPLETED', 'LOCKED_IN_FLIGHT'
        """
        idempotency_key = key or (self._generate_fingerprint(payload) if payload else None)
        if not idempotency_key:
            raise ValueError("Must provide either an idempotency key or a serializable payload.")

        with self.lock:
            record = self.store.get(idempotency_key)

            # Check if existing key has expired
            if record and record.is_expired():
                del self.store[idempotency_key]
                record = None

            if record is None:
                # Key is fresh: mark as IN_FLIGHT
                self.store[idempotency_key] = IdempotencyRecord(idempotency_key, self.default_ttl)
                return ("ACQUIRED", idempotency_key, None)

            if record.state == IdempotencyState.COMPLETED:
                return ("ALREADY_COMPLETED", idempotency_key, record.response)

            if record.state == IdempotencyState.IN_FLIGHT:
                return ("LOCKED_IN_FLIGHT", idempotency_key, None)

            # If previous attempt failed, allow fresh re-acquisition
            self.store[idempotency_key] = IdempotencyRecord(idempotency_key, self.default_ttl)
            return ("ACQUIRED", idempotency_key, None)

    def complete(self, key, response):
        """Marks the operation complete and caches the response."""
        with self.lock:
            if key in self.store:
                self.store[key].state = IdempotencyState.COMPLETED
                self.store[key].response = response

    def fail(self, key):
        """Marks the operation failed, clearing the lock for immediate retry."""
        with self.lock:
            if key in self.store:
                self.store[key].state = IdempotencyState.FAILED

    def purge_expired(self):
        """Active garbage collection loop to prune dead keys."""
        with self.lock:
            now = time.time()
            expired_keys = [k for k, v in self.store.items() if now > v.expires_at]
            for k in expired_keys:
                del self.store[k]
            return len(expired_keys)


if __name__ == "__main__":
    print("--- Systems Resilience: Idempotency Key & Deduplication Engine ---\n")

    store = IdempotencyKeyStore(default_ttl_seconds=2)

    def process_inference_pipeline(client_key, payload):
        status, key, cached_data = store.acquire_or_get(key=client_key, payload=payload)

        if status == "ALREADY_COMPLETED":
            print(f" [CACHE HIT - DEDUPLICATED] Key: '{key}' | Returning cached result: {cached_data}")
            return cached_data

        if status == "LOCKED_IN_FLIGHT":
            print(f"[CONCURRENT LOCK] Key: '{key}' is already executing on another worker. Fast-failing duplicate.")
            return None

        print(f"[FIRST RUN] Executing model inference for Key: '{key}'...")
        # Simulate heavy inference computation
        time.sleep(0.05)
        result = {"prediction": "fraud_detected", "score": 0.941, "features_evaluated": len(payload)}

        store.complete(key, result)
        print(f"[COMPLETED & SAVED] Key: '{key}' persisted to idempotency ledger.")
        return result

    # 1. Initial Request
    sample_payload = {"user_id": 4012, "transaction_amount": 950.0}
    test_key = "idemp_tokyo_tx_001"

    print("Step 1: Processing initial model evaluation:")
    process_inference_pipeline(test_key, sample_payload)

    # 2. Duplicate Request within TTL window (Simulated network retry)
    print("\nStep 2: Client retry arrives with identical key:")
    process_inference_pipeline(test_key, sample_payload)

    # 3. Automatic hash deduction when no key is explicitly passed
    print("\nStep 3: Auto-fingerprinting raw payload (No key explicitly supplied):")
    auto_payload = {"user_id": 9999, "action": "predict"}
    process_inference_pipeline(None, auto_payload)
    process_inference_pipeline(None, auto_payload)  # Duplicate call

    # 4. Wait for TTL expiration and re-execute
    print("\nStep 4: Sleeping 2.2 seconds to allow TTL expiration...")
    time.sleep(2.2)
    purged_count = store.purge_expired()
    print(f"Cleaned up {purged_count} expired keys via active GC sweep.")

    print("\nStep 5: Retrying original key post-expiration (Runs fresh):")
    process_inference_pipeline(test_key, sample_payload)