# MLOps Infrastructure & Network Resilience: Full jitter backoff, transient error classification, and DLQ routing

import asyncio
import random
import time
import json
from typing import Any, Callable, Dict, List, Optional, Tuple, Type


class TransientSystemError(Exception):
    """Raised for transient issues eligible for retry (e.g., 503, connection drops)."""
    pass


class NonTransientClientError(Exception):
    """Raised for deterministic, non-recoverable errors (e.g., bad schemas, 400s)."""
    pass


class RetryJitterEngine:
    """
    Asynchronous retry coordinator using exponential backoff with full jitter
    and selective exception classification.
    """
    def __init__(
        self,
        max_retries: int = 4,
        base_delay_sec: float = 0.05,
        max_delay_sec: float = 1.0,
        retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay_sec
        self.max_delay = max_delay_sec
        self.retryable_exceptions = retryable_exceptions or (TransientSystemError, TimeoutError, ConnectionError)
        self.dead_letter_queue: List[Dict[str, Any]] = []

    def _calculate_jitter_delay(self, attempt: int) -> float:
        """Calculates Full Jitter backoff delay: Uniform(0, Min(MaxDelay, BaseDelay * 2^attempt))."""
        exponential_cap = min(self.max_delay, self.base_delay * (2 ** attempt))
        return random.uniform(0.0, exponential_cap)

    async def execute_with_retry(
        self,
        task_fn: Callable[..., Any],
        payload: Dict[str, Any],
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Executes task_fn with exponential backoff and jitter for transient errors.
        Routes to DLQ if max retries are exceeded or non-transient error occurs.
        """
        attempt = 0
        execution_trace = []

        while attempt <= self.max_retries:
            t0 = time.time()
            try:
                result = await task_fn(payload, *args, **kwargs)
                return {
                    "status": "SUCCESS",
                    "attempts_taken": attempt + 1,
                    "result": result,
                    "trace": execution_trace
                }

            except self.retryable_exceptions as exc:
                elapsed = round((time.time() - t0) * 1000.0, 2)
                attempt += 1
                if attempt > self.max_retries:
                    dlq_record = {
                        "payload": payload,
                        "reason": f"Max retries exhausted ({self.max_retries})",
                        "final_error": str(exc),
                        "total_attempts": attempt
                    }
                    self.dead_letter_queue.append(dlq_record)
                    print(f"[RETRIES EXHAUSTED] Diverting payload to DLQ. Error: {exc}")
                    return {"status": "DLQ_ESCALATED", "attempts_taken": attempt, "error": str(exc)}

                sleep_duration = self._calculate_jitter_delay(attempt)
                execution_trace.append({
                    "attempt": attempt,
                    "error": str(exc),
                    "backoff_sleep_sec": round(sleep_duration, 4)
                })
                print(
                    f"[TRANSIENT ERROR] Attempt {attempt} failed ({exc}). "
                    f"Sleeping {round(sleep_duration * 1000, 1)}ms (Full Jitter)..."
                )
                await asyncio.sleep(sleep_duration)

            except Exception as non_transient_exc:
                dlq_record = {
                    "payload": payload,
                    "reason": "Non-transient error encountered",
                    "final_error": str(non_transient_exc),
                    "total_attempts": attempt + 1
                }
                self.dead_letter_queue.append(dlq_record)
                print(f"[FAST-FAIL] Non-transient error: '{non_transient_exc}'. No retries performed.")
                return {
                    "status": "NON_TRANSIENT_FAILURE",
                    "attempts_taken": attempt + 1,
                    "error": str(non_transient_exc)
                }


if __name__ == "__main__":
    print("--- MLOps Network Resilience: Exponential Backoff & Full Jitter Engine ---\n")

    async def main():
        engine = RetryJitterEngine(max_retries=3, base_delay_sec=0.04, max_delay_sec=0.5)

        call_counter = 0

        async def unstable_vector_db_fetch(payload: dict) -> dict:
            nonlocal call_counter
            call_counter += 1
            if call_counter < 3:
                raise TransientSystemError(f"503 Service Unavailable: Remote shard warming (Attempt #{call_counter})")
            return {"embedding_vector": [0.12, 0.44, 0.98], "dim": 3}

        print("Scenario 1: Executing query against transiently failing service:")
        res1 = await engine.execute_with_retry(unstable_vector_db_fetch, {"entity_id": "tokyo_vec_01"})
        print(f"  Final Status: {res1['status']} in {res1['attempts_taken']} attempts.\n")

        async def malformed_query(payload: dict) -> dict:
            raise NonTransientClientError("400 Bad Request: Missing required feature tensor dimension")

        print("Scenario 2: Executing non-transient corrupt query:")
        res2 = await engine.execute_with_retry(malformed_query, {"invalid_key": True})
        print(f"  Final Status: {res2['status']} in {res2['attempts_taken']} attempt.\n")

        async def down_service(payload: dict) -> dict:
            raise TransientSystemError("504 Gateway Timeout: Database host unresponsive")

        print("Scenario 3: Complete service outage (Exhausts retries):")
        res3 = await engine.execute_with_retry(down_service, {"entity_id": "tokyo_vec_99"})
        print(f"  Final Status: {res3['status']}\n")

        print("Final Dead-Letter Queue (DLQ) Dump:")
        print(json.dumps(engine.dead_letter_queue, indent=2))

    asyncio.run(main())