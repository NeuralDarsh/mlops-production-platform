# MLOps API Gateway & Traffic Governance: Weighted rolling estimation, O(1) window rotation, and 429 headers

import json
import math
import threading
import time
from typing import Any, Dict, Optional, Tuple

class TenantWindowState:
    def __init__(self, window_size_sec: float, current_window_start: float):
        self.window_size_sec = window_size_sec
        self.current_window_start = current_window_start
        self.previous_count = 0
        self.current_count = 0

class SlidingWindowRateLimiter:
    def __init__(self, default_limit: int = 10, window_size_sec: float = 1.0):
        if default_limit <= 0 or window_size_sec <= 0:
            raise ValueError("default_limit and window_size_sec must be strictly positive.")
        self.default_limit = default_limit
        self.window_size_sec = window_size_sec   
        self._tenant_limits: Dict[str, int] = {}
        self._states: Dict[str, TenantWindowState] = {}
        self._lock = threading.Lock()

    def configure_tenant_quota(self, tenant_id: str, max_requests: int):
        if max_requests <= 0:
            raise ValueError("max_requests must be strictly positive.")
        with self._lock:
            self._tenant_limits[tenant_id] = max_requests

    def _align_window(self, state: TenantWindowState, now: float):
        elapsed_since_start = now - state.current_window_start
        windows_passed = int(elapsed_since_start // state.window_size_sec)

        if windows_passed == 1:
            state.previous_count = state.current_count
            state.current_count = 0
            state.current_window_start += state.window_size_sec
        elif windows_passed >= 2:
            state.previous_count = 0
            state.current_count = 0
            state.current_window_start += windows_passed * state.window_size_sec

    def evaluate_request(self, tenant_id: str, now: Optional[float] = None) -> Tuple[bool, Dict[str, Any]]:
        ts = now if now is not None else time.time()

        with self._lock:
            limit = self._tenant_limits.get(tenant_id, self.default_limit)
            state = self._states.get(tenant_id)

            if state is None:
                aligned_start = math.floor(ts / self.window_size_sec) * self.window_size_sec
                state = TenantWindowState(self.window_size_sec, aligned_start)
                self._states[tenant_id] = state

            self._align_window(state, ts)
            elapsed_in_current = max(0.0, ts - state.current_window_start)
            current_fraction = min(1.0, elapsed_in_current / state.window_size_sec)
            previous_weight = 1.0 - current_fraction
            estimated_rolling_count = (state.previous_count * previous_weight) + state.current_count

            if estimated_rolling_count + 1.0 > limit:
                if state.previous_count > 0 and state.current_count < limit:
                    needed_drop = (estimated_rolling_count + 1.0) - limit
                    retry_after = (needed_drop / state.previous_count) * state.window_size_sec
                else:
                    retry_after = state.window_size_sec - elapsed_in_current

                headers = {
                    "HTTP_Status": 429,
                    "X-RateLimit-Limit": limit,
                    "X-RateLimit-Remaining": 0,
                    "Retry-After-Sec": round(max(0.01, retry_after), 3),
                    "Estimated-Rolling-Load": round(estimated_rolling_count, 2)
                }
                return False, headers
            state.current_count += 1
            updated_estimate = (state.previous_count * previous_weight) + state.current_count
            remaining = max(0, int(math.floor(limit - updated_estimate)))

            headers = {
                "HTTP_Status": 200,
                "X-RateLimit-Limit": limit,
                "X-RateLimit-Remaining": remaining,
                "Estimated-Rolling-Load": round(updated_estimate, 2)
            }
            return True, headers
if __name__ == "__main__":
    print("--- MLOps API Gateway: Sliding-Window Counter Rate Limiter ---\n")

    limiter = SlidingWindowRateLimiter(default_limit=10, window_size_sec=1.0)
    limiter.configure_tenant_quota("tenant_enterprise_tokyo", max_requests=10)
    limiter.configure_tenant_quota("tenant_free_tier", max_requests=3)

    print("Step 1: Testing window-boundary burst protection (Limit = 10 req / 1.0s):")
    t0 = 1000.0  
    admitted_w0 = 0
    for _ in range(8):
        ok, _ = limiter.evaluate_request("tenant_enterprise_tokyo", now=t0 + 0.8)
        if ok:
            admitted_w0 += 1
    print(f"  Window 0 (t=1000.8s): Admitted {admitted_w0}/8 requests near boundary.")
    print("\nStep 2: Bursting 6 requests immediately across boundary at t=1001.1s:")
    for req_idx in range(1, 7):
        allowed, meta = limiter.evaluate_request("tenant_enterprise_tokyo", now=t0 + 1.1)
        status_icon = "200 OK" if allowed else "429 BLOCKED"
        print(
            f"  Req #{req_idx} -> {status_icon} | Rolling Load: {meta['Estimated-Rolling-Load']} "
            f"| Remaining: {meta['X-RateLimit-Remaining']}"
            + (f" | Retry-After: {meta['Retry-After-Sec']}s" if not allowed else "")
        )
    print("\nStep 3: Verifying multi-tenant isolation ('tenant_free_tier' limit = 3):")
    free_results = []
    for i in range(4):
        allowed, meta = limiter.evaluate_request("tenant_free_tier", now=t0 + 2.0)
        free_results.append((i + 1, allowed, meta["HTTP_Status"]))
    print(f"Free-tier admission trace: {json.dumps(free_results)}")