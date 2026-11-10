# Distributed Systems & MLOps Infrastructure: Mutex leases, heartbeat renewals, and fencing-token split-brain guards

import asyncio
import time
import json
from typing import Any, Dict, Optional, Tuple


class LockLease:
    """Represents an active or expired distributed lease holding an exclusive resource."""
    def __init__(self, resource_id: str, holder_id: str, ttl_seconds: float, fencing_token: int):
        self.resource_id = resource_id
        self.holder_id = holder_id
        self.fencing_token = fencing_token
        self.granted_at = time.time()
        self.expires_at = self.granted_at + ttl_seconds

    def is_valid(self) -> bool:
        return time.time() < self.expires_at

    def renew(self, ttl_seconds: float):
        self.expires_at = time.time() + ttl_seconds


class DistributedLockManager:
    """
    Centralized coordinator managing distributed resource locks with monotonic
    fencing tokens and non-blocking lease renewals.
    """
    def __init__(self):
        self._locks: Dict[str, LockLease] = {}
        self._global_token_counter = 0
        self._lock_mutex = asyncio.Lock()

    async def acquire(
        self, resource_id: str, client_id: str, ttl_seconds: float = 1.0
    ) -> Tuple[bool, Optional[int]]:
        """
        Attempts to acquire an exclusive lease on a resource.
        Returns: (success: bool, fencing_token: Optional[int])
        """
        async with self._lock_mutex:
            current_lease = self._locks.get(resource_id)

            if current_lease and current_lease.is_valid():
                if current_lease.holder_id != client_id:
                    return False, None
                current_lease.renew(ttl_seconds)
                return True, current_lease.fencing_token
            self._global_token_counter += 1
            token = self._global_token_counter
            self._locks[resource_id] = LockLease(resource_id, client_id, ttl_seconds, token)
            print(f"[LEASE GRANTED] Resource '{resource_id}' -> Client '{client_id}' (Token #{token}, TTL: {ttl_seconds}s)")
            return True, token

    async def renew_heartbeat(self, resource_id: str, client_id: str, ttl_seconds: float = 1.0) -> bool:
        """Extends an existing valid lease to prevent premature expiration."""
        async with self._lock_mutex:
            lease = self._locks.get(resource_id)
            if lease and lease.is_valid() and lease.holder_id == client_id:
                lease.renew(ttl_seconds)
                return True
            return False

    async def release(self, resource_id: str, client_id: str, fencing_token: int) -> bool:
        """Releases the lease voluntarily if the fencing token matches."""
        async with self._lock_mutex:
            lease = self._locks.get(resource_id)
            if lease and lease.holder_id == client_id and lease.fencing_token == fencing_token:
                del self._locks[resource_id]
                print(f"[LEASE RELEASED] Resource '{resource_id}' released by '{client_id}'.")
                return True
            return False


class ProtectedStorageTarget:
    """
    Simulates a downstream storage tier (e.g., Model Weight Bucket / Feature Store)
    enforcing fencing token validations to prevent zombie process overwrites.
    """
    def __init__(self):
        self.last_committed_token = 0
        self.stored_data = {}

    def write_payload(self, client_id: str, fencing_token: int, payload: Dict[str, Any]) -> bool:
        """Only permits writes if fencing token is strictly greater than or equal to highest observed token."""
        if fencing_token < self.last_committed_token:
            print(
                f" [ZOMBIE WRITE REJECTED] Client '{client_id}' attempted write with stale Token #{fencing_token}. "
                f"Storage highest observed Token is #{self.last_committed_token}."
            )
            return False

        self.last_committed_token = fencing_token
        self.stored_data = payload
        print(f"[WRITE SUCCESS] Client '{client_id}' committed state with Token #{fencing_token}.")
        return True


if __name__ == "__main__":
    print("--- MLOps Distributed Systems: Distributed Lock Lease & Fencing Engine ---\n")

    async def main():
        lock_manager = DistributedLockManager()
        storage = ProtectedStorageTarget()
        resource = "model_weights_checkpoint_v4"

        print("Scenario 1: Worker 1 acquires lock and performs clean checkpoint write:")
        ok1, token1 = await lock_manager.acquire(resource, "worker_01", ttl_seconds=0.6)
        if ok1 and token1:
            await asyncio.sleep(0.3)
            renewed = await lock_manager.renew_heartbeat(resource, "worker_01", ttl_seconds=0.6)
            print(f"  Heartbeat renewed for worker_01: {renewed}")

            storage.write_payload("worker_01", token1, {"weights_version": "4.0.1", "val_loss": 0.18})
            await lock_manager.release(resource, "worker_01", token1)

        print("\nScenario 2: Worker 2 freezes; lease times out; Worker 3 takes over with new token:")
        ok2, token2 = await lock_manager.acquire(resource, "worker_02", ttl_seconds=0.4)
        print(f"  Worker 2 acquired lease (Token #{token2}). Simulating 0.8s freeze/pause...")

        await asyncio.sleep(0.6)

        ok3, token3 = await lock_manager.acquire(resource, "worker_03", ttl_seconds=0.8)
        print(f"  Worker 3 acquisition: {ok3} (New Fencing Token #{token3})")

        storage.write_payload("worker_03", token3, {"weights_version": "4.0.2", "val_loss": 0.14})

        print("\nScenario 3: Frozen Worker 2 unpauses and attempts zombie write:")
        zombie_write_success = storage.write_payload(
            "worker_02", token2, {"weights_version": "4.0.1_corrupted", "val_loss": 0.99}
        )
        print(f"  Zombie write execution result: {zombie_write_success}")

        await lock_manager.release(resource, "worker_03", token3)

        print("\nFinal Protected Storage State:")
        print(json.dumps({
            "highest_token": storage.last_committed_token,
            "data": storage.stored_data
        }, indent=2))

    asyncio.run(main())