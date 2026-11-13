# Distributed Systems & Storage: Tunable R+W>N quorum consensus, version reconciliation, and asynchronous read-repair

import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Tuple

class VersionedRecord:
    """Represents a stored feature payload paired with a monotonic version stamp."""
    def __init__(self, value: Any, version: int, updated_at: float):
        self.value = value
        self.version = version
        self.updated_at = updated_at


class ReplicaStorageNode:
    """Simulates a physical storage replica in a distributed feature store cluster."""
    def __init__(self, node_id: str, latency_ms: float = 10.0, is_online: bool = True):
        self.node_id = node_id
        self.latency_ms = latency_ms
        self.is_online = is_online
        self.store: Dict[str, VersionedRecord] = {}

    async def write(self, key: str, record: VersionedRecord) -> bool:
        await asyncio.sleep(self.latency_ms / 1000.0)
        if not self.is_online:
            raise ConnectionError(f"Replica '{self.node_id}' unreachable during write.")

        existing = self.store.get(key)
        if existing is None or record.version > existing.version:
            self.store[key] = record
        return True

    async def read(self, key: str) -> Optional[VersionedRecord]:
        await asyncio.sleep(self.latency_ms / 1000.0)
        if not self.is_online:
            raise ConnectionError(f"Replica '{self.node_id}' unreachable during read.")
        return self.store.get(key)


class QuorumReplicationCoordinator:
    """
    Coordinates reads and writes across N replicas enforcing R + W > N quorum
    guarantees and triggering background read-repair on stale replicas.
    """
    def __init__(self, replicas: List[ReplicaStorageNode], read_quorum: int, write_quorum: int):
        self.replicas = replicas
        self.n = len(replicas)
        self.r = read_quorum
        self.w = write_quorum

        if self.r + self.w <= self.n:
            raise ValueError(
                f"Weak consistency configuration: R ({self.r}) + W ({self.w}) must exceed N ({self.n})."
            )
        self._version_counter = 0

    async def quorum_write(self, key: str, value: Any) -> Tuple[bool, int]:
        """
        Broadcasts write to all N replicas and succeeds once at least W acknowledge.
        Returns: (quorum_met: bool, assigned_version: int)
        """
        self._version_counter += 1
        version = self._version_counter
        record = VersionedRecord(value=value, version=version, updated_at=time.time())

        tasks = [asyncio.create_task(node.write(key, record)) for node in self.replicas]
        acks = 0

        for completed_task in asyncio.as_completed(tasks):
            try:
                if await completed_task:
                    acks += 1
                    if acks >= self.w:
                        print(f"[WRITE QUORUM MET] Key '{key}' v{version} committed ({acks}/{self.n} ACKs, W={self.w}).")
                        return True, version
            except Exception as exc:
                print(f"[REPLICA WRITE ERROR] {exc}")

        print(f"[WRITE QUORUM FAILED] Only {acks}/{self.w} required ACKs received for '{key}'.")
        return False, version

    async def _perform_read_repair(self, key: str, latest_record: VersionedRecord, stale_nodes: List[ReplicaStorageNode]):
        """Asynchronously updates lagging replicas in the background."""
        for node in stale_nodes:
            try:
                await node.write(key, latest_record)
                print(f"[READ-REPAIR COMPLETE] Synchronized '{node.node_id}' to v{latest_record.version} for '{key}'.")
            except Exception as exc:
                print(f"[READ-REPAIR FAILED] Could not repair '{node.node_id}': {exc}")

    async def quorum_read(self, key: str) -> Tuple[bool, Optional[Any], int]:
        """
        Gathers responses from at least R replicas, reconciles highest version,
        and schedules read-repair for stale nodes.
        Returns: (quorum_met: bool, latest_value: Any, version: int)
        """
        node_tasks = {
            asyncio.create_task(node.read(key)): node for node in self.replicas
        }

        responses: List[Tuple[ReplicaStorageNode, Optional[VersionedRecord]]] = []

        for completed_task in asyncio.as_completed(node_tasks.keys()):
            for task_obj, node in node_tasks.items():
                if task_obj is completed_task:
                    try:
                        rec = await task_obj
                        responses.append((node, rec))
                    except Exception as exc:
                        print(f"[REPLICA READ ERROR] {exc}")
                    break

            if len(responses) >= self.r:
                break

        if len(responses) < self.r:
            print(f"[READ QUORUM FAILED] Only {len(responses)}/{self.r} replicas responded.")
            return False, None, 0

        # Reconcile highest version among quorum responses
        valid_records = [rec for _, rec in responses if rec is not None]
        if not valid_records:
            return True, None, 0

        latest_record = max(valid_records, key=lambda r: r.version)

        # Identify lagging replicas for background read-repair
        stale_nodes = [
            node for node, rec in responses
            if rec is None or rec.version < latest_record.version
        ]

        if stale_nodes:
            stale_names = [n.node_id for n in stale_nodes]
            print(f" [STALE REPLICA DETECTED] Triggering read-repair on {stale_names} -> v{latest_record.version}")
            asyncio.create_task(self._perform_read_repair(key, latest_record, stale_nodes))

        return True, latest_record.value, latest_record.version


if __name__ == "__main__":
    print("--- MLOps Distributed Storage: Quorum Replication & Read-Repair Coordinator ---\n")

    async def main():
        node_1 = ReplicaStorageNode("replica_tokyo_01", latency_ms=10.0)
        node_2 = ReplicaStorageNode("replica_osaka_02", latency_ms=15.0)
        node_3 = ReplicaStorageNode("replica_singapore_03", latency_ms=25.0)

        coordinator = QuorumReplicationCoordinator(
            replicas=[node_1, node_2, node_3],
            read_quorum=2,
            write_quorum=2
        )

        print("Step 1: Writing feature vector with Quorum W=2:")
        ok, ver = await coordinator.quorum_write("user_emb_771", {"risk_score": 0.14, "segment": "prime"})
        await asyncio.sleep(0.05) 

        print("\nStep 2: Simulating outage on 'replica_singapore_03' during update:")
        node_3.is_online = False
        ok_degraded, ver_2 = await coordinator.quorum_write("user_emb_771", {"risk_score": 0.89, "segment": "high_risk"})

        print("\nStep 3: 'replica_singapore_03' rejoins cluster with stale v1 data; reading with R=2:")
        node_3.is_online = True
        node_1.is_online = False

        read_ok, value, resolved_ver = await coordinator.quorum_read("user_emb_771")
        print(f"[CLIENT READ RESULT] Resolved v{resolved_ver}: {value}")

        await asyncio.sleep(0.05)

        print("\nFinal Replica Consistency Audit:")
        audit = {
            n.node_id: {
                "version": n.store["user_emb_771"].version if "user_emb_771" in n.store else None,
                "value": n.store["user_emb_771"].value if "user_emb_771" in n.store else None
            }
            for n in [node_1, node_2, node_3]
        }
        print(json.dumps(audit, indent=2))

    asyncio.run(main())