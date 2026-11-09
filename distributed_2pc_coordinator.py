# Distributed Systems & Feature Stores: Atomic multi-node mutations, prepare-commit consensus, and rollback hooks

import json
import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class Vote(Enum):
    VOTE_COMMIT = "VOTE_COMMIT"
    VOTE_ABORT = "VOTE_ABORT"


class TransactionState(Enum):
    INITIATED = "INITIATED"
    PREPARING = "PREPARING"
    COMMITTED = "COMMITTED"
    ABORTED = "ABORTED"


class FeatureStoreParticipant:
    """
    Represents an independent distributed storage node participating in 2PC transactions
    (e.g., In-Memory Cache Node, Vector Database Node, or Relational Metadata DB).
    """
    def __init__(self, node_id: str, should_fail_prepare: bool = False):
        self.node_id = node_id
        self.should_fail_prepare = should_fail_prepare
        self.committed_storage: Dict[str, Any] = {}
        self.staged_transactions: Dict[str, Dict[str, Any]] = {}

    def prepare(self, tx_id: str, key: str, value: Any) -> Vote:
        """Phase 1: Validates feasibility, checks locks, and stages data in a scratchpad."""
        if self.should_fail_prepare:
            print(f"[{self.node_id}] Prepare rejected: Simulating capacity or lock contention.")
            return Vote.VOTE_ABORT
        self.staged_transactions[tx_id] = {"key": key, "value": value}
        print(f"[{self.node_id}] Staged tx '{tx_id}'. Voted: VOTE_COMMIT.")
        return Vote.VOTE_COMMIT

    def commit(self, tx_id: str):
        """Phase 2 (Success): Finalizes staged changes into active committed storage."""
        if tx_id in self.staged_transactions:
            entry = self.staged_transactions.pop(tx_id)
            self.committed_storage[entry["key"]] = entry["value"]
            print(f"[{self.node_id}] Committed tx '{tx_id}' (Key: '{entry['key']}').")

    def abort(self, tx_id: str):
        """Phase 2 (Failure): Discards staged scratchpad mutations with zero state change."""
        if tx_id in self.staged_transactions:
            self.staged_transactions.pop(tx_id)
            print(f"[{self.node_id}] Aborted and rolled back tx '{tx_id}'.")


class TwoPhaseCommitCoordinator:
    """
    Coordinates distributed atomic writes across multiple storage participants
    using the Two-Phase Commit (2PC) protocol.
    """
    def __init__(self, participants: List[FeatureStoreParticipant]):
        self.participants = participants
        self.transaction_logs: List[Dict[str, Any]] = []

    def execute_transaction(self, key: str, value: Any) -> Tuple[bool, str]:
        """
        Executes a distributed atomic write:
        Phase 1: Prepare (Collect votes)
        Phase 2: Commit if all vote yes; Abort if any participant rejects.
        """
        tx_id = f"tx_{uuid.uuid4().hex[:8]}"
        t0 = time.time()
        print(f"\n[TX BEGIN] Starting 2PC Transaction '{tx_id}' for Key: '{key}'...")

        votes: Dict[str, Vote] = {}
        for node in self.participants:
            vote = node.prepare(tx_id, key, value)
            votes[node.node_id] = vote

        all_agreed = all(v == Vote.VOTE_COMMIT for v in votes.values())

        if all_agreed:
            final_state = TransactionState.COMMITTED
            print(f"[CONSENSUS REACHED] All nodes voted COMMIT. Broadcasting GLOBAL_COMMIT...")
            for node in self.participants:
                node.commit(tx_id)
            success = True
        else:
            final_state = TransactionState.ABORTED
            print(f"[CONSENSUS FAILED] One or more nodes aborted. Broadcasting GLOBAL_ABORT...")
            for node in self.participants:
                node.abort(tx_id)
            success = False

        duration_ms = round((time.time() - t0) * 1000.0, 2)
        log_entry = {
            "tx_id": tx_id,
            "key": key,
            "state": final_state.value,
            "votes": {nid: v.value for nid, v in votes.items()},
            "duration_ms": duration_ms
        }
        self.transaction_logs.append(log_entry)
        return success, tx_id


if __name__ == "__main__":
    print("--- MLOps Distributed Storage: Two-Phase Commit (2PC) Coordinator ---\n")

    print("Scenario 1: All nodes healthy (Cache, Vector DB, Metadata Store):")
    cache_node = FeatureStoreParticipant("L1_Cache_Redis")
    vector_node = FeatureStoreParticipant("Vector_DB_Qdrant")
    meta_node = FeatureStoreParticipant("Metadata_Store_Postgres")

    coordinator_healthy = TwoPhaseCommitCoordinator([cache_node, vector_node, meta_node])

    success_tx, tx_id_1 = coordinator_healthy.execute_transaction(
        key="user_vector_9981",
        value={"embedding": [0.12, 0.44, 0.98], "version": 1}
    )
    print(f"  Transaction '{tx_id_1}' Result: {'SUCCESSFUL COMMIT' if success_tx else 'FAILED'}")

    print(f"  Cache Node State: {cache_node.committed_storage.get('user_vector_9981') is not None}")
    print(f"  Vector Node State: {vector_node.committed_storage.get('user_vector_9981') is not None}")
    print(f"  Metadata Node State: {meta_node.committed_storage.get('user_vector_9981') is not None}")

    print("\nScenario 2: Downstream node rejection (Simulating Vector DB failure):")
    failing_vector_node = FeatureStoreParticipant("Vector_DB_Qdrant", should_fail_prepare=True)
    coordinator_with_failure = TwoPhaseCommitCoordinator([cache_node, failing_vector_node, meta_node])

    fail_tx, tx_id_2 = coordinator_with_failure.execute_transaction(
        key="user_vector_9982",
        value={"embedding": [0.81, 0.22, 0.55], "version": 2}
    )
    print(f"  Transaction '{tx_id_2}' Result: {'SUCCESSFUL COMMIT' if fail_tx else 'ROLLED BACK CLEANLY'}")

    print(f"  Cache Key Present: {'user_vector_9982' in cache_node.committed_storage}")
    print(f"  Vector Key Present: {'user_vector_9982' in failing_vector_node.committed_storage}")
    print(f"  Metadata Key Present: {'user_vector_9982' in meta_node.committed_storage}")

    print("\n2PC Coordinator Transaction Journal:")
    print(json.dumps(coordinator_healthy.transaction_logs + coordinator_with_failure.transaction_logs, indent=2))