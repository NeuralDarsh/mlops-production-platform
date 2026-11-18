# Distributed Systems & Edge MLOps: Causal ordering, concurrent conflict detection, and PN-Counter convergence

import json
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class CausalRelation(Enum):
    BEFORE = "BEFORE"          
    AFTER = "AFTER"             
    EQUAL = "EQUAL"             
    CONCURRENT = "CONCURRENT"   


class VectorClock:
    """Tracks logical causality across distributed edge replicas."""
    def __init__(self, clock: Optional[Dict[str, int]] = None):
        self.clock: Dict[str, int] = dict(clock) if clock else {}

    def increment(self, node_id: str):
        self.clock[node_id] = self.clock.get(node_id, 0) + 1

    def merge(self, other: "VectorClock"):
        all_nodes = set(self.clock.keys()) | set(other.clock.keys())
        for nid in all_nodes:
            self.clock[nid] = max(self.clock.get(nid, 0), other.clock.get(nid, 0))

    def compare(self, other: "VectorClock") -> CausalRelation:
        """Determines exact causal relationship between two vector clocks."""
        all_nodes = set(self.clock.keys()) | set(other.clock.keys())
        has_less = False
        has_greater = False

        for nid in all_nodes:
            v1 = self.clock.get(nid, 0)
            v2 = other.clock.get(nid, 0)
            if v1 < v2:
                has_less = True
            elif v1 > v2:
                has_greater = True

        if not has_less and not has_greater:
            return CausalRelation.EQUAL
        if has_less and not has_greater:
            return CausalRelation.BEFORE
        if has_greater and not has_less:
            return CausalRelation.AFTER
        return CausalRelation.CONCURRENT

    def copy(self) -> "VectorClock":
        return VectorClock(self.clock)

    def to_dict(self) -> Dict[str, int]:
        return dict(sorted(self.clock.items()))


class PNCounterCRDT:
    """
    State-based Positive-Negative Counter CRDT.
    Guarantees strong eventual consistency across partitioned edge nodes via element-wise max joins.
    """
    def __init__(self):
        self.p_vector: Dict[str, int] = {}  
        self.n_vector: Dict[str, int] = {} 

    def increment(self, node_id: str, amount: int = 1):
        if amount < 0:
            raise ValueError("Use decrement() for negative adjustments.")
        self.p_vector[node_id] = self.p_vector.get(node_id, 0) + amount

    def decrement(self, node_id: str, amount: int = 1):
        if amount < 0:
            raise ValueError("Amount must be non-negative.")
        self.n_vector[node_id] = self.n_vector.get(node_id, 0) + amount

    def value(self) -> int:
        return sum(self.p_vector.values()) - sum(self.n_vector.values())

    def merge(self, other: "PNCounterCRDT"):
        """Commutative, associative, and idempotent join operation."""
        for nid, val in other.p_vector.items():
            self.p_vector[nid] = max(self.p_vector.get(nid, 0), val)
        for nid, val in other.n_vector.items():
            self.n_vector[nid] = max(self.n_vector.get(nid, 0), val)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "net_value": self.value(),
            "increments": dict(sorted(self.p_vector.items())),
            "decrements": dict(sorted(self.n_vector.items()))
        }


class EdgeFeatureReplica:
    """
    Represents an edge inference node maintaining causal vector clocks,
    CRDT session counters, and concurrent metadata versions.
    """
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.vector_clock = VectorClock()
        self.click_counter = PNCounterCRDT()
        self.active_model_tag: str = "baseline_v1"
        self.tag_clock = VectorClock()
        self.conflict_siblings: List[Tuple[str, Dict[str, int]]] = []

    def record_user_interaction(self, clicks: int = 1, model_tag: Optional[str] = None):
        """Applies local mutation while disconnected or online."""
        self.vector_clock.increment(self.node_id)
        if clicks > 0:
            self.click_counter.increment(self.node_id, clicks)
        elif clicks < 0:
            self.click_counter.decrement(self.node_id, abs(clicks))

        if model_tag is not None:
            self.active_model_tag = model_tag
            self.tag_clock = self.vector_clock.copy()
            self.conflict_siblings.clear()

    def anti_entropy_sync_from(self, peer: "EdgeFeatureReplica"):
        """
        Merges state from a peer replica:
        1. Joins PN-Counter CRDT losslessly.
        2. Detects causal precedence vs. concurrent conflict on scalar tags.
        3. Advances local vector clock.
        """
        relation = self.tag_clock.compare(peer.tag_clock)

        if relation == CausalRelation.BEFORE:
            self.active_model_tag = peer.active_model_tag
            self.tag_clock = peer.tag_clock.copy()
            self.conflict_siblings.clear()
        elif relation == CausalRelation.CONCURRENT and self.active_model_tag != peer.active_model_tag:
            siblings = [
                (self.active_model_tag, self.tag_clock.to_dict()),
                (peer.active_model_tag, peer.tag_clock.to_dict())
            ]
            self.conflict_siblings = siblings
            self.active_model_tag = max(self.active_model_tag, peer.active_model_tag)
            self.tag_clock.merge(peer.tag_clock)
            print(
                f"[{self.node_id}] Concurrent conflict detected with '{peer.node_id}'! "
                f"Preserved siblings & converged on '{self.active_model_tag}'."
            )

        self.click_counter.merge(peer.click_counter)
        self.vector_clock.merge(peer.vector_clock)


if __name__ == "__main__":
    print("--- MLOps Edge State Convergence: Vector Clock & CRDT Engine ---\n")

    tokyo = EdgeFeatureReplica("edge_tokyo_01")
    mumbai = EdgeFeatureReplica("edge_mumbai_02")
    frankfurt = EdgeFeatureReplica("edge_frankfurt_03")

    print("Step 1: Simulating network partition with independent regional mutations:")
    tokyo.record_user_interaction(clicks=5, model_tag="recsys_kanji_v2")
    tokyo.record_user_interaction(clicks=3)

    mumbai.record_user_interaction(clicks=4, model_tag="recsys_indic_v2")
    mumbai.record_user_interaction(clicks=-1)

    frankfurt.record_user_interaction(clicks=2)

    print(f"Tokyo pre-sync-> Clicks: {tokyo.click_counter.value()} | Clock: {tokyo.vector_clock.to_dict()}")
    print(f"Mumbai pre-sync-> Clicks: {mumbai.click_counter.value()} | Clock: {mumbai.vector_clock.to_dict()}")
    print(f"Frankfurt pre-sync-> Clicks: {frankfurt.click_counter.value()} | Clock: {frankfurt.vector_clock.to_dict()}")
    print("\nStep 2: Partition heals-> Running bidirectional anti-entropy CRDT sync:")
    tokyo.anti_entropy_sync_from(mumbai)
    tokyo.anti_entropy_sync_from(frankfurt)

    mumbai.anti_entropy_sync_from(tokyo)
    frankfurt.anti_entropy_sync_from(tokyo)
    tokyo.anti_entropy_sync_from(mumbai)

    print("\nPost-Convergence State Audit (All 3 Edge Replicas Identical):")
    audit = {
        node.node_id: {
            "vector_clock": node.vector_clock.to_dict(),
            "converged_clicks": node.click_counter.snapshot(),
            "active_model_tag": node.active_model_tag
        }
        for node in [tokyo, mumbai, frankfurt]
    }
    print(json.dumps(audit, indent=2))