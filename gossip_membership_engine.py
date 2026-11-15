# Distributed Systems & MLOps Infrastructure: Peer-to-peer PING/PING-REQ probing, incarnation refutation, and gossip dissemination

import json
import random
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

class MemberState(Enum):
    ALIVE = "ALIVE"
    SUSPECT = "SUSPECT"
    DEAD = "DEAD"

class MemberRecord:
    """Represents a node's view of a peer's health state and monotonic incarnation."""
    def __init__(self, node_id: str, state: MemberState = MemberState.ALIVE, incarnation: int = 0):
        self.node_id = node_id
        self.state = state
        self.incarnation = incarnation
        self.updated_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "state": self.state.value,
            "incarnation": self.incarnation
        }
class GossipClusterNode:
    """
    Decentralized cluster participant implementing SWIM-inspired failure detection
    (PING, indirect PING-REQ, suspicion grace periods, and incarnation refutation).
    """
    def __init__(self, node_id: str, indirect_probe_k: int = 2):
        self.node_id = node_id
        self.indirect_probe_k = indirect_probe_k
        self.incarnation = 0
        self.is_online = True
        self.blocked_peers: Set[str] = set()
        self.membership: Dict[str, MemberRecord] = {
            node_id: MemberRecord(node_id, MemberState.ALIVE, self.incarnation)
        }
        self.gossip_queue: List[Dict[str, Any]] = []

    def join_seed(self, seed_node: "GossipClusterNode"):
        """Bootstraps cluster membership by exchanging initial tables with a seed peer."""
        for peer_id, rec in seed_node.membership.items():
            self._merge_gossip_update(rec.to_dict())
        seed_node._merge_gossip_update(self.membership[self.node_id].to_dict())

    def _enqueue_gossip(self, record: MemberRecord):
        payload = record.to_dict()
        self.gossip_queue = [g for g in self.gossip_queue if g["node_id"] != record.node_id]
        self.gossip_queue.append(payload)

    def _merge_gossip_update(self, update: Dict[str, Any]):
        """
        Applies SWIM state precedence rules:
        - Higher incarnation always wins.
        - Within same incarnation: DEAD > SUSPECT > ALIVE.
        - If self is marked SUSPECT, increment incarnation and refute with ALIVE.
        """
        target_id = update["node_id"]
        incoming_state = MemberState(update["state"])
        incoming_inc = update["incarnation"]

        if target_id == self.node_id and incoming_state == MemberState.SUSPECT:
            if incoming_inc >= self.incarnation:
                self.incarnation = incoming_inc + 1
                self_rec = MemberRecord(self.node_id, MemberState.ALIVE, self.incarnation)
                self.membership[self.node_id] = self_rec
                self._enqueue_gossip(self_rec)
                print(f"[{self.node_id}] Refuted false SUSPECT! Incremented incarnation -> {self.incarnation}.")
            return

        existing = self.membership.get(target_id)
        if existing is None:
            new_rec = MemberRecord(target_id, incoming_state, incoming_inc)
            self.membership[target_id] = new_rec
            self._enqueue_gossip(new_rec)
            return

        state_rank = {MemberState.ALIVE: 1, MemberState.SUSPECT: 2, MemberState.DEAD: 3}
        should_overwrite = False

        if incoming_inc > existing.incarnation:
            should_overwrite = True
        elif incoming_inc == existing.incarnation and state_rank[incoming_state] > state_rank[existing.state]:
            should_overwrite = True

        if should_overwrite:
            existing.state = incoming_state
            existing.incarnation = incoming_inc
            existing.updated_at = time.time()
            self._enqueue_gossip(existing)

    def receive_ping(self, sender_id: str, piggybacked_gossip: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        """Handles incoming direct or indirect PING; returns ACK with local gossip."""
        if not self.is_online or sender_id in self.blocked_peers:
            return None

        for item in piggybacked_gossip:
            self._merge_gossip_update(item)

        return list(self.gossip_queue)

    def run_protocol_tick(self, cluster_directory: Dict[str, "GossipClusterNode"]):
        """Executes one SWIM failure-detection cycle against a randomly chosen peer."""
        if not self.is_online:
            return

        candidates = [
            pid for pid, rec in self.membership.items()
            if pid != self.node_id and rec.state != MemberState.DEAD
        ]
        if not candidates:
            return

        target_id = random.choice(candidates)
        target_node = cluster_directory.get(target_id)
        ack_gossip = target_node.receive_ping(self.node_id, self.gossip_queue) if target_node else None

        if ack_gossip is not None:
            for item in ack_gossip:
                self._merge_gossip_update(item)
            return
        print(f"[{self.node_id}] Direct PING to '{target_id}' timed out. Initiating indirect PING-REQ...")
        helpers = [pid for pid in candidates if pid != target_id]
        selected_helpers = random.sample(helpers, min(self.indirect_probe_k, len(helpers)))

        indirect_success = False
        for helper_id in selected_helpers:
            helper_node = cluster_directory.get(helper_id)
            if helper_node and helper_node.is_online and helper_id not in self.blocked_peers:
                proxied_ack = target_node.receive_ping(helper_id, self.gossip_queue) if target_node else None
                if proxied_ack is not None:
                    indirect_success = True
                    for item in proxied_ack:
                        self._merge_gossip_update(item)
                    print(f"[{self.node_id}] Indirect PING-REQ via '{helper_id}' confirmed '{target_id}' is ALIVE!")
                    break

        if not indirect_success:
            rec = self.membership[target_id]
            if rec.state == MemberState.ALIVE:
                rec.state = MemberState.SUSPECT
                self._enqueue_gossip(rec)
                print(f"[{self.node_id}] Marked '{target_id}' as SUSPECT (Incarnation {rec.incarnation}).")
            elif rec.state == MemberState.SUSPECT:
                rec.state = MemberState.DEAD
                self._enqueue_gossip(rec)
                print(f"[{self.node_id}] Confirmed '{target_id}' is DEAD -> Gossiping eviction.")


if __name__ == "__main__":
    print("--- MLOps Cluster Coordination: SWIM Gossip Membership Engine ---\n")
    random.seed(42)

    nodes = {
        f"inference_node_0{i}": GossipClusterNode(f"inference_node_0{i}")
        for i in range(1, 5)
    }
    seed = nodes["inference_node_01"]

    print("Step 1: Bootstrapping 4-node decentralized cluster via seed node:")
    for nid, node in nodes.items():
        if nid != "inference_node_01":
            node.join_seed(seed)
    for _ in range(4):
        for node in nodes.values():
            node.run_protocol_tick(nodes)

    print(f"  Node 04 known alive peers: {[k for k, v in nodes['inference_node_04'].membership.items() if v.state == MemberState.ALIVE]}")

    print("\nStep 2: Simulating asymmetric rack link drop (Node 01 <-> Node 02 blocked):")
    nodes["inference_node_01"].blocked_peers.add("inference_node_02")
    nodes["inference_node_02"].blocked_peers.add("inference_node_01")

    for _ in range(5):
        nodes["inference_node_01"].run_protocol_tick(nodes)

    print("\nStep 3: Simulating hard crash on 'inference_node_03' and gossip convergence:")
    nodes["inference_node_03"].is_online = False

    for round_idx in range(1, 12):
        for node in nodes.values():
            node.run_protocol_tick(nodes)

    print("\nFinal Cluster Membership View (from 'inference_node_04'):")
    final_view = {
        peer_id: rec.to_dict()
        for peer_id, rec in sorted(nodes["inference_node_04"].membership.items())
    }
    print(json.dumps(final_view, indent=2))