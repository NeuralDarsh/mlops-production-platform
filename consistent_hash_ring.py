# Distributed Systems & Storage: Ring hashing, v-node distribution, O(log N) routing, and minimal-churn rebalancing

import bisect
import hashlib
import json
from typing import Dict, List, Optional, Set

class ConsistentHashRing:
    """
    Consistent Hash Ring mapping distributed feature store keys across
    physical nodes with virtual node replicas to ensure uniform distribution.
    """
    def __init__(self, replicas: int = 100):
        self.replicas = replicas
        self.ring: Dict[int, str] = {}
        self.sorted_keys: List[int] = []
        self.nodes: Set[str] = set()

    def _hash(self, key: str) -> int:
        """Computes a deterministic 32-bit integer hash on the continuous ring [0, 2^32 - 1]."""
        digest = hashlib.md5(key.encode("utf-8")).hexdigest()
        return int(digest[:8], 16)

    def add_node(self, node: str):
        """Adds a physical node by projecting its virtual replicas onto the ring."""
        if node in self.nodes:
            return
        self.nodes.add(node)
        for i in range(self.replicas):
            vnode_key = f"{node}#vnode_{i}"
            ring_hash = self._hash(vnode_key)
            self.ring[ring_hash] = node
            bisect.insort(self.sorted_keys, ring_hash)
        print(f"[NODE ADDED] '{node}' registered with {self.replicas} virtual nodes.")

    def remove_node(self, node: str):
        """Removes a physical node and prunes its associated virtual nodes."""
        if node not in self.nodes:
            return
        self.nodes.remove(node)
        for i in range(self.replicas):
            vnode_key = f"{node}#vnode_{i}"
            ring_hash = self._hash(vnode_key)
            if ring_hash in self.ring:
                del self.ring[ring_hash]
                idx = bisect.bisect_left(self.sorted_keys, ring_hash)
                if idx < len(self.sorted_keys) and self.sorted_keys[idx] == ring_hash:
                    self.sorted_keys.pop(idx)
        print(f"[NODE REMOVED] '{node}' and virtual replicas unallocated from ring.")

    def get_node(self, key: str) -> Optional[str]:
        """
        Locates the designated storage node for a feature key by searching
        clockwise on the ring using binary search in O(log N) time.
        """
        if not self.ring:
            return None

        key_hash = self._hash(key)
        idx = bisect.bisect_right(self.sorted_keys, key_hash)

        if idx == len(self.sorted_keys):
            idx = 0

        target_hash = self.sorted_keys[idx]
        return self.ring[target_hash]


if __name__ == "__main__":
    print("--- MLOps Distributed Systems: Consistent Hash Ring Partitioner ---\n")

    ring = ConsistentHashRing(replicas=150)
    print("Step 1: Initializing distributed cluster topology:")
    ring.add_node("shard_us_east_01")
    ring.add_node("shard_eu_central_02")
    ring.add_node("shard_ap_northeast_03")

    keys = [f"entity_user_feature_id_{i}" for i in range(1000)]
    initial_allocations: Dict[str, str] = {}
    distribution_counts: Dict[str, int] = {node: 0 for node in ring.nodes}

    for k in keys:
        node = ring.get_node(k)
        initial_allocations[k] = node
        distribution_counts[node] += 1

    print("\nInitial Key Distribution (1,000 Features):")
    for node, count in distribution_counts.items():
        print(f" Node '{node}': {count} keys ({count / 10.0:.1f}%)")

    print("\nStep 2: Scaling cluster horizontally (Adding 'shard_ap_south_04'):")
    ring.add_node("shard_ap_south_04")

    remapped_count = 0
    new_distribution_counts: Dict[str, int] = {node: 0 for node in ring.nodes}

    for k in keys:
        node = ring.get_node(k)
        new_distribution_counts[node] += 1
        if node != initial_allocations[k]:
            remapped_count += 1

    print("\nRebalanced Key Distribution (4 Shards):")
    for node, count in new_distribution_counts.items():
        print(f"  Node '{node}': {count} keys ({count / 10.0:.1f}%)")

    rebalance_pct = (remapped_count / len(keys)) * 100.0
    print(f"\n[CHURN METRIC] Only {remapped_count} keys ({rebalance_pct:.1f}%) remapped!")
    print(f"Theoretical ideal churn for 1/4 nodes = 25.0%. Consistent hashing minimized cache thrashing.")

    print("\nStep 3: Simulating node failure (Removing 'shard_eu_central_02'):")
    ring.remove_node("shard_eu_central_02")

    post_failure_node = ring.get_node("entity_user_feature_id_42")
    print(f"Key 'entity_user_feature_id_42' routed seamlessly to: '{post_failure_node}'")