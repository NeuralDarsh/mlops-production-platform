# MLOps & High-Performance Serving: L1 in-memory LRU, L2 simulated shared tier, and Stale-While-Revalidate patterns

import collections
import time
import json
from typing import Any, Callable, Dict, Optional, Tuple

class CacheEntry:
    """Stores feature payload with creation timestamp and fresh/stale lifetime thresholds."""
    def __init__(self, value: Dict[str, Any], fresh_ttl_sec: float, stale_ttl_sec: float):
        self.value = value
        self.created_at = time.time()
        self.fresh_until = self.created_at + fresh_ttl_sec
        self.stale_until = self.fresh_until + stale_ttl_sec

    def is_fresh(self) -> bool:
        return time.time() <= self.fresh_until

    def is_stale_allowed(self) -> bool:
        now = time.time()
        return self.fresh_until < now <= self.stale_until

    def is_expired(self) -> bool:
        return time.time() > self.stale_until


class TieredFeatureCache:
    """
    Two-tiered caching system combining an L1 in-process LRU cache
    and an L2 simulated distributed cache with Stale-While-Revalidate (SWR) semantics.
    """
    def __init__(
        self,
        l1_capacity: int = 4,
        fresh_ttl_sec: float = 1.0,
        stale_ttl_sec: float = 2.0
    ):
        self.l1_capacity = l1_capacity
        self.fresh_ttl = fresh_ttl_sec
        self.stale_ttl = stale_ttl_sec

        # L1: In-process LRU cache (OrderedDict maintains recency order)
        self.l1_cache: collections.OrderedDict[str, CacheEntry] = collections.OrderedDict()
        # L2: Simulated distributed shared memory store (e.g., Redis)
        self.l2_cache: Dict[str, CacheEntry] = {}

        self.stats = {"l1_hits": 0, "l2_hits": 0, "db_fetches": 0, "swr_refreshes": 0}

    def _l1_put(self, key: str, entry: CacheEntry):
        """Pushes entry into L1, executing LRU eviction if capacity is exceeded."""
        if key in self.l1_cache:
            self.l1_cache.move_to_end(key)
        self.l1_cache[key] = entry
        if len(self.l1_cache) > self.l1_capacity:
            evicted_key, _ = self.l1_cache.popitem(last=False)
            print(f"[L1 LRU EVICTION] Purged cold entity: '{evicted_key}'")

    def get_features(
        self,
        entity_id: str,
        authoritative_db_fetcher: Callable[[str], Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], str]:
        """
        Retrieves features using tiered cascade:
        1. L1 LRU -> 2. L2 Shared -> 3. Authoritative DB fetch.
        """
        now = time.time()

        # Tier 1: Check L1
        if entity_id in self.l1_cache:
            entry = self.l1_cache[entity_id]
            if entry.is_fresh():
                self.stats["l1_hits"] += 1
                self.l1_cache.move_to_end(entity_id)
                return entry.value, "L1_HIT_FRESH"

            if entry.is_stale_allowed():
                # Stale-While-Revalidate: serve stale immediately, revalidate
                self.stats["l1_hits"] += 1
                self.stats["swr_refreshes"] += 1
                self.l1_cache.move_to_end(entity_id)
                self._revalidate(entity_id, authoritative_db_fetcher)
                return entry.value, "L1_HIT_STALE_REVALIDATING"

            # Expired: remove from L1
            del self.l1_cache[entity_id]

        # Tier 2: Check L2
        if entity_id in self.l2_cache:
            entry = self.l2_cache[entity_id]
            if not entry.is_expired():
                self.stats["l2_hits"] += 1
                # Backfill L1
                self._l1_put(entity_id, entry)
                tier_msg = "L2_HIT_FRESH" if entry.is_fresh() else "L2_HIT_STALE"
                return entry.value, tier_msg
            else:
                del self.l2_cache[entity_id]

        # Tier 3: Dual Miss -> Query Authoritative DB
        self.stats["db_fetches"] += 1
        fresh_data = authoritative_db_fetcher(entity_id)
        new_entry = CacheEntry(fresh_data, self.fresh_ttl, self.stale_ttl)

        # Populate both cache tiers
        self._l1_put(entity_id, new_entry)
        self.l2_cache[entity_id] = new_entry
        return fresh_data, "DB_FETCH_MISS"

    def _revalidate(self, entity_id: str, fetcher: Callable[[str], Dict[str, Any]]):
        """Executes a revalidation sweep against the primary feature store."""
        print(f" [SWR BACKGROUND REVALIDATION] Refreshing features for '{entity_id}'")
        fresh_data = fetcher(entity_id)
        refreshed_entry = CacheEntry(fresh_data, self.fresh_ttl, self.stale_ttl)
        self._l1_put(entity_id, refreshed_entry)
        self.l2_cache[entity_id] = refreshed_entry


if __name__ == "__main__":
    print("--- MLOps Feature Serving: Tiered L1/L2 Feature Cache Engine ---\n")

    # Authoritative mock database fetcher
    db_access_counter = 0

    def query_feature_store(entity: str) -> dict:
        global db_access_counter
        db_access_counter += 1
        time.sleep(0.04)  # Simulate DB query latency
        return {
            "entity": entity,
            "user_score": round(0.75 + (db_access_counter * 0.01), 3),
            "txn_velocity": 4,
            "version": db_access_counter
        }

    cache = TieredFeatureCache(l1_capacity=2, fresh_ttl_sec=0.5, stale_ttl_sec=1.0)

    # 1. First fetch (Cold start -> DB fetch)
    print("Step 1: Initial cold fetch for 'user_tokyo_01':")
    data1, status1 = cache.get_features("user_tokyo_01", query_feature_store)
    print(f"  Result: {data1} | Status: {status1}")

    # 2. Immediate second fetch (Warm -> L1 Hit)
    print("\nStep 2: Immediate repeated fetch for 'user_tokyo_01':")
    data2, status2 = cache.get_features("user_tokyo_01", query_feature_store)
    print(f"  Result: {data2} | Status: {status2}")

    # 3. Fill L1 capacity to observe LRU eviction
    print("\nStep 3: Populating other users to trigger L1 LRU eviction:")
    cache.get_features("user_osaka_02", query_feature_store)
    cache.get_features("user_kyoto_03", query_feature_store)  # Pushes 'user_tokyo_01' out of L1

    print("\nStep 4: Fetching evicted 'user_tokyo_01' (Resolves from L2):")
    data3, status3 = cache.get_features("user_tokyo_01", query_feature_store)
    print(f"  Result: {data3} | Status: {status3}")

    # 4. Demonstrate Stale-While-Revalidate (SWR)
    print("\nStep 5: Sleeping 0.6s to enter Stale window (Fresh TTL = 0.5s):")
    time.sleep(0.6)
    data4, status4 = cache.get_features("user_tokyo_01", query_feature_store)
    print(f"  Result: {data4} | Status: {status4}")

    print("\nCache Telemetry & Hit Statistics:")
    print(json.dumps(cache.stats, indent=2))