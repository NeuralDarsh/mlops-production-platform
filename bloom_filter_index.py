# Distributed Storage & MLOps Infrastructure: Optimal bit-array sizing, Kirsch-Mitzenmacher double hashing, and I/O short-circuiting

import hashlib
import json
import math
from typing import Any, Dict, List, Optional, Tuple


class BloomFilterIndex:
    """
    Space-efficient probabilistic set membership filter eliminating unnecessary
    disk reads on negative lookups in LSM-Tree SSTables and feature stores.
    """
    def __init__(self, expected_items: int, false_positive_rate: float = 0.01):
        if expected_items <= 0:
            raise ValueError("expected_items must be greater than 0.")
        if not (0.0 < false_positive_rate < 1.0):
            raise ValueError("false_positive_rate must be strictly between 0.0 and 1.0.")

        self.expected_items = expected_items
        self.target_fp_rate = false_positive_rate

        self.bit_size = max(8, int(math.ceil(-(expected_items * math.log(false_positive_rate)) / (math.log(2) ** 2))))
        self.hash_count = max(1, int(round((self.bit_size / expected_items) * math.log(2))))

        self._byte_array = bytearray((self.bit_size + 7) // 8)
        self.inserted_count = 0

    def _get_bit_positions(self, item: str) -> List[int]:
        """
        Generates k bit indices using Kirsch-Mitzenmacher double hashing:
        pos_i = (h1 + i * h2) mod m
        """
        encoded = item.encode("utf-8")
        h1 = int(hashlib.md5(encoded).hexdigest(), 16)
        h2 = int(hashlib.sha256(encoded).hexdigest(), 16)

        return [(h1 + i * h2) % self.bit_size for i in range(self.hash_count)]

    def add(self, item: str):
        """Inserts a key into the Bloom filter by setting k bits to 1."""
        for bit_idx in self._get_bit_positions(item):
            byte_idx = bit_idx // 8
            bit_offset = bit_idx % 8
            self._byte_array[byte_idx] |= (1 << bit_offset)
        self.inserted_count += 1

    def might_contain(self, item: str) -> bool:
        """
        Checks set membership:
        - Returns False: Key is 100% guaranteed absent (zero false negatives).
        - Returns True: Key is probably present (subject to target_fp_rate).
        """
        for bit_idx in self._get_bit_positions(item):
            byte_idx = bit_idx // 8
            bit_offset = bit_idx % 8
            if not (self._byte_array[byte_idx] & (1 << bit_offset)):
                return False
        return True

    def estimated_current_fp_rate(self) -> float:
        """Computes theoretical false positive probability: (1 - e^(-k*n/m))^k."""
        exponent = -(self.hash_count * self.inserted_count) / self.bit_size
        return round((1.0 - math.exp(exponent)) ** self.hash_count, 6)

    def get_diagnostics(self) -> Dict[str, Any]:
        return {
            "expected_items": self.expected_items,
            "inserted_items": self.inserted_count,
            "bit_array_size_bits": self.bit_size,
            "memory_footprint_bytes": len(self._byte_array),
            "hash_functions_k": self.hash_count,
            "target_fp_rate": self.target_fp_rate,
            "estimated_current_fp_rate": self.estimated_current_fp_rate()
        }


class IndexedSSTableSegment:
    """Simulates an on-disk SSTable guarded by an in-memory BloomFilterIndex."""
    def __init__(self, segment_id: str, records: Dict[str, Any], fp_rate: float = 0.01):
        self.segment_id = segment_id
        self._disk_records = dict(records)
        self.bloom_filter = BloomFilterIndex(expected_items=max(1, len(records)), false_positive_rate=fp_rate)
        self.disk_reads_executed = 0
        self.disk_reads_saved = 0

        for key in records.keys():
            self.bloom_filter.add(key)

    def lookup(self, key: str) -> Optional[Any]:
        """Queries Bloom filter first; short-circuits disk I/O if negative."""
        if not self.bloom_filter.might_contain(key):
            self.disk_reads_saved += 1
            return None
        self.disk_reads_executed += 1
        return self._disk_records.get(key)


if __name__ == "__main__":
    print("--- MLOps Storage Optimization: Bloom Filter Index Engine ---\n")

    known_records = {f"user_embedding_key_{i}": [round(i * 0.01, 3), 0.95] for i in range(500)}
    segment = IndexedSSTableSegment("sstable_segment_001", known_records, fp_rate=0.01)

    print("Bloom Filter Sizing & Memory Diagnostics:")
    print(json.dumps(segment.bloom_filter.get_diagnostics(), indent=2))

    print("\nStep 1: Verifying 100% recall across all 500 inserted keys:")
    misses = sum(1 for k in known_records if segment.lookup(k) is None)
    print(f"False Negatives Observed: {misses} (Must strictly equal 0)")

    print("\nStep 2: Querying 1,000 missing keys to measure disk I/O short-circuiting:")
    segment.disk_reads_executed = 0
    segment.disk_reads_saved = 0

    for j in range(1000):
        missing_key = f"cold_start_unknown_user_{j}"
        segment.lookup(missing_key)

    empirical_fp = (segment.disk_reads_executed / 1000.0) * 100.0
    io_savings_pct = (segment.disk_reads_saved / 1000.0) * 100.0

    print(f"Disk Reads Eliminated by Bloom Filter: {segment.disk_reads_saved}/1000 ({io_savings_pct:.1f}% I/O saved)")
    print(f"Unnecessary Disk Reads (False Positives): {segment.disk_reads_executed}/1000 ({empirical_fp:.2f}% empirical FP rate)")