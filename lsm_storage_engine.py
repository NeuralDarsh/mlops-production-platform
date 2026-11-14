# Distributed Storage & MLOps Infrastructure: Write-Ahead Log recovery, sorted MemTables, SSTable flushing, and compaction

import json
import os
import tempfile
import time
import zlib
from typing import Any, Dict, List, Optional, Tuple

TOMBSTONE = "__LSM_TOMBSTONE__"

class WriteAheadLog:
    """Append-only crash recovery log storing CRC32-checksummed mutation records."""
    def __init__(self, wal_path: str):
        self.wal_path = wal_path

    def append(self, operation: str, key: str, value: Any):
        entry = {"op": operation, "key": key, "val": value, "ts": time.time()}
        raw_json = json.dumps(entry, separators=(",", ":"))
        checksum = zlib.crc32(raw_json.encode("utf-8"))
        line = f"{checksum}|{raw_json}\n"
        with open(self.wal_path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())

    def recover(self) -> List[Dict[str, Any]]:
        """Replays valid entries from disk on startup, skipping corrupted tail lines."""
        if not os.path.exists(self.wal_path):
            return []

        recovered = []
        with open(self.wal_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f,start=1):
                line = line.strip()
                if not line or "|" not in line:
                    continue
                checksum_str, raw_json = line.split("|", 1)
                expected_crc = int(checksum_str)
                actual_crc = zlib.crc32(raw_json.encode("utf-8"))
                if expected_crc == actual_crc:
                    recovered.append(json.loads(raw_json))
                else:
                    print(f"[WAL CORRUPTION] Skipping damaged record at line {line_num}.")
        return recovered

    def clear(self):
        """Truncates WAL safely once MemTable has flushed to an immutable SSTable."""
        with open(self.wal_path, "w", encoding="utf-8") as f:
            f.truncate(0)


class LSMStorageEngine:
    """
    Log-Structured Merge-Tree key-value engine for high-throughput feature persistence.
    Combines a durable WAL, in-memory sorted MemTable, SSTable flushes, and compaction.
    """
    def __init__(self, storage_dir: str, memtable_flush_threshold: int = 4):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

        self.wal = WriteAheadLog(os.path.join(self.storage_dir, "active.wal"))
        self.memtable: Dict[str, Any] = {}
        self.memtable_flush_threshold = memtable_flush_threshold
        self.sstables: List[str] = self._discover_existing_sstables()
        self._replay_wal_on_startup()

    def _discover_existing_sstables(self) -> List[str]:
        files = [
            os.path.join(self.storage_dir, fn)
            for fn in os.listdir(self.storage_dir)
            if fn.startswith("sstable_") and fn.endswith(".json")
        ]
        return sorted(files)

    def _replay_wal_on_startup(self):
        entries = self.wal.recover()
        if entries:
            print(f" [CRASH RECOVERY] Replaying {len(entries)} uncommitted mutations from WAL...")
            for entry in entries:
                if entry["op"] == "PUT":
                    self.memtable[entry["key"]] = entry["val"]
                elif entry["op"] == "DEL":
                    self.memtable[entry["key"]] = TOMBSTONE

    def put(self, key: str, value: Any):
        """Writes mutation to WAL first, then updates sorted MemTable."""
        self.wal.append("PUT", key, value)
        self.memtable[key] = value
        if len(self.memtable) >= self.memtable_flush_threshold:
            self.flush_memtable()

    def delete(self, key: str):
        """Appends a tombstone deletion marker rather than seeking on disk."""
        self.wal.append("DEL", key, None)
        self.memtable[key] = TOMBSTONE
        if len(self.memtable) >= self.memtable_flush_threshold:
            self.flush_memtable()

    def get(self, key: str) -> Optional[Any]:
        """Reads MemTable first, then scans SSTables from newest to oldest."""
        if key in self.memtable:
            val = self.memtable[key]
            return None if val == TOMBSTONE else val

        for sstable_path in reversed(self.sstables):
            with open(sstable_path, "r", encoding="utf-8") as f:
                table_data = json.load(f)
                if key in table_data:
                    val = table_data[key]
                    return None if val == TOMBSTONE else val

        return None

    def flush_memtable(self):
        """Sorts MemTable keys and writes an immutable SSTable segment to disk."""
        if not self.memtable:
            return

        sorted_items = dict(sorted(self.memtable.items()))
        sstable_id = f"sstable_{int(time.time() * 1_000_000)}.json"
        sstable_path = os.path.join(self.storage_dir, sstable_id)

        with open(sstable_path, "w", encoding="utf-8") as f:
            json.dump(sorted_items, f, indent=2)

        self.sstables.append(sstable_path)
        self.memtable.clear()
        self.wal.clear()
        print(f" [SSTABLE FLUSH] Flushed sorted segment '{sstable_id}' ({len(sorted_items)} keys).")

    def compact_sstables(self):
        """
        Merges all immutable SSTables into a single compacted segment,
        purging overwritten keys and dropping tombstones.
        """
        if len(self.sstables) < 2:
            print(" [COMPACTION] Fewer than 2 SSTables present; skipping compaction.")
            return

        merged_state: Dict[str, Any] = {}
        for path in self.sstables:
            with open(path, "r", encoding="utf-8") as f:
                merged_state.update(json.load(f))

        compacted_data = {
            k: v for k, v in sorted(merged_state.items()) if v != TOMBSTONE
        }

        compacted_name = f"sstable_{int(time.time() * 1_000_000)}_compacted.json"
        compacted_path = os.path.join(self.storage_dir, compacted_name)
        with open(compacted_path, "w", encoding="utf-8") as f:
            json.dump(compacted_data, f, indent=2)

        old_count = len(self.sstables)
        for old_path in self.sstables:
            if os.path.exists(old_path):
                os.remove(old_path)

        self.sstables = [compacted_path]
        print(
            f" [COMPACTION COMPLETE] Merged {old_count} SSTables -> '{compacted_name}' "
            f"({len(compacted_data)} live keys retained)."
        )


if __name__ == "__main__":
    print("--- MLOps Storage Engine: LSM-Tree with WAL & SSTable Compaction ---\n")

    with tempfile.TemporaryDirectory() as tmp_dir:
        engine = LSMStorageEngine(storage_dir=tmp_dir, memtable_flush_threshold=3)

        print("Step 1: Writing initial feature records (Flush threshold = 3):")
        engine.put("user:1001:score", 0.91)
        engine.put("user:1002:score", 0.42)
        engine.put("user:1003:score", 0.77)  

        print("\nStep 2: Updating 'user:1001:score' and deleting 'user:1002:score':")
        engine.put("user:1001:score", 0.98)
        engine.delete("user:1002:score")
        engine.put("user:1004:score", 0.65)  

        print(f"  Lookup 'user:1001:score' (Updated): {engine.get('user:1001:score')}")
        print(f"  Lookup 'user:1002:score' (Deleted): {engine.get('user:1002:score')}")

        print("\n Step 3: Writing uncommitted record & simulating sudden process crash:")
        engine.put("user:9999:score", 0.85)  
        del engine 

        recovered_engine = LSMStorageEngine(storage_dir=tmp_dir, memtable_flush_threshold=3)
        print(f"  Post-Crash Lookup 'user:9999:score': {recovered_engine.get('user:9999:score')}")

        print("\nStep 4: Flushing remaining MemTable and executing SSTable compaction:")
        recovered_engine.flush_memtable()
        recovered_engine.compact_sstables()
        print(f"  Active SSTable count post-compaction: {len(recovered_engine.sstables)}")