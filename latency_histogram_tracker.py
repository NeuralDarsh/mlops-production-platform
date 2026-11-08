# MLOps Telemetry & Infrastructure: Histogram binning, p50/p90/p99 rank interpolation, and SLA tail latency budgets

import bisect
import json
import math
import time
from typing import Dict, List, Optional, Tuple


class LatencyHistogramTracker:
    """
    Fixed-memory streaming latency histogram providing sub-millisecond p50, p90,
    and p99 percentile estimations via cumulative distribution interpolation.
    """
    def __init__(self, bucket_boundaries: Optional[List[float]] = None):
        self.boundaries = bucket_boundaries or [
            1.0, 5.0, 10.0, 20.0, 35.0, 50.0, 75.0, 100.0, 150.0, 250.0, 500.0, 1000.0
        ]
        self.bucket_counts = [0] * (len(self.boundaries) + 1)
        self.total_samples = 0
        self.min_latency = float("inf")
        self.max_latency = 0.0
        self.latency_sum = 0.0

    def record_latency(self, latency_ms: float):
        """Records a single latency observation in O(log B) time."""
        val = max(0.0, float(latency_ms))
        self.total_samples += 1
        self.latency_sum += val
        self.min_latency = min(self.min_latency, val)
        self.max_latency = max(self.max_latency, val)

        idx = bisect.bisect_right(self.boundaries, val)
        self.bucket_counts[idx] += 1

    def calculate_percentile(self, q: float) -> float:
        """
        Estimates the q-th quantile (0.0 to 1.0) using linear interpolation
        across histogram bucket ranks.
        """
        if self.total_samples == 0:
            return 0.0
        if not (0.0 <= q <= 1.0):
            raise ValueError("Quantile must be between 0.0 and 1.0.")

        target_rank = q * self.total_samples
        cumulative = 0

        for i, count in enumerate(self.bucket_counts):
            cumulative += count
            if cumulative >= target_rank:
                lower_bound = 0.0 if i == 0 else self.boundaries[i - 1]
                upper_bound = self.boundaries[i] if i < len(self.boundaries) else max(self.max_latency, lower_bound + 10.0)
                count_in_bucket = count
                rank_within_bucket = target_rank - (cumulative - count)
                fraction = rank_within_bucket / count_in_bucket if count_in_bucket > 0 else 0.0

                return round(lower_bound + (fraction * (upper_bound - lower_bound)), 2)

        return round(self.max_latency, 2)

    def get_summary(self) -> Dict[str, float]:
        """Returns structured statistical summary and key SLA percentiles."""
        avg_latency = (self.latency_sum / self.total_samples) if self.total_samples > 0 else 0.0
        return {
            "samples_count": self.total_samples,
            "min_ms": round(self.min_latency if self.total_samples > 0 else 0.0, 2),
            "max_ms": round(self.max_latency, 2),
            "mean_ms": round(avg_latency, 2),
            "p50_ms": self.calculate_percentile(0.50),
            "p90_ms": self.calculate_percentile(0.90),
            "p95_ms": self.calculate_percentile(0.95),
            "p99_ms": self.calculate_percentile(0.99)
        }

    def verify_sla(self, max_p95_ms: float = 50.0, max_p99_ms: float = 100.0) -> Tuple[bool, List[str]]:
        """Audits current tail latencies against target SLA budget thresholds."""
        summary = self.get_summary()
        violations = []

        if summary["p95_ms"] > max_p95_ms:
            violations.append(f"p95 breached: {summary['p95_ms']}ms > {max_p95_ms}ms target")
        if summary["p99_ms"] > max_p99_ms:
            violations.append(f"p99 breached: {summary['p99_ms']}ms > {max_p99_ms}ms target")

        return len(violations) == 0, violations


if __name__ == "__main__":
    print("--- MLOps Telemetry: Streaming Latency Histogram & SLA Tracker ---\n")

    tracker = LatencyHistogramTracker()

    print("Step 1: Recording 1,000 synthetic inference request latencies:")
    for i in range(950):
        sample = 12.0 + (i % 13) + (i % 4) * 0.5
        tracker.record_latency(sample)

    for i in range(50):
        outlier = 55.0 + (i * 3.2)
        tracker.record_latency(outlier)

    summary = tracker.get_summary()
    print("Latency Distribution Summary:")
    print(json.dumps(summary, indent=2))

    print("\nStep 2: Evaluating Service Level Agreement (SLA) Compliance:")
    sla_passed, violations = tracker.verify_sla(max_p95_ms=40.0, max_p99_ms=180.0)
    print(f"SLA Audit Verdict: {'COMPLIANT' if sla_passed else 'NON-COMPLIANT'}")
    if violations:
        for v in violations:
            print(f"{v}")
    print("\nStep 3: Simulating downstream network outage / tail blowout (100 spikes):")
    for _ in range(100):
        tracker.record_latency(350.0)

    updated_summary = tracker.get_summary()
    sla_passed_spike, violations_spike = tracker.verify_sla(max_p95_ms=40.0, max_p99_ms=180.0)
    print(f"  Updated p99: {updated_summary['p99_ms']}ms")
    print(f"  SLA Audit Post-Spike: {'COMPLIANT' if sla_passed_spike else 'BREACHED'}")
    for v in violations_spike:
        print(f"{v}")