# MLOps Serving Infrastructure: Active heartbeat probing, dynamic least-loaded routing, and automatic quarantine

import time
import uuid
import json
from typing import Any, Dict, List, Optional


class WorkerStatus:
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"


class ModelWorkerNode:
    """Represents an individual inference worker instance executing ML models."""
    def __init__(self, worker_id: str, capacity: int = 10):
        self.worker_id = worker_id
        self.capacity = capacity
        self.active_requests = 0
        self.total_processed = 0
        self.status = WorkerStatus.HEALTHY
        self.last_heartbeat = time.time()
        self.latencies: List[float] = []

    def get_average_latency_ms(self) -> float:
        if not self.latencies:
            return 10.0  
        recent = self.latencies[-10:]
        return sum(recent) / len(recent)

    def calculate_load_score(self) -> float:
        """Lower score indicates higher suitability to receive the next query."""
        if self.status != WorkerStatus.HEALTHY:
            return float("inf")
        saturation = self.active_requests / self.capacity
        latency_factor = self.get_average_latency_ms() / 100.0
        return (0.7 * saturation) + (0.3 * latency_factor)


class ModelFleetSupervisor:
    """
    Supervises a pool of inference worker nodes, actively monitoring liveness
    and dispatching requests via adaptive least-loaded balancing.
    """
    def __init__(self, heartbeat_timeout_sec: float = 2.0):
        self.heartbeat_timeout_sec = heartbeat_timeout_sec
        self.workers: Dict[str, ModelWorkerNode] = {}

    def register_worker(self, worker_id: str, capacity: int = 10) -> ModelWorkerNode:
        worker = ModelWorkerNode(worker_id=worker_id, capacity=capacity)
        self.workers[worker_id] = worker
        print(f"[REGISTERED] Model Worker: '{worker_id}' (Capacity: {capacity})")
        return worker

    def send_heartbeat(self, worker_id: str):
        if worker_id in self.workers:
            self.workers[worker_id].last_heartbeat = time.time()
            if self.workers[worker_id].status == WorkerStatus.UNHEALTHY:
                self.workers[worker_id].status = WorkerStatus.HEALTHY
                print(f"[RECOVERED] Worker '{worker_id}' restored to HEALTHY.")

    def run_health_check_sweep(self):
        """Scans the fleet and marks non-responsive workers as UNHEALTHY."""
        now = time.time()
        for worker in self.workers.values():
            if now - worker.last_heartbeat > self.heartbeat_timeout_sec:
                if worker.status != WorkerStatus.UNHEALTHY:
                    worker.status = WorkerStatus.UNHEALTHY
                    print(f"[QUARANTINE] Worker '{worker.worker_id}' missed heartbeats -> Marked UNHEALTHY.")

    def select_best_worker(self) -> Optional[ModelWorkerNode]:
        """Adaptive load balancer selecting the healthiest worker with lowest load score."""
        self.run_health_check_sweep()
        available_workers = [w for w in self.workers.values() if w.status == WorkerStatus.HEALTHY]
        if not available_workers:
            return None

        return min(available_workers, key=lambda w: w.calculate_load_score())

    def dispatch_inference(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatches an inference query to the optimal active worker node."""
        worker = self.select_best_worker()
        if not worker:
            return {"error": "503 Service Unavailable: No healthy model workers active"}

        worker.active_requests += 1
        t0 = time.time()
        time.sleep(0.02)
        elapsed_ms = (time.time() - t0) * 1000.0

        worker.active_requests = max(0, worker.active_requests - 1)
        worker.total_processed += 1
        worker.latencies.append(elapsed_ms)

        return {
            "worker_id": worker.worker_id,
            "prediction": "positive_signal",
            "latency_ms": round(elapsed_ms, 2),
            "status": "success"
        }


if __name__ == "__main__":
    print("--- MLOps Fleet Management: Model Worker Supervisor & Adaptive Balancer ---\n")

    supervisor = ModelFleetSupervisor(heartbeat_timeout_sec=0.8)

    # 1. Register 3 model inference nodes
    w1 = supervisor.register_worker("worker-gpu-01", capacity=5)
    w2 = supervisor.register_worker("worker-gpu-02", capacity=5)
    w3 = supervisor.register_worker("worker-gpu-03", capacity=5)

    # 2. Dispatch queries across balanced fleet
    print("\nStep 1: Dispatching 6 incoming queries across healthy fleet:")
    for i in range(1, 7):
        res = supervisor.dispatch_inference({"features": [0.1, 0.4, 0.9]})
        print(f"  Req #{i} -> Handled by: {res.get('worker_id')} ({res.get('latency_ms')}ms)")

    # 3. Simulate failure: worker-gpu-02 stops emitting heartbeats
    print("\nStep 2: Simulating heartbeat failure on 'worker-gpu-02'...")
    supervisor.send_heartbeat("worker-gpu-01")
    supervisor.send_heartbeat("worker-gpu-03")

    time.sleep(1.0)

    print("\nStep 3: Dispatching queries while 'worker-gpu-02' is quarantined:")
    for i in range(7, 10):
        res = supervisor.dispatch_inference({"features": [0.5, 0.2, 0.8]})
        print(f"  Req #{i} -> Handled by: {res.get('worker_id')}")

    # 4. Recover worker-gpu-02
    print("\nStep 4: Re-transmitting heartbeat for 'worker-gpu-02':")
    supervisor.send_heartbeat("worker-gpu-02")
    res_post_recovery = supervisor.dispatch_inference({"features": [0.3, 0.6, 0.7]})
    print(f"  Query after recovery handled by: {res_post_recovery.get('worker_id')}")

    print("\nFinal Fleet State & Processing Metrics:")
    fleet_summary = {
        w.worker_id: {
            "status": w.status,
            "total_processed": w.total_processed,
            "avg_latency_ms": round(w.get_average_latency_ms(), 2)
        }
        for w in supervisor.workers.values()
    }
    print(json.dumps(fleet_summary, indent=2))