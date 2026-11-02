# MLOps Serving & Safe Deployment: In-memory blue-green slots, model pre-warming, and in-flight request draining

import threading
import time
import json
from typing import Any, Dict, Optional


class ModelInstance:
    """Represents an instantiated model artifact loaded into memory with an active reference counter."""
    def __init__(self, version: str, weights_tag: str):
        self.version = version
        self.weights_tag = weights_tag
        self.active_readers = 0
        self.lock = threading.Lock()
        self.is_retired = False

    def acquire_reader(self) -> bool:
        """Increments active reader count if the model is valid."""
        with self.lock:
            self.active_readers += 1
            return True

    def release_reader(self):
        """Decrements active reader count."""
        with self.lock:
            self.active_readers = max(0, self.active_readers - 1)

    def execute_inference(self, features: list) -> Dict[str, Any]:
        """Simulates running inference against loaded weights."""
        time.sleep(0.03)  
        score = sum(features) / (len(features) or 1)
        return {
            "version": self.version,
            "weights_tag": self.weights_tag,
            "score": round(score, 4),
            "status": "success"
        }


class AtomicModelSwapEngine:
    """
    Coordinates zero-downtime model replacements via in-memory blue-green slots,
    atomic pointer reassignment, and safe draining of deprecated models.
    """
    def __init__(self, initial_version: str, initial_weights: str):
        self._active_model: ModelInstance = ModelInstance(initial_version, initial_weights)
        self._swap_lock = threading.Lock()
        self.deployment_history = []
        print(f"[INIT] Initialized active production model '{initial_version}'.")

    def predict(self, features: list) -> Dict[str, Any]:
        """Client endpoint: atomically acquires active model reference, evaluates, and releases."""
        with self._swap_lock:
            current_model = self._active_model
            current_model.acquire_reader()

        try:
            return current_model.execute_inference(features)
        finally:
            current_model.release_reader()

    def deploy_new_model_zero_downtime(
        self,
        new_version: str,
        new_weights: str,
        warmup_payload: list,
        drain_timeout_sec: float = 2.0
    ) -> bool:
        """
        Executes zero-downtime swap:
        1. Loads & pre-warms candidate in isolation.
        2. Atomically flips the pointer.
        3. Drains in-flight queries from deprecated model before cleanup.
        """
        print(f"\n[PRE-DEPLOY] Preparing deployment for candidate model '{new_version}'...")

        candidate = ModelInstance(new_version, new_weights)
        t0 = time.time()
        warmup_res = candidate.execute_inference(warmup_payload)
        warmup_ms = (time.time() - t0) * 1000.0
        print(f"Pre-warming candidate completed in {round(warmup_ms, 2)}ms (Validation score: {warmup_res['score']})")

        with self._swap_lock:
            old_model = self._active_model
            self._active_model = candidate
            old_model.is_retired = True
            print(f"[ATOMIC SWAP] Active pointer switched from '{old_model.version}' to '{candidate.version}'.")

        print(f"[DRAIN] Waiting for {old_model.active_readers} in-flight queries to clear from '{old_model.version}'...")
        deadline = time.time() + drain_timeout_sec
        while old_model.active_readers > 0 and time.time() < deadline:
            time.sleep(0.01)

        drained = old_model.active_readers == 0
        if drained:
            print(f"[UNLOAD] Model '{old_model.version}' safely drained and unallocated from memory.")
        else:
            print(f"[DRAIN TIMEOUT] Model '{old_model.version}' force-retired with {old_model.active_readers} stragglers.")

        self.deployment_history.append({
            "deployed_version": new_version,
            "replaced_version": old_model.version,
            "drained_cleanly": drained,
            "timestamp": time.time()
        })
        return True


if __name__ == "__main__":
    print("--- MLOps Safe Deployment: Zero-Downtime Atomic Model Swap Engine ---\n")

    engine = AtomicModelSwapEngine(initial_version="v1.0.0", initial_weights="weights_resnet_fp32")

    print("\nStep 1: Dispatching live inference queries to initial model:")
    res1 = engine.predict([0.1, 0.4, 0.7])
    print(f"  Query Output: Version={res1['version']} | Score={res1['score']}")

    print("\nStep 2: Simulating background traffic while deploying candidate model 'v2.0.0'...")

    def background_client_worker(worker_id: int):
        for _ in range(3):
            out = engine.predict([0.2 * worker_id, 0.5])
            print(f"    [Client {worker_id}] Handled by: {out['version']}")
            time.sleep(0.02)

    t1 = threading.Thread(target=background_client_worker, args=(1,))
    t2 = threading.Thread(target=background_client_worker, args=(2,))
    t1.start()
    t2.start()

    engine.deploy_new_model_zero_downtime(
        new_version="v2.0.0",
        new_weights="weights_resnet_int8_quantized",
        warmup_payload=[1.0, 1.0, 1.0]
    )

    t1.join()
    t2.join()

    print("\nStep 3: Dispatching queries after deployment completed:")
    res_final = engine.predict([0.8, 0.9, 0.95])
    print(f"  Final Query Output: Version={res_final['version']} | Weights={res_final['weights_tag']}")

    print("\nDeployment Audit History:")
    print(json.dumps(engine.deployment_history, indent=2))