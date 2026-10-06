# MLOps & High-Throughput Serving: Dynamic batching with SLA timeouts, vectorized batch evaluation, and future demultiplexing

import asyncio
import time
import uuid
from typing import Any, List

class InferenceRequest:
    """Encapsulates a single client inference payload and its asynchronous response handle."""
    def __init__(self, features: List[float]):
        self.request_id = f"req_{uuid.uuid4().hex[:6]}"
        self.features = features
        self.arrival_time = time.time()
        self.future = asyncio.get_running_loop().create_future()


class DynamicBatchScheduler:
    """
    Asynchronous micro-batch scheduler optimizing inference throughput.
    Batches evaluate when either max_batch_size is reached or max_latency_ms expires.
    """
    def __init__(self, max_batch_size: int = 4, max_latency_ms: float = 40.0):
        self.max_batch_size = max_batch_size
        self.max_latency_sec = max_latency_ms / 1000.0
        self.queue: asyncio.Queue[InferenceRequest] = asyncio.Queue()
        self.running = False
        self.worker_task: asyncio.Task | None = None
        self.batches_processed = 0

    async def start(self):
        """Starts the dynamic micro-batch dispatcher loop."""
        self.running = True
        self.worker_task = asyncio.create_task(self._batch_dispatcher_loop())
        print("[DYNAMIC SCHEDULER] Inference batch worker active.")

    async def predict(self, features: List[float]) -> Any:
        """Client API: Submits single input vector and awaits batched evaluation."""
        req = InferenceRequest(features)
        await self.queue.put(req)
        return await req.future

    async def _mock_vectorized_model_inference(self, batch_matrix: List[List[float]]) -> List[dict]:
        """
        Simulates vectorized batch model execution (e.g., PyTorch/ONNX matrix multiply).
        Amortizes model overhead across all rows in a single forward pass.
        """
        await asyncio.sleep(0.02)  # Simulated vectorized compute latency (20ms)
        results = []
        for row in batch_matrix:
            # Simple linear model mock: score = sigmoid(sum(x))
            val = sum(row)
            score = 1.0 / (1.0 + 2.71828 ** (-val))
            prediction = "positive" if score >= 0.5 else "negative"
            results.append({"prediction": prediction, "confidence": round(score, 4)})
        return results

    async def _batch_dispatcher_loop(self):
        """Monitors queue and flushes batches based on size capacity or SLA window expiration."""
        while self.running:
            batch: List[InferenceRequest] = []
            
            # Wait for at least one item
            try:
                first_item = await asyncio.wait_for(self.queue.get(), timeout=0.1)
                batch.append(first_item)
            except asyncio.TimeoutError:
                continue

            deadline = time.time() + self.max_latency_sec

            # Opportunistically assemble subsequent items up to max_batch_size or deadline
            while len(batch) < self.max_batch_size:
                timeout_remaining = deadline - time.time()
                if timeout_remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self.queue.get(), timeout=timeout_remaining)
                    batch.append(item)
                except asyncio.TimeoutError:
                    break

            if not batch:
                continue

            # Process assembled micro-batch
            self.batches_processed += 1
            batch_inputs = [r.features for r in batch]
            print(f"\n[BATCH #{self.batches_processed} DISPATCH] Size: {len(batch)} items | SLA Wait: {round((time.time() - batch[0].arrival_time) * 1000, 2)}ms")

            # Single vectorized forward pass
            batch_outputs = await self._mock_vectorized_model_inference(batch_inputs)

            # Scatter / Demultiplex outputs back to individual client futures
            for req, output in zip(batch, batch_outputs):
                req.future.set_result(output)
                self.queue.task_done()

    async def stop(self):
        """Flushes remaining items and stops dispatcher."""
        await self.queue.join()
        self.running = False
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
        print(f"\n[SCHEDULER STOPPED] Processed {self.batches_processed} total batches.")


if __name__ == "__main__":
    print("--- MLOps Serving Infrastructure: Dynamic Micro-Batch Scheduler ---\n")

    async def main():
        scheduler = DynamicBatchScheduler(max_batch_size=4, max_latency_ms=50.0)
        await scheduler.start()

        async def client_request_task(client_id: str, inputs: List[float], delay: float):
            await asyncio.sleep(delay)
            t0 = time.time()
            res = await scheduler.predict(inputs)
            latency = (time.time() - t0) * 1000
            print(f" Client [{client_id}] finished in {round(latency, 2)}ms -> Result: {res}")

        # Simulate concurrent client queries arriving with small jitters
        tasks = [
            client_request_task("Client_A", [0.5, 1.2, -0.3], delay=0.00),
            client_request_task("Client_B", [-0.8, -1.1, 0.2], delay=0.01),
            client_request_task("Client_C", [1.0, 0.4, 0.9], delay=0.02),
            client_request_task("Client_D", [0.1, -0.2, 0.4], delay=0.03),  # Triggers max_batch_size=4
            client_request_task("Client_E", [-0.2, 0.5, 0.1], delay=0.08),  # Triggers by SLA timeout
        ]

        await asyncio.gather(*tasks)
        await scheduler.stop()

    asyncio.run(main())