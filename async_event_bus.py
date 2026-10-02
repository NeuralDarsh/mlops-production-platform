# Distributed Systems & Messaging: Non-blocking pub/sub event bus with wildcard routing and subscriber error boundaries

import asyncio
import fnmatch
import json
import time
import uuid

class Event:
    """Represents an immutable event message dispatched across the bus."""
    def __init__(self, topic, payload):
        self.event_id = f"evt_{uuid.uuid4().hex[:8]}"
        self.topic = topic
        self.payload = payload
        self.timestamp = time.time()

    def to_dict(self):
        return {
            "event_id": self.event_id,
            "topic": self.topic,
            "payload": self.payload,
            "timestamp": round(self.timestamp, 4)
        }


class AsyncEventBus:
    """
    Asynchronous in-memory event bus supporting wildcard topic routing,
    concurrent consumer dispatch, and isolated subscriber error handling.
    """
    def __init__(self, maxsize=100):
        self.queue = asyncio.Queue(maxsize=maxsize)
        self.subscribers = {}  # pattern -> list of async callback functions
        self.running = False
        self.worker_task = None
        self.dispatched_count = 0

    def subscribe(self, pattern, callback):
        """Registers an asynchronous callback to a topic pattern (supports wildcards like 'order.*')."""
        if pattern not in self.subscribers:
            self.subscribers[pattern] = []
        self.subscribers[pattern].append(callback)
        print(f"Subscribed handler '{callback.__name__}' to pattern '{pattern}'")

    async def start(self):
        """Starts the background event dispatch loop."""
        self.running = True
        self.worker_task = asyncio.create_task(self._dispatch_loop())
        print("[EVENT BUS] Dispatch worker started.")

    async def publish(self, topic, payload):
        """Pushes an event into the queue non-blockingly."""
        event = Event(topic, payload)
        await self.queue.put(event)
        print(f"  [PUBLISHED] Topic='{topic}' | ID={event.event_id}")
        return event

    async def _safe_execute_subscriber(self, callback, event):
        """Executes a subscriber inside an isolated try/except error boundary."""
        try:
            await callback(event)
        except Exception as exc:
            print(f"[HANDLER ERROR] '{callback.__name__}' failed on event '{event.event_id}': {exc}")

    async def _dispatch_loop(self):
        """Continuously pulls events from the queue and dispatches to matching handlers."""
        while self.running or not self.queue.empty():
            try:
                event = await asyncio.wait_for(self.queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue

            matching_handlers = []
            for pattern, handlers in self.subscribers.items():
                if fnmatch.fnmatch(event.topic, pattern):
                    matching_handlers.extend(handlers)

            if matching_handlers:
                # Concurrent execution of matching subscribers
                await asyncio.gather(
                    *(self._safe_execute_subscriber(h, event) for h in matching_handlers)
                )

            self.dispatched_count += 1
            self.queue.task_done()

    async def stop(self):
        """Drains remaining events and cleanly shuts down the dispatch loop."""
        print("\n[EVENT BUS] Draining queue and initiating shutdown...")
        await self.queue.join()
        self.running = False
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
        print(f"[EVENT BUS] Safely stopped. Total events dispatched: {self.dispatched_count}\n")


if __name__ == "__main__":
    print("--- Systems Architecture: Asynchronous Non-Blocking Event Bus ---\n")

    async def main():
        bus = AsyncEventBus()
        await bus.start()

        # Define microservice handlers
        async def order_processor(event):
            await asyncio.sleep(0.02)
            print(f"[OrderService] Processed order payload: {event.payload}")

        async def audit_logger(event):
            print(f" [AuditService] Audit logged topic '{event.topic}' (Event ID: {event.event_id})")

        async def faulty_analytics_tracker(event):
            # Simulates a failing subscriber
            raise RuntimeError("Connection dropped to analytics database!")

        # Wire up subscriptions
        bus.subscribe("order.*", order_processor)
        bus.subscribe("*", audit_logger)
        bus.subscribe("order.checkout", faulty_analytics_tracker)

        print("\nPublishing sample events across topics:")
        await bus.publish("order.created", {"order_id": "ORD-7001", "amount": 250})
        await bus.publish("order.checkout", {"order_id": "ORD-7002", "amount": 1200})
        await bus.publish("auth.user_login", {"user_id": "usr_tokyo_44"})

        # Graceful drain and shutdown
        await bus.stop()

    asyncio.run(main())