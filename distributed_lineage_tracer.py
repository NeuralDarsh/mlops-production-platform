# MLOps Observability & Tracing: Asynchronous contextvars propagation, span trees, and trace lineage auditing

import asyncio
import contextvars
import json
import time
import uuid
from typing import Any, Dict, List, Optional

_CURRENT_SPAN: contextvars.ContextVar[Optional["Span"]] = contextvars.ContextVar("current_span", default=None)


class Span:
    """Represents a discrete timed execution block within a distributed trace."""
    def __init__(self, name: str, trace_id: str, parent_id: Optional[str] = None):
        self.span_id = f"span_{uuid.uuid4().hex[:8]}"
        self.trace_id = trace_id
        self.parent_id = parent_id
        self.name = name
        self.start_time = 0.0
        self.end_time = 0.0
        self.attributes: Dict[str, Any] = {}
        self.events: List[Dict[str, Any]] = []
        self.error: Optional[str] = None

    def set_attribute(self, key: str, value: Any):
        self.attributes[key] = value

    def add_event(self, name: str, payload: Optional[Dict[str, Any]] = None):
        self.events.append({
            "name": name,
            "timestamp": round(time.time(), 4),
            "payload": payload or {}
        })

    def to_dict(self) -> Dict[str, Any]:
        duration_ms = (self.end_time - self.start_time) * 1000.0 if self.end_time else 0.0
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "duration_ms": round(duration_ms, 2),
            "attributes": self.attributes,
            "events": self.events,
            "error": self.error
        }


class LineageTracer:
    """
    Context manager and tracer coordinating trace propagation across asynchronous boundaries.
    """
    def __init__(self):
        self.completed_spans: List[Dict[str, Any]] = []

    def start_span(self, name: str, trace_id: Optional[str] = None) -> "SpanContextManager":
        return SpanContextManager(self, name, trace_id)


class SpanContextManager:
    """Manages span lifecycle and context variable binding."""
    def __init__(self, tracer: LineageTracer, name: str, explicit_trace_id: Optional[str] = None):
        self.tracer = tracer
        self.name = name
        self.explicit_trace_id = explicit_trace_id
        self.span: Optional[Span] = None
        self.token: Optional[contextvars.Token] = None

    async def __aenter__(self) -> Span:
        parent_span = _CURRENT_SPAN.get()
        if parent_span:
            trace_id = parent_span.trace_id
            parent_id = parent_span.span_id
        else:
            trace_id = self.explicit_trace_id or f"trace_{uuid.uuid4().hex[:12]}"
            parent_id = None

        self.span = Span(name=self.name, trace_id=trace_id, parent_id=parent_id)
        self.span.start_time = time.time()
        self.token = _CURRENT_SPAN.set(self.span)
        return self.span

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.span:
            self.span.end_time = time.time()
            if exc_val:
                self.span.error = str(exc_val)
            self.tracer.completed_spans.append(self.span.to_dict())
        if self.token:
            _CURRENT_SPAN.reset(self.token)


if __name__ == "__main__":
    print("--- MLOps Observability: Distributed Context Propagation & Lineage Tracer ---\n")

    tracer = LineageTracer()

    async def fetch_feature_vector(entity_id: str) -> List[float]:
        async with tracer.start_span("FeatureStore.FetchAsOf") as span:
            span.set_attribute("entity_id", entity_id)
            await asyncio.sleep(0.02)
            features = [1.25, 0.45, 9.80]
            span.add_event("features_resolved", {"dimensions": len(features)})
            return features

    async def run_model_forward_pass(features: List[float]) -> Dict[str, Any]:
        async with tracer.start_span("InferenceEngine.ForwardPass") as span:
            span.set_attribute("batch_size", 1)
            span.set_attribute("precision", "float32")
            await asyncio.sleep(0.03)
            prediction = {"score": 0.942, "class": "positive"}
            span.add_event("tensor_evaluated", prediction)
            return prediction

    async def execute_inference_pipeline(request_id: str, entity_id: str):
 
        async with tracer.start_span("API.PredictEndpoint") as root_span:
            root_span.set_attribute("request_id", request_id)
            print(f"[ROOT SPAN STARTED] TraceID: {root_span.trace_id} | Name: {root_span.name}")

            features = await fetch_feature_vector(entity_id)

            result = await run_model_forward_pass(features)

            root_span.set_attribute("prediction_class", result["class"])
            return result

    async def main():
        print("Running end-to-end traced inference request:")
        output = await execute_inference_pipeline(
            request_id="client_req_tokyo_001",
            entity_id="user_japan_44"
        )
        print(f"Pipeline Execution Finished. Result: {output}\n")

        print("Distributed Trace Spans Hierarchy (Audit Log):")
        print(json.dumps(tracer.completed_spans, indent=2))

    asyncio.run(main())