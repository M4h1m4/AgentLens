"""OTel setup and span emission for AgentLens.

Two things live here:
  1. setup_otel() — configures the TracerProvider once at startup
  2. Span emitters — emit AGENT and HANDOFF spans from the stream

LLM and tool spans are captured automatically by LangChainInstrumentor
(openinference-instrumentation-langchain). AgentLens only adds what the
auto-instrumentor cannot see: node-level state and lateral handoffs between
sibling agents.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

AGENTLENS_TRACER_NAME = "agentlens"
PHOENIX_OTLP_ENDPOINT = "http://localhost:6006/v1/traces"


def setup_otel(
    endpoint: str | None = None,
    exporter: Any = None,
) -> TracerProvider:
    """Configure the global OTel TracerProvider.

    Call once at startup (CLI entry point or test fixture).

    Args:
        endpoint: OTLP HTTP endpoint. Defaults to Phoenix at localhost:6006.
                  Reads AGENTLENS_OTLP_ENDPOINT env var if set.
        exporter: Inject a custom exporter (e.g. InMemorySpanExporter for tests).
                  When provided, ``endpoint`` is ignored.

    Returns:
        The configured TracerProvider.
    """
    provider = TracerProvider()

    if exporter is None:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        ep = endpoint or os.environ.get("AGENTLENS_OTLP_ENDPOINT", PHOENIX_OTLP_ENDPOINT)
        exporter = OTLPSpanExporter(endpoint=ep)

    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return provider


def get_tracer() -> trace.Tracer:
    """Return the AgentLens OTel tracer."""
    return trace.get_tracer(AGENTLENS_TRACER_NAME)


def instrument_langchain() -> None:
    """Auto-instrument LangChain/LangGraph. Idempotent — safe to call multiple times."""
    from openinference.instrumentation.langchain import LangChainInstrumentor
    LangChainInstrumentor().instrument()


def _jsonable(value: Any) -> Any:
    """Coerce arbitrary values into something JSON-serialisable."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def emit_agent(
    node: str,
    update: Any,
    duration_ms: float,
    start_time_ns: int,
    end_time_ns: int,
) -> None:
    """Emit an AGENT span recording one node's execution and produced state."""
    tracer = get_tracer()
    span = tracer.start_span(
        f"agentlens.agent.{node}",
        start_time=start_time_ns,
    )
    span.set_attribute("openinference.span.kind", "AGENT")
    span.set_attribute("agentlens.node", node)
    span.set_attribute("agentlens.duration_ms", duration_ms)
    try:
        produced = json.dumps(_jsonable(update))[:4000]  # cap at 4KB per span
    except Exception:
        produced = str(update)[:4000]
    span.set_attribute("agentlens.produced", produced)
    span.end(end_time=end_time_ns)


def emit_handoff(source: str, target: str) -> None:
    """Emit a HANDOFF span as a child of the current active span."""
    tracer = get_tracer()
    with tracer.start_as_current_span("agentlens.handoff") as span:
        span.set_attribute("openinference.span.kind", "HANDOFF")
        span.set_attribute("handoff.source", source)
        span.set_attribute("handoff.target", target)
