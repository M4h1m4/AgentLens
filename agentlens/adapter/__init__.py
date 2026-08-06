"""AgentLens universal adapter — OTel-native trace capture for LangGraph.

Quickstart::

    from agentlens.adapter import setup_otel, instrument_langchain, traced_invoke
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    # For tests — use in-memory exporter
    exporter = InMemorySpanExporter()
    setup_otel(exporter=exporter)
    instrument_langchain()

    trace_id, state = traced_invoke(app, {"query": "..."})
    spans = exporter.get_finished_spans()

    # For production — point at Phoenix (default)
    setup_otel()   # exports to http://localhost:6006/v1/traces
    instrument_langchain()
    trace_id, state = traced_invoke(app, {"query": "..."})
"""

from agentlens.adapter.otel import (
    PHOENIX_OTLP_ENDPOINT,
    emit_agent,
    emit_handoff,
    get_tracer,
    instrument_langchain,
    setup_otel,
)
from agentlens.adapter.langgraph_adapter import traced_invoke
from agentlens.adapter.store import InMemorySpanExporter, PhoenixTraceReader

__all__ = [
    "setup_otel",
    "instrument_langchain",
    "get_tracer",
    "emit_agent",
    "emit_handoff",
    "traced_invoke",
    "InMemorySpanExporter",
    "PhoenixTraceReader",
    "PHOENIX_OTLP_ENDPOINT",
]
