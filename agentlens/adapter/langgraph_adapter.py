"""The AgentLens entry point for LangGraph apps.

    trace_id, final_state = traced_invoke(app, initial)

Spans flow to Phoenix (or the configured OTel backend) automatically.
Call setup_otel() and instrument_langchain() once before using this.

What this function does:
  - Wraps the full run in a root OTel span (agentlens.run)
  - Streams the graph to detect node boundaries
  - Emits an AGENT span per node (with produced state + duration)
  - Emits a HANDOFF span at each node transition
  - LLM + tool spans are captured automatically by LangChainInstrumentor
"""

from __future__ import annotations

import time
from typing import Any, Optional

from opentelemetry import context as otel_context

from agentlens.adapter.otel import emit_agent, emit_handoff, get_tracer


def traced_invoke(
    app: Any,
    initial: dict[str, Any],
    session_id: str = "default",
    framework: str = "langgraph",
    recursion_limit: int = 50,
) -> tuple[str, dict[str, Any]]:
    """Run a compiled LangGraph app under AgentLens observation.

    Returns:
        (trace_id, final_state) — trace_id is the OTel W3C hex trace ID.

    Requires setup_otel() and instrument_langchain() to have been called first.
    """
    tracer = get_tracer()

    with tracer.start_as_current_span("agentlens.run") as root_span:
        ctx = root_span.get_span_context()
        trace_id = format(ctx.trace_id, "032x")

        root_span.set_attribute("openinference.span.kind", "AGENT")
        root_span.set_attribute("agentlens.query", initial.get("query", ""))
        root_span.set_attribute("agentlens.session_id", session_id)
        root_span.set_attribute("agentlens.framework", framework)

        final_state: dict[str, Any] = dict(initial)
        prev_node: Optional[str] = None
        prev_time_ns: int = time.time_ns()
        config = {"recursion_limit": recursion_limit}

        for chunk in app.stream(initial, config=config, stream_mode="updates"):
            now_ns = time.time_ns()
            for node, update in chunk.items():
                duration_ms = (now_ns - prev_time_ns) / 1_000_000

                if prev_node is not None:
                    emit_handoff(prev_node, node)

                emit_agent(node, update, duration_ms, prev_time_ns, now_ns)

                if isinstance(update, dict):
                    final_state.update(update)

                prev_node = node
                prev_time_ns = now_ns

        root_span.set_attribute(
            "agentlens.output",
            str(final_state.get("report", ""))[:500],
        )

    return trace_id, final_state
