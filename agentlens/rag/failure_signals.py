"""Failure signal extraction from OTel spans.

A FailureSignal is a structured + natural-language summary of what a trace
showed. It is the unit that gets embedded and stored in the failure library.

Extraction is best-effort and framework-agnostic:
- Agent loop counts come from counting AGENT span occurrences per node
- Context drop detection comes from comparing produced output sizes across handoffs
- Tool usage comes from TOOL spans
- The summary is a natural-language paragraph suitable for semantic embedding
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class FailureSignal:
    """Structured representation of what one trace run showed."""

    trace_id: str
    session_id: str = "default"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Structural facts extracted from spans
    agents_observed: list[str] = field(default_factory=list)
    handoffs: list[tuple[str, str]] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)

    # Quantitative signals
    loop_counts: dict[str, int] = field(default_factory=dict)
    # agent_name -> how many times that node's AGENT span appeared

    produced_sizes: dict[str, int] = field(default_factory=dict)
    # agent_name -> byte length of agentlens.produced JSON (proxy for context size)

    # Derived failure indicators
    context_drop_detected: bool = False
    # True if produced_size decreases by >20% across a handoff

    delegation_loop_detected: bool = False
    # True if any agent ran more than once

    # Natural language summary — this is what gets embedded
    summary: str = ""

    # Post-diagnosis fields — filled in after the Diagnosis Agent runs
    diagnosis: str | None = None
    fix_applied: str | None = None
    pass_rate_delta: float | None = None  # pass_rate_after - pass_rate_before


# ── public API ────────────────────────────────────────────────────────────────

def extract_signals(
    spans: list,
    trace_id: str = "",
    session_id: str = "default",
) -> FailureSignal:
    """Extract a FailureSignal from a list of OTel spans (or span proxies).

    Works with ReadableSpan objects from InMemorySpanExporter and with
    _SpanProxy objects from IntakePhoenixReader — anything with .name
    and .attributes.

    Args:
        spans:      OTel spans from one complete run.
        trace_id:   The run's trace ID (hex string from traced_invoke).
        session_id: The session identifier.
    """
    agents = _agent_order(spans)
    handoffs = _handoffs(spans)
    tools = _tools(spans)
    loop_counts = _loop_counts(spans)
    produced_sizes = _produced_sizes(spans)
    context_drop = _detect_context_drop(handoffs, produced_sizes)
    delegation_loop = any(count > 1 for count in loop_counts.values())

    summary = _build_summary(
        agents, handoffs, tools, loop_counts, produced_sizes,
        context_drop, delegation_loop,
    )

    return FailureSignal(
        trace_id=trace_id,
        session_id=session_id,
        agents_observed=agents,
        handoffs=handoffs,
        tools_used=tools,
        loop_counts=loop_counts,
        produced_sizes=produced_sizes,
        context_drop_detected=context_drop,
        delegation_loop_detected=delegation_loop,
        summary=summary,
    )


# ── extraction helpers ────────────────────────────────────────────────────────

def _attrs(span) -> dict:
    raw = getattr(span, "attributes", None) or {}
    return dict(raw)


def _agent_order(spans: list) -> list[str]:
    """Return unique agent node names in the order they first appeared."""
    seen: list[str] = []
    for span in spans:
        attrs = _attrs(span)
        if attrs.get("openinference.span.kind") == "HANDOFF":
            for key in ("handoff.source", "handoff.target"):
                name = attrs.get(key)
                if name and str(name) not in seen:
                    seen.append(str(name))
    return seen


def _handoffs(spans: list) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    for span in spans:
        attrs = _attrs(span)
        if attrs.get("openinference.span.kind") == "HANDOFF":
            src = attrs.get("handoff.source")
            tgt = attrs.get("handoff.target")
            if src and tgt:
                result.append((str(src), str(tgt)))
    return result


def _tools(spans: list) -> list[str]:
    seen: list[str] = []
    for span in spans:
        attrs = _attrs(span)
        if attrs.get("openinference.span.kind") == "TOOL":
            tool = attrs.get("tool.name") or getattr(span, "name", None)
            if tool and str(tool) not in seen:
                seen.append(str(tool))
    return seen


def _loop_counts(spans: list) -> dict[str, int]:
    counts: dict[str, int] = {}
    for span in spans:
        attrs = _attrs(span)
        if attrs.get("openinference.span.kind") == "AGENT":
            node = attrs.get("agentlens.node")
            if node:
                counts[str(node)] = counts.get(str(node), 0) + 1
    return counts


def _produced_sizes(spans: list) -> dict[str, int]:
    """Return byte length of agentlens.produced JSON per agent node."""
    sizes: dict[str, int] = {}
    for span in spans:
        attrs = _attrs(span)
        if attrs.get("openinference.span.kind") == "AGENT":
            node = attrs.get("agentlens.node")
            produced = attrs.get("agentlens.produced", "")
            if node and produced:
                key = str(node)
                # Use max if the node ran multiple times (loop scenario)
                sizes[key] = max(sizes.get(key, 0), len(str(produced)))
    return sizes


def _detect_context_drop(
    handoffs: list[tuple[str, str]],
    produced_sizes: dict[str, int],
    threshold: float = 0.20,
) -> bool:
    """Return True if produced size drops by more than threshold% across any handoff."""
    for src, tgt in handoffs:
        src_size = produced_sizes.get(src, 0)
        tgt_size = produced_sizes.get(tgt, 0)
        if src_size > 0 and tgt_size > 0:
            drop = (src_size - tgt_size) / src_size
            if drop > threshold:
                return True
    return False


def _build_summary(
    agents: list[str],
    handoffs: list[tuple[str, str]],
    tools: list[str],
    loop_counts: dict[str, int],
    produced_sizes: dict[str, int],
    context_drop: bool,
    delegation_loop: bool,
) -> str:
    """Build a natural-language summary suitable for semantic embedding."""
    parts: list[str] = []

    # Agent structure
    n = len(agents)
    if agents:
        flow = " → ".join(agents)
        parts.append(f"{n} agent{'s' if n != 1 else ''} observed: {flow}.")
    else:
        parts.append("No agents observed.")

    # Handoffs
    parts.append(f"{len(handoffs)} handoff{'s' if len(handoffs) != 1 else ''} detected.")

    # Tools
    if tools:
        parts.append(f"Tools used: {', '.join(tools)}.")
    else:
        parts.append("No tool calls detected.")

    # Context sizes across handoffs
    if produced_sizes:
        size_descriptions = []
        for agent in agents:
            sz = produced_sizes.get(agent)
            if sz:
                size_descriptions.append(f"{agent} (~{sz}B)")
        if size_descriptions:
            parts.append(f"Output sizes: {', '.join(size_descriptions)}.")

    # Loop counts
    loops = {a: c for a, c in loop_counts.items() if c > 1}
    if loops:
        loop_desc = ", ".join(f"{a} ran {c}x" for a, c in loops.items())
        parts.append(f"Delegation loops: {loop_desc}.")
    else:
        parts.append("No delegation loops.")

    # Failure indicators
    indicators = []
    if context_drop:
        indicators.append("context_drop=True")
    if delegation_loop:
        indicators.append("delegation_loop=True")
    if not indicators:
        indicators.append("no failure indicators detected")
    parts.append(f"Failure indicators: {', '.join(indicators)}.")

    return " ".join(parts)
