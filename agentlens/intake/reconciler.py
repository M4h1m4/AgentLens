"""Trace reconciler — updates a draft AgentProfile with observed OTel span evidence.

After traced_invoke() runs, collect spans from InMemorySpanExporter or
PhoenixTraceReader, then call reconcile(profile, spans) to get a new profile
version where:
  - Fields confirmed by the trace are promoted to confidence="high", source="trace_observed"
  - Discrepancies between description and observation are logged
  - Unknown fields are filled in from the trace

The trace is always the ground truth. User descriptions are priors.
"""

from __future__ import annotations

from agentlens.intake.profile import AgentProfile, ConfidentValue


def reconcile(profile: AgentProfile, spans: list) -> AgentProfile:
    """Return a new AgentProfile (version bumped) reconciled against OTel spans.

    Args:
        profile: Draft profile from the intake agent.
        spans:   List of OTel ReadableSpan objects (from InMemorySpanExporter
                 or converted from PhoenixTraceReader).

    Does not mutate the input profile.
    """
    p = profile.bump_version()
    p.reconciled_with_trace = True

    observed_nodes = _nodes(spans)
    observed_handoffs = _handoffs(spans)
    observed_tools = _tools(spans)
    observed_framework = _framework(spans)

    _reconcile_framework(p, observed_framework)
    _reconcile_agents(p, observed_nodes, profile)
    _reconcile_handoffs(p, observed_handoffs)
    _reconcile_tools(p, observed_tools, profile)

    return p


# ── per-field reconciliation ─────────────────────────────────────────────────

def _reconcile_framework(p: AgentProfile, observed: str | None) -> None:
    if not observed:
        return
    if (
        p.framework.value
        and p.framework.confidence != "unknown"
        and p.framework.value != observed
    ):
        p.discrepancies.append(
            f"Framework: described as '{p.framework.value}', "
            f"observed as '{observed}'. Profile updated to observed value."
        )
    p.framework = ConfidentValue(
        value=observed, confidence="high", source="trace_observed"
    )


def _reconcile_agents(
    p: AgentProfile, observed_nodes: list[str], original: AgentProfile
) -> None:
    if not observed_nodes:
        return

    described_count = original.agent_count.value
    observed_count = len(observed_nodes)

    if described_count is not None and described_count != observed_count:
        p.discrepancies.append(
            f"Agent count: described as {described_count}, "
            f"observed {observed_count} nodes in trace "
            f"({', '.join(observed_nodes)}). Profile updated."
        )

    p.agent_count = ConfidentValue(
        value=observed_count, confidence="high", source="trace_observed"
    )

    draft_roles: dict[str, str] = {
        a["name"]: a.get("role", "") for a in original.agents
    }
    p.agents = [
        {
            "name": node,
            "role": draft_roles.get(node, "observed in trace — role not described"),
            "confidence": "high",
            "source": "trace_observed",
        }
        for node in observed_nodes
    ]

    described_names = {a["name"] for a in original.agents}
    observed_names = set(observed_nodes)

    for name in described_names - observed_names:
        p.discrepancies.append(
            f"Agent '{name}' was described but not observed in trace."
        )
    for name in observed_names - described_names:
        p.discrepancies.append(
            f"Agent '{name}' observed in trace but not mentioned in description."
        )


def _reconcile_handoffs(
    p: AgentProfile, observed: list[tuple[str, str]]
) -> None:
    if not observed:
        return
    p.delegation_patterns = ConfidentValue(
        value=observed,
        confidence="high",
        source="trace_observed",
    )


def _reconcile_tools(
    p: AgentProfile,
    observed_tools: list[str],
    original: AgentProfile,
) -> None:
    if not observed_tools:
        return

    described_tools: list[str] = original.tool_inventory.value or []
    described_set = set(described_tools)
    observed_set = set(observed_tools)

    for tool in described_set - observed_set:
        p.discrepancies.append(
            f"Tool '{tool}' was described but not observed in trace."
        )
    for tool in observed_set - described_set:
        p.discrepancies.append(
            f"Tool '{tool}' observed in trace but not mentioned in description."
        )

    p.tool_inventory = ConfidentValue(
        value=observed_tools, confidence="high", source="trace_observed"
    )


# ── span extraction helpers ───────────────────────────────────────────────────

def _attrs(span) -> dict:
    """Safely extract span attributes as a plain dict."""
    raw = getattr(span, "attributes", None) or {}
    return dict(raw)


def _nodes(spans: list) -> list[str]:
    """Extract unique agent node names from HANDOFF spans (source + target)."""
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
    """Extract (source, target) pairs from HANDOFF spans."""
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
    """Extract tool names from TOOL spans."""
    seen: list[str] = []
    for span in spans:
        attrs = _attrs(span)
        if attrs.get("openinference.span.kind") == "TOOL":
            tool = attrs.get("tool.name") or getattr(span, "name", None)
            if tool and str(tool) not in seen:
                seen.append(str(tool))
    return seen


def _framework(spans: list) -> str | None:
    """Extract framework from the root agentlens.run span."""
    for span in spans:
        name = getattr(span, "name", None)
        if name == "agentlens.run":
            attrs = _attrs(span)
            fw = attrs.get("agentlens.framework")
            if fw:
                return str(fw)
    return None
