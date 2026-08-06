"""Agent 2 — Synthesis agent.

Receives sources from Agent 1 and produces a structured summary: key claims,
supporting evidence, and conflicting viewpoints. Hosts three seeded failures:

  * Failure 1 (context loss): when it receives more sources than its cap it
    keeps only the most recent ones and silently drops the rest — no signal.
  * Failure 3 (contradiction ignored): when sources conflict it picks one and
    leaves the conflicting-viewpoints field empty instead of flagging it.
  * Failure 4 (delegation loop): it increments a loop counter used by the
    graph to decide whether to re-query Agent 1 (see graph.py). The bug is the
    missing termination condition, not this node itself.
"""

from __future__ import annotations

from typing import Any, Callable

from reference_agent.config import AgentConfig
from reference_agent.state import AgentState, Synthesis


def _detect_conflicts(sources: list[dict[str, Any]]) -> list[str]:
    stances = {s.get("stance") for s in sources if s.get("stance")}
    if len(stances) <= 1:
        return []
    return [
        f"{s['title']} takes the '{s['stance']}' position"
        for s in sources
        if s.get("stance")
    ]


def make_synthesis_agent(config: AgentConfig) -> Callable[[AgentState], AgentState]:
    failures = config.failures

    def synthesis_agent(state: AgentState) -> AgentState:
        sources = state.get("sources", [])
        cap = config.synthesis_source_cap

        # Failure 1 — context loss: silently keep only the most recent `cap`.
        dropped: list[dict[str, Any]] = []
        if failures.context_loss and len(sources) > cap:
            dropped = sources[:-cap]
            used = sources[-cap:]
        else:
            used = sources

        key_claims = [s["title"] for s in used]
        supporting_evidence = [s["excerpt"] for s in used]

        # Failure 3 — contradiction ignored: drop the conflict signal.
        if failures.contradiction_ignored:
            conflicting_viewpoints: list[str] = []
        else:
            conflicting_viewpoints = _detect_conflicts(used)

        complete: Synthesis = {
            "key_claims": key_claims,
            "supporting_evidence": supporting_evidence,
            "conflicting_viewpoints": conflicting_viewpoints,
        }
        # The "first chunk" a streaming synthesis would emit: claims only,
        # before evidence and conflicts are resolved. Failure 2 consumes this.
        partial: Synthesis = {
            "key_claims": key_claims,
            "supporting_evidence": [],
            "conflicting_viewpoints": [],
        }

        # Failure 4 — track how many times synthesis has run this session.
        loop_count = state.get("loop_count", 0) + 1

        return {
            "synthesis_complete": complete,
            "synthesis_partial": partial,
            "dropped_sources": dropped,
            "loop_count": loop_count,
        }

    return synthesis_agent
