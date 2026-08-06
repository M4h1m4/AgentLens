"""Agent 3 — Writer agent.

Receives the synthesis from Agent 2 and produces a final report. Hosts two
seeded failures:

  * Failure 2 (race condition): it starts writing from the *partial* synthesis
    (the first chunk) rather than waiting for Agent 2 to finish, so evidence
    and conflicts never reach the report.
  * Failure 5 (context bleed): a module-level buffer, not keyed by session,
    leaks a previous run's report into the current one when sessions are
    reused.

The node accepts `config` so LangGraph propagates the AgentLens callbacks into
the LLM call, letting the adapter capture tokens/latency automatically.
"""

from __future__ import annotations

from typing import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig

from reference_agent.config import AgentConfig
from reference_agent.state import AgentState, Synthesis

# Failure 5 — a global buffer that is (incorrectly) not scoped per session.
_LAST_REPORT: str = ""


def reset_bleed_cache() -> None:
    """Clear the cross-run buffer. Correct code would scope by session id."""
    global _LAST_REPORT
    _LAST_REPORT = ""


def _render(query: str, synthesis: Synthesis, prose: str) -> str:
    lines = [f"# Report: {query}", "", "## Key claims"]
    lines += [f"- {c}" for c in synthesis.get("key_claims", [])]

    evidence = synthesis.get("supporting_evidence", [])
    if evidence:
        lines += ["", "## Supporting evidence"]
        lines += [f"- {e}" for e in evidence]

    conflicts = synthesis.get("conflicting_viewpoints", [])
    if conflicts:
        lines += ["", "## Conflicting viewpoints"]
        lines += [f"- {c}" for c in conflicts]

    lines += ["", "## Summary", prose]
    return "\n".join(lines)


def make_writer_agent(
    agent_config: AgentConfig, llm: BaseChatModel
) -> Callable[..., AgentState]:
    failures = agent_config.failures

    def writer_agent(state: AgentState, config: RunnableConfig) -> AgentState:
        global _LAST_REPORT

        # Failure 2 — race condition: read the partial synthesis if the bug is on.
        if failures.race_condition:
            synthesis = state.get("synthesis_partial", {})
            writer_input = "partial"
        else:
            synthesis = state.get("synthesis_complete", {})
            writer_input = "complete"

        query = state["query"]
        response = llm.invoke(
            f"Write a concise report summary for the query: {query}\n"
            f"Key claims: {synthesis.get('key_claims', [])}",
            config=config,
        )
        prose = str(response.content)
        report = _render(query, synthesis, prose)

        # Failure 5 — context bleed: prepend the previous run's report.
        if failures.context_bleed and _LAST_REPORT:
            report = (
                "<!-- carried over from previous run -->\n"
                + _LAST_REPORT
                + "\n\n"
                + report
            )
        _LAST_REPORT = report

        return {"report": report, "writer_input": writer_input}

    return writer_agent
