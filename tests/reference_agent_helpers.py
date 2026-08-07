"""Reference-agent-specific test helpers — NOT part of the AgentLens public API.

This module contains test scaffolding that is specific to the reference agent
used to validate AgentLens. Nothing in this file should be imported from the
agentlens package itself. Any agent-agnostic utilities live in agentlens/eval/.

  StateInspectingExecutor  — runs the reference agent directly and converts
                             final state to synthetic spans. Avoids global OTel.
  PatternJudgeLLM          — fake judge that checks reference-agent output
                             patterns (e.g. "## Supporting evidence").
  make_*_suite()           — benchmark suite factories targeting the five
                             seeded failures in the reference agent.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from agentlens.benchmark.schema import BenchmarkCase, BenchmarkSuite
from agentlens.eval.checker import _Verdict
from agentlens.eval.runner import _SpanProxy
from reference_agent.agents.writer import reset_bleed_cache
from reference_agent.config import AgentConfig


# ── synthetic span builder ────────────────────────────────────────────────────

def _state_to_spans(state: dict[str, Any]) -> list[_SpanProxy]:
    """Convert reference agent final state into synthetic spans for the checker.

    Produces the same span shape the real OTel adapter would emit, so
    checker.py's deterministic patterns (context drop, delegation loop)
    work identically against these spans.
    """
    spans: list[_SpanProxy] = []
    loop_count = max(int(state.get("loop_count", 1)), 1)

    # Search AGENT span — carries raw source list
    spans.append(_SpanProxy("agentlens.agent.search", {
        "openinference.span.kind": "AGENT",
        "agentlens.node": "search",
        "agentlens.produced": json.dumps(state.get("sources", []))[:4000],
    }))

    # Handoff: search → synthesis
    spans.append(_SpanProxy("agentlens.handoff.search_synthesis", {
        "openinference.span.kind": "HANDOFF",
        "handoff.source": "search",
        "handoff.target": "synthesis",
    }))

    # Synthesis AGENT spans — one per loop iteration.
    # Multiple spans for the same node are what delegation-loop detection counts.
    for i in range(loop_count):
        spans.append(_SpanProxy(f"agentlens.agent.synthesis.{i}", {
            "openinference.span.kind": "AGENT",
            "agentlens.node": "synthesis",
            "agentlens.produced": json.dumps(
                state.get("synthesis_complete", {})
            )[:4000],
        }))

    # Handoff: synthesis → writer
    spans.append(_SpanProxy("agentlens.handoff.synthesis_writer", {
        "openinference.span.kind": "HANDOFF",
        "handoff.source": "synthesis",
        "handoff.target": "writer",
    }))

    # Writer AGENT span
    spans.append(_SpanProxy("agentlens.agent.writer", {
        "openinference.span.kind": "AGENT",
        "agentlens.node": "writer",
        "agentlens.produced": json.dumps(
            {"report": state.get("report", "")[:500]}
        ),
    }))

    return spans


# ── state-inspecting executor ─────────────────────────────────────────────────

class StateInspectingExecutor:
    """Run the reference agent and convert state to synthetic spans.

    No global OTel provider is touched. Spans are built from the final agent
    state and fed directly to EvalRunner, which passes them to checker.py.

    Args:
        config:         AgentConfig controlling which failures are active.
        llm:            LangChain LLM — use FakeListChatModel in tests.
        prewarm_bleed:  If set, run this query first to prime the context-bleed
                        cache before each real invocation (needed to detect
                        Failure 5).
    """

    def __init__(
        self,
        config: AgentConfig,
        llm: Any,
        *,
        prewarm_bleed: str | None = None,
    ) -> None:
        self._config = config
        self._llm = llm
        self._prewarm_bleed = prewarm_bleed

    def execute(self, query: str, session_id: str) -> tuple[str, str, list]:
        from reference_agent.graph import build_graph

        reset_bleed_cache()
        app = build_graph(self._config, self._llm)

        if self._prewarm_bleed:
            # Seed the bleed buffer so the next run picks it up.
            app.invoke({
                "query": self._prewarm_bleed,
                "session_id": "prewarm",
                "loop_count": 0,
            })

        state = app.invoke({
            "query": query,
            "session_id": session_id,
            "loop_count": 0,
        })

        spans = _state_to_spans(state)
        output = str(state.get("report", ""))
        return f"synthetic-{uuid.uuid4().hex[:8]}", output, spans


# ── pattern-matching LLM judge ────────────────────────────────────────────────

class PatternJudgeLLM:
    """Fake structured LLM judge using deterministic string patterns.

    Each criterion phrase maps to a specific marker to look for (or not) in
    the agent output. Eliminates API calls from Day 7 verification tests while
    still exercising the real checker.py → _llm_judge → CriterionResult path.
    """

    # (criterion keyword, output pattern, expect_present)
    _RULES: list[tuple[str, str | None, bool]] = [
        ("supporting evidence",      "## Supporting evidence",        True),
        ("conflicting viewpoints",   "## Conflicting viewpoints",     True),
        ("previous session",         "carried over from previous run", False),
        ("previous run",             "carried over from previous run", False),
        ("all available sources",    None,                            True),  # special
    ]

    # First unique words from each of the 5 default fixture source titles
    _SOURCE_KEYWORDS = [
        "grid-scale battery",
        "solar overtakes",
        "transmission bottleneck",
        "curtailment rising",
        "policy incentives",
    ]

    def invoke(self, messages: list) -> _Verdict:
        content = messages[-1].content if messages else ""

        output_match = re.search(
            r"Agent output:\n(.*?)\n\nCriterion to check:", content, re.DOTALL
        )
        criterion_match = re.search(
            r"Criterion to check: (.+)$", content, re.MULTILINE
        )

        output = output_match.group(1).strip() if output_match else ""
        criterion = criterion_match.group(1).strip() if criterion_match else ""
        return self._evaluate(criterion.lower(), output.lower())

    def _evaluate(self, criterion_lower: str, output_lower: str) -> _Verdict:
        for keyword, pattern, expect_present in self._RULES:
            if keyword not in criterion_lower:
                continue

            if pattern is None:
                # Special case: all 5 source titles must appear in the output
                found = sum(1 for kw in self._SOURCE_KEYWORDS if kw in output_lower)
                total = len(self._SOURCE_KEYWORDS)
                passed = found == total
                return _Verdict(
                    passed=passed,
                    reason=f"{found}/{total} source titles found in output.",
                )

            found = pattern.lower() in output_lower
            passed = found if expect_present else not found
            return _Verdict(
                passed=passed,
                reason=(
                    f"'{pattern}' {'found' if found else 'not found'} in output. "
                    f"Expected: {'present' if expect_present else 'absent'}."
                ),
            )

        return _Verdict(passed=True, reason="No pattern matched — criterion not evaluated.")


# ── benchmark suite factories ─────────────────────────────────────────────────

def make_context_loss_suite(n: int = 5) -> BenchmarkSuite:
    """Cases targeting Failure 1: synthesis silently drops sources beyond cap."""
    queries = [
        "Summarize renewable energy trends in 2025",
        "What is the state of battery storage adoption?",
        "Overview of solar energy growth this year",
        "Renewable energy policy updates and outlook",
        "Grid infrastructure challenges for renewables",
    ]
    cases = [
        BenchmarkCase.new(
            query=queries[i % len(queries)],
            expected_criteria=["Output references all available sources from the search"],
            failure_mode="context_loss",
            severity="high",
            rationale="5 sources fetched but cap is 3; dropped sources must not silently vanish",
        )
        for i in range(n)
    ]
    return BenchmarkSuite.new(
        trace_id="seeded-context-loss",
        session_id="eval-day7",
        cases=cases,
        failure_modes=["context_loss"],
    )


def make_delegation_loop_suite(n: int = 5) -> BenchmarkSuite:
    """Cases targeting Failure 4: delegation loop without terminator."""
    queries = [
        "exhaustive review of grid storage options",
        "exhaustive analysis of renewable capacity",
        "exhaustive comparison of energy storage policies",
        "exhaustive study of solar vs wind economics",
        "exhaustive breakdown of battery technology trends",
    ]
    cases = [
        BenchmarkCase.new(
            query=queries[i % len(queries)],
            expected_criteria=["No agent node ran more than once"],
            failure_mode="delegation_loop",
            severity="high",
            rationale="'exhaustive' keyword triggers loop; synthesis must run exactly once",
        )
        for i in range(n)
    ]
    return BenchmarkSuite.new(
        trace_id="seeded-delegation-loop",
        session_id="eval-day7",
        cases=cases,
        failure_modes=["delegation_loop"],
    )


def make_race_condition_suite(n: int = 5) -> BenchmarkSuite:
    """Cases targeting Failure 2: writer consumes partial synthesis."""
    queries = [
        "Summarize renewable energy adoption trends",
        "What are the key findings on solar energy capacity?",
        "Report on grid-scale battery deployment progress",
        "Summary of renewable policy developments in 2025",
        "Analysis of transmission infrastructure challenges",
    ]
    cases = [
        BenchmarkCase.new(
            query=queries[i % len(queries)],
            expected_criteria=["Output includes supporting evidence from sources"],
            failure_mode="race_condition",
            severity="high",
            rationale="Writer using partial synthesis omits the supporting evidence section",
        )
        for i in range(n)
    ]
    return BenchmarkSuite.new(
        trace_id="seeded-race-condition",
        session_id="eval-day7",
        cases=cases,
        failure_modes=["race_condition"],
    )


def make_contradiction_suite(n: int = 5) -> BenchmarkSuite:
    """Cases targeting Failure 3: conflicting sources silently ignored."""
    queries = [
        "nuclear energy debate and conflict in grid planning",
        "conflicting views on nuclear power for decarbonization",
        "debate around nuclear conflict in energy policy",
        "nuclear energy: conflicting perspectives from experts",
        "conflict between pro and anti nuclear viewpoints",
    ]
    cases = [
        BenchmarkCase.new(
            query=queries[i % len(queries)],
            expected_criteria=["Output acknowledges conflicting viewpoints on the topic"],
            failure_mode="contradiction_ignored",
            severity="medium",
            rationale="Conflict fixture has opposing stances; synthesis must flag them",
        )
        for i in range(n)
    ]
    return BenchmarkSuite.new(
        trace_id="seeded-contradiction",
        session_id="eval-day7",
        cases=cases,
        failure_modes=["contradiction_ignored"],
    )


def make_context_bleed_suite(n: int = 5) -> BenchmarkSuite:
    """Cases targeting Failure 5: previous session content bleeds into report."""
    queries = [
        "overview of battery storage deployment outlook",
        "solar energy adoption statistics in 2025",
        "renewable energy policy landscape overview",
        "grid infrastructure investment challenges",
        "wind energy growth and capacity trends",
    ]
    cases = [
        BenchmarkCase.new(
            query=queries[i % len(queries)],
            expected_criteria=["Output does not contain content from a previous session"],
            failure_mode="context_bleed",
            severity="high",
            rationale="Module-level buffer carries prior report into current session output",
        )
        for i in range(n)
    ]
    return BenchmarkSuite.new(
        trace_id="seeded-context-bleed",
        session_id="eval-day7",
        cases=cases,
        failure_modes=["context_bleed"],
    )
