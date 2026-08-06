"""Benchmark Gen Agent — generates a targeted test suite from a failure signal.

Design (from DESIGN_DECISIONS.md):
- LangGraph graph, not a plain class — runs autonomously to completion with no
  human input between steps (contrast with IntakeAgent which needs human turns).
- Graph: generate → route → (loop back | finalize)
- Loops until it has MIN_CASES (20) or exhausts MAX_ATTEMPTS (3).
- Uses MAST taxonomy as a prompt string (not RAG — 8 categories fit in context).
- Each case is traceable to the failure signal that motivated it.

Injection point for tests: pass `structured_llm` to `BenchmarkGenAgent` to
bypass the real LLM — same pattern as IntakeAgent.
"""

from __future__ import annotations

from typing import Any, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel
from typing_extensions import TypedDict

from agentlens.benchmark.schema import BenchmarkCase, BenchmarkSuite
from agentlens.rag.failure_signals import FailureSignal
from agentlens.rag.mast import as_prompt_section, category_names

MIN_CASES = 20
MAX_CASES = 30
MAX_ATTEMPTS = 3


# ── structured output schema ──────────────────────────────────────────────────

class _Case(BaseModel):
    query: str
    expected_criteria: list[str]
    failure_mode: str   # must be a valid MAST category name
    severity: str       # "high" | "medium" | "low"
    rationale: str


class _GenerationBatch(BaseModel):
    failure_modes_identified: list[str]
    cases: list[_Case]


# ── LangGraph state ───────────────────────────────────────────────────────────

class _BenchmarkState(TypedDict):
    signal_summary: str
    signal_failure_modes: list[str]   # MAST categories pre-detected from signal
    cases: list[dict]                 # serializable BenchmarkCase dicts
    failure_modes_identified: list[str]
    attempts: int
    trace_id: str
    session_id: str


# ── prompts ───────────────────────────────────────────────────────────────────

def _system_prompt(target_modes: list[str], existing_count: int) -> str:
    needed = max(MIN_CASES - existing_count, 5)
    return f"""\
You are the AgentLens Benchmark Gen Agent. Your job is to generate concrete test
cases that detect coordination failures in a multi-agent system.

{as_prompt_section()}

OBSERVED FAILURE MODES: {', '.join(target_modes) if target_modes else 'unknown — generate broad coverage'}

GENERATION RULES
1. Generate exactly {needed} test cases.
2. Each case MUST have all five fields:
   - query: a specific, concrete input string a user would send to the agent
   - expected_criteria: 2–4 natural language assertions a CORRECT output must satisfy
   - failure_mode: the MAST category this case is designed to catch
   - severity: "high" if this failure produces wrong/harmful output, "medium" if
     degraded quality, "low" if minor
   - rationale: one sentence explaining why this query triggers or exposes the failure
3. Prioritize the observed failure modes listed above.
4. Queries must be diverse — different topics, lengths, and complexity.
5. Expected criteria must be specific and checkable.
   Good: "output references at least 5 distinct sources"
   Bad:  "output is good"
6. In failure_modes_identified, list ALL MAST categories you believe apply to
   this trace (may include modes beyond the ones listed above).

VALID FAILURE MODE NAMES (use only these):
{', '.join(category_names())}
"""


# ── node functions ─────────────────────────────────────────────────────────────

_VALID_SEVERITIES = {"high", "medium", "low"}


def _make_generate_node(llm: Any):
    def generate(state: _BenchmarkState) -> dict:
        existing = state.get("cases", [])
        existing_count = len(existing)

        if existing_count >= MIN_CASES:
            return {}   # routing will send us to finalize

        target_modes = state.get("signal_failure_modes", [])
        system = _system_prompt(target_modes, existing_count)
        prompt = (
            f"Here is a summary of what the observed trace run showed:\n\n"
            f"{state['signal_summary']}\n\n"
            f"Generate test cases targeting these failure patterns. "
            f"Focus on: {', '.join(target_modes) if target_modes else 'all MAST categories'}."
        )

        result: _GenerationBatch = llm.invoke(
            [SystemMessage(content=system), HumanMessage(content=prompt)]
        )

        valid_modes = set(category_names())
        new_cases: list[dict] = []
        for c in result.cases:
            # Sanitize failure_mode — fall back to first target mode if invalid
            mode = c.failure_mode if c.failure_mode in valid_modes else (
                target_modes[0] if target_modes else "context_loss"
            )
            severity = c.severity if c.severity in _VALID_SEVERITIES else "medium"
            new_cases.append(
                BenchmarkCase.new(
                    query=c.query,
                    expected_criteria=c.expected_criteria,
                    failure_mode=mode,
                    severity=severity,
                    rationale=c.rationale,
                ).to_dict()
            )

        return {
            "cases": existing + new_cases,
            "failure_modes_identified": result.failure_modes_identified,
            "attempts": state.get("attempts", 0) + 1,
        }

    return generate


def _finalize(state: _BenchmarkState) -> dict:
    """Cap at MAX_CASES — suite is ready."""
    return {"cases": state.get("cases", [])[:MAX_CASES]}


def _route(state: _BenchmarkState) -> str:
    cases = state.get("cases", [])
    attempts = state.get("attempts", 0)
    if len(cases) >= MIN_CASES or attempts >= MAX_ATTEMPTS:
        return "finalize"
    return "generate"


# ── graph builder ─────────────────────────────────────────────────────────────

def _default_llm() -> ChatAnthropic:
    return ChatAnthropic(model="claude-sonnet-4-6", temperature=0)


def build_benchmark_graph(llm: Any = None, structured_llm: Any = None):
    """Build and compile the benchmark gen LangGraph.

    Args:
        llm:            Base LLM (ChatAnthropic or compatible). Defaults to
                        claude-sonnet-4-6.
        structured_llm: Pre-wrapped structured LLM — use this in tests to
                        inject a fake that returns _GenerationBatch objects.
    Returns:
        Compiled LangGraph application.
    """
    base = llm or _default_llm()
    sllm = structured_llm or base.with_structured_output(_GenerationBatch)
    generate_node = _make_generate_node(sllm)

    g: StateGraph = StateGraph(_BenchmarkState)
    g.add_node("generate", generate_node)
    g.add_node("finalize", _finalize)
    g.set_entry_point("generate")
    g.add_conditional_edges(
        "generate",
        _route,
        {"generate": "generate", "finalize": "finalize"},
    )
    g.add_edge("finalize", END)
    return g.compile()


# ── public class ──────────────────────────────────────────────────────────────

class BenchmarkGenAgent:
    """Generate a benchmark suite from an observed failure signal.

    Usage::

        agent = BenchmarkGenAgent()
        suite = agent.generate(signal)
        # suite.cases → list of 20-30 BenchmarkCase objects

    For tests — inject a fake structured LLM to avoid hitting the API::

        agent = BenchmarkGenAgent(structured_llm=FakeStructuredLLM([...]))
        suite = agent.generate(signal)
    """

    def __init__(self, llm: Any = None, structured_llm: Any = None) -> None:
        self._app = build_benchmark_graph(llm=llm, structured_llm=structured_llm)

    def generate(self, signal: FailureSignal) -> BenchmarkSuite:
        """Generate a benchmark suite targeting the failure modes in the signal.

        Args:
            signal: FailureSignal extracted from a completed trace run.

        Returns:
            BenchmarkSuite with 20–30 test cases.
        """
        # Derive targeted MAST categories from the signal's detected indicators
        failure_modes: list[str] = []
        if signal.context_drop_detected:
            failure_modes.append("context_loss")
        if signal.delegation_loop_detected:
            failure_modes.append("delegation_loop")
        # Always include these common coordination failures for broader coverage
        for mode in ("contradiction_ignored", "context_bleed", "race_condition"):
            if mode not in failure_modes:
                failure_modes.append(mode)

        initial: _BenchmarkState = {
            "signal_summary": signal.summary,
            "signal_failure_modes": failure_modes,
            "cases": [],
            "failure_modes_identified": [],
            "attempts": 0,
            "trace_id": signal.trace_id,
            "session_id": signal.session_id,
        }

        final = self._app.invoke(initial)

        cases = [BenchmarkCase.from_dict(c) for c in final.get("cases", [])]
        modes = final.get("failure_modes_identified") or failure_modes

        return BenchmarkSuite.new(
            trace_id=signal.trace_id,
            session_id=signal.session_id,
            cases=cases,
            failure_modes=modes,
        )
