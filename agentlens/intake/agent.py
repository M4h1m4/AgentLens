"""Multi-turn NL intake agent.

Takes a developer's plain-English description of their agent system — however
complete or incomplete — and produces an AgentProfile through targeted
follow-up questions.

Optionally accepts a codebase path (for static inspection) and a Phoenix
endpoint (for pulling existing traces). Both are used to pre-fill profile
fields before the LLM asks any questions, so the LLM only asks about
what is genuinely unknown.

Design decision (D3.1): plain Python class, not a LangGraph graph.
Design decision (D3.2): LLM re-extracts from full conversation each turn.
Design decision (D3.3): MAX_TURNS = 4 before forcing a draft.
Design decision (D3.7): pre-fill from code inspection + existing traces first,
  ask LLM only about remaining gaps. Three evidence sources in priority order:
  trace_observed > code_inspected > user_stated.
"""

from __future__ import annotations

from typing import Any, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from agentlens.intake.profile import AgentProfile, ConfidentValue


# ── structured output schema ────────────────────────────────────────────────

class _ExtractedAgent(BaseModel):
    name: str
    role: str
    confidence: str = "medium"


class _ExtractionResult(BaseModel):
    """What the LLM extracts from the full conversation so far."""

    framework: Optional[str] = None
    framework_confidence: str = "unknown"

    agent_count: Optional[int] = None
    agent_count_confidence: str = "unknown"

    agents: list[_ExtractedAgent] = []

    tools: list[str] = []
    tools_confidence: str = "unknown"

    expected_output: Optional[str] = None
    delegation_pattern: Optional[str] = None
    suspected_failure_modes: list[str] = []

    ready_for_draft: bool = False
    next_question: Optional[str] = None


# ── system prompt ─────────────────────────────────────────────────────────────

_BASE_PROMPT = """\
You are an AgentLens intake agent. Extract a structured profile of a multi-agent
system from the developer's description.

EXTRACTION RULES
- Extract only what is explicitly stated or very strongly implied. Never invent details.
- Confidence levels:
    high   — explicitly stated and clear
    medium — strongly implied but not explicit
    low    — guessed from weak signals
    unknown — no information at all

WHEN TO SET ready_for_draft = True
When ANY of these are true:
- You know the purpose of the system (even vaguely)
- You have at least one concrete fact (framework, agent count, a tool name, a symptom)
- The conversation has gone on for 3+ turns

QUESTION RULES (when NOT ready_for_draft)
- Ask exactly ONE question per turn — never multiple.
- Focus on the most important unknown: agent structure first, then framework,
  then tools, then symptoms.
- Do NOT ask about anything already listed in ALREADY KNOWN below.
- Do NOT ask about things the developer said they don't know.

The profile will be corrected by an observed trace. An incomplete draft is fine.\
"""


def _build_system_prompt(pre_filled: dict) -> str:
    """Build system prompt including pre-filled context from inspection/traces."""
    if not pre_filled:
        return _BASE_PROMPT

    lines = ["\nALREADY KNOWN (do not ask about these — skip directly to unknowns):"]
    for field_name, info in pre_filled.items():
        val = info.get("value")
        conf = info.get("confidence", "")
        source = info.get("source", "")
        note = info.get("note", "")
        line = f"  {field_name}: {val}  [{conf} confidence, from {source}]"
        if note:
            line += f"  — note: {note}"
        lines.append(line)

    return _BASE_PROMPT + "\n" + "\n".join(lines)


# ── helpers ───────────────────────────────────────────────────────────────────

def _default_llm() -> ChatAnthropic:
    return ChatAnthropic(model="claude-sonnet-4-6", temperature=0)


def _build_profile(raw: str, result: _ExtractionResult, pre_filled: dict) -> AgentProfile:
    """Merge LLM extraction result with pre-filled hints into an AgentProfile."""

    def _cv(value: Any, confidence: str, source: str = "user_stated") -> ConfidentValue:
        if value is None or value == [] or value == "":
            return ConfidentValue.unknown()
        return ConfidentValue(value=value, confidence=confidence, source=source)

    def _best(llm_value, llm_conf, hint_key: str) -> ConfidentValue:
        """Return whichever evidence source is more reliable."""
        hint = pre_filled.get(hint_key)
        if hint and hint.get("value") is not None:
            # code_inspected and trace_observed beat user_stated/LLM extraction
            if hint["source"] in ("code_inspected", "trace_observed"):
                return ConfidentValue(
                    value=hint["value"],
                    confidence=hint["confidence"],
                    source=hint["source"],
                    note=hint.get("note"),
                )
        return _cv(llm_value, llm_conf)

    # Build agents list — prefer code_inspected names if available
    agents_hint = pre_filled.get("agent_names", {}).get("value", [])
    if agents_hint and not result.agents:
        agents = [
            {
                "name": name,
                "role": "detected from codebase — role not described",
                "confidence": pre_filled["agent_names"]["confidence"],
                "source": "code_inspected",
            }
            for name in agents_hint
        ]
    else:
        agents = [
            {
                "name": a.name,
                "role": a.role,
                "confidence": a.confidence,
                "source": "user_stated",
            }
            for a in result.agents
        ]

    # Prefer code_inspected tools if LLM found none
    tools_hint = pre_filled.get("tool_names", {}).get("value", [])
    effective_tools = result.tools or tools_hint or None
    tools_source = (
        "code_inspected" if (not result.tools and tools_hint) else "user_stated"
    )
    tools_conf = (
        pre_filled.get("tool_names", {}).get("confidence", "unknown")
        if tools_source == "code_inspected"
        else result.tools_confidence
    )

    return AgentProfile(
        raw_description=raw,
        framework=_best(result.framework, result.framework_confidence, "framework"),
        agent_count=_cv(result.agent_count, result.agent_count_confidence),
        agents=agents,
        delegation_patterns=_cv(result.delegation_pattern, "medium"),
        tool_inventory=_cv(effective_tools, tools_conf, tools_source),
        expected_output=_cv(result.expected_output, "medium"),
        suspected_failure_modes=_cv(
            result.suspected_failure_modes or None, "low"
        ),
    )


def _pre_fill_from_inspection(codebase_path: str) -> dict:
    """Run codebase inspector and return pre-filled hints dict."""
    try:
        from agentlens.intake.inspector import inspect_codebase
        result = inspect_codebase(codebase_path)
        return result.to_profile_hints()
    except Exception:
        return {}


def _pre_fill_from_phoenix(phoenix_endpoint: str, description: str) -> dict:
    """Pull existing Phoenix traces and reconcile to get pre-filled hints."""
    try:
        from agentlens.intake.phoenix_reader import IntakePhoenixReader
        from agentlens.intake.reconciler import reconcile

        reader = IntakePhoenixReader(phoenix_endpoint)
        if not reader.is_available():
            return {}

        spans = reader.get_recent_spans()
        if not spans:
            return {}

        empty = AgentProfile(raw_description=description)
        reconciled = reconcile(empty, spans)

        hints: dict = {}
        if reconciled.framework.value:
            hints["framework"] = reconciled.framework.to_dict()
        if reconciled.agent_count.value is not None:
            hints["agent_count"] = reconciled.agent_count.to_dict()
        if reconciled.agents:
            hints["agent_names"] = {
                "value": [a["name"] for a in reconciled.agents],
                "confidence": "high",
                "source": "trace_observed",
            }
        if reconciled.tool_inventory.value:
            hints["tool_names"] = reconciled.tool_inventory.to_dict()
        if reconciled.delegation_patterns.value:
            hints["delegation_patterns"] = reconciled.delegation_patterns.to_dict()
        return hints
    except Exception:
        return {}


# ── main class ────────────────────────────────────────────────────────────────

class IntakeAgent:
    """Multi-turn intake conversation that produces an AgentProfile.

    Usage::

        agent = IntakeAgent()
        question = agent.start(
            "I have a research system — the output is losing context",
            codebase_path="/path/to/project",       # optional
            phoenix_endpoint="http://localhost:6006" # optional
        )
        while question is not None:
            question = agent.respond(input(question + " "))
        profile = agent.get_profile()

    For testing, inject ``_structured_llm`` to avoid hitting the API::

        agent = IntakeAgent(_structured_llm=FakeStructuredLLM([...]))
    """

    MAX_TURNS = 4

    def __init__(
        self,
        llm: Any = None,
        _structured_llm: Any = None,
    ) -> None:
        if _structured_llm is not None:
            self._structured_llm = _structured_llm
        else:
            base = llm or _default_llm()
            self._structured_llm = base.with_structured_output(_ExtractionResult)

        self._messages: list = []
        self._raw_parts: list[str] = []
        self._turn: int = 0
        self._profile: Optional[AgentProfile] = None
        self._pre_filled: dict = {}

    @property
    def done(self) -> bool:
        return self._profile is not None

    def start(
        self,
        description: str,
        codebase_path: str | None = None,
        phoenix_endpoint: str | None = None,
    ) -> Optional[str]:
        """Start the intake conversation.

        Args:
            description:      Developer's plain-English description (required).
            codebase_path:    Path to the project root for static inspection (optional).
            phoenix_endpoint: Phoenix URL to pull existing traces (optional).

        Returns:
            The first clarifying question, or None if enough is already known.
        """
        self._messages = [HumanMessage(content=description)]
        self._raw_parts = [description]
        self._turn = 0
        self._profile = None
        self._pre_filled = {}

        # Pre-fill from codebase inspection
        if codebase_path:
            self._pre_filled.update(_pre_fill_from_inspection(codebase_path))

        # Pre-fill from existing Phoenix traces (higher priority — overrides code inspection)
        if phoenix_endpoint:
            trace_hints = _pre_fill_from_phoenix(phoenix_endpoint, description)
            self._pre_filled.update(trace_hints)

        return self._step()

    def respond(self, user_message: str) -> Optional[str]:
        """Feed the developer's answer. Returns next question or None when complete."""
        if self.done:
            raise RuntimeError("Intake is already complete. Call get_profile().")
        self._messages.append(HumanMessage(content=user_message))
        self._raw_parts.append(user_message)
        self._turn += 1
        return self._step()

    def get_profile(self) -> AgentProfile:
        """Return the completed AgentProfile. Raises if intake is not done yet."""
        if not self.done:
            raise RuntimeError(
                "Intake is not complete. Keep calling respond() until it returns None."
            )
        return self._profile

    def _step(self) -> Optional[str]:
        system_prompt = _build_system_prompt(self._pre_filled)
        result: _ExtractionResult = self._structured_llm.invoke(
            [SystemMessage(content=system_prompt)] + self._messages
        )

        force_done = self._turn >= self.MAX_TURNS
        if result.ready_for_draft or force_done:
            self._profile = _build_profile(
                raw="\n".join(self._raw_parts),
                result=result,
                pre_filled=self._pre_filled,
            )
            return None

        question = result.next_question or "Can you tell me more about your agent system?"
        self._messages.append(AIMessage(content=question))
        return question
