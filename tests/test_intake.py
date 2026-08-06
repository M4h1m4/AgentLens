"""Tests for Day 3: NL Intake Agent, AgentProfile, ProfileStore, reconciler.

Fake LLM pattern: inject a _structured_llm that returns pre-built
_ExtractionResult objects, so tests never hit the real Anthropic API.
"""

from __future__ import annotations

import os
import pytest

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from agentlens.intake.agent import IntakeAgent, _ExtractionResult, _ExtractedAgent
from agentlens.intake.inspector import inspect_codebase, InspectionResult
from agentlens.intake.phoenix_reader import IntakePhoenixReader, _SpanProxy
from agentlens.intake.profile import AgentProfile, ConfidentValue
from agentlens.intake.reconciler import reconcile
from agentlens.intake.store import ProfileStore


# ── fake structured LLM ──────────────────────────────────────────────────────

class _FakeStructuredLLM:
    """Returns a sequence of _ExtractionResult objects without calling the API."""

    def __init__(self, responses: list[_ExtractionResult]) -> None:
        self._responses = iter(responses)

    def invoke(self, messages: list) -> _ExtractionResult:
        return next(self._responses)


# ── AgentProfile tests ───────────────────────────────────────────────────────

class TestAgentProfile:
    def test_defaults_are_unknown(self):
        p = AgentProfile()
        assert p.framework.confidence == "unknown"
        assert p.agent_count.value is None
        assert p.agents == []
        assert not p.reconciled_with_trace

    def test_roundtrip(self):
        p = AgentProfile(
            raw_description="3-agent research system on LangGraph",
            framework=ConfidentValue("langgraph", "high", "user_stated"),
            agent_count=ConfidentValue(3, "medium", "user_stated"),
        )
        assert AgentProfile.from_dict(p.to_dict()).framework.value == "langgraph"
        assert AgentProfile.from_dict(p.to_dict()).agent_count.value == 3

    def test_bump_version_does_not_mutate(self):
        p = AgentProfile()
        p2 = p.bump_version()
        assert p.version == 1
        assert p2.version == 2
        assert p2.profile_id == p.profile_id

    def test_confident_value_unknown_factory(self):
        cv = ConfidentValue.unknown()
        assert cv.value is None
        assert cv.confidence == "unknown"

    def test_confident_value_roundtrip(self):
        cv = ConfidentValue("langgraph", "high", "trace_observed", note="seen in trace")
        cv2 = ConfidentValue.from_dict(cv.to_dict())
        assert cv2.value == "langgraph"
        assert cv2.note == "seen in trace"


# ── ProfileStore tests ───────────────────────────────────────────────────────

class TestProfileStore:
    def test_save_and_load(self, tmp_path):
        store = ProfileStore(base_dir=tmp_path)
        p = AgentProfile(raw_description="test system")
        store.save(p)
        loaded = store.load(p.profile_id)
        assert loaded.raw_description == "test system"
        assert loaded.version == 1

    def test_load_specific_version(self, tmp_path):
        store = ProfileStore(base_dir=tmp_path)
        p = AgentProfile(raw_description="v1")
        store.save(p)
        p2 = p.bump_version()
        p2.raw_description = "v2"
        store.save(p2)
        loaded_v1 = store.load(p.profile_id, version=1)
        loaded_v2 = store.load(p.profile_id, version=2)
        assert loaded_v1.raw_description == "v1"
        assert loaded_v2.raw_description == "v2"

    def test_load_latest_returns_highest_version(self, tmp_path):
        store = ProfileStore(base_dir=tmp_path)
        p = AgentProfile()
        store.save(p)
        p2 = p.bump_version()
        store.save(p2)
        latest = store.load(p.profile_id)
        assert latest.version == 2

    def test_load_latest_convenience(self, tmp_path):
        store = ProfileStore(base_dir=tmp_path)
        p = AgentProfile(raw_description="my system")
        store.save(p)
        assert store.load_latest().raw_description == "my system"

    def test_list_profiles(self, tmp_path):
        store = ProfileStore(base_dir=tmp_path)
        p1 = AgentProfile()
        p2 = AgentProfile()
        store.save(p1)
        store.save(p2)
        ids = store.list_profiles()
        assert p1.profile_id in ids
        assert p2.profile_id in ids

    def test_history_returns_all_versions_in_order(self, tmp_path):
        store = ProfileStore(base_dir=tmp_path)
        p = AgentProfile()
        p2 = p.bump_version()
        p3 = p2.bump_version()
        store.save(p)
        store.save(p2)
        store.save(p3)
        history = store.history(p.profile_id)
        assert [h.version for h in history] == [1, 2, 3]

    def test_load_missing_raises(self, tmp_path):
        store = ProfileStore(base_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            store.load("nonexistent-id")


# ── IntakeAgent tests ─────────────────────────────────────────────────────────

class TestIntakeAgent:
    def test_complete_description_produces_profile_immediately(self):
        fake = _FakeStructuredLLM([
            _ExtractionResult(
                framework="langgraph",
                framework_confidence="high",
                agent_count=3,
                agent_count_confidence="high",
                agents=[
                    _ExtractedAgent(name="search", role="searches the web", confidence="high"),
                    _ExtractedAgent(name="synthesis", role="synthesizes sources", confidence="high"),
                    _ExtractedAgent(name="writer", role="writes the report", confidence="high"),
                ],
                tools=["web_search"],
                tools_confidence="high",
                expected_output="research report",
                ready_for_draft=True,
                next_question=None,
            )
        ])
        agent = IntakeAgent(_structured_llm=fake)
        question = agent.start("I have a 3-agent LangGraph research system...")
        assert question is None
        assert agent.done
        profile = agent.get_profile()
        assert profile.framework.value == "langgraph"
        assert profile.agent_count.value == 3
        assert len(profile.agents) == 3
        assert profile.agents[0]["name"] == "search"

    def test_incomplete_description_triggers_follow_up(self):
        fake = _FakeStructuredLLM([
            # turn 0: not ready, ask a question
            _ExtractionResult(
                framework=None,
                agent_count=None,
                ready_for_draft=False,
                next_question="How many agents does your system have?",
            ),
            # turn 1: now ready
            _ExtractionResult(
                framework="langgraph",
                framework_confidence="medium",
                agent_count=2,
                agent_count_confidence="high",
                agents=[
                    _ExtractedAgent(name="search", role="searches", confidence="high"),
                    _ExtractedAgent(name="writer", role="writes", confidence="high"),
                ],
                ready_for_draft=True,
                next_question=None,
            ),
        ])
        agent = IntakeAgent(_structured_llm=fake)
        question = agent.start("My system does research.")
        assert question == "How many agents does your system have?"
        assert not agent.done

        question2 = agent.respond("It has 2 agents — search and writer.")
        assert question2 is None
        assert agent.done
        profile = agent.get_profile()
        assert profile.agent_count.value == 2

    def test_max_turns_forces_draft(self):
        # LLM keeps saying not ready — agent should force a draft after MAX_TURNS
        responses = [
            _ExtractionResult(ready_for_draft=False, next_question=f"Question {i}?")
            for i in range(IntakeAgent.MAX_TURNS + 2)
        ]
        fake = _FakeStructuredLLM(responses)
        agent = IntakeAgent(_structured_llm=fake)
        agent.start("Some system.")
        for _ in range(IntakeAgent.MAX_TURNS):
            if agent.done:
                break
            agent.respond("I don't know.")
        assert agent.done

    def test_get_profile_before_done_raises(self):
        fake = _FakeStructuredLLM([
            _ExtractionResult(ready_for_draft=False, next_question="How many agents?"),
        ])
        agent = IntakeAgent(_structured_llm=fake)
        agent.start("Some system.")
        with pytest.raises(RuntimeError):
            agent.get_profile()

    def test_respond_after_done_raises(self):
        fake = _FakeStructuredLLM([
            _ExtractionResult(ready_for_draft=True),
        ])
        agent = IntakeAgent(_structured_llm=fake)
        agent.start("Done immediately.")
        with pytest.raises(RuntimeError):
            agent.respond("extra message")

    def test_raw_description_accumulates(self):
        fake = _FakeStructuredLLM([
            _ExtractionResult(ready_for_draft=False, next_question="Tell me more?"),
            _ExtractionResult(ready_for_draft=True),
        ])
        agent = IntakeAgent(_structured_llm=fake)
        agent.start("Initial description.")
        agent.respond("Additional detail.")
        profile = agent.get_profile()
        assert "Initial description." in profile.raw_description
        assert "Additional detail." in profile.raw_description

    def test_symptom_only_description_produces_profile(self):
        """User describes only symptoms — intake should still produce a valid profile."""
        fake = _FakeStructuredLLM([
            _ExtractionResult(
                suspected_failure_modes=["context loss", "incomplete output"],
                ready_for_draft=True,
            )
        ])
        agent = IntakeAgent(_structured_llm=fake)
        question = agent.start("The output is always missing half the information.")
        assert question is None
        profile = agent.get_profile()
        assert profile.suspected_failure_modes.value == ["context loss", "incomplete output"]


# ── span factory for reconciler tests ─────────────────────────────────────────

@pytest.fixture
def _span_exporter():
    """OTel in-memory exporter for reconciler span tests."""
    exporter = InMemorySpanExporter()
    yield exporter
    exporter.clear()


def _make_spans(
    exporter: InMemorySpanExporter,
    nodes: list[str],
    handoffs: list[tuple[str, str]],
    tools: list[str],
    framework: str = "langgraph",
) -> list:
    """Create real OTel spans for reconciler tests.

    Uses a local TracerProvider so this function never fights over the
    global OTel provider set by test_adapter.py.
    """
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    local_provider = TracerProvider()
    local_provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = local_provider.get_tracer("agentlens.test")
    exporter.clear()

    with tracer.start_as_current_span("agentlens.run") as root:
        root.set_attribute("agentlens.framework", framework)
        root.set_attribute("openinference.span.kind", "AGENT")

        for src, tgt in handoffs:
            with tracer.start_as_current_span("agentlens.handoff") as span:
                span.set_attribute("openinference.span.kind", "HANDOFF")
                span.set_attribute("handoff.source", src)
                span.set_attribute("handoff.target", tgt)

        for tool in tools:
            with tracer.start_as_current_span(tool) as span:
                span.set_attribute("openinference.span.kind", "TOOL")
                span.set_attribute("tool.name", tool)

    return list(exporter.get_finished_spans())


class TestReconciler:
    def test_bumps_version(self, _span_exporter):
        profile = AgentProfile()
        spans = _make_spans(_span_exporter, ["search"], [], [])
        updated = reconcile(profile, spans)
        assert updated.version == profile.version + 1

    def test_does_not_mutate_original(self, _span_exporter):
        profile = AgentProfile()
        spans = _make_spans(_span_exporter, ["search", "writer"], [("search", "writer")], [])
        reconcile(profile, spans)
        assert profile.agent_count.value is None

    def test_sets_reconciled_flag(self, _span_exporter):
        profile = AgentProfile()
        spans = _make_spans(_span_exporter, ["search"], [], [])
        updated = reconcile(profile, spans)
        assert updated.reconciled_with_trace is True

    def test_fills_unknown_agent_count_from_trace(self, _span_exporter):
        profile = AgentProfile()
        spans = _make_spans(
            _span_exporter,
            ["search", "synthesis", "writer"],
            [("search", "synthesis"), ("synthesis", "writer")],
            [],
        )
        updated = reconcile(profile, spans)
        assert updated.agent_count.value == 3
        assert updated.agent_count.confidence == "high"
        assert updated.agent_count.source == "trace_observed"

    def test_fills_unknown_framework_from_trace(self, _span_exporter):
        profile = AgentProfile()
        spans = _make_spans(_span_exporter, ["search"], [], [], framework="langgraph")
        updated = reconcile(profile, spans)
        assert updated.framework.value == "langgraph"
        assert updated.framework.confidence == "high"

    def test_fills_tools_from_trace(self, _span_exporter):
        profile = AgentProfile()
        spans = _make_spans(_span_exporter, ["search"], [], ["web_search"])
        updated = reconcile(profile, spans)
        assert "web_search" in updated.tool_inventory.value
        assert updated.tool_inventory.confidence == "high"

    def test_fills_delegation_patterns_from_trace(self, _span_exporter):
        profile = AgentProfile()
        spans = _make_spans(
            _span_exporter, ["search", "writer"], [("search", "writer")], []
        )
        updated = reconcile(profile, spans)
        assert ("search", "writer") in updated.delegation_patterns.value

    def test_discrepancy_logged_when_agent_count_wrong(self, _span_exporter):
        profile = AgentProfile(
            agent_count=ConfidentValue(2, "high", "user_stated")
        )
        spans = _make_spans(
            _span_exporter,
            ["search", "synthesis", "writer"],
            [("search", "synthesis"), ("synthesis", "writer")],
            [],
        )
        updated = reconcile(profile, spans)
        assert any("2" in d and "3" in d for d in updated.discrepancies)

    def test_discrepancy_logged_for_agent_not_in_trace(self, _span_exporter):
        profile = AgentProfile(
            agents=[
                {"name": "phantom_agent", "role": "?", "confidence": "high", "source": "user_stated"}
            ]
        )
        spans = _make_spans(
            _span_exporter, ["search", "writer"], [("search", "writer")], []
        )
        updated = reconcile(profile, spans)
        assert any("phantom_agent" in d for d in updated.discrepancies)

    def test_discrepancy_logged_for_unmentioned_agent_in_trace(self, _span_exporter):
        profile = AgentProfile(
            agents=[
                {"name": "search", "role": "searches", "confidence": "high", "source": "user_stated"}
            ]
        )
        spans = _make_spans(
            _span_exporter,
            ["search", "synthesis"],
            [("search", "synthesis")],
            [],
        )
        updated = reconcile(profile, spans)
        assert any("synthesis" in d for d in updated.discrepancies)

    def test_discrepancy_logged_for_tool_name_mismatch(self, _span_exporter):
        profile = AgentProfile(
            tool_inventory=ConfidentValue(["brave_search"], "medium", "user_stated")
        )
        spans = _make_spans(_span_exporter, ["search"], [], ["web_search"])
        updated = reconcile(profile, spans)
        assert any("brave_search" in d for d in updated.discrepancies)
        assert any("web_search" in d for d in updated.discrepancies)

    def test_preserves_role_descriptions_from_draft(self, _span_exporter):
        profile = AgentProfile(
            agents=[
                {"name": "search", "role": "searches the web for sources",
                 "confidence": "high", "source": "user_stated"}
            ]
        )
        spans = _make_spans(
            _span_exporter, ["search", "writer"], [("search", "writer")], []
        )
        updated = reconcile(profile, spans)
        search = next(a for a in updated.agents if a["name"] == "search")
        assert search["role"] == "searches the web for sources"

    def test_empty_spans_does_not_overwrite_known_fields(self, _span_exporter):
        profile = AgentProfile(
            framework=ConfidentValue("langgraph", "high", "user_stated"),
            agent_count=ConfidentValue(3, "high", "user_stated"),
        )
        updated = reconcile(profile, [])
        assert updated.framework.value == "langgraph"
        assert updated.agent_count.value == 3

    def test_no_discrepancies_when_trace_matches_description(self, _span_exporter):
        profile = AgentProfile(
            framework=ConfidentValue("langgraph", "high", "user_stated"),
            agent_count=ConfidentValue(2, "high", "user_stated"),
            agents=[
                {"name": "search", "role": "searches", "confidence": "high", "source": "user_stated"},
                {"name": "writer", "role": "writes", "confidence": "high", "source": "user_stated"},
            ],
            tool_inventory=ConfidentValue(["web_search"], "high", "user_stated"),
        )
        spans = _make_spans(
            _span_exporter,
            ["search", "writer"],
            [("search", "writer")],
            ["web_search"],
            framework="langgraph",
        )
        updated = reconcile(profile, spans)
        assert updated.discrepancies == []


# ── inspector tests ───────────────────────────────────────────────────────────

class TestInspector:
    def test_detects_langgraph_framework(self, tmp_path):
        (tmp_path / "graph.py").write_text(
            "from langgraph.graph import StateGraph\nfrom langgraph import something\n"
        )
        result = inspect_codebase(tmp_path)
        assert result.framework == "langgraph"
        assert result.framework_confidence == "high"

    def test_detects_crewai_framework(self, tmp_path):
        (tmp_path / "crew.py").write_text("from crewai import Agent, Task\n")
        result = inspect_codebase(tmp_path)
        assert result.framework == "crewai"

    def test_extracts_node_names_from_add_node(self, tmp_path):
        (tmp_path / "graph.py").write_text(
            'from langgraph.graph import StateGraph\n'
            'graph = StateGraph(State)\n'
            'graph.add_node("search", search_agent)\n'
            'graph.add_node("synthesis", synthesis_agent)\n'
            'graph.add_node("writer", writer_agent)\n'
        )
        result = inspect_codebase(tmp_path)
        assert "search" in result.agent_names
        assert "synthesis" in result.agent_names
        assert "writer" in result.agent_names

    def test_detects_dynamic_node_construction(self, tmp_path):
        (tmp_path / "graph.py").write_text(
            'from langgraph.graph import StateGraph\n'
            'for name in agent_names:\n'
            '    graph.add_node(name, make_agent(name))\n'
        )
        result = inspect_codebase(tmp_path)
        assert result.has_dynamic_nodes is True

    def test_extracts_tool_names(self, tmp_path):
        (tmp_path / "tools.py").write_text(
            'from langchain_core.tools import tool\n\n'
            '@tool\n'
            'def web_search(query: str) -> list:\n'
            '    pass\n\n'
            '@tool\n'
            'def calculator(expr: str) -> float:\n'
            '    pass\n'
        )
        result = inspect_codebase(tmp_path)
        assert "web_search" in result.tool_names
        assert "calculator" in result.tool_names

    def test_detects_entry_point_from_compile(self, tmp_path):
        (tmp_path / "run.py").write_text(
            'from langgraph.graph import StateGraph\n'
            'builder = StateGraph(State)\n'
            'app = builder.compile()\n'
        )
        result = inspect_codebase(tmp_path)
        assert any("app" in ep for ep in result.entry_points)

    def test_skips_venv_directory(self, tmp_path):
        venv = tmp_path / ".venv" / "lib" / "site-packages" / "somelib"
        venv.mkdir(parents=True)
        (venv / "graph.py").write_text(
            'from langgraph.graph import StateGraph\n'
            'graph.add_node("venv_node", fn)\n'
        )
        result = inspect_codebase(tmp_path)
        assert "venv_node" not in result.agent_names

    def test_nonexistent_path_returns_empty_result_with_note(self, tmp_path):
        result = inspect_codebase(tmp_path / "does_not_exist")
        assert result.framework is None
        assert result.agent_names == []
        assert any("does not exist" in n for n in result.notes)

    def test_files_scanned_count(self, tmp_path):
        (tmp_path / "a.py").write_text("x = 1\n")
        (tmp_path / "b.py").write_text("y = 2\n")
        result = inspect_codebase(tmp_path)
        assert result.files_scanned == 2

    def test_to_profile_hints_framework(self, tmp_path):
        (tmp_path / "graph.py").write_text("from langgraph.graph import StateGraph\n")
        result = inspect_codebase(tmp_path)
        hints = result.to_profile_hints()
        assert hints["framework"]["value"] == "langgraph"
        assert hints["framework"]["source"] == "code_inspected"

    def test_to_profile_hints_empty_when_nothing_found(self, tmp_path):
        (tmp_path / "empty.py").write_text("x = 1\n")
        result = inspect_codebase(tmp_path)
        hints = result.to_profile_hints()
        assert hints == {}

    def test_note_added_for_dynamic_nodes(self, tmp_path):
        (tmp_path / "graph.py").write_text(
            'from langgraph.graph import StateGraph\n'
            'for name in names:\n'
            '    graph.add_node(name, fn)\n'
        )
        result = inspect_codebase(tmp_path)
        assert any("dynamic" in n.lower() for n in result.notes)

    def test_multiple_frameworks_adds_note(self, tmp_path):
        (tmp_path / "graph.py").write_text(
            "from langgraph.graph import StateGraph\nfrom crewai import Agent\n"
        )
        result = inspect_codebase(tmp_path)
        assert any("Multiple frameworks" in n for n in result.notes)

    def test_intake_agent_uses_codebase_hints(self, tmp_path):
        """IntakeAgent.start() with codebase_path pre-fills framework."""
        (tmp_path / "graph.py").write_text(
            'from langgraph.graph import StateGraph\n'
            'graph.add_node("search", search_fn)\n'
            'graph.add_node("writer", writer_fn)\n'
        )
        fake = _FakeStructuredLLM([
            _ExtractionResult(ready_for_draft=True)
        ])
        agent = IntakeAgent(_structured_llm=fake)
        agent.start("My system loses context.", codebase_path=str(tmp_path))
        profile = agent.get_profile()
        # Framework and agents came from code inspection
        assert profile.framework.value == "langgraph"
        assert profile.framework.source == "code_inspected"
        assert any(a["name"] == "search" for a in profile.agents)
        assert any(a["name"] == "writer" for a in profile.agents)


# ── phoenix reader tests ───────────────────────────────────────────────────────

class TestIntakePhoenixReader:
    def test_is_available_returns_false_when_phoenix_not_running(self):
        reader = IntakePhoenixReader("http://localhost:19999")  # nothing on this port
        assert reader.is_available() is False

    def test_get_recent_spans_returns_empty_when_unavailable(self):
        reader = IntakePhoenixReader("http://localhost:19999")
        spans = reader.get_recent_spans()
        assert spans == []

    def test_get_trace_count_returns_zero_when_unavailable(self):
        reader = IntakePhoenixReader("http://localhost:19999")
        assert reader.get_trace_count() == 0

    def test_span_proxy_has_name_and_attributes(self):
        proxy = _SpanProxy(
            name="agentlens.handoff",
            attributes={
                "openinference.span.kind": "HANDOFF",
                "handoff.source": "search",
                "handoff.target": "writer",
            },
        )
        assert proxy.name == "agentlens.handoff"
        assert proxy.attributes["handoff.source"] == "search"

    def test_span_proxy_compatible_with_reconciler(self, _span_exporter):
        """_SpanProxy objects are accepted by reconcile() without errors."""
        proxies = [
            _SpanProxy(
                name="agentlens.run",
                attributes={"agentlens.framework": "langgraph", "openinference.span.kind": "AGENT"},
            ),
            _SpanProxy(
                name="agentlens.handoff",
                attributes={
                    "openinference.span.kind": "HANDOFF",
                    "handoff.source": "search",
                    "handoff.target": "writer",
                },
            ),
            _SpanProxy(
                name="web_search",
                attributes={"openinference.span.kind": "TOOL", "tool.name": "web_search"},
            ),
        ]
        profile = AgentProfile()
        updated = reconcile(profile, proxies)
        assert updated.framework.value == "langgraph"
        assert updated.tool_inventory.value == ["web_search"]
        assert ("search", "writer") in updated.delegation_patterns.value
