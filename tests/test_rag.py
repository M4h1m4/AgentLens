"""Day 4 — failure library and signal extraction tests.

FailureLibrary.ephemeral() + _FakeEmbeddingFunction: no ChromaDB disk writes,
no sentence-transformers model download. Tests run offline.
"""

from __future__ import annotations

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from agentlens.rag.failure_signals import (
    FailureSignal,
    extract_signals,
    _agent_order,
    _handoffs,
    _tools,
    _loop_counts,
    _produced_sizes,
    _detect_context_drop,
)
from agentlens.rag.library import FailureLibrary, _FakeEmbeddingFunction
from agentlens.rag.mast import (
    as_prompt_section,
    category_names,
    describe,
    CATEGORIES,
)


# ── span factory ──────────────────────────────────────────────────────────────

@pytest.fixture
def _exporter():
    """Fresh in-memory exporter per test — no global OTel state touched."""
    exporter = InMemorySpanExporter()
    yield exporter
    exporter.clear()


def _make_spans(
    exporter: InMemorySpanExporter,
    nodes: list[str],
    handoffs: list[tuple[str, str]],
    tools: list[str],
    produced: dict[str, str] | None = None,  # node -> produced JSON string
    loop_nodes: list[str] | None = None,      # nodes that ran twice
    framework: str = "langgraph",
) -> list:
    """Create real OTel spans for RAG tests.

    Uses a local TracerProvider so this never conflicts with the global
    OTel provider configured by test_adapter.py.
    """
    local_provider = TracerProvider()
    local_provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = local_provider.get_tracer("agentlens.rag.test")
    exporter.clear()
    produced = produced or {}
    loop_nodes = loop_nodes or []

    with tracer.start_as_current_span("agentlens.run") as root:
        root.set_attribute("agentlens.framework", framework)
        root.set_attribute("openinference.span.kind", "AGENT")

        # Emit AGENT spans
        all_node_runs = nodes + loop_nodes  # loop_nodes appear twice
        for node in all_node_runs:
            with tracer.start_as_current_span(f"agentlens.agent.{node}") as span:
                span.set_attribute("openinference.span.kind", "AGENT")
                span.set_attribute("agentlens.node", node)
                if node in produced:
                    span.set_attribute("agentlens.produced", produced[node])

        # Emit HANDOFF spans
        for src, tgt in handoffs:
            with tracer.start_as_current_span("agentlens.handoff") as span:
                span.set_attribute("openinference.span.kind", "HANDOFF")
                span.set_attribute("handoff.source", src)
                span.set_attribute("handoff.target", tgt)

        # Emit TOOL spans
        for tool in tools:
            with tracer.start_as_current_span(tool) as span:
                span.set_attribute("openinference.span.kind", "TOOL")
                span.set_attribute("tool.name", tool)

    return list(exporter.get_finished_spans())


# ── FailureSignal extraction tests ────────────────────────────────────────────

class TestFailureSignalExtraction:
    def test_extracts_agents_from_handoff_spans(self, _exporter):
        spans = _make_spans(
            _exporter,
            nodes=["search", "synthesis", "writer"],
            handoffs=[("search", "synthesis"), ("synthesis", "writer")],
            tools=[],
        )
        signal = extract_signals(spans, trace_id="t1")
        assert "search" in signal.agents_observed
        assert "synthesis" in signal.agents_observed
        assert "writer" in signal.agents_observed

    def test_extracts_handoffs(self, _exporter):
        spans = _make_spans(
            _exporter,
            nodes=["search", "writer"],
            handoffs=[("search", "writer")],
            tools=[],
        )
        signal = extract_signals(spans, trace_id="t2")
        assert ("search", "writer") in signal.handoffs

    def test_extracts_tools(self, _exporter):
        spans = _make_spans(
            _exporter,
            nodes=["search"],
            handoffs=[],
            tools=["web_search"],
        )
        signal = extract_signals(spans, trace_id="t3")
        assert "web_search" in signal.tools_used

    def test_loop_count_incremented_for_repeated_node(self, _exporter):
        spans = _make_spans(
            _exporter,
            nodes=["search", "synthesis"],
            handoffs=[("search", "synthesis")],
            tools=[],
            loop_nodes=["synthesis"],  # synthesis runs twice
        )
        signal = extract_signals(spans, trace_id="t4")
        assert signal.loop_counts.get("synthesis", 0) >= 2
        assert signal.delegation_loop_detected is True

    def test_no_loop_when_each_node_runs_once(self, _exporter):
        spans = _make_spans(
            _exporter,
            nodes=["search", "synthesis", "writer"],
            handoffs=[("search", "synthesis"), ("synthesis", "writer")],
            tools=[],
        )
        signal = extract_signals(spans, trace_id="t5")
        assert signal.delegation_loop_detected is False

    def test_context_drop_detected_when_output_shrinks(self, _exporter):
        # search produces large output, synthesis produces small output
        large = '{"sources": [' + ','.join([f'"source{i}"' for i in range(20)]) + ']}'
        small = '{"claims": ["one claim"]}'
        spans = _make_spans(
            _exporter,
            nodes=["search", "synthesis"],
            handoffs=[("search", "synthesis")],
            tools=[],
            produced={"search": large, "synthesis": small},
        )
        signal = extract_signals(spans, trace_id="t6")
        assert signal.context_drop_detected is True

    def test_no_context_drop_when_output_size_preserved(self, _exporter):
        same = '{"data": "' + "x" * 100 + '"}'
        spans = _make_spans(
            _exporter,
            nodes=["search", "synthesis"],
            handoffs=[("search", "synthesis")],
            tools=[],
            produced={"search": same, "synthesis": same},
        )
        signal = extract_signals(spans, trace_id="t7")
        assert signal.context_drop_detected is False

    def test_summary_is_non_empty_string(self, _exporter):
        spans = _make_spans(
            _exporter,
            nodes=["search", "writer"],
            handoffs=[("search", "writer")],
            tools=["web_search"],
        )
        signal = extract_signals(spans, trace_id="t8")
        assert isinstance(signal.summary, str)
        assert len(signal.summary) > 20

    def test_summary_mentions_agents(self, _exporter):
        spans = _make_spans(
            _exporter,
            nodes=["search", "writer"],
            handoffs=[("search", "writer")],
            tools=[],
        )
        signal = extract_signals(spans, trace_id="t9")
        assert "search" in signal.summary
        assert "writer" in signal.summary

    def test_summary_flags_context_drop(self, _exporter):
        large = '{"data": "' + "x" * 500 + '"}'
        small = '{"data": "x"}'
        spans = _make_spans(
            _exporter,
            nodes=["search", "synthesis"],
            handoffs=[("search", "synthesis")],
            tools=[],
            produced={"search": large, "synthesis": small},
        )
        signal = extract_signals(spans, trace_id="t10")
        assert "context_drop=True" in signal.summary

    def test_summary_flags_delegation_loop(self, _exporter):
        spans = _make_spans(
            _exporter,
            nodes=["search", "synthesis"],
            handoffs=[("search", "synthesis")],
            tools=[],
            loop_nodes=["synthesis"],
        )
        signal = extract_signals(spans, trace_id="t11")
        assert "delegation_loop=True" in signal.summary

    def test_trace_id_and_session_id_set(self, _exporter):
        spans = _make_spans(_exporter, nodes=[], handoffs=[], tools=[])
        signal = extract_signals(spans, trace_id="abc-123", session_id="session-1")
        assert signal.trace_id == "abc-123"
        assert signal.session_id == "session-1"

    def test_empty_spans_produces_valid_signal(self, _exporter):
        signal = extract_signals([], trace_id="empty")
        assert signal.agents_observed == []
        assert signal.handoffs == []
        assert signal.context_drop_detected is False
        assert signal.delegation_loop_detected is False
        assert isinstance(signal.summary, str)


# ── context drop detection unit tests ────────────────────────────────────────

class TestContextDropDetection:
    def test_drop_detected_above_threshold(self):
        handoffs = [("a", "b")]
        sizes = {"a": 1000, "b": 100}  # 90% drop
        assert _detect_context_drop(handoffs, sizes) is True

    def test_no_drop_below_threshold(self):
        handoffs = [("a", "b")]
        sizes = {"a": 100, "b": 90}  # 10% drop, below 20% threshold
        assert _detect_context_drop(handoffs, sizes) is False

    def test_no_drop_with_missing_sizes(self):
        handoffs = [("a", "b")]
        sizes = {"a": 100}  # b not measured
        assert _detect_context_drop(handoffs, sizes) is False

    def test_increase_is_not_a_drop(self):
        handoffs = [("a", "b")]
        sizes = {"a": 100, "b": 200}  # b is larger
        assert _detect_context_drop(handoffs, sizes) is False


# ── FailureLibrary tests ───────────────────────────────────────────────────────

@pytest.fixture
def lib():
    return FailureLibrary.ephemeral(embedding_fn=_FakeEmbeddingFunction())


def _make_signal(trace_id: str, summary: str = "", **kwargs) -> FailureSignal:
    return FailureSignal(
        trace_id=trace_id,
        summary=summary or f"Test signal for trace {trace_id}",
        **kwargs,
    )


class TestFailureLibrary:
    def test_empty_library_count_is_zero(self, lib):
        assert lib.count() == 0

    def test_store_increases_count(self, lib):
        lib.store(_make_signal("tr1"))
        assert lib.count() == 1

    def test_store_multiple(self, lib):
        lib.store(_make_signal("tr2"))
        lib.store(_make_signal("tr3"))
        assert lib.count() >= 2

    def test_retrieve_similar_returns_empty_when_library_empty(self):
        empty_lib = FailureLibrary.ephemeral(embedding_fn=_FakeEmbeddingFunction())
        signal = _make_signal("query")
        result = empty_lib.retrieve_similar(signal, k=5)
        assert result == []

    def test_retrieve_similar_returns_stored_signals(self, lib):
        lib.store(_make_signal("tr4", summary="context loss at search synthesis handoff"))
        lib.store(_make_signal("tr5", summary="delegation loop in synthesis agent"))
        query = _make_signal("q", summary="context loss detected")
        results = lib.retrieve_similar(query, k=5)
        assert len(results) >= 1
        assert all(isinstance(r, FailureSignal) for r in results)

    def test_retrieve_similar_respects_k_limit(self, lib):
        for i in range(10, 15):
            lib.store(_make_signal(f"tr{i}", summary=f"failure signal number {i}"))
        query = _make_signal("q2", summary="coordination failure")
        results = lib.retrieve_similar(query, k=2)
        assert len(results) <= 2

    def test_store_is_idempotent_on_trace_id(self, lib):
        before = lib.count()
        lib.store(_make_signal("same-id", summary="original"))
        lib.store(_make_signal("same-id", summary="updated"))
        assert lib.count() == before + 1  # upsert, not insert

    def test_update_diagnosis_sets_diagnosis_field(self, lib):
        lib.store(_make_signal("diag-test"))
        lib.update_diagnosis(
            "diag-test",
            diagnosis="Context loss at synthesis handoff.",
            fix_applied="Raised source cap from 3 to 5.",
            pass_rate_delta=0.32,
        )
        results = lib.retrieve_similar(_make_signal("q", summary="context loss"), k=10)
        match = next((r for r in results if r.trace_id == "diag-test"), None)
        assert match is not None
        assert match.diagnosis == "Context loss at synthesis handoff."
        assert match.fix_applied == "Raised source cap from 3 to 5."
        assert abs(match.pass_rate_delta - 0.32) < 0.001

    def test_update_diagnosis_no_op_for_missing_trace(self, lib):
        # Should not raise even if trace_id doesn't exist
        lib.update_diagnosis("nonexistent-id", diagnosis="whatever")

    def test_retrieve_by_type_context_drop(self, lib):
        lib.store(_make_signal(
            "cd1", summary="context drop run",
            context_drop_detected=True,
        ))
        lib.store(_make_signal(
            "ncd1", summary="clean run",
            context_drop_detected=False,
        ))
        results = lib.retrieve_by_type(context_drop=True)
        assert all(r.context_drop_detected for r in results)

    def test_retrieve_by_type_delegation_loop(self, lib):
        lib.store(_make_signal(
            "dl1", summary="loop run",
            delegation_loop_detected=True,
        ))
        results = lib.retrieve_by_type(delegation_loop=True)
        assert all(r.delegation_loop_detected for r in results)

    def test_signal_roundtrip_preserves_agents(self, lib):
        signal = FailureSignal(
            trace_id="rt1",
            summary="roundtrip test",
            agents_observed=["search", "synthesis", "writer"],
            tools_used=["web_search"],
            context_drop_detected=True,
        )
        lib.store(signal)
        results = lib.retrieve_similar(signal, k=5)
        match = next((r for r in results if r.trace_id == "rt1"), None)
        assert match is not None
        assert "search" in match.agents_observed
        assert "web_search" in match.tools_used


# ── MAST taxonomy tests ────────────────────────────────────────────────────────

class TestMASTTaxonomy:
    def test_all_eight_categories_present(self):
        names = category_names()
        expected = [
            "context_loss", "race_condition", "contradiction_ignored",
            "delegation_loop", "context_bleed", "stale_context",
            "premature_termination", "over_delegation",
        ]
        for name in expected:
            assert name in names, f"Missing MAST category: {name}"

    def test_describe_returns_string_for_known_category(self):
        desc = describe("context_loss")
        assert isinstance(desc, str)
        assert len(desc) > 20

    def test_describe_returns_none_for_unknown_category(self):
        assert describe("made_up_category") is None

    def test_prompt_section_is_non_empty(self):
        section = as_prompt_section()
        assert isinstance(section, str)
        assert "MAST" in section
        assert len(section) > 100

    def test_prompt_section_contains_all_categories(self):
        section = as_prompt_section()
        for name in category_names():
            assert name in section, f"Category {name} missing from prompt"

    def test_all_categories_have_descriptions(self):
        for name in category_names():
            desc = describe(name)
            assert desc is not None
            assert len(desc) > 20
