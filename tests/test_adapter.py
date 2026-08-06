"""Day 2 (reworked) — adapter produces OTel spans that make coordination failures
observable. Uses InMemorySpanExporter so no Phoenix instance is needed.
"""

from __future__ import annotations

import json
import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from agentlens.adapter import setup_otel, instrument_langchain, traced_invoke
from reference_agent.agents.writer import reset_bleed_cache
from reference_agent.config import AgentConfig, SeededFailureConfig
from reference_agent.graph import build_graph

LLM = FakeListChatModel(responses=["[auto-generated summary]"])

_EXPORTER: InMemorySpanExporter


@pytest.fixture(scope="module", autouse=True)
def _otel_setup():
    """Configure OTel with in-memory exporter once for the whole module."""
    global _EXPORTER
    _EXPORTER = InMemorySpanExporter()
    setup_otel(exporter=_EXPORTER)
    instrument_langchain()
    yield


@pytest.fixture(autouse=True)
def _reset():
    reset_bleed_cache()
    _EXPORTER.clear()
    yield
    reset_bleed_cache()


def _run(query, failures=None, session_id="s1"):
    config = AgentConfig(failures=failures or SeededFailureConfig())
    app = build_graph(config, LLM)
    initial = {"query": query, "session_id": session_id, "loop_count": 0}
    return traced_invoke(app, initial, session_id=session_id)


def _spans_for(trace_id: str):
    return [
        s for s in _EXPORTER.get_finished_spans()
        if format(s.context.trace_id, "032x") == trace_id
    ]


def _by_kind(spans, kind: str):
    return [
        s for s in spans
        if (s.attributes or {}).get("openinference.span.kind") == kind
    ]


def _handoff_spans(spans):
    return _by_kind(spans, "HANDOFF")


def _agent_spans(spans):
    return _by_kind(spans, "AGENT")


def _agent_for_node(spans, node: str):
    return [
        s for s in _agent_spans(spans)
        if (s.attributes or {}).get("agentlens.node") == node
    ]


# ── trace shape ──────────────────────────────────────────────────────────────

def test_root_span_exists():
    trace_id, state = _run("renewable energy")
    spans = _spans_for(trace_id)
    root = [s for s in spans if s.name == "agentlens.run"]
    assert root, "no root agentlens.run span found"
    assert root[0].attributes["agentlens.query"] == "renewable energy"
    assert state["report"]


def test_all_three_agent_spans_present():
    trace_id, _ = _run("renewable energy")
    spans = _spans_for(trace_id)
    for node in ("search", "synthesis", "writer"):
        assert _agent_for_node(spans, node), f"missing AGENT span for node '{node}'"


def test_both_handoff_spans_present():
    trace_id, _ = _run("renewable energy")
    spans = _spans_for(trace_id)
    handoffs = {
        (s.attributes["handoff.source"], s.attributes["handoff.target"])
        for s in _handoff_spans(spans)
    }
    assert ("search", "synthesis") in handoffs
    assert ("synthesis", "writer") in handoffs


def test_agent_spans_carry_produced_state():
    trace_id, _ = _run("renewable energy")
    spans = _spans_for(trace_id)
    search_spans = _agent_for_node(spans, "search")
    assert search_spans
    produced_raw = search_spans[0].attributes.get("agentlens.produced", "{}")
    produced = json.loads(produced_raw)
    assert "sources" in produced


def test_framework_recorded_on_root_span():
    trace_id, _ = _run("renewable energy")
    spans = _spans_for(trace_id)
    root = [s for s in spans if s.name == "agentlens.run"][0]
    assert root.attributes["agentlens.framework"] == "langgraph"


# ── failures observable from the trace ───────────────────────────────────────

def test_context_loss_visible_in_trace():
    trace_id, _ = _run(
        "renewable energy",
        failures=SeededFailureConfig(context_loss=True),
    )
    spans = _spans_for(trace_id)

    search_produced = json.loads(
        _agent_for_node(spans, "search")[0].attributes["agentlens.produced"]
    )
    synth_produced = json.loads(
        _agent_for_node(spans, "synthesis")[0].attributes["agentlens.produced"]
    )

    n_search_sources = len(search_produced["sources"])
    n_synth_claims = len(synth_produced["synthesis_complete"]["key_claims"])

    assert n_search_sources == 5
    assert n_synth_claims == 3   # silent drop is now on camera


def test_delegation_loop_visible_in_trace():
    trace_id, _ = _run(
        "exhaustive review of grid storage",
        failures=SeededFailureConfig(delegation_loop=True),
    )
    spans = _spans_for(trace_id)
    assert len(_agent_for_node(spans, "synthesis")) > 1


def test_clean_run_shows_no_divergence():
    trace_id, _ = _run(
        "renewable energy",
        failures=SeededFailureConfig.clean(),
    )
    spans = _spans_for(trace_id)

    search_produced = json.loads(
        _agent_for_node(spans, "search")[0].attributes["agentlens.produced"]
    )
    synth_produced = json.loads(
        _agent_for_node(spans, "synthesis")[0].attributes["agentlens.produced"]
    )

    assert len(search_produced["sources"]) == 5
    assert len(synth_produced["synthesis_complete"]["key_claims"]) == 5
    assert len(_agent_for_node(spans, "synthesis")) == 1
