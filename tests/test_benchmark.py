"""Tests for Day 5 — Benchmark Gen Agent.

All tests use a fake structured LLM — no API key or network required.
The fake returns pre-built _GenerationBatch objects, exactly mirroring
the IntakeAgent test pattern.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from agentlens.benchmark import (
    BenchmarkCase,
    BenchmarkGenAgent,
    BenchmarkStore,
    BenchmarkSuite,
    _Case,
    _GenerationBatch,
)
from agentlens.rag.failure_signals import FailureSignal
from agentlens.rag.mast import category_names


# ── fake LLM ─────────────────────────────────────────────────────────────────

def _make_case(
    query: str,
    failure_mode: str,
    severity: str = "high",
) -> _Case:
    return _Case(
        query=query,
        expected_criteria=[
            "Output references all sources returned by search",
            "No more than 15% reduction in content size across handoffs",
        ],
        failure_mode=failure_mode,
        severity=severity,
        rationale=f"This query triggers {failure_mode} by design.",
    )


def _make_batch(n: int, failure_mode: str = "context_loss") -> _GenerationBatch:
    """Build a GenerationBatch with n cases."""
    return _GenerationBatch(
        failure_modes_identified=[failure_mode, "delegation_loop"],
        cases=[
            _make_case(f"Query about topic {i} for {failure_mode}", failure_mode)
            for i in range(n)
        ],
    )


class _FakeStructuredLLM:
    """Returns pre-built GenerationBatch objects from a list."""

    def __init__(self, batches: list[_GenerationBatch]) -> None:
        self._batches = iter(batches)
        self._default = _make_batch(25)

    def invoke(self, messages) -> _GenerationBatch:
        try:
            return next(self._batches)
        except StopIteration:
            return self._default


def _agent_with(batches: list[_GenerationBatch]) -> BenchmarkGenAgent:
    return BenchmarkGenAgent(structured_llm=_FakeStructuredLLM(batches))


# ── fixtures ──────────────────────────────────────────────────────────────────

def _signal(
    context_drop: bool = True,
    delegation_loop: bool = True,
    trace_id: str = "trace-001",
    session_id: str = "test-session",
) -> FailureSignal:
    return FailureSignal(
        trace_id=trace_id,
        session_id=session_id,
        agents_observed=["search", "synthesis", "writer"],
        handoffs=[("search", "synthesis"), ("synthesis", "writer")],
        tools_used=["web_search"],
        loop_counts={"synthesis": 2},
        produced_sizes={"search": 1000, "synthesis": 400, "writer": 380},
        context_drop_detected=context_drop,
        delegation_loop_detected=delegation_loop,
        summary=(
            "3 agents observed: search → synthesis → writer. "
            "2 handoffs detected. Context drop at synthesis→writer. "
            "Delegation loop: synthesis ran 2x."
        ),
    )


# ── TestBenchmarkCase ─────────────────────────────────────────────────────────

class TestBenchmarkCase:
    def test_new_assigns_uuid(self):
        c = BenchmarkCase.new(
            query="test query",
            expected_criteria=["output has sources"],
            failure_mode="context_loss",
        )
        assert len(c.case_id) == 36  # UUID format

    def test_default_severity_is_medium(self):
        c = BenchmarkCase.new(
            query="q", expected_criteria=["x"], failure_mode="context_loss"
        )
        assert c.severity == "medium"

    def test_roundtrip(self):
        c = BenchmarkCase.new(
            query="What caused the 2008 financial crisis?",
            expected_criteria=["mentions at least 3 causes", "cites sources"],
            failure_mode="contradiction_ignored",
            severity="high",
            rationale="Conflicting viewpoints on causes should be flagged.",
        )
        assert BenchmarkCase.from_dict(c.to_dict()) == c

    def test_to_dict_contains_all_fields(self):
        c = BenchmarkCase.new(query="q", expected_criteria=["x"], failure_mode="context_loss")
        d = c.to_dict()
        assert set(d.keys()) == {
            "case_id", "query", "expected_criteria", "failure_mode", "severity", "rationale"
        }


# ── TestBenchmarkSuite ────────────────────────────────────────────────────────

class TestBenchmarkSuite:
    def _suite(self, n: int = 5) -> BenchmarkSuite:
        cases = [
            BenchmarkCase.new(
                query=f"Query {i}",
                expected_criteria=["criterion"],
                failure_mode="context_loss" if i % 2 == 0 else "delegation_loop",
                severity="high" if i < 2 else "medium",
            )
            for i in range(n)
        ]
        return BenchmarkSuite.new(
            trace_id="trace-abc",
            session_id="sess-1",
            cases=cases,
            failure_modes=["context_loss", "delegation_loop"],
        )

    def test_new_assigns_uuid(self):
        s = self._suite()
        assert len(s.suite_id) == 36

    def test_roundtrip(self):
        s = self._suite(10)
        assert BenchmarkSuite.from_dict(s.to_dict()).suite_id == s.suite_id
        assert len(BenchmarkSuite.from_dict(s.to_dict()).cases) == 10

    def test_cases_by_mode(self):
        s = self._suite(6)
        cl = s.cases_by_mode("context_loss")
        assert all(c.failure_mode == "context_loss" for c in cl)

    def test_high_severity(self):
        s = self._suite(5)
        hs = s.high_severity()
        assert all(c.severity == "high" for c in hs)

    def test_to_dict_cases_list(self):
        s = self._suite(3)
        d = s.to_dict()
        assert isinstance(d["cases"], list)
        assert len(d["cases"]) == 3


# ── TestBenchmarkGenAgent ─────────────────────────────────────────────────────

class TestBenchmarkGenAgent:
    def test_generates_suite_from_signal(self):
        agent = _agent_with([_make_batch(25)])
        suite = agent.generate(_signal())
        assert isinstance(suite, BenchmarkSuite)

    def test_suite_has_cases(self):
        agent = _agent_with([_make_batch(25)])
        suite = agent.generate(_signal())
        assert len(suite.cases) > 0

    def test_suite_capped_at_max_cases(self):
        # LLM returns 35 cases — should be capped at 30
        agent = _agent_with([_make_batch(35)])
        suite = agent.generate(_signal())
        assert len(suite.cases) <= 30

    def test_case_ids_are_unique(self):
        agent = _agent_with([_make_batch(25)])
        suite = agent.generate(_signal())
        ids = [c.case_id for c in suite.cases]
        assert len(ids) == len(set(ids))

    def test_all_failure_modes_are_valid_mast_categories(self):
        valid = set(category_names())
        agent = _agent_with([_make_batch(25)])
        suite = agent.generate(_signal())
        for case in suite.cases:
            assert case.failure_mode in valid, f"Invalid mode: {case.failure_mode}"

    def test_all_severities_are_valid(self):
        agent = _agent_with([_make_batch(25)])
        suite = agent.generate(_signal())
        for case in suite.cases:
            assert case.severity in {"high", "medium", "low"}

    def test_all_cases_have_expected_criteria(self):
        agent = _agent_with([_make_batch(25)])
        suite = agent.generate(_signal())
        for case in suite.cases:
            assert isinstance(case.expected_criteria, list)
            assert len(case.expected_criteria) >= 1

    def test_suite_trace_id_matches_signal(self):
        signal = _signal(trace_id="my-trace-xyz")
        agent = _agent_with([_make_batch(25)])
        suite = agent.generate(signal)
        assert suite.trace_id == "my-trace-xyz"

    def test_suite_session_id_matches_signal(self):
        signal = _signal(session_id="session-42")
        agent = _agent_with([_make_batch(25)])
        suite = agent.generate(signal)
        assert suite.session_id == "session-42"

    def test_failure_modes_targeted_non_empty(self):
        agent = _agent_with([_make_batch(25)])
        suite = agent.generate(_signal())
        assert len(suite.failure_modes_targeted) > 0

    def test_context_drop_signal_targets_context_loss(self):
        """When context drop is detected, context_loss must be in targeted modes."""
        agent = _agent_with([_make_batch(25)])
        suite = agent.generate(_signal(context_drop=True, delegation_loop=False))
        assert "context_loss" in suite.failure_modes_targeted

    def test_delegation_loop_signal_targets_delegation_loop(self):
        """When delegation loop is detected, delegation_loop must be in targeted modes."""
        agent = _agent_with([_make_batch(25)])
        suite = agent.generate(_signal(context_drop=False, delegation_loop=True))
        assert "delegation_loop" in suite.failure_modes_targeted

    def test_loops_until_min_cases_reached(self):
        """If first batch has <20 cases, agent loops to generate more."""
        # First call returns 10, second returns 15 — total 25 after two passes
        agent = _agent_with([_make_batch(10), _make_batch(15)])
        suite = agent.generate(_signal())
        assert len(suite.cases) >= 20

    def test_stops_after_max_attempts(self):
        """Even if min cases never reached, agent stops at MAX_ATTEMPTS."""
        # Each batch only has 5 cases — will stop after 3 attempts (15 total)
        agent = _agent_with([_make_batch(5), _make_batch(5), _make_batch(5)])
        suite = agent.generate(_signal())
        # Should have stopped after 3 attempts (15 cases < 20, but max_attempts hit)
        assert len(suite.cases) <= 30

    def test_invalid_failure_mode_sanitized(self):
        """Cases with invalid failure_mode get sanitized to a valid MAST category."""
        bad_batch = _GenerationBatch(
            failure_modes_identified=["context_loss"],
            cases=[
                _Case(
                    query="test",
                    expected_criteria=["x"],
                    failure_mode="made_up_failure_mode",  # invalid
                    severity="high",
                    rationale="test",
                )
                for _ in range(25)
            ],
        )
        agent = _agent_with([bad_batch])
        suite = agent.generate(_signal(context_drop=True))
        valid = set(category_names())
        for case in suite.cases:
            assert case.failure_mode in valid

    def test_invalid_severity_sanitized(self):
        """Cases with invalid severity get sanitized to 'medium'."""
        bad_batch = _GenerationBatch(
            failure_modes_identified=["context_loss"],
            cases=[
                _Case(
                    query="test",
                    expected_criteria=["x"],
                    failure_mode="context_loss",
                    severity="CRITICAL",  # invalid
                    rationale="test",
                )
                for _ in range(25)
            ],
        )
        agent = _agent_with([bad_batch])
        suite = agent.generate(_signal())
        for case in suite.cases:
            assert case.severity in {"high", "medium", "low"}

    def test_empty_signal_still_produces_suite(self):
        """Even with no detected failures, the agent generates broad coverage."""
        empty_signal = FailureSignal(
            trace_id="empty-trace",
            session_id="test",
            summary="No agents observed. No handoffs detected.",
        )
        agent = _agent_with([_make_batch(25)])
        suite = agent.generate(empty_signal)
        assert isinstance(suite, BenchmarkSuite)
        assert len(suite.cases) > 0


# ── TestBenchmarkStore ────────────────────────────────────────────────────────

class TestBenchmarkStore:
    def _sample_suite(self, trace_id: str = "trace-001") -> BenchmarkSuite:
        return BenchmarkSuite.new(
            trace_id=trace_id,
            session_id="sess",
            cases=[
                BenchmarkCase.new(
                    query="Sample query",
                    expected_criteria=["criterion a"],
                    failure_mode="context_loss",
                )
            ],
            failure_modes=["context_loss"],
        )

    def test_save_creates_file(self, tmp_path):
        store = BenchmarkStore(base_dir=tmp_path)
        suite = self._sample_suite()
        path = store.save(suite)
        assert path.exists()

    def test_load_roundtrip(self, tmp_path):
        store = BenchmarkStore(base_dir=tmp_path)
        suite = self._sample_suite()
        store.save(suite)
        loaded = store.load(suite.suite_id)
        assert loaded.suite_id == suite.suite_id
        assert loaded.trace_id == suite.trace_id
        assert len(loaded.cases) == len(suite.cases)
        assert loaded.cases[0].query == suite.cases[0].query

    def test_load_latest_returns_most_recent(self, tmp_path):
        store = BenchmarkStore(base_dir=tmp_path)
        s1 = self._sample_suite(trace_id="t-1")
        s2 = self._sample_suite(trace_id="t-1")
        store.save(s1)
        store.save(s2)
        latest = store.load_latest("t-1")
        assert latest.suite_id == s2.suite_id

    def test_load_latest_returns_none_when_not_found(self, tmp_path):
        store = BenchmarkStore(base_dir=tmp_path)
        assert store.load_latest("nonexistent") is None

    def test_list_suites_excludes_pointer_files(self, tmp_path):
        store = BenchmarkStore(base_dir=tmp_path)
        s1 = self._sample_suite("t-a")
        s2 = self._sample_suite("t-b")
        store.save(s1)
        store.save(s2)
        ids = store.list_suites()
        assert s1.suite_id in ids
        assert s2.suite_id in ids
        assert not any(i.startswith("latest_") for i in ids)

    def test_different_trace_ids_have_independent_latest(self, tmp_path):
        store = BenchmarkStore(base_dir=tmp_path)
        sa = self._sample_suite("trace-A")
        sb = self._sample_suite("trace-B")
        store.save(sa)
        store.save(sb)
        assert store.load_latest("trace-A").suite_id == sa.suite_id
        assert store.load_latest("trace-B").suite_id == sb.suite_id

    def test_saved_json_is_valid(self, tmp_path):
        store = BenchmarkStore(base_dir=tmp_path)
        suite = self._sample_suite()
        path = store.save(suite)
        data = json.loads(path.read_text())
        assert "suite_id" in data
        assert "cases" in data
        assert isinstance(data["cases"], list)
