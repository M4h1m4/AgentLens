"""Tests for Day 6 — Eval Runner, criterion checker, schema, and store.

All tests use:
- FakeExecutor    — returns pre-built spans, no real agent run
- FakeJudgeLLM   — returns pre-built _Verdict objects, no API key needed
- tmp_path        — pytest built-in, no disk cleanup needed

This means the full test suite runs offline with zero external dependencies.
"""

from __future__ import annotations

import json
import math

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from agentlens.benchmark.schema import BenchmarkCase, BenchmarkSuite
from agentlens.eval import (
    CaseResult,
    CriterionResult,
    EvalResult,
    EvalRunner,
    EvalStore,
    check_all_criteria,
    check_criterion,
    wilson_ci,
)
from agentlens.eval.checker import _Verdict


# ── span factory ──────────────────────────────────────────────────────────────

def _make_spans(
    handoffs: list[tuple[str, str]] | None = None,
    produced: dict[str, str] | None = None,
    loop_nodes: list[str] | None = None,
    tools: list[str] | None = None,
) -> list:
    """Build real OTel spans using a local TracerProvider (no global state)."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("eval.test")

    handoffs = handoffs or []
    produced = produced or {}
    loop_nodes = loop_nodes or []
    tools = tools or []

    with tracer.start_as_current_span("agentlens.run") as root:
        root.set_attribute("openinference.span.kind", "AGENT")

        for node, content in produced.items():
            with tracer.start_as_current_span(f"agent.{node}") as span:
                span.set_attribute("openinference.span.kind", "AGENT")
                span.set_attribute("agentlens.node", node)
                span.set_attribute("agentlens.produced", content)

        for node in loop_nodes:
            # Create two spans for the node to simulate a delegation loop
            for i in range(2):
                with tracer.start_as_current_span(f"agent.{node}.run{i}") as span:
                    span.set_attribute("openinference.span.kind", "AGENT")
                    span.set_attribute("agentlens.node", node)
                    span.set_attribute("agentlens.produced", "{}")

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


# ── fake executor ─────────────────────────────────────────────────────────────

class FakeExecutor:
    """Returns pre-built spans without running the real agent."""

    def __init__(
        self,
        spans: list | None = None,
        output: str = "Agent output text",
        trace_id: str = "fake-trace-001",
        raise_on_execute: Exception | None = None,
    ) -> None:
        self._spans = spans or []
        self._output = output
        self._trace_id = trace_id
        self._raise = raise_on_execute

    def execute(self, query: str, session_id: str) -> tuple[str, str, list]:
        if self._raise:
            raise self._raise
        return self._trace_id, self._output, self._spans


# ── fake judge LLM ────────────────────────────────────────────────────────────

class FakeJudgeLLM:
    """Returns pre-built _Verdict objects without calling an LLM."""

    def __init__(self, verdicts: list[_Verdict]) -> None:
        self._verdicts = iter(verdicts)
        self._default = _Verdict(passed=True, reason="default fake verdict")

    def invoke(self, messages) -> _Verdict:
        try:
            return next(self._verdicts)
        except StopIteration:
            return self._default


# ── suite/case factories ──────────────────────────────────────────────────────

def _case(
    query: str = "test query",
    criteria: list[str] | None = None,
    failure_mode: str = "context_loss",
    severity: str = "high",
) -> BenchmarkCase:
    return BenchmarkCase.new(
        query=query,
        expected_criteria=criteria or ["output is complete and accurate"],
        failure_mode=failure_mode,
        severity=severity,
    )


def _suite(cases: list[BenchmarkCase] | None = None) -> BenchmarkSuite:
    return BenchmarkSuite.new(
        trace_id="suite-trace-001",
        session_id="test-session",
        cases=cases or [_case()],
        failure_modes=["context_loss"],
    )


# ── TestWilsonCI ──────────────────────────────────────────────────────────────

class TestWilsonCI:
    def test_perfect_pass_rate(self):
        lo, hi = wilson_ci(10, 10)
        assert lo > 0.7
        assert hi == 1.0

    def test_zero_pass_rate(self):
        lo, hi = wilson_ci(0, 10)
        assert lo == 0.0
        assert hi < 0.3

    def test_zero_trials(self):
        assert wilson_ci(0, 0) == (0.0, 1.0)

    def test_ci_contains_true_rate(self):
        lo, hi = wilson_ci(14, 22)
        assert lo < 14 / 22 < hi

    def test_bounds_are_in_zero_one(self):
        for s, t in [(0, 1), (1, 1), (5, 10), (20, 25), (100, 100)]:
            lo, hi = wilson_ci(s, t)
            assert 0.0 <= lo <= hi <= 1.0

    def test_more_trials_narrower_ci(self):
        lo1, hi1 = wilson_ci(5, 10)
        lo2, hi2 = wilson_ci(50, 100)
        assert (hi2 - lo2) < (hi1 - lo1)


# ── TestCriterionResult ───────────────────────────────────────────────────────

class TestCriterionResult:
    def test_roundtrip(self):
        cr = CriterionResult(
            criterion="output references all sources",
            passed=True,
            reason="All 5 sources mentioned.",
            check_type="llm_judge",
        )
        assert CriterionResult.from_dict(cr.to_dict()) == cr

    def test_to_dict_fields(self):
        cr = CriterionResult("x", True, "y", "deterministic")
        d = cr.to_dict()
        assert set(d.keys()) == {"criterion", "passed", "reason", "check_type"}


# ── TestCaseResult ────────────────────────────────────────────────────────────

class TestCaseResult:
    def _result(self, criteria_passed: list[bool]) -> CaseResult:
        criteria = [
            CriterionResult(f"criterion {i}", p, "reason", "deterministic")
            for i, p in enumerate(criteria_passed)
        ]
        return CaseResult(
            case_id="c-1",
            query="q",
            failure_mode="context_loss",
            passed=all(criteria_passed),
            criteria_results=criteria,
            trace_id="t-1",
            spans_captured=5,
        )

    def test_roundtrip(self):
        r = self._result([True, False])
        assert CaseResult.from_dict(r.to_dict()).passed == r.passed

    def test_failing_criteria(self):
        r = self._result([True, False, True])
        assert len(r.failing_criteria) == 1

    def test_all_pass(self):
        r = self._result([True, True])
        assert r.passed is True
        assert r.failing_criteria == []


# ── TestEvalResult ────────────────────────────────────────────────────────────

class TestEvalResult:
    def _build(self, passed_flags: list[bool], modes: list[str] | None = None) -> EvalResult:
        modes = modes or ["context_loss"] * len(passed_flags)
        results = [
            CaseResult(
                case_id=f"c-{i}",
                query="q",
                failure_mode=modes[i],
                passed=p,
                criteria_results=[CriterionResult("c", p, "r", "deterministic")],
                trace_id="t",
                spans_captured=3,
            )
            for i, p in enumerate(passed_flags)
        ]
        return EvalResult.build(suite_id="s-1", results=results)

    def test_pass_rate(self):
        r = self._build([True, True, False, False])
        assert r.pass_rate == 0.5

    def test_pass_rate_all_pass(self):
        assert self._build([True, True, True]).pass_rate == 1.0

    def test_pass_rate_all_fail(self):
        assert self._build([False, False]).pass_rate == 0.0

    def test_wilson_ci_computed(self):
        r = self._build([True] * 14 + [False] * 8)
        assert 0.0 < r.ci_low < r.ci_high < 1.0

    def test_failures_by_mode(self):
        r = self._build(
            [True, False, False],
            modes=["context_loss", "context_loss", "delegation_loop"],
        )
        assert r.failures_by_mode["context_loss"] == 1
        assert r.failures_by_mode["delegation_loop"] == 1

    def test_failing_cases(self):
        r = self._build([True, False, True])
        assert len(r.failing_cases) == 1

    def test_roundtrip(self):
        r = self._build([True, False, True])
        r2 = EvalResult.from_dict(r.to_dict())
        assert r2.eval_id == r.eval_id
        assert r2.pass_rate == r.pass_rate
        assert len(r2.results) == 3

    def test_ci_overlaps_with(self):
        # Same pass rate → CIs overlap
        r1 = self._build([True] * 14 + [False] * 8)
        r2 = self._build([True] * 14 + [False] * 8)
        assert r1.ci_overlaps_with(r2) is True

    def test_ci_non_overlapping_very_different_rates(self):
        # 0% vs 100% pass rate — CIs should not overlap
        r_low = self._build([False] * 20)
        r_high = self._build([True] * 20)
        assert r_low.ci_overlaps_with(r_high) is False


# ── TestDeterministicChecker ──────────────────────────────────────────────────

class TestDeterministicChecker:
    def test_context_drop_detected(self):
        large = "x" * 1000
        small = "x" * 100
        spans = _make_spans(
            handoffs=[("search", "synthesis")],
            produced={"search": large, "synthesis": small},
        )
        result = check_criterion(
            "No more than 20% reduction in content size across handoffs",
            query="q", output="o", spans=spans,
        )
        assert result.passed is False
        assert result.check_type == "deterministic"

    def test_context_drop_within_threshold(self):
        large = "x" * 1000
        similar = "x" * 900  # 10% drop — within 20% threshold
        spans = _make_spans(
            handoffs=[("search", "synthesis")],
            produced={"search": large, "synthesis": similar},
        )
        result = check_criterion(
            "No more than 20% reduction in content size",
            query="q", output="o", spans=spans,
        )
        assert result.passed is True

    def test_delegation_loop_detected(self):
        spans = _make_spans(
            handoffs=[("search", "synthesis")],
            loop_nodes=["synthesis"],
        )
        result = check_criterion(
            "Synthesis agent ran only once — no delegation loop",
            query="q", output="o", spans=spans,
        )
        assert result.passed is False
        assert result.check_type == "deterministic"
        assert "synthesis" in result.reason

    def test_no_loop_passes(self):
        spans = _make_spans(
            handoffs=[("search", "synthesis")],
            produced={"search": "s", "synthesis": "t"},
        )
        result = check_criterion(
            "Agent ran only once with no loops",
            query="q", output="o", spans=spans,
        )
        assert result.passed is True

    def test_custom_threshold_parsed(self):
        large = "x" * 1000
        medium = "x" * 860  # 14% drop — within 15%, outside 10%
        spans = _make_spans(
            handoffs=[("a", "b")],
            produced={"a": large, "b": medium},
        )
        result_10 = check_criterion(
            "No more than 10% reduction in content size",
            query="q", output="o", spans=spans,
        )
        result_15 = check_criterion(
            "No more than 15% reduction in content size",
            query="q", output="o", spans=spans,
        )
        assert result_10.passed is False
        assert result_15.passed is True

    def test_open_ended_without_judge_is_skipped(self):
        spans = _make_spans()
        result = check_criterion(
            "Output references all original sources by name",
            query="q", output="some output", spans=spans,
            judge_llm=None,
        )
        assert result.check_type == "skipped"
        assert result.passed is True  # optimistic default


# ── TestLLMJudgeChecker ───────────────────────────────────────────────────────

class TestLLMJudgeChecker:
    def test_judge_called_for_open_ended_criterion(self):
        judge = FakeJudgeLLM([_Verdict(passed=True, reason="Sources present.")])
        result = check_criterion(
            "Output mentions all sources by name",
            query="q", output="mentions source A and source B",
            spans=_make_spans(),
            judge_llm=judge,
        )
        assert result.passed is True
        assert result.check_type == "llm_judge"
        assert result.reason == "Sources present."

    def test_judge_can_fail(self):
        judge = FakeJudgeLLM([_Verdict(passed=False, reason="Sources missing.")])
        result = check_criterion(
            "Output mentions all sources by name",
            query="q", output="vague summary",
            spans=_make_spans(),
            judge_llm=judge,
        )
        assert result.passed is False

    def test_deterministic_takes_priority_over_judge(self):
        """Judge should NOT be called when deterministic check applies."""
        call_count = {"n": 0}

        class CountingJudge:
            def invoke(self, messages):
                call_count["n"] += 1
                return _Verdict(passed=True, reason="x")

        spans = _make_spans(loop_nodes=["synthesis"])
        check_criterion(
            "No delegation loops — synthesis ran once",
            query="q", output="o",
            spans=spans,
            judge_llm=CountingJudge(),
        )
        assert call_count["n"] == 0  # deterministic matched first

    def test_check_all_criteria_mixed(self):
        judge = FakeJudgeLLM([
            _Verdict(passed=True, reason="Good output."),
        ])
        spans = _make_spans(
            handoffs=[("a", "b")],
            produced={"a": "x" * 1000, "b": "x" * 100},  # big drop
        )
        results = check_all_criteria(
            criteria=[
                "No more than 20% reduction in content size",  # deterministic → fail
                "Output is comprehensive and well-structured",  # judge → pass
            ],
            query="q", output="output text",
            spans=spans,
            judge_llm=judge,
        )
        assert len(results) == 2
        assert results[0].check_type == "deterministic"
        assert results[0].passed is False
        assert results[1].check_type == "llm_judge"
        assert results[1].passed is True


# ── TestEvalRunner ────────────────────────────────────────────────────────────

class TestEvalRunner:
    def _simple_suite(self, n: int = 3) -> BenchmarkSuite:
        return _suite([
            _case(
                query=f"query {i}",
                criteria=["No more than 20% reduction in content size"],
                failure_mode="context_loss",
            )
            for i in range(n)
        ])

    def test_run_returns_eval_result(self):
        executor = FakeExecutor(spans=_make_spans())
        runner = EvalRunner(executor=executor)
        result = runner.run(self._simple_suite())
        assert isinstance(result, EvalResult)

    def test_result_has_correct_case_count(self):
        executor = FakeExecutor(spans=_make_spans())
        runner = EvalRunner(executor=executor)
        result = runner.run(self._simple_suite(5))
        assert result.total_cases == 5

    def test_suite_id_propagated(self):
        suite = self._simple_suite()
        executor = FakeExecutor(spans=_make_spans())
        runner = EvalRunner(executor=executor)
        result = runner.run(suite)
        assert result.suite_id == suite.suite_id

    def test_all_pass_when_no_drop(self):
        # Spans with no context drop → all deterministic checks pass
        spans = _make_spans(
            handoffs=[("a", "b")],
            produced={"a": "x" * 100, "b": "x" * 95},  # <20% drop
        )
        executor = FakeExecutor(spans=spans)
        runner = EvalRunner(executor=executor)
        result = runner.run(self._simple_suite(3))
        assert result.pass_rate == 1.0

    def test_all_fail_when_big_drop(self):
        spans = _make_spans(
            handoffs=[("a", "b")],
            produced={"a": "x" * 1000, "b": "x" * 100},  # 90% drop
        )
        executor = FakeExecutor(spans=spans)
        runner = EvalRunner(executor=executor)
        result = runner.run(self._simple_suite(3))
        assert result.pass_rate == 0.0

    def test_execution_error_marks_case_failed(self):
        executor = FakeExecutor(raise_on_execute=RuntimeError("agent crashed"))
        runner = EvalRunner(executor=executor)
        result = runner.run(self._simple_suite(2))
        assert result.pass_rate == 0.0
        for r in result.results:
            assert not r.passed
            assert "error" in r.criteria_results[0].check_type

    def test_trace_id_recorded_per_case(self):
        executor = FakeExecutor(spans=_make_spans(), trace_id="my-trace")
        runner = EvalRunner(executor=executor)
        result = runner.run(self._simple_suite(2))
        for r in result.results:
            assert r.trace_id == "my-trace"

    def test_spans_captured_count(self):
        spans = _make_spans(
            handoffs=[("a", "b")],
            produced={"a": "x", "b": "y"},
        )
        executor = FakeExecutor(spans=spans)
        runner = EvalRunner(executor=executor)
        result = runner.run(self._simple_suite(1))
        assert result.results[0].spans_captured == len(spans)

    def test_failures_by_mode_aggregated(self):
        spans_drop = _make_spans(
            handoffs=[("a", "b")],
            produced={"a": "x" * 1000, "b": "x" * 50},
        )
        suite = _suite([
            _case(
                criteria=["No more than 20% reduction in content size"],
                failure_mode="context_loss",
            ),
            _case(
                criteria=["No more than 20% reduction in content size"],
                failure_mode="delegation_loop",
            ),
        ])
        executor = FakeExecutor(spans=spans_drop)
        runner = EvalRunner(executor=executor)
        result = runner.run(suite)
        assert "context_loss" in result.failures_by_mode
        assert "delegation_loop" in result.failures_by_mode

    def test_with_judge_llm(self):
        judge = FakeJudgeLLM([
            _Verdict(passed=True, reason="Good."),
            _Verdict(passed=False, reason="Missing sources."),
            _Verdict(passed=True, reason="Good."),
        ])
        suite = _suite([
            _case(criteria=["Output is comprehensive"], failure_mode="context_loss")
            for _ in range(3)
        ])
        executor = FakeExecutor(spans=_make_spans())
        runner = EvalRunner(executor=executor, judge_llm=judge)
        result = runner.run(suite)
        assert result.passed_cases == 2
        assert result.pass_rate == pytest.approx(2 / 3)


# ── TestEvalStore ─────────────────────────────────────────────────────────────

class TestEvalStore:
    def _sample_result(self, suite_id: str = "suite-1") -> EvalResult:
        return EvalResult.build(
            suite_id=suite_id,
            results=[
                CaseResult(
                    case_id="c-1",
                    query="q",
                    failure_mode="context_loss",
                    passed=True,
                    criteria_results=[CriterionResult("c", True, "r", "deterministic")],
                    trace_id="t-1",
                    spans_captured=3,
                )
            ],
        )

    def test_save_creates_file(self, tmp_path):
        store = EvalStore(base_dir=tmp_path)
        result = self._sample_result()
        path = store.save(result)
        assert path.exists()

    def test_load_roundtrip(self, tmp_path):
        store = EvalStore(base_dir=tmp_path)
        result = self._sample_result()
        store.save(result)
        loaded = store.load(result.eval_id)
        assert loaded.eval_id == result.eval_id
        assert loaded.pass_rate == result.pass_rate
        assert len(loaded.results) == 1

    def test_load_latest(self, tmp_path):
        store = EvalStore(base_dir=tmp_path)
        r1 = self._sample_result("suite-A")
        r2 = self._sample_result("suite-A")
        store.save(r1)
        store.save(r2)
        latest = store.load_latest("suite-A")
        assert latest.eval_id == r2.eval_id

    def test_load_latest_none_when_missing(self, tmp_path):
        store = EvalStore(base_dir=tmp_path)
        assert store.load_latest("nonexistent") is None

    def test_list_results_excludes_pointers(self, tmp_path):
        store = EvalStore(base_dir=tmp_path)
        r1 = self._sample_result("s-1")
        r2 = self._sample_result("s-2")
        store.save(r1)
        store.save(r2)
        ids = store.list_results()
        assert r1.eval_id in ids
        assert r2.eval_id in ids
        assert not any(i.startswith("latest_") for i in ids)

    def test_saved_json_is_valid(self, tmp_path):
        store = EvalStore(base_dir=tmp_path)
        result = self._sample_result()
        path = store.save(result)
        data = json.loads(path.read_text())
        assert "eval_id" in data
        assert "results" in data
        assert "pass_rate" in data
        assert "ci_low" in data
        assert "ci_high" in data
