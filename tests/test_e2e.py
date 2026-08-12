"""End-to-end integration tests — components chained together, no API calls.

Each test class exercises a pipeline slice. All external dependencies are replaced
with fakes so the suite runs offline and deterministically:
  - FakeListChatModel      — for the reference agent (no Anthropic API)
  - Inline fake structured LLMs — for Intake, BenchmarkGen, Diagnosis
  - InMemorySpanExporter   — OTel spans in memory (no Phoenix)
  - FailureLibrary.ephemeral() — ChromaDB in memory (no disk)
  - InProcessExecutor      — EvalRunner runs agent in-process (no E2B sandbox)

Test slices
-----------
  E2E-1  Intake → Adapter → Reconcile
  E2E-2  Adapter → FailureSignal extraction
  E2E-3  FailureSignal → BenchmarkGen
  E2E-4  BenchmarkSuite → EvalRunner → EvalResult
  E2E-5  EvalResult → DiagnosisAgent → Diagnoses
  E2E-6  Diagnosis → Library → pipeline.update_library_after_improvement (D8.1)
  E2E-7  Full pipeline (all components chained)
"""
from __future__ import annotations

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from agentlens.adapter import setup_otel, instrument_langchain, traced_invoke
from agentlens.benchmark.gen_agent import BenchmarkGenAgent, _GenerationBatch, _Case
from agentlens.benchmark.schema import BenchmarkCase, BenchmarkSuite
from agentlens.diagnosis import DiagnosisAgent, _DiagnosisLLMOutput
from agentlens.eval.runner import EvalRunner, InProcessExecutor
from agentlens.eval.schema import CaseResult, CriterionResult, EvalResult
from agentlens.improvement.schema import ImprovementResult
from agentlens.intake.agent import IntakeAgent, _ExtractionResult, _ExtractedAgent
from agentlens.intake.profile import AgentProfile
from agentlens.intake.reconciler import reconcile
from agentlens.pipeline import update_library_after_improvement
from agentlens.rag.failure_signals import FailureSignal, extract_signals
from agentlens.rag.library import FailureLibrary, _FakeEmbeddingFunction
from reference_agent.agents.writer import reset_bleed_cache
from reference_agent.config import AgentConfig, SeededFailureConfig
from reference_agent.graph import build_graph


# ── module-level OTel setup ───────────────────────────────────────────────────
# One exporter for the whole module, cleared between tests.

_EXPORTER: InMemorySpanExporter
_AGENT_LLM = FakeListChatModel(responses=["[auto-generated summary]"])


@pytest.fixture(scope="module", autouse=True)
def _otel_setup():
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


# ── shared helpers ────────────────────────────────────────────────────────────

def _run_reference_agent(
    query: str,
    failures: SeededFailureConfig | None = None,
    session_id: str = "e2e",
) -> tuple[str, dict, list]:
    """Run the reference agent and return (trace_id, final_state, spans)."""
    config = AgentConfig(failures=failures or SeededFailureConfig())
    app = build_graph(config, _AGENT_LLM)
    initial = {"query": query, "session_id": session_id, "loop_count": 0}
    trace_id, final_state = traced_invoke(app, initial, session_id=session_id)
    spans = list(_EXPORTER.get_finished_spans())
    return trace_id, final_state, spans


def _make_suite(
    failure_mode: str = "context_loss",
    criteria: list[str] | None = None,
    n: int = 3,
) -> BenchmarkSuite:
    """Build a small benchmark suite with deterministic-checkable criteria."""
    effective_criteria = criteria or [
        "context drop does not exceed 20% across any handoff",
        "each agent runs only once — no delegation loop",
    ]
    cases = [
        BenchmarkCase.new(
            query=f"research topic {i}",
            expected_criteria=effective_criteria,
            failure_mode=failure_mode,
            severity="high",
            rationale="E2E test case",
        )
        for i in range(n)
    ]
    return BenchmarkSuite.new("trace-e2e", "e2e", cases, [failure_mode])


def _make_eval_result_with_failures(failure_mode: str = "context_loss") -> EvalResult:
    """Build an EvalResult that has one failing and one passing case."""
    failing = CaseResult(
        case_id="case-fail",
        query="test query",
        failure_mode=failure_mode,
        passed=False,
        criteria_results=[
            CriterionResult(
                "context drop must not exceed 20%", False,
                "40% drop detected at search→synthesis", "deterministic",
            ),
        ],
        trace_id="trace-fail",
        spans_captured=5,
    )
    passing = CaseResult(
        case_id="case-pass",
        query="clean query",
        failure_mode=failure_mode,
        passed=True,
        criteria_results=[
            CriterionResult(
                "context drop must not exceed 20%", True,
                "No measurable size reduction", "deterministic",
            ),
        ],
        trace_id="trace-pass",
        spans_captured=5,
    )
    return EvalResult.build(suite_id="suite-e2e", results=[failing, passing])


def _make_improvement_result(
    promoted: bool,
    hypothesis: str = "Raise synthesis source cap from 3 to 5",
) -> ImprovementResult:
    """Build a fake ImprovementResult for library update tests."""
    baseline = EvalResult.build("s-base", [
        CaseResult(
            "c0", "q", "context_loss", False,
            [CriterionResult("cr", False, "fail", "deterministic")], "t0", 3,
        )
    ])
    variant_results = [
        CaseResult(
            f"c{i}", "q", "context_loss", i < 4,
            [CriterionResult("cr", i < 4, "ok" if i < 4 else "fail", "deterministic")],
            f"tv{i}", 3,
        )
        for i in range(5)
    ]
    variant = EvalResult.build("s-var", variant_results)
    return ImprovementResult(
        failure_mode="context_loss",
        promoted=promoted,
        iterations_run=1,
        cost_dollars=0.01,
        hypothesis=hypothesis,
        variant_files={},
        baseline_eval=baseline,
        variant_eval=variant,
        regression_detected=False,
        reason="Test result",
    )


# ── fake structured LLMs ──────────────────────────────────────────────────────

class _FakeIntakeLLM:
    """Returns a ready-for-draft _ExtractionResult on first invoke."""

    def invoke(self, messages: list) -> _ExtractionResult:
        return _ExtractionResult(
            framework="langgraph",
            framework_confidence="high",
            agent_count=3,
            agent_count_confidence="high",
            agents=[
                _ExtractedAgent(name="search", role="searches for sources", confidence="high"),
                _ExtractedAgent(name="synthesis", role="synthesizes sources", confidence="high"),
                _ExtractedAgent(name="writer", role="writes the report", confidence="high"),
            ],
            tools=["web_search"],
            tools_confidence="high",
            expected_output="research report",
            ready_for_draft=True,
        )


class _FakeBenchmarkLLM:
    """Returns 20 deterministic-criteria cases in one batch."""

    def invoke(self, messages: list) -> _GenerationBatch:
        cases = [
            _Case(
                query=f"research query about topic {i}",
                expected_criteria=[
                    "context drop does not exceed 20% across any handoff",
                    "each agent runs only once — no delegation loop",
                ],
                failure_mode="context_loss",
                severity="high",
                rationale="Tests that synthesis does not silently drop sources",
            )
            for i in range(20)
        ]
        return _GenerationBatch(
            failure_modes_identified=["context_loss", "race_condition"],
            cases=cases,
        )


class _FakeDiagnosisLLM:
    """Returns a fixed _DiagnosisLLMOutput."""

    def invoke(self, messages: list) -> _DiagnosisLLMOutput:
        return _DiagnosisLLMOutput(
            root_cause="Synthesis agent silently caps key claims to 3 instead of passing all 5 sources",
            evidence=[
                "Search produced 5 sources",
                "Synthesis produced 3 key claims",
                "88% size drop detected at search→synthesis handoff",
            ],
            suggested_fix="Remove the source cap in synthesis.py — raise limit from 3 to 5",
            confidence="high",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# E2E-1: Intake → Adapter → Reconcile
# ═══════════════════════════════════════════════════════════════════════════════

class TestE2E_IntakeAdapterReconcile:
    """Developer describes the reference agent → Intake produces a draft profile
    → Adapter runs the agent → reconcile() merges trace evidence into the profile."""

    def test_reconcile_upgrades_framework_source_to_trace_observed(self):
        intake = IntakeAgent(_structured_llm=_FakeIntakeLLM())
        intake.start("I have a 3-agent LangGraph research system.")
        profile = intake.get_profile()
        assert profile.framework.source == "user_stated"

        _, _, spans = _run_reference_agent("renewable energy")
        updated = reconcile(profile, spans)

        assert updated.framework.value == "langgraph"
        assert updated.framework.source == "trace_observed"
        assert updated.framework.confidence == "high"

    def test_reconcile_fills_agent_count_from_trace(self):
        intake = IntakeAgent(_structured_llm=_FakeIntakeLLM())
        intake.start("My agent system does research.")
        profile = intake.get_profile()

        _, _, spans = _run_reference_agent("climate change")
        updated = reconcile(profile, spans)

        assert updated.agent_count.value == 3
        assert updated.agent_count.source == "trace_observed"

    def test_reconcile_fills_handoffs_from_trace(self):
        intake = IntakeAgent(_structured_llm=_FakeIntakeLLM())
        intake.start("Research system.")
        profile = intake.get_profile()

        _, _, spans = _run_reference_agent("ocean pollution")
        updated = reconcile(profile, spans)

        handoffs = updated.delegation_patterns.value
        assert ("search", "synthesis") in handoffs
        assert ("synthesis", "writer") in handoffs

    def test_reconcile_marks_profile_as_confirmed(self):
        intake = IntakeAgent(_structured_llm=_FakeIntakeLLM())
        intake.start("Research agent.")
        profile = intake.get_profile()
        assert profile.reconciled_with_trace is False

        _, _, spans = _run_reference_agent("AI trends")
        updated = reconcile(profile, spans)

        assert updated.reconciled_with_trace is True
        assert updated.version == profile.version + 1

    def test_reconcile_does_not_mutate_original_profile(self):
        intake = IntakeAgent(_structured_llm=_FakeIntakeLLM())
        intake.start("Research system.")
        profile = intake.get_profile()
        original_version = profile.version

        _, _, spans = _run_reference_agent("quantum computing")
        reconcile(profile, spans)  # discard result

        assert profile.version == original_version
        assert profile.reconciled_with_trace is False


# ═══════════════════════════════════════════════════════════════════════════════
# E2E-2: Adapter → FailureSignal extraction
# ═══════════════════════════════════════════════════════════════════════════════

class TestE2E_AdapterFailureSignal:
    """Adapter runs the reference agent with seeded failures → FailureSignal
    correctly classifies what happened in the trace."""

    def test_context_drop_detected_from_seeded_trace(self):
        trace_id, _, spans = _run_reference_agent(
            "renewable energy",
            failures=SeededFailureConfig(context_loss=True),
        )
        signal = extract_signals(spans, trace_id=trace_id)
        assert signal.context_drop_detected is True
        assert signal.trace_id == trace_id

    def test_delegation_loop_detected_from_seeded_trace(self):
        trace_id, _, spans = _run_reference_agent(
            "exhaustive review of grid storage",
            failures=SeededFailureConfig(delegation_loop=True),
        )
        signal = extract_signals(spans, trace_id=trace_id)
        assert signal.delegation_loop_detected is True

    def test_clean_run_no_delegation_loop_detected(self):
        # NOTE: context_drop_detected fires on clean runs too because the writer
        # node's output (~802B) is naturally ~30% smaller than synthesis (~1140B),
        # exceeding the 20% heuristic threshold. This is a known limitation of the
        # size-based heuristic — it cannot distinguish design-time compression from
        # actual information loss. Only delegation_loop_detected is asserted here.
        trace_id, _, spans = _run_reference_agent(
            "renewable energy",
            failures=SeededFailureConfig.clean(),
        )
        signal = extract_signals(spans, trace_id=trace_id)
        assert signal.delegation_loop_detected is False

    def test_signal_captures_all_three_agent_nodes(self):
        trace_id, _, spans = _run_reference_agent("battery technology")
        signal = extract_signals(spans, trace_id=trace_id)
        for node in ("search", "synthesis", "writer"):
            assert node in signal.agents_observed, f"'{node}' missing from signal.agents_observed"

    def test_signal_summary_is_non_empty_and_embeddable(self):
        trace_id, _, spans = _run_reference_agent("solar panels")
        signal = extract_signals(spans, trace_id=trace_id)
        assert isinstance(signal.summary, str)
        assert len(signal.summary) > 20

    def test_signal_trace_id_matches_adapter_output(self):
        trace_id, _, spans = _run_reference_agent("wind power")
        signal = extract_signals(spans, trace_id=trace_id)
        assert signal.trace_id == trace_id


# ═══════════════════════════════════════════════════════════════════════════════
# E2E-3: FailureSignal → BenchmarkGen
# ═══════════════════════════════════════════════════════════════════════════════

class TestE2E_FailureSignalBenchmarkGen:
    """FailureSignal extracted from a trace → BenchmarkGenAgent generates a
    targeted suite of test cases without hitting the real LLM."""

    def test_suite_meets_minimum_case_count(self):
        signal = FailureSignal(
            trace_id="trace-bg-1",
            summary="Context drop at search→synthesis handoff",
            context_drop_detected=True,
        )
        suite = BenchmarkGenAgent(structured_llm=_FakeBenchmarkLLM()).generate(signal)
        assert len(suite.cases) >= 20

    def test_suite_targets_detected_failure_mode(self):
        signal = FailureSignal(
            trace_id="trace-bg-2",
            summary="Context drop detected",
            context_drop_detected=True,
        )
        suite = BenchmarkGenAgent(structured_llm=_FakeBenchmarkLLM()).generate(signal)
        assert "context_loss" in suite.failure_modes_targeted

    def test_suite_inherits_trace_id_from_signal(self):
        signal = FailureSignal(
            trace_id="my-trace-xyz",
            summary="Delegation loop in synthesis",
            delegation_loop_detected=True,
        )
        suite = BenchmarkGenAgent(structured_llm=_FakeBenchmarkLLM()).generate(signal)
        assert suite.trace_id == "my-trace-xyz"

    def test_all_cases_have_valid_mast_failure_modes(self):
        from agentlens.rag.mast import category_names
        signal = FailureSignal(
            trace_id="trace-bg-3",
            summary="Multiple failures detected",
            context_drop_detected=True,
            delegation_loop_detected=True,
        )
        suite = BenchmarkGenAgent(structured_llm=_FakeBenchmarkLLM()).generate(signal)
        valid_modes = set(category_names())
        for case in suite.cases:
            assert case.failure_mode in valid_modes, \
                f"Case has invalid failure_mode: '{case.failure_mode}'"

    def test_all_cases_have_non_empty_criteria(self):
        signal = FailureSignal(trace_id="trace-bg-4", summary="test signal")
        suite = BenchmarkGenAgent(structured_llm=_FakeBenchmarkLLM()).generate(signal)
        for case in suite.cases:
            assert len(case.expected_criteria) >= 1
            for criterion in case.expected_criteria:
                assert criterion.strip() != "", "Empty criterion string found"

    def test_signal_with_no_detected_failures_still_generates_suite(self):
        """Even with no flags set, BenchmarkGen produces broad coverage cases."""
        signal = FailureSignal(trace_id="trace-bg-5", summary="no specific failure detected")
        suite = BenchmarkGenAgent(structured_llm=_FakeBenchmarkLLM()).generate(signal)
        assert len(suite.cases) >= 20


# ═══════════════════════════════════════════════════════════════════════════════
# E2E-4: BenchmarkSuite → EvalRunner → EvalResult
# ═══════════════════════════════════════════════════════════════════════════════

class TestE2E_EvalRunner:
    """BenchmarkSuite with deterministic criteria → EvalRunner runs each case
    against the reference agent in-process → EvalResult captures pass/fail."""

    def _executor(self, failures: SeededFailureConfig | None = None) -> InProcessExecutor:
        config = AgentConfig(failures=failures or SeededFailureConfig())
        app = build_graph(config, _AGENT_LLM)
        return InProcessExecutor(
            app=app,
            exporter=_EXPORTER,
            output_key="report",
            extra_initial_state={"loop_count": 0},
        )

    def test_context_drop_criterion_checked_deterministically(self):
        suite = _make_suite(
            criteria=["context drop does not exceed 20% across any handoff"],
            n=2,
        )
        result = EvalRunner(executor=self._executor()).run(suite)
        assert result.total_cases == 2
        for case_result in result.results:
            drop_crs = [
                cr for cr in case_result.criteria_results
                if "context drop" in cr.criterion.lower()
            ]
            assert drop_crs[0].check_type == "deterministic"

    def test_context_drop_criterion_fails_on_seeded_context_loss(self):
        # context_loss seeded: synthesis caps to 3 key_claims, causing ~88% size drop
        suite = _make_suite(
            criteria=["context drop does not exceed 20% across any handoff"],
            n=3,
        )
        result = EvalRunner(executor=self._executor(SeededFailureConfig(context_loss=True))).run(suite)
        failing_drop_criteria = [
            cr
            for case_result in result.results
            for cr in case_result.criteria_results
            if "context drop" in cr.criterion.lower() and not cr.passed
        ]
        assert len(failing_drop_criteria) > 0, \
            "Expected context_loss to fail the context drop criterion"

    def test_loop_criterion_passes_on_clean_run(self):
        # The 20%-threshold context_drop criterion also fires on clean runs because
        # the writer node naturally compresses synthesis output by ~30%. Use the loop
        # criterion instead — it is unambiguous: no seeded loop means no repeated nodes.
        suite = _make_suite(
            criteria=["each agent runs only once — no delegation loop"],
            n=2,
        )
        result = EvalRunner(executor=self._executor(SeededFailureConfig.clean())).run(suite)
        passing_loop_criteria = [
            cr
            for case_result in result.results
            for cr in case_result.criteria_results
            if "delegation loop" in cr.criterion.lower() and cr.passed
        ]
        assert len(passing_loop_criteria) > 0, \
            "Expected clean run to pass the delegation-loop criterion"

    def test_loop_criterion_fails_on_seeded_delegation_loop(self):
        loop_suite = BenchmarkSuite.new(
            "t", "s",
            [BenchmarkCase.new(
                query="exhaustive review of grid storage",
                expected_criteria=["each agent runs only once — no delegation loop"],
                failure_mode="delegation_loop",
            )],
            ["delegation_loop"],
        )
        result = EvalRunner(
            executor=self._executor(SeededFailureConfig(delegation_loop=True))
        ).run(loop_suite)
        failing = [
            cr for case_r in result.results
            for cr in case_r.criteria_results
            if not cr.passed
        ]
        assert len(failing) > 0, "Expected delegation loop to fail the loop criterion"

    def test_eval_result_stats_are_internally_consistent(self):
        suite = _make_suite(
            criteria=["context drop does not exceed 20% across any handoff"],
            n=4,
        )
        result = EvalRunner(executor=self._executor()).run(suite)
        assert result.total_cases == 4
        assert result.passed_cases + len(result.failing_cases) == result.total_cases
        assert abs(result.pass_rate - result.passed_cases / result.total_cases) < 1e-9
        assert 0.0 <= result.ci_low <= result.ci_high <= 1.0

    def test_failure_modes_recorded_in_eval_result(self):
        suite = _make_suite(
            failure_mode="context_loss",
            criteria=["context drop does not exceed 20% across any handoff"],
            n=2,
        )
        result = EvalRunner(
            executor=self._executor(SeededFailureConfig(context_loss=True))
        ).run(suite)
        if result.failing_cases:
            assert "context_loss" in result.failures_by_mode


# ═══════════════════════════════════════════════════════════════════════════════
# E2E-5: EvalResult → DiagnosisAgent → Diagnoses
# ═══════════════════════════════════════════════════════════════════════════════

class TestE2E_EvalDiagnosis:
    """Failing EvalResult → DiagnosisAgent produces structured Diagnoses grounded
    in the MAST taxonomy. Library retrieval is also exercised."""

    def test_one_diagnosis_per_failure_mode(self):
        eval_result = _make_eval_result_with_failures("context_loss")
        diagnoses = DiagnosisAgent(structured_llm=_FakeDiagnosisLLM()).diagnose(eval_result)
        assert len(diagnoses) == 1
        assert diagnoses[0].failure_mode == "context_loss"

    def test_diagnosis_fields_are_populated_from_llm(self):
        eval_result = _make_eval_result_with_failures("context_loss")
        [diag] = DiagnosisAgent(structured_llm=_FakeDiagnosisLLM()).diagnose(eval_result)
        assert diag.root_cause.strip() != ""
        assert diag.suggested_fix.strip() != ""
        assert diag.confidence in ("high", "medium", "low")
        assert len(diag.evidence) <= 3

    def test_diagnosis_case_ids_match_failing_cases(self):
        eval_result = _make_eval_result_with_failures("context_loss")
        [diag] = DiagnosisAgent(structured_llm=_FakeDiagnosisLLM()).diagnose(eval_result)
        failing_ids = {c.case_id for c in eval_result.failing_cases}
        assert set(diag.case_ids).issubset(failing_ids)

    def test_no_diagnoses_for_fully_passing_eval(self):
        passing = CaseResult(
            case_id="p1", query="q", failure_mode="context_loss",
            passed=True,
            criteria_results=[CriterionResult("cr", True, "ok", "deterministic")],
            trace_id="t", spans_captured=3,
        )
        eval_result = EvalResult.build("s", [passing])
        assert DiagnosisAgent(structured_llm=_FakeDiagnosisLLM()).diagnose(eval_result) == []

    def test_two_failure_modes_produce_two_diagnoses(self):
        context_loss_case = CaseResult(
            case_id="cl-1", query="q1", failure_mode="context_loss", passed=False,
            criteria_results=[CriterionResult("cr", False, "drop", "deterministic")],
            trace_id="t1", spans_captured=3,
        )
        loop_case = CaseResult(
            case_id="dl-1", query="q2", failure_mode="delegation_loop", passed=False,
            criteria_results=[CriterionResult("cr", False, "loop", "deterministic")],
            trace_id="t2", spans_captured=3,
        )
        eval_result = EvalResult.build("s", [context_loss_case, loop_case])
        diagnoses = DiagnosisAgent(structured_llm=_FakeDiagnosisLLM()).diagnose(eval_result)
        assert len(diagnoses) == 2
        modes = {d.failure_mode for d in diagnoses}
        assert modes == {"context_loss", "delegation_loop"}

    def test_proven_library_signals_reflected_in_similar_past_count(self):
        """D8.2: only signals with pass_rate_delta > 0 are counted."""
        library = FailureLibrary.ephemeral(embedding_fn=_FakeEmbeddingFunction())
        proven = FailureSignal(trace_id="proven-1", summary="Context loss at synthesis handoff")
        proven.diagnosis = "Source cap too low"
        proven.fix_applied = "Raised cap from 3 to 5"
        proven.pass_rate_delta = 0.4
        library.store(proven)

        unresolved = FailureSignal(trace_id="unresolved-1", summary="Context loss suspected")
        library.store(unresolved)  # no pass_rate_delta → filtered by D8.2

        eval_result = _make_eval_result_with_failures("context_loss")
        agent = DiagnosisAgent(
            structured_llm=_FakeDiagnosisLLM(),
            library=library,
            retrieve_k=5,
        )
        [diag] = agent.diagnose(eval_result)
        assert diag.similar_past_count == 1  # only the proven signal, not the unresolved one


# ═══════════════════════════════════════════════════════════════════════════════
# E2E-6: Diagnosis → Library → pipeline.update_library_after_improvement (D8.1)
# ═══════════════════════════════════════════════════════════════════════════════

class TestE2E_LibraryPipeline:
    """After ImprovementAgent promotes a fix, update_library_after_improvement()
    overwrites the weak stored hypothesis with the proven one (D8.1).
    Future retrievals return ground truth, not the original speculation."""

    def test_promoted_result_overwrites_weak_hypothesis(self):
        library = FailureLibrary.ephemeral(embedding_fn=_FakeEmbeddingFunction())
        signal = FailureSignal(trace_id="tr-D81-a", summary="context drop in synthesis")
        signal.diagnosis = "weak initial hypothesis"
        library.store(signal)

        result = _make_improvement_result(
            promoted=True,
            hypothesis="Raise synthesis cap from 3 to 5 — confirmed fix",
        )
        update_library_after_improvement(library, "tr-D81-a", result)

        retrieved = library.retrieve_similar(signal, k=5)
        match = next((r for r in retrieved if r.trace_id == "tr-D81-a"), None)
        assert match is not None
        assert match.diagnosis == "Raise synthesis cap from 3 to 5 — confirmed fix"

    def test_unpromoted_result_does_not_overwrite_library(self):
        library = FailureLibrary.ephemeral(embedding_fn=_FakeEmbeddingFunction())
        signal = FailureSignal(trace_id="tr-D81-b", summary="context drop")
        signal.diagnosis = "original weak hypothesis"
        library.store(signal)

        result = _make_improvement_result(promoted=False, hypothesis="a failed attempt")
        update_library_after_improvement(library, "tr-D81-b", result)

        retrieved = library.retrieve_similar(signal, k=5)
        match = next((r for r in retrieved if r.trace_id == "tr-D81-b"), None)
        assert match is not None
        assert match.diagnosis == "original weak hypothesis"  # unchanged

    def test_pass_rate_delta_stored_after_promotion(self):
        library = FailureLibrary.ephemeral(embedding_fn=_FakeEmbeddingFunction())
        signal = FailureSignal(trace_id="tr-D81-c", summary="context loss")
        library.store(signal)

        result = _make_improvement_result(promoted=True)
        update_library_after_improvement(library, "tr-D81-c", result)

        retrieved = library.retrieve_similar(signal, k=5)
        match = next((r for r in retrieved if r.trace_id == "tr-D81-c"), None)
        assert match is not None
        assert match.pass_rate_delta is not None
        assert match.pass_rate_delta > 0

    def test_updated_signal_now_survives_d8_2_filter(self):
        """After update, the signal has pass_rate_delta > 0 and will be retrieved
        by DiagnosisAgent (D8.2 filter only keeps proven signals)."""
        library = FailureLibrary.ephemeral(embedding_fn=_FakeEmbeddingFunction())
        signal = FailureSignal(trace_id="tr-D81-d", summary="context loss at synthesis")
        library.store(signal)  # no pass_rate_delta yet → filtered by D8.2

        # Before update: no proven signals → similar_past_count == 0
        eval_result = _make_eval_result_with_failures("context_loss")
        agent = DiagnosisAgent(structured_llm=_FakeDiagnosisLLM(), library=library, retrieve_k=5)
        [diag_before] = agent.diagnose(eval_result)
        assert diag_before.similar_past_count == 0

        # Promote a fix → overwrite in library
        result = _make_improvement_result(promoted=True)
        update_library_after_improvement(library, "tr-D81-d", result)

        # After update: signal now has pass_rate_delta > 0 → survives D8.2
        [diag_after] = agent.diagnose(eval_result)
        assert diag_after.similar_past_count == 1


# ═══════════════════════════════════════════════════════════════════════════════
# E2E-7: Full pipeline
# ═══════════════════════════════════════════════════════════════════════════════

class TestE2E_FullPipeline:
    """Complete AgentLens pipeline from developer description to library update.

    Intake → Adapter → Reconcile → FailureSignal → BenchmarkGen →
    EvalRunner → DiagnosisAgent → pipeline.update_library_after_improvement

    Uses the reference agent with context_loss=True as the target agent.
    """

    def test_full_pipeline_context_loss_end_to_end(self):
        # ── Step 1: Intake ────────────────────────────────────────────────────
        intake = IntakeAgent(_structured_llm=_FakeIntakeLLM())
        intake.start("I have a 3-agent LangGraph research system — output loses context.")
        profile = intake.get_profile()
        assert profile.framework.value == "langgraph"

        # ── Step 2: Adapter — run agent with seeded context_loss ──────────────
        config = AgentConfig(failures=SeededFailureConfig(context_loss=True))
        app = build_graph(config, _AGENT_LLM)
        initial = {"query": "renewable energy sources", "session_id": "e2e-full", "loop_count": 0}
        trace_id, _ = traced_invoke(app, initial, session_id="e2e-full")
        spans = list(_EXPORTER.get_finished_spans())
        assert spans, "No spans captured from adapter run"

        # ── Step 3: Reconcile profile with trace ──────────────────────────────
        profile = reconcile(profile, spans)
        assert profile.reconciled_with_trace is True
        assert profile.agent_count.value == 3
        assert ("search", "synthesis") in profile.delegation_patterns.value

        # ── Step 4: Extract FailureSignal from spans ──────────────────────────
        signal = extract_signals(spans, trace_id=trace_id)
        assert signal.context_drop_detected is True
        assert len(signal.agents_observed) == 3

        # ── Step 5: Store signal in library ───────────────────────────────────
        library = FailureLibrary.ephemeral(embedding_fn=_FakeEmbeddingFunction())
        library.store(signal)
        assert library.count() == 1

        # ── Step 6: BenchmarkGen — generate targeted suite ────────────────────
        suite = BenchmarkGenAgent(structured_llm=_FakeBenchmarkLLM()).generate(signal)
        assert len(suite.cases) >= 20
        assert "context_loss" in suite.failure_modes_targeted
        assert suite.trace_id == trace_id

        # ── Step 7: Eval — run a small deterministic sub-suite ────────────────
        # Use a 3-case suite with context_drop criterion for speed
        small_suite = _make_suite(
            failure_mode="context_loss",
            criteria=["context drop does not exceed 20% across any handoff"],
            n=3,
        )
        executor = InProcessExecutor(
            app=app,
            exporter=_EXPORTER,
            output_key="report",
            extra_initial_state={"loop_count": 0},
        )
        eval_result = EvalRunner(executor=executor).run(small_suite)
        assert eval_result.total_cases == 3
        assert len(eval_result.failing_cases) > 0, \
            "Expected context_loss seeding to produce failing cases"

        # ── Step 8: Diagnose failing cases ────────────────────────────────────
        diagnoses = DiagnosisAgent(
            structured_llm=_FakeDiagnosisLLM(),
            library=library,
            retrieve_k=5,
        ).diagnose(eval_result)
        assert len(diagnoses) >= 1
        assert diagnoses[0].failure_mode == "context_loss"
        assert diagnoses[0].root_cause.strip() != ""

        # ── Step 9: Update library with proven fix (D8.1) ─────────────────────
        improvement = ImprovementResult(
            failure_mode="context_loss",
            promoted=True,
            iterations_run=1,
            cost_dollars=0.01,
            hypothesis="Raised synthesis cap from 3 to 5 sources",
            variant_files={},
            baseline_eval=eval_result,
            variant_eval=EvalResult.build("v", [
                CaseResult("v1", "q", "context_loss", True,
                           [CriterionResult("cr", True, "pass", "deterministic")], "tv", 3)
            ]),
            regression_detected=False,
            reason="Wilson CIs non-overlapping. No regressions.",
        )
        update_library_after_improvement(library, trace_id, improvement)

        # ── Step 10: Verify library reflects ground truth ─────────────────────
        retrieved = library.retrieve_similar(signal, k=5)
        match = next((r for r in retrieved if r.trace_id == trace_id), None)
        assert match is not None, "Original signal not found in library after update"
        assert match.diagnosis == "Raised synthesis cap from 3 to 5 sources"
        assert match.pass_rate_delta > 0

    def test_full_pipeline_clean_run_produces_no_loop_failures(self):
        """With no seeded failures, delegation-loop criteria should all pass.

        NOTE: The 20%-threshold context_drop criterion is NOT used here. On a clean
        run the writer node's output (~802B) is naturally ~30% smaller than synthesis
        (~1140B), which exceeds the heuristic threshold. This is a known limitation:
        the size-based heuristic cannot distinguish design-time compression from actual
        information loss. Bug tracked: size_heuristic_writer_false_positive.
        """
        # Intake
        intake = IntakeAgent(_structured_llm=_FakeIntakeLLM())
        intake.start("Research agent system, no known issues.")
        profile = intake.get_profile()

        # Adapter — clean run
        config = AgentConfig(failures=SeededFailureConfig.clean())
        app = build_graph(config, _AGENT_LLM)
        trace_id, _ = traced_invoke(
            app,
            {"query": "renewable energy", "session_id": "clean", "loop_count": 0},
            session_id="clean",
        )
        spans = list(_EXPORTER.get_finished_spans())

        # Reconcile
        profile = reconcile(profile, spans)
        assert profile.reconciled_with_trace is True

        # Signal — clean run must show no delegation loop
        signal = extract_signals(spans, trace_id=trace_id)
        assert signal.delegation_loop_detected is False

        # Eval — loop criterion passes cleanly (no seeded loop)
        suite = _make_suite(
            criteria=["each agent runs only once — no delegation loop"],
            n=3,
        )
        executor = InProcessExecutor(
            app=app, exporter=_EXPORTER,
            output_key="report", extra_initial_state={"loop_count": 0},
        )
        eval_result = EvalRunner(executor=executor).run(suite)
        assert eval_result.total_cases == 3
        passing_loop = [
            cr
            for case_r in eval_result.results
            for cr in case_r.criteria_results
            if "delegation loop" in cr.criterion.lower() and cr.passed
        ]
        assert len(passing_loop) == 3
