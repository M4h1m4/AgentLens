"""Tests for the Diagnosis Agent (Day 8).

The DiagnosisAgent is tested with:
  - A FakeStructuredLLM that returns fixed _DiagnosisLLMOutput objects
  - A FakeLibrary that returns a configurable list of FailureSignals
  - Real EvalResult / CaseResult / CriterionResult objects (no mocking)

This keeps tests fast (no API calls), deterministic, and focused on the
logic that prevents hallucination:
  - Evidence is capped at 3 items
  - Diagnoses are only produced for failure modes that actually appear in
    the failing cases
  - Library retrieval errors degrade gracefully (no diagnosis crash)
  - Empty EvalResult → empty diagnosis list
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from agentlens.diagnosis import Diagnosis, DiagnosisAgent, _DiagnosisLLMOutput
from agentlens.diagnosis.agent import (
    _build_eval_context,
    _build_past_context,
    _failure_signal_from_cases,
)
from agentlens.eval.schema import CaseResult, CriterionResult, EvalResult
from agentlens.rag.failure_signals import FailureSignal


# ── test helpers ──────────────────────────────────────────────────────────────

def _criterion(passed: bool, reason: str = "test reason") -> CriterionResult:
    return CriterionResult(
        criterion="Test criterion",
        passed=passed,
        reason=reason,
        check_type="deterministic",
    )


def _case(
    failure_mode: str,
    passed: bool = False,
    query: str = "test query",
    reasons: list[str] | None = None,
) -> CaseResult:
    criteria = []
    if not passed:
        for r in (reasons or ["observed failure"]):
            criteria.append(_criterion(passed=False, reason=r))
    else:
        criteria.append(_criterion(passed=True))
    return CaseResult(
        case_id=str(uuid.uuid4()),
        query=query,
        failure_mode=failure_mode,
        passed=passed,
        criteria_results=criteria,
        trace_id="t1",
        spans_captured=5,
    )


def _eval_result(cases: list[CaseResult]) -> EvalResult:
    return EvalResult.build(suite_id=str(uuid.uuid4()), results=cases)


def _fixed_output(
    root_cause: str = "Test root cause",
    evidence: list[str] | None = None,
    suggested_fix: str = "Flip the context_loss flag to False",
    confidence: str = "high",
) -> _DiagnosisLLMOutput:
    return _DiagnosisLLMOutput(
        root_cause=root_cause,
        evidence=evidence or ["Evidence A", "Evidence B"],
        suggested_fix=suggested_fix,
        confidence=confidence,
    )


class _FakeLLM:
    """Returns a fixed _DiagnosisLLMOutput per .invoke() call."""

    def __init__(self, outputs: list[_DiagnosisLLMOutput] | None = None) -> None:
        self._outputs = outputs or [_fixed_output()]
        self._call_count = 0

    def invoke(self, messages: list) -> _DiagnosisLLMOutput:
        output = self._outputs[self._call_count % len(self._outputs)]
        self._call_count += 1
        return output

    @property
    def call_count(self) -> int:
        return self._call_count


class _FakeLibrary:
    """Returns a configurable list of FailureSignals from retrieve_similar."""

    def __init__(self, signals: list[FailureSignal] | None = None) -> None:
        self._signals = signals or []

    def retrieve_similar(self, signal: FailureSignal, k: int = 5) -> list[FailureSignal]:
        return self._signals[:k]


def _signal(summary: str, diagnosis: str | None = None) -> FailureSignal:
    sig = FailureSignal(trace_id="past-1", session_id="test", summary=summary)
    sig.diagnosis = diagnosis
    return sig


# ═══════════════════════════════════════════════════════════════════════════════
# DiagnosisSchema
# ═══════════════════════════════════════════════════════════════════════════════

class TestDiagnosisSchema:

    def test_to_dict_roundtrip(self):
        d = Diagnosis(
            failure_mode="context_loss",
            root_cause="Synthesis cap silently drops sources",
            evidence=["Search produced 5 sources", "Only 3 in output"],
            suggested_fix="Raise synthesis_source_cap to match search_source_limit",
            confidence="high",
            similar_past_count=3,
            case_ids=["abc", "def"],
        )
        restored = Diagnosis.from_dict(d.to_dict())
        assert restored.failure_mode == d.failure_mode
        assert restored.root_cause == d.root_cause
        assert restored.evidence == d.evidence
        assert restored.suggested_fix == d.suggested_fix
        assert restored.confidence == d.confidence
        assert restored.similar_past_count == d.similar_past_count
        assert restored.case_ids == d.case_ids

    def test_confidence_values(self):
        for level in ("high", "medium", "low"):
            d = Diagnosis(
                failure_mode="delegation_loop",
                root_cause="Loop without terminator",
                evidence=[],
                suggested_fix="Add loop guard",
                confidence=level,
            )
            assert d.confidence == level

    def test_default_case_ids_is_empty_list(self):
        d = Diagnosis(
            failure_mode="race_condition",
            root_cause="r",
            evidence=[],
            suggested_fix="s",
            confidence="low",
        )
        assert d.case_ids == []
        assert d.similar_past_count == 0


# ═══════════════════════════════════════════════════════════════════════════════
# _DiagnosisLLMOutput validation
# ═══════════════════════════════════════════════════════════════════════════════

class TestDiagnosisLLMOutput:

    def test_valid_output_accepted(self):
        out = _DiagnosisLLMOutput(
            root_cause="Context cap drops sources silently",
            evidence=["A", "B", "C"],
            suggested_fix="Remove cap",
            confidence="high",
        )
        assert out.confidence == "high"

    def test_evidence_capped_at_three(self):
        out = _DiagnosisLLMOutput(
            root_cause="r",
            evidence=["a", "b", "c", "d", "e"],
            suggested_fix="s",
            confidence="medium",
        )
        assert len(out.evidence) == 3

    def test_empty_root_cause_rejected(self):
        with pytest.raises(Exception):
            _DiagnosisLLMOutput(
                root_cause="",
                evidence=[],
                suggested_fix="fix",
                confidence="low",
            )

    def test_invalid_confidence_rejected(self):
        with pytest.raises(Exception):
            _DiagnosisLLMOutput(
                root_cause="r",
                evidence=[],
                suggested_fix="s",
                confidence="very_high",  # type: ignore[arg-type]
            )

    def test_whitespace_root_cause_rejected(self):
        with pytest.raises(Exception):
            _DiagnosisLLMOutput(
                root_cause="   ",
                evidence=[],
                suggested_fix="s",
                confidence="low",
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Context builders
# ═══════════════════════════════════════════════════════════════════════════════

class TestContextBuilders:

    def test_eval_context_contains_failure_mode(self):
        cases = [_case("context_loss", reasons=["2 sources dropped"])]
        ctx = _build_eval_context("context_loss", cases)
        assert "context_loss" in ctx
        assert "2 sources dropped" in ctx

    def test_eval_context_caps_at_five_cases(self):
        cases = [_case("delegation_loop") for _ in range(10)]
        ctx = _build_eval_context("delegation_loop", cases)
        # Should only mention up to 5 queries
        assert ctx.count("Query:") <= 5

    def test_eval_context_includes_mast_description(self):
        cases = [_case("context_loss")]
        ctx = _build_eval_context("context_loss", cases)
        # MAST description for context_loss mentions "truncated" or "Information"
        assert "Information" in ctx or "truncated" in ctx or "absent" in ctx

    def test_past_context_no_library(self):
        ctx = _build_past_context([])
        assert "None found" in ctx

    def test_past_context_includes_summaries(self):
        signals = [
            _signal("synthesis dropped 3 sources", diagnosis="Raise the cap"),
            _signal("loop detected in search node"),
        ]
        ctx = _build_past_context(signals)
        assert "synthesis dropped 3 sources" in ctx
        assert "Raise the cap" in ctx
        assert "loop detected" in ctx

    def test_failure_signal_from_cases_summary_contains_mode(self):
        cases = [_case("race_condition", query="q1"), _case("race_condition", query="q2")]
        sig = _failure_signal_from_cases("race_condition", cases, eval_id="e1")
        assert "race_condition" in sig.summary
        assert "q1" in sig.summary

    def test_unknown_failure_mode_in_context(self):
        """Unknown MAST category degrades gracefully (mast_describe returns empty)."""
        cases = [_case("not_a_real_mode")]
        ctx = _build_eval_context("not_a_real_mode", cases)
        assert "not_a_real_mode" in ctx


# ═══════════════════════════════════════════════════════════════════════════════
# DiagnosisAgent
# ═══════════════════════════════════════════════════════════════════════════════

class TestDiagnosisAgent:

    def test_empty_eval_returns_no_diagnoses(self):
        result = _eval_result([_case("context_loss", passed=True)])
        agent = DiagnosisAgent(structured_llm=_FakeLLM())
        assert agent.diagnose(result) == []

    def test_single_failure_mode_produces_one_diagnosis(self):
        result = _eval_result([
            _case("context_loss"),
            _case("context_loss"),
        ])
        agent = DiagnosisAgent(structured_llm=_FakeLLM())
        diagnoses = agent.diagnose(result)
        assert len(diagnoses) == 1
        assert diagnoses[0].failure_mode == "context_loss"

    def test_two_failure_modes_produce_two_diagnoses(self):
        result = _eval_result([
            _case("context_loss"),
            _case("delegation_loop"),
        ])
        llm = _FakeLLM(outputs=[
            _fixed_output(root_cause="Drop in context"),
            _fixed_output(root_cause="Loop without terminator"),
        ])
        agent = DiagnosisAgent(structured_llm=llm)
        diagnoses = agent.diagnose(result)
        assert len(diagnoses) == 2
        modes = {d.failure_mode for d in diagnoses}
        assert modes == {"context_loss", "delegation_loop"}

    def test_llm_called_once_per_failure_mode(self):
        result = _eval_result([
            _case("context_loss"),
            _case("context_loss"),
            _case("race_condition"),
        ])
        llm = _FakeLLM(outputs=[_fixed_output()] * 5)
        agent = DiagnosisAgent(structured_llm=llm)
        agent.diagnose(result)
        assert llm.call_count == 2  # one per distinct failure_mode

    def test_diagnosis_contains_correct_case_ids(self):
        case1 = _case("context_loss")
        case2 = _case("context_loss")
        result = _eval_result([case1, case2])
        agent = DiagnosisAgent(structured_llm=_FakeLLM())
        [diag] = agent.diagnose(result)
        assert set(diag.case_ids) == {case1.case_id, case2.case_id}

    def test_diagnosis_fields_from_llm_output(self):
        result = _eval_result([_case("context_loss")])
        output = _fixed_output(
            root_cause="Cap silently truncates sources",
            evidence=["5 fetched", "3 used"],
            suggested_fix="Align caps",
            confidence="high",
        )
        agent = DiagnosisAgent(structured_llm=_FakeLLM(outputs=[output]))
        [diag] = agent.diagnose(result)
        assert diag.root_cause == "Cap silently truncates sources"
        assert diag.evidence == ["5 fetched", "3 used"]
        assert diag.suggested_fix == "Align caps"
        assert diag.confidence == "high"

    def test_passes_only_cases_not_in_results(self):
        """Passing cases must not trigger a diagnosis."""
        result = _eval_result([
            _case("context_loss", passed=True),
            _case("context_loss", passed=True),
        ])
        agent = DiagnosisAgent(structured_llm=_FakeLLM())
        assert agent.diagnose(result) == []

    def test_mixed_pass_fail_only_diagnoses_failing_mode(self):
        result = _eval_result([
            _case("context_loss", passed=True),
            _case("delegation_loop", passed=False),
        ])
        agent = DiagnosisAgent(structured_llm=_FakeLLM())
        diagnoses = agent.diagnose(result)
        assert len(diagnoses) == 1
        assert diagnoses[0].failure_mode == "delegation_loop"


# ═══════════════════════════════════════════════════════════════════════════════
# Library integration
# ═══════════════════════════════════════════════════════════════════════════════

class TestLibraryIntegration:

    def test_no_library_similar_past_count_is_zero(self):
        result = _eval_result([_case("context_loss")])
        agent = DiagnosisAgent(structured_llm=_FakeLLM(), library=None)
        [diag] = agent.diagnose(result)
        assert diag.similar_past_count == 0

    def test_library_count_reflected_in_diagnosis(self):
        signals = [_signal(f"past failure {i}") for i in range(3)]
        result = _eval_result([_case("context_loss")])
        agent = DiagnosisAgent(
            structured_llm=_FakeLLM(),
            library=_FakeLibrary(signals),
            retrieve_k=5,
        )
        [diag] = agent.diagnose(result)
        assert diag.similar_past_count == 3

    def test_retrieve_k_limits_signals(self):
        """retrieve_k=2 means at most 2 signals returned even if library has more."""
        signals = [_signal(f"signal {i}") for i in range(10)]
        result = _eval_result([_case("delegation_loop")])
        agent = DiagnosisAgent(
            structured_llm=_FakeLLM(),
            library=_FakeLibrary(signals),
            retrieve_k=2,
        )
        [diag] = agent.diagnose(result)
        assert diag.similar_past_count == 2

    def test_library_failure_degrades_gracefully(self):
        """A broken library must not crash the diagnosis."""
        class _BrokenLibrary:
            def retrieve_similar(self, signal, k=5):
                raise RuntimeError("ChromaDB connection failed")

        result = _eval_result([_case("context_loss")])
        agent = DiagnosisAgent(
            structured_llm=_FakeLLM(),
            library=_BrokenLibrary(),
        )
        [diag] = agent.diagnose(result)
        # Diagnosis still produced, just with no historical context
        assert diag.similar_past_count == 0
        assert diag.root_cause  # non-empty


# ═══════════════════════════════════════════════════════════════════════════════
# Anti-hallucination properties
# ═══════════════════════════════════════════════════════════════════════════════

class TestAntiHallucinationProperties:
    """Structural assertions that constrain what the diagnosis can claim."""

    def test_evidence_never_exceeds_three_items(self):
        """LLM returning 10 evidence items must be capped to 3."""
        result = _eval_result([_case("context_loss")])
        output = _DiagnosisLLMOutput(
            root_cause="r",
            evidence=[f"item {i}" for i in range(10)],
            suggested_fix="s",
            confidence="medium",
        )
        agent = DiagnosisAgent(structured_llm=_FakeLLM(outputs=[output]))
        [diag] = agent.diagnose(result)
        assert len(diag.evidence) <= 3

    def test_failure_mode_matches_failing_cases(self):
        """Diagnosis failure_mode must come from the eval data, not the LLM."""
        result = _eval_result([_case("race_condition")])
        agent = DiagnosisAgent(structured_llm=_FakeLLM())
        [diag] = agent.diagnose(result)
        # The mode comes from CaseResult.failure_mode, never from the LLM output
        assert diag.failure_mode == "race_condition"

    def test_case_ids_are_real_case_ids_from_eval(self):
        """Case IDs in diagnosis must match actual CaseResult.case_id values."""
        case = _case("context_bleed", query="topic A")
        result = _eval_result([case])
        agent = DiagnosisAgent(structured_llm=_FakeLLM())
        [diag] = agent.diagnose(result)
        assert case.case_id in diag.case_ids

    def test_no_diagnosis_for_modes_not_in_eval(self):
        """DiagnosisAgent must not hallucinate diagnoses for failure modes not in the data."""
        result = _eval_result([_case("delegation_loop")])
        llm = _FakeLLM(outputs=[_fixed_output()] * 10)
        agent = DiagnosisAgent(structured_llm=llm)
        diagnoses = agent.diagnose(result)
        modes = {d.failure_mode for d in diagnoses}
        # Only delegation_loop appears in the eval — no extras
        assert modes == {"delegation_loop"}
        assert llm.call_count == 1

    def test_eval_context_caps_case_display_at_five(self):
        """Context fed to LLM never shows more than 5 cases (context window guard)."""
        cases = [_case("context_loss", query=f"query {i}") for i in range(20)]
        ctx = _build_eval_context("context_loss", cases)
        # Count "Query:" occurrences — should be ≤ 5
        assert ctx.count("Query:") <= 5

    def test_all_five_seeded_modes_diagnosable(self):
        """All five seeded failure modes from the reference agent must produce a Diagnosis."""
        modes = [
            "context_loss",
            "race_condition",
            "contradiction_ignored",
            "delegation_loop",
            "context_bleed",
        ]
        result = _eval_result([_case(m) for m in modes])
        outputs = [_fixed_output(root_cause=f"root cause for {m}") for m in modes]
        agent = DiagnosisAgent(structured_llm=_FakeLLM(outputs=outputs))
        diagnoses = agent.diagnose(result)
        assert len(diagnoses) == 5
        diagnosed_modes = {d.failure_mode for d in diagnoses}
        assert diagnosed_modes == set(modes)
