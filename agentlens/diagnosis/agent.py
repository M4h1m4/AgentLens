"""Diagnosis Agent — produces structured diagnoses from failing eval results.

The agent is grounded against two sources to prevent hallucination:

  1. MAST taxonomy (hardcoded in agentlens/rag/mast.py) — constrains which
     failure category a diagnosis can claim and what language to use.

  2. FailureLibrary (ChromaDB) — retrieves the top-5 most similar past
     FailureSignals so the LLM can cite historical precedent rather than
     inventing explanations.

Pipeline per failing failure_mode:

    failing_cases  →  build_context()
                          ↓
                   retrieve_similar()  (FailureLibrary, k=5)
                          ↓
                   _llm_diagnose()     (structured LLM call → _DiagnosisLLMOutput)
                          ↓
                   Diagnosis           (stored + returned)

The structured output schema (_DiagnosisLLMOutput) constrains the LLM to
return root_cause (one sentence), evidence (≤3 items), suggested_fix, and
confidence ("high"|"medium"|"low"). Pydantic validation at the boundary
prevents malformed diagnoses from propagating downstream.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, field_validator
from typing_extensions import Literal

from agentlens.diagnosis.schema import Diagnosis
from agentlens.eval.schema import CaseResult, EvalResult
from agentlens.rag.failure_signals import FailureSignal
from agentlens.rag.mast import as_prompt_section, describe as mast_describe


# ── structured LLM output schema ─────────────────────────────────────────────

class _DiagnosisLLMOutput(BaseModel):
    """What the LLM must return — validated before creating a Diagnosis."""

    root_cause: str
    evidence: list[str]
    suggested_fix: str
    confidence: Literal["high", "medium", "low"]

    @field_validator("evidence")
    @classmethod
    def cap_evidence(cls, v: list[str]) -> list[str]:
        """Never store more than 3 evidence items — keeps diagnosis focused."""
        return v[:3]

    @field_validator("root_cause", "suggested_fix")
    @classmethod
    def non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("root_cause and suggested_fix must be non-empty")
        return v.strip()


# ── system prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = f"""\
You are a multi-agent system failure analyst. Your job is to produce a
structured diagnosis of why a benchmark evaluation flagged failures.

RULES — follow these strictly to avoid hallucination:
1. Root cause must be ONE sentence describing the concrete mechanism.
   Do not speculate beyond what the evidence shows.
2. Evidence must be at most 3 items. Each must be a specific observation
   directly quoted or paraphrased from the provided eval results or past cases.
   Do not invent observations not present in the data.
3. Suggested fix must be a concrete, actionable change (a configuration flag,
   a code-level constraint, or a validation step). Not a vague recommendation.
4. Confidence must be "high" (clear, consistent evidence), "medium" (partial
   evidence with some ambiguity), or "low" (sparse evidence, inference only).
5. You may only classify failures into the MAST categories listed below.
   Do not introduce your own failure categories.

MAST FAILURE TAXONOMY:
{as_prompt_section()}
"""


# ── context builders ──────────────────────────────────────────────────────────

def _build_eval_context(failure_mode: str, cases: list[CaseResult]) -> str:
    """Build a compact text block from failing eval cases for the LLM."""
    lines = [
        f"FAILURE MODE: {failure_mode}",
        f"MAST description: {mast_describe(failure_mode)}",
        f"Failing cases: {len(cases)}",
        "",
        "FAILING CASES (query → failed criteria → checker reason):",
    ]
    for case in cases[:5]:  # cap at 5 to stay within context
        lines.append(f"\n  Query: {case.query}")
        for cr in case.failing_criteria[:3]:
            lines.append(f"    ✗ {cr.criterion}")
            lines.append(f"      Reason: {cr.reason}")
    return "\n".join(lines)


def _build_past_context(similar: list[FailureSignal]) -> str:
    """Summarise the top-k similar past failures for the LLM."""
    if not similar:
        return "PAST SIMILAR CASES: None found in failure library."
    lines = ["PAST SIMILAR CASES (most similar first):"]
    for i, sig in enumerate(similar, 1):
        lines.append(f"\n  [{i}] {sig.summary}")
        if sig.diagnosis:
            lines.append(f"      Prior diagnosis: {sig.diagnosis}")
    return "\n".join(lines)


def _failure_signal_from_cases(
    failure_mode: str,
    cases: list[CaseResult],
    eval_id: str,
) -> FailureSignal:
    """Build a minimal FailureSignal from failing cases for library retrieval.

    The summary is natural-language so ChromaDB can embed it for semantic
    similarity search against stored signals.
    """
    reasons = [
        cr.reason
        for case in cases
        for cr in case.failing_criteria
    ]
    summary = (
        f"Failure mode: {failure_mode}. "
        f"Queries: {[c.query for c in cases[:3]]}. "
        f"Observed: {'; '.join(reasons[:5])}."
    )
    return FailureSignal(
        trace_id=eval_id,
        session_id="diagnosis",
        summary=summary,
    )


# ── LLM call ─────────────────────────────────────────────────────────────────

def _llm_diagnose(
    failure_mode: str,
    eval_context: str,
    past_context: str,
    structured_llm: Any,
) -> _DiagnosisLLMOutput:
    """Call the structured LLM and return a validated diagnosis output."""
    from langchain_core.messages import HumanMessage, SystemMessage

    prompt = (
        f"{eval_context}\n\n"
        f"{past_context}\n\n"
        "Diagnose the root cause of the failure and provide a concrete fix."
    )
    return structured_llm.invoke(
        [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=prompt)]
    )


# ── public agent ──────────────────────────────────────────────────────────────

class DiagnosisAgent:
    """Diagnose all failure modes in a failing EvalResult.

    For each distinct failure_mode in the failing cases:
      1. Retrieves top-k similar past FailureSignals from the library.
      2. Calls the structured LLM to produce root_cause, evidence, fix.
      3. Returns a Diagnosis per failure mode.

    Args:
        structured_llm:   A LangChain LLM wrapped with .with_structured_output(
                          _DiagnosisLLMOutput) — or a fake that returns
                          _DiagnosisLLMOutput directly (for tests).
        library:          FailureLibrary for historical precedent retrieval.
                          If None, no past cases are retrieved.
        retrieve_k:       How many similar past cases to retrieve (default 5).
    """

    def __init__(
        self,
        structured_llm: Any,
        library: Any = None,
        retrieve_k: int = 5,
    ) -> None:
        self._llm = structured_llm
        self._library = library
        self._k = retrieve_k

    def diagnose(self, eval_result: EvalResult) -> list[Diagnosis]:
        """Produce one Diagnosis per failing failure_mode in the EvalResult.

        Returns an empty list if there are no failing cases.
        """
        if not eval_result.failing_cases:
            return []

        # Group failing cases by failure_mode
        by_mode: dict[str, list[CaseResult]] = {}
        for case in eval_result.failing_cases:
            by_mode.setdefault(case.failure_mode, []).append(case)

        diagnoses: list[Diagnosis] = []
        for failure_mode, cases in by_mode.items():
            diagnosis = self._diagnose_mode(
                failure_mode, cases, eval_result.eval_id
            )
            diagnoses.append(diagnosis)

        return diagnoses

    def _diagnose_mode(
        self,
        failure_mode: str,
        cases: list[CaseResult],
        eval_id: str,
    ) -> Diagnosis:
        # Retrieve similar past failures for grounding
        similar: list[FailureSignal] = []
        if self._library is not None:
            signal = _failure_signal_from_cases(failure_mode, cases, eval_id)
            try:
                similar = self._library.retrieve_similar(signal, k=self._k)
            except Exception:
                similar = []  # library unavailable — degrade gracefully

        eval_context = _build_eval_context(failure_mode, cases)
        past_context = _build_past_context(similar)

        llm_output: _DiagnosisLLMOutput = _llm_diagnose(
            failure_mode, eval_context, past_context, self._llm
        )

        return Diagnosis(
            failure_mode=failure_mode,
            root_cause=llm_output.root_cause,
            evidence=llm_output.evidence,
            suggested_fix=llm_output.suggested_fix,
            confidence=llm_output.confidence,
            similar_past_count=len(similar),
            case_ids=[c.case_id for c in cases],
        )
