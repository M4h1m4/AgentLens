"""Diagnosis schema — structured output of the Diagnosis Agent.

A Diagnosis is the answer to: "Given a set of failing eval cases for one
MAST failure mode, what is the root cause and how do you fix it?"

Fields are kept small and concrete so the Improvement Agent (Day 9) can
act on them directly without re-parsing free text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class Diagnosis:
    """Structured diagnosis for one failure mode across one eval run.

    Produced by DiagnosisAgent for each distinct failure_mode in the
    EvalResult's failing cases. The Improvement Agent consumes this to
    generate and rank code fix hypotheses.

    Attributes:
        failure_mode:        MAST category (e.g. "context_loss").
        root_cause:          One sentence — the concrete mechanism that caused
                             the failure (grounded in evidence, not inferred).
        evidence:            Up to three specific observations from eval results
                             or past cases that support the root_cause claim.
        suggested_fix:       A concrete, actionable code-level change or
                             configuration adjustment to resolve the failure.
        confidence:          Diagnosis certainty — "high" if evidence is clear
                             and consistent, "medium" if partial, "low" if sparse.
        similar_past_count:  How many similar past FailureSignals were retrieved
                             from the library (0 means no historical context).
        case_ids:            The EvalResult case_ids this diagnosis covers.
    """

    failure_mode: str
    root_cause: str
    evidence: list[str]
    suggested_fix: str
    confidence: Literal["high", "medium", "low"]
    similar_past_count: int = 0
    case_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "failure_mode": self.failure_mode,
            "root_cause": self.root_cause,
            "evidence": self.evidence,
            "suggested_fix": self.suggested_fix,
            "confidence": self.confidence,
            "similar_past_count": self.similar_past_count,
            "case_ids": self.case_ids,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Diagnosis":
        return cls(
            failure_mode=d["failure_mode"],
            root_cause=d["root_cause"],
            evidence=d["evidence"],
            suggested_fix=d["suggested_fix"],
            confidence=d["confidence"],
            similar_past_count=d.get("similar_past_count", 0),
            case_ids=d.get("case_ids", []),
        )
