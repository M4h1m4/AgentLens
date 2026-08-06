"""EvalResult schema — the structured output of one benchmark suite run.

CriterionResult  — pass/fail for one expected_criterion in one BenchmarkCase
CaseResult       — aggregate of all criteria for one case
EvalResult       — aggregate of all cases in one BenchmarkSuite run

Wilson score confidence interval is computed here because the Improvement
Agent (Day 9) needs it to decide whether a code variant's pass rate
improvement is statistically real.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


# ── Wilson CI ─────────────────────────────────────────────────────────────────

def wilson_ci(
    successes: int,
    trials: int,
    z: float = 1.96,  # 95% confidence
) -> tuple[float, float]:
    """Return the Wilson score 95% confidence interval for a pass rate.

    More reliable than a naive p ± margin for small sample sizes.
    Used by the Improvement Agent to decide if a fix is statistically real:
    non-overlapping CIs between before/after → improvement is genuine.

    Args:
        successes: Number of passing cases.
        trials:    Total cases run.
        z:         Z-score for the desired confidence level (1.96 = 95%).

    Returns:
        (lower_bound, upper_bound) both in [0.0, 1.0].
    """
    if trials == 0:
        return (0.0, 1.0)
    p = successes / trials
    n = trials
    center = (p + z ** 2 / (2 * n)) / (1 + z ** 2 / n)
    margin = (z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))) / (
        1 + z ** 2 / n
    )
    return (max(0.0, center - margin), min(1.0, center + margin))


# ── result dataclasses ────────────────────────────────────────────────────────

@dataclass
class CriterionResult:
    """Pass/fail for one expected_criterion string."""

    criterion: str
    passed: bool
    reason: str
    check_type: str  # "deterministic" | "llm_judge" | "skipped"

    def to_dict(self) -> dict[str, Any]:
        return {
            "criterion": self.criterion,
            "passed": self.passed,
            "reason": self.reason,
            "check_type": self.check_type,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CriterionResult":
        return cls(**d)


@dataclass
class CaseResult:
    """Outcome of running one BenchmarkCase against the agent."""

    case_id: str
    query: str
    failure_mode: str            # MAST category this case targeted
    passed: bool                 # True only if ALL criteria passed
    criteria_results: list[CriterionResult]
    trace_id: str
    spans_captured: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "query": self.query,
            "failure_mode": self.failure_mode,
            "passed": self.passed,
            "criteria_results": [c.to_dict() for c in self.criteria_results],
            "trace_id": self.trace_id,
            "spans_captured": self.spans_captured,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CaseResult":
        return cls(
            case_id=d["case_id"],
            query=d["query"],
            failure_mode=d["failure_mode"],
            passed=d["passed"],
            criteria_results=[CriterionResult.from_dict(c) for c in d["criteria_results"]],
            trace_id=d["trace_id"],
            spans_captured=d["spans_captured"],
        )

    @property
    def failing_criteria(self) -> list[CriterionResult]:
        return [c for c in self.criteria_results if not c.passed]


@dataclass
class EvalResult:
    """Outcome of running a full BenchmarkSuite against the agent."""

    eval_id: str
    suite_id: str
    created_at: datetime
    results: list[CaseResult]

    # Aggregate stats
    total_cases: int
    passed_cases: int
    pass_rate: float                      # passed_cases / total_cases
    ci_low: float                         # Wilson 95% CI lower bound
    ci_high: float                        # Wilson 95% CI upper bound
    failures_by_mode: dict[str, int]      # MAST category → count of failing cases

    @classmethod
    def build(cls, suite_id: str, results: list[CaseResult]) -> "EvalResult":
        """Compute all aggregate stats from a list of CaseResults."""
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        rate = passed / total if total > 0 else 0.0
        lo, hi = wilson_ci(passed, total)

        failures: dict[str, int] = {}
        for r in results:
            if not r.passed:
                failures[r.failure_mode] = failures.get(r.failure_mode, 0) + 1

        return cls(
            eval_id=str(uuid.uuid4()),
            suite_id=suite_id,
            created_at=datetime.now(timezone.utc),
            results=results,
            total_cases=total,
            passed_cases=passed,
            pass_rate=rate,
            ci_low=lo,
            ci_high=hi,
            failures_by_mode=failures,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "eval_id": self.eval_id,
            "suite_id": self.suite_id,
            "created_at": self.created_at.isoformat(),
            "results": [r.to_dict() for r in self.results],
            "total_cases": self.total_cases,
            "passed_cases": self.passed_cases,
            "pass_rate": self.pass_rate,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "failures_by_mode": self.failures_by_mode,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EvalResult":
        return cls(
            eval_id=d["eval_id"],
            suite_id=d["suite_id"],
            created_at=datetime.fromisoformat(d["created_at"]),
            results=[CaseResult.from_dict(r) for r in d["results"]],
            total_cases=d["total_cases"],
            passed_cases=d["passed_cases"],
            pass_rate=d["pass_rate"],
            ci_low=d["ci_low"],
            ci_high=d["ci_high"],
            failures_by_mode=d["failures_by_mode"],
        )

    @property
    def failing_cases(self) -> list[CaseResult]:
        return [r for r in self.results if not r.passed]

    def ci_overlaps_with(self, other: "EvalResult") -> bool:
        """Return True if the Wilson CIs of two EvalResults overlap.

        Used by the Improvement Agent: non-overlapping CIs mean the pass
        rate improvement is statistically real at the 95% level.
        """
        return self.ci_low <= other.ci_high and other.ci_low <= self.ci_high
