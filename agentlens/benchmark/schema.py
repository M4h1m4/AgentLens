"""BenchmarkCase and BenchmarkSuite — the output of the Benchmark Gen Agent.

Each BenchmarkCase is a concrete test: a query to run against the agent and a
set of natural-language criteria that a correct output must satisfy.

A BenchmarkSuite groups 20-30 cases generated from one observed trace, each
traceable back to the failure signal that motivated it.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class BenchmarkCase:
    """One test case — a query plus assertions about the expected output."""

    case_id: str
    query: str
    expected_criteria: list[str]  # natural language assertions a correct output must pass
    failure_mode: str             # MAST category this case is designed to catch
    severity: str                 # "high" | "medium" | "low"
    rationale: str                # why this query exposes this failure mode

    @classmethod
    def new(
        cls,
        query: str,
        expected_criteria: list[str],
        failure_mode: str,
        severity: str = "medium",
        rationale: str = "",
    ) -> "BenchmarkCase":
        return cls(
            case_id=str(uuid.uuid4()),
            query=query,
            expected_criteria=expected_criteria,
            failure_mode=failure_mode,
            severity=severity,
            rationale=rationale,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "query": self.query,
            "expected_criteria": self.expected_criteria,
            "failure_mode": self.failure_mode,
            "severity": self.severity,
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BenchmarkCase":
        return cls(
            case_id=d["case_id"],
            query=d["query"],
            expected_criteria=d["expected_criteria"],
            failure_mode=d["failure_mode"],
            severity=d["severity"],
            rationale=d["rationale"],
        )


@dataclass
class BenchmarkSuite:
    """A set of benchmark cases generated from one observed trace run."""

    suite_id: str
    trace_id: str                      # the source trace that motivated this suite
    session_id: str
    created_at: datetime
    cases: list[BenchmarkCase]
    failure_modes_targeted: list[str]  # MAST categories covered by this suite

    @classmethod
    def new(
        cls,
        trace_id: str,
        session_id: str,
        cases: list[BenchmarkCase],
        failure_modes: list[str],
    ) -> "BenchmarkSuite":
        return cls(
            suite_id=str(uuid.uuid4()),
            trace_id=trace_id,
            session_id=session_id,
            created_at=datetime.now(timezone.utc),
            cases=cases,
            failure_modes_targeted=failure_modes,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_id": self.suite_id,
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "cases": [c.to_dict() for c in self.cases],
            "failure_modes_targeted": self.failure_modes_targeted,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BenchmarkSuite":
        return cls(
            suite_id=d["suite_id"],
            trace_id=d["trace_id"],
            session_id=d["session_id"],
            created_at=datetime.fromisoformat(d["created_at"]),
            cases=[BenchmarkCase.from_dict(c) for c in d["cases"]],
            failure_modes_targeted=d["failure_modes_targeted"],
        )

    # ── convenience ──────────────────────────────────────────────────────────

    def cases_by_mode(self, failure_mode: str) -> list[BenchmarkCase]:
        return [c for c in self.cases if c.failure_mode == failure_mode]

    def high_severity(self) -> list[BenchmarkCase]:
        return [c for c in self.cases if c.severity == "high"]
