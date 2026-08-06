"""AgentProfile — the structured output of the NL intake agent.

Every field that comes from user input carries an explicit confidence level and
source, because a developer describing their own system is the least reliable
evidence source. Fields get promoted to "high" confidence only after being
confirmed by an observed trace (see reconciler.py).
"""

from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return str(uuid.uuid4())


@dataclass
class ConfidentValue:
    """A profile field that knows how confident we are and where the value came from."""

    value: Any
    confidence: str  # "high" | "medium" | "low" | "unknown"
    source: str      # "user_stated" | "trace_observed" | "inferred" | "code_inspected"
    note: Optional[str] = None

    @classmethod
    def unknown(cls) -> "ConfidentValue":
        return cls(value=None, confidence="unknown", source="not_provided")

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "confidence": self.confidence,
            "source": self.source,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ConfidentValue":
        return cls(
            value=d["value"],
            confidence=d["confidence"],
            source=d["source"],
            note=d.get("note"),
        )


@dataclass
class AgentProfile:
    """Structured description of a multi-agent system, built incrementally.

    Starts as a rough draft from the user's description (low confidence).
    Gets promoted field-by-field as an observed trace confirms or contradicts
    what the user said (see reconciler.py).
    """

    profile_id: str = field(default_factory=_new_id)
    version: int = 1
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)

    # raw input preserved verbatim
    raw_description: str = ""

    # structured fields — each carries a confidence level
    framework: ConfidentValue = field(default_factory=ConfidentValue.unknown)
    agent_count: ConfidentValue = field(default_factory=ConfidentValue.unknown)
    agents: list[dict[str, Any]] = field(default_factory=list)
    # each agent dict: {name, role, confidence, source}
    delegation_patterns: ConfidentValue = field(default_factory=ConfidentValue.unknown)
    tool_inventory: ConfidentValue = field(default_factory=ConfidentValue.unknown)
    expected_output: ConfidentValue = field(default_factory=ConfidentValue.unknown)
    suspected_failure_modes: ConfidentValue = field(default_factory=ConfidentValue.unknown)

    # reconciliation tracking
    reconciled_with_trace: bool = False
    discrepancies: list[str] = field(default_factory=list)

    def bump_version(self) -> "AgentProfile":
        """Return a deep copy with version incremented and updated_at refreshed."""
        p = copy.deepcopy(self)
        p.version += 1
        p.updated_at = _now()
        return p

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "raw_description": self.raw_description,
            "framework": self.framework.to_dict(),
            "agent_count": self.agent_count.to_dict(),
            "agents": self.agents,
            "delegation_patterns": self.delegation_patterns.to_dict(),
            "tool_inventory": self.tool_inventory.to_dict(),
            "expected_output": self.expected_output.to_dict(),
            "suspected_failure_modes": self.suspected_failure_modes.to_dict(),
            "reconciled_with_trace": self.reconciled_with_trace,
            "discrepancies": self.discrepancies,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AgentProfile":
        return cls(
            profile_id=d["profile_id"],
            version=d["version"],
            created_at=datetime.fromisoformat(d["created_at"]),
            updated_at=datetime.fromisoformat(d["updated_at"]),
            raw_description=d["raw_description"],
            framework=ConfidentValue.from_dict(d["framework"]),
            agent_count=ConfidentValue.from_dict(d["agent_count"]),
            agents=d["agents"],
            delegation_patterns=ConfidentValue.from_dict(d["delegation_patterns"]),
            tool_inventory=ConfidentValue.from_dict(d["tool_inventory"]),
            expected_output=ConfidentValue.from_dict(d["expected_output"]),
            suspected_failure_modes=ConfidentValue.from_dict(d["suspected_failure_modes"]),
            reconciled_with_trace=d["reconciled_with_trace"],
            discrepancies=d["discrepancies"],
        )
