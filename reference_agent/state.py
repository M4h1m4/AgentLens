"""Shared LangGraph state passed between the three agents.

Node boundaries are kept clean so the Day 2 universal adapter can intercept
handoffs (search -> synthesis -> writer) without refactoring.
"""

from __future__ import annotations

from typing import Any, TypedDict


class Synthesis(TypedDict, total=False):
    key_claims: list[str]
    supporting_evidence: list[str]
    conflicting_viewpoints: list[str]


class AgentState(TypedDict, total=False):
    # Inputs
    query: str
    session_id: str

    # Agent 1 output
    sources: list[dict[str, Any]]

    # Agent 2 output
    synthesis_partial: Synthesis  # first chunk — key claims only
    synthesis_complete: Synthesis  # full synthesis
    dropped_sources: list[dict[str, Any]]  # observability for failure 1
    loop_count: int  # observability for failure 4

    # Agent 3 output
    report: str
    writer_input: str  # "partial" | "complete" — observability for failure 2
