"""Fixture-backed web search tool (Agent 1's tool).

Option B (see DESIGN_DECISIONS.md D2.1): exposed as a LangChain tool so the
AgentLens adapter auto-captures tool calls via framework callbacks, exactly as
it would on a real system. Day 1 uses fixtures rather than a live web API so
the seeded failures stay reproducible: failure 3 needs conflicting sources and
failure 1 needs more sources than the synthesis cap, both of which must be
controllable.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

_FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "sources.json"


@lru_cache(maxsize=1)
def _load() -> dict[str, list[dict[str, Any]]]:
    return json.loads(_FIXTURES.read_text())


@tool
def web_search(query: str, limit: int = 5) -> list[dict[str, Any]]:
    """Search the web and return up to `limit` source documents for a query.

    Queries mentioning "conflict" return contradictory sources so the
    contradiction-handling failure can be exercised; everything else returns
    the default set (five sources, enough to trip the context-loss cap).
    """
    fixtures = _load()
    key = "conflict" if "conflict" in query.lower() else "default"
    return [dict(source) for source in fixtures[key][:limit]]
