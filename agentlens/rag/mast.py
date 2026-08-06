"""MAST multi-agent failure taxonomy — hardcoded as a prompt string.

MAST (Multi-Agent System Testing) classifies coordination failures into
categories. Small enough to fit directly in the Diagnosis Agent's system
prompt — no RAG needed.

Reference: Cemri et al., "Multi-Agent System Testing" (2024).
"""

from __future__ import annotations

# ── taxonomy ──────────────────────────────────────────────────────────────────

CATEGORIES: dict[str, str] = {
    "context_loss": (
        "Information that was present in one agent's output is absent or "
        "truncated in the next agent's input. The receiving agent operates "
        "on an incomplete view of the world without knowing it."
    ),
    "race_condition": (
        "An agent begins processing before a dependency has completed. "
        "The agent acts on a partial or intermediate state rather than "
        "the final output of the upstream agent."
    ),
    "contradiction_ignored": (
        "An agent receives conflicting signals from multiple sources or "
        "upstream agents and picks one without flagging the conflict. "
        "Downstream agents inherit an unresolved inconsistency."
    ),
    "delegation_loop": (
        "An agent delegates a task to another agent which re-delegates "
        "back, creating a cycle. The system makes no forward progress "
        "and may exhaust its budget or hit a recursion limit."
    ),
    "context_bleed": (
        "State or context from a previous session or run is carried into "
        "the current run. The agent operates on stale information that "
        "should have been cleared between sessions."
    ),
    "stale_context": (
        "An agent uses information that was valid when retrieved but has "
        "since been superseded by a later event within the same run. "
        "The agent's reasoning is correct given its snapshot but wrong "
        "given the current state."
    ),
    "premature_termination": (
        "An agent declares completion and hands off before it has actually "
        "finished its task. Downstream agents receive an incomplete artifact "
        "and may not detect the incompleteness."
    ),
    "over_delegation": (
        "An agent delegates a task that it could and should handle itself, "
        "introducing an unnecessary handoff that adds latency, loses context, "
        "or amplifies errors."
    ),
}

# ── prompt helpers ─────────────────────────────────────────────────────────────

def as_prompt_section() -> str:
    """Return the MAST taxonomy formatted for inclusion in a system prompt."""
    lines = [
        "KNOWN MULTI-AGENT FAILURE CATEGORIES (MAST taxonomy):",
        "Use these to label and classify failures you observe in traces.",
        "",
    ]
    for name, description in CATEGORIES.items():
        lines.append(f"  {name}:")
        lines.append(f"    {description}")
        lines.append("")
    return "\n".join(lines)


def category_names() -> list[str]:
    """Return the list of MAST category names."""
    return list(CATEGORIES.keys())


def describe(category: str) -> str | None:
    """Return the description of a MAST category, or None if not found."""
    return CATEGORIES.get(category)
