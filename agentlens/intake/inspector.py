"""Codebase inspector — extracts structural facts about an agent system from source files.

Best-effort: uses regex pattern matching on Python source files. Works well for
clean, framework-native codebases. Degrades gracefully for dynamic construction,
multi-repo systems, or heavy custom wrappers — notes are added to the result
explaining what could not be reliably detected.

Confidence level produced: "code_inspected" — more reliable than user_stated
(developer may misremember) but less reliable than trace_observed (code may
not reflect what's actually running in production).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


# ── skip lists ───────────────────────────────────────────────────────────────

_SKIP_DIRS = {
    ".git", ".venv", "venv", "__pycache__", "node_modules",
    ".pytest_cache", "dist", "build", ".mypy_cache", "*.egg-info",
    ".tox", "htmlcov", ".eggs",
}

# ── regex patterns ────────────────────────────────────────────────────────────

# Framework detection — imports at the top of any file
_FRAMEWORK_PATTERNS: list[tuple[str, str]] = [
    (r"(?:import|from)\s+langgraph\b", "langgraph"),
    (r"(?:import|from)\s+crewai\b", "crewai"),
    (r"(?:import|from)\s+autogen\b", "autogen"),
    (r"(?:import|from)\s+llama_index\b", "llamaindex"),
    (r"(?:import|from)\s+llamaindex\b", "llamaindex"),
]

# LangGraph node names — .add_node("name", fn) with string literal first arg
_NODE_LITERAL = re.compile(
    r'\.add_node\(\s*["\']([a-zA-Z_][a-zA-Z0-9_]*)["\']'
)
# Dynamic node construction — .add_node( with a variable, not a string literal
_NODE_DYNAMIC = re.compile(
    r'\.add_node\(\s*(?!["\'])[a-zA-Z_]'
)

# Tool names — @tool decorator followed by def function_name
_TOOL_DEF = re.compile(
    r'@tool(?:\([^)]*\))?\s*\n\s*(?:async\s+)?def\s+([a-zA-Z_][a-zA-Z0-9_]*)'
)

# Entry point — variable = something.compile()
_COMPILE_CALL = re.compile(
    r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*\w+\.compile\('
)


# ── result type ───────────────────────────────────────────────────────────────

@dataclass
class InspectionResult:
    """Structural facts extracted from a codebase scan."""

    framework: str | None = None
    framework_confidence: str = "unknown"  # "high" if unambiguous import found

    agent_names: list[str] = field(default_factory=list)
    # Names extracted from .add_node("name", ...) calls with string literals.
    # Empty if construction is fully dynamic.

    tool_names: list[str] = field(default_factory=list)
    # Names of @tool-decorated functions found.

    entry_points: list[str] = field(default_factory=list)
    # Candidate entry point variables: "path/to/file.py::variable_name"
    # from `var = something.compile()` patterns.

    files_scanned: int = 0
    has_dynamic_nodes: bool = False
    # True if any .add_node() call used a variable (not a string literal).

    notes: list[str] = field(default_factory=list)
    # Human-readable caveats about what could not be reliably detected.

    def to_profile_hints(self) -> dict:
        """Convert to a dict of pre-filled hints for the IntakeAgent system prompt.

        Format: {field_name: {value, confidence, source}} — same shape as
        ConfidentValue.to_dict() so the prompt builder can render them uniformly.
        """
        hints: dict = {}

        if self.framework:
            hints["framework"] = {
                "value": self.framework,
                "confidence": self.framework_confidence,
                "source": "code_inspected",
            }

        if self.agent_names:
            hints["agent_names"] = {
                "value": self.agent_names,
                "confidence": "high" if not self.has_dynamic_nodes else "low",
                "source": "code_inspected",
                "note": "may be incomplete — dynamic node construction detected"
                        if self.has_dynamic_nodes else None,
            }

        if self.tool_names:
            hints["tool_names"] = {
                "value": self.tool_names,
                "confidence": "high",
                "source": "code_inspected",
            }

        if self.entry_points:
            hints["entry_points"] = {
                "value": self.entry_points,
                "confidence": "medium",
                "source": "code_inspected",
            }

        return hints


# ── public API ────────────────────────────────────────────────────────────────

def inspect_codebase(path: str | Path) -> InspectionResult:
    """Scan a directory tree and extract multi-agent structural facts.

    Args:
        path: Root directory of the developer's project.

    Returns:
        InspectionResult with best-effort findings and explanatory notes.
    """
    root = Path(path).resolve()
    result = InspectionResult()

    if not root.exists():
        result.notes.append(f"Path does not exist: {root}")
        return result

    if not root.is_dir():
        result.notes.append(f"Path is a file, not a directory: {root}")
        return result

    frameworks_found: dict[str, int] = {}  # framework -> file count

    for py_file in _iter_python_files(root):
        result.files_scanned += 1
        try:
            source = py_file.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        _scan_framework(source, frameworks_found)
        _scan_nodes(source, result)
        _scan_tools(source, result)
        _scan_entry_points(source, py_file, root, result)

    _resolve_framework(frameworks_found, result)
    _add_notes(result)

    return result


# ── internals ─────────────────────────────────────────────────────────────────

def _iter_python_files(root: Path):
    """Yield .py files, skipping common non-source directories."""
    for item in root.rglob("*.py"):
        if any(skip in item.parts for skip in _SKIP_DIRS):
            continue
        # skip egg-info directories (glob pattern matching)
        if any(part.endswith(".egg-info") for part in item.parts):
            continue
        yield item


def _scan_framework(source: str, frameworks_found: dict) -> None:
    for pattern, name in _FRAMEWORK_PATTERNS:
        if re.search(pattern, source):
            frameworks_found[name] = frameworks_found.get(name, 0) + 1


def _scan_nodes(source: str, result: InspectionResult) -> None:
    for match in _NODE_LITERAL.finditer(source):
        name = match.group(1)
        if name not in result.agent_names:
            result.agent_names.append(name)

    if _NODE_DYNAMIC.search(source):
        result.has_dynamic_nodes = True


def _scan_tools(source: str, result: InspectionResult) -> None:
    for match in _TOOL_DEF.finditer(source):
        name = match.group(1)
        if name not in result.tool_names:
            result.tool_names.append(name)


def _scan_entry_points(
    source: str, py_file: Path, root: Path, result: InspectionResult
) -> None:
    for match in _COMPILE_CALL.finditer(source):
        var_name = match.group(1)
        rel = py_file.relative_to(root)
        module = str(rel).replace("/", ".").replace("\\", ".").removesuffix(".py")
        entry = f"{module}:{var_name}"
        if entry not in result.entry_points:
            result.entry_points.append(entry)


def _resolve_framework(frameworks_found: dict, result: InspectionResult) -> None:
    if not frameworks_found:
        return
    # Pick the framework with the most files importing it
    best = max(frameworks_found, key=lambda k: frameworks_found[k])
    result.framework = best
    result.framework_confidence = "high" if frameworks_found[best] >= 1 else "medium"
    if len(frameworks_found) > 1:
        others = [f for f in frameworks_found if f != best]
        result.notes.append(
            f"Multiple frameworks detected: {best} (primary), {', '.join(others)} (also imported). "
            f"Reported framework may not be the main orchestration layer."
        )


def _add_notes(result: InspectionResult) -> None:
    if result.has_dynamic_nodes:
        result.notes.append(
            "Dynamic node construction detected (.add_node with a variable). "
            "Agent names list may be incomplete — trace observation will give the full picture."
        )
    if not result.agent_names and not result.has_dynamic_nodes and result.framework == "langgraph":
        result.notes.append(
            "No .add_node() calls found despite LangGraph import. "
            "Graph may be built in a custom wrapper — agent names could not be extracted."
        )
    if result.files_scanned == 0:
        result.notes.append("No Python files found in the specified directory.")
