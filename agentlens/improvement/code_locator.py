"""CodeLocator — local, non-LLM source file selector for the Improvement Agent.

Given an agent directory and a Diagnosis, CodeLocator returns the minimal set of
source files that are most likely to contain the bug:

1. Scan all .py files in agent_dir for LangGraph add_node("name", ...) calls.
2. Filter to node names mentioned in the diagnosis text.
3. For each matched file, follow one level of local imports (AST-parsed).
4. Return {relative_path: file_content} — typically 1–4 files.

This runs entirely locally before anything is sent to an LLM. The developer can
review exactly which files will be sent.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from agentlens.diagnosis.schema import Diagnosis


class CodeLocator:
    """Find source files implicated by a Diagnosis without browsing the full codebase."""

    # Matches both add_node("name", ...) and add_node('name', ...)
    _ADD_NODE_RE = re.compile(r'add_node\(\s*["\']([^"\']+)["\']')

    def locate(self, agent_dir: str, diagnosis: Diagnosis) -> dict[str, str]:
        """Find files implicated by the diagnosis.

        Args:
            agent_dir:  Path to the root of the agent package to scan.
            diagnosis:  Structured diagnosis from DiagnosisAgent.

        Returns:
            Mapping of relative_path (from agent_dir) → file content.
            Empty dict if no .py files exist in agent_dir.
        """
        agent_path = Path(agent_dir).resolve()

        # ── Step 1: scan for add_node registrations ───────────────────────────
        all_py_files = [
            f for f in agent_path.rglob("*.py")
            if "__pycache__" not in f.parts
        ]

        # {node_name: registration_file_path}
        node_registry: dict[str, Path] = {}
        for py_file in all_py_files:
            try:
                source = py_file.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for match in self._ADD_NODE_RE.finditer(source):
                node_name = match.group(1)
                if node_name not in node_registry:
                    node_registry[node_name] = py_file

        # ── Step 2: filter to nodes mentioned in diagnosis text ───────────────
        diagnosis_text = " ".join(
            [diagnosis.root_cause]
            + list(diagnosis.evidence)
            + [diagnosis.suggested_fix]
        )

        matched_nodes: set[str] = set()
        for node_name in node_registry:
            pattern = r"\b" + re.escape(node_name) + r"\b"
            if re.search(pattern, diagnosis_text, re.IGNORECASE):
                matched_nodes.add(node_name)

        # ── Step 3: fallback — if no nodes matched, use all discovered nodes ──
        if not matched_nodes:
            matched_nodes = set(node_registry.keys())

        # ── Step 4: collect seed files ────────────────────────────────────────
        # For each matched node: registration file + any .py file containing a
        # function/class whose name contains the node_name as a substring.
        seed_files: set[Path] = set()

        for node_name in matched_nodes:
            seed_files.add(node_registry[node_name])

            # Search for functions/classes containing node_name as substring
            for py_file in all_py_files:
                try:
                    source = py_file.read_text(encoding="utf-8", errors="replace")
                    tree = ast.parse(source, filename=str(py_file))
                except (OSError, SyntaxError):
                    continue
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        if node_name.lower() in node.name.lower():
                            seed_files.add(py_file)
                            break

        # ── Step 5: follow one level of local imports ─────────────────────────
        project_root = agent_path.parent  # one level up so we can resolve sibling packages

        local_import_files: set[Path] = set()
        for seed_file in seed_files:
            local_import_files.update(
                self._resolve_local_imports(seed_file, project_root, agent_path)
            )

        # ── Step 6: union and build result dict ───────────────────────────────
        all_files = seed_files | local_import_files

        result: dict[str, str] = {}
        for f in sorted(all_files):
            try:
                content = f.read_text(encoding="utf-8", errors="replace")
                rel = str(f.relative_to(agent_path))
                result[rel] = content
            except (OSError, ValueError):
                continue

        return result

    # ── helpers ───────────────────────────────────────────────────────────────

    def _resolve_local_imports(
        self,
        source_file: Path,
        project_root: Path,
        agent_path: Path,
    ) -> list[Path]:
        """Parse source_file with AST and return local .py files it imports."""
        try:
            source = source_file.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source, filename=str(source_file))
        except (OSError, SyntaxError):
            return []

        resolved: list[Path] = []

        for node in ast.walk(tree):
            module_str: str | None = None

            if isinstance(node, ast.ImportFrom) and node.module:
                module_str = node.module
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    module_str = alias.name
                    candidate = self._module_to_path(module_str, project_root, agent_path)
                    if candidate is not None:
                        resolved.append(candidate)
                continue  # handled names above; skip the module_str block below

            if module_str is not None:
                candidate = self._module_to_path(module_str, project_root, agent_path)
                if candidate is not None:
                    resolved.append(candidate)

        return resolved

    def _module_to_path(
        self,
        module: str,
        project_root: Path,
        agent_path: Path,
    ) -> Path | None:
        """Convert a dotted module name to a .py Path if it resolves locally."""
        rel_path = module.replace(".", "/") + ".py"

        # Try resolving relative to agent_path first, then project_root
        for base in (agent_path, project_root):
            candidate = base / rel_path
            if candidate.exists() and candidate.is_file():
                return candidate.resolve()

        # Also try as a package (__init__.py)
        pkg_path = module.replace(".", "/") + "/__init__.py"
        for base in (agent_path, project_root):
            candidate = base / pkg_path
            if candidate.exists() and candidate.is_file():
                return candidate.resolve()

        return None
