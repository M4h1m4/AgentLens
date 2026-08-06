"""BenchmarkStore — save and load BenchmarkSuites as versioned JSON files.

Files are written to <base_dir>/benchmarks/<suite_id>.json.
A latest-pointer file per trace_id allows quick retrieval of the most recent
suite for a given run without scanning all files.
"""

from __future__ import annotations

import json
from pathlib import Path

from agentlens.benchmark.schema import BenchmarkSuite


class BenchmarkStore:
    """Persist BenchmarkSuites to local JSON files.

    Usage::

        store = BenchmarkStore()            # defaults to .agentlens/benchmarks/
        store.save(suite)
        suite = store.load_latest(trace_id)
    """

    def __init__(self, base_dir: str | Path = ".agentlens") -> None:
        self._dir = Path(base_dir) / "benchmarks"
        self._dir.mkdir(parents=True, exist_ok=True)

    def save(self, suite: BenchmarkSuite) -> Path:
        """Write suite to disk. Returns the path written."""
        path = self._dir / f"{suite.suite_id}.json"
        path.write_text(json.dumps(suite.to_dict(), indent=2))
        # Update latest pointer for this trace
        pointer = self._dir / f"latest_{suite.trace_id}.json"
        pointer.write_text(suite.suite_id)
        return path

    def load(self, suite_id: str) -> BenchmarkSuite:
        """Load a suite by its suite_id."""
        path = self._dir / f"{suite_id}.json"
        return BenchmarkSuite.from_dict(json.loads(path.read_text()))

    def load_latest(self, trace_id: str) -> BenchmarkSuite | None:
        """Return the most recently saved suite for a trace, or None."""
        pointer = self._dir / f"latest_{trace_id}.json"
        if not pointer.exists():
            return None
        suite_id = pointer.read_text().strip()
        return self.load(suite_id)

    def list_suites(self) -> list[str]:
        """Return all suite IDs stored (excludes pointer files)."""
        return [
            p.stem
            for p in sorted(self._dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
            if not p.name.startswith("latest_")
        ]
