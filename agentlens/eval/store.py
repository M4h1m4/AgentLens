"""EvalStore — persist EvalResult objects as versioned JSON files."""

from __future__ import annotations

import json
from pathlib import Path

from agentlens.eval.schema import EvalResult


class EvalStore:
    """Save and load EvalResult objects to local JSON files.

    Files written to <base_dir>/eval_results/<eval_id>.json.
    A latest-pointer per suite_id enables quick retrieval without scanning.

    Usage::

        store = EvalStore()
        store.save(result)
        latest = store.load_latest(suite_id)
    """

    def __init__(self, base_dir: str | Path = ".agentlens") -> None:
        self._dir = Path(base_dir) / "eval_results"
        self._dir.mkdir(parents=True, exist_ok=True)

    def save(self, result: EvalResult) -> Path:
        """Persist an EvalResult. Returns the path written."""
        path = self._dir / f"{result.eval_id}.json"
        path.write_text(json.dumps(result.to_dict(), indent=2))
        pointer = self._dir / f"latest_{result.suite_id}.json"
        pointer.write_text(result.eval_id)
        return path

    def load(self, eval_id: str) -> EvalResult:
        path = self._dir / f"{eval_id}.json"
        return EvalResult.from_dict(json.loads(path.read_text()))

    def load_latest(self, suite_id: str) -> EvalResult | None:
        """Return the most recent EvalResult for a suite, or None."""
        pointer = self._dir / f"latest_{suite_id}.json"
        if not pointer.exists():
            return None
        return self.load(pointer.read_text().strip())

    def list_results(self) -> list[str]:
        """Return all eval_ids stored (sorted by modification time)."""
        return [
            p.stem
            for p in sorted(self._dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
            if not p.name.startswith("latest_")
        ]
