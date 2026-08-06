"""Versioned JSON profile store.

Profiles are saved as .agentlens/{profile_id}.v{n}.json in the user's project
directory. Each reconciliation or update creates a new version rather than
overwriting, so the full history is preserved.

Using local JSON (not Postgres) deliberately: the profile store must work for
the CLI user who has no database running.  Postgres can be added later as an
optional backend via the same TraceStore pattern used in the adapter.
"""

from __future__ import annotations

import json
from pathlib import Path

from agentlens.intake.profile import AgentProfile


class ProfileStore:
    """Save and load versioned AgentProfiles as JSON files."""

    def __init__(self, base_dir: str | Path = ".agentlens") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # ── write ──────────────────────────────────────────────────────────────

    def save(self, profile: AgentProfile) -> Path:
        path = self.base_dir / f"{profile.profile_id}.v{profile.version}.json"
        path.write_text(json.dumps(profile.to_dict(), indent=2))
        # keep a pointer to the latest profile so callers don't need to track IDs
        (self.base_dir / "latest").write_text(profile.profile_id)
        return path

    # ── read ───────────────────────────────────────────────────────────────

    def load(self, profile_id: str, version: int | None = None) -> AgentProfile:
        """Load a profile by ID. Loads the latest version when version is None."""
        if version is not None:
            path = self.base_dir / f"{profile_id}.v{version}.json"
            return AgentProfile.from_dict(json.loads(path.read_text()))

        candidates = sorted(
            self.base_dir.glob(f"{profile_id}.v*.json"),
            key=lambda p: int(p.stem.rsplit(".v", 1)[-1]),
        )
        if not candidates:
            raise FileNotFoundError(f"No profile found for id={profile_id}")
        return AgentProfile.from_dict(json.loads(candidates[-1].read_text()))

    def load_latest(self) -> AgentProfile:
        """Load whichever profile was saved most recently."""
        pointer = self.base_dir / "latest"
        if not pointer.exists():
            raise FileNotFoundError("No profiles saved yet.")
        return self.load(pointer.read_text().strip())

    def list_profiles(self) -> list[str]:
        """Return all unique profile IDs that have been saved."""
        ids: set[str] = set()
        for p in self.base_dir.glob("*.v*.json"):
            ids.add(p.stem.rsplit(".v", 1)[0])
        return sorted(ids)

    def history(self, profile_id: str) -> list[AgentProfile]:
        """Return all saved versions of a profile, oldest first."""
        candidates = sorted(
            self.base_dir.glob(f"{profile_id}.v*.json"),
            key=lambda p: int(p.stem.rsplit(".v", 1)[-1]),
        )
        return [AgentProfile.from_dict(json.loads(p.read_text())) for p in candidates]
