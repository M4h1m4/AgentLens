"""Configuration for the reference agent.

The five seeded failures are each gated behind a boolean flag. Default is
ON (buggy) because this system is the "test subject" whose known bugs the
AgentLens eval engine must detect. Flipping a flag OFF restores correct
behavior, which is how later components (the Improvement Agent) express and
test fixes, and how evals establish ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SeededFailureConfig:
    """Toggles for the five deliberately planted failure modes."""

    # Failure 1 — Agent 2 silently drops sources beyond the cap (no error signal).
    context_loss: bool = True
    # Failure 2 — Agent 3 writes from the partial synthesis instead of the full one.
    race_condition: bool = True
    # Failure 3 — Agent 2 ignores conflicting sources instead of flagging them.
    contradiction_ignored: bool = True
    # Failure 4 — Agent 2 re-queries Agent 1 in a loop without a proper terminator.
    delegation_loop: bool = True
    # Failure 5 — Agent 3 carries context from a previous run into the current one.
    context_bleed: bool = True

    @classmethod
    def clean(cls) -> "SeededFailureConfig":
        """A correct configuration with every seeded failure disabled."""
        return cls(
            context_loss=False,
            race_condition=False,
            contradiction_ignored=False,
            delegation_loop=False,
            context_bleed=False,
        )


@dataclass
class AgentConfig:
    """Runtime configuration for the reference agent."""

    failures: SeededFailureConfig = None  # type: ignore[assignment]

    # Agent 1: how many sources the search tool returns. Kept above the
    # synthesis cap so the context-loss failure can manifest.
    search_source_limit: int = 5
    # Agent 2: max sources it will actually process before dropping.
    synthesis_source_cap: int = 3
    # Failure 4 safety valve: even a "non-terminating" loop halts here so dev
    # runs and tests do not hang. The bug is the missing terminator, which the
    # eval detects via the elevated loop count — not an actual infinite loop.
    max_loop_iterations: int = 10
    # Query type that triggers the delegation loop (failure 4).
    loop_trigger_keyword: str = "exhaustive"

    def __post_init__(self) -> None:
        if self.failures is None:
            self.failures = SeededFailureConfig()
