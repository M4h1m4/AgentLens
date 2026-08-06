"""Reference 3-agent research system for AgentLens (test subject).

A deliberately buggy multi-agent system: search -> synthesis -> writer,
with five seeded coordination failures that serve as ground truth for the
AgentLens eval engine.
"""

from reference_agent.config import AgentConfig, SeededFailureConfig
from reference_agent.run import invoke, run, reset_bleed_cache

__all__ = [
    "AgentConfig",
    "SeededFailureConfig",
    "invoke",
    "run",
    "reset_bleed_cache",
]
