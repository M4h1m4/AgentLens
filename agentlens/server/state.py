"""In-memory store for the current pipeline run's results.

The pipeline writes into server_state after each run completes.
The Vapi webhook handler reads from server_state to answer Q&A questions.

Why a module-level singleton: the FastAPI server and the pipeline run in the
same process. A module-level object is shared across all imports within that
process, so both sides see the same instance without needing a database.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentlens.eval.schema import EvalResult
    from agentlens.diagnosis.schema import Diagnosis
    from agentlens.improvement.schema import ImprovementResult


@dataclass
class ServerState:
    """Holds the outputs of the most recent pipeline run.

    Written by VoiceAgent.run() after the pipeline completes.
    Read by the Vapi webhook handler on every Q&A turn.

    Attributes:
        audio_bytes:   Raw MP3 bytes of the generated briefing. Served at /briefing.mp3.
        eval_result:   The EvalResult from the most recent eval run.
        diagnoses:     List of Diagnosis objects, one per failing failure mode.
        improvement:   The ImprovementResult from the most recent improvement run.
        conversation:  Running list of (role, text) tuples — the Q&A history.
                       Grows with each Vapi turn so the LLM has context.
    """

    audio_bytes: bytes = b""
    eval_result: "EvalResult | None" = None
    diagnoses: "list[Diagnosis]" = field(default_factory=list)
    improvement: "ImprovementResult | None" = None
    conversation: list[tuple[str, str]] = field(default_factory=list)

    def clear_conversation(self) -> None:
        """Reset conversation history at the start of each new briefing."""
        self.conversation = []


# Module-level singleton — imported by both voice.py and main.py
server_state = ServerState()
