"""Pipeline utilities — functions that wire together multiple AgentLens components.

These helpers are called at the orchestration layer (CLI or integration tests)
after individual agents have completed their work.
"""

from __future__ import annotations

from agentlens.improvement.schema import ImprovementResult
from agentlens.rag.library import FailureLibrary


def update_library_after_improvement(
    library: FailureLibrary,
    trace_id: str,
    result: ImprovementResult,
) -> None:
    """Write the improvement outcome back into the failure library.

    If the improvement was promoted (fix confirmed by Wilson CI + no regressions),
    overwrite the stored signal's weak initial hypothesis with the proven fix.
    If not promoted, leave the library unchanged — an unconfirmed attempt is not
    evidence of a fix.

    Design decision D8.1: only promoted fixes update the library. This keeps the
    library as a source of ground truth rather than speculation.

    Args:
        library:   The FailureLibrary to update.
        trace_id:  The trace_id of the stored FailureSignal to update.
        result:    The ImprovementResult from ImprovementAgent.improve().
    """
    if not result.promoted:
        return

    pass_rate_delta = (
        result.variant_eval.pass_rate - result.baseline_eval.pass_rate
    )

    library.update_diagnosis(
        trace_id=trace_id,
        diagnosis=result.hypothesis,
        fix_applied=result.hypothesis,
        pass_rate_delta=pass_rate_delta,
    )
