"""Pipeline orchestration helpers — glue between AgentLens components.

After the Improvement Agent promotes a fix, this module handles writing the
proven hypothesis back to the FailureLibrary so future Diagnosis Agent runs
reason from ground truth rather than the original weak hypothesis.

Design decision D8.1: overwrite the stored diagnosis with the proven fix.
See DESIGN_DECISIONS.md for the rationale.
"""

from __future__ import annotations

from agentlens.improvement.schema import ImprovementResult
from agentlens.rag.library import FailureLibrary


def update_library_after_improvement(
    library: FailureLibrary,
    trace_id: str,
    improvement_result: ImprovementResult,
) -> None:
    """Overwrite the stored diagnosis with the proven hypothesis.

    Called by the Day 11 CLI after ImprovementAgent.improve() returns a
    promoted result. The original diagnosis stored alongside the FailureSignal
    was a weak LLM hypothesis. Once the Improvement Agent has found a fix that
    is statistically confirmed (Wilson CIs non-overlapping, no regressions),
    the hypothesis IS the ground truth — overwrite it so future retrievals
    return the proven explanation, not the speculative one.

    Only runs when improvement_result.promoted is True. No-op otherwise,
    because an unpromoted result has no confirmed root cause to store.

    Args:
        library:           FailureLibrary instance (ChromaDB-backed).
        trace_id:          Trace ID used when the FailureSignal was originally
                           stored (matches the eval run that surfaced the failure).
        improvement_result: Result from ImprovementAgent.improve(). Must have
                            promoted=True for any write to occur.
    """
    if not improvement_result.promoted:
        return

    pass_rate_delta = (
        improvement_result.variant_eval.pass_rate
        - improvement_result.baseline_eval.pass_rate
    )

    library.update_diagnosis(
        trace_id=trace_id,
        diagnosis=improvement_result.hypothesis,   # proven ground truth
        fix_applied=improvement_result.hypothesis,
        pass_rate_delta=pass_rate_delta,
    )
