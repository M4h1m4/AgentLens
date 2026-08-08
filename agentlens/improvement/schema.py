"""Schema dataclasses for the Improvement Agent.

ImprovementResult  — full result of one improve() run
ExecutorConfig     — SandboxExecutor constructor parameters, passed by the caller
                     so the ImprovementAgent can spin up fresh executors for each
                     variant without knowing how to construct the agent beyond these
                     values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentlens.eval.schema import EvalResult


@dataclass
class ExecutorConfig:
    """Parameters for constructing a SandboxExecutor for variant runs.

    Mirrors the SandboxExecutor constructor signature exactly so the caller can
    express "here is how to spin up my agent" without the ImprovementAgent
    needing to know any agent-specific details.

    Attributes:
        agent_dir:           Local path to the agent package directory.
        import_lines:        Python import statements for the sandbox runner script.
        build_app_expr:      One-line Python expression that builds the compiled
                             LangGraph application.
        output_key:          State key holding the final text output. Default "report".
        extra_initial_state: Agent-specific fields merged into the initial state dict.
        pip_packages:        Extra pip packages to install in each sandbox.
        e2b_api_key:         E2B API key. Falls back to E2B_API_KEY env var if None.
    """

    agent_dir: str
    import_lines: str
    build_app_expr: str
    output_key: str = "report"
    extra_initial_state: dict[str, Any] | None = None
    pip_packages: list[str] | None = None
    e2b_api_key: str | None = None


@dataclass
class ImprovementResult:
    """Full result of one ImprovementAgent.improve() run.

    Attributes:
        failure_mode:        MAST category that was targeted.
        promoted:            True if a statistically-real, regression-free fix was found.
        iterations_run:      Number of LLM iterations consumed.
        cost_dollars:        Estimated cost of all LLM calls in this run.
        hypothesis:          The winning hypothesis sentence (or last attempted if not promoted).
        variant_files:       The modified source files (relative path → complete content).
                             Empty dict if no iterations completed.
        baseline_eval:       The original EvalResult passed in by the caller.
        variant_eval:        The EvalResult from the best variant found (full suite if
                             promoted, failing-cases-only otherwise).
        regression_detected: True if the last promoted candidate caused regressions.
                             Always False for a promoted result (regressions block promotion).
        reason:              Human-readable explanation of the outcome.
    """

    failure_mode: str
    promoted: bool
    iterations_run: int
    cost_dollars: float
    hypothesis: str
    variant_files: dict[str, str]
    baseline_eval: EvalResult
    variant_eval: EvalResult
    regression_detected: bool
    reason: str
