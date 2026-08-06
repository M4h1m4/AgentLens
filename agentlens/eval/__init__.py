"""AgentLens eval — Eval Runner, result schema, criterion checker, and store.

Public API::

    from agentlens.eval import (
        EvalRunner, InProcessExecutor, SandboxExecutor,
        EvalResult, CaseResult, CriterionResult,
        EvalStore, wilson_ci,
    )

    # Run a benchmark suite against the agent (in-process)
    executor = InProcessExecutor(app=build_graph(config), exporter=exporter)
    runner = EvalRunner(executor=executor)
    result = runner.run(suite)

    print(f"Pass rate: {result.pass_rate:.0%}  "
          f"CI: [{result.ci_low:.2f}, {result.ci_high:.2f}]")

    # Persist
    store = EvalStore()
    store.save(result)
"""

from agentlens.eval.checker import check_criterion, check_all_criteria
from agentlens.eval.runner import EvalRunner, InProcessExecutor, SandboxExecutor, AgentExecutor
from agentlens.eval.schema import (
    CriterionResult, CaseResult, EvalResult, wilson_ci,
)
from agentlens.eval.store import EvalStore

__all__ = [
    "EvalRunner",
    "InProcessExecutor",
    "SandboxExecutor",
    "AgentExecutor",
    "EvalResult",
    "CaseResult",
    "CriterionResult",
    "EvalStore",
    "wilson_ci",
    "check_criterion",
    "check_all_criteria",
]
