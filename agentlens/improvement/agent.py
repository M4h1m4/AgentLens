"""ImprovementAgent — generates, validates, and ranks code fixes for diagnosed failures.

Given:
  - A Diagnosis (failure_mode, root_cause, evidence, suggested_fix)
  - A baseline EvalResult (all cases, all results)
  - A BenchmarkSuite (full test suite)
  - component_files: {rel_path: content} from CodeLocator

The agent runs up to max_iterations of:
  1. LLM call → hypothesis + modified_files
  2. Apply files to temp copy of agent_dir
  3. Phase 1: run failing-cases-only sub-suite via SandboxExecutor
  4. Wilson CI: non-overlapping → Phase 2
  5. Phase 2: run FULL suite → regression check
  6. If no regressions → promoted=True, return

Budget is tracked via character-based token estimates and capped at limit_dollars.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, field_validator

from agentlens.benchmark.schema import BenchmarkSuite
from agentlens.diagnosis.schema import Diagnosis
from agentlens.eval.runner import EvalRunner, SandboxExecutor
from agentlens.eval.schema import CaseResult, EvalResult
from agentlens.improvement.schema import ExecutorConfig, ImprovementResult


# ── budget tracking ───────────────────────────────────────────────────────────

class _BudgetExceeded(Exception):
    """Raised when the cost estimate exceeds the configured limit."""


class _BudgetTracker:
    """Character-based token estimate for approximate cost tracking.

    Rates (Claude Sonnet approximate):
        Input:  $0.003 / 1000 tokens
        Output: $0.015 / 1000 tokens
    Token estimate: len(text) / 4 (standard approximation).
    """

    _INPUT_RATE = 0.003 / 1000   # dollars per token
    _OUTPUT_RATE = 0.015 / 1000  # dollars per token

    def __init__(self, limit_dollars: float) -> None:
        self._limit = limit_dollars
        self.spent: float = 0.0

    def charge(self, prompt_text: str, response_text: str) -> None:
        """Add cost of one LLM call. Raises _BudgetExceeded if limit hit."""
        input_tokens = len(prompt_text) / 4
        output_tokens = len(response_text) / 4
        cost = input_tokens * self._INPUT_RATE + output_tokens * self._OUTPUT_RATE
        self.spent += cost
        if self.spent > self._limit:
            raise _BudgetExceeded(
                f"Budget exceeded: spent ${self.spent:.4f} against limit ${self._limit:.2f}."
            )


# ── LLM output schema ─────────────────────────────────────────────────────────

class _VariantOutput(BaseModel):
    """Structured output from the improvement LLM call."""

    hypothesis: str  # one sentence — what code change fixes this
    modified_files: dict[str, str]  # filename relative to agent_dir → complete new content

    @field_validator("hypothesis")
    @classmethod
    def non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("hypothesis must be a non-empty string")
        return v.strip()

    @field_validator("modified_files")
    @classmethod
    def at_least_one(cls, v: dict[str, str]) -> dict[str, str]:
        if not v:
            raise ValueError("modified_files must contain at least one file")
        return v


# ── system prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a multi-agent system code improvement specialist.

Given a failure diagnosis and the source files implicated by OTel span analysis,
produce a minimal, targeted code fix.

RULES:
1. Hypothesis: ONE precise sentence — what exact code change fixes this failure.
2. Return ONLY files that need to change. Do not return unchanged files.
3. Return COMPLETE file content — not diffs or partial snippets.
4. Your fix must address the specific root_cause in the diagnosis.
5. Do not change function signatures, class interfaces, or module-level exports
   that other files depend on — unless the diagnosis explicitly requires it.
6. Minimal change: fix the failure, touch nothing else.
"""


# ── prompt builder ────────────────────────────────────────────────────────────

def _build_prompt(
    diagnosis: Diagnosis,
    component_files: dict[str, str],
    failing_cases: list[CaseResult],
    prev_attempts: list[dict[str, Any]],
) -> str:
    """Build the human-turn prompt for one improvement iteration."""
    lines: list[str] = []

    # ── Diagnosis block ───────────────────────────────────────────────────────
    lines.append("## Failure Diagnosis")
    lines.append(f"- **failure_mode**: {diagnosis.failure_mode}")
    lines.append(f"- **root_cause**: {diagnosis.root_cause}")
    lines.append(f"- **confidence**: {diagnosis.confidence}")
    lines.append("- **evidence**:")
    for ev in diagnosis.evidence:
        lines.append(f"  - {ev}")
    lines.append(f"- **suggested_fix**: {diagnosis.suggested_fix}")
    lines.append("")

    # ── Source files block ────────────────────────────────────────────────────
    lines.append("## Implicated Source Files")
    lines.append(
        "These files were selected by OTel span analysis as most likely to "
        "contain the bug. Return complete replacement content for any file "
        "you change."
    )
    lines.append("")
    for rel_path, content in component_files.items():
        lines.append(f"### File: {rel_path}")
        lines.append("```python")
        lines.append(content)
        lines.append("```")
        lines.append("")

    # ── Failing cases block ───────────────────────────────────────────────────
    lines.append("## Failing Cases")
    lines.append(
        "These cases fail under the current code. Your fix must make them pass."
    )
    lines.append("")
    for case in failing_cases:
        lines.append(f"**Case {case.case_id}**")
        lines.append(f"- query: {case.query}")
        failing_criteria = [cr for cr in case.criteria_results if not cr.passed]
        if failing_criteria:
            lines.append("- failing criteria:")
            for cr in failing_criteria[:3]:
                lines.append(f"  - [{cr.criterion}] reason: {cr.reason}")
        lines.append("")

    # ── Previous attempts block ───────────────────────────────────────────────
    if prev_attempts:
        lines.append("## PREVIOUS ATTEMPTS (all failed)")
        lines.append(
            "These approaches did not produce a statistically significant improvement "
            "or caused regressions. Try a different approach."
        )
        lines.append("")
        for attempt in prev_attempts:
            lines.append(f"### Attempt {attempt['iteration']}")
            lines.append(f"- hypothesis: {attempt['hypothesis']}")
            lines.append(f"- files changed: {', '.join(attempt['files_modified'])}")
            lines.append(f"- pass rate after change: {attempt['pass_rate']:.1%}")

            if attempt.get("still_failing"):
                lines.append("- still-failing cases (sample):")
                for sf in attempt["still_failing"]:
                    lines.append(f"  - query: {sf['query']}")
                    for reason in sf.get("reasons", []):
                        lines.append(f"    - {reason}")

            if attempt.get("regressions"):
                lines.append("- REGRESSIONS caused (previously-passing cases now fail):")
                for reg in attempt["regressions"]:
                    lines.append(f"  - {reg}")

            lines.append("")

    # ── Output format instruction ─────────────────────────────────────────────
    lines.append("## Required Output")
    lines.append(
        "Respond with JSON matching this schema exactly:\n"
        "```json\n"
        "{\n"
        '  "hypothesis": "ONE sentence describing the exact code change",\n'
        '  "modified_files": {\n'
        '    "relative/path/to/file.py": "complete new file content here"\n'
        "  }\n"
        "}\n"
        "```"
    )
    lines.append(
        "Keys in modified_files must be paths relative to the agent directory "
        "(matching the paths shown above in 'Implicated Source Files')."
    )

    return "\n".join(lines)


# ── Improvement Agent ─────────────────────────────────────────────────────────

class ImprovementAgent:
    """Autonomous code fix generator for diagnosed multi-agent failures.

    Args:
        llm:            LangChain chat model with structured output support.
        judge_llm:      LangChain chat model used as the eval judge.
        config:         ExecutorConfig — how to build SandboxExecutors for variants.
        max_iterations: Maximum LLM iterations before giving up. Default 5.
        budget_dollars: Maximum estimated spend in dollars. Default $2.00.
    """

    def __init__(
        self,
        llm: Any,
        judge_llm: Any,
        config: ExecutorConfig,
        max_iterations: int = 5,
        budget_dollars: float = 2.00,
    ) -> None:
        self._llm = llm
        self._structured_llm = llm.with_structured_output(_VariantOutput)
        self._judge_llm = judge_llm
        self._config = config
        self._max_iterations = max_iterations
        self._budget = _BudgetTracker(budget_dollars)

    # ── public API ────────────────────────────────────────────────────────────

    def improve(
        self,
        diagnosis: Diagnosis,
        baseline_result: EvalResult,
        suite: BenchmarkSuite,
        component_files: dict[str, str],
    ) -> ImprovementResult:
        """Run the improvement loop and return a result.

        Args:
            diagnosis:        Structured diagnosis for one failure mode.
            baseline_result:  Full EvalResult from the current (unmodified) agent.
            suite:            Full BenchmarkSuite used to generate baseline_result.
            component_files:  {rel_path: content} from CodeLocator — files to send
                              the LLM and potentially modify.

        Returns:
            ImprovementResult — promoted=True if a valid fix was found.
        """
        # Determine which cases are failing for this failure mode
        failing_case_ids: set[str] = (
            set(diagnosis.case_ids)
            or {
                c.case_id
                for c in baseline_result.failing_cases
                if c.failure_mode == diagnosis.failure_mode
            }
        )
        failing_cases = [
            c for c in baseline_result.failing_cases
            if c.case_id in failing_case_ids
        ]

        # Build a sub-suite of only the failing cases
        failing_suite_cases = [c for c in suite.cases if c.case_id in failing_case_ids]
        failing_suite = BenchmarkSuite.new(
            suite.trace_id,
            suite.session_id,
            failing_suite_cases,
            [diagnosis.failure_mode],
        )

        # Baseline eval restricted to just the failing cases (pass_rate = 0)
        baseline_failing_eval = EvalResult.build(
            "baseline_failing",
            [c for c in baseline_result.results if c.case_id in failing_case_ids],
        )

        prev_attempts: list[dict[str, Any]] = []
        last_variant_eval = baseline_failing_eval
        last_variant_files: dict[str, str] = {}

        for iteration in range(1, self._max_iterations + 1):
            # ── 1. Generate variant ───────────────────────────────────────────
            prompt = _build_prompt(diagnosis, component_files, failing_cases, prev_attempts)
            try:
                output: _VariantOutput = self._structured_llm.invoke(
                    [SystemMessage(_SYSTEM_PROMPT), HumanMessage(prompt)]
                )
                self._budget.charge(prompt, json.dumps(output.model_dump()))
            except _BudgetExceeded as exc:
                return ImprovementResult(
                    failure_mode=diagnosis.failure_mode,
                    promoted=False,
                    iterations_run=iteration - 1,
                    cost_dollars=self._budget.spent,
                    hypothesis=prev_attempts[-1]["hypothesis"] if prev_attempts else "",
                    variant_files=last_variant_files,
                    baseline_eval=baseline_result,
                    variant_eval=last_variant_eval,
                    regression_detected=False,
                    reason=str(exc),
                )
            except Exception as exc:
                return ImprovementResult(
                    failure_mode=diagnosis.failure_mode,
                    promoted=False,
                    iterations_run=iteration - 1,
                    cost_dollars=self._budget.spent,
                    hypothesis=prev_attempts[-1]["hypothesis"] if prev_attempts else "",
                    variant_files=last_variant_files,
                    baseline_eval=baseline_result,
                    variant_eval=last_variant_eval,
                    regression_detected=False,
                    reason=f"LLM call failed at iteration {iteration}: {exc}",
                )

            # ── 2. Apply variant to temp dir ──────────────────────────────────
            temp_root, new_agent_dir = self._apply_variant(output.modified_files)
            try:
                variant_exec = SandboxExecutor(
                    agent_dir=new_agent_dir,
                    import_lines=self._config.import_lines,
                    build_app_expr=self._config.build_app_expr,
                    output_key=self._config.output_key,
                    extra_initial_state=self._config.extra_initial_state,
                    pip_packages=self._config.pip_packages,
                    e2b_api_key=self._config.e2b_api_key,
                )
                runner = EvalRunner(executor=variant_exec, judge_llm=self._judge_llm)

                # ── Phase 1: failing cases only ───────────────────────────────
                variant_eval = runner.run(failing_suite)
                last_variant_eval = variant_eval
                last_variant_files = output.modified_files

                ci_improved = (
                    not baseline_failing_eval.ci_overlaps_with(variant_eval)
                    and variant_eval.pass_rate > baseline_failing_eval.pass_rate
                )

                if ci_improved:
                    # ── Phase 2: full suite regression check ──────────────────
                    full_variant_eval = runner.run(suite)

                    regressions = [
                        c for c in full_variant_eval.results
                        if c.case_id not in failing_case_ids
                        and not c.passed
                        and any(
                            bc.case_id == c.case_id and bc.passed
                            for bc in baseline_result.results
                        )
                    ]

                    if not regressions:
                        return ImprovementResult(
                            failure_mode=diagnosis.failure_mode,
                            promoted=True,
                            iterations_run=iteration,
                            cost_dollars=self._budget.spent,
                            hypothesis=output.hypothesis,
                            variant_files=output.modified_files,
                            baseline_eval=baseline_result,
                            variant_eval=full_variant_eval,
                            regression_detected=False,
                            reason=(
                                "Wilson CIs non-overlapping — improvement is statistically real. "
                                "No regressions in full suite."
                            ),
                        )
                    else:
                        # Regressions found — record and loop
                        regression_names = [
                            f"{c.case_id}: {c.failure_mode}" for c in regressions[:3]
                        ]
                        prev_attempts.append({
                            "iteration": iteration,
                            "hypothesis": output.hypothesis,
                            "files_modified": list(output.modified_files.keys()),
                            "pass_rate": variant_eval.pass_rate,
                            "still_failing": [
                                {
                                    "query": c.query,
                                    "reasons": [
                                        cr.reason for cr in c.failing_criteria[:2]
                                    ],
                                }
                                for c in variant_eval.failing_cases[:3]
                            ],
                            "regressions": regression_names,
                        })
                        continue

                # CI still overlaps — record attempt with feedback
                prev_attempts.append({
                    "iteration": iteration,
                    "hypothesis": output.hypothesis,
                    "files_modified": list(output.modified_files.keys()),
                    "pass_rate": variant_eval.pass_rate,
                    "still_failing": [
                        {
                            "query": c.query,
                            "reasons": [
                                cr.reason for cr in c.failing_criteria[:2]
                            ],
                        }
                        for c in variant_eval.failing_cases[:3]
                    ],
                    "regressions": [],
                })

            finally:
                shutil.rmtree(temp_root, ignore_errors=True)

        # Exhausted all iterations
        return ImprovementResult(
            failure_mode=diagnosis.failure_mode,
            promoted=False,
            iterations_run=self._max_iterations,
            cost_dollars=self._budget.spent,
            hypothesis=prev_attempts[-1]["hypothesis"] if prev_attempts else "",
            variant_files=last_variant_files,
            baseline_eval=baseline_result,
            variant_eval=last_variant_eval,
            regression_detected=False,
            reason=(
                f"No statistically significant improvement after "
                f"{self._max_iterations} iterations."
            ),
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _apply_variant(
        self, modified_files: dict[str, str]
    ) -> tuple[str, str]:
        """Copy agent_dir to a temp directory and apply modified_files.

        Returns:
            (temp_root, new_agent_dir) — temp_root should be rmtree'd when done.
        """
        agent_path = Path(self._config.agent_dir).resolve()
        temp_root = tempfile.mkdtemp()
        new_agent_dir = Path(temp_root) / agent_path.name
        shutil.copytree(str(agent_path), str(new_agent_dir))

        for rel_path, content in modified_files.items():
            target = new_agent_dir / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")

        return temp_root, str(new_agent_dir)
