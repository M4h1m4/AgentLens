"""Criterion checker — evaluates expected_criteria against OTel spans and output.

Two check types in priority order:

1. Deterministic — keyword-pattern matching on known span signals. Fast, no LLM,
   100% reproducible. Covers context drop and delegation loop thresholds.

2. LLM-as-judge — for open-ended criteria that can't be inferred from spans alone
   (e.g. "output references all sources"). A separate structured LLM call returns
   {passed: bool, reason: str}. Uses a different model/prompt than the agent under
   test so the judge is independent.

If no judge LLM is provided, open-ended criteria are skipped (marked passed=True
with check_type="skipped"). This keeps the runner usable in unit tests without an
API key — all deterministic checks still run.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel

from agentlens.eval.schema import CriterionResult


# ── LLM judge schema ──────────────────────────────────────────────────────────

class _Verdict(BaseModel):
    passed: bool
    reason: str


# ── span helpers ───────────────────────────────────────────────────────────────

def _attrs(span) -> dict:
    return dict(getattr(span, "attributes", None) or {})


def _produced_sizes(spans: list) -> dict[str, int]:
    """Return len(agentlens.produced) per agent node."""
    sizes: dict[str, int] = {}
    for span in spans:
        attrs = _attrs(span)
        if attrs.get("openinference.span.kind") == "AGENT":
            node = attrs.get("agentlens.node")
            produced = attrs.get("agentlens.produced", "")
            if node and produced:
                key = str(node)
                sizes[key] = max(sizes.get(key, 0), len(str(produced)))
    return sizes


def _handoffs(spans: list) -> list[tuple[str, str]]:
    result = []
    for span in spans:
        attrs = _attrs(span)
        if attrs.get("openinference.span.kind") == "HANDOFF":
            src = attrs.get("handoff.source")
            tgt = attrs.get("handoff.target")
            if src and tgt:
                result.append((str(src), str(tgt)))
    return result


def _loop_counts(spans: list) -> dict[str, int]:
    counts: dict[str, int] = {}
    for span in spans:
        attrs = _attrs(span)
        if attrs.get("openinference.span.kind") == "AGENT":
            node = attrs.get("agentlens.node")
            if node:
                counts[str(node)] = counts.get(str(node), 0) + 1
    return counts


# ── deterministic patterns ────────────────────────────────────────────────────

_CONTEXT_DROP_PATTERNS = re.compile(
    r"(context.?drop|reduction.in.content|content.size|token.drop|size.reduc|"
    r"no.more.than.*reduc|output.size|produced.size)",
    re.IGNORECASE,
)

_LOOP_PATTERNS = re.compile(
    r"(loop|ran.more.than|delegation.loop|more.than.once|ran.only.once|"
    r"no.more.than.*time|execut.*once|runs.once)",
    re.IGNORECASE,
)

_THRESHOLD_RE = re.compile(r"(\d+)\s*%")


def _parse_threshold(criterion: str, default: float = 0.20) -> float:
    """Extract a percentage threshold from the criterion text."""
    match = _THRESHOLD_RE.search(criterion)
    if match:
        return int(match.group(1)) / 100.0
    return default


def _check_context_drop(criterion: str, spans: list) -> CriterionResult | None:
    """Check if criterion is about context/content size reduction across handoffs."""
    if not _CONTEXT_DROP_PATTERNS.search(criterion):
        return None

    threshold = _parse_threshold(criterion, default=0.20)
    sizes = _produced_sizes(spans)
    hops = _handoffs(spans)

    if not hops or not sizes:
        return CriterionResult(
            criterion=criterion,
            passed=True,
            reason="No handoffs or produced sizes found in spans — cannot verify, assuming pass.",
            check_type="deterministic",
        )

    worst_drop = 0.0
    worst_pair = ("", "")
    for src, tgt in hops:
        src_size = sizes.get(src, 0)
        tgt_size = sizes.get(tgt, 0)
        if src_size > 0 and tgt_size > 0:
            drop = (src_size - tgt_size) / src_size
            if drop > worst_drop:
                worst_drop = drop
                worst_pair = (src, tgt)

    passed = worst_drop <= threshold
    if worst_drop == 0.0:
        reason = "No measurable size reduction across any handoff."
    else:
        pct = f"{worst_drop * 100:.1f}%"
        limit = f"{threshold * 100:.0f}%"
        reason = (
            f"Largest content drop: {worst_pair[0]}→{worst_pair[1]} ({pct}). "
            f"Threshold: {limit}. {'Within limit.' if passed else 'Exceeds limit.'}"
        )

    return CriterionResult(
        criterion=criterion,
        passed=passed,
        reason=reason,
        check_type="deterministic",
    )


def _check_loop(criterion: str, spans: list) -> CriterionResult | None:
    """Check if criterion is about delegation loops / repeated agent execution."""
    if not _LOOP_PATTERNS.search(criterion):
        return None

    counts = _loop_counts(spans)
    looping = {node: c for node, c in counts.items() if c > 1}

    if not looping:
        return CriterionResult(
            criterion=criterion,
            passed=True,
            reason="No agent node ran more than once.",
            check_type="deterministic",
        )

    desc = ", ".join(f"{n} ran {c}×" for n, c in looping.items())
    return CriterionResult(
        criterion=criterion,
        passed=False,
        reason=f"Delegation loop detected: {desc}.",
        check_type="deterministic",
    )


# ── LLM judge ─────────────────────────────────────────────────────────────────

_JUDGE_SYSTEM = """\
You are an impartial evaluator assessing whether a multi-agent system's output
satisfies a specific quality criterion.

Rules:
- Be strict but fair. Only mark passed=True if the criterion is clearly satisfied.
- Your reason must be one specific sentence citing evidence from the output.
- Do not assume information not present in the output.
"""


def _llm_judge(
    criterion: str,
    query: str,
    output: str,
    structured_llm: Any,
) -> CriterionResult:
    from langchain_core.messages import HumanMessage, SystemMessage

    prompt = (
        f"Query given to the agent: {query}\n\n"
        f"Agent output:\n{output}\n\n"
        f"Criterion to check: {criterion}"
    )
    verdict: _Verdict = structured_llm.invoke(
        [SystemMessage(content=_JUDGE_SYSTEM), HumanMessage(content=prompt)]
    )
    return CriterionResult(
        criterion=criterion,
        passed=verdict.passed,
        reason=verdict.reason,
        check_type="llm_judge",
    )


# ── public API ────────────────────────────────────────────────────────────────

def check_criterion(
    criterion: str,
    query: str,
    output: str,
    spans: list,
    judge_llm: Any = None,
) -> CriterionResult:
    """Check one criterion. Deterministic first, LLM judge as fallback.

    Args:
        criterion:  Natural language assertion from BenchmarkCase.expected_criteria.
        query:      The query that was run.
        output:     The agent's text output (e.g. the report field).
        spans:      OTel spans from the run.
        judge_llm:  Pre-wrapped structured LLM returning _Verdict. Optional —
                    if None, open-ended criteria are skipped.
    """
    # 1. Try deterministic checks
    result = _check_context_drop(criterion, spans) or _check_loop(criterion, spans)
    if result is not None:
        return result

    # 2. LLM judge for open-ended criteria
    if judge_llm is not None:
        return _llm_judge(criterion, query, output, judge_llm)

    # 3. No judge available — skip
    return CriterionResult(
        criterion=criterion,
        passed=True,
        reason="No LLM judge configured — criterion not evaluated.",
        check_type="skipped",
    )


def check_all_criteria(
    criteria: list[str],
    query: str,
    output: str,
    spans: list,
    judge_llm: Any = None,
) -> list[CriterionResult]:
    """Check all criteria for one benchmark case."""
    return [
        check_criterion(criterion, query, output, spans, judge_llm)
        for criterion in criteria
    ]
