"""Tests for the Improvement Agent (Day 9).

All tests are:
  - No API keys required
  - Deterministic
  - Fast — SandboxExecutor is patched; no real E2B calls

Real file I/O is used only where needed:
  - CodeLocator: creates .py files in tmp_path to test file scanning
  - ImprovementAgent._apply_variant: copies agent_dir to temp dir (real shutil)

Test groups:
  TestVariantOutput        — Pydantic schema validation for LLM output
  TestBudgetTracker        — cost tracking and limit enforcement
  TestExecutorConfig       — ExecutorConfig dataclass
  TestImprovementResult    — ImprovementResult dataclass
  TestCodeLocator          — file selection from add_node + diagnosis text
  TestImprovementAgent     — improve() loop: promotion, exhaustion, regressions, budget
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agentlens.benchmark.schema import BenchmarkCase, BenchmarkSuite
from agentlens.diagnosis.schema import Diagnosis
from agentlens.eval.schema import CaseResult, CriterionResult, EvalResult
from agentlens.improvement import CodeLocator, ExecutorConfig, ImprovementAgent, ImprovementResult
from agentlens.improvement.agent import _BudgetExceeded, _BudgetTracker, _VariantOutput


# ── shared helpers ────────────────────────────────────────────────────────────

def _make_diagnosis(
    failure_mode: str = "context_loss",
    case_ids: list[str] | None = None,
    root_cause: str = "The synthesis agent silently drops sources",
    evidence: list[str] | None = None,
    suggested_fix: str = "Remove the [:3] slice in synthesis.py",
    confidence: str = "high",
) -> Diagnosis:
    return Diagnosis(
        failure_mode=failure_mode,
        root_cause=root_cause,
        evidence=evidence or ["5 sources fetched, 3 forwarded"],
        suggested_fix=suggested_fix,
        confidence=confidence,
        similar_past_count=0,
        case_ids=case_ids or [],
    )


def _criterion(passed: bool, reason: str = "ok") -> CriterionResult:
    return CriterionResult(
        criterion="output is well-structured",
        passed=passed,
        reason=reason,
        check_type="deterministic",
    )


def _make_case(
    failure_mode: str,
    passed: bool,
    query: str = "test query",
    case_id: str | None = None,
) -> CaseResult:
    return CaseResult(
        case_id=case_id or str(uuid.uuid4()),
        query=query,
        failure_mode=failure_mode,
        passed=passed,
        criteria_results=[_criterion(passed)],
        trace_id="t-" + uuid.uuid4().hex[:6],
        spans_captured=3,
    )


def _make_eval_result(cases: list[CaseResult]) -> EvalResult:
    return EvalResult.build(suite_id=str(uuid.uuid4()), results=cases)


def _make_suite(
    cases_data: list[tuple[str, str]],   # (query, failure_mode)
    case_ids: list[str] | None = None,
) -> BenchmarkSuite:
    """Build a BenchmarkSuite from (query, failure_mode) pairs."""
    ids = case_ids or [str(uuid.uuid4()) for _ in cases_data]
    cases = [
        BenchmarkCase(
            case_id=ids[i],
            query=q,
            expected_criteria=["output is well-structured"],
            failure_mode=fm,
            severity="high",
            rationale="test",
        )
        for i, (q, fm) in enumerate(cases_data)
    ]
    modes = list({fm for _, fm in cases_data})
    return BenchmarkSuite.new("trace-1", "session-1", cases, modes)


def _variant_output(
    hypothesis: str = "Remove the [:3] slice",
    files: dict[str, str] | None = None,
) -> _VariantOutput:
    return _VariantOutput(
        hypothesis=hypothesis,
        modified_files=files or {"agents/synthesis.py": "# fixed\n"},
    )


class _FakeStructuredLLM:
    """Returns pre-configured _VariantOutput instances on each invoke()."""

    def __init__(self, outputs: list[_VariantOutput]) -> None:
        self._outputs = outputs
        self._call_count = 0

    def invoke(self, messages: list) -> _VariantOutput:
        out = self._outputs[self._call_count % len(self._outputs)]
        self._call_count += 1
        return out

    @property
    def call_count(self) -> int:
        return self._call_count


class _FakeLLM:
    """Fake LLM whose with_structured_output() returns a _FakeStructuredLLM."""

    def __init__(self, outputs: list[_VariantOutput]) -> None:
        self._structured = _FakeStructuredLLM(outputs)

    def with_structured_output(self, schema: Any) -> _FakeStructuredLLM:
        return self._structured

    @property
    def structured(self) -> _FakeStructuredLLM:
        return self._structured


def _mock_executor(fail_queries: set[str] | None = None) -> MagicMock:
    """Mock SandboxExecutor instance: raises for fail_queries, succeeds otherwise.

    Criteria in test suites use "output is well-structured" — no deterministic
    pattern matches and judge_llm=None → checker skips it → passed=True.
    Raising causes all criteria to be marked failed (execution error path).
    """
    failing = fail_queries or set()

    def execute(query: str, session_id: str) -> tuple[str, str, list]:
        if query in failing:
            raise RuntimeError(f"Simulated execution failure for: {query}")
        return f"trace-{uuid.uuid4().hex[:8]}", "good output", []

    mock = MagicMock()
    mock.execute.side_effect = execute
    return mock


def _make_executor_config(agent_dir: str) -> ExecutorConfig:
    return ExecutorConfig(
        agent_dir=agent_dir,
        import_lines="from my_agent.graph import build_graph",
        build_app_expr="build_graph()",
        output_key="report",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# _VariantOutput schema validation
# ═══════════════════════════════════════════════════════════════════════════════

class TestVariantOutput:

    def test_valid_output_accepted(self):
        out = _variant_output()
        assert out.hypothesis == "Remove the [:3] slice"
        assert "agents/synthesis.py" in out.modified_files

    def test_empty_hypothesis_rejected(self):
        with pytest.raises(Exception):
            _VariantOutput(hypothesis="", modified_files={"f.py": "x"})

    def test_whitespace_hypothesis_rejected(self):
        with pytest.raises(Exception):
            _VariantOutput(hypothesis="   ", modified_files={"f.py": "x"})

    def test_empty_modified_files_rejected(self):
        with pytest.raises(Exception):
            _VariantOutput(hypothesis="fix it", modified_files={})

    def test_hypothesis_is_stripped(self):
        out = _VariantOutput(hypothesis="  fix it  ", modified_files={"f.py": "x"})
        assert out.hypothesis == "fix it"

    def test_multiple_files_accepted(self):
        out = _VariantOutput(
            hypothesis="Fix two files",
            modified_files={"a.py": "x", "b.py": "y"},
        )
        assert len(out.modified_files) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# _BudgetTracker
# ═══════════════════════════════════════════════════════════════════════════════

class TestBudgetTracker:

    def test_initial_spent_is_zero(self):
        tracker = _BudgetTracker(limit_dollars=2.0)
        assert tracker.spent == 0.0

    def test_charge_accumulates(self):
        tracker = _BudgetTracker(limit_dollars=10.0)
        tracker.charge("a" * 4000, "b" * 400)
        assert tracker.spent > 0.0

    def test_charge_raises_when_limit_exceeded(self):
        tracker = _BudgetTracker(limit_dollars=0.0001)
        with pytest.raises(_BudgetExceeded):
            tracker.charge("x" * 10_000, "y" * 1_000)

    def test_charge_does_not_raise_under_limit(self):
        tracker = _BudgetTracker(limit_dollars=100.0)
        tracker.charge("short", "short")
        assert tracker.spent < 100.0

    def test_spent_grows_with_each_call(self):
        tracker = _BudgetTracker(limit_dollars=100.0)
        tracker.charge("a" * 100, "b" * 100)
        after_first = tracker.spent
        tracker.charge("a" * 100, "b" * 100)
        assert tracker.spent > after_first

    def test_budget_exceeded_message_contains_amounts(self):
        tracker = _BudgetTracker(limit_dollars=0.0)
        try:
            tracker.charge("x" * 100_000, "y" * 10_000)
        except _BudgetExceeded as exc:
            assert "$" in str(exc)


# ═══════════════════════════════════════════════════════════════════════════════
# ExecutorConfig and ImprovementResult dataclasses
# ═══════════════════════════════════════════════════════════════════════════════

class TestExecutorConfig:

    def test_required_fields_set(self):
        cfg = ExecutorConfig(
            agent_dir="/path/to/agent",
            import_lines="from agent import build_graph",
            build_app_expr="build_graph()",
        )
        assert cfg.agent_dir == "/path/to/agent"
        assert cfg.output_key == "report"          # default
        assert cfg.extra_initial_state is None     # default
        assert cfg.pip_packages is None            # default
        assert cfg.e2b_api_key is None             # default

    def test_all_optional_fields_set(self):
        cfg = ExecutorConfig(
            agent_dir="/path",
            import_lines="import x",
            build_app_expr="x.build()",
            output_key="result",
            extra_initial_state={"loop_count": 0},
            pip_packages=["langchain"],
            e2b_api_key="sk-test",
        )
        assert cfg.output_key == "result"
        assert cfg.extra_initial_state == {"loop_count": 0}
        assert cfg.pip_packages == ["langchain"]
        assert cfg.e2b_api_key == "sk-test"


class TestImprovementResult:

    def _ev(self) -> EvalResult:
        return _make_eval_result([_make_case("context_loss", True)])

    def test_fields_accessible(self):
        ev = self._ev()
        result = ImprovementResult(
            failure_mode="context_loss",
            promoted=True,
            iterations_run=2,
            cost_dollars=0.05,
            hypothesis="Remove the cap",
            variant_files={"f.py": "x"},
            baseline_eval=ev,
            variant_eval=ev,
            regression_detected=False,
            reason="CIs non-overlapping",
        )
        assert result.promoted is True
        assert result.iterations_run == 2
        assert result.failure_mode == "context_loss"
        assert result.regression_detected is False
        assert result.cost_dollars == 0.05


# ═══════════════════════════════════════════════════════════════════════════════
# CodeLocator
# ═══════════════════════════════════════════════════════════════════════════════

class TestCodeLocator:

    def test_empty_dir_returns_empty_dict(self, tmp_path: Path):
        result = CodeLocator().locate(str(tmp_path), _make_diagnosis())
        assert result == {}

    def test_finds_registration_file_with_add_node(self, tmp_path: Path):
        (tmp_path / "graph.py").write_text(
            'graph.add_node("synthesis", make_synthesis_agent(config))\n'
        )
        diagnosis = _make_diagnosis(
            root_cause="The synthesis agent drops sources",
            suggested_fix="Fix synthesis",
        )
        result = CodeLocator().locate(str(tmp_path), diagnosis)
        assert "graph.py" in result

    def test_filters_to_nodes_mentioned_in_diagnosis(self, tmp_path: Path):
        (tmp_path / "graph.py").write_text(
            'graph.add_node("synthesis", fn1)\n'
            'graph.add_node("writer", fn2)\n'
            'graph.add_node("search", fn3)\n'
        )
        (tmp_path / "synthesis.py").write_text("def make_synthesis_agent(): pass\n")
        (tmp_path / "writer.py").write_text("def make_writer_agent(): pass\n")

        # Diagnosis only mentions "synthesis" — writer and search unrelated
        diagnosis = _make_diagnosis(
            root_cause="The synthesis agent drops sources",
            evidence=["synthesis produced fewer tokens"],
            suggested_fix="Fix synthesis agent transformation",
        )
        result = CodeLocator().locate(str(tmp_path), diagnosis)

        assert any("synthesis" in k for k in result)
        assert "graph.py" in result  # registration file always included

    def test_falls_back_to_all_nodes_when_none_in_diagnosis(self, tmp_path: Path):
        (tmp_path / "graph.py").write_text(
            'graph.add_node("synthesis", fn1)\n'
            'graph.add_node("writer", fn2)\n'
        )
        (tmp_path / "synthesis.py").write_text("def make_synthesis_agent(): pass\n")

        # Diagnosis text mentions neither "synthesis" nor "writer"
        diagnosis = _make_diagnosis(
            root_cause="An unexpected error occurred",
            evidence=["something went wrong"],
            suggested_fix="Check the implementation",
        )
        result = CodeLocator().locate(str(tmp_path), diagnosis)
        # Fallback: graph.py (the registration file) included
        assert "graph.py" in result

    def test_follows_local_imports_one_level(self, tmp_path: Path):
        (tmp_path / "graph.py").write_text(
            'graph.add_node("synthesis", make_synthesis_agent)\n'
        )
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        (agents_dir / "__init__.py").write_text("")
        (agents_dir / "synthesis.py").write_text(
            "from agents.state import AgentState\n"
            "def make_synthesis_agent(): pass\n"
        )
        (agents_dir / "state.py").write_text("class AgentState: pass\n")

        diagnosis = _make_diagnosis(
            root_cause="synthesis drops sources",
            suggested_fix="fix synthesis",
        )
        result = CodeLocator().locate(str(tmp_path), diagnosis)

        # synthesis.py must be found (function name contains "synthesis")
        assert any("synthesis" in k for k in result)

    def test_skips_pycache(self, tmp_path: Path):
        pycache = tmp_path / "__pycache__"
        pycache.mkdir()
        (pycache / "graph.cpython-311.pyc").write_bytes(b"\xde\xad\xbe\xef")
        (tmp_path / "graph.py").write_text('graph.add_node("synthesis", fn)\n')

        result = CodeLocator().locate(str(tmp_path), _make_diagnosis())
        assert not any("pycache" in k for k in result)

    def test_word_boundary_matching(self, tmp_path: Path):
        """Node 'search' must NOT be selected when diagnosis only says 'research'."""
        (tmp_path / "graph.py").write_text(
            'graph.add_node("search", fn1)\n'
            'graph.add_node("synthesis", fn2)\n'
        )
        (tmp_path / "synthesis.py").write_text("def make_synthesis_agent(): pass\n")

        # "research" contains "search" as a substring — must not trigger the "search" node
        diagnosis = _make_diagnosis(
            root_cause="The research shows synthesis drops sources",
            evidence=["synthesis output was smaller"],
            suggested_fix="Fix synthesis research transformation",
        )
        result = CodeLocator().locate(str(tmp_path), diagnosis)

        # synthesis matched → synthesis.py included
        assert any("synthesis" in k for k in result)
        # "search" node should NOT be matched (only "research" appears, not bare "search")
        assert not any(k == "search.py" for k in result)

    def test_returns_file_contents(self, tmp_path: Path):
        content = 'graph.add_node("synthesis", fn)\n'
        (tmp_path / "graph.py").write_text(content)
        result = CodeLocator().locate(str(tmp_path), _make_diagnosis())
        assert result.get("graph.py") == content

    def test_single_quotes_in_add_node_detected(self, tmp_path: Path):
        """add_node with single-quoted name must also be detected."""
        (tmp_path / "graph.py").write_text(
            "graph.add_node('synthesis', fn)\n"
        )
        diagnosis = _make_diagnosis(root_cause="synthesis drops sources")
        result = CodeLocator().locate(str(tmp_path), diagnosis)
        assert "graph.py" in result


# ═══════════════════════════════════════════════════════════════════════════════
# ImprovementAgent
# ═══════════════════════════════════════════════════════════════════════════════

class TestImprovementAgent:
    """Tests patch SandboxExecutor so no real E2B calls are made.

    Each test uses a real tmp_path agent_dir because _apply_variant calls
    shutil.copytree — the directory must exist on disk.
    """

    @pytest.fixture()
    def agent_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "my_agent"
        d.mkdir()
        (d / "__init__.py").write_text("")
        (d / "graph.py").write_text('graph.add_node("synthesis", fn)\n')
        return d

    @pytest.fixture()
    def cfg(self, agent_dir: Path) -> ExecutorConfig:
        return _make_executor_config(str(agent_dir))

    def _agent(
        self,
        cfg: ExecutorConfig,
        outputs: list[_VariantOutput],
        max_iterations: int = 3,
        budget_dollars: float = 10.0,
    ) -> ImprovementAgent:
        return ImprovementAgent(
            llm=_FakeLLM(outputs),
            judge_llm=None,
            config=cfg,
            max_iterations=max_iterations,
            budget_dollars=budget_dollars,
        )

    # ── promotion ─────────────────────────────────────────────────────────────

    def test_promoted_when_ci_non_overlapping_and_no_regressions(self, cfg: ExecutorConfig):
        """4 failing → variant makes all 4 pass → Wilson CIs non-overlapping → promoted."""
        ids = [str(uuid.uuid4()) for _ in range(4)]
        baseline = _make_eval_result([_make_case("context_loss", False, f"q{i}", ids[i]) for i in range(4)])
        suite = _make_suite([(f"q{i}", "context_loss") for i in range(4)], case_ids=ids)
        diagnosis = _make_diagnosis("context_loss", case_ids=ids)

        agent = self._agent(cfg, [_variant_output()])

        with patch("agentlens.improvement.agent.SandboxExecutor") as MockExec:
            MockExec.return_value = _mock_executor()
            result = agent.improve(diagnosis, baseline, suite, {"graph.py": "x"})

        assert result.promoted is True
        assert result.iterations_run == 1
        assert result.regression_detected is False
        assert result.failure_mode == "context_loss"
        assert "non-overlapping" in result.reason.lower() or "statistically" in result.reason.lower()

    def test_promoted_result_carries_variant_files(self, cfg: ExecutorConfig):
        """variant_files in ImprovementResult must match what the LLM returned."""
        ids = [str(uuid.uuid4()) for _ in range(4)]
        baseline = _make_eval_result([_make_case("context_loss", False, f"q{i}", ids[i]) for i in range(4)])
        suite = _make_suite([(f"q{i}", "context_loss") for i in range(4)], case_ids=ids)
        diagnosis = _make_diagnosis("context_loss", case_ids=ids)

        sentinel = "# VARIANT CONTENT\ndef fixed(): pass\n"
        agent = self._agent(cfg, [_variant_output(files={"graph.py": sentinel})])

        with patch("agentlens.improvement.agent.SandboxExecutor") as MockExec:
            MockExec.return_value = _mock_executor()
            result = agent.improve(diagnosis, baseline, suite, {"graph.py": "x"})

        assert result.variant_files.get("graph.py") == sentinel

    def test_result_carries_baseline_and_variant_evals(self, cfg: ExecutorConfig):
        ids = [str(uuid.uuid4()) for _ in range(4)]
        baseline = _make_eval_result([_make_case("context_loss", False, f"q{i}", ids[i]) for i in range(4)])
        suite = _make_suite([(f"q{i}", "context_loss") for i in range(4)], case_ids=ids)
        diagnosis = _make_diagnosis("context_loss", case_ids=ids)

        agent = self._agent(cfg, [_variant_output()])

        with patch("agentlens.improvement.agent.SandboxExecutor") as MockExec:
            MockExec.return_value = _mock_executor()
            result = agent.improve(diagnosis, baseline, suite, {"graph.py": "x"})

        assert result.baseline_eval is baseline
        assert isinstance(result.variant_eval, EvalResult)

    # ── exhaustion ────────────────────────────────────────────────────────────

    def test_not_promoted_after_max_iterations_exhausted(self, cfg: ExecutorConfig):
        """Executor always raises → variant always 0/4 → CIs overlap → exhaust iterations."""
        ids = [str(uuid.uuid4()) for _ in range(4)]
        baseline = _make_eval_result([_make_case("context_loss", False, f"q{i}", ids[i]) for i in range(4)])
        suite = _make_suite([(f"q{i}", "context_loss") for i in range(4)], case_ids=ids)
        diagnosis = _make_diagnosis("context_loss", case_ids=ids)

        agent = self._agent(cfg, [_variant_output()] * 5, max_iterations=2)

        with patch("agentlens.improvement.agent.SandboxExecutor") as MockExec:
            MockExec.return_value = _mock_executor(fail_queries={f"q{i}" for i in range(4)})
            result = agent.improve(diagnosis, baseline, suite, {"graph.py": "x"})

        assert result.promoted is False
        assert result.iterations_run == 2
        assert "No statistically significant" in result.reason

    def test_iterations_run_reflects_actual_attempts(self, cfg: ExecutorConfig):
        ids = [str(uuid.uuid4()) for _ in range(4)]
        baseline = _make_eval_result([_make_case("context_loss", False, f"q{i}", ids[i]) for i in range(4)])
        suite = _make_suite([(f"q{i}", "context_loss") for i in range(4)], case_ids=ids)
        diagnosis = _make_diagnosis("context_loss", case_ids=ids)

        agent = self._agent(cfg, [_variant_output()] * 10, max_iterations=3)

        with patch("agentlens.improvement.agent.SandboxExecutor") as MockExec:
            MockExec.return_value = _mock_executor(fail_queries={f"q{i}" for i in range(4)})
            result = agent.improve(diagnosis, baseline, suite, {"graph.py": "x"})

        assert result.iterations_run == 3

    # ── budget ────────────────────────────────────────────────────────────────

    def test_budget_exceeded_returns_early_with_reason(self, cfg: ExecutorConfig):
        """Budget of $0 triggers _BudgetExceeded on the first LLM charge."""
        ids = [str(uuid.uuid4()) for _ in range(4)]
        baseline = _make_eval_result([_make_case("context_loss", False, f"q{i}", ids[i]) for i in range(4)])
        suite = _make_suite([(f"q{i}", "context_loss") for i in range(4)], case_ids=ids)
        diagnosis = _make_diagnosis("context_loss", case_ids=ids)

        agent = self._agent(cfg, [_variant_output()], budget_dollars=0.0)

        with patch("agentlens.improvement.agent.SandboxExecutor"):
            result = agent.improve(diagnosis, baseline, suite, {"graph.py": "x"})

        assert result.promoted is False
        assert "Budget exceeded" in result.reason
        assert result.iterations_run == 0

    # ── LLM errors ────────────────────────────────────────────────────────────

    def test_llm_error_returns_gracefully(self, cfg: ExecutorConfig):
        """An LLM exception must not crash improve() — returns promoted=False."""
        ids = [str(uuid.uuid4()) for _ in range(2)]
        baseline = _make_eval_result([_make_case("context_loss", False, f"q{i}", ids[i]) for i in range(2)])
        suite = _make_suite([(f"q{i}", "context_loss") for i in range(2)], case_ids=ids)
        diagnosis = _make_diagnosis("context_loss", case_ids=ids)

        class _ErrorLLM:
            def with_structured_output(self, schema: Any):
                class _Raiser:
                    def invoke(self, messages):
                        raise ValueError("LLM API unavailable")
                return _Raiser()

        agent = ImprovementAgent(
            llm=_ErrorLLM(),
            judge_llm=None,
            config=cfg,
            max_iterations=3,
        )

        with patch("agentlens.improvement.agent.SandboxExecutor"):
            result = agent.improve(diagnosis, baseline, suite, {"graph.py": "x"})

        assert result.promoted is False
        assert "LLM call failed" in result.reason

    # ── regressions ───────────────────────────────────────────────────────────

    def test_regression_blocks_promotion(self, cfg: ExecutorConfig):
        """Variant passes Phase 1 (CI improves) but Phase 2 reveals regressions → not promoted."""
        # 2 context_loss failing + 2 delegation_loop passing in baseline
        cl_ids = [str(uuid.uuid4()) for _ in range(2)]
        dl_ids = [str(uuid.uuid4()) for _ in range(2)]

        cl_cases = [_make_case("context_loss", False, f"cl-q{i}", cl_ids[i]) for i in range(2)]
        dl_cases = [_make_case("delegation_loop", True, f"dl-q{i}", dl_ids[i]) for i in range(2)]
        baseline = _make_eval_result(cl_cases + dl_cases)

        suite = _make_suite(
            [(f"cl-q{i}", "context_loss") for i in range(2)]
            + [(f"dl-q{i}", "delegation_loop") for i in range(2)],
            case_ids=cl_ids + dl_ids,
        )
        diagnosis = _make_diagnosis("context_loss", case_ids=cl_ids)

        # Executor: delegation_loop queries fail → previously-passing cases now fail
        agent = self._agent(cfg, [_variant_output()] * 5, max_iterations=2)

        with patch("agentlens.improvement.agent.SandboxExecutor") as MockExec:
            MockExec.return_value = _mock_executor(
                fail_queries={f"dl-q{i}" for i in range(2)}
            )
            result = agent.improve(diagnosis, baseline, suite, {"graph.py": "x"})

        # Regression blocked promotion even though Phase 1 improved
        assert result.promoted is False

    # ── case filtering ────────────────────────────────────────────────────────

    def test_only_targeted_cases_run_in_phase1(self, cfg: ExecutorConfig):
        """Phase 1 runs only diagnosis.case_ids cases, not the full suite.

        10 total cases, 4 targeted. With 4/4 passing vs 0/4 baseline the
        Wilson CIs are non-overlapping (0.51 > 0.49), so Phase 2 also runs.
        Total executions = 4 (Phase 1) + 10 (Phase 2) = 14.
        If Phase 1 had run all 10 the total would be 10 + 10 = 20.
        """
        all_ids = [str(uuid.uuid4()) for _ in range(10)]
        targeted_ids = all_ids[:4]   # 4 targeted — enough for CI separation

        baseline = _make_eval_result([
            _make_case("context_loss", False, f"q{i}", all_ids[i]) for i in range(10)
        ])
        suite = _make_suite(
            [(f"q{i}", "context_loss") for i in range(10)],
            case_ids=all_ids,
        )
        diagnosis = _make_diagnosis("context_loss", case_ids=targeted_ids)

        agent = self._agent(cfg, [_variant_output()], max_iterations=1)

        execute_count = 0

        def counting_execute(query: str, session_id: str) -> tuple[str, str, list]:
            nonlocal execute_count
            execute_count += 1
            return f"trace-{uuid.uuid4().hex[:8]}", "out", []

        with patch("agentlens.improvement.agent.SandboxExecutor") as MockExec:
            mock_inst = MagicMock()
            mock_inst.execute.side_effect = counting_execute
            MockExec.return_value = mock_inst
            agent.improve(diagnosis, baseline, suite, {"graph.py": "x"})

        # Phase 1: 4 targeted. Phase 2: 10 full suite. Total = 14.
        assert execute_count == 4 + 10

    def test_diagnosis_case_ids_take_priority_over_failure_mode_filter(self, cfg: ExecutorConfig):
        """diagnosis.case_ids is used when non-empty, even if baseline has more failing cases.

        8 total cases, 4 targeted. Phase 1 runs only the 4 targeted cases,
        achieves CI separation (4/4 vs 0/4), then Phase 2 runs all 8.
        Total = 4 + 8 = 12 (vs 8 + 8 = 16 if all cases were used in Phase 1).
        """
        all_ids = [str(uuid.uuid4()) for _ in range(8)]
        targeted_ids = all_ids[:4]   # 4 of 8

        baseline = _make_eval_result([
            _make_case("context_loss", False, f"q{i}", all_ids[i]) for i in range(8)
        ])
        suite = _make_suite(
            [(f"q{i}", "context_loss") for i in range(8)],
            case_ids=all_ids,
        )
        diagnosis = _make_diagnosis("context_loss", case_ids=targeted_ids)

        agent = self._agent(cfg, [_variant_output()], max_iterations=1)

        execute_count = 0

        def counting_execute(query: str, session_id: str) -> tuple[str, str, list]:
            nonlocal execute_count
            execute_count += 1
            return f"trace-{uuid.uuid4().hex[:8]}", "out", []

        with patch("agentlens.improvement.agent.SandboxExecutor") as MockExec:
            mock_inst = MagicMock()
            mock_inst.execute.side_effect = counting_execute
            MockExec.return_value = mock_inst
            agent.improve(diagnosis, baseline, suite, {"graph.py": "x"})

        # Phase 1: 4 targeted. Phase 2: 8 full suite. Total = 12.
        assert execute_count == 4 + 8
