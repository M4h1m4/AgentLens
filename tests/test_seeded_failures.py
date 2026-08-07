"""Ground-truth harness for the five seeded failures (Day 7).

Part 1 — Agent state tests (existing):
    Each test asserts the failure manifests in agent state when its flag is ON
    and disappears when OFF.

Part 2 — Eval pipeline detection tests (Day 7):
    Each test runs EvalRunner with StateInspectingExecutor against the buggy
    and clean configs, then asserts the runner correctly flags failures and
    produces statistically distinct Wilson CI intervals.
"""

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from reference_agent.config import AgentConfig, SeededFailureConfig
from reference_agent.run import invoke, reset_bleed_cache

# Deterministic stand-in so tests need no API key and never vary. The seeded
# failures are structural, so the model's output does not affect them.
LLM = FakeListChatModel(responses=["[auto-generated summary]"])


@pytest.fixture(autouse=True)
def _isolate_bleed():
    # Failure 5 uses a module-global buffer; clear it around every test.
    reset_bleed_cache()
    yield
    reset_bleed_cache()


def _config(**flags) -> AgentConfig:
    return AgentConfig(failures=SeededFailureConfig(**flags))


# --- Failure 1: context loss ------------------------------------------------

def test_context_loss_drops_sources_when_on():
    state = invoke("renewable energy", config=_config(context_loss=True), llm=LLM)
    # 5 sources returned, cap is 3 -> only 3 survive, 2 silently dropped.
    assert len(state["synthesis_complete"]["key_claims"]) == 3
    assert len(state["dropped_sources"]) == 2


def test_context_loss_absent_when_off():
    state = invoke("renewable energy", config=_config(context_loss=False), llm=LLM)
    assert len(state["synthesis_complete"]["key_claims"]) == 5
    assert state["dropped_sources"] == []


# --- Failure 2: race condition ----------------------------------------------

def test_race_condition_uses_partial_when_on():
    state = invoke("renewable energy", config=_config(race_condition=True), llm=LLM)
    assert state["writer_input"] == "partial"
    # Partial synthesis carries no evidence, so the report omits that section.
    assert "Supporting evidence" not in state["report"]


def test_race_condition_uses_complete_when_off():
    state = invoke("renewable energy", config=_config(race_condition=False), llm=LLM)
    assert state["writer_input"] == "complete"
    assert "Supporting evidence" in state["report"]


# --- Failure 3: contradiction ignored ---------------------------------------

def test_contradiction_ignored_when_on():
    state = invoke("nuclear conflict debate", config=_config(contradiction_ignored=True), llm=LLM)
    assert state["synthesis_complete"]["conflicting_viewpoints"] == []


def test_contradiction_flagged_when_off():
    state = invoke("nuclear conflict debate", config=_config(contradiction_ignored=False), llm=LLM)
    assert len(state["synthesis_complete"]["conflicting_viewpoints"]) > 0


# --- Failure 4: delegation loop ---------------------------------------------

def test_delegation_loop_when_on():
    state = invoke("exhaustive review of grid storage", config=_config(delegation_loop=True), llm=LLM)
    # Trigger query with no terminator -> synthesis runs many times.
    assert state["loop_count"] > 1


def test_no_loop_when_off():
    state = invoke("exhaustive review of grid storage", config=_config(delegation_loop=False), llm=LLM)
    assert state["loop_count"] == 1


def test_no_loop_for_non_trigger_query():
    state = invoke("quick summary of grid storage", config=_config(delegation_loop=True), llm=LLM)
    assert state["loop_count"] == 1


# --- Failure 5: context bleed -----------------------------------------------

def test_context_bleed_when_on():
    cfg = _config(context_bleed=True)
    invoke("alpha topic", session_id="s1", config=cfg, llm=LLM)
    second = invoke("beta topic", session_id="s1", config=cfg, llm=LLM)
    # The second report leaks the first run's content.
    assert "alpha topic" in second["report"]
    assert "carried over from previous run" in second["report"]


def test_no_context_bleed_when_off():
    cfg = _config(context_bleed=False)
    invoke("alpha topic", session_id="s1", config=cfg, llm=LLM)
    second = invoke("beta topic", session_id="s1", config=cfg, llm=LLM)
    assert "alpha topic" not in second["report"]


# --- All five off: a fully correct run --------------------------------------

def test_clean_config_has_no_failures():
    cfg = AgentConfig(failures=SeededFailureConfig.clean())
    state = invoke("nuclear conflict debate", config=cfg, llm=LLM)
    assert len(state["synthesis_complete"]["key_claims"]) == 3  # 3 conflict sources
    assert state["dropped_sources"] == []
    assert state["writer_input"] == "complete"
    assert len(state["synthesis_complete"]["conflicting_viewpoints"]) > 0
    assert state["loop_count"] == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Part 2 — Eval pipeline detection tests (Day 7)
#
# These tests run EvalRunner with StateInspectingExecutor against real agent
# configs, verifying that the criterion checker catches each seeded failure
# through spans and output content — no global OTel, no API key required.
# ═══════════════════════════════════════════════════════════════════════════════

from agentlens.eval import EvalRunner
from agentlens.eval.seeded_suite import (
    PatternJudgeLLM,
    StateInspectingExecutor,
    make_context_bleed_suite,
    make_context_loss_suite,
    make_contradiction_suite,
    make_delegation_loop_suite,
    make_race_condition_suite,
)


def _buggy_config(**only_these_on) -> AgentConfig:
    """All failures OFF except the ones specified."""
    clean = SeededFailureConfig.clean()
    for k, v in only_these_on.items():
        setattr(clean, k, v)
    return AgentConfig(failures=clean)


def _clean_config() -> AgentConfig:
    return AgentConfig(failures=SeededFailureConfig.clean())


def _fake_llm(n: int = 20) -> FakeListChatModel:
    """Enough responses to cover n LLM (writer) invocations."""
    return FakeListChatModel(responses=["[auto-generated summary]"] * n)


class TestEvalPipelineDetection:
    """Verify EvalRunner catches each seeded failure via spans + output content.

    Each test:
      1. Runs the suite against the BUGGY config  → all cases should FAIL.
      2. Runs the same suite against the CLEAN config → all cases should PASS.
    """

    # ── Failure 1: context loss ────────────────────────────────────────────────

    def test_context_loss_detected_by_eval(self):
        suite = make_context_loss_suite(n=5)
        judge = PatternJudgeLLM()

        # Buggy: synthesis cap=3, 5 sources fetched → 2 silently dropped
        buggy_exec = StateInspectingExecutor(
            _buggy_config(context_loss=True), _fake_llm()
        )
        buggy_result = EvalRunner(executor=buggy_exec, judge_llm=judge).run(suite)
        assert buggy_result.pass_rate == 0.0, (
            f"Expected all cases to fail with context_loss ON; got {buggy_result.pass_rate:.0%}"
        )

        # Clean: all 5 sources processed → all titles in output
        clean_exec = StateInspectingExecutor(_clean_config(), _fake_llm())
        clean_result = EvalRunner(executor=clean_exec, judge_llm=judge).run(suite)
        assert clean_result.pass_rate == 1.0, (
            f"Expected all cases to pass with context_loss OFF; got {clean_result.pass_rate:.0%}"
        )

    # ── Failure 2: race condition ──────────────────────────────────────────────

    def test_race_condition_detected_by_eval(self):
        suite = make_race_condition_suite(n=5)
        judge = PatternJudgeLLM()

        buggy_exec = StateInspectingExecutor(
            _buggy_config(race_condition=True), _fake_llm()
        )
        buggy_result = EvalRunner(executor=buggy_exec, judge_llm=judge).run(suite)
        assert buggy_result.pass_rate == 0.0, (
            f"Expected all cases to fail with race_condition ON; got {buggy_result.pass_rate:.0%}"
        )

        clean_exec = StateInspectingExecutor(_clean_config(), _fake_llm())
        clean_result = EvalRunner(executor=clean_exec, judge_llm=judge).run(suite)
        assert clean_result.pass_rate == 1.0, (
            f"Expected all cases to pass with race_condition OFF; got {clean_result.pass_rate:.0%}"
        )

    # ── Failure 3: contradiction ignored ──────────────────────────────────────

    def test_contradiction_detected_by_eval(self):
        suite = make_contradiction_suite(n=5)
        judge = PatternJudgeLLM()

        buggy_exec = StateInspectingExecutor(
            _buggy_config(contradiction_ignored=True), _fake_llm()
        )
        buggy_result = EvalRunner(executor=buggy_exec, judge_llm=judge).run(suite)
        assert buggy_result.pass_rate == 0.0, (
            f"Expected all cases to fail with contradiction_ignored ON; got {buggy_result.pass_rate:.0%}"
        )

        clean_exec = StateInspectingExecutor(_clean_config(), _fake_llm())
        clean_result = EvalRunner(executor=clean_exec, judge_llm=judge).run(suite)
        assert clean_result.pass_rate == 1.0, (
            f"Expected all cases to pass with contradiction_ignored OFF; got {clean_result.pass_rate:.0%}"
        )

    # ── Failure 4: delegation loop ─────────────────────────────────────────────

    def test_delegation_loop_detected_by_eval(self):
        suite = make_delegation_loop_suite(n=5)
        # Loop detection is deterministic — no LLM judge needed, but passing
        # one is harmless and exercises the full code path.
        judge = PatternJudgeLLM()

        # Low max_loop_iterations so the test finishes quickly (2 loops per case).
        loop_config = AgentConfig(
            failures=SeededFailureConfig(
                context_loss=False,
                race_condition=False,
                contradiction_ignored=False,
                delegation_loop=True,
                context_bleed=False,
            ),
            max_loop_iterations=2,
        )
        buggy_exec = StateInspectingExecutor(loop_config, _fake_llm(30))
        buggy_result = EvalRunner(executor=buggy_exec, judge_llm=judge).run(suite)
        assert buggy_result.pass_rate == 0.0, (
            f"Expected all cases to fail with delegation_loop ON; got {buggy_result.pass_rate:.0%}"
        )

        clean_exec = StateInspectingExecutor(_clean_config(), _fake_llm())
        clean_result = EvalRunner(executor=clean_exec, judge_llm=judge).run(suite)
        assert clean_result.pass_rate == 1.0, (
            f"Expected all cases to pass with delegation_loop OFF; got {clean_result.pass_rate:.0%}"
        )

    # ── Failure 5: context bleed ───────────────────────────────────────────────

    def test_context_bleed_detected_by_eval(self):
        suite = make_context_bleed_suite(n=5)
        judge = PatternJudgeLLM()

        # prewarm_bleed seeds the bleed cache before each case so the "carried
        # over" marker always appears in the report when the bug is ON.
        buggy_exec = StateInspectingExecutor(
            _buggy_config(context_bleed=True),
            _fake_llm(30),
            prewarm_bleed="alpha topic seed run",
        )
        buggy_result = EvalRunner(executor=buggy_exec, judge_llm=judge).run(suite)
        assert buggy_result.pass_rate == 0.0, (
            f"Expected all cases to fail with context_bleed ON; got {buggy_result.pass_rate:.0%}"
        )

        clean_exec = StateInspectingExecutor(_clean_config(), _fake_llm())
        clean_result = EvalRunner(executor=clean_exec, judge_llm=judge).run(suite)
        assert clean_result.pass_rate == 1.0, (
            f"Expected all cases to pass with context_bleed OFF; got {clean_result.pass_rate:.0%}"
        )

    # ── All five failures: clean agent passes everything ───────────────────────

    def test_clean_agent_passes_all_eval_checks(self):
        """A fully correct agent should pass all five suites end-to-end."""
        judge = PatternJudgeLLM()
        clean_cfg = _clean_config()

        suites = [
            make_context_loss_suite(n=3),
            make_race_condition_suite(n=3),
            make_contradiction_suite(n=3),
            make_delegation_loop_suite(n=3),
        ]
        for suite in suites:
            executor = StateInspectingExecutor(clean_cfg, _fake_llm(10))
            result = EvalRunner(executor=executor, judge_llm=judge).run(suite)
            assert result.pass_rate == 1.0, (
                f"Clean agent failed suite {suite.suite_id}: {result.pass_rate:.0%}"
            )


class TestWilsonCICalibration:
    """Verify Wilson CI intervals are non-overlapping between clean and buggy.

    5 cases is enough to produce non-overlapping CIs when pass_rate is 0% vs
    100%, which is exactly the signal the Day 9 Improvement Agent relies on
    to gate fix promotions.
    """

    def test_delegation_loop_ci_non_overlapping(self):
        """Clean (0 loops) and buggy (>1 loops) CIs must not overlap."""
        suite = make_delegation_loop_suite(n=5)
        judge = PatternJudgeLLM()

        loop_config = AgentConfig(
            failures=SeededFailureConfig(
                context_loss=False,
                race_condition=False,
                contradiction_ignored=False,
                delegation_loop=True,
                context_bleed=False,
            ),
            max_loop_iterations=2,
        )

        buggy_result = EvalRunner(
            executor=StateInspectingExecutor(loop_config, _fake_llm(20)),
            judge_llm=judge,
        ).run(suite)

        clean_result = EvalRunner(
            executor=StateInspectingExecutor(_clean_config(), _fake_llm()),
            judge_llm=judge,
        ).run(suite)

        # The two CIs must not overlap — this is what gates fix promotion in Day 9.
        assert not clean_result.ci_overlaps_with(buggy_result), (
            f"CIs overlap: clean=[{clean_result.ci_low:.2f},{clean_result.ci_high:.2f}] "
            f"buggy=[{buggy_result.ci_low:.2f},{buggy_result.ci_high:.2f}]"
        )

    def test_race_condition_ci_non_overlapping(self):
        suite = make_race_condition_suite(n=5)
        judge = PatternJudgeLLM()

        buggy_result = EvalRunner(
            executor=StateInspectingExecutor(_buggy_config(race_condition=True), _fake_llm()),
            judge_llm=judge,
        ).run(suite)

        clean_result = EvalRunner(
            executor=StateInspectingExecutor(_clean_config(), _fake_llm()),
            judge_llm=judge,
        ).run(suite)

        assert not clean_result.ci_overlaps_with(buggy_result), (
            f"CIs overlap: clean=[{clean_result.ci_low:.2f},{clean_result.ci_high:.2f}] "
            f"buggy=[{buggy_result.ci_low:.2f},{buggy_result.ci_high:.2f}]"
        )

    def test_all_five_failures_produce_zero_pass_rate(self):
        """All five bugs ON simultaneously → every suite fails every case."""
        all_buggy = AgentConfig(
            failures=SeededFailureConfig(),  # all ON by default
            max_loop_iterations=2,
        )
        judge = PatternJudgeLLM()

        suites_and_prewarmed = [
            (make_context_loss_suite(n=3),    False),
            (make_race_condition_suite(n=3),  False),
            (make_contradiction_suite(n=3),   False),
            (make_delegation_loop_suite(n=3), False),
            (make_context_bleed_suite(n=3),   True),
        ]

        for suite, needs_prewarm in suites_and_prewarmed:
            executor = StateInspectingExecutor(
                all_buggy,
                _fake_llm(30),
                prewarm_bleed="alpha topic seed" if needs_prewarm else None,
            )
            result = EvalRunner(executor=executor, judge_llm=judge).run(suite)
            assert result.pass_rate == 0.0, (
                f"Expected 0% pass rate for {suite.suite_id} with all bugs ON; "
                f"got {result.pass_rate:.0%}"
            )
