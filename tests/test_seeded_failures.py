"""Ground-truth harness for the five seeded failures.

Each test asserts the failure manifests when its flag is ON and disappears
when OFF. This is the mini-harness the Day 7 "5/5 detection" milestone builds
on, and it lets later fix variants prove they actually resolved a failure.
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
