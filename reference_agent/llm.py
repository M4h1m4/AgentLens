"""LLM plumbing for the reference agent.

Option B (see DESIGN_DECISIONS.md D2.1): the reference agent uses a
framework-standard LangChain chat model so the AgentLens adapter can
auto-capture LLM calls and token usage the way it would on a real system.
Tests inject a deterministic fake chat model (see tests), so runs stay cheap
and reproducible. The seeded failures are structural and do not depend on the
model.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel


def default_llm(model: str = "claude-sonnet-4-6") -> BaseChatModel:
    """Real chat model for production runs. Tests pass their own fake model."""
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(model=model, max_tokens=1024)
