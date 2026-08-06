"""Entrypoint for running the reference agent end to end."""

from __future__ import annotations

from typing import Optional

from langchain_core.language_models import BaseChatModel

from reference_agent.agents.writer import reset_bleed_cache
from reference_agent.config import AgentConfig
from reference_agent.graph import build_graph
from reference_agent.llm import default_llm
from reference_agent.state import AgentState

__all__ = ["invoke", "run", "reset_bleed_cache"]


def invoke(
    query: str,
    session_id: str = "default",
    config: Optional[AgentConfig] = None,
    llm: Optional[BaseChatModel] = None,
) -> AgentState:
    """Run the full graph and return the final state (for inspection/evals)."""
    config = config or AgentConfig()
    llm = llm or default_llm()
    app = build_graph(config, llm)
    # Failure 4's loop counter must start fresh per invocation.
    initial: AgentState = {"query": query, "session_id": session_id, "loop_count": 0}
    return app.invoke(initial)


def run(
    query: str,
    session_id: str = "default",
    config: Optional[AgentConfig] = None,
    llm: Optional[BaseChatModel] = None,
) -> str:
    """Run the full graph and return just the final report."""
    return invoke(query, session_id, config, llm)["report"]


if __name__ == "__main__":
    print(run("The state of renewable energy adoption in 2025"))
