"""Agent 1 — Search agent.

Takes a research query, calls the web search tool, and returns a list of
sources. Its `search_source_limit` (Agent 1's configurable source limit)
is deliberately set above Agent 2's cap so the downstream context-loss
failure can manifest.

The node accepts `config` so LangGraph propagates the AgentLens callbacks into
the tool call, letting the adapter capture the tool invocation automatically.
"""

from __future__ import annotations

from typing import Callable

from langchain_core.runnables import RunnableConfig

from reference_agent.config import AgentConfig
from reference_agent.state import AgentState
from reference_agent.tools.search_tool import web_search


def make_search_agent(agent_config: AgentConfig) -> Callable[..., AgentState]:
    def search_agent(state: AgentState, config: RunnableConfig) -> AgentState:
        query = state["query"]
        sources = web_search.invoke(
            {"query": query, "limit": agent_config.search_source_limit},
            config=config,
        )
        return {"sources": sources}

    return search_agent
