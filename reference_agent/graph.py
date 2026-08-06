"""LangGraph wiring for the reference agent: search -> synthesis -> writer.

The conditional edge out of synthesis hosts seeded failure 4 (delegation
loop): under a trigger query type, synthesis routes back to search without a
correct termination condition. A hard safety cap guarantees the graph still
halts — the eval detects the bug via the elevated loop count, not a hang.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from reference_agent.agents.search import make_search_agent
from reference_agent.agents.synthesis import make_synthesis_agent
from reference_agent.agents.writer import make_writer_agent
from langchain_core.language_models import BaseChatModel

from reference_agent.config import AgentConfig
from reference_agent.state import AgentState


def build_graph(config: AgentConfig, llm: BaseChatModel):
    failures = config.failures

    def route_after_synthesis(state: AgentState) -> str:
        query = state.get("query", "")
        is_trigger = config.loop_trigger_keyword in query.lower()
        loop_count = state.get("loop_count", 0)

        # Failure 4 — missing terminator: on a trigger query the buggy path
        # keeps looping back to search until only the safety cap stops it.
        if (
            failures.delegation_loop
            and is_trigger
            and loop_count < config.max_loop_iterations
        ):
            return "search"
        return "writer"

    graph = StateGraph(AgentState)
    graph.add_node("search", make_search_agent(config))
    graph.add_node("synthesis", make_synthesis_agent(config))
    graph.add_node("writer", make_writer_agent(config, llm))

    graph.add_edge(START, "search")
    graph.add_edge("search", "synthesis")
    graph.add_conditional_edges(
        "synthesis",
        route_after_synthesis,
        {"search": "search", "writer": "writer"},
    )
    graph.add_edge("writer", END)

    return graph.compile()
