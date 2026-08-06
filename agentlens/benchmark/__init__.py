"""AgentLens benchmark — Benchmark Gen Agent, suite schema, and store.

Public API::

    from agentlens.benchmark import (
        BenchmarkGenAgent,
        BenchmarkSuite,
        BenchmarkCase,
        BenchmarkStore,
    )

    # Generate a suite from an observed failure signal
    agent = BenchmarkGenAgent()
    suite = agent.generate(signal)

    # Persist it
    store = BenchmarkStore()
    store.save(suite)

    # Retrieve later
    suite = store.load_latest(trace_id)
"""

from agentlens.benchmark.gen_agent import BenchmarkGenAgent, _GenerationBatch, _Case
from agentlens.benchmark.schema import BenchmarkCase, BenchmarkSuite
from agentlens.benchmark.store import BenchmarkStore

__all__ = [
    "BenchmarkGenAgent",
    "BenchmarkSuite",
    "BenchmarkCase",
    "BenchmarkStore",
    "_GenerationBatch",
    "_Case",
]
