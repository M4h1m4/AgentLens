"""AgentLens intake — NL intake agent, codebase inspector, profile schema,
Phoenix reader, and trace reconciler.

Public API::

    from agentlens.intake import (
        IntakeAgent, AgentProfile, ProfileStore, reconcile,
        inspect_codebase, IntakePhoenixReader,
    )

    # Full intake with all three evidence sources
    agent = IntakeAgent()
    question = agent.start(
        "My research agent loses context on long queries",
        codebase_path="/path/to/project",
        phoenix_endpoint="http://localhost:6006",
    )
    while question is not None:
        question = agent.respond(input(question + " "))
    profile = agent.get_profile()

    # Save and version the profile
    store = ProfileStore()
    store.save(profile)

    # After a run: reconcile with fresh OTel spans
    from agentlens.adapter import setup_otel, instrument_langchain, traced_invoke
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    exporter = InMemorySpanExporter()
    setup_otel(exporter=exporter)
    instrument_langchain()
    trace_id, _ = traced_invoke(app, initial)
    updated = reconcile(profile, list(exporter.get_finished_spans()))
    store.save(updated)
"""

from agentlens.intake.agent import IntakeAgent
from agentlens.intake.inspector import InspectionResult, inspect_codebase
from agentlens.intake.phoenix_reader import IntakePhoenixReader
from agentlens.intake.profile import AgentProfile, ConfidentValue
from agentlens.intake.reconciler import reconcile
from agentlens.intake.store import ProfileStore

__all__ = [
    "IntakeAgent",
    "AgentProfile",
    "ConfidentValue",
    "reconcile",
    "ProfileStore",
    "inspect_codebase",
    "InspectionResult",
    "IntakePhoenixReader",
]
