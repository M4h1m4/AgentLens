"""AgentLens RAG — failure library and MAST taxonomy.

The failure library stores embedded failure signals from every eval run
and retrieves semantically similar past failures during diagnosis.

Usage::

    from agentlens.rag import (
        extract_signals, FailureSignal,
        FailureLibrary,
        as_mast_prompt, mast_categories,
    )

    # After a run: extract signals and store them
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    spans = list(exporter.get_finished_spans())
    signal = extract_signals(spans, trace_id=trace_id, session_id="dev")
    lib = FailureLibrary()
    lib.store(signal)

    # During diagnosis: retrieve similar past failures
    similar = lib.retrieve_similar(signal, k=5)
    context = "\\n\\n".join(s.summary for s in similar)

    # In any system prompt that needs failure classification
    prompt = as_mast_prompt()
"""

from agentlens.rag.failure_signals import FailureSignal, extract_signals
from agentlens.rag.library import FailureLibrary, _FakeEmbeddingFunction
from agentlens.rag.mast import (
    CATEGORIES as mast_categories,
    as_prompt_section as as_mast_prompt,
    category_names as mast_category_names,
    describe as mast_describe,
)

__all__ = [
    "FailureSignal",
    "extract_signals",
    "FailureLibrary",
    "_FakeEmbeddingFunction",
    "mast_categories",
    "as_mast_prompt",
    "mast_category_names",
    "mast_describe",
]
