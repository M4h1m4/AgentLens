"""Trace reading backends for AgentLens.

Writing is handled automatically by the OTel exporter (configured in otel.py).
This module provides readers that query stored traces back for analysis.

InMemorySpanExporter  — re-exported for use in tests (no backend needed).
PhoenixTraceReader    — reads from a running Phoenix instance via its Python API.
"""

from __future__ import annotations

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


class PhoenixTraceReader:
    """Query traces stored in a local Phoenix instance.

    Phoenix must be running: ``pip install arize-phoenix && phoenix serve``
    """

    def __init__(self, endpoint: str = "http://localhost:6006") -> None:
        self.endpoint = endpoint

    def _client(self):
        import phoenix as px  # lazy — not needed for tests
        return px.Client(endpoint=self.endpoint)

    def get_spans(self, trace_id: str):
        """Return a pandas DataFrame of all spans for a given trace_id."""
        df = self._client().get_spans_dataframe()
        if df is None or df.empty:
            return df
        col = "context.trace_id"
        if col not in df.columns:
            return df.iloc[0:0]  # empty
        return df[df[col] == trace_id].reset_index(drop=True)

    def get_handoffs(self, trace_id: str) -> list[dict]:
        """Return dicts of {source, target} for every HANDOFF span in the trace."""
        df = self.get_spans(trace_id)
        if df is None or df.empty:
            return []
        kind_col = "attributes.openinference.span.kind"
        if kind_col not in df.columns:
            return []
        hdf = df[df[kind_col] == "HANDOFF"]
        result = []
        for _, row in hdf.iterrows():
            result.append({
                "source": row.get("attributes.handoff.source"),
                "target": row.get("attributes.handoff.target"),
            })
        return result

    def get_agents(self, trace_id: str) -> list[str]:
        """Return node names observed in the trace, in order."""
        handoffs = self.get_handoffs(trace_id)
        seen: list[str] = []
        for h in handoffs:
            for name in (h["source"], h["target"]):
                if name and name not in seen:
                    seen.append(str(name))
        return seen


__all__ = ["InMemorySpanExporter", "PhoenixTraceReader"]
