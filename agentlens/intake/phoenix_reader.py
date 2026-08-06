"""Phoenix intake reader — pulls existing traces from a running Phoenix instance
for use during the intake conversation.

This is the intake-side companion to the adapter's PhoenixTraceReader. While
the adapter's reader is used AFTER a run to query specific traces by ID, this
reader is used DURING intake to check whether any traces already exist and
pull them for immediate profile reconciliation.

If Phoenix already has 10 runs from the developer's agent, the profile is
mostly complete before the first clarifying question is asked.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# ── span proxy ────────────────────────────────────────────────────────────────

@dataclass
class _SpanProxy:
    """Minimal span-like object the reconciler can consume.

    The reconciler only reads span.name and span.attributes — this proxy
    satisfies that interface using data from a Phoenix DataFrame row.
    """
    name: str
    attributes: dict[str, Any]


# ── reader ────────────────────────────────────────────────────────────────────

class IntakePhoenixReader:
    """Pull existing traces from Phoenix for intake profile pre-population.

    Usage::

        reader = IntakePhoenixReader("http://localhost:6006")
        if reader.is_available():
            spans = reader.get_recent_spans()
            # pass to reconcile(profile, spans)
    """

    def __init__(self, endpoint: str = "http://localhost:6006") -> None:
        self.endpoint = endpoint.rstrip("/")

    def is_available(self) -> bool:
        """Return True if Phoenix is reachable at the configured endpoint."""
        try:
            import urllib.request
            with urllib.request.urlopen(
                f"{self.endpoint}/healthz", timeout=2
            ) as resp:
                return resp.status == 200
        except Exception:
            return False

    def get_recent_spans(self, max_traces: int = 10) -> list[_SpanProxy]:
        """Return span proxies from the most recent traces in Phoenix.

        Returns an empty list if Phoenix is unavailable or has no data.
        The returned objects satisfy the reconciler's interface (name + attributes).

        Args:
            max_traces: How many recent traces to pull. More traces = more
                        complete node/tool discovery, but slower.
        """
        try:
            import phoenix as px
        except ImportError:
            return []

        try:
            client = px.Client(endpoint=self.endpoint)
            df = client.get_spans_dataframe()
        except Exception:
            return []

        if df is None or df.empty:
            return []

        return _dataframe_to_proxies(df, max_traces)

    def get_trace_count(self) -> int:
        """Return the number of distinct traces stored in Phoenix. 0 if unavailable."""
        try:
            import phoenix as px
            client = px.Client(endpoint=self.endpoint)
            df = client.get_spans_dataframe()
            if df is None or df.empty:
                return 0
            col = "context.trace_id"
            if col not in df.columns:
                return 0
            return df[col].nunique()
        except Exception:
            return 0


# ── helpers ───────────────────────────────────────────────────────────────────

def _dataframe_to_proxies(df, max_traces: int) -> list[_SpanProxy]:
    """Convert a Phoenix spans DataFrame into span proxies for the reconciler."""
    proxies: list[_SpanProxy] = []

    # Take spans from the most recent N traces only
    trace_id_col = "context.trace_id"
    if trace_id_col in df.columns:
        recent_traces = (
            df.groupby(trace_id_col)["start_time"].max()
            .nlargest(max_traces)
            .index.tolist()
            if "start_time" in df.columns
            else df[trace_id_col].unique().tolist()[:max_traces]
        )
        df = df[df[trace_id_col].isin(recent_traces)]

    for _, row in df.iterrows():
        name = str(row.get("name", ""))
        attrs = _extract_attributes(row)
        proxies.append(_SpanProxy(name=name, attributes=attrs))

    return proxies


def _extract_attributes(row) -> dict[str, Any]:
    """Pull relevant attributes from a DataFrame row into a plain dict."""
    attrs: dict[str, Any] = {}

    # Standard AgentLens attributes
    for col in (
        "attributes.openinference.span.kind",
        "attributes.handoff.source",
        "attributes.handoff.target",
        "attributes.agentlens.framework",
        "attributes.agentlens.node",
        "attributes.agentlens.produced",
        "attributes.tool.name",
    ):
        val = row.get(col)
        if val is not None and str(val) not in ("nan", "None", ""):
            # Strip the "attributes." prefix so reconciler reads the same keys
            key = col.removeprefix("attributes.")
            attrs[key] = val

    return attrs
