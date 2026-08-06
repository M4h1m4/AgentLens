"""ChromaDB failure library — store and retrieve failure signals by semantic similarity.

The library starts empty and gets smarter with every eval run. After enough
runs, "this failure looks like X from 3 weeks ago, and what fixed it was Y"
becomes a real, grounded output.

For tests: use FailureLibrary.ephemeral() — in-memory ChromaDB, no disk,
no model download. Inject a _FakeEmbeddingFunction to avoid sentence-transformers.

For production: FailureLibrary(persist_dir=".agentlens/failures") — persists
to disk, uses the default sentence-transformers embedding model (all-MiniLM-L6-v2,
~80MB, downloaded on first use).
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from agentlens.rag.failure_signals import FailureSignal


# ── embedding functions ────────────────────────────────────────────────────────

class _FakeEmbeddingFunction:
    """Deterministic fake embeddings for tests — no model download needed.

    Implements the ChromaDB EmbeddingFunction interface including the
    newer embed_documents/embed_query split introduced in ChromaDB 1.x.
    """

    def name(self) -> str:
        return "fake"

    def _embed(self, texts: list[str]) -> list[list[float]]:
        import hashlib
        embeddings = []
        for text in texts:
            raw = hashlib.sha256(text.encode()).digest()  # 32 bytes
            # Repeat to reach 384 dims (ChromaDB default), normalize to [0, 1]
            repeated = (list(raw) * 12)[:384]
            embeddings.append([b / 255.0 for b in repeated])
        return embeddings

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self._embed(input)

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        return self._embed(input)

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return self._embed(input)


def _default_embedding_fn():
    """Use ChromaDB's built-in sentence-transformers embedding (all-MiniLM-L6-v2)."""
    try:
        from chromadb.utils import embedding_functions
        return embedding_functions.DefaultEmbeddingFunction()
    except Exception:
        # Fallback if sentence-transformers not installed
        return _FakeEmbeddingFunction()


# ── metadata serialization ────────────────────────────────────────────────────
# ChromaDB metadata values must be str | int | float | bool — no lists or dicts.

def _signal_to_metadata(signal: FailureSignal) -> dict[str, Any]:
    return {
        "trace_id": signal.trace_id or "",
        "session_id": signal.session_id,
        "timestamp": signal.timestamp.isoformat(),
        "agents_observed": ",".join(signal.agents_observed),
        "tools_used": ",".join(signal.tools_used),
        "loop_counts": str(signal.loop_counts),
        "context_drop_detected": signal.context_drop_detected,
        "delegation_loop_detected": signal.delegation_loop_detected,
        "diagnosis": signal.diagnosis or "",
        "fix_applied": signal.fix_applied or "",
        "pass_rate_delta": signal.pass_rate_delta
            if signal.pass_rate_delta is not None else 0.0,
    }


def _metadata_to_signal(doc: str, meta: dict) -> FailureSignal:
    agents_raw = meta.get("agents_observed", "")
    tools_raw = meta.get("tools_used", "")
    return FailureSignal(
        trace_id=meta.get("trace_id", ""),
        session_id=meta.get("session_id", "default"),
        agents_observed=[a for a in agents_raw.split(",") if a],
        tools_used=[t for t in tools_raw.split(",") if t],
        context_drop_detected=bool(meta.get("context_drop_detected", False)),
        delegation_loop_detected=bool(meta.get("delegation_loop_detected", False)),
        summary=doc,
        diagnosis=meta.get("diagnosis") or None,
        fix_applied=meta.get("fix_applied") or None,
        pass_rate_delta=float(meta["pass_rate_delta"])
            if meta.get("pass_rate_delta") else None,
    )


# ── library ───────────────────────────────────────────────────────────────────

class FailureLibrary:
    """Store and retrieve failure signals by semantic similarity.

    Usage (production)::

        lib = FailureLibrary()          # persists to .agentlens/failures/
        lib.store(signal)               # after every eval run
        similar = lib.retrieve_similar(new_signal, k=5)  # during diagnosis

    Usage (tests)::

        lib = FailureLibrary.ephemeral(embedding_fn=_FakeEmbeddingFunction())
    """

    _COLLECTION = "agentlens_failures"

    def __init__(
        self,
        persist_dir: str | Path = ".agentlens/failures",
        embedding_fn: Any = None,
    ) -> None:
        import chromadb
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=self._COLLECTION,
            embedding_function=embedding_fn or _default_embedding_fn(),
        )

    @classmethod
    def ephemeral(cls, embedding_fn: Any = None) -> "FailureLibrary":
        """Create an in-memory library for tests — nothing written to disk.

        Each call produces a fully isolated library: ChromaDB EphemeralClient
        instances share a process-level singleton, so we use a unique collection
        name per call to guarantee test isolation.
        """
        import chromadb
        obj = cls.__new__(cls)
        obj._client = chromadb.EphemeralClient()
        # Unique collection name prevents cross-test contamination
        collection_name = f"{cls._COLLECTION}_{uuid.uuid4().hex}"
        obj._collection = obj._client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_fn or _FakeEmbeddingFunction(),
        )
        return obj

    # ── write ────────────────────────────────────────────────────────────────

    def store(self, signal: FailureSignal) -> None:
        """Embed and store a failure signal. Idempotent on trace_id."""
        doc_id = signal.trace_id or str(uuid.uuid4())
        self._collection.upsert(
            ids=[doc_id],
            documents=[signal.summary],
            metadatas=[_signal_to_metadata(signal)],
        )

    def update_diagnosis(
        self,
        trace_id: str,
        diagnosis: str,
        fix_applied: str | None = None,
        pass_rate_delta: float | None = None,
    ) -> None:
        """Attach diagnosis results to a stored signal after the Diagnosis Agent runs."""
        existing = self._collection.get(
            ids=[trace_id], include=["documents", "metadatas"]
        )
        if not existing["ids"]:
            return
        meta = existing["metadatas"][0]
        meta["diagnosis"] = diagnosis
        if fix_applied is not None:
            meta["fix_applied"] = fix_applied
        if pass_rate_delta is not None:
            meta["pass_rate_delta"] = pass_rate_delta
        self._collection.upsert(
            ids=[trace_id],
            documents=existing["documents"],
            metadatas=[meta],
        )

    # ── read ─────────────────────────────────────────────────────────────────

    def retrieve_similar(
        self, signal: FailureSignal, k: int = 5
    ) -> list[FailureSignal]:
        """Return up to k most semantically similar past failures.

        Returns an empty list when the library has fewer than 2 entries
        (can't retrieve meaningfully similar results from a single-entry library).
        """
        total = self.count()
        if total < 1:
            return []
        n = min(k, total)
        results = self._collection.query(
            query_texts=[signal.summary],
            n_results=n,
            include=["documents", "metadatas"],
        )
        signals = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        for doc, meta in zip(docs, metas):
            signals.append(_metadata_to_signal(doc, meta))
        return signals

    def retrieve_by_type(
        self,
        context_drop: bool | None = None,
        delegation_loop: bool | None = None,
        k: int = 10,
    ) -> list[FailureSignal]:
        """Filter signals by failure type — useful for building diagnosis context."""
        where: dict = {}
        if context_drop is not None:
            where["context_drop_detected"] = context_drop
        if delegation_loop is not None:
            where["delegation_loop_detected"] = delegation_loop

        kwargs: dict = {"include": ["documents", "metadatas"]}
        if where:
            kwargs["where"] = where

        results = self._collection.get(**kwargs)
        docs = results.get("documents", [])
        metas = results.get("metadatas", [])
        signals = [_metadata_to_signal(d, m) for d, m in zip(docs, metas)]
        return signals[:k]

    def count(self) -> int:
        """Return total number of stored failure signals."""
        return self._collection.count()
