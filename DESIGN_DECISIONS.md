# AgentLens — Design Decisions

A running log of the architectural decisions we make while building AgentLens,
with the reasoning behind each so we can revisit them later. Newest sections
appended as the project grows.

---

## Context: two separate things

- **AgentLens** = the product. Autonomous eval infrastructure for multi-agent
  systems. Lives in the `agentlens/` package.
- **Reference agent** = the *test subject*. A deliberately buggy 3-agent
  research system (`reference_agent/`) whose known failures let us prove
  AgentLens actually works. It is **not** the product.

**Guiding principle (applies to every decision below):** prefer designs that
generalize to arbitrary real-world agent systems over shortcuts that only serve
the demo. AgentLens's value is being framework-agnostic and working on systems
where nobody knows where the bugs are.

---

## Day 1 — the reference agent

**D1.1 — Bugs are planted in orchestration CODE, not in prompts.**
The five seeded failures live in how the agents hand off to each other, not in
LLM prompt wording. Rationale: prompt bugs are *model-output* failures
(non-deterministic, prompt-tunable); AgentLens's thesis is detecting
*coordination* failures, which are structural and fire regardless of model
quality. This is why the tests catch every bug even with a fake LLM.

**D1.2 — Failures are deterministic and toggleable.**
Each failure is gated by a boolean flag in `SeededFailureConfig`, default ON
(buggy). Flag OFF restores correct behavior. Rationale: the failures are
*ground truth*; they must reproduce identically every run, and later components
(the Improvement Agent) express and test fixes by flipping flags.

**D1.3 — The delegation loop has a hard safety cap.**
Failure 4 (missing loop terminator) is bounded by `max_loop_iterations` so dev
runs and tests halt. The bug is the *missing terminator*, detected via an
elevated loop count — not an actual infinite loop.

**D1.4 — Observability fields are the "answer key," not surfaced output.**
`dropped_sources`, `loop_count`, `writer_input` exist in state only so our
tests can assert ground truth. They are NOT shown to the end user, and
AgentLens is not allowed to read them. AgentLens must detect failures from the
observed *behavioral trace*, not from these fields. (The state is the answer
key; the trace is the exam.)

**D1.5 — Pluggable LLM + fixture search for testability.**
Real runs use a real model/search; tests use deterministic stand-ins. Chosen so
tests need no API key and give identical results, and so the controllable
inputs (conflicting sources, >cap source counts) needed for ground truth are
reproducible.

---

## Day 2 — the universal adapter

**D2.1 — Option B: make the reference agent framework-native.**
We rebuild the reference agent's *plumbing* (its LLM wrapper and search tool) to
use framework-standard parts (LangChain chat model + LangChain tool). The five
seeded bugs are untouched — they live in orchestration, not plumbing.

- *Why:* In a real customer system, AgentLens cannot rewrite the target agent.
  What makes the adapter universal is hooking the framework's own
  instrumentation (which real systems already use) + generic hooks for custom
  code. Our Day-1 agent was unusually hand-built (custom LLM, plain-function
  tool), so the framework couldn't auto-report its tool/token activity — making
  it a poor stand-in for a real system. Option B makes it realistic AND forces
  us to actually build and test the rich auto-capture path real customers rely
  on. (Rejected Option A, which captured only agent-to-agent handoffs — enough
  for the demo's known bugs, but it would never exercise the real-world capture
  path.)

**D2.2 — Interception mechanism: LangGraph streaming + callback handler.**
- Node boundaries + handoffs come from `app.stream(stream_mode="updates")` —
  100% reliable node identification from the streamed `{node: update}` keys.
- Tool calls and token counts come from a LangChain `BaseCallbackHandler`,
  attributed to the right agent via `metadata["langgraph_node"]`.
- *Why:* streaming gives bulletproof node/handoff detection without fragile
  callback filtering; callbacks give reliable tool/LLM capture. Together, zero
  changes to the agent's business logic. Callbacks propagate automatically to
  child tool/LLM calls via the node config.
- *Known approximation:* handoff events may interleave slightly with a node's
  own tool/LLM events. We attribute by `source_agent`, not exact position, so
  analysis is unaffected.

**D2.3 — Normalized event schema: custom, coordination-first, OTel-mappable.**
We use our own lean event format rather than adopting an industry standard
wholesale, but every field maps cleanly onto OpenTelemetry / OpenInference.

- *Standards considered:* OpenTelemetry (+ GenAI semantic conventions),
  OpenInference (Arize), LangSmith run tree, W3C Trace Context.
- *Why not adopt one wholesale:* (1) They model a run as a **call tree**
  (parent/child spans). AgentLens's failures are **lateral edges between
  sibling agents** (context lost across a handoff, delegation loops). No
  standard makes a **handoff a first-class event** — the exact signal we need.
  (2) Full OTel means a collector + exporters + heavier SDK: real infra
  overhead for a 14-day build.
- *Why not pure custom:* real production agents increasingly already emit OTel;
  a snowflake format could never ingest what a customer already produces.
- *Decision:* thin coordination layer (first-class `source_agent`,
  `target_agent`, `HANDOFF`) on top of fields that map 1:1 to OTel, so we get
  simplicity now and interoperability (ingest customer OTel traces) later.

  | Our field            | OTel / OpenInference equivalent        |
  |----------------------|----------------------------------------|
  | `trace_id`           | `trace_id`                             |
  | `event_id`           | `span_id`                              |
  | `ts` + `duration_ms` | span start / end time                  |
  | `event_type`         | OpenInference span kind (LLM/TOOL/AGENT)|
  | `token_count`        | `gen_ai.usage.*`                       |
  | `content`            | span attributes                        |
  | `source_agent` / `target_agent` / `HANDOFF` | **our addition — no standard has this** |

**D2.4 — Pluggable trace storage (Postgres for real, in-memory for tests).**
`TraceStore` is an interface with `PostgresTraceStore` (the "traces flowing
into Postgres" deliverable, run via `docker-compose`) and `InMemoryTraceStore`
(so tests need no database). Same pattern as D1.5, which we validated on Day 1.

**D2.5 — One-line entry point.**
`traced_invoke(app, initial, store=...)` runs the graph, captures the full
normalized trace, and persists it — the "one-line decorator on the agent entry
point" the spec calls for.

---

## Day 3 — the NL intake agent

**D3.1 — Intake is a plain Python class, not a LangGraph graph.**
The multi-turn intake conversation is driven by external human input between
turns. LangGraph's state machine fits autonomous agents that run to completion
in one `invoke()` call. A human-in-the-loop conversation fits a stateful class
where each call to `respond()` runs one LLM pass. Using LangGraph here would
add interrupt/resume complexity for no benefit. LangGraph is used for the
autonomous agents in Days 5–9.

**D3.2 — LLM re-extracts from full conversation each turn, no mutable draft.**
Each `_step()` call sends the complete message history to the LLM and gets back
a fresh `_ExtractionResult`. We do not accumulate a partial draft between turns.
Rationale: simpler, avoids stale-state bugs, and the LLM with full context
produces better extractions than merging incremental deltas.

**D3.3 — MAX_TURNS = 4 before forcing a draft.**
An incomplete profile is always acceptable because the trace reconciler fills
in unknowns from observed reality. Asking more than 4 questions creates
friction without meaningfully improving the profile. When MAX_TURNS is hit,
the agent produces whatever profile it can from the conversation so far.

**D3.4 — Profile fields carry explicit confidence levels and sources.**
Every `ConfidentValue` has `confidence` ("high"/"medium"/"low"/"unknown") and
`source` ("user_stated"/"trace_observed"/"inferred"/"code_inspected"). This
lets downstream agents (benchmark gen, diagnosis) weight evidence correctly and
surfaces exactly what came from the user vs. what was confirmed by observation.

**D3.5 — Trace is always the ground truth; user description is a prior.**
`reconcile()` promotes user-stated fields to trace-observed confidence when
the trace confirms them, logs discrepancies when it contradicts them, and fills
in unknowns from the trace. After one trace run, the profile reflects observed
reality regardless of how incomplete the initial description was.

**D3.6 — Profile storage is local JSON, not Postgres.**
The `.agentlens/` directory in the user's project folder stores versioned JSON
files. Rationale: the profile store must work for any user with no database
running (CLI use case). Postgres can be added as an optional backend later via
the same interface pattern used in `TraceStore`.

---

## Architecture revision — supersedes parts of Day 2

*Decided: 2026-08-03. D2.2, D2.3, D2.4, D2.5 are superseded by the decisions
below. D2.1 (make the reference agent framework-native) remains valid and
unchanged. Day 1 decisions are all unchanged.*

**Why we're revising:** the Day 2 adapter was built with a closed custom schema
(only AgentLens can produce or consume it) and a private Postgres store. The
right architecture for a real observability tool is one where traces flow
through open standards and developers who already have observability
infrastructure get AgentLens without re-instrumenting. These decisions align
the build with that goal.

---

## Revised adapter architecture

**D-R1 — Trace format: OTel spans with custom handoff attributes.**
*(Supersedes D2.3)*

Adopt OpenTelemetry as the wire format for all traces. The custom `TraceEvent`
dataclass and `EventType` enum are replaced with OTel spans. Every field maps
to OTel GenAI semantic conventions, with one addition: HANDOFF spans carry
`handoff.source` and `handoff.target` as custom span attributes. This is the
only thing no existing standard models — everything else is standard OTel.

- *Why not stay custom:* the custom schema is a closed system. Real-world
  developers already emit OTel. Asking them to instrument twice is a blocker.
- *Why not OpenInference wholesale:* OpenInference (Arize) is OTel + LLM
  attributes and is the closest existing standard. It still models runs as
  call trees (parent/child spans). HANDOFF between sibling agents — the signal
  AgentLens is built around — is not a first-class concept anywhere. Our
  custom span kind `HANDOFF` with `handoff.source` / `handoff.target` is the
  minimal addition needed on top of standard OTel.

| Old field | OTel equivalent |
|---|---|
| `trace_id` | `trace_id` (W3C TraceContext) |
| `event_id` | `span_id` |
| `ts` + `duration_ms` | span `start_time` / `end_time` |
| `event_type` | OTel span kind + `openinference.span.kind` |
| `token_count` | `gen_ai.usage.input_tokens` + `gen_ai.usage.output_tokens` |
| `content` | span attributes |
| `source_agent` / `target_agent` / `HANDOFF` | custom `handoff.source`, `handoff.target` on a `HANDOFF`-kind span — our only addition |

Files that change: `agentlens/adapter/events.py` (replaced), `agentlens/adapter/tracer.py` (replaced), `agentlens/adapter/schema.sql` (deleted).

**D-R2 — Instrumentation: LangchainInstrumentor + thin HANDOFF layer.**
*(Supersedes D2.2, D2.5)*

Replace `AgentLensTracer` (custom `BaseCallbackHandler`) and `traced_invoke()`
with `LangchainInstrumentor` for auto-instrumentation, plus a thin AgentLens
layer that emits HANDOFF spans on top.

`LangchainInstrumentor` captures automatically, with zero code change to the
developer's agent:
- Every LLM call — model, prompt, completion, token counts
- Every tool call — tool name, input, output, duration
- Every LangGraph node execution — node name, duration, state produced

AgentLens adds only:
- HANDOFF spans — detected from `app.stream(stream_mode="updates")` exactly
  as today, but emitted as OTel spans with `handoff.source` and
  `handoff.target` attributes

Developer experience:
```python
# If they already have OTel: zero changes — existing traces flow to Phoenix
# If they have no OTel: one line at app startup (CLI handles this automatically)
from opentelemetry.instrumentation.langchain import LangchainInstrumentor
LangchainInstrumentor().instrument()
```

Files that change: `agentlens/adapter/tracer.py` (replaced with OTel span
emitter for HANDOFF only), `agentlens/adapter/langgraph_adapter.py`
(simplified — LLM/tool capture delegated to `LangchainInstrumentor`).

**D-R3 — Local trace backend: Phoenix (Arize).**
*(Supersedes D2.4)*

Replace `PostgresTraceStore` with Phoenix as the local observability backend.
`InMemoryTraceStore` is kept for unit tests — no database required for tests.

- *Why not Langtrace:* Langtrace is a full observability product with a good
  instrumentation SDK (`langtrace-python-sdk`). However, it has no stable
  programmatic read API. AgentLens needs to query traces analytically across
  runs (e.g. "find all HANDOFF spans where token count dropped between agents")
  — Langtrace does not expose this. Langtrace can optionally be used as the
  instrumentation layer in place of `LangchainInstrumentor`, but not as the
  storage backend.
- *Why not ClickHouse:* ClickHouse is a database, not an observability
  platform. It requires an OTel Collector in front and a UI layer (Signoz,
  Uptrace) on top — a 6-container Docker Compose stack. Too heavy for a local
  dev tool aimed at zero-friction setup.
- *Why Phoenix:* `pip install arize-phoenix`, starts with `phoenix serve`, no
  Docker required. Provides `px.Client().get_spans_dataframe()` — a Python
  query API returning a pandas DataFrame that AgentLens can query directly.
  OTel-native, LLM-aware UI at `localhost:6006`, open source.
- *Production backend:* deferred. Not in scope for current build.

Files that change: `agentlens/adapter/store.py` (replace `PostgresTraceStore`
with `PhoenixTraceReader`), `docker-compose.yml` (remove Postgres service),
`agentlens/adapter/schema.sql` (deleted).

**D-R4 — Entry point: CLI tool.**

CLI is the primary entry point. Zero code changes required in the developer's
codebase. The spec's FastAPI web UI is replaced by a local CLI that imports
and wraps the developer's agent from outside.

```bash
pip install agentlens
agentlens observe --entry graph:build_graph --query "..."
agentlens diagnose --trace-id abc-123
agentlens run    # observe + diagnose in one command
```

- *Why not hosted web UI:* requires sending agent code or API keys to a
  server. Hard blocker for most companies on privacy and security grounds.
  Every serious observability tool is local-first for this reason.
- *Voice briefing:* CLI spins up a local FastAPI server on `localhost:7432`,
  opens a browser tab automatically after each eval run. Voice plays in
  browser, follow-up Q&A works as a chat interface. Server shuts down when
  the tab is closed. No hosted infrastructure.

Files to build: `agentlens/cli/` (Click or Typer commands), `agentlens/server/`
(minimal local FastAPI for voice briefing tab). Neither exists yet.

---

## Day 9 — Improvement Agent

**D9.1 — CodeLocator: span-driven file selection (supersedes the "caller passes files" approach)**

The Improvement Agent needs source files to generate a fix. Two naive approaches were rejected:

- *Option A (caller passes files)*: requires the caller to know which file contains the bug. In a real agentic system, if the developer knew which file to pass in, they'd probably fix it themselves. This defeats AgentLens's core value proposition.
- *Option B (agent browses the entire codebase)*: large codebases are expensive to analyze, and sending an entire proprietary codebase to an external LLM API is a policy blocker for most enterprise customers.

The right approach: **CodeLocator** — a local, non-LLM component that uses the diagnosis text + Python code analysis to derive the minimal relevant file set without browsing the whole codebase.

Algorithm:
1. Scan agent_dir for LangGraph `add_node("name", ...)` calls → build {node_name: file_path} map.
2. Filter to node names that appear in the diagnosis text (root_cause + evidence + suggested_fix).
3. For matched files, follow one level of local imports (AST-parsed, resolved to actual .py files in the project).
4. Return only the resulting file set — typically 1–4 files.

This runs entirely locally before anything is sent to the LLM. The developer can review exactly which files will be sent.

**D9.2 — Three-phase validation before promotion**

Sending a small file set means the fix might inadvertently break files that depend on the changed code. Three phases:

1. **Wilson CI on failing cases only** (fast): run the variant against only the failing benchmark cases. Non-overlapping 95% CIs between baseline (0/N) and variant → improvement is statistically real. If CIs overlap → next iteration.
2. **Full suite regression check**: run the variant against ALL benchmark cases. If any previously-passing case now fails → variant is rejected (causes regressions).
3. Only if both pass → promote.

**D9.3 — Iteration feedback loop (max 5 iterations)**

If Phase 1 or Phase 2 fails, the LLM receives concrete feedback: the previous hypothesis, which files were changed, the new pass rate, and the still-failing criteria with reasons. The next iteration can take a different approach. Budget capped at $2.00 by default.

**D9.4 — Return, don't apply**

The ImprovementAgent returns variant_files (the modified code) and an ImprovementResult. It does NOT write to disk. The CLI layer (Day 11+) shows the developer a diff and asks for confirmation before applying. The agent's job is to find a fix, not to apply it.

**D9.5 — ExecutorConfig decouples variant execution from agent construction**

Variant runs use SandboxExecutor (E2B) rather than InProcessExecutor. Unreviewed LLM-generated code must run in an isolated sandbox. The caller provides an ExecutorConfig (agent_dir, import_lines, build_app_expr, etc.) so the ImprovementAgent can spin up a fresh SandboxExecutor for each variant without knowing how to construct the agent beyond those config values.

---

## Day 8+9 — Failure Library and Diagnosis Accuracy

**D8.1 — Overwrite weak diagnosis with proven hypothesis after promotion.**

When the Improvement Agent promotes a fix, the original Diagnosis Agent hypothesis is
replaced in ChromaDB with the Improvement Agent's winning hypothesis — not stored
alongside it.

Rationale: the Diagnosis Agent produces a hypothesis before the fix is proven. The
Improvement Agent empirically validates what actually caused the failure. Storing both
creates a contradiction in the failure library: the weak original hypothesis and the
proven fix coexist, and future Diagnosis Agents may reason from the weak hypothesis
rather than the ground truth.

The correct ground truth for a fixed failure is: "what the Improvement Agent's winning
hypothesis said caused it." The pipeline orchestrator (Day 11 CLI) is responsible for
calling `library.update_diagnosis()` with the Improvement Agent's winning hypothesis
as the new diagnosis string, not the Diagnosis Agent's original weak hypothesis.

Implementation: in the Day 11 pipeline orchestrator, after `improvement_result.promoted
== True`:

```python
library.update_diagnosis(
    trace_id=trace_id,
    diagnosis=improvement_result.hypothesis,   # proven ground truth, not weak hypothesis
    fix_applied=improvement_result.hypothesis,
    pass_rate_delta=improvement_result.variant_eval.pass_rate
                    - improvement_result.baseline_eval.pass_rate,
)
```

**D8.2 — Weight retrieved failures by pass_rate_delta, exclude unresolved ones.**

When the Diagnosis Agent retrieves similar past failures from ChromaDB, it must not
treat all retrieved signals equally. Failures where no fix was found (pass_rate_delta
= 0 or None) are noise — they show a failure pattern but provide no actionable evidence.

The pipeline sorts retrieved signals by pass_rate_delta descending and passes only the
top signals where a fix was confirmed (pass_rate_delta > 0) to the Diagnosis Agent's
context. Signals with no proven fix are excluded or shown last.

Rationale: across many runs, the failure library accumulates entries with contradicting
hypotheses — some where the fix worked, some where it did not. Passing all of them
equally to the LLM creates contradicting context that degrades diagnosis quality.
Weighting by proven outcome ensures the Diagnosis Agent reasons from validated evidence
rather than from a mix of confirmed and unconfirmed hypotheses.

Implementation: in the Diagnosis Agent, after `library.retrieve_similar()`:

```python
similar = library.retrieve_similar(signal, k=5)
# Sort by proven outcome — failures with working fixes ranked first
similar.sort(key=lambda s: s.pass_rate_delta or 0.0, reverse=True)
# Only pass signals where a fix was confirmed
grounded = [s for s in similar if s.pass_rate_delta and s.pass_rate_delta > 0][:3]
```
