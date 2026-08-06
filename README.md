# AgentLens

**Autonomous eval infrastructure for multi-agent systems.**

Most observability tools tell you *what* happened in an agent run. AgentLens tells you *what broke*, *why it broke*, and *what to change* — automatically, after observing a single real execution.

Point AgentLens at your agent system. It instruments it without touching your code, watches one run, detects coordination failures across agent handoffs, generates a benchmark suite targeting exactly those failures, runs the benchmarks, diagnoses what went wrong in plain English, and autonomously tests fixes. All from one command.

---

## The problem AgentLens solves

Multi-agent systems fail silently. A wrong tool call raises an exception — visible immediately. A wrong *handoff* passes bad data to the next agent, which produces a plausible-looking but wrong output. The system never errors. Nobody knows.

Traditional evals don't catch this because they are written by hand, before you know where the bugs are. AgentLens inverts this: observe first, generate the harness from what you saw.

---

## How it works

```
agentlens run --entry my_agent.graph:build_graph --query "climate change impacts"

Observing run...
  search → synthesis → writer  [3 handoffs, 2 tool calls]
  ⚠ Context drop at synthesis→writer: 892 tokens in, 340 out (-62%)
  ⚠ Delegation loop: synthesis ran 2×

Generating benchmarks...  25 cases across 4 failure modes

Running eval suite...
  21/25 passed  (84%)
  Failures: context_loss ×3, delegation_loop ×1

Diagnosis:
  The synthesis agent silently drops sources when the count exceeds its
  internal cap of 3. The writer receives an incomplete view and produces
  reports missing key evidence. This pattern appeared in 4 of your last
  6 failing runs.

Opening briefing at localhost:7432 ...
```

Zero changes to your codebase. AgentLens imports your agent from outside, wraps OTel instrumentation around it in memory, and runs it.

---

## Architecture

```
Your agent (LangGraph / any OTel-emitting framework)
    │
    ├── LangchainInstrumentor
    │     auto-captures: every LLM call, tool call, node execution
    │
    ├── AgentLens HANDOFF layer
    │     emits: HANDOFF spans with handoff.source / handoff.target
    │     (the only coordination signal no existing OTel standard captures)
    │
    └── All spans → Phoenix (localhost:6006)
            │
            ├── NL Intake Agent
            │     Profiles your system from a plain English description.
            │     Pre-fills from codebase inspection + existing traces.
            │     Stores an AgentProfile with per-field confidence levels.
            │
            ├── Failure Library (ChromaDB)
            │     Embeds and stores a FailureSignal after every run.
            │     Retrieves semantically similar past failures during diagnosis.
            │
            ├── Benchmark Gen Agent
            │     Reads the FailureSignal. Generates 20-30 test cases targeting
            │     exactly the failure modes observed — each traceable to a span.
            │
            ├── Eval Runner                              ← in progress
            │     Runs each benchmark case against the agent.
            │     Checks expected criteria against OTel spans + output.
            │     Records pass/fail with reasons.
            │
            ├── Diagnosis Agent                          ← in progress
            │     Reads failing traces + top-5 similar past failures from ChromaDB.
            │     Produces a grounded plain English diagnosis paragraph.
            │
            ├── Improvement Agent                        ← in progress
            │     Forms a hypothesis. Writes a code variant. Runs it against
            │     failing cases. Uses Wilson CI to decide if the fix is real.
            │     Promotes if statistically significant. Loops up to 5× if not.
            │
            └── Voice Agent                              ← in progress
                  Spoken briefing after every eval run via ElevenLabs.
                  Conversational Q&A over trace data via Vapi + WebSocket.
                  Served from a local FastAPI tab at localhost:7432.
```

---

## Key design choices

**OTel as the wire format.**
Every span AgentLens produces is standard OpenTelemetry. Developers who already emit OTel get AgentLens with zero additional instrumentation — their existing traces flow straight in. The only custom addition is the `HANDOFF` span with `handoff.source` / `handoff.target` attributes, because no existing standard models lateral agent-to-agent coordination.

**Phoenix as the local backend.**
`pip install arize-phoenix && phoenix serve` — no Docker required. Provides a Python query API (`px.Client().get_spans_dataframe()`) that AgentLens queries analytically, plus a built-in LLM-aware UI at `localhost:6006` the developer can use independently. Chosen over Langtrace (no programmatic read API) and ClickHouse (requires a 6-container stack).

**CLI as the primary entry point.**
`agentlens run --entry my_agent.graph:build_graph --query "..."` — the developer touches nothing in their codebase. AgentLens adds `cwd` to `sys.path`, imports their module dynamically, and wraps it from outside.

**Confidence-weighted profiles.**
Every field in an `AgentProfile` carries a confidence level (`high/medium/low/unknown`) and a source (`user_stated/code_inspected/trace_observed`). Trace-observed evidence always wins over what a developer describes. After one real run, the profile reflects observed reality regardless of how incomplete the initial description was.

**Failure library grows smarter over time.**
Every run produces a `FailureSignal` — a structured + natural language summary of what was observed. These are embedded and stored in ChromaDB. The Diagnosis Agent retrieves the top-5 most semantically similar past failures as grounding context. "This same pattern appeared in 4 of your last 6 runs" is a retrieved fact, not a hallucination.

---

## Components

### Built

**Universal Adapter** (`agentlens/adapter/`)
OTel setup, LangchainInstrumentor auto-instrumentation, HANDOFF span emission from LangGraph streaming, Phoenix backend. The single entry point `traced_invoke()` runs your graph, captures the full trace, and persists it — one line at the agent entry point.

**NL Intake Agent** (`agentlens/intake/`)
Multi-turn conversation that produces an `AgentProfile` from a plain English description. Pre-fills known fields from static codebase inspection and existing Phoenix traces before asking a single question. Reconciles the profile against observed OTel spans after a run — trace always overrides user description.

**Failure Library** (`agentlens/rag/`)
Extracts a `FailureSignal` from OTel spans: agent order, handoff sequence, tool usage, context drop detection (>20% output size reduction across a handoff), delegation loop detection. Embeds and stores signals in ChromaDB. Retrieves by semantic similarity or failure type filter. MAST 8-category taxonomy hardcoded as a prompt string (small enough to fit in context; no RAG needed for the taxonomy itself).

**Benchmark Gen Agent** (`agentlens/benchmark/`)
LangGraph agent that reads a `FailureSignal` and generates a `BenchmarkSuite` of 20–30 test cases. Each case has a concrete query, 2–4 checkable expected criteria, a MAST failure mode label, severity, and a rationale traceable to the observed span. Loops until the minimum case count is reached, sanitizes invalid LLM outputs.

**Reference Agent** (`reference_agent/`)
A deliberately buggy 3-agent research system (search → synthesis → writer) used as the test subject. Five seeded coordination failures, all toggleable via `SeededFailureConfig`. AgentLens must detect them from observed spans alone — it has no access to the agent's internal state.

| Failure | What happens |
|---|---|
| Context loss | Synthesis silently drops sources when count exceeds its cap |
| Race condition | Writer starts from partial synthesis instead of the completed one |
| Contradiction ignored | Conflicting sources not flagged — one is silently picked |
| Delegation loop | Synthesis re-queries search without a termination condition |
| Context bleed | State from a previous session leaks into the current run |

---

### In progress

**Eval Runner** (`agentlens/eval/`)
Runs each `BenchmarkCase` against the agent. Checks `expected_criteria` against OTel spans and output — deterministic checks first (token ratios, loop counts, source counts), LLM-as-judge for open-ended criteria. Produces an `EvalResult` with per-case pass/fail, reasons, and aggregate pass rate.

**Diagnosis Agent**
Reads failing traces from Phoenix. Retrieves top-5 semantically similar past failures from the ChromaDB library. Combines these with the AgentProfile to produce a grounded plain English diagnosis. Stores the diagnosis back into the failure library so future runs benefit from it.

**Improvement Agent**
Reads the diagnosis and forms a fix hypothesis. Sends only the failing component to the LLM (not the full codebase). Runs the variant against failing cases in isolated E2B sandboxes. Computes Wilson score confidence intervals on the pass rate delta. Promotes the fix if CIs are non-overlapping. Loops up to 5 iterations if not.

**Voice Agent**
Spoken briefing delivered via ElevenLabs after every eval run. Conversational Q&A over trace data via Vapi (STT → LLM → TTS). Served from a local FastAPI server at `localhost:7432` that opens automatically in the browser.

**CLI** (`agentlens/cli/`)
Typer-based CLI with `observe`, `diagnose`, and `run` commands. Dynamically imports the developer's agent module via `sys.path` manipulation — zero changes to their codebase. Detection rate gate for CI/CD integration (`--assert-detection-rate 5/5`).

---

## Getting started

### Prerequisites

- Python 3.10+
- Anthropic API key: `export ANTHROPIC_API_KEY=...`
- Phoenix (optional, for the trace UI): `pip install arize-phoenix && phoenix serve`

### Install

```bash
git clone https://github.com/M4h1m4/AgentLens.git
cd AgentLens
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### Run the tests

```bash
pytest tests/ -v
```

140 tests pass with no API keys and no running services. The test suite covers the adapter, intake agent, failure library, and benchmark generator — all using fake LLMs and in-memory exporters.

### Try the reference agent

```bash
python -c "
from reference_agent.run import run
print(run('The state of renewable energy in 2025')[:300])
"
```

### Observe a run with OTel tracing

```python
from agentlens.adapter import setup_otel, instrument_langchain, traced_invoke
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from reference_agent.config import AgentConfig
from reference_agent.graph import build_graph

exporter = InMemorySpanExporter()
setup_otel(exporter=exporter)
instrument_langchain()

app = build_graph(AgentConfig())
trace_id, result = traced_invoke(app, {"query": "climate change", "session_id": "s1"})

spans = list(exporter.get_finished_spans())
print(f"Trace: {trace_id} — {len(spans)} spans captured")
```

---

## Tech stack

| Concern | Choice |
|---|---|
| Agent framework | LangGraph |
| LLM | Claude (Anthropic) via `langchain-anthropic` |
| Tracing | OpenTelemetry + `openinference-instrumentation-langchain` |
| Local trace backend | Phoenix (Arize) |
| Vector store | ChromaDB |
| Failure taxonomy | MAST (Multi-Agent System Testing) |
| Sandboxed execution | E2B |
| Voice | ElevenLabs (TTS) + Vapi (conversational) |
| CLI | Typer |
| API server | FastAPI |

---

## License

MIT
