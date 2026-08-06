# AgentLens

Autonomous eval infrastructure for multi-agent systems.

AgentLens observes one real execution of your agent system, auto-generates a benchmark suite targeting the failure patterns it saw, runs those benchmarks, diagnoses failures in plain English, and autonomously tests fixes — all with zero changes to your codebase.

---

## What it does

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
  The synthesis agent drops sources beyond its internal cap of 3,
  causing the writer to produce reports missing key evidence.
  This pattern appeared in 4 of your last 6 failing runs.

Opening briefing at localhost:7432 ...
```

---

## How it works

AgentLens instruments your agent without touching your code. It hooks into the framework's own telemetry (via OpenTelemetry + LangchainInstrumentor), watches for coordination failures across handoffs, and builds an eval harness from what it observes.

```
Your agent (LangGraph / any OTel-emitting framework)
    │
    ├── LangchainInstrumentor — auto-captures every LLM call, tool call, node
    │
    ├── AgentLens HANDOFF layer — emits HANDOFF spans with source/target
    │
    └── All spans → Phoenix (localhost:6006)
            │
            ├── NL Intake Agent        — profiles your system from a description
            ├── Benchmark Gen Agent    — generates test cases from the trace
            ├── Eval Runner            — runs cases in isolated E2B sandboxes
            ├── Diagnosis Agent        — plain English failure explanation
            ├── Improvement Agent      — generates and tests fixes, Wilson CI gating
            └── Voice Agent            — spoken briefing + Q&A in browser tab
```

---

## Architecture

| Layer | Technology | Purpose |
|---|---|---|
| Trace format | OpenTelemetry (OTel) | Industry-standard spans, OTLP wire format |
| Auto-instrumentation | LangchainInstrumentor | Zero-code-change LLM/tool/node capture |
| HANDOFF detection | LangGraph streaming | Lateral agent coordination — not in any standard |
| Local trace backend | Phoenix (Arize) | LLM-aware UI + Python query API, no Docker needed |
| Failure library | ChromaDB | Semantic similarity search over past failures |
| Failure taxonomy | MAST (8 categories) | Context loss, race condition, delegation loop, and more |
| Agent orchestration | LangGraph | All AgentLens agents are LangGraph state machines |
| Sandboxed execution | E2B | Eval cases run in isolated ephemeral environments |
| Entry point | CLI (Typer) | Zero code changes in the developer's project |

---

## Project structure

```
agentlens/
├── adapter/          # OTel setup, LangGraph adapter, HANDOFF span emission
├── intake/           # NL intake agent, codebase inspector, Phoenix reader,
│                     # agent profile schema, trace reconciler, profile store
├── rag/              # Failure signal extraction, ChromaDB library, MAST taxonomy
└── benchmark/        # Benchmark Gen Agent, BenchmarkSuite schema, store
                      # (Day 5 — coming next)

reference_agent/      # Deliberately buggy 3-agent research system used as
                      # the test subject (search → synthesis → writer)
                      # Five seeded coordination failures, all toggleable

tests/
├── test_seeded_failures.py   # Ground truth: all 5 bugs must be detectable
├── test_adapter.py           # OTel spans, HANDOFF emission, traced_invoke
├── test_intake.py            # NL intake, inspector, reconciler, profile store
└── test_rag.py               # Failure signals, ChromaDB library, MAST taxonomy
```

---

## The reference agent

AgentLens ships with a deliberately buggy 3-agent research system. Its five seeded failures are the ground truth for proving the eval engine works:

| # | Failure | What happens |
|---|---|---|
| 1 | Context loss | Synthesis silently drops sources beyond its cap of 3 |
| 2 | Race condition | Writer starts from partial synthesis instead of complete one |
| 3 | Contradiction ignored | Conflicting sources are not flagged — one is silently chosen |
| 4 | Delegation loop | Synthesis re-queries search without a termination condition |
| 5 | Context bleed | State from a previous session leaks into the current run |

All failures are toggleable via `SeededFailureConfig`. AgentLens must detect them from observed spans alone — it has no access to the agent's internal state fields.

---

## Getting started

### Prerequisites

- Python 3.10+
- An Anthropic API key (`ANTHROPIC_API_KEY`)
- Phoenix for the local trace UI (optional): `pip install arize-phoenix && phoenix serve`

### Install

```bash
git clone https://github.com/M4h1m4/AgentLens.git
cd AgentLens
pip install -e ".[dev]"
```

### Run the tests

```bash
pytest tests/ -v
```

All 87 tests run without API keys or a running Phoenix instance.

### Observe a run

```bash
# Coming in Day 5+ when the CLI is built
agentlens run --entry reference_agent.graph:build_graph --query "climate change"
```

---

## Build progress

| Day | Component | Status |
|---|---|---|
| 1 | Reference agent (3-agent system, 5 seeded failures) | Done |
| 2 | Universal adapter (OTel spans, HANDOFF layer, Phoenix) | Done |
| 3 | NL Intake Agent (multi-turn profile, codebase inspection, reconciliation) | Done |
| 4 | RAG infrastructure (failure library, ChromaDB, MAST taxonomy) | Done |
| 5 | Benchmark Gen Agent | Next |
| 6 | Eval Runner + E2B sandboxes | Planned |
| 7 | Seeded failure detection verification | Planned |
| 8 | Diagnosis Agent | Planned |
| 9 | Improvement Agent | Planned |
| 10 | Voice Agent (ElevenLabs + Vapi) | Planned |
| 11–14 | Docker, Kubernetes, CI/CD, Prometheus + Grafana | Planned |

---

## Design decisions

Key architectural choices and the reasoning behind them are documented in `DESIGN_DECISIONS.md`. The most consequential ones:

- **OTel as the trace format** — industry standard, developers who already emit OTel get AgentLens for free with zero additional instrumentation
- **Phoenix as the local backend** — `pip install arize-phoenix`, Python query API, no Docker required
- **CLI as the entry point** — zero code changes in the developer's project; AgentLens imports and wraps the agent from outside
- **HANDOFF spans as a custom addition** — the only thing no existing OTel standard captures; lateral agent coordination is AgentLens's core signal

---

## License

MIT
