"""AgentLens CLI — one command that runs the full eval pipeline.

Usage:
    agentlens run --entry graph:build_graph --query "climate change impacts"

What it does (in order):
    1. Imports the user's module and calls their build function
    2. Instruments LangChain with OTel (no code change in their project)
    3. Runs the agent and captures spans
    4. Extracts a FailureSignal from the spans
    5. Generates a BenchmarkSuite targeting detected failure modes
    6. Runs the suite through EvalRunner and prints pass/fail results
    7. Diagnoses failing cases using the MAST taxonomy + failure library
    8. Optionally runs the ImprovementAgent (--improve, requires E2B)
    9. Updates the failure library with the proven fix
    10. Optionally opens a voice briefing in the browser (--voice, requires ElevenLabs)
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import time
import threading
import webbrowser
from pathlib import Path
from typing import Optional

import typer
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from agentlens.adapter import instrument_langchain, setup_otel, traced_invoke
from agentlens.benchmark.gen_agent import BenchmarkGenAgent
from agentlens.diagnosis.agent import DiagnosisAgent
from agentlens.eval.runner import EvalRunner, InProcessExecutor
from agentlens.pipeline import update_library_after_improvement
from agentlens.rag.failure_signals import extract_signals
from agentlens.rag.library import FailureLibrary

app = typer.Typer(
    name="agentlens",
    help="Autonomous eval infrastructure for multi-agent systems.",
    add_completion=False,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_entry(entry: str, cwd: str):
    """Dynamically import module and return the build function.

    Args:
        entry: "module.path:function_name" — e.g. "my_agent.graph:build_graph"
        cwd:   The directory to add to sys.path so the module can be found.

    Returns:
        The callable (build function) from the module.
    """
    if ":" not in entry:
        typer.echo(
            f"[error] --entry must be in the format 'module:function'. Got: {entry}",
            err=True,
        )
        raise typer.Exit(1)

    module_path, func_name = entry.rsplit(":", 1)

    # Add the working directory to sys.path so the user's module is importable.
    # This is what lets AgentLens work without any code change in the user's project.
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        typer.echo(f"[error] Could not import '{module_path}': {exc}", err=True)
        raise typer.Exit(1)

    func = getattr(module, func_name, None)
    if func is None:
        typer.echo(
            f"[error] '{func_name}' not found in module '{module_path}'", err=True
        )
        raise typer.Exit(1)

    return func


def _print_eval_summary(eval_result) -> None:
    """Print a compact eval result table to the terminal."""
    typer.echo("")
    typer.echo("── Eval Results ──────────────────────────────────────────")
    typer.echo(
        f"  {eval_result.passed_cases}/{eval_result.total_cases} passed  "
        f"({round(eval_result.pass_rate * 100)}%)  "
        f"CI: [{eval_result.ci_low:.2f}, {eval_result.ci_high:.2f}]"
    )
    if eval_result.failures_by_mode:
        typer.echo("  Failures by mode:")
        for mode, count in eval_result.failures_by_mode.items():
            typer.echo(f"    {mode}: {count}")
    typer.echo("")


def _print_diagnoses(diagnoses: list) -> None:
    """Print diagnosis results to the terminal."""
    if not diagnoses:
        typer.echo("── Diagnosis ─────────────────────────────────────────────")
        typer.echo("  No failures to diagnose.")
        typer.echo("")
        return

    typer.echo("── Diagnosis ─────────────────────────────────────────────")
    for d in diagnoses:
        typer.echo(f"  [{d.failure_mode}] confidence={d.confidence}")
        typer.echo(f"  root_cause:    {d.root_cause}")
        typer.echo(f"  suggested_fix: {d.suggested_fix}")
        if d.similar_past_count:
            typer.echo(f"  grounded by {d.similar_past_count} proven past fix(es)")
        typer.echo("")


def _launch_voice(eval_result, diagnoses, improvement) -> None:
    """Generate audio and open browser tab. Blocks until Ctrl+C."""
    import uvicorn
    from agentlens.server.main import app as fastapi_app
    from agentlens.server.voice import run_briefing

    run_briefing(eval_result, diagnoses, improvement)

    server_thread = threading.Thread(
        target=lambda: uvicorn.run(
            fastapi_app, host="0.0.0.0", port=7432, log_level="error"
        ),
        daemon=True,
    )
    server_thread.start()

    time.sleep(1)
    webbrowser.open("http://localhost:7432")
    typer.echo("Voice briefing active at http://localhost:7432  (Ctrl+C to exit)")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


# ── run command ───────────────────────────────────────────────────────────────

@app.command()
def run(
    entry: str = typer.Option(
        ...,
        "--entry",
        help="Module and build function: 'my_agent.graph:build_graph'",
    ),
    query: str = typer.Option(
        ...,
        "--query",
        help="The query to run the agent with.",
    ),
    output_key: str = typer.Option(
        "output",
        "--output-key",
        help="Key in the final agent state that holds the output text.",
    ),
    extra_state: Optional[str] = typer.Option(
        None,
        "--extra-state",
        help="JSON string of extra fields to merge into the initial state. "
             "Example: '{\"loop_count\": 0}'",
    ),
    library_dir: str = typer.Option(
        ".agentlens/failures",
        "--library-dir",
        help="Directory for the persistent failure library (ChromaDB).",
    ),
    improve: bool = typer.Option(
        False,
        "--improve",
        help="Run the Improvement Agent after diagnosis. Requires E2B_API_KEY.",
    ),
    voice: bool = typer.Option(
        False,
        "--voice",
        help="Open a voice briefing in the browser after the run. "
             "Requires ELEVENLABS_API_KEY, VAPI_PUBLIC_KEY, VAPI_ASSISTANT_ID.",
    ),
    session_id: str = typer.Option(
        "agentlens-run",
        "--session-id",
        help="Session identifier attached to spans and library entries.",
    ),
) -> None:
    """Run the full AgentLens eval pipeline against a LangGraph agent."""

    cwd = os.getcwd()

    # ── Step 1: Load the user's build function ────────────────────────────────
    typer.echo(f"Loading '{entry}' from {cwd} ...")
    build_fn = _load_entry(entry, cwd)

    # ── Step 2: Parse extra initial state ─────────────────────────────────────
    extra_initial: dict = {}
    if extra_state:
        try:
            extra_initial = json.loads(extra_state)
        except json.JSONDecodeError as exc:
            typer.echo(f"[error] --extra-state is not valid JSON: {exc}", err=True)
            raise typer.Exit(1)

    # ── Step 3: Set up OTel ───────────────────────────────────────────────────
    exporter = InMemorySpanExporter()
    setup_otel(exporter=exporter)
    instrument_langchain()

    # ── Step 4: Build the app and run the agent ───────────────────────────────
    typer.echo(f"Running agent with query: \"{query}\"")
    app_instance = build_fn()
    initial_state = {"query": query, "session_id": session_id, **extra_initial}

    trace_id, final_state = traced_invoke(
        app_instance, initial_state, session_id=session_id
    )
    spans = list(exporter.get_finished_spans())

    typer.echo(f"Trace captured: {trace_id}  ({len(spans)} spans)")

    output_text = str(final_state.get(output_key, ""))
    if output_text:
        typer.echo(f"\nAgent output preview:\n{output_text[:300]}{'...' if len(output_text) > 300 else ''}")

    # ── Step 5: Extract FailureSignal and store in library ────────────────────
    signal = extract_signals(spans, trace_id=trace_id, session_id=session_id)
    typer.echo(
        f"\nFailure signal: context_drop={signal.context_drop_detected} "
        f"delegation_loop={signal.delegation_loop_detected}"
    )

    library = FailureLibrary(persist_dir=library_dir)
    library.store(signal)

    # ── Step 6: Generate benchmark suite ──────────────────────────────────────
    typer.echo("Generating benchmark suite ...")
    suite = BenchmarkGenAgent().generate(signal)
    typer.echo(f"Suite ready: {len(suite.cases)} cases targeting {suite.failure_modes_targeted}")

    # ── Step 7: Run eval ──────────────────────────────────────────────────────
    typer.echo("Running eval ...")
    executor = InProcessExecutor(
        app=app_instance,
        exporter=exporter,
        output_key=output_key,
        extra_initial_state=extra_initial or None,
    )
    eval_result = EvalRunner(executor=executor).run(suite)
    _print_eval_summary(eval_result)

    # ── Step 8: Diagnose ──────────────────────────────────────────────────────
    diagnoses = DiagnosisAgent(
        structured_llm=_default_diagnosis_llm(),
        library=library,
    ).diagnose(eval_result)
    _print_diagnoses(diagnoses)

    # ── Step 9: Improve (optional) ────────────────────────────────────────────
    improvement = None
    if improve and diagnoses:
        from agentlens.improvement.agent import ImprovementAgent
        from agentlens.improvement.code_locator import CodeLocator
        from agentlens.improvement.schema import ExecutorConfig

        typer.echo("Running Improvement Agent ...")
        agent_dir = _find_agent_dir(entry, cwd)
        component_files = CodeLocator().locate(agent_dir, diagnoses[0])

        config = ExecutorConfig(
            agent_dir=agent_dir,
            import_lines=_build_import_lines(entry),
            build_app_expr=_build_app_expr(entry),
            output_key=output_key,
            extra_initial_state=extra_initial or None,
        )

        improvement = ImprovementAgent(
            llm=_default_llm(),
            judge_llm=_default_llm(),
            config=config,
        ).improve(
            diagnosis=diagnoses[0],
            baseline_result=eval_result,
            suite=suite,
            component_files=component_files,
        )

        typer.echo(f"Improvement: promoted={improvement.promoted}")
        typer.echo(f"  {improvement.reason}")

        update_library_after_improvement(library, trace_id, improvement)

    # ── Step 10: Voice briefing (optional) ───────────────────────────────────
    if voice:
        _launch_voice(eval_result, diagnoses, improvement)


# ── LLM factory helpers ───────────────────────────────────────────────────────

def _default_llm():
    """Return the default Claude LLM. Fails fast if ANTHROPIC_API_KEY is missing."""
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(model="claude-sonnet-4-6", temperature=0)


def _default_diagnosis_llm():
    """Return the structured LLM for the DiagnosisAgent."""
    from agentlens.diagnosis.agent import _DiagnosisLLMOutput
    return _default_llm().with_structured_output(_DiagnosisLLMOutput)


# ── improvement agent helpers ─────────────────────────────────────────────────

def _find_agent_dir(entry: str, cwd: str) -> str:
    """Derive the agent package directory from the entry point string.

    For "my_agent.graph:build_graph" → returns "<cwd>/my_agent"
    For "graph:build_graph"          → returns "<cwd>"
    """
    module_path = entry.rsplit(":", 1)[0]
    top_package = module_path.split(".")[0]
    candidate = Path(cwd) / top_package
    if candidate.is_dir():
        return str(candidate)
    return cwd


def _build_import_lines(entry: str) -> str:
    """Build the import statement for the SandboxExecutor runner script.

    For "my_agent.graph:build_graph" → "from my_agent.graph import build_graph"
    """
    module_path, func_name = entry.rsplit(":", 1)
    return f"from {module_path} import {func_name}"


def _build_app_expr(entry: str) -> str:
    """Build the app construction expression for the SandboxExecutor.

    For "my_agent.graph:build_graph" → "build_graph()"
    """
    func_name = entry.rsplit(":", 1)[1]
    return f"{func_name}()"


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    app()


if __name__ == "__main__":
    main()
