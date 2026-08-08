"""Eval Runner — runs a BenchmarkSuite against an agent and produces an EvalResult.

The runner is structured around an AgentExecutor protocol so the execution
backend can be swapped without changing any eval logic:

  InProcessExecutor   — runs any compiled LangGraph application in the current
                        Python process. Requires setup_otel() + instrument_langchain()
                        to have been called once before use.

  SandboxExecutor     — runs any LangGraph application inside an E2B microVM.
                        Used by the Improvement Agent when running LLM-generated
                        code variants that have not been reviewed. The agent to run
                        is fully parameterised — no specific agent is hardcoded.

Both executors return the same (trace_id, output, spans) tuple. The EvalRunner
never knows or cares which backend ran the code.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from agentlens.benchmark.schema import BenchmarkCase, BenchmarkSuite
from agentlens.eval.checker import check_all_criteria
from agentlens.eval.schema import CaseResult, EvalResult


# ── executor protocol ─────────────────────────────────────────────────────────

@runtime_checkable
class AgentExecutor(Protocol):
    """Interface for running the agent with a query.

    Returns:
        trace_id: Hex trace ID string from the run.
        output:   Agent's final text output (for LLM judge).
        spans:    OTel ReadableSpan list from the run.
    """

    def execute(
        self, query: str, session_id: str
    ) -> tuple[str, str, list]:
        ...


# ── in-process executor ───────────────────────────────────────────────────────

class InProcessExecutor:
    """Runs any compiled LangGraph application in the current Python process.

    Requires ``setup_otel(exporter=exporter)`` and ``instrument_langchain()``
    to have been called once before the first ``execute()`` call.

    Args:
        app:                 Compiled LangGraph application (``graph.compile()``).
        exporter:            InMemorySpanExporter — cleared before each run so
                             spans from one case don't bleed into the next.
        output_key:          State key that holds the final text output.
        extra_initial_state: Additional key/value pairs merged into the initial
                             state dict alongside ``query`` and ``session_id``.
                             Use this for agent-specific fields your graph requires
                             (e.g. ``{"loop_count": 0}`` for graphs that track loops).
    """

    def __init__(
        self,
        app: Any,
        exporter: Any,
        output_key: str = "report",
        extra_initial_state: dict | None = None,
    ) -> None:
        self._app = app
        self._exporter = exporter
        self._output_key = output_key
        self._extra = extra_initial_state or {}

    def execute(self, query: str, session_id: str) -> tuple[str, str, list]:
        from agentlens.adapter import traced_invoke

        self._exporter.clear()
        initial = {"query": query, "session_id": session_id, **self._extra}
        trace_id, result = traced_invoke(self._app, initial, session_id=session_id)
        spans = list(self._exporter.get_finished_spans())
        output = str(result.get(self._output_key, ""))
        return trace_id, output, spans


# ── sandbox executor (E2B) ────────────────────────────────────────────────────

class SandboxExecutor:
    """Runs any LangGraph application inside an E2B microVM sandbox.

    Each call to execute():
      1. Spins up a fresh E2B sandbox (~500ms)
      2. Uploads the entire agent_dir to /app in the sandbox
      3. Runs the agent with the given query via the AgentLens adapter
      4. Reads back the output and OTel spans exported as JSON
      5. Destroys the sandbox

    The sandbox cannot affect the host filesystem, environment variables,
    or other processes. Used by the Improvement Agent when running
    LLM-generated code variants that have not been reviewed.

    Args:
        agent_dir:           Local path to the agent package to upload.
                             The entire directory tree is uploaded to /app.
        import_lines:        Python import statements placed at the top of the
                             sandbox runner script. Must import everything needed
                             to build the app.
                             Example::
                               "from my_agent.graph import build_graph\\n"
                               "from my_agent.config import MyConfig"
        build_app_expr:      Python expression (one line) that constructs and
                             returns the compiled LangGraph application.
                             Example: ``"build_graph(MyConfig())"``
        output_key:          State key that holds the final text output.
        extra_initial_state: Agent-specific fields added to the initial state
                             dict alongside ``query`` and ``session_id``.
                             Must be JSON-serialisable.
                             Example: ``{"loop_count": 0}``
        pip_packages:        Extra pip packages to install in the sandbox beyond
                             the AgentLens baseline. Useful for agent-specific
                             dependencies not bundled in agent_dir.
        e2b_api_key:         E2B API key. Reads E2B_API_KEY env var if not set.
    """

    def __init__(
        self,
        agent_dir: str,
        import_lines: str,
        build_app_expr: str,
        output_key: str = "report",
        extra_initial_state: dict | None = None,
        pip_packages: list[str] | None = None,
        e2b_api_key: str | None = None,
    ) -> None:
        self._agent_dir = agent_dir
        self._import_lines = import_lines
        self._build_app_expr = build_app_expr
        self._output_key = output_key
        self._extra = extra_initial_state or {}
        self._pip_packages = pip_packages or []
        self._api_key = e2b_api_key

    def execute(self, query: str, session_id: str) -> tuple[str, str, list]:
        import json
        import os
        from pathlib import Path

        try:
            from e2b_code_interpreter import Sandbox
        except ImportError:
            raise ImportError(
                "e2b_code_interpreter is required for SandboxExecutor. "
                "Install it with: pip install e2b-code-interpreter"
            )

        api_key = self._api_key or os.environ.get("E2B_API_KEY")

        # Merge query + session_id with any agent-specific extra state fields
        initial_state = {"query": query, "session_id": session_id, **self._extra}

        runner_script = f"""
import sys, json
sys.path.insert(0, "/app")

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from agentlens.adapter import setup_otel, instrument_langchain, traced_invoke

{self._import_lines}

exporter = InMemorySpanExporter()
setup_otel(exporter=exporter)
instrument_langchain()

app = {self._build_app_expr}
initial = {json.dumps(initial_state)}
trace_id, result = traced_invoke(app, initial, session_id={json.dumps(session_id)})

spans_data = []
for span in exporter.get_finished_spans():
    spans_data.append({{
        "name": span.name,
        "attributes": dict(span.attributes or {{}}),
        "trace_id": format(span.context.trace_id, "032x"),
    }})

output = {{
    "trace_id": trace_id,
    "output": str(result.get({json.dumps(self._output_key)}, "")),
    "spans": spans_data,
}}
print(json.dumps(output))
"""

        baseline_packages = (
            "agentlens langchain-anthropic langgraph "
            "opentelemetry-sdk openinference-instrumentation-langchain"
        )
        extra_packages = " ".join(self._pip_packages)
        install_cmd = f"pip install {baseline_packages} {extra_packages} -q"

        with Sandbox(api_key=api_key) as sbx:
            agent_path = Path(self._agent_dir)
            for f in agent_path.rglob("*.py"):
                rel = f.relative_to(agent_path.parent)
                sbx.files.write(f"/app/{rel}", f.read_text())

            sbx.run_code(install_cmd)
            result_raw = sbx.run_code(runner_script)

            if result_raw.exit_code != 0:
                raise RuntimeError(
                    f"Sandbox execution failed:\n{result_raw.logs.stderr}"
                )

            data = json.loads(result_raw.logs.stdout.strip())

        spans = [_SpanProxy(s["name"], s["attributes"]) for s in data["spans"]]
        return data["trace_id"], data["output"], spans


class _SpanProxy:
    """Duck-typed span compatible with checker.py — mirrors IntakePhoenixReader._SpanProxy."""

    def __init__(self, name: str, attributes: dict) -> None:
        self.name = name
        self.attributes = attributes


# ── eval runner ───────────────────────────────────────────────────────────────

class EvalRunner:
    """Run a BenchmarkSuite against an agent and produce an EvalResult.

    Usage (in-process, for known safe agent)::

        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
        from agentlens.adapter import setup_otel, instrument_langchain
        from agentlens.eval import EvalRunner, InProcessExecutor

        exporter = InMemorySpanExporter()
        setup_otel(exporter=exporter)
        instrument_langchain()

        executor = InProcessExecutor(app=build_graph(config), exporter=exporter)
        runner = EvalRunner(executor=executor)
        result = runner.run(suite)

    Usage (sandbox, for LLM-generated variants)::

        executor = SandboxExecutor(
            agent_dir="./my_agent",
            import_lines="from my_agent.graph import build_graph\\nfrom my_agent.config import Config",
            build_app_expr="build_graph(Config())",
            output_key="report",
            extra_initial_state={"loop_count": 0},   # only if your graph needs it
        )
        runner = EvalRunner(executor=executor)
        result = runner.run(suite)

    For tests — inject a FakeExecutor and optional fake judge LLM::

        runner = EvalRunner(executor=FakeExecutor(spans), judge_llm=FakeJudge())
        result = runner.run(suite)
    """

    def __init__(
        self,
        executor: AgentExecutor,
        judge_llm: Any = None,
        session_prefix: str = "eval",
    ) -> None:
        self._executor = executor
        self._judge_llm = judge_llm
        self._session_prefix = session_prefix

    def run(self, suite: BenchmarkSuite) -> EvalResult:
        """Run all cases in the suite. Returns a complete EvalResult."""
        case_results: list[CaseResult] = []
        for i, case in enumerate(suite.cases):
            result = self._run_case(case, session_id=f"{self._session_prefix}-{i}")
            case_results.append(result)
        return EvalResult.build(suite_id=suite.suite_id, results=case_results)

    def _run_case(self, case: BenchmarkCase, session_id: str) -> CaseResult:
        try:
            trace_id, output, spans = self._executor.execute(
                query=case.query, session_id=session_id
            )
        except Exception as exc:
            # Execution failure → all criteria fail with the error as reason
            criteria_results = [
                type("CR", (), {
                    "criterion": c, "passed": False,
                    "reason": f"Execution error: {exc}", "check_type": "error",
                    "to_dict": lambda self: {
                        "criterion": self.criterion, "passed": self.passed,
                        "reason": self.reason, "check_type": self.check_type,
                    }
                })()
                for c in case.expected_criteria
            ]
            from agentlens.eval.schema import CriterionResult
            return CaseResult(
                case_id=case.case_id,
                query=case.query,
                failure_mode=case.failure_mode,
                passed=False,
                criteria_results=[
                    CriterionResult(c, False, f"Execution error: {exc}", "error")
                    for c in case.expected_criteria
                ],
                trace_id="",
                spans_captured=0,
            )

        criteria_results = check_all_criteria(
            criteria=case.expected_criteria,
            query=case.query,
            output=output,
            spans=spans,
            judge_llm=self._judge_llm,
        )

        return CaseResult(
            case_id=case.case_id,
            query=case.query,
            failure_mode=case.failure_mode,
            passed=all(r.passed for r in criteria_results),
            criteria_results=criteria_results,
            trace_id=trace_id,
            spans_captured=len(spans),
        )
