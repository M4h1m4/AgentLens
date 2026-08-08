"""AgentLens Improvement Agent — autonomous code fix generation.

Given a Diagnosis, a baseline EvalResult, a BenchmarkSuite, and source files
from the target agent, the ImprovementAgent proposes and validates code fixes
using a Wilson CI-gated, regression-checked iteration loop.

Public API:
    ImprovementAgent  — main agent class
    ImprovementResult — structured result dataclass
    ExecutorConfig    — SandboxExecutor constructor params
    CodeLocator       — local, non-LLM file selector
"""

from __future__ import annotations

from agentlens.improvement.agent import ImprovementAgent
from agentlens.improvement.code_locator import CodeLocator
from agentlens.improvement.schema import ExecutorConfig, ImprovementResult

__all__ = [
    "ImprovementAgent",
    "ImprovementResult",
    "ExecutorConfig",
    "CodeLocator",
]
