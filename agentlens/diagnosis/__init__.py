"""AgentLens diagnosis — structured failure diagnosis grounded in MAST + past cases.

Public API::

    from agentlens.diagnosis import DiagnosisAgent, Diagnosis

    # Wire up with a real structured LLM
    from langchain_anthropic import ChatAnthropic
    llm = ChatAnthropic(model="claude-sonnet-4-6").with_structured_output(
        agentlens.diagnosis.agent._DiagnosisLLMOutput
    )

    # Or inject a fake for tests (returns _DiagnosisLLMOutput directly)
    agent = DiagnosisAgent(structured_llm=fake_llm, library=failure_library)
    diagnoses = agent.diagnose(eval_result)

    for d in diagnoses:
        print(f"[{d.confidence}] {d.failure_mode}: {d.root_cause}")
        print(f"  Fix: {d.suggested_fix}")
"""

from agentlens.diagnosis.schema import Diagnosis
from agentlens.diagnosis.agent import DiagnosisAgent, _DiagnosisLLMOutput

__all__ = [
    "Diagnosis",
    "DiagnosisAgent",
    "_DiagnosisLLMOutput",
]
