"""Voice Agent — briefing generation and Vapi webhook handler.

Two entry points:
    generate_briefing(eval_result, diagnoses, improvement)
        → fills server_state.audio_bytes with ElevenLabs-generated MP3

    vapi_webhook(request)
        → FastAPI route handler, reads server_state context, answers via Claude
"""

from __future__ import annotations

import os

from elevenlabs import ElevenLabs
from fastapi import Request
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agentlens.diagnosis.schema import Diagnosis
from agentlens.eval.schema import EvalResult
from agentlens.improvement.schema import ImprovementResult
from agentlens.server.state import server_state


# ── Clients ───────────────────────────────────────────────────────────────────
# Initialised once at import time. Both read from environment variables so
# no credentials are hardcoded.

_elevenlabs = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])

_llm = ChatAnthropic(model="claude-sonnet-4-6")


# ── Briefing text generation ──────────────────────────────────────────────────

def _verbalize_percent(rate: float) -> str:
    """Convert 0.636 → 'sixty-three percent'. TTS reads digits unnaturally."""
    return f"{round(rate * 100)} percent"


def _verbalize_fraction(numerator: int, denominator: int) -> str:
    """Convert (14, 22) → 'fourteen of twenty-two'. More natural in speech."""
    # For small numbers, words sound better than digits in TTS.
    # For larger numbers, digits are fine — TTS handles them correctly.
    words = {
        0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
        5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
        10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen",
        14: "fourteen", 15: "fifteen", 16: "sixteen", 17: "seventeen",
        18: "eighteen", 19: "nineteen", 20: "twenty",
    }
    n = words.get(numerator, str(numerator))
    d = words.get(denominator, str(denominator))
    return f"{n} of {d}"


def generate_briefing_text(
    eval_result: EvalResult,
    diagnoses: list[Diagnosis],
    improvement: ImprovementResult | None,
) -> str:
    """Format pipeline outputs into spoken sentences for ElevenLabs.

    Structure: key result → failure breakdown → diagnosis → improvement outcome.
    This ordering follows the rule: listeners hear linearly, so lead with what
    matters most.

    Args:
        eval_result:  The EvalResult from the most recent eval run.
        diagnoses:    One Diagnosis per failing failure mode.
        improvement:  The ImprovementResult, or None if improvement was not run.

    Returns:
        A plain text string ready to send to ElevenLabs. No markdown, no bullets.
    """
    sentences: list[str] = []

    # ── Key result — always first ─────────────────────────────────────────────
    fraction = _verbalize_fraction(
        eval_result.passed_cases, eval_result.total_cases
    )
    sentences.append(f"Eval complete. {fraction} benchmarks passed.")

    # ── Failure breakdown — one sentence per failing MAST category ────────────
    if eval_result.failures_by_mode:
        for mode, count in eval_result.failures_by_mode.items():
            mode_readable = mode.replace("_", " ")
            noun = "failure" if count == 1 else "failures"
            sentences.append(f"{count} {noun} in the {mode_readable} category.")
    else:
        sentences.append("All benchmark cases passed. No failures detected.")

    # ── Diagnosis — root cause only, one sentence per diagnosis ──────────────
    # We use root_cause (one sentence) rather than evidence (list) because
    # TTS of a list sounds robotic. The user can ask for detail via Q&A.
    for d in diagnoses:
        sentences.append(d.root_cause)

    # ── Improvement outcome ───────────────────────────────────────────────────
    if improvement is not None:
        if improvement.promoted:
            old_pct = _verbalize_percent(improvement.baseline_eval.pass_rate)
            new_pct = _verbalize_percent(improvement.variant_eval.pass_rate)
            sentences.append(
                f"A fix was found and promoted. "
                f"Pass rate improved from {old_pct} to {new_pct}."
            )
        else:
            sentences.append(
                f"No statistically significant improvement was found. "
                f"{improvement.reason}"
            )

    # ── Invitation to ask questions ───────────────────────────────────────────
    sentences.append("Ask me any questions about the results.")

    # Join with a space. Each sentence already ends with a period so TTS
    # handles the pause between them naturally.
    return " ".join(sentences)


def generate_audio(text: str) -> bytes:
    """Send text to ElevenLabs and return raw MP3 bytes.

    ElevenLabs returns a generator of byte chunks. We join them into a single
    bytes object so FastAPI can serve it as a complete audio file.

    Args:
        text: The briefing text from generate_briefing_text().

    Returns:
        MP3 audio bytes ready to write to a file or serve via HTTP.
    """
    audio_generator = _elevenlabs.generate(
        text=text,
        voice="Rachel",                  # change to any ElevenLabs voice name
        model="eleven_monolingual_v1",   # standard quality — sufficient for briefings
    )
    return b"".join(audio_generator)


def run_briefing(
    eval_result: EvalResult,
    diagnoses: list[Diagnosis],
    improvement: ImprovementResult | None,
) -> None:
    """Generate briefing text + audio and store in server_state.

    Called by the CLI after the pipeline completes. Does NOT open the browser —
    that is the CLI's job because it knows when the FastAPI server is ready.

    Args:
        eval_result:  Passed from EvalRunner.
        diagnoses:    Passed from DiagnosisAgent.
        improvement:  Passed from ImprovementAgent, or None.
    """
    # Store pipeline outputs so the Vapi webhook can access them later
    server_state.eval_result = eval_result
    server_state.diagnoses = diagnoses
    server_state.improvement = improvement
    server_state.clear_conversation()   # fresh Q&A history for each run

    # Generate and store the briefing audio
    text = generate_briefing_text(eval_result, diagnoses, improvement)
    server_state.audio_bytes = generate_audio(text)


# ── Context summary for the Q&A LLM ──────────────────────────────────────────

def _build_context_summary() -> str:
    """Build a compact text summary of server_state for the LLM's context window.

    The LLM in the Vapi webhook does not receive the Python objects directly —
    it receives text. This function serialises the key facts into a format the
    LLM can reason about clearly.

    Returns:
        Multi-line string injected into every Q&A LLM call as context.
        Returns a placeholder if no run data is stored yet.
    """
    if server_state.eval_result is None:
        return "No eval results available yet."

    e = server_state.eval_result
    lines = [
        f"Pass rate: {e.passed_cases} of {e.total_cases} "
        f"({round(e.pass_rate * 100)}%)",
        f"Wilson 95% CI: [{e.ci_low:.2f}, {e.ci_high:.2f}]",
        f"Failures by mode: {e.failures_by_mode}",
    ]

    for d in server_state.diagnoses:
        lines += [
            f"",
            f"Failure mode: {d.failure_mode}",
            f"Root cause: {d.root_cause}",
            f"Evidence: {'; '.join(d.evidence)}",
            f"Suggested fix: {d.suggested_fix}",
            f"Diagnosis confidence: {d.confidence}",
            f"Similar past failures found: {d.similar_past_count}",
        ]

    if server_state.improvement is not None:
        imp = server_state.improvement
        lines += [
            f"",
            f"Improvement promoted: {imp.promoted}",
            f"Iterations run: {imp.iterations_run}",
            f"Cost: ${imp.cost_dollars:.4f}",
            f"Hypothesis: {imp.hypothesis}",
            f"Reason: {imp.reason}",
        ]

    return "\n".join(lines)


# ── Vapi webhook handler ──────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a concise voice assistant explaining the results of an automated "
    "eval run on a multi-agent system. "
    "Answer in one to three spoken sentences. "
    "No bullet points. No markdown. No asterisks. "
    "Speak as if the listener cannot see a screen — verbalize numbers and "
    "percentages rather than writing them as digits."
)


async def handle_vapi_webhook(request: Request) -> dict:
    """Handle the POST request Vapi sends when the user asks a question.

    Vapi flow:
        user speaks → Vapi STT → transcript → POST /vapi/webhook
        ← your response text ← Vapi TTS ← spoken to user

    The request body from Vapi looks like:
        {
            "message": {
                "type": "transcript",
                "role": "user",
                "transcript": "Which benchmark case failed?"
            }
        }

    We return:
        { "result": "<answer text>" }

    Vapi reads the "result" field and speaks it aloud.

    Conversation history is accumulated in server_state.conversation so the
    LLM has context from earlier turns in the same session.

    Args:
        request: The raw FastAPI Request object from the POST /vapi/webhook route.

    Returns:
        Dict with a "result" key containing the answer text for Vapi to speak.
    """
    body = await request.json()

    # Extract the user's question from Vapi's request format
    message = body.get("message", {})
    question = message.get("transcript", "")

    if not question:
        return {"result": "I didn't catch that. Could you ask again?"}

    # Append user turn to conversation history
    server_state.conversation.append(("user", question))

    # Build the messages list for Claude:
    # 1. System prompt — defines role and output constraints
    # 2. Context block — the current run's eval data
    # 3. Conversation history — all prior turns in this session
    # 4. Current question — the new user message
    messages = [
        SystemMessage(SYSTEM_PROMPT),
        HumanMessage(
            f"Here is the context for this eval run:\n\n"
            f"{_build_context_summary()}"
        ),
    ]

    # Add prior conversation turns so the LLM remembers what was already said
    for role, text in server_state.conversation[:-1]:  # exclude the current turn
        if role == "user":
            messages.append(HumanMessage(text))
        else:
            messages.append(AIMessage(text))

    # Add the current question
    messages.append(HumanMessage(question))

    # Call Claude
    response = _llm.invoke(messages)
    answer = response.content

    # Append assistant turn to conversation history for future turns
    server_state.conversation.append(("assistant", answer))

    # Return in the format Vapi expects
    return {"result": answer}
