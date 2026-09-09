from __future__ import annotations

import json

from src.agent.llm import call_llm
from src.agent.schemas import ChatMessage, SubmitAnswer, TurnResult
from src.agent.tools import get_tool_definitions, run_tool
from src.config.constants import MAX_TOOL_ROUNDS, PROMPT_VERSION, get_system_prompt
from src.data.document import format_document
from src.data.models import ConvFinQARecord


def format_history(history: list[TurnResult]) -> str:
    if not history:
        return "No prior turns."
    lines = []
    for index, turn in enumerate(history, start=1):
        value = turn.numeric_value
        if value is None:
            value = turn.answer
        lines.append(f"Turn {index} Q: {turn.question}\nTurn {index} A: {value}")
    return "\n".join(lines)


def answer_turn(
    record: ConvFinQARecord,
    question: str,
    history: list[TurnResult],
    provider: str | None = None,
    model: str | None = None,
    prompt_version: str | None = None,
) -> TurnResult:
    doc = record.doc
    tools = get_tool_definitions()
    messages: list[ChatMessage] = [
        {"role": "system", "content": get_system_prompt(prompt_version or PROMPT_VERSION)},
        {
            "role": "user",
            "content": (
                f"Document:\n{format_document(doc)}\n\n"
                f"Conversation so far:\n{format_history(history)}\n\n"
                f"Current question: {question}"
            ),
        },
    ]

    submitted: SubmitAnswer | None = None
    for _ in range(MAX_TOOL_ROUNDS):
        response = call_llm(messages, tools, provider=provider, model=model)

        if response["tool_calls"]:
            messages.append(
                {
                    "role": "assistant",
                    "content": response["content"],
                    "tool_calls": response["tool_calls"],
                }
            )

            for call in response["tool_calls"]:
                result = run_tool(doc, call["name"], call["arguments"])
                if call["name"] == "submit_answer":
                    submitted = SubmitAnswer.model_validate(call["arguments"])
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": json.dumps(result),
                    }
                )
            if submitted is not None:
                break
            continue

        if response["content"]:
            print("Model replied without submit_answer, retrying...")
            messages.append(
                {
                    "role": "user",
                    "content": "You must call submit_answer. Do not reply with plain text.",
                }
            )

    if submitted is None:
        raise RuntimeError("Agent did not submit an answer within tool round limit.")

    return TurnResult(
        question=question,
        answer=submitted.answer_text,
        numeric_value=submitted.numeric_value,
        reasoning=submitted.reasoning,
    )
