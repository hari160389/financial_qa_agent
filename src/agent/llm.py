from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any, TypeVar

from anthropic import Anthropic
from openai import OpenAI

from src.agent.schemas import ChatMessage, LLMResponse, OpenAIToolDefinition, ToolCall
from src.config.constants import LLM_BASE_URL, get_api_key, get_model

T = TypeVar("T")


def _retry(func: Callable[[], T], max_attempts: int = 6) -> T:
    last_error: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return func()
        except Exception as exc:
            last_error = exc
            message = str(exc).lower()
            if "rate_limit" in message or "429" in message or "overloaded" in message:
                time.sleep(min(2**attempt, 30))
                continue
            raise
    if last_error is None:
        raise RuntimeError("Retry failed without capturing an error")
    raise last_error


def _to_anthropic_tools(openai_tools: list[OpenAIToolDefinition]) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for tool in openai_tools:
        function = tool["function"]
        tools.append(
            {
                "name": function["name"],
                "description": function["description"],
                "input_schema": function["parameters"],
            }
        )
    return tools


def _call_openai(
    messages: list[ChatMessage],
    tools: list[OpenAIToolDefinition],
    model: str,
    api_key: str,
) -> LLMResponse:
    kwargs: dict[str, Any] = {"api_key": api_key}
    if LLM_BASE_URL:
        kwargs["base_url"] = LLM_BASE_URL
    client = OpenAI(**kwargs)

    openai_messages: list[dict[str, Any]] = []
    for message in messages:
        if message["role"] == "tool":
            openai_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": message["tool_call_id"],
                    "content": message.get("content") or "",
                }
            )
            continue

        item: dict[str, Any] = {"role": message["role"], "content": message.get("content")}
        if message.get("tool_calls"):
            item["tool_calls"] = [
                {
                    "id": call["id"],
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": json.dumps(call["arguments"]),
                    },
                }
                for call in message["tool_calls"]
            ]
        openai_messages.append(item)

    response = _retry(
        lambda: client.chat.completions.create(
            model=model,
            messages=openai_messages,
            tools=tools,
            tool_choice="auto",
        )
    )
    message = response.choices[0].message
    tool_calls: list[ToolCall] = []
    for call in message.tool_calls or []:
        tool_calls.append(
            {
                "id": call.id,
                "name": call.function.name,
                "arguments": json.loads(call.function.arguments),
            }
        )
    return {"content": message.content, "tool_calls": tool_calls}


def _call_anthropic(
    messages: list[ChatMessage],
    tools: list[OpenAIToolDefinition],
    model: str,
    api_key: str,
) -> LLMResponse:
    client = Anthropic(api_key=api_key)
    system_parts: list[str] = []
    anthropic_messages: list[dict[str, Any]] = []

    for message in messages:
        if message["role"] == "system":
            if message.get("content"):
                system_parts.append(message["content"])
            continue

        if message["role"] == "tool":
            anthropic_messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": message["tool_call_id"],
                            "content": message.get("content") or "",
                        }
                    ],
                }
            )
            continue

        if message["role"] == "assistant" and message.get("tool_calls"):
            blocks: list[dict[str, Any]] = []
            if message.get("content"):
                blocks.append({"type": "text", "text": message["content"]})
            for call in message["tool_calls"]:
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": call["id"],
                        "name": call["name"],
                        "input": call["arguments"],
                    }
                )
            anthropic_messages.append({"role": "assistant", "content": blocks})
            continue

        anthropic_messages.append(
            {"role": message["role"], "content": message.get("content") or ""}
        )

    response = _retry(
        lambda: client.messages.create(
            model=model,
            max_tokens=4096,
            system="\n\n".join(system_parts),
            messages=anthropic_messages,
            tools=_to_anthropic_tools(tools),
        )
    )

    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_input = block.input
            if isinstance(tool_input, str):
                arguments = json.loads(tool_input)
            else:
                arguments = dict(tool_input)
            tool_calls.append({"id": block.id, "name": block.name, "arguments": arguments})

    content = "\n".join(text_parts).strip() or None
    return {"content": content, "tool_calls": tool_calls}


def call_llm(
    messages: list[ChatMessage],
    tools: list[OpenAIToolDefinition],
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> LLMResponse:
    resolved_provider = provider or "openai"
    resolved_model = model or get_model(resolved_provider)
    resolved_api_key = api_key or get_api_key(resolved_provider)
    if not resolved_api_key:
        raise ValueError(f"No API key found for provider: {resolved_provider}")

    if resolved_provider == "openai":
        return _call_openai(messages, tools, resolved_model, resolved_api_key)
    if resolved_provider == "anthropic":
        return _call_anthropic(messages, tools, resolved_model, resolved_api_key)
    raise ValueError(f"Unsupported provider: {resolved_provider}")
