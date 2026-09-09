from __future__ import annotations

from typing import Any, Literal, NotRequired, TypedDict

from pydantic import BaseModel

ProviderName = Literal["openai", "anthropic"]
ArithmeticOp = Literal["add", "subtract", "multiply", "divide"]


class TurnResult(BaseModel):
    question: str
    answer: str
    numeric_value: float | None = None
    reasoning: str | None = None


class SubmitAnswer(BaseModel):
    answer_text: str
    numeric_value: float | None = None
    reasoning: str


class LookupTableValueArgs(BaseModel):
    column_header: str
    row_label: str


class CalculateArgs(BaseModel):
    operation: ArithmeticOp
    left: float
    right: float


class ToolCall(TypedDict):
    id: str
    name: str
    arguments: dict[str, Any]


class LLMResponse(TypedDict):
    content: str | None
    tool_calls: list[ToolCall]


class ChatMessage(TypedDict):
    role: str
    content: NotRequired[str | None]
    tool_calls: NotRequired[list[ToolCall]]
    tool_call_id: NotRequired[str]


class OpenAIToolDefinition(TypedDict):
    type: Literal["function"]
    function: dict[str, Any]


def tool_parameters(model: type[BaseModel] | None) -> dict[str, Any]:
    if model is None:
        return {"type": "object", "properties": {}, "required": []}
    schema = model.model_json_schema()
    schema.pop("title", None)
    return schema


def openai_tool(name: str, description: str, args_model: type[BaseModel] | None) -> OpenAIToolDefinition:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": tool_parameters(args_model),
        },
    }
