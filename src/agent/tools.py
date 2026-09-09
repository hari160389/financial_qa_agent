from __future__ import annotations

from typing import Any

from src.agent.schemas import (
    ArithmeticOp,
    CalculateArgs,
    LookupTableValueArgs,
    OpenAIToolDefinition,
    SubmitAnswer,
    openai_tool,
)
from src.data.document import list_table_structure
from src.data.models import Document


def calculate(operation: ArithmeticOp, left: float | int | str, right: float | int | str) -> float:
    left_value = float(left)
    right_value = float(right)
    if operation == "add":
        return left_value + right_value
    if operation == "subtract":
        return left_value - right_value
    if operation == "multiply":
        return left_value * right_value
    if operation == "divide":
        if right_value == 0:
            raise ZeroDivisionError("Division by zero")
        return left_value / right_value
    raise ValueError(f"Unsupported operation: {operation}")


def _match_key(query: str, options: list[str] | Any) -> str | None:
    normalized = query.strip().lower()
    for option in options:
        if option.strip().lower() == normalized:
            return option
    for option in options:
        if normalized in option.strip().lower():
            return option
    return None


def lookup_table_value(
    doc: Document,
    column_header: str,
    row_label: str,
) -> dict[str, Any]:
    table = doc.table
    column = _match_key(column_header, table.keys())
    if column is None:
        return {
            "error": f"Column not found: {column_header}",
            "available_columns": list(table.keys()),
        }

    rows = table[column]
    row = _match_key(row_label, rows.keys())
    if row is None:
        return {
            "error": f"Row not found: {row_label}",
            "available_rows": list(rows.keys()),
        }

    value = rows[row]
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = None
    return {"column": column, "row": row, "value": value, "numeric_value": numeric}


def get_tool_definitions() -> list[OpenAIToolDefinition]:
    return [
        openai_tool("list_table_structure", "List table column headers and row labels.", None),
        openai_tool(
            "lookup_table_value",
            "Look up a table cell by column and row.",
            LookupTableValueArgs,
        ),
        openai_tool("calculate", "Perform arithmetic on two numbers.", CalculateArgs),
        openai_tool(
            "submit_answer",
            "Submit final answer with chain-of-thought reasoning.",
            SubmitAnswer,
        ),
    ]


def run_tool(doc: Document, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name == "list_table_structure":
        return list_table_structure(doc)
    if name == "lookup_table_value":
        args = LookupTableValueArgs.model_validate(arguments)
        return lookup_table_value(doc, args.column_header, args.row_label)
    if name == "calculate":
        args = CalculateArgs.model_validate(arguments)
        return {"result": calculate(args.operation, args.left, args.right)}
    if name == "submit_answer":
        submit = SubmitAnswer.model_validate(arguments)
        return {"status": "accepted", **submit.model_dump()}
    raise ValueError(f"Unknown tool: {name}")
