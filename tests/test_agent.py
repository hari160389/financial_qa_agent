from unittest.mock import patch

from src.agent import answer_turn
from src.data.models import ConvFinQARecord


def test_answer_turn_with_mock_llm() -> None:
    record = ConvFinQARecord.from_dict(
        {
            "id": "test",
            "doc": {
                "pre_text": "text",
                "post_text": "",
                "table": {"2009": {"net income": 100.0}},
            },
            "dialogue": {
                "conv_questions": [],
                "conv_answers": [],
                "turn_program": [],
                "executed_answers": [],
                "qa_split": [],
            },
            "features": {
                "num_dialogue_turns": 0,
                "has_type2_question": False,
                "has_duplicate_columns": False,
                "has_non_numeric_values": False,
            },
        }
    )

    responses = [
        {
            "content": None,
            "tool_calls": [
                {
                    "id": "1",
                    "name": "lookup_table_value",
                    "arguments": {"column_header": "2009", "row_label": "net income"},
                }
            ],
        },
        {
            "content": None,
            "tool_calls": [
                {
                    "id": "2",
                    "name": "submit_answer",
                    "arguments": {
                        "answer_text": "100",
                        "numeric_value": 100.0,
                        "reasoning": "PLAN lookup net income 2009",
                    },
                }
            ],
        },
    ]

    with patch("src.agent.chat.call_llm", side_effect=responses):
        turn = answer_turn(record, "what is net income in 2009?", [])

    assert turn.answer == "100"
    assert turn.numeric_value == 100.0
