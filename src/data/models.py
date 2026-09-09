from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

TableCell = float | str | int
Table = dict[str, dict[str, TableCell]]


class Document(BaseModel):
    pre_text: str = ""
    post_text: str = ""
    table: Table = Field(default_factory=dict)


class Dialogue(BaseModel):
    conv_questions: list[str] = Field(default_factory=list)
    conv_answers: list[str] = Field(default_factory=list)
    turn_program: list[str] = Field(default_factory=list)
    executed_answers: list[float | str] = Field(default_factory=list)
    qa_split: list[bool] = Field(default_factory=list)


class Features(BaseModel):
    num_dialogue_turns: int = 0
    has_type2_question: bool = False
    has_duplicate_columns: bool = False
    has_non_numeric_values: bool = False


class ConvFinQARecord(BaseModel):
    id: str
    doc: Document
    dialogue: Dialogue
    features: Features

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConvFinQARecord:
        return cls.model_validate(data)
