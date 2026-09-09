from __future__ import annotations

import json
import math
import os
from datetime import UTC, datetime
from typing import Any, TypedDict

from src.agent import answer_turn
from src.agent.schemas import TurnResult
from src.config.constants import (
    ANSWER_ABS_TOL,
    ANSWER_REL_TOL,
    EVAL_SAMPLE_SIZE,
    LLM_PROVIDER,
    PROMPT_VERSION,
    RESULTS_DIR,
    get_model,
)
from src.data.dataset import get_split_records


class TurnEvalResult(TypedDict):
    record_id: str
    turn_index: int
    question: str
    expected: float | str
    predicted: TurnResult
    correct: bool


class EvalReport(TypedDict):
    split: str
    provider: str
    model: str
    prompt_version: str
    conversation_count: int
    turn_count: int
    turn_accuracy: float
    conversation_accuracy: float
    turn_results: list[TurnEvalResult]


def normalize_answer(value: float | str | int | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip().replace(",", "").replace("$", "")
    if not text:
        return None
    if text.endswith("%"):
        try:
            return float(text[:-1]) / 100.0
        except ValueError:
            return None
    try:
        return float(text)
    except ValueError:
        return None


def answers_match(predicted: TurnResult, expected: float | str) -> bool:
    pred_value = predicted.numeric_value
    if pred_value is None:
        pred_value = normalize_answer(predicted.answer)

    expected_value = normalize_answer(expected)
    if pred_value is not None and expected_value is not None:
        return math.isclose(pred_value, expected_value, rel_tol=ANSWER_REL_TOL, abs_tol=ANSWER_ABS_TOL)

    return predicted.answer.strip().lower() == str(expected).strip().lower()


def evaluate_split(
    split: str = "dev",
    sample_size: int | None = None,
    provider: str | None = None,
    model: str | None = None,
    prompt_version: str | None = None,
) -> EvalReport:
    resolved_provider = provider or LLM_PROVIDER
    resolved_model = model or get_model(resolved_provider)
    resolved_sample_size = sample_size if sample_size is not None else EVAL_SAMPLE_SIZE
    records = get_split_records(split, resolved_sample_size)

    turn_results: list[TurnEvalResult] = []
    conversations_correct = 0

    for index, record in enumerate(records, start=1):
        print(f"Evaluating record {index}/{len(records)}: {record.id}")
        history: list[TurnResult] = []
        record_results: list[bool] = []
        all_correct = True

        for turn_index, question in enumerate(record.dialogue.conv_questions):
            expected = record.dialogue.executed_answers[turn_index]
            try:
                predicted = answer_turn(
                    record,
                    question,
                    history,
                    provider=resolved_provider,
                    model=resolved_model,
                    prompt_version=prompt_version,
                )
                history.append(predicted)
                correct = answers_match(predicted, expected)
            except Exception as exc:
                print(f"Failed {record.id} turn {turn_index}: {exc}")
                predicted = TurnResult(
                    question=question,
                    answer="ERROR",
                    numeric_value=None,
                    reasoning=str(exc),
                )
                correct = False
                all_correct = False

            if not correct:
                all_correct = False

            turn_results.append(
                {
                    "record_id": record.id,
                    "turn_index": turn_index,
                    "question": question,
                    "expected": expected,
                    "predicted": predicted,
                    "correct": correct,
                }
            )
            record_results.append(correct)

        if record_results and all(item for item in record_results):
            conversations_correct += 1

    turn_count = len(turn_results)
    turn_accuracy = sum(item["correct"] for item in turn_results) / turn_count if turn_count else 0.0
    conversation_count = len(records)
    conversation_accuracy = conversations_correct / conversation_count if conversation_count else 0.0

    return {
        "split": split,
        "provider": resolved_provider,
        "model": resolved_model,
        "prompt_version": prompt_version or PROMPT_VERSION,
        "conversation_count": conversation_count,
        "turn_count": turn_count,
        "turn_accuracy": turn_accuracy,
        "conversation_accuracy": conversation_accuracy,
        "turn_results": turn_results,
    }


def save_results(report: EvalReport) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now(UTC).isoformat().replace(":", "-")
    model_name = report["model"].replace("/", "-")
    path = os.path.join(
        RESULTS_DIR,
        f"{report['provider']}_{model_name}_{timestamp}.json",
    )

    summary: dict[str, Any] = {
        "metadata": {
            "timestamp": datetime.now(UTC).isoformat(),
            "split": report["split"],
            "provider": report["provider"],
            "model": report["model"],
            "prompt_version": report["prompt_version"],
            "conversation_count": report["conversation_count"],
            "turn_count": report["turn_count"],
        },
        "summary": {
            "turn_accuracy": report["turn_accuracy"],
            "conversation_accuracy": report["conversation_accuracy"],
        },
        "failures": [
            {
                "record_id": item["record_id"],
                "turn_index": item["turn_index"],
                "question": item["question"],
                "expected": item["expected"],
                "predicted": item["predicted"].answer,
            }
            for item in report["turn_results"]
            if not item["correct"]
        ][:25],
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved results to {path}")
    return path


class CompareReport(EvalReport):
    output_path: str


def compare_providers(split: str = "dev", sample_size: int | None = None) -> list[CompareReport]:
    results: list[CompareReport] = []
    for provider in ["openai", "anthropic"]:
        print(f"\n=== Evaluating {provider} ===")
        report = evaluate_split(split=split, sample_size=sample_size, provider=provider)
        output_path = save_results(report)
        compare_report: CompareReport = {**report, "output_path": output_path}
        results.append(compare_report)
        print(
            f"{provider}: turn_acc={report['turn_accuracy']:.2%} "
            f"conv_acc={report['conversation_accuracy']:.2%}"
        )
    return results
