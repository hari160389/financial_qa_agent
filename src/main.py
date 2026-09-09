from __future__ import annotations

import sys

from src.agent import answer_turn
from src.agent.schemas import TurnResult
from src.config.constants import LLM_PROVIDER, PROMPT_VERSION, get_model
from src.data.dataset import get_record, list_record_ids
from src.evaluation import compare_providers, evaluate_split, save_results


def print_usage() -> None:
    print("Usage:")
    print("  python -m src.main list-records [split] [limit]")
    print("  python -m src.main chat <record_id>")
    print("  python -m src.main evaluate [split] [sample_size]")
    print("  python -m src.main compare-eval [split] [sample_size]")
    print("  python -m src.main serve [host] [port]")


def cmd_list_records(split: str = "dev", limit: int = 10) -> None:
    ids = list_record_ids(split)[:limit]
    print(f"Records in {split} (showing {len(ids)}):")
    for record_id in ids:
        print(record_id)


def cmd_chat(record_id: str) -> None:
    record = get_record(record_id)
    history: list[TurnResult] = []
    print(f"Loaded: {record.id}")
    print("Type your questions. Use exit or quit to stop.\n")

    while True:
        question = input(">>> ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        turn = answer_turn(record, question, history)
        history.append(turn)
        print(f"assistant: {turn.answer}")
        if turn.reasoning:
            print(f"reasoning: {turn.reasoning}")


def cmd_evaluate(split: str = "dev", sample_size: str | int | None = None) -> None:
    resolved_sample_size = int(sample_size) if sample_size is not None else None
    print(
        f"Evaluating split={split} provider={LLM_PROVIDER} "
        f"model={get_model()} prompt={PROMPT_VERSION}"
    )
    report = evaluate_split(split=split, sample_size=resolved_sample_size)
    save_results(report)
    print(f"Turn accuracy: {report['turn_accuracy']:.2%}")
    print(f"Conversation accuracy: {report['conversation_accuracy']:.2%}")


def cmd_compare_eval(split: str = "dev", sample_size: int = 75) -> None:
    compare_providers(split=split, sample_size=int(sample_size))


def cmd_serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run("src.api.app:app", host=host, port=int(port), reload=False)


def main() -> None:
    if len(sys.argv) < 2:
        print_usage()
        return

    command = sys.argv[1]

    if command == "list-records":
        split = sys.argv[2] if len(sys.argv) > 2 else "dev"
        limit = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        cmd_list_records(split, limit)
    elif command == "chat":
        if len(sys.argv) < 3:
            print("Missing record_id")
            return
        cmd_chat(sys.argv[2])
    elif command == "evaluate":
        split = sys.argv[2] if len(sys.argv) > 2 else "dev"
        sample_size = sys.argv[3] if len(sys.argv) > 3 else None
        cmd_evaluate(split, sample_size)
    elif command == "compare-eval":
        split = sys.argv[2] if len(sys.argv) > 2 else "dev"
        sample_size = sys.argv[3] if len(sys.argv) > 3 else 75
        cmd_compare_eval(split, sample_size)
    elif command == "serve":
        host = sys.argv[2] if len(sys.argv) > 2 else "127.0.0.1"
        port = sys.argv[3] if len(sys.argv) > 3 else 8000
        cmd_serve(host, port)
    else:
        print(f"Unknown command: {command}")
        print_usage()


if __name__ == "__main__":
    main()
