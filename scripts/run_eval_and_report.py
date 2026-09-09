"""Run compare eval and write REPORT.md from results."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import compare_providers


def write_report(results):
    lines = [
        "# ConvFinQA Report",
        "",
        "## Method",
        "",
        "Tool-augmented LLM agent with chain-of-thought prompting (PLAN -> ACT -> VERIFY -> SUBMIT).",
        "Supports OpenAI and Anthropic via a simple provider switch in config/constants.py.",
        "",
        "## Evaluation",
        "",
        f"Dev split, {results[0]['conversation_count']} conversations, prompt {results[0]['prompt_version']}.",
        "",
        "| Provider | Model | Turn Acc | Conv Acc |",
        "|----------|-------|----------|----------|",
    ]

    for report in results:
        lines.append(
            f"| {report['provider']} | {report['model']} | "
            f"{report['turn_accuracy']:.2%} | {report['conversation_accuracy']:.2%} |"
        )

    lines.extend(
        [
            "",
            "## AI Tool Disclosure",
            "",
            "Cursor AI was used to help build and simplify this solution.",
            "",
        ]
    )

    Path("REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote REPORT.md")


def main():
    results = compare_providers(split="dev")
    write_report(results)


if __name__ == "__main__":
    main()
