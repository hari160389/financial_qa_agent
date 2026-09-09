from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

DATASET_PATH = os.getenv("DATASET_PATH", "data/convfinqa_dataset.json")
RESULTS_DIR = os.getenv("RESULTS_DIR", "data/results")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "") or None
PROMPT_VERSION = os.getenv("PROMPT_VERSION", "v3")

MAX_TOOL_ROUNDS = int(os.getenv("MAX_TOOL_ROUNDS", "12"))
EVAL_SAMPLE_SIZE = int(os.getenv("EVAL_SAMPLE_SIZE", "75"))
ANSWER_REL_TOL = float(os.getenv("ANSWER_REL_TOL", "0.001"))
ANSWER_ABS_TOL = float(os.getenv("ANSWER_ABS_TOL", "0.5"))

DEFAULT_MODELS = {
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-5-20250929",
}

PROMPT_V3 = """You are a financial analyst assistant answering questions about a single financial document.

Use chain-of-thought reasoning before every tool call:
1. PLAN: metric, column/year, prior turns needed
2. ACT: use list_table_structure, lookup_table_value, or calculate
3. VERIFY: correct row, column, operation
4. SUBMIT: call submit_answer with answer_text, numeric_value, reasoning

Multi-turn rules:
- "what about in 2008?" -> same metric, new column
- "what is the difference?" -> subtract recent values from context
- percentage change -> numeric_value as decimal (0.141 for 14.1%)

Rules:
- Use ONLY the document, table, and prior turns
- Always use tools for numbers
- You MUST call submit_answer to finish
"""

PROMPTS = {
    "v1": "Answer using tools only. Call submit_answer when done.",
    "v2": PROMPT_V3.replace("Multi-turn rules:", "Follow PLAN -> ACT -> VERIFY -> SUBMIT."),
    "v3": PROMPT_V3,
}


def get_model(provider: str | None = None) -> str:
    resolved_provider = provider or LLM_PROVIDER
    if LLM_MODEL:
        return LLM_MODEL
    return DEFAULT_MODELS[resolved_provider]


def get_api_key(provider: str | None = None) -> str:
    resolved_provider = provider or LLM_PROVIDER
    if resolved_provider == "openai":
        return OPENAI_API_KEY
    if resolved_provider == "anthropic":
        return ANTHROPIC_API_KEY
    return ""


def get_system_prompt(version: str | None = None) -> str:
    version = version or PROMPT_VERSION
    return PROMPTS.get(version, PROMPT_V3)
