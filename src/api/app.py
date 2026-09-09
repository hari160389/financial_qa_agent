from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from src.agent import answer_turn
from src.agent.schemas import TurnResult
from src.config.constants import LLM_PROVIDER, PROMPT_VERSION, get_model
from src.data.dataset import get_record, list_record_ids
from src.evaluation import evaluate_split

app = FastAPI(title="ConvFinQA API", version="0.1.0")
_sessions: dict[str, list[TurnResult]] = {}


class ChatRequest(BaseModel):
    record_id: str
    question: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    record_id: str
    session_id: str
    answer: str
    numeric_value: float | None = None
    reasoning: str | None = None


class EvaluateResponse(BaseModel):
    split: str
    provider: str
    model: str
    prompt_version: str
    conversation_count: int
    turn_count: int
    turn_accuracy: float
    conversation_accuracy: float


class RecordsResponse(BaseModel):
    split: str
    count: int
    record_ids: list[str]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/records", response_model=RecordsResponse)
def records(split: str = "dev", limit: int = 20) -> RecordsResponse:
    ids = list_record_ids(split)[:limit]
    return RecordsResponse(split=split, count=len(ids), record_ids=ids)


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        record = get_record(request.record_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    key = f"{request.session_id}:{request.record_id}"
    history = _sessions.get(key, [])
    turn = answer_turn(record, request.question, history)
    history.append(turn)
    _sessions[key] = history

    return ChatResponse(
        record_id=request.record_id,
        session_id=request.session_id,
        answer=turn.answer,
        numeric_value=turn.numeric_value,
        reasoning=turn.reasoning,
    )


@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate(split: str = "dev", sample_size: int = Query(default=75, ge=1)) -> EvaluateResponse:
    report = evaluate_split(split=split, sample_size=sample_size)
    return EvaluateResponse(
        split=report["split"],
        provider=LLM_PROVIDER,
        model=get_model(),
        prompt_version=PROMPT_VERSION,
        conversation_count=report["conversation_count"],
        turn_count=report["turn_count"],
        turn_accuracy=report["turn_accuracy"],
        conversation_accuracy=report["conversation_accuracy"],
    )
