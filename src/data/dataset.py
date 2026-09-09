from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.config.constants import DATASET_PATH
from src.data.models import ConvFinQARecord

_dataset: dict[str, list[dict[str, Any]]] | None = None


def load_dataset(path: str | Path | None = None) -> dict[str, list[dict[str, Any]]]:
    global _dataset
    if _dataset is not None and path is None:
        return _dataset

    dataset_path = Path(path or DATASET_PATH)
    with dataset_path.open(encoding="utf-8") as f:
        _dataset = json.load(f)
    return _dataset


def get_record(record_id: str, split: str | None = None) -> ConvFinQARecord:
    data = load_dataset()
    if split:
        records = data.get(split, [])
        for record in records:
            if record["id"] == record_id:
                return ConvFinQARecord.from_dict(record)
        raise KeyError(f"Record not found in {split}: {record_id}")

    for records in data.values():
        for record in records:
            if record["id"] == record_id:
                return ConvFinQARecord.from_dict(record)
    raise KeyError(f"Record not found: {record_id}")


def list_record_ids(split: str = "dev") -> list[str]:
    data = load_dataset()
    return [record["id"] for record in data.get(split, [])]


def get_split_records(split: str = "dev", sample_size: int | None = None) -> list[ConvFinQARecord]:
    records = [ConvFinQARecord.from_dict(record) for record in load_dataset().get(split, [])]
    if sample_size:
        return records[:sample_size]
    return records
