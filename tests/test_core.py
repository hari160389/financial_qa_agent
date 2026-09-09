from src.agent.tools import lookup_table_value, run_tool
from src.data.dataset import get_record, list_record_ids, load_dataset
from src.data.document import format_table
from src.data.models import Document


def test_dataset_loads() -> None:
    data = load_dataset()
    assert "train" in data
    assert "dev" in data
    assert len(list_record_ids("dev")) > 0


def test_get_record() -> None:
    record_id = list_record_ids("dev")[0]
    record = get_record(record_id)
    assert record.id == record_id


def test_format_table() -> None:
    record = get_record(list_record_ids("dev")[0])
    rendered = format_table(record.doc)
    assert "|" in rendered


def test_lookup_table_value() -> None:
    doc = Document(
        table={
            "2009": {"net income": 100.0},
            "2008": {"net income": 80.0},
        }
    )
    result = lookup_table_value(doc, "2009", "net income")
    assert result["numeric_value"] == 100.0


def test_calculate_tool() -> None:
    doc = Document(table={"2009": {"net income": 100.0}})
    result = run_tool(doc, "calculate", {"operation": "subtract", "left": 10, "right": 3})
    assert result["result"] == 7.0
