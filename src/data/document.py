from __future__ import annotations

from src.data.models import Document


def format_table(doc: Document) -> str:
    table = doc.table
    columns = list(table.keys())
    if not columns:
        return "_Empty table_"

    row_labels: list[str] = []
    seen: set[str] = set()
    for column in columns:
        for row in table[column]:
            if row not in seen:
                seen.add(row)
                row_labels.append(row)

    header = "| Metric | " + " | ".join(columns) + " |"
    separator = "| --- | " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for label in row_labels:
        cells = [str(table[column].get(label, "")) for column in columns]
        rows.append("| " + label + " | " + " | ".join(cells) + " |")
    return "\n".join([header, separator, *rows])


def format_document(doc: Document) -> str:
    return "\n".join(
        [
            "## Narrative (before table)",
            doc.pre_text.strip() or "_None_",
            "",
            "## Table",
            format_table(doc),
            "",
            "## Narrative (after table)",
            doc.post_text.strip() or "_None_",
        ]
    )


def list_table_structure(doc: Document) -> dict[str, list[str]]:
    table = doc.table
    columns = list(table.keys())
    row_labels: list[str] = []
    seen: set[str] = set()
    for column in columns:
        for row in table[column]:
            if row not in seen:
                seen.add(row)
                row_labels.append(row)
    return {"columns": columns, "rows": row_labels}
