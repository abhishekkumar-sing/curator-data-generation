"""Stable JSONL presentation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def citations_last(row: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow copy with top-level citations serialized last."""
    if "citations" not in row:
        return row
    return {
        **{key: value for key, value in row.items() if key != "citations"},
        "citations": row["citations"],
    }


def write_jsonl_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write JSONL while enforcing the stable citation-last presentation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(citations_last(row), ensure_ascii=False) + "\n")
