"""Leakage-safe splitting and task-specific JSONL exports."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _components(manuals: list[dict[str, Any]]) -> dict[str, str]:
    parent = {manual["manual_id"]: manual["manual_id"] for manual in manuals}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        left, right = find(left), find(right)
        if left != right:
            parent[max(left, right)] = min(left, right)

    for manual in manuals:
        for target in manual.get("amends", []):
            if target in parent:
                union(manual["manual_id"], target)
    return {manual_id: find(manual_id) for manual_id in parent}


def assign_splits(
    records: list[dict[str, Any]],
    manuals: list[dict[str, Any]],
    train: float,
    validation: float,
    seed: str,
) -> None:
    """Assign whole amendment-connected manual groups to one split."""
    components = _components(manuals)
    split_by_component: dict[str, str] = {}
    for component in set(components.values()):
        value = int(hashlib.sha256(f"{seed}:{component}".encode()).hexdigest()[:12], 16)
        fraction = value / float(16**12)
        split_by_component[component] = (
            "train"
            if fraction < train
            else "validation"
            if fraction < train + validation
            else "test"
        )
    for record in records:
        record["split"] = split_by_component[components[record["manual_id"]]]


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def export_records(
    records: list[dict[str, Any]], manuals: list[dict[str, Any]], output_dir: Path
) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write(output_dir / "canonical.jsonl", records)
    qa, cot, rag, evaluation = [], [], [], []
    for row in records:
        provenance = {
            "record_id": row["record_id"],
            "split": row["split"],
            "manual_id": row["manual_id"],
            "source_chunk_ids": row["source_chunk_ids"],
        }
        qa.append(
            {
                **provenance,
                "messages": [
                    {"role": "user", "content": row["question"]},
                    {"role": "assistant", "content": row["answer"]},
                ],
            }
        )
        if row["task_type"] == "qa_cot":
            rationale = "\n".join(
                f"{index}. {step['statement']}"
                for index, step in enumerate(row["reasoning_steps"], 1)
            )
            cot.append(
                {
                    **provenance,
                    "messages": [
                        {"role": "user", "content": row["question"]},
                        {
                            "role": "assistant",
                            "content": f"Rationale:\n{rationale}\n\nAnswer: {row['answer']}",
                        },
                    ],
                }
            )
        rag.append(
            {
                **provenance,
                "question": row["question"],
                "contexts": row["evidence"],
                "answer": row["answer"],
                "answerable": row["answerable"],
            }
        )
        evaluation.append(
            {
                **provenance,
                "question": row["question"],
                "reference_answer": row["answer"],
                "evidence": row["evidence"],
                "question_type": row["question_type"],
            }
        )
    _write(output_dir / "qa_sft.jsonl", qa)
    _write(output_dir / "qa_cot_sft.jsonl", cot)
    _write(output_dir / "rag.jsonl", rag)
    _write(output_dir / "eval.jsonl", evaluation)
    counts = defaultdict(int)
    for row in records:
        counts[f"split_{row['split']}"] += 1
        counts[f"task_{row['task_type']}"] += 1
    stats = {"records": len(records), **dict(sorted(counts.items()))}
    (output_dir / "manifest.json").write_text(
        json.dumps({"statistics": stats, "manuals": manuals}, indent=2), encoding="utf-8"
    )
    return stats
