"""Leakage-safe splitting and task-specific JSONL exports."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _components(manuals: list[dict[str, Any]], records: list[dict[str, Any]]) -> dict[str, str]:
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
    for record in records:
        manual_ids = [document["manual_id"] for document in record.get("source_documents", [])]
        for manual_id in manual_ids[1:]:
            if manual_ids[0] in parent and manual_id in parent:
                union(manual_ids[0], manual_id)
    return {manual_id: find(manual_id) for manual_id in parent}


def assign_splits(
    records: list[dict[str, Any]],
    manuals: list[dict[str, Any]],
    train: float,
    validation: float,
    seed: str,
) -> None:
    """Assign whole amendment-connected manual groups to one split."""
    components = _components(manuals, records)
    split_by_component: dict[str, str] = {}
    for component in set(components.values()):
        value = int(hashlib.sha256(f"{seed}:{component}".encode()).hexdigest()[:12], 16)
        fraction = value / float(16**12)
        split_by_component[component] = "train" if fraction < train else "validation" if fraction < train + validation else "test"
    for record in records:
        manual_ids = [document["manual_id"] for document in record.get("source_documents", [])] or [record["manual_id"]]
        record["split"] = split_by_component[components[manual_ids[0]]]


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_manifest(output_dir: Path, manifest: dict[str, Any]) -> None:
    """Atomically write the terminal or in-progress run manifest."""
    path = output_dir / "manifest.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def export_records(
    records: list[dict[str, Any]],
    manuals: list[dict[str, Any]],
    output_dir: Path,
    run_id: str,
) -> dict[str, int]:
    """Write canonical and task-specific datasets plus their manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _write(output_dir / "canonical.jsonl", records)
    qa, cot, rag, evaluation = [], [], [], []
    cross_qa, cross_cot = [], []
    for row in records:
        manual_ids = [document["manual_id"] for document in row.get("source_documents", [])] or [row["manual_id"]]
        provenance = {
            "record_id": row["record_id"],
            "split": row["split"],
            "task": row["task"],
            "persona": row["persona"],
            "task_type": row["task_type"],
            "manual_ids": manual_ids,
            "source_chunk_ids": row["source_chunk_ids"],
        }
        qa_row = {
            **provenance,
            "messages": [
                {"role": "user", "content": row["question"]},
                {"role": "assistant", "content": row["answer"]},
            ],
        }
        is_cross = row["task_type"].startswith("cross_document_")
        if row["task_type"] == "qa":
            qa.append(qa_row)
        elif row["task_type"] == "cross_document_qa":
            cross_qa.append(
                {
                    **qa_row,
                    "relationship_type": row["relationship_type"],
                    "source_bundle_id": row["source_bundle_id"],
                }
            )
        if row["task_type"] in {"qa_cot", "cross_document_qa_cot"}:
            rationale = "\n".join(f"{index}. {step['statement']}" for index, step in enumerate(row["reasoning_steps"], 1))
            cot_row = {
                **provenance,
                "messages": [
                    {"role": "user", "content": row["question"]},
                    {
                        "role": "assistant",
                        "content": f"Rationale:\n{rationale}\n\nAnswer: {row['answer']}",
                    },
                ],
            }
            if row["task_type"] == "qa_cot":
                cot.append(cot_row)
            elif is_cross:
                cross_cot.append(
                    {
                        **cot_row,
                        "relationship_type": row["relationship_type"],
                        "source_bundle_id": row["source_bundle_id"],
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
                "task": row["task"],
                "persona": row["persona"],
            }
        )
    _write(output_dir / "qa_sft.jsonl", qa)
    _write(output_dir / "qa_cot_sft.jsonl", cot)
    _write(output_dir / "cross_document_qa_sft.jsonl", cross_qa)
    _write(output_dir / "cross_document_qa_cot_sft.jsonl", cross_cot)
    _write(output_dir / "rag.jsonl", rag)
    _write(output_dir / "eval.jsonl", evaluation)
    counts = defaultdict(int)
    for task_type in ("qa", "qa_cot", "cross_document_qa", "cross_document_qa_cot"):
        counts[f"task_{task_type}"] = 0
    for row in records:
        counts[f"split_{row['split']}"] += 1
        counts[f"task_{row['task_type']}"] += 1
        counts[f"procurement_task_{row['task']}"] += 1
        counts[f"persona_{row['persona']}"] += 1
        counts[f"answerable_{str(bool(row['answerable'])).lower()}"] += 1
        for manual_id in [document["manual_id"] for document in row.get("source_documents", [])] or [row["manual_id"]]:
            counts[f"manual_{manual_id}"] += 1
    stats = {"records": len(records), **dict(sorted(counts.items()))}
    write_manifest(
        output_dir,
        {
            "run_id": run_id,
            "status": "partial",
            "statistics": stats,
            "manuals": manuals,
        },
    )
    return stats
