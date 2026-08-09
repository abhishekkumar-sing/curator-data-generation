"""Leakage-safe splitting and task-specific JSONL exports."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from jsonl_io import write_jsonl_rows
from provenance import build_reasoning_graph, leakage_audit
from validation import is_extractive_answer, question_opener_key


def question_opener_diversity(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Report how concentrated accepted questions are on one opening phrase.

    This does not reject anything; `validation.enforce_question_opener_diversity`
    already does the active rejection pre-judge. This is a post-export
    visibility signal in case residual concentration remains after judging
    and deduplication, which the pre-judge pass cannot see.
    """
    opener_counts: dict[str, int] = defaultdict(int)
    for row in records:
        opener_counts[question_opener_key(str(row.get("question", "")))] += 1
    if not records:
        return {
            "unique_openers": 0,
            "top_opener": "",
            "top_opener_count": 0,
            "top_opener_share": 0.0,
        }
    top_opener, top_count = max(opener_counts.items(), key=lambda item: item[1])
    return {
        "unique_openers": len(opener_counts),
        "top_opener": top_opener,
        "top_opener_count": top_count,
        "top_opener_share": round(top_count / len(records), 4),
    }


def answer_style_diversity(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Report full-answer and partial evidence-span copying."""
    eligible = [row for row in records if row.get("task_type") in {"qa", "qa_cot"}]
    extractive = sum(is_extractive_answer(row) for row in eligible)
    eight_gram_coverages: list[float] = []
    copied_sentence_fractions: list[float] = []
    for row in eligible:
        answer_tokens = re.findall(r"[a-z0-9]+", str(row.get("answer", "")).casefold())
        evidence_text = " ".join(
            str(item.get("quote", "")) for item in row.get("evidence", [])
        )
        evidence_tokens = re.findall(r"[a-z0-9]+", evidence_text.casefold())
        evidence_normalized = " ".join(evidence_tokens)
        grams = [
            " ".join(answer_tokens[index : index + 8])
            for index in range(max(0, len(answer_tokens) - 7))
        ]
        eight_gram_coverages.append(
            sum(gram in evidence_normalized for gram in grams) / len(grams)
            if grams
            else 0.0
        )
        sentences = [
            " ".join(re.findall(r"[a-z0-9]+", sentence.casefold()))
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", str(row.get("answer", "")))
            if len(re.findall(r"[a-z0-9]+", sentence.casefold())) >= 5
        ]
        copied_sentence_fractions.append(
            sum(sentence in evidence_normalized for sentence in sentences) / len(sentences)
            if sentences
            else 0.0
        )
    return {
        "records_evaluated": len(eligible),
        "extractive_answers": extractive,
        "non_extractive_answers": len(eligible) - extractive,
        "extractive_answer_share": round(extractive / len(eligible), 4) if eligible else 0.0,
        "mean_eight_gram_source_coverage": (
            round(sum(eight_gram_coverages) / len(eight_gram_coverages), 4)
            if eight_gram_coverages
            else 0.0
        ),
        "mean_copied_sentence_fraction": (
            round(sum(copied_sentence_fractions) / len(copied_sentence_fractions), 4)
            if copied_sentence_fractions
            else 0.0
        ),
    }


_RELEVANCE_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does",
        "for", "from", "has", "have", "how", "if", "in", "into", "is", "it",
        "its", "may", "must", "not", "of", "on", "or", "shall", "should",
        "that", "the", "their", "there", "this", "to", "under", "was", "were",
        "what", "when", "where", "which", "who", "whom", "why", "will",
        "with", "would",
        # Source-citation boilerplate that dominates question wording (see
        # T8's opener-collapse finding) without being the fact actually
        # being asked about; without stripping these, term overlap is
        # systematically diluted regardless of true relevance.
        "according", "accordance", "per", "manual", "manuals", "procurement",
        "goods", "works", "services", "consultancy", "issued", "government",
        "india", "updated", "gazette", "notification", "order", "circular",
        "ministry", "department",
    }
)
_RELEVANCE_WORD = re.compile(r"[a-z][a-z0-9_-]{2,}")


def _relevance_terms(text: str) -> set[str]:
    return {word for word in _RELEVANCE_WORD.findall(str(text).casefold()) if word not in _RELEVANCE_STOPWORDS}


def question_answer_relevance_diagnostics(
    records: list[dict[str, Any]],
    *,
    near_zero_overlap_ratio: float = 0.05,
    sample_size: int = 20,
) -> dict[str, Any]:
    """Report how much question-vs-answer key-term overlap accepted records have.

    Finding V3: nothing deterministic checks that an answer is actually
    about what the question asked; that is entirely delegated to the judge's
    self-reported `relevant` boolean. This is a **non-blocking, aggregate**
    diagnostic, not a per-record hard-reject gate and not a new
    `validate_record` reason code -- real-data validation against both
    `qa-qacot-full-002` and `qa-qacot-full-003` found that per-record token
    overlap between a question and its answer+evidence is not precise enough
    to safely reject on individually: even at the strictest possible cutoff
    (zero shared key terms), the large majority of "flagged" records in this
    corpus were genuinely correct, judge-accepted answers -- terse
    direct-fact/tabular extractions (e.g. question "what is the validity
    period of a Proprietary Article Certificate?" / answer "Valid for the
    Current Financial Year") legitimately do not restate the question's
    vocabulary at all. Duplicating the judge's `relevant` boolean with an
    unreliable lexical proxy would create false rejections, which the task
    explicitly warns against; reporting the aggregate distribution instead
    gives real visibility (a corpus-level regression to a lot of near-zero-
    overlap records would be visible here) without trusting any single
    record's flag. `flagged_sample` is included for optional human spot-
    checking, not as a reliable per-record verdict.
    """
    eligible = [row for row in records if row.get("task_type") in {"qa", "qa_cot"} and row.get("answerable", True)]
    ratios: list[float] = []
    flagged: list[dict[str, Any]] = []
    for row in eligible:
        question_terms = _relevance_terms(row.get("question", ""))
        if not question_terms:
            continue
        support_text = " ".join(
            [str(row.get("answer", ""))] + [str(item.get("quote", "")) for item in row.get("evidence", [])]
        )
        support_terms = _relevance_terms(support_text)
        overlap_ratio = len(question_terms & support_terms) / len(question_terms)
        ratios.append(overlap_ratio)
        if overlap_ratio <= near_zero_overlap_ratio:
            flagged.append(
                {
                    "record_id": row.get("record_id", ""),
                    "overlap_ratio": round(overlap_ratio, 4),
                }
            )
    ratios.sort()
    return {
        "records_evaluated": len(ratios),
        "near_zero_overlap_ratio_threshold": near_zero_overlap_ratio,
        "flagged_near_zero_overlap": len(flagged),
        "flagged_share": round(len(flagged) / len(ratios), 4) if ratios else 0.0,
        "mean_overlap_ratio": (round(sum(ratios) / len(ratios), 4) if ratios else 0.0),
        "median_overlap_ratio": (round(ratios[len(ratios) // 2], 4) if ratios else 0.0),
        "flagged_sample": sorted(flagged, key=lambda item: item["overlap_ratio"])[:sample_size],
    }


def categorical_diversity(
    records: list[dict[str, Any]],
    field: str,
) -> dict[str, Any]:
    """Summarize category concentration and normalized entropy."""
    counts = Counter(str(row.get(field, "")).strip() or "missing" for row in records)
    if not records:
        return {
            "counts": {},
            "top_category": "",
            "top_count": 0,
            "top_share": 0.0,
            "normalized_entropy": 0.0,
        }
    top_category, top_count = max(counts.items(), key=lambda item: item[1])
    probabilities = [count / len(records) for count in counts.values()]
    entropy = -sum(probability * math.log(probability) for probability in probabilities)
    normalized_entropy = entropy / math.log(len(counts)) if len(counts) > 1 else 0.0
    return {
        "counts": dict(sorted(counts.items())),
        "top_category": top_category,
        "top_count": top_count,
        "top_share": round(top_count / len(records), 4),
        "normalized_entropy": round(normalized_entropy, 4),
    }


def answer_length_statistics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Report answer lengths globally and by deterministic format."""
    by_format: dict[str, list[int]] = defaultdict(list)
    for row in records:
        count = len(re.findall(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)?", str(row.get("answer", ""))))
        by_format[str(row.get("answer_format", "missing"))].append(count)

    def summarize(values: list[int]) -> dict[str, Any]:
        ordered = sorted(values)
        if not ordered:
            return {"records": 0, "minimum": 0, "median": 0, "p90": 0, "maximum": 0}
        def at(share: float) -> int:
            return ordered[round((len(ordered) - 1) * share)]
        return {
            "records": len(ordered),
            "minimum": ordered[0],
            "median": at(0.5),
            "p90": at(0.9),
            "maximum": ordered[-1],
        }

    all_values = [value for values in by_format.values() for value in values]
    return {
        "overall": summarize(all_values),
        "by_format": {
            answer_format: summarize(values)
            for answer_format, values in sorted(by_format.items())
        },
    }


def batch_efficiency_stats(
    generated_records: int,
    accepted_records: int,
    removed_by_reason: dict[str, int],
) -> dict[str, Any]:
    """Report how much of one generated batch survives to the accepted export.

    `generated_records` should count every parsed generation candidate before
    judging, deduplication, or diversity-cap removal; `accepted_records` is
    the final exported pool. The gap between the two is already tracked
    elsewhere in the manifest under several separately-named removal counts
    (near-duplicates, opener/type/style/extractive overrepresentation); this
    reports the generation-to-acceptance ratio and an overall removal rate so
    a reader doesn't have to manually diff several scattered manifest fields
    to see where generated candidates went.
    """
    total_removed = sum(removed_by_reason.values())
    return {
        "generated_records": generated_records,
        "accepted_records": accepted_records,
        "generation_to_acceptance_ratio": (round(accepted_records / generated_records, 4) if generated_records else 0.0),
        "removed_by_reason": dict(sorted(removed_by_reason.items())),
        "total_removed": total_removed,
        "removal_rate": (round(total_removed / generated_records, 4) if generated_records else 0.0),
    }


def assert_unique_record_ids(
    rows: list[dict[str, Any]],
    *,
    key: str = "record_id",
    dataset_name: str = "records",
) -> None:
    """Fail closed when stable record identities are absent or duplicated."""
    missing = [index for index, row in enumerate(rows) if not str(row.get(key, "")).strip()]
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        if str(row.get(key, "")).strip():
            counts[str(row[key])] += 1
    duplicates = sorted(record_id for record_id, count in counts.items() if count > 1)
    issues = []
    if missing:
        issues.append(f"missing {key} at row indexes {missing}")
    if duplicates:
        issues.append(f"duplicate {key} values: {duplicates}")
    if issues:
        raise ValueError(f"Invalid {dataset_name}: {'; '.join(issues)}")


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


def drafting_manual_documents(
    chunk_ids: list[str],
    chunk_manuals: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    """Resolve drafting `manual_chunk_ids` to `source_documents`.

    `assign_splits`/`leakage_audit` already know how to read `source_documents`.
    """
    documents: list[dict[str, str]] = []
    seen: set[str] = set()
    for chunk_id in chunk_ids:
        info = chunk_manuals.get(str(chunk_id))
        if info is None:
            raise ValueError(f"Drafting record references unknown corpus chunk: {chunk_id}")
        manual_id = str(info.get("manual_id", ""))
        if manual_id in seen:
            continue
        seen.add(manual_id)
        documents.append(dict(info))
    return documents


def assign_drafting_splits(
    records: list[dict[str, Any]],
    manuals: list[dict[str, Any]],
    chunk_manuals: dict[str, dict[str, str]],
    train: float,
    validation: float,
    seed: str,
    manual_folds: dict[str, str] | None,
) -> None:
    """Give accepted drafting records the same split guarantee QA records get.

    Drafting rows do not naturally carry `source_documents`/`record_id`/`question` —
    the fields `assign_splits`/`leakage_audit` read. Rather than reimplement either
    gate for a second schema, this attaches those fields (derived from
    `manual_chunk_ids`, the only field that actually identifies which manuals a
    drafting record draws from) and then calls the existing, unmodified
    `assign_splits`. A drafting record's split is therefore always the split of the
    manual(s) it cites — the identical rule QA/cross-document records already use.
    """
    for record in records:
        chunk_ids = [str(chunk_id) for chunk_id in record.get("manual_chunk_ids", [])]
        record["record_id"] = record["id"]
        record["question"] = record["instruction"]
        record["source_documents"] = drafting_manual_documents(chunk_ids, chunk_manuals)
        record["source_chunk_ids"] = chunk_ids
    assign_splits(records, manuals, train, validation, seed, manual_folds=manual_folds)


def assign_splits(
    records: list[dict[str, Any]],
    manuals: list[dict[str, Any]],
    train: float,
    validation: float,
    seed: str,
    manual_folds: dict[str, str] | None = None,
) -> None:
    """Assign whole connected components near configured record targets."""
    if manual_folds is not None:
        for record in records:
            manual_ids = [
                str(document["manual_id"])
                for document in record.get("source_documents", [])
            ] or [str(record["manual_id"])]
            folds = {manual_folds[manual_id] for manual_id in manual_ids}
            if len(folds) != 1:
                raise ValueError(
                    "One record spans explicit manual folds: "
                    f"{record.get('record_id')}: {sorted(folds)}"
                )
            record["split"] = folds.pop()
        return
    components = _components(manuals, records)
    test = 1.0 - train - validation
    fractions = {"train": train, "validation": validation, "test": test}
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        manual_ids = [
            document["manual_id"]
            for document in record.get("source_documents", [])
        ] or [record["manual_id"]]
        counts[components[manual_ids[0]]] += 1
    targets = {
        split: len(records) * fraction for split, fraction in fractions.items()
    }
    assigned = {split: 0 for split in fractions}
    split_by_component: dict[str, str] = {}
    populated = sorted(
        counts,
        key=lambda component: (
            -counts[component],
            hashlib.sha256(f"{seed}:{component}".encode()).hexdigest(),
        ),
    )
    for component in populated:
        size = counts[component]

        def allocation_cost(
            split: str,
            component: str = component,
            size: int = size,
        ) -> tuple[float, str]:
            projected = {**assigned, split: assigned[split] + size}
            cost = sum(
                (
                    (projected[name] - targets[name])
                    / max(targets[name], 1.0)
                )
                ** 2
                for name in fractions
            )
            tie = hashlib.sha256(
                f"{seed}:{component}:{split}".encode()
            ).hexdigest()
            return cost, tie

        selected = min(fractions, key=allocation_cost)
        split_by_component[component] = selected
        assigned[selected] += size
    for component in set(components.values()) - set(populated):
        split_by_component[component] = "train"
    for record in records:
        manual_ids = [document["manual_id"] for document in record.get("source_documents", [])] or [record["manual_id"]]
        record["split"] = split_by_component[components[manual_ids[0]]]


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    write_jsonl_rows(path, rows)


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
    assert_unique_record_ids(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_rejected = []
    graph_valid = []
    for row in records:
        graph = build_reasoning_graph(row)
        row["reasoning_graph"] = graph
        if graph["validation"]["passed"]:
            graph_valid.append(row)
        else:
            graph_rejected.append(
                {
                    "record_id": row["record_id"],
                    "issues": graph["validation"]["issues"],
                }
            )
    if graph_rejected:
        _write(output_dir / "reasoning_graph_rejected.jsonl", graph_rejected)
    # A record whose reasoning graph fails structural validation is dropped
    # rather than aborting the whole export: one malformed record must not
    # discard every other accepted record's real generation/judge lineage.
    records[:] = graph_valid
    leakage = leakage_audit(records)
    (output_dir / "leakage_audit.json").write_text(
        json.dumps(leakage, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if not leakage["passed"]:
        raise ValueError("Cross-split leakage detected; see leakage_audit.json")
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
            "citations": row["citations"],
        }
        qa_row = {
            **provenance,
            "messages": [
                {"role": "user", "content": row["question"]},
                {"role": "assistant", "content": row["answer"]},
            ],
        }
        is_cross = row["task_type"].startswith("cross_document_")
        is_train = row["split"] == "train"
        if row["task_type"] == "qa" and is_train:
            qa.append(qa_row)
        elif row["task_type"] == "cross_document_qa" and is_train:
            cross_qa.append(
                {
                    **qa_row,
                    "relationship_type": row["relationship_type"],
                    "source_bundle_id": row["source_bundle_id"],
                }
            )
        if row["task_type"] in {"qa_cot", "cross_document_qa_cot"} and is_train:
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
        if is_train:
            rag.append(
                {
                    **provenance,
                    "question": row["question"],
                    "contexts": row["evidence"],
                    "answer": row["answer"],
                    "answerable": row["answerable"],
                }
            )
        else:
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
    # Every *_sft.jsonl and rag.jsonl file is train-split only; eval.jsonl is
    # validation+test only. canonical.jsonl above retains every split so the
    # split assignment itself stays fully auditable; only the ready-to-use
    # training/evaluation exports must never overlap on record_id.
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
    stats["reasoning_graphs_valid"] = len(records)
    stats["reasoning_graphs_rejected"] = len(graph_rejected)
    stats["leakage_audit_passed"] = leakage["passed"]
    # Explicit per-export-file counts so a train-only/eval-only mismatch
    # against `records` (the full accepted, all-split pool) is visible in
    # the manifest without re-deriving it from the exported files by hand.
    stats["qa_sft_records"] = len(qa)
    stats["qa_cot_sft_records"] = len(cot)
    stats["cross_document_qa_sft_records"] = len(cross_qa)
    stats["cross_document_qa_cot_sft_records"] = len(cross_cot)
    stats["rag_records"] = len(rag)
    stats["eval_records"] = len(evaluation)
    stats["question_opener_diversity"] = question_opener_diversity(records)
    stats["answer_style_diversity"] = answer_style_diversity(records)
    stats["question_answer_relevance_diagnostics"] = question_answer_relevance_diagnostics(records)
    single_document_qa = [
        row for row in records if row.get("task_type") in {"qa", "qa_cot"}
    ]
    stats["question_type_diversity"] = categorical_diversity(
        single_document_qa,
        "question_type",
    )
    stats["question_style_diversity"] = categorical_diversity(
        single_document_qa,
        "question_style",
    )
    stats["answer_format_diversity"] = categorical_diversity(
        single_document_qa,
        "answer_format",
    )
    stats["manual_diversity"] = categorical_diversity(records, "manual_id")
    stats["answer_length"] = answer_length_statistics(single_document_qa)
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
