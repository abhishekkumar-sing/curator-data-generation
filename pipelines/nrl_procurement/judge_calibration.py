"""Calibrate the procurement judge threshold on immutable human reviews."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from review import validate_reviews

CONTRACT_VERSION = "nrl-judge-calibration-v1"
JUDGE_FEATURES = (
    "supported",
    "relevant",
    "preserves_qualifications",
    "authority_correct",
    "reasoning_valid",
    "question_natural",
    "persona_relevant",
    "task_correct",
    "persona_correct",
    "answerability_correct",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_complete_reviews(path: Path, minimum_records: int) -> list[dict[str, Any]]:
    validation = validate_reviews(path, minimum_accepted=0)
    if validation["issues"]:
        raise ValueError(f"Invalid review file {path}: {validation['issues'][:5]}")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) < minimum_records:
        raise ValueError(
            f"Judge calibration requires at least {minimum_records} records in {path}"
        )
    labels = {bool(row["overall_accept"]) for row in rows}
    if labels != {False, True}:
        raise ValueError(f"Judge calibration split {path} needs both label classes")
    for row in rows:
        judge = row.get("record", {}).get("judge", {})
        missing = [feature for feature in JUDGE_FEATURES if feature not in judge]
        if "score" not in judge or missing:
            raise ValueError(
                f"Review {row.get('review_id')} lacks judge features: {missing}"
            )
    return rows


def _prediction(row: dict[str, Any], threshold: int) -> bool:
    judge = row["record"]["judge"]
    return all(bool(judge[feature]) for feature in JUDGE_FEATURES) and int(
        judge["score"]
    ) >= threshold


def calibration_metrics(
    rows: list[dict[str, Any]], threshold: int
) -> dict[str, Any]:
    """Return confusion counts and standard binary classification metrics."""
    tp = fp = tn = fn = 0
    for row in rows:
        predicted = _prediction(row, threshold)
        actual = bool(row["overall_accept"])
        tp += predicted and actual
        fp += predicted and not actual
        tn += not predicted and not actual
        fn += not predicted and actual
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": threshold,
        "count": len(rows),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / len(rows) if rows else 0.0,
    }


def calibrate_judge(
    development: Path,
    holdout: Path,
    *,
    minimum_precision: float = 0.9,
    minimum_records: int = 25,
) -> dict[str, Any]:
    """Choose on development data, then evaluate once on a disjoint holdout."""
    development = development.resolve()
    holdout = holdout.resolve()
    if development == holdout or _sha256(development) == _sha256(holdout):
        raise ValueError("Development and holdout reviews must be distinct")
    dev_rows = _read_complete_reviews(development, minimum_records)
    holdout_rows = _read_complete_reviews(holdout, minimum_records)
    overlap = {
        str(row["record_id"]) for row in dev_rows
    } & {str(row["record_id"]) for row in holdout_rows}
    if overlap:
        raise ValueError(
            f"Development/holdout record overlap: {sorted(overlap)[:5]}"
        )
    development_metrics = [calibration_metrics(dev_rows, score) for score in range(1, 6)]
    eligible = [
        item
        for item in development_metrics
        if item["precision"] >= minimum_precision
        and item["confusion"]["tp"] > 0
    ]
    if not eligible:
        raise ValueError("No judge threshold reaches development precision target")
    selected = max(
        eligible,
        key=lambda item: (item["f1"], item["recall"], -item["threshold"]),
    )
    holdout_metrics = calibration_metrics(holdout_rows, selected["threshold"])
    feature_hash = hashlib.sha256(
        json.dumps(JUDGE_FEATURES, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "contract_version": CONTRACT_VERSION,
        "feature_contract": list(JUDGE_FEATURES),
        "feature_contract_sha256": feature_hash,
        "minimum_precision": minimum_precision,
        "recommended_threshold": selected["threshold"],
        "development": {
            "path": str(development),
            "sha256": _sha256(development),
            "metrics_by_threshold": development_metrics,
            "selected_metrics": selected,
        },
        "holdout": {
            "path": str(holdout),
            "sha256": _sha256(holdout),
            "metrics": holdout_metrics,
        },
        "passed": holdout_metrics["precision"] >= minimum_precision
        and holdout_metrics["confusion"]["tp"] > 0,
    }


def load_judge_calibration(
    config: dict[str, Any], *, required: bool
) -> dict[str, Any]:
    """Verify a hash-pinned passing artifact without trusting its file paths."""
    calibration = config.get("judge_calibration", {})
    path = Path(str(calibration.get("path", "")))
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    expected_sha = calibration.get("sha256")
    if not path.is_file() or not expected_sha:
        if required:
            raise SystemExit(
                "Full run blocked: configure a hash-pinned judge calibration artifact"
            )
        return {"verified": False, "path": str(path)}
    actual_sha = _sha256(path)
    if actual_sha != expected_sha:
        raise SystemExit("Judge calibration artifact SHA-256 does not match config")
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("contract_version") != CONTRACT_VERSION or not artifact.get("passed"):
        raise SystemExit("Judge calibration artifact is incompatible or not passing")
    expected_features = list(JUDGE_FEATURES)
    if artifact.get("feature_contract") != expected_features:
        raise SystemExit("Judge calibration feature contract is stale")
    minimum_precision = float(
        calibration.get("minimum_holdout_precision", 0.9)
    )
    minimum_records = int(calibration.get("minimum_records_per_split", 25))
    holdout_metrics = artifact.get("holdout", {}).get("metrics", {})
    development_count = int(
        artifact.get("development", {})
        .get("selected_metrics", {})
        .get("count", 0)
    )
    if (
        float(holdout_metrics.get("precision", 0.0)) < minimum_precision
        or int(holdout_metrics.get("count", 0)) < minimum_records
        or development_count < minimum_records
    ):
        raise SystemExit(
            "Judge calibration artifact does not meet configured holdout policy"
        )
    return {
        "verified": True,
        "path": str(path),
        "sha256": actual_sha,
        "recommended_threshold": int(artifact["recommended_threshold"]),
        "holdout_metrics": holdout_metrics,
    }


def main(argv: list[str] | None = None) -> None:
    """Write a calibration artifact from development and held-out reviews."""
    parser = argparse.ArgumentParser()
    parser.add_argument("development", type=Path)
    parser.add_argument("holdout", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--minimum-precision", type=float, default=0.9)
    parser.add_argument("--minimum-records", type=int, default=25)
    args = parser.parse_args(argv)
    artifact = calibrate_judge(
        args.development,
        args.holdout,
        minimum_precision=args.minimum_precision,
        minimum_records=args.minimum_records,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(artifact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
