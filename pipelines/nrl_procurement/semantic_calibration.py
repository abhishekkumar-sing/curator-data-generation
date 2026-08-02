"""Probe embeddings and summarize human-reviewed semantic calibration pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml
from semantic_diversity import (
    calibration_report,
    load_embedding_settings,
    probe_embedding_endpoint,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into dictionaries with useful line errors."""
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        rows.append(value)
    return rows


def _write_report(report: dict[str, Any], output: Path | None) -> None:
    """Print a report and optionally persist the same secret-free JSON."""
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(f"{rendered}\n", encoding="utf-8")
    print(rendered)


def main() -> None:
    """Run the requested embedding maintenance command."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    probe = subparsers.add_parser("probe")
    probe.add_argument("--config", type=Path, default=Path("config.yaml"))

    calibrate = subparsers.add_parser("calibrate")
    calibrate.add_argument("--input", type=Path, required=True)
    calibrate.add_argument("--output", type=Path)
    calibrate.add_argument("--minimum-precision", type=float, default=0.95)
    calibrate.add_argument("--minimum-labeled-pairs", type=int, default=50)
    calibrate.add_argument("--minimum-class-pairs", type=int, default=10)

    args = parser.parse_args()
    if args.command == "probe":
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        settings = load_embedding_settings(config)
        if settings is None:
            raise SystemExit("Embedding profile is disabled in the configuration")
        _write_report(probe_embedding_endpoint(settings), None)
        return
    report = calibration_report(
        _read_jsonl(args.input),
        minimum_precision=args.minimum_precision,
        minimum_labeled_pairs=args.minimum_labeled_pairs,
        minimum_class_pairs=args.minimum_class_pairs,
    )
    _write_report(report, args.output)


if __name__ == "__main__":
    main()
