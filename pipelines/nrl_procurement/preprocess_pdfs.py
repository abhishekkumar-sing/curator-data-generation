"""Convert source PDFs to structured Markdown with Chandra OCR 2."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from pypdf import PdfReader
from settings import CONFIG, PROJECT_ROOT, require_private_endpoint, require_setting

PATH_CONFIG = CONFIG["paths"]
OCR_CONFIG = CONFIG["models"]["ocr"]
DEFAULT_SOURCE_DIR = PROJECT_ROOT / PATH_CONFIG["source_dir"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_provenance(
    pdf: Path,
    output_dir: Path,
    model: str,
    model_revision: str,
    package_revision: str,
    command: list[str],
    batch_size: int,
    max_workers: int,
) -> None:
    markdown = sorted(path for path in output_dir.rglob("*.md") if path.stem == pdf.stem)
    if len(markdown) != 1:
        raise RuntimeError(f"Expected one OCR Markdown output for {pdf.name}; found {len(markdown)}")
    metadata = sorted(path for path in output_dir.rglob("*_metadata.json") if path.name == f"{pdf.stem}_metadata.json")
    try:
        package_version = importlib.metadata.version("chandra-ocr")
    except importlib.metadata.PackageNotFoundError:
        package_version = "unknown"
    payload = {
        "contract_version": "nrl-ocr-provenance-v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_file": str(pdf),
        "source_sha256": _sha256(pdf),
        "source_page_count": len(PdfReader(str(pdf)).pages),
        "model": model,
        "model_revision": model_revision,
        "engine": command[0],
        "chandra_ocr_version": package_version,
        "package_revision": package_revision,
        "method": "vllm",
        "batch_size": batch_size,
        "max_workers": max_workers,
        "arguments": command[3:],
        "markdown_file": str(markdown[0].relative_to(output_dir)),
        "markdown_sha256": _sha256(markdown[0]),
        "markdown_page_count": len(
            re.findall(
                r"(?m)^\s*\d+-{20,}\s*$",
                markdown[0].read_text(encoding="utf-8"),
            )
        )
        + 1,
        "metadata_file": (str(metadata[0].relative_to(output_dir)) if len(metadata) == 1 else None),
        "metadata_sha256": _sha256(metadata[0]) if len(metadata) == 1 else None,
    }
    cache = output_dir / pdf.stem / ".chandra-cache.json"
    temporary = cache.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(cache)


def main() -> None:
    """Run Chandra OCR 2 against every PDF in the source directory."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=28)
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()

    base_url_name = OCR_CONFIG["base_url_env"]
    base_url = require_private_endpoint(base_url_name) if OCR_CONFIG.get("private_endpoint_only", True) else require_setting(base_url_name)
    model = require_setting(OCR_CONFIG["model_env"])
    model_revision = require_setting(OCR_CONFIG["model_revision_env"])
    package_revision = require_setting(OCR_CONFIG["package_revision_env"])
    api_key = require_setting(OCR_CONFIG["api_key_env"])
    command_env = OCR_CONFIG["command_env"]
    ocr_command = os.environ.get(command_env, OCR_CONFIG["engine"]).strip() or OCR_CONFIG["engine"]
    output_dir_env = OCR_CONFIG["output_dir_env"]
    output_dir = args.output_dir or PROJECT_ROOT / os.environ.get(
        output_dir_env,
        PATH_CONFIG["ocr_dir"],
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    environment = os.environ.copy()
    environment.update(
        {
            "VLLM_API_BASE": base_url,
            "VLLM_MODEL_NAME": model,
            "VLLM_API_KEY": api_key,
            "OPENAI_API_KEY": api_key,
        }
    )

    pdfs = sorted(args.source_dir.resolve().rglob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found under {args.source_dir}")

    for pdf in pdfs:
        command = [
            ocr_command,
            str(pdf),
            str(output_dir.resolve()),
            "--method",
            "vllm",
            "--no-images",
            "--paginate_output",
            "--batch-size",
            str(args.batch_size),
            "--max-workers",
            str(args.max_workers),
        ]
        print(f"OCR: {pdf.name}", flush=True)
        subprocess.run(command, check=True, env=environment)
        _write_provenance(
            pdf,
            output_dir.resolve(),
            model,
            model_revision,
            package_revision,
            command,
            args.batch_size,
            args.max_workers,
        )

    print(f"Chandra outputs written to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
