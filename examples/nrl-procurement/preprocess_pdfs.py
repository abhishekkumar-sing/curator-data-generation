"""Convert source PDFs to structured Markdown with Chandra OCR 2."""

import argparse
import os
import subprocess
from pathlib import Path

from generate import DEFAULT_SOURCE_DIR, PROJECT_ROOT, load_dotenv


def require_setting(name: str) -> str:
    """Return a required non-empty environment setting."""
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"Set {name} in {PROJECT_ROOT / '.env'}")
    return value


def main() -> None:
    """Run Chandra OCR 2 against every PDF in the source directory."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=28)
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    base_url = require_setting("OCR_BASE_URL")
    model = require_setting("OCR_MODEL")
    api_key = require_setting("OCR_API_KEY")
    ocr_command = os.environ.get("OCR_COMMAND", "chandra").strip() or "chandra"
    output_dir = args.output_dir or PROJECT_ROOT / os.environ.get(
        "OCR_OUTPUT_DIR",
        "data/interim/ocr",
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
            "--batch-size",
            str(args.batch_size),
            "--max-workers",
            str(args.max_workers),
        ]
        print(f"OCR: {pdf.name}", flush=True)
        subprocess.run(command, check=True, env=environment)

    print(f"Chandra outputs written to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
