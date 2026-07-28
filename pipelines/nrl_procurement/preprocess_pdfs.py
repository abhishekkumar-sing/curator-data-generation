"""Convert source PDFs to structured Markdown with Chandra OCR 2."""

import argparse
import os
import subprocess
from pathlib import Path

from settings import CONFIG, PROJECT_ROOT, require_private_endpoint, require_setting

PATH_CONFIG = CONFIG["paths"]
OCR_CONFIG = CONFIG["models"]["ocr"]
DEFAULT_SOURCE_DIR = PROJECT_ROOT / PATH_CONFIG["source_dir"]


def main() -> None:
    """Run Chandra OCR 2 against every PDF in the source directory."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=28)
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()

    base_url_name = OCR_CONFIG["base_url_env"]
    base_url = (
        require_private_endpoint(base_url_name)
        if OCR_CONFIG.get("private_endpoint_only", True)
        else require_setting(base_url_name)
    )
    model = require_setting(OCR_CONFIG["model_env"])
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
