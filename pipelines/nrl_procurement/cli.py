"""Packaged command-line entry point for the NRL procurement pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _pipeline_path() -> Path:
    return Path(__file__).resolve().parent


def _load_module(name: str):
    path = str(_pipeline_path())
    if path not in sys.path:
        sys.path.insert(0, path)
    return __import__(name)


def main(argv: list[str] | None = None) -> None:
    """Dispatch commands without wrapping pipeline stages in subprocesses."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="nrl-curate")
    parser.add_argument(
        "command",
        choices=("all", "probe-structure", "validate-run", "regress"),
    )
    if not arguments or arguments[0] in {"-h", "--help"}:
        parser.print_help()
        return
    # Parse only the command token so subcommand flags such as `all --help`
    # are forwarded to the shared in-process implementation.
    known = parser.parse_args(arguments[:1])
    remaining = arguments[1:]
    if known.command == "all":
        _load_module("generate").main(remaining)
        return
    if known.command == "probe-structure":
        _load_module("probe_structure").main(remaining)
        return
    if known.command == "regress":
        _load_module("evaluation").main(remaining)
        return
    _load_module("validate_run").main(remaining)


if __name__ == "__main__":
    main()
