"""Packaged NRL CLI dispatch regressions."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from pipelines.nrl_procurement import cli  # noqa: E402


def test_top_level_help_lists_in_process_commands(capsys) -> None:
    cli.main(["--help"])
    output = capsys.readouterr().out
    assert "all" in output
    assert "probe-structure" in output
    assert "regress" in output


def test_all_help_is_forwarded_to_generation_parser(capsys) -> None:
    with pytest.raises(SystemExit) as raised:
        cli.main(["all", "--help"])
    assert raised.value.code == 0
    output = capsys.readouterr().out
    assert "--max-passes" in output
    assert "--cross-document-limit" in output
    assert "--review-file" in output


def test_all_help_lists_every_stage_skip_flag(capsys) -> None:
    # T15: propositions/temporal/path_qa/reasoning_paths must be independently
    # disableable at the CLI the same way cross-document/drafting already are.
    with pytest.raises(SystemExit) as raised:
        cli.main(["all", "--help"])
    assert raised.value.code == 0
    output = capsys.readouterr().out
    assert "--skip-cross-document" in output
    assert "--skip-drafting" in output
    assert "--skip-propositions" in output
    assert "--skip-temporal" in output
    assert "--skip-path-qa" in output
    assert "--skip-reasoning-paths" in output
