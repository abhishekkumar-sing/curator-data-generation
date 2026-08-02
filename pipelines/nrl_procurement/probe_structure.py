"""Run live structured-output capability probes for configured model roles."""

from __future__ import annotations

import argparse
import json
from typing import Any

from generate import (
    CACHE_ROOT,
    GENERATION,
    JUDGE,
    _llm_kwargs,
    _role_profile,
)
from settings import CONFIG
from structure_probe import run_structure_probe


def selected_probe_profiles(args: argparse.Namespace) -> list[tuple[str, dict[str, Any]]]:
    """Resolve active, explicit, or every configured role/profile pairing."""
    active = {"generation": GENERATION, "judge": JUDGE}
    roles = args.role or ["generation", "judge"]
    configured_names = sorted(CONFIG.get("model_profiles", {}))
    selected: list[tuple[str, dict[str, Any]]] = []
    for role in roles:
        explicit = getattr(args, f"{role}_profile") or []
        if args.all_configured_profiles:
            names = configured_names
        elif explicit:
            names = explicit
        else:
            selected.append((role, active[role]))
            continue
        selected.extend((role, _role_profile(role, name)) for name in names)
    return selected


def main(argv: list[str] | None = None) -> None:
    """Probe one or both configured roles and return a useful exit status."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--role",
        action="append",
        choices=["generation", "judge"],
        help="Role to probe; repeat for both. Defaults to both roles.",
    )
    parser.add_argument(
        "--generation-profile",
        action="append",
        metavar="NAME",
        help="Probe a named generation profile; repeat as needed.",
    )
    parser.add_argument(
        "--judge-profile",
        action="append",
        metavar="NAME",
        help="Probe a named judge profile; repeat as needed.",
    )
    parser.add_argument(
        "--all-configured-profiles",
        action="store_true",
        help="Probe every configured profile for each selected role.",
    )
    args = parser.parse_args(argv)
    results = []
    for role, profile in selected_probe_profiles(args):
        result = run_structure_probe(
            CACHE_ROOT,
            role,
            profile,
            _llm_kwargs(profile),
        )
        results.append(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if all(result["passed"] for result in results) else 1)


if __name__ == "__main__":
    main()
