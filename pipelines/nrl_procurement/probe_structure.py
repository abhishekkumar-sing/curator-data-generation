"""Run live structured-output capability probes for configured model roles."""

from __future__ import annotations

import argparse
import json

from generate import CACHE_ROOT, GENERATION, JUDGE, _llm_kwargs
from structure_probe import run_structure_probe


def main() -> None:
    """Probe one or both configured roles and return a useful exit status."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--role",
        action="append",
        choices=["generation", "judge"],
        help="Role to probe; repeat for both. Defaults to both roles.",
    )
    args = parser.parse_args()
    profiles = {"generation": GENERATION, "judge": JUDGE}
    roles = args.role or ["generation", "judge"]
    results = []
    for role in roles:
        profile = profiles[role]
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
