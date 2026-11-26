"""Regression: CI must trigger on the repository's default branch.

ci.yml listed ``main`` while the repo's only long-lived branch is ``master``,
so no push or pull request ever ran CI. Parsed as text: no YAML dependency.
"""

import re
from pathlib import Path

CI = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "ci.yml"
DEFAULT_BRANCH = "master"  # origin/HEAD of daedalus/fuzzer
EVENTS = ("push", "pull_request")


def _branches(text: str, event: str) -> list[str]:
    """Branch filter of `event` under `on:` (``branches: [a, b]`` form)."""
    m = re.search(rf"^  {event}:\n    branches: \[([^\]]*)\]", text, re.MULTILINE)
    assert m, f"no branches filter for {event}"
    return [b.strip().strip("'\"") for b in m.group(1).split(",")]


def test_regression_ci_triggers_on_default_branch():
    text = CI.read_text()
    for event in EVENTS:
        assert DEFAULT_BRANCH in _branches(text, event), event


def test_parser_rejects_a_missing_filter():
    """Falsification: the helper must fail loudly, not pass vacuously."""
    try:
        _branches("on:\n  push:\n    paths: [src]\n", "push")
    except AssertionError:
        return
    raise AssertionError("missing filter was accepted")
