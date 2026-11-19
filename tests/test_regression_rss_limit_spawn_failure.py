"""Regression: --rss-limit-mb turned a spawn failure into exit 127.

The limit runs the target behind ``/bin/sh -c 'ulimit -v; exec ...'``; sh
spawned fine, its exec failed, and the caller read rc 127 (a target exit)
instead of -2 (infrastructure). Seen as a suite flake: a Fuzzer earlier in
the worker left the limit on.
"""

from __future__ import annotations

import pytest

from fuzzer_tool.adapters import process

_LIMIT_MB = 64
_INFRA_RC = -2


@pytest.fixture
def limit():
    process.set_rss_limit(_LIMIT_MB)
    yield
    process.set_rss_limit(0)


@pytest.mark.usefixtures("limit")
def test_regression_missing_target_is_infra_error():
    """Falsification: no such binary, limit on -> -2 and no pid."""
    rc, _, pid = process.run_target_fast("/nonexistent/binary", b"", timeout=10.0)

    assert (rc, pid) == (_INFRA_RC, 0)


@pytest.mark.usefixtures("limit")
def test_non_executable_target_is_infra_error(tmp_path):
    """Adversarial: the file exists but has no exec bit."""
    target = tmp_path / "plain"
    target.write_text("#!/bin/sh\nexit 0\n")
    target.chmod(0o644)

    rc, _, pid = process.run_target_fast(str(target), b"", timeout=10.0)

    assert (rc, pid) == (_INFRA_RC, 0)
