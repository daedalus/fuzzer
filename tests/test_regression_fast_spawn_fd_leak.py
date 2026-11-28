"""Regression: run_target_fast leaked its stdin fd when os.pipe() failed.

``stdin_fd`` was opened outside every cleanup; an EMFILE from the stderr
pipe left it open, making the next exec's EMFILE likelier.
"""

from __future__ import annotations

import os

from fuzzer_tool.adapters import process

_INFRA_RC = -2
_TRUE = "/bin/true"
_TIMEOUT_S = 10.0


def _open_fds() -> set[int]:
    return {int(n) for n in os.listdir("/proc/self/fd")}


def test_regression_pipe_failure_leaks_no_fd(monkeypatch):
    """Falsification: EMFILE on the stderr pipe -> -2, no child, no fd left open."""
    process.run_target_fast(_TRUE, b"", timeout=_TIMEOUT_S)  # opens the reused input file
    before = _open_fds()

    def no_pipe():
        raise OSError(24, "Too many open files")

    monkeypatch.setattr(process.os, "pipe", no_pipe)
    rc, _, pid = process.run_target_fast(_TRUE, b"x", timeout=_TIMEOUT_S)

    assert (rc, pid) == (_INFRA_RC, 0)
    assert _open_fds() == before


def test_run_after_pipe_failure_is_clean(monkeypatch):
    """Adversarial: the exec after a failed one runs normally and leaks nothing either."""
    process.run_target_fast(_TRUE, b"", timeout=_TIMEOUT_S)
    before = _open_fds()
    real_pipe = os.pipe
    calls = [0]

    def flaky_pipe():
        calls[0] += 1
        if calls[0] == 1:
            raise OSError(24, "Too many open files")
        return real_pipe()

    monkeypatch.setattr(process.os, "pipe", flaky_pipe)
    assert process.run_target_fast(_TRUE, b"x", timeout=_TIMEOUT_S)[0] == _INFRA_RC
    rc, _, pid = process.run_target_fast(_TRUE, b"x", timeout=_TIMEOUT_S)

    assert rc == 0 and pid > 0
    assert _open_fds() == before
