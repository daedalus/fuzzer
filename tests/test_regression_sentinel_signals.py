"""Regression: SIGHUP / SIGINT deaths collided with the -1 / -2 sentinels.

Wait statuses decoded to ``-WTERMSIG``, while -1 means timeout and -2 means
infrastructure failure: a target killed by SIGHUP was filed as a hang, one
killed by SIGINT was dropped unread. Those two now decode as 128+sig, the
shell convention, so they are plain non-zero exits.
"""

from __future__ import annotations

import shutil
import signal

import pytest

from fuzzer_tool.adapters import process
from fuzzer_tool.adapters.forkserver import ForkserverRunner, _ensure_compiled
from fuzzer_tool.services.runner import TargetRunner

_EXIT_BASE = 128
_TIMEOUT, _INFRA = -1, -2


class _Fuzzer:
    extra_crash_codes: set[int] = set()
    last_report = None


def _script(tmp_path, sig: signal.Signals) -> str:
    path = tmp_path / f"kill_{sig.name}.sh"
    path.write_text(f"#!/bin/sh\ntrap - {sig.name[3:]}\nkill -{sig.name[3:]} $$\nsleep 5\n")
    path.chmod(0o755)
    return str(path)


def _runs(target: str) -> list[int]:
    rcs = [
        process.run_target_stdin(target, b"x", 10.0)[0],
        process.run_target_fast(target, b"x", timeout=10.0)[0],
    ]
    if (shutil.which("cc") or shutil.which("clang")) and _ensure_compiled() is not None:
        fs = ForkserverRunner(target, timeout=10.0)
        try:
            assert fs.start()
            rcs.append(fs.run_one(b"x")[0])
        finally:
            fs.stop()
    return rcs


@pytest.mark.parametrize("sig", [signal.SIGHUP, signal.SIGINT])
def test_regression_sentinel_signal_not_sentinel(tmp_path, sig) -> None:
    rcs = _runs(_script(tmp_path, sig))

    assert rcs and all(rc == _EXIT_BASE + sig for rc in rcs)
    runner = TargetRunner(_Fuzzer())  # type: ignore[arg-type]
    assert not runner.is_crash(rcs[0], "")


def test_crash_signal_still_negative(tmp_path) -> None:
    # Falsification: fault signals keep -sig and stay crashes.
    rcs = _runs(_script(tmp_path, signal.SIGSEGV))

    assert all(rc == -signal.SIGSEGV for rc in rcs)


@pytest.mark.parametrize("sig", list(signal.Signals)[:40])
def test_signal_rc_never_a_sentinel(sig) -> None:
    # Edge: no signal decodes to a sentinel.
    assert process.signal_rc(int(sig)) not in (_TIMEOUT, _INFRA)
