"""Per-child CPU time for the bench harnesses (algorithmica handover §4.1).

Wall time drifts with frequency scaling and neighbours; user CPU time of
the campaign process tree does not. ``run_timed`` reads it from
``os.wait4`` so concurrent cells cannot leak into each other.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "lib"))

from child_rusage import run_timed  # noqa: E402

PY = sys.executable
BURN = "x = 0\nfor i in range(6_000_000):\n    x += i\n"


def _zombies() -> set[int]:
    """Pids of this process's unreaped (state Z) children, from /proc."""
    me = str(os.getpid())
    out = set()
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            # "pid (comm) state ppid ...": comm may hold spaces, split after ")".
            fields = stat.read_text().rsplit(")", 1)[1].split()
        except OSError:
            continue  # process exited between glob and read
        if fields[0] == "Z" and fields[1] == me:
            out.add(int(stat.parent.name))
    return out


class TestRunTimed:
    def test_burn_is_cpu_sleep_is_not(self):
        """Falsification: CPU time must separate burning from sleeping."""
        burn = run_timed([PY, "-c", BURN], timeout=60)
        sleep = run_timed([PY, "-c", "import time; time.sleep(0.4)"], timeout=60)

        assert burn.cpu_user >= 0.5 * burn.wall
        assert sleep.wall >= 0.4
        assert sleep.cpu_user + sleep.cpu_sys < sleep.wall / 2

    def test_returncode_and_output(self):
        r = run_timed(
            [PY, "-c", "import sys; print('out'); print('err', file=sys.stderr); sys.exit(3)"],
            timeout=60,
        )

        assert r.rc == 3
        assert "out" in r.log
        assert "err" in r.log

    def test_grandchild_cpu_counted(self):
        """The fuzzer reaps its own children; their CPU must reach the cell."""
        inner = f"import subprocess, sys; subprocess.run([sys.executable, '-c', {BURN!r}])"
        r = run_timed([PY, "-c", inner], timeout=60)

        assert r.cpu_user >= 0.5 * r.wall

    def test_regression_timeout_kills_and_reaps(self):
        """Adversarial: a hung child is killed, reported rc=-1, and reaped."""
        before = _zombies()
        hang = (
            "import sys, time; print('started', flush=True); "
            "print('ASAN: diag', file=sys.stderr, flush=True); time.sleep(60)"
        )
        r = run_timed([PY, "-c", hang], timeout=0.5)

        assert r.rc == -1
        assert r.wall < 30
        assert "started" in r.log
        assert "ASAN: diag" in r.log  # stderr survives the kill
        assert _zombies() <= before

    def test_missing_binary_raises(self):
        with pytest.raises(FileNotFoundError):
            run_timed(["/nonexistent/binary"], timeout=5)
