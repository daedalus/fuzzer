"""Regressions from the 2026-11-13 audit: fd leak on failed spawn, stderr on in-process timeout."""

import os
import shutil
import subprocess

import pytest

from fuzzer_tool.adapters.inprocess import InProcessRunner
from fuzzer_tool.adapters.process import run_target_fast

_SPAWN_FAILED_RC = -2
_TIMEOUT_RC = -1
_FAILED_SPAWNS = 20
_REPORT = "==1==ERROR: AddressSanitizer: heap-buffer-overflow"

# Prints a report, then overruns the deadline ('H'); stays silent and hangs ('S'); returns ('A').
_SRC = (
    r"""
#include <stdio.h>
#include <unistd.h>
int f(const unsigned char *d, unsigned long n) {
    if (n && d[0] == 'A') return 0;
    if (n && d[0] == 'H') { fputs("%s\n", stderr); fflush(stderr); }
    usleep(600000);
    return 0;
}
"""
    % _REPORT
)
_DEADLINE_S = 0.2


def _open_fds() -> int:
    return len(os.listdir("/proc/self/fd"))


class TestFastSpawnFds:
    """``run_target_fast`` leaked stdin + stderr-write fds when ``posix_spawn`` raised."""

    def test_regression_failed_spawn_leaks_no_fd(self):
        run_target_fast("/bin/true", b"x", timeout=2)  # opens the cached input fd
        before = _open_fds()

        rcs = {
            run_target_fast("/nonexistent/target", b"x", timeout=2)[0]
            for _ in range(_FAILED_SPAWNS)
        }

        assert rcs == {_SPAWN_FAILED_RC}
        assert _open_fds() == before

    def test_successful_spawn_leaks_no_fd(self):
        """Falsification: the success path still closes everything and runs the target."""
        run_target_fast("/bin/true", b"x", timeout=2)
        before = _open_fds()

        rcs = {run_target_fast("/bin/true", b"x", timeout=2)[0] for _ in range(_FAILED_SPAWNS)}

        assert rcs == {0}
        assert _open_fds() == before


@pytest.fixture(scope="module")
def so(tmp_path_factory):
    cc = shutil.which("clang")
    if not cc:
        pytest.skip("no clang")
    d = tmp_path_factory.mktemp("inproc")
    src, out = d / "s.c", d / "s.so"
    src.write_text(_SRC)
    r = subprocess.run(
        [cc, "-shared", "-fPIC", "-o", str(out), str(src)], capture_output=True, text=True
    )
    if r.returncode != 0:
        pytest.skip(f"build failed: {r.stderr[:300]}")
    return str(out)


@pytest.fixture(
    params=[{"direct": True}, {"direct_lite": True, "capture_stderr": True}],
    ids=["direct", "direct_lite"],
)
def runner(so, request):
    r = InProcessRunner(so, "f", timeout=_DEADLINE_S, **request.param)
    yield r
    r.stop()


class TestInprocessTimeoutStderr:
    """A sanitizer report written before the deadline was replaced by "timeout"."""

    def test_regression_report_survives_timeout(self, runner):
        rc, err = runner.run_one(b"H")

        assert rc == _TIMEOUT_RC
        assert _REPORT in err

    def test_silent_hang_keeps_sentinel(self, runner):
        """Falsification: a target that wrote nothing is still a plain timeout.

        The text is "timeout" unless the host logged to fd 2 during the call
        (the one-time shared-heap warning), so only the report's absence is pinned.
        """
        rc, err = runner.run_one(b"S")

        assert rc == _TIMEOUT_RC
        assert err
        assert _REPORT not in err

    def test_report_does_not_leak_into_next_run(self, runner):
        """Adversarial: the timed-out run's stderr must not show up one exec later."""
        runner.run_one(b"H")

        assert runner.run_one(b"A") == (0, "")


class TestPersistentLoaderTempFiles:
    """Each hang restart left a ``fuzz_persist_*.py`` and a ``fuzz_child_*.pid`` in the temp dir."""

    _HANG_SRC = "int f(const unsigned char *d, unsigned long n) { if (n && d[0] == 'H') for (;;); return 0; }"
    _HANGS = 2

    @pytest.fixture
    def loader(self, tmp_path, monkeypatch):
        from fuzzer_tool.adapters.persistent_subprocess import PersistentLoader

        cc = shutil.which("clang")
        if not cc:
            pytest.skip("no clang")
        src, out = tmp_path / "h.c", tmp_path / "h.so"
        src.write_text(self._HANG_SRC)
        r = subprocess.run(
            [cc, "-shared", "-fPIC", "-o", str(out), str(src)], capture_output=True, text=True
        )
        if r.returncode != 0:
            pytest.skip(f"build failed: {r.stderr[:300]}")

        # Private temp dir: the count below sees this loader's files only.
        tmp = tmp_path / "tmp"
        tmp.mkdir()
        monkeypatch.setattr("tempfile.tempdir", str(tmp))
        pl = PersistentLoader(str(out), "f", timeout=0.5)
        yield pl, tmp
        pl.stop()

    @staticmethod
    def _files(tmp):
        return sorted(p.name.split("_")[1] for p in tmp.iterdir() if p.name.startswith("fuzz_"))

    def test_regression_hangs_do_not_accumulate_files(self, loader):
        pl, tmp = loader
        assert pl.start()
        for _ in range(self._HANGS):
            assert pl.run_one(b"H")[0] == _TIMEOUT_RC

        assert pl.run_one(b"A")[0] == 0
        # One live loader: at most one script and one pid file, whatever the hang count.
        files = self._files(tmp)
        assert "persist" in files
        assert len(files) == len(set(files))

    def test_regression_stop_removes_all(self, loader):
        pl, tmp = loader
        assert pl.start()
        pl.run_one(b"H")
        pl.stop()

        assert self._files(tmp) == []
