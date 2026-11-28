"""Regressions from the 2026-11-24 external code review, in-process layer.

Three defects, all on rarely-taken paths in the mode where a bug corrupts
the fuzzer's own process rather than a child:

* The subprocess loader script called ``shim.cov_get_bitmap()`` /
  ``cov_get_size()`` — symbols no shim this tree builds has ever exported
  (shim_factory's stub defines only ``__sanitizer_cov_8bit_counters_init``)
  — and the resulting AttributeError escaped the ``except OSError`` meant
  to contain it. Every clean run through the ``_COV_SHM_PATH`` fallback
  exited 1 with a traceback instead of the target's own exit code, and no
  coverage was ever written by that path.
* ``_run_c_direct_lite`` installed a SIGALRM handler once and saved the
  previous disposition to ``_old_alarm_handler``, but nothing ever restored
  it: the displaced disposition leaked for the process lifetime.
* ``update_shm_after_resize``'s fallback (target lacks ``__afl_map_shm``)
  stored the raw shmat base into ``__afl_area``, whose every consumer
  treats as base + SHM_METADATA_SIZE (32): edge writes would land in the
  segment header instead of the edge table.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from fuzzer_tool.adapters.inprocess import _LOADER_SCRIPT, InProcessRunner
from fuzzer_tool.adapters.shm import SHM_METADATA_SIZE, ShmCoverage

REPO_ROOT = Path(__file__).resolve().parents[1]
SHIM = REPO_ROOT / "src" / "fuzzer_tool" / "adapters" / "afl_shim.c"

# Any C toolchain does for the stub (no sanitizer coverage needed).
_SRC_PLAIN = r"""
#include <stddef.h>
__attribute__((visibility("default")))
int fuzz_test(const unsigned char *buf, size_t len) { (void)buf; (void)len; return 0; }
"""

# __afl_area without __afl_map_shm: exactly what reaches the resize fallback.
_SRC_NO_MAP_SHM = r"""
__attribute__((visibility("default")))
void *__afl_area = (void *)0;
"""


def _compiler() -> str | None:
    for cc in ("clang", "gcc", "cc"):
        if shutil.which(cc):
            return cc
    return None


def _build(tmp_path: Path, src: str, name: str) -> str:
    cc = _compiler()
    if cc is None:
        pytest.skip("no C compiler available")
    src_path = tmp_path / f"{name}.c"
    src_path.write_text(src)
    so = tmp_path / f"{name}.so"
    proc = subprocess.run(
        [cc, "-O1", "-fPIC", "-shared", "-o", str(so), str(src_path)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or not so.is_file():
        pytest.skip(f"stub failed to build: {proc.stderr[-400:]}")
    return str(so)


class TestLoaderDeadShimPathIsContained:
    """A shim without cov_get_bitmap must not kill the loader script."""

    def test_clean_run_returns_targets_exit_code(self, tmp_path):
        """Regression: AttributeError escaped except OSError -> exit 1.

        Before the fix every run with _COV_SHM_PATH set (and the shim
        exporting no cov_get_bitmap — i.e. every shim this tree builds)
        died with a traceback and exit code 1, masking the target's own 0.
        """
        so = _build(tmp_path, _SRC_PLAIN, "plain")
        loader = tmp_path / "loader.py"
        loader.write_text(_LOADER_SCRIPT)
        bitmap_out = tmp_path / "bitmap.cov"

        env = {k: v for k, v in os.environ.items() if k != "__AFL_SHM_ID"}
        env["_COV_SHM_PATH"] = so
        env["_COV_BITMAP_OUT"] = str(bitmap_out)

        r = subprocess.run(
            [sys.executable, str(loader), so, "fuzz_test"],
            input=b"AAAA",
            capture_output=True,
            env=env,
            timeout=30,
        )
        stderr = r.stderr.decode("utf-8", "replace")
        assert r.returncode == 0, f"loader exited {r.returncode} on a clean run:\n{stderr[-800:]}"
        assert "AttributeError" not in stderr
        assert "Traceback" not in stderr


class TestSigalrmDispositionRestored:
    """direct_lite's install-once SIGALRM handler must be un-installed."""

    def test_stop_restores_the_displaced_disposition(self, tmp_path):
        so = _build(tmp_path, _SRC_PLAIN, "alarm")
        shm = ShmCoverage(size=65536)
        prev = {k: os.environ.get(k) for k in ("__AFL_SHM_ID", "AFL_MAP_SIZE")}
        os.environ["__AFL_SHM_ID"] = str(shm.env_id)
        os.environ["AFL_MAP_SIZE"] = "65536"
        # A third party owned SIGALRM before the runner did.
        prev_disposition = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, signal.SIG_IGN)
        try:
            runner = InProcessRunner(
                so,
                "fuzz_test",
                direct_lite=True,
                shm_size=65536,
                coverage_env_id=shm.env_id,
                timeout=5.0,
            )
            runner._start()
            runner.run_one(b"x")
            # Install-once happened: the runner owns SIGALRM now, and
            # SIG_IGN is the disposition stop() must put back.
            assert signal.getsignal(signal.SIGALRM) is runner._alarm_handler
            assert runner._old_alarm_handler is signal.SIG_IGN

            runner.stop()

            assert signal.getsignal(signal.SIGALRM) is signal.SIG_IGN, (
                "stop() left the runner's SIGALRM handler installed — the "
                "disposition it displaced leaks for the process lifetime"
            )
        finally:
            signal.signal(signal.SIGALRM, prev_disposition)
            for k, v in prev.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            shm.cleanup()

    def test_stop_without_handler_is_a_noop(self, tmp_path):
        """A runner that never ran direct_lite must not touch SIGALRM."""
        so = _build(tmp_path, _SRC_PLAIN, "noalarm")
        before = signal.getsignal(signal.SIGALRM)
        runner = InProcessRunner(
            so,
            "fuzz_test",
            direct_lite=True,
            shm_size=65536,
            timeout=5.0,
        )
        assert not hasattr(runner, "_alarm_handler")
        runner.stop()
        assert signal.getsignal(signal.SIGALRM) is before


class TestResizeFallbackPointsAtTable:
    """The __afl_area fallback must add the 32-byte header offset."""

    def test_fallback_writes_base_plus_metadata_size(self, tmp_path):
        so = _build(tmp_path, _SRC_NO_MAP_SHM, "nomapshm")
        runner = InProcessRunner.__new__(InProcessRunner)
        runner._persistent = None
        runner._forkserver = None
        runner._lib = ctypes.CDLL(so)
        runner._shim_handle = ctypes.CDLL(so)
        runner._shm_ptr = None
        runner.shm_size = 8192
        runner.coverage_env_id = ""

        base = 0x7F0000000000
        runner.update_shm_after_resize(base, 16384, "4242")

        area = ctypes.c_void_p.in_dll(runner._lib, "__afl_area")
        assert area.value == base + SHM_METADATA_SIZE, (
            "__afl_area is the edge TABLE pointer (base + 32); the raw "
            "shmat base would aim every edge write into the segment header"
        )
        # Bookkeeping updated alongside.
        assert runner.shm_size == 16384
        assert runner.coverage_env_id == "4242"
