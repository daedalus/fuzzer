"""Regression tests: subprocess loader's standalone-exec clamp flaw.

The standalone-executable branch of _LOADER_SCRIPT used to exit with
max(0, min(proc.returncode, 125)): a target killed by SIGSEGV reports
returncode -11, which the clamp turned into a clean exit 0 — so the crash
was invisible to the fuzzer. Signal-killed targets must now exit 128+signum
(139 for SIGSEGV, 134 for SIGABRT), which is_crash() recognizes via
SIGNAL_CRASH_CODES.
"""

import subprocess
import sys
from pathlib import Path

from fuzzer_tool.adapters.inprocess import _LOADER_SCRIPT
from fuzzer_tool.adapters.process import SIGNAL_CRASH_CODES
from tests.conftest import requires_clang

pytestmark = requires_clang

TARGETS_DIR = Path(__file__).parent.parent / "targets"


def _compile_standalone(src: Path, out: Path):
    subprocess.run(
        ["clang", "-o", str(out), str(src)],
        check=True,
        capture_output=True,
    )


class TestSubprocessExecClamp:
    def _run_loader(self, tmp_path, data: bytes) -> int:
        binary = tmp_path / "tgt"
        _compile_standalone(TARGETS_DIR / "test_target.c", binary)
        loader = tmp_path / "loader.py"
        loader.write_text(_LOADER_SCRIPT)
        proc = subprocess.run(
            [sys.executable, str(loader), str(binary), "LLVMFuzzerTestOneInput"],
            input=data,
            capture_output=True,
            timeout=30,
        )
        return proc.returncode

    def test_sigsegv_target_exits_139(self, tmp_path):
        rc = self._run_loader(tmp_path, b"CRASHS")
        assert rc == 139, f"expected 128+SIGSEGV, got {rc}"
        assert rc in SIGNAL_CRASH_CODES

    def test_sigabrt_target_exits_134(self, tmp_path):
        rc = self._run_loader(tmp_path, b"CRASHA")
        assert rc == 134, f"expected 128+SIGABRT, got {rc}"
        assert rc in SIGNAL_CRASH_CODES

    def test_safe_target_exits_0(self, tmp_path):
        rc = self._run_loader(tmp_path, b"SAFEXXX00")
        assert rc == 0

    def test_timeout_not_reported_as_crash(self, tmp_path):
        # A SIGKILL'd (timed-out) target exits 137, which is NOT in
        # SIGNAL_CRASH_CODES — the clamp fix must not turn timeouts into
        # false-positive crashes.
        binary = tmp_path / "tgt"
        _compile_standalone(TARGETS_DIR / "test_target.c", binary)
        loader = tmp_path / "loader.py"
        loader.write_text(_LOADER_SCRIPT)
        import os

        env = os.environ.copy()
        env["_TIMEOUT"] = "0"  # communicate times out immediately, then kills
        proc = subprocess.run(
            [sys.executable, str(loader), str(binary), "LLVMFuzzerTestOneInput"],
            input=b"SAFEXXX00",
            capture_output=True,
            timeout=30,
            env=env,
        )
        assert proc.returncode not in SIGNAL_CRASH_CODES
        assert proc.returncode != 0  # the old clamp hid the timeout as exit 0
