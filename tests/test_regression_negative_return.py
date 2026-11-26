"""A harness's negative return value is neither a crash nor a timeout.

The guard returned the entry's value unmasked, so -6 read as SIGABRT and -1
(libFuzzer's "reject input") as a timeout. A return is now reported as
exit() would report it: rc & 0xFF (-6 -> 250, -1 -> 255).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from fuzzer_tool.cli import commands
from fuzzer_tool.services.minimize import ExecMode

REPO_ROOT = Path(__file__).resolve().parents[1]
SHIM = REPO_ROOT / "src" / "fuzzer_tool" / "adapters" / "afl_shim.c"

# 'n' returns -SIGABRT, 'm' returns -1, 'x' aborts; anything else returns 0.
TARGET_SRC = r"""
#include <stddef.h>
#include <stdlib.h>
__attribute__((visibility("default")))
int fuzz_test(const unsigned char *buf, size_t len) {
    if (!len) return 0;
    if (buf[0] == 'n') return -6;
    if (buf[0] == 'm') return -1;
    if (buf[0] == 'x') abort();
    return 0;
}
"""

NEG_SIG = b"n"
NEG_ONE = b"m"
CRASH = b"x"
REPRODUCED = 0
NOT_REPRODUCED = 1
EXIT_MASK = 0xFF

MODES = {
    ExecMode.LOADER: ("--inprocess",),
    ExecMode.DIRECT_LITE: ("--inprocess-direct",),
    ExecMode.DIRECT: ("--inprocess", "--inprocess-direct"),
}


@pytest.fixture(autouse=True)
def _restore_env():
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


def _build(tmp_path: Path, shim: bool) -> str:
    if shutil.which("clang") is None:
        pytest.skip("clang not available")
    d = tmp_path / ("shim" if shim else "bare")
    d.mkdir()
    src = d / "t.c"
    src.write_text(TARGET_SRC)
    so = d / "t.so"
    cmd = ["clang", "-O1", "-fPIC", "-shared", "-o", str(so), str(src)]
    if shim:
        cmd.append(str(SHIM))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.skip(f"build failed: {proc.stderr[-400:]}")
    return str(so)


def _replay(monkeypatch, tmp_path, so, data, flags):
    inp = tmp_path / f"in_{data.hex()}.bin"
    inp.write_bytes(data)
    argv = ["fuzzer-tool", "replay", so, str(inp), "--inprocess-func", "fuzz_test", *flags]
    monkeypatch.setattr(sys, "argv", argv)
    return commands.main()


@pytest.mark.parametrize("mode", list(MODES), ids=lambda m: m.value)
class TestGuarded:
    def test_regression_neg_sig_clean(self, monkeypatch, tmp_path, mode, capsys):
        so = _build(tmp_path, shim=True)
        assert _replay(monkeypatch, tmp_path, so, NEG_SIG, MODES[mode]) == NOT_REPRODUCED
        assert "Crash reproduced" not in capsys.readouterr().out

    def test_regression_neg_one_no_hang(self, monkeypatch, tmp_path, mode, capsys):
        so = _build(tmp_path, shim=True)
        assert _replay(monkeypatch, tmp_path, so, NEG_ONE, MODES[mode]) == NOT_REPRODUCED
        out = capsys.readouterr().out
        assert "timed out" not in out
        assert "Crash reproduced" not in out

    def test_crash_still_found(self, monkeypatch, tmp_path, mode, capsys):
        # Control: a real abort still reproduces in every mode.
        so = _build(tmp_path, shim=True)
        assert _replay(monkeypatch, tmp_path, so, CRASH, MODES[mode]) == REPRODUCED
        assert "signal 6" in capsys.readouterr().out


@pytest.mark.parametrize("mode", [ExecMode.DIRECT_LITE, ExecMode.DIRECT], ids=lambda m: m.value)
def test_regression_unguarded_mask(tmp_path, mode):
    # Adversarial: no shim, so no guard; the runner masks the raw return itself.
    from fuzzer_tool.adapters.inprocess import InProcessRunner

    so = _build(tmp_path, shim=False)
    runner = InProcessRunner(
        target=so,
        function_name="fuzz_test",
        timeout=1,
        direct=mode is ExecMode.DIRECT,
        direct_lite=mode is ExecMode.DIRECT_LITE,
    )
    try:
        got = [runner.run_one(d)[0] for d in (NEG_SIG, NEG_ONE)]
    finally:
        runner.stop()
    assert got == [-6 & EXIT_MASK, -1 & EXIT_MASK]
