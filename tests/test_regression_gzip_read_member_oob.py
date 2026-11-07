"""``targets/gzip_read.c``: the multi-member check read one byte past the input.

``member_end < size`` guarded ``buf[member_end]`` but not ``buf[member_end + 1]``.
A gzip member followed by a lone 0x1F byte made ASAN report a 1-byte
heap-buffer-overflow READ in the harness, not in zlib (found by the
13-target campaign, 2026-11-06).
"""

from __future__ import annotations

import gzip
import subprocess
from pathlib import Path

import pytest

from tests.conftest import requires_clang

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "targets" / "gzip_read.c"
SHIM = ROOT / "src" / "fuzzer_tool" / "adapters" / "afl_shim.c"

GZIP_MAGIC = b"\x1f\x8b"
ASAN_OOB = "heap-buffer-overflow"


@pytest.fixture(scope="module")
def harness(tmp_path_factory) -> Path:
    """ASAN gzip_read executable; reads a file into an exact-size heap buffer."""
    out = tmp_path_factory.mktemp("gzip_read") / "gzip_read"
    cmd = [
        "clang", "-O1", "-g", "-fsanitize=address", "-fno-omit-frame-pointer",
        "-include", str(SHIM), str(TARGET), "-o", str(out), "-lz", "-ldl",
    ]  # fmt: skip
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    return out


def _run(harness: Path, tmp_path: Path, data: bytes) -> subprocess.CompletedProcess:
    f = tmp_path / "input.gz"
    f.write_bytes(data)
    return subprocess.run(
        [str(harness), str(f)], capture_output=True, text=True, timeout=30,
        env={"ASAN_OPTIONS": "detect_leaks=0"},
    )  # fmt: skip


@requires_clang
def test_regression_gzip_read_member_oob(harness, tmp_path):
    """Falsification: one trailing magic byte must not read past the buffer."""
    r = _run(harness, tmp_path, gzip.compress(b"x") + GZIP_MAGIC[:1])

    assert ASAN_OOB not in r.stderr
    assert r.returncode == 0, r.stderr[-2000:]


@requires_clang
def test_regression_gzip_read_full_magic_at_end(harness, tmp_path):
    """Adversarial: the full 2-byte magic ending exactly at the buffer end."""
    r = _run(harness, tmp_path, gzip.compress(b"x") + GZIP_MAGIC)

    assert ASAN_OOB not in r.stderr
    assert r.returncode == 0, r.stderr[-2000:]
