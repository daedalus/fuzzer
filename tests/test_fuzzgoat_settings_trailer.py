"""targets/fuzzgoat_read.c settings trailer: make json_settings branches fuzzable.

The frontier measurement (DEEP_DIVE §Coverage frontier) found 32 one-sided
branch conditions in fuzzgoat; about 26 were unreachable from input because
the harness always called json_parse(): comments off, no memory limit, no
error buffer, malloc never failing. A trailing ``\\0 F`` pair now selects
json_parse_ex() settings; inputs without it parse exactly as before.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tests.conftest import requires_clang

REPO = Path(__file__).resolve().parents[1]
HARNESS = REPO / "targets" / "fuzzgoat_read.c"
VENDOR = (
    Path(os.environ.get("FUZZ_VENDOR_ROOT", Path.home() / "fuzzing" / "vendoring")) / "fuzzgoat"
)

# Harness edge ids (targets/fuzzgoat_read.c).
PARSED, PARSE_FAILED = 0x1002, 0x1004
TRAILER, COMMENTS, MEMORY_LIMIT, ERROR_BUF, FAIL_ALLOC = 0x1006, 0x1007, 0x1008, 0x1009, 0x100A

COMMENTS_ON, LIMIT_ON, ERRBUF_ON, FAILALLOC_ON = 0x01, 0x02, 0x04, 0x08

_STUB = """
#include <string.h>
static unsigned char hits[0x10000];
void __afl_map_edge(unsigned int id) { hits[id & 0xFFFF] = 1; }
int edge_hit(unsigned int id) { return hits[id & 0xFFFF]; }
void edges_reset(void) { memset(hits, 0, sizeof hits); }
"""


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    if not (VENDOR / "fuzzgoat.c").exists():
        pytest.skip("fuzzgoat not vendored (tools/vendor_fuzzgoat.sh)")
    tmp = tmp_path_factory.mktemp("goat")
    stub = tmp / "stub.c"
    stub.write_text(_STUB)
    so = tmp / "goat.so"
    cmd = [
        "clang", "-O1", "-g", "-shared", "-fPIC", f"-I{VENDOR}",
        str(HARNESS), str(VENDOR / "fuzzgoat.c"), str(stub), "-lm", "-o", str(so),
    ]  # fmt: skip
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return so


# fuzzgoat crashes by design: run each input in a child, report the edges hit.
_DRIVER = """
import ctypes, sys
lib = ctypes.CDLL(sys.argv[1])
data = bytes.fromhex(sys.argv[2])
lib.fuzz_shm_run(data, len(data))
print(",".join(str(i) for i in map(int, sys.argv[3:]) if lib.edge_hit(i)), flush=True)
"""
_IDS = (PARSED, PARSE_FAILED, TRAILER, COMMENTS, MEMORY_LIMIT, ERROR_BUF, FAIL_ALLOC)


def _run(lib, data: bytes) -> set[int]:
    cmd = [sys.executable, "-c", textwrap.dedent(_DRIVER), str(lib), data.hex(), *map(str, _IDS)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, f"target crashed on {data!r}: rc={r.returncode}"
    return {int(x) for x in r.stdout.strip().split(",") if x}


@requires_clang
def test_no_trailer_unchanged(lib):
    # Control: plain JSON parses as before, no settings path taken. Objects,
    # [] and one-byte strings hit fuzzgoat's planted bugs: avoid them.
    assert _run(lib, b"[1,2]") == {PARSED}
    assert _run(lib, b"/*c*/1") == {PARSE_FAILED}


@requires_clang
def test_comments_flag_enables_comments(lib):
    # Falsification: the same comment input only parses with bit 0 set.
    assert _run(lib, b"/*c*/1\x00" + bytes([COMMENTS_ON])) == {TRAILER, COMMENTS, PARSED}
    assert _run(lib, b"/*c*/1\x00\x00") == {TRAILER, PARSE_FAILED}


@requires_clang
def test_failing_allocator_is_survived(lib):
    # Adversarial: allocation 1 returns NULL; the parser must report failure, not crash.
    out = _run(lib, b'[1,"ab"]\x00' + bytes([FAILALLOC_ON]))
    assert {TRAILER, FAIL_ALLOC, PARSE_FAILED} <= out


@requires_clang
def test_error_buffer_reports_failure(lib):
    out = _run(lib, b"[1,\x00" + bytes([ERRBUF_ON]))
    assert {TRAILER, PARSE_FAILED, ERROR_BUF} <= out


@requires_clang
def test_memory_limit_rejects_large_input(lib):
    # High nibble 0 -> 64-byte budget: a long array exceeds it.
    big = b"[" + b"1," * 200 + b"1]"
    assert {TRAILER, MEMORY_LIMIT, PARSE_FAILED} <= _run(lib, big + b"\x00" + bytes([LIMIT_ON]))


@requires_clang
def test_short_inputs_and_nul_only(lib):
    # Adversarial: a lone trailer or a 1-byte input must not read out of bounds.
    assert _run(lib, b"\x00") <= {PARSE_FAILED, PARSED}
    assert TRAILER in _run(lib, b"\x00\x0f")
