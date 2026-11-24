"""Follow-ups to the 2026-11-24 review (commit aa9f1e3).

Four loose ends, each pinned by a test that failed before its fix:

1. Fast-path temp file: a forked child must not unlink the parent's file
   (atexit ran in the child) nor share the parent's file offset.
2. ELF ``_cstr``: a symbol name is clipped at the end of its string table,
   not read on into whatever section follows.
3. ``__AFL_CMPLOG_SITE_CAP``: digits-only, like every other numeric knob.
4. Preload shim: a stack-overflow SIGSEGV still flushes the cmplog counters
   (the crash handler runs on an alternate stack).
"""

from __future__ import annotations

import os
import signal
import struct
import subprocess
from pathlib import Path

import pytest

from fuzzer_tool.adapters import process
from fuzzer_tool.core import elf
from tests.conftest import requires_gcc

SHIM = Path(__file__).resolve().parents[1] / "src/fuzzer_tool/adapters/afl_shim.c"


# ── 1. fast-path file vs fork ─────────────────────────────────────────


@pytest.fixture
def fast_file():
    """A fresh fast-path temp file, torn down by the owning process."""
    process._close_fast_path_file()
    name = process._get_fast_path_file()
    yield name
    process._close_fast_path_file()


def _in_child(fn) -> int:
    """Run *fn* in a forked child; its exit status (0 = fn returned truthy)."""
    pid = os.fork()
    if pid == 0:
        code = 1
        try:
            code = 0 if fn() else 1
        finally:
            os._exit(code)
    _, status = os.waitpid(pid, 0)
    return os.waitstatus_to_exitcode(status)


def test_forked_child_exit_keeps_parent_file(fast_file):
    """The child's atexit-style close must not unlink the parent's name."""
    assert _in_child(lambda: process._close_fast_path_file() or True) == 0
    assert os.path.exists(fast_file)


def test_owner_close_unlinks_the_file(fast_file):
    process._close_fast_path_file()
    assert not os.path.exists(fast_file)


def test_forked_child_gets_its_own_file(fast_file):
    """Sharing the parent's fd shares its offset: lseek+write would interleave."""
    assert _in_child(lambda: process._get_fast_path_file() != fast_file) == 0


# ── 2. _cstr clamps at the string table ───────────────────────────────


def test_cstr_stops_at_limit():
    blob = b"abcdef\x00tail\x00"
    assert elf._cstr(blob, 0, len(blob)) == b"abcdef"
    assert elf._cstr(blob, 0, 3) == b"abc"
    assert elf._cstr(blob, 4, 3) == b""
    assert elf._cstr(blob, 99, 120) == b""


_FUNC = "zz_unique_function_name"
_TARGET = f"void _start(void) {{}}\nint {_FUNC}(void) {{ return 1; }}\n"


def _sections(raw: bytes) -> dict[str, tuple[int, int]]:
    """name -> (header offset, payload offset) for a 64-bit LE ELF."""
    (shoff,) = struct.unpack_from("<Q", raw, 0x28)
    entsz, num, strndx = struct.unpack_from("<HHH", raw, 0x3A)
    shstr = struct.unpack_from("<Q", raw, shoff + strndx * entsz + 24)[0]
    out = {}
    for i in range(num):
        hdr = shoff + i * entsz
        nm = struct.unpack_from("<I", raw, hdr)[0]
        end = raw.index(b"\x00", shstr + nm)
        out[raw[shstr + nm : end].decode()] = (hdr, struct.unpack_from("<Q", raw, hdr + 24)[0])
    return out


@requires_gcc
def test_func_name_is_clipped_to_strtab(tmp_path):
    src = tmp_path / "t.c"
    src.write_text(_TARGET)
    exe = tmp_path / "t"
    r = subprocess.run(
        ["gcc", "-nostdlib", "-static", "-O0", str(src), "-o", str(exe)],
        capture_output=True, text=True,
    )  # fmt: skip
    if r.returncode != 0:
        pytest.skip(r.stderr[-300:])
    raw = bytearray(exe.read_bytes())
    assert _FUNC in elf.func_table(bytes(raw)).names

    hdr, off = _sections(bytes(raw))[".strtab"]
    at = raw.index(_FUNC.encode() + b"\x00", off) - off
    struct.pack_into("<Q", raw, hdr + 32, at + 4)  # strtab ends 4 bytes in

    names = elf.func_table(bytes(raw)).names
    assert _FUNC not in names
    assert _FUNC[:4] in names


# ── 3 + 4. shim env parsing and the preload crash handler ─────────────

_PROBE = r"""
#include <stdio.h>
int main(void) {
    printf("%u %u\n", (unsigned)__afl_cmplog_site_cap, (unsigned)CMPLOG_SITE_CAP_MAX);
    return 0;
}
"""

CAP_CASES = [
    ("12", 12),
    ("1", 1),
    ("2000000000", 0xFFFF),  # valid but huge: clamped to the max
    ("12junk", 0),
    (" 12", 0),
    ("+12", 0),
    ("-5", 0),
    ("0x10", 0),
    ("99999999999999999999", 0),
    ("", 0),
]


@pytest.fixture(scope="module")
def probe(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("cap")
    src = d / "p.c"
    src.write_text(_PROBE)
    exe = d / "p"
    r = subprocess.run(
        ["gcc", "-O1", "-D__AFL_PRELOAD_ONLY", "-include", str(SHIM), str(src), "-o", str(exe), "-ldl"],
        capture_output=True, text=True,
    )  # fmt: skip
    if r.returncode != 0:
        pytest.skip(r.stderr[-300:])
    return exe


@requires_gcc
@pytest.mark.parametrize(("val", "want"), CAP_CASES)
def test_site_cap_is_digits_only(probe, val, want):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("__AFL_", "_CMPLOG"))}
    env["__AFL_CMPLOG_SITE_CAP"] = val
    out = subprocess.run([str(probe)], capture_output=True, text=True, env=env, timeout=30)
    assert out.stdout.split()[0] == str(want), out.stderr


_OVERFLOW = r"""
#include <string.h>
static int deep(int n) {
    volatile char pad[4096];
    pad[0] = (char)n;
    return deep(n + 1) + pad[0];
}
int main(int argc, char **argv) {
    volatile int r = 0;
    r += memcmp("HELLO_WORLD", argv[0], 11) == 0;
    r += memcmp("HELLO_WORLD", argv[0], 11) == 0;
    r += memcmp("HELLO_WORLD", argv[0], 11) == 0;
    return deep(r);
}
"""


@requires_gcc
def test_stack_overflow_still_flushes_counts(tmp_path):
    """The handler needs an alternate stack: the faulting one has no room."""
    src = tmp_path / "o.c"
    src.write_text(_OVERFLOW)
    obj = tmp_path / "shim.o"
    exe = tmp_path / "o"
    steps = (
        ["gcc", "-O2", "-fPIC", "-D__AFL_PRELOAD_ONLY", "-c", str(SHIM), "-o", str(obj)],
        ["gcc", "-O0", "-fno-builtin-memcmp", str(src), str(obj), "-o", str(exe), "-ldl"],
    )  # fmt: skip
    for cmd in steps:
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            pytest.skip(r.stderr[-300:])

    counts = tmp_path / "counts.txt"
    env = {k: v for k, v in os.environ.items() if not k.startswith(("__AFL_", "_CMPLOG"))}
    env["_CMPLOG_COUNTS"] = str(counts)
    env["_CMPLOG_OUT"] = str(tmp_path / "records.cmplog")
    run = subprocess.run([str(exe)], capture_output=True, text=True, env=env, timeout=60)

    assert run.returncode == -signal.SIGSEGV
    assert counts.exists(), "overflow crash lost the cmplog counters"
    assert "CNT memcmp" in counts.read_text()
