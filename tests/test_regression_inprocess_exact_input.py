"""Regression: in-process inputs lived in Python-managed memory ASAN can't see.

``(c_uint8 * n).from_buffer_copy`` stores n <= 16 bytes inside the ctypes
object and n <= 512 in pymalloc arenas: no redzone after the input, so
``buf[n]`` read neighbouring objects silently. Inputs now live in an
exact-size libc ``malloc`` block (``adapters/heap_input.py``).
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys

import pytest

from fuzzer_tool.adapters.heap_input import HeapInput
from tests.support.asan_overread import (
    OVERREAD,
    REPORT,
    asan_env,
    asan_reports,
    build_overread_so,
)

_TIMEOUT_S = 60

# Sizes: inline in the ctypes object, pymalloc arena, libc malloc.
_INLINE = 8
_PYMALLOC = 100
_LARGE = 600

_SCRIPT = r"""
import sys
from fuzzer_tool.adapters.inprocess import InProcessRunner
so, mode, *sizes = sys.argv[1:]
r = InProcessRunner(so, "fuzz_shm_run", direct=True, direct_lite=mode == "lite")
for n in map(int, sizes):
    first = int(n > 0 and n % 2 == 0)  # even size: overread probe
    data = (bytes([OVERREAD]) if first else b"S") * n
    rc, _ = r.run_one(data)
    print("RC", rc, flush=True)
r.stop()
""".replace("OVERREAD", str(OVERREAD))


def _run(so: str, log_dir, mode: str, *sizes: int) -> subprocess.CompletedProcess:
    """Drive *sizes* through a direct runner in an ASAN-preloaded child."""
    return subprocess.run(
        [sys.executable, "-c", _SCRIPT, so, mode, *map(str, sizes)],
        capture_output=True,
        text=True,
        env=asan_env(dict(os.environ), log_dir),
        timeout=_TIMEOUT_S,
    )


@pytest.mark.parametrize("mode", ["direct", "lite"])
@pytest.mark.parametrize("n", [_INLINE, _PYMALLOC, _LARGE])
def test_regression_small_overread_caught(tmp_path, mode, n) -> None:
    """Falsification: buf[n] past an n-byte input must hit ASAN's redzone."""
    so = build_overread_so(tmp_path)
    _run(so, tmp_path, mode, n)
    assert REPORT in asan_reports(tmp_path)


@pytest.mark.parametrize("mode", ["direct", "lite"])
def test_in_bounds_sizes_clean(tmp_path, mode) -> None:
    """Control + adversarial: odd sizes (incl. 0, 1) and repeats stay clean."""
    so = build_overread_so(tmp_path)
    sizes = (0, 1, 17, 17, 513, 3, 0)
    r = _run(so, tmp_path, mode, *sizes)
    assert r.returncode == 0, asan_reports(tmp_path)[-500:]
    assert r.stdout.count("RC 0") == len(sizes)
    assert asan_reports(tmp_path) == ""


class TestHeapInput:
    @pytest.mark.parametrize("n", [0, 1, 255, 2048, 20000])
    def test_exact_size_and_content(self, n) -> None:
        data = bytes((i * 31) & 0xFF for i in range(n))
        h = HeapInput()
        try:
            buf = h.place(data)
            assert ctypes.sizeof(buf) == n
            assert bytes(buf) == data
        finally:
            h.close()

    def test_outside_python_object(self) -> None:
        """Falsification: the bytes must not sit inside the ctypes object."""
        h = HeapInput()
        try:
            buf = h.place(b"x" * _INLINE)
            start, end = id(buf), id(buf) + sys.getsizeof(buf)
            assert not start <= ctypes.addressof(buf) < end
        finally:
            h.close()

    @pytest.mark.parametrize("kind", [bytearray, memoryview])
    def test_buffer_protocol_copies(self, kind) -> None:
        """Adversarial: mutators hand bytearrays; views must copy, not alias."""
        src = bytearray(b"\x00\x01\xfe\xff")
        h = HeapInput()
        try:
            buf = h.place(kind(src))
            src[0] = 0x7F
            assert bytes(buf) == b"\x00\x01\xfe\xff"
        finally:
            h.close()

    @pytest.mark.parametrize("n", [4, _LARGE])
    def test_same_length_reuses_block(self, n) -> None:
        h = HeapInput()
        try:
            h.place(b"a" * n)  # large: the first repeat fills the slot
            a = ctypes.addressof(h.place(b"b" * n))
            buf = h.place(b"z" * n)
            assert ctypes.addressof(buf) == a
            assert bytes(buf) == b"z" * n
        finally:
            h.close()

    def test_large_length_change_exact(self) -> None:
        """Adversarial: a shorter large input must not reuse the longer slot."""
        h = HeapInput()
        try:
            h.place(b"a" * (_LARGE + 1))
            buf = h.place(b"b" * _LARGE)
            assert ctypes.sizeof(buf) == _LARGE
            assert bytes(buf) == b"b" * _LARGE
        finally:
            h.close()

    def test_close_idempotent(self) -> None:
        h = HeapInput()
        h.place(b"abc")
        h.close()
        h.close()
        assert bytes(h.place(b"de")) == b"de"
        h.close()
