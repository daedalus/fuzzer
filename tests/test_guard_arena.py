"""Guard-page input placement (``--input-placement guard``).

The input ends flush against a ``PROT_NONE`` page, so a read of the first
byte past it faults with ``si_addr`` inside that page; the runner labels such
a fault ``guard-overread``. Exact placement (heap copy) misses the same read
in a build without ASAN, which is the gap this closes.
"""

from __future__ import annotations

import ast
import ctypes
import mmap
import shutil
import signal
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from fuzzer_tool.adapters.guard_arena import GUARD_OVERREAD, GuardArena

REPO_ROOT = Path(__file__).resolve().parents[1]
SHIM = REPO_ROOT / "src" / "fuzzer_tool" / "adapters" / "afl_shim.c"
PAGE = mmap.PAGESIZE

# Reads buf[len]: one byte past the input, the smallest overread.
OVERREAD_SRC = r"""
#include <stddef.h>
__attribute__((visibility("default")))
int fuzz_test(const unsigned char *buf, size_t len) {
    volatile unsigned char past = buf[len];
    return past == 0x7f ? 1 : 0;
}
"""

# Faults far from any guard: a plain wild-pointer crash.
WILD_SRC = r"""
#include <stddef.h>
__attribute__((visibility("default")))
int fuzz_test(const unsigned char *buf, size_t len) {
    volatile unsigned char *p = (volatile unsigned char *)16;
    return *p + (int)len + buf[0];
}
"""


@pytest.fixture
def arena():
    a = GuardArena()
    yield a
    a.close()


def _end(buf: ctypes.Array) -> int:
    return ctypes.addressof(buf) + len(buf)


class TestPlacement:
    def test_input_ends_at_guard(self, arena):
        buf = arena.place(b"abc")

        assert _end(buf) == arena.guard_addr
        assert bytes(buf) == b"abc"

    @pytest.mark.parametrize("n", [0, 1, PAGE - 1, PAGE, PAGE + 1, 3 * PAGE + 7])
    def test_sizes_around_page(self, arena, n):
        data = bytes(i & 0xFF for i in range(n))
        buf = arena.place(data)

        assert len(buf) == n
        assert bytes(buf) == data
        assert _end(buf) == arena.guard_addr

    def test_shrink_after_grow_reuses_mapping(self, arena):
        arena.place(b"x" * (4 * PAGE))
        guard = arena.guard_addr

        buf = arena.place(b"yz")

        assert arena.guard_addr == guard
        assert bytes(buf) == b"yz"

    @pytest.mark.parametrize("kind", [bytes, bytearray, memoryview])
    def test_buffer_types(self, arena, kind):
        buf = arena.place(kind(b"hello"))

        assert bytes(buf) == b"hello"

    def test_cached_length_follows_remap(self, arena):
        """Adversarial: a length seen before a grow must land in the new mapping."""
        arena.place(b"ab")
        arena.place(b"x" * (2 * PAGE))

        buf = arena.place(b"cd")

        assert _end(buf) == arena.guard_addr
        assert bytes(buf) == b"cd"

    def test_cache_bounded(self, arena):
        from fuzzer_tool.adapters.guard_arena import _MAX_CACHED

        for n in range(_MAX_CACHED + 5):
            arena.place(bytes(n % PAGE))

        assert len(arena._bufs) <= _MAX_CACHED

    def test_copy_not_alias(self, arena):
        src = bytearray(b"abcd")
        buf = arena.place(src)

        src[0] = ord("Z")

        assert bytes(buf) == b"abcd"


class TestOwns:
    def test_guard_page_owned(self, arena):
        g = arena.guard_addr

        assert arena.owns(g)
        assert arena.owns(g + PAGE - 1)

    def test_neighbours_not_owned(self, arena):
        """Falsification: the last input byte and the byte past the guard are not overreads."""
        g = arena.guard_addr

        assert not arena.owns(g - 1)
        assert not arena.owns(g + PAGE)
        assert not arena.owns(None)
        assert not arena.owns(0)

    def test_closed_owns_nothing(self):
        a = GuardArena()
        g = a.guard_addr
        a.close()

        assert not a.owns(g)
        a.close()  # idempotent


def _cc() -> str | None:
    return shutil.which("clang") or shutil.which("cc")


def _build(tmp_path: Path, name: str, src: str) -> str:
    cc = _cc()
    if cc is None:
        pytest.skip("no C compiler")
    c = tmp_path / f"{name}.c"
    c.write_text(src)
    so = tmp_path / f"{name}.so"
    proc = subprocess.run(
        [cc, "-O1", "-fPIC", "-shared", "-o", str(so), str(c), str(SHIM)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        pytest.skip(f"target build failed: {proc.stderr[-400:]}")
    return str(so)


# Runs in a child: the guarded call recovers SIGSEGV, but a regression that
# lets the fault escape must kill a throwaway process, not pytest.
_PROBE = textwrap.dedent(
    """
    import sys
    from fuzzer_tool.adapters.inprocess import InProcessRunner
    from fuzzer_tool.core.run_mode import InputPlacement

    so, placement, mode = sys.argv[1], sys.argv[2], sys.argv[3]
    r = InProcessRunner(
        so, "fuzz_test", timeout=5.0,
        direct=(mode == "direct"), direct_lite=(mode == "lite"),
        placement=InputPlacement[placement],
    )
    out = []
    for data in (b"A" * 5, b"B" * 4096, b"C"):
        rc, _ = r.run_one(data)
        arena = r._arena
        out.append({
            "rc": rc,
            "label": r._last_fault_label,
            "at_guard": bool(arena) and arena.owns(r._last_fault_addr),
        })
    r.stop()
    print(repr(out))
    """
)


def _probe(so: str, placement: str, mode: str) -> list[dict]:
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE, so, placement, mode],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return ast.literal_eval(proc.stdout.strip().splitlines()[-1])


SEGV = int(signal.SIGSEGV)


class TestRunner:
    @pytest.mark.parametrize("mode", ["lite", "direct"])
    def test_overread_labelled(self, tmp_path, mode):
        so = _build(tmp_path, "overread", OVERREAD_SRC)
        crash_rc = -SEGV if mode == "lite" else 128 + SEGV

        runs = _probe(so, "GUARD", mode)

        for run in runs:
            assert run["rc"] == crash_rc
            assert run["label"] == GUARD_OVERREAD
            assert run["at_guard"]

    @pytest.mark.parametrize("mode", ["lite", "direct"])
    def test_exact_misses_overread(self, tmp_path, mode):
        """Falsification: without the guard the same one-byte overread is silent."""
        so = _build(tmp_path, "overread", OVERREAD_SRC)

        runs = _probe(so, "EXACT", mode)

        for run in runs:
            assert run["rc"] in (0, 1)
            assert run["label"] == ""

    def test_wild_fault_not_labelled(self, tmp_path):
        """Adversarial: a SEGV away from the guard stays a plain crash."""
        so = _build(tmp_path, "wild", WILD_SRC)

        runs = _probe(so, "GUARD", "lite")

        for run in runs:
            assert run["rc"] == -SEGV
            assert run["label"] == ""
            assert not run["at_guard"]


class TestCrashLabel:
    """``corpus_manager._label_fault``: the runner's label reaches the sidecar."""

    def _meta(self):
        from fuzzer_tool.core.crash_metadata import CrashMetadata

        return CrashMetadata()

    def test_guard_label_sets_error_and_access(self):
        from types import SimpleNamespace

        from fuzzer_tool.services.corpus_manager import _label_fault

        meta = self._meta()
        _label_fault(meta, SimpleNamespace(_last_fault_label=GUARD_OVERREAD))

        assert meta.error_type == GUARD_OVERREAD
        assert meta.access_type == "READ"

    def test_sanitizer_type_wins(self):
        """Adversarial: an ASAN report already named the bug; keep it."""
        from types import SimpleNamespace

        from fuzzer_tool.services.corpus_manager import _label_fault

        meta = self._meta()
        meta.error_type = "heap-buffer-overflow"
        _label_fault(meta, SimpleNamespace(_last_fault_label=GUARD_OVERREAD))

        assert meta.error_type == "heap-buffer-overflow"

    def test_no_runner_or_label_is_noop(self):
        from types import SimpleNamespace

        from fuzzer_tool.services.corpus_manager import _label_fault

        meta = self._meta()
        _label_fault(meta, None)
        _label_fault(meta, SimpleNamespace(_last_fault_label=""))

        assert not meta.error_type
        assert not meta.access_type


# ── CLI wiring ─────────────────────────────────────────────────────────


def _fuzz_kwargs(monkeypatch, tmp_path, extra: list[str]) -> dict:
    from fuzzer_tool.cli import commands

    exe = tmp_path / "t"
    exe.write_bytes(b"\x7fELF")
    exe.chmod(0o755)
    captured: dict = {}
    monkeypatch.setattr(commands, "Fuzzer", lambda **kw: captured.update(kw) or MagicMock())
    argv = ["fuzzer-tool", "fuzz", str(exe), "-d", str(tmp_path / "c"), *extra]
    monkeypatch.setattr(sys, "argv", argv)
    commands.main()
    return captured


class TestCli:
    def test_default_exact(self, monkeypatch, tmp_path):
        from fuzzer_tool.core.run_mode import InputPlacement

        kw = _fuzz_kwargs(monkeypatch, tmp_path, [])

        assert kw["input_placement"] is InputPlacement.EXACT

    def test_hail_mary_keeps_exact(self, monkeypatch, tmp_path):
        from fuzzer_tool.core.run_mode import InputPlacement

        kw = _fuzz_kwargs(monkeypatch, tmp_path, ["--hail-mary"])

        assert kw["input_placement"] is InputPlacement.EXACT

    def test_guard_reaches_fuzzer(self, monkeypatch, tmp_path):
        from fuzzer_tool.core.run_mode import InputPlacement

        kw = _fuzz_kwargs(monkeypatch, tmp_path, ["--input-placement", "guard"])

        assert kw["input_placement"] is InputPlacement.GUARD

    def test_rejects_unknown(self, monkeypatch, tmp_path):
        """Adversarial: a typo fails loudly instead of running unguarded."""
        with pytest.raises(SystemExit):
            _fuzz_kwargs(monkeypatch, tmp_path, ["--input-placement", "guards"])
