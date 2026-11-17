"""C11 seed weight (AFL++ C11 mode, ICSE'26): locals of the deepest function.

DWARF  function_local_counts: low_pc -> params + locals it owns
table  guard_table: pc-table entry guard -> locals (u8)
shim   per-exec max over entered functions, stamped with the generation
weight c11_weight: 1.1 + 0.9 * t^2, t = min-max normalised c11
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from fuzzer_tool.adapters.sancov_tables import SANCOV_ENV, SancovTables
from fuzzer_tool.adapters.shm import BlockBitmapShm, C11Shm, ShmCoverage
from fuzzer_tool.core.c11 import C11_MAX, c11_bounds, c11_weight, guard_table
from fuzzer_tool.core.dwarf import function_local_counts
from tests.conftest import requires_clang

SHIM = Path(__file__).parent.parent / "src" / "fuzzer_tool" / "adapters" / "afl_shim.c"
MAP_ENTRIES = 8192
BLOCK_BITS = 4096

# Locals = params + variables a function owns, nested blocks included,
# inlined callees' variables excluded.
_SRC = r"""
#include <string.h>
volatile int sink;
__attribute__((noinline)) void none(void) { sink++; }
__attribute__((noinline)) void two(int a) { int b = a; sink += b; }
__attribute__((noinline)) void nested(int a) {
    int x = a;
    { int y = x; { int z = y; sink += z; } }
}
static inline __attribute__((always_inline)) int helper(int q) { int w = q * 2; return w; }
__attribute__((noinline)) void caller(int a) { sink += helper(a); }
__attribute__((noinline)) void five(int a, int b) {
    int c = a, d = b, e = c + d;
    sink += e;
}
int main(int argc, char **argv) {
    const char *in = argc > 1 ? argv[1] : "";
    none();
    two(1);
    nested(2);
    caller(3);
    if (in[0] == 'B') five(4, 5);
    return 0;
}
"""

EXPECTED = {"none": 0, "two": 2, "nested": 4, "caller": 1, "five": 5, "main": 3}


def _compile(tmp_path, *flags, name="t"):
    src = tmp_path / f"{name}.c"
    src.write_text(_SRC)
    exe = tmp_path / name
    r = subprocess.run(
        ["clang", "-O0", *flags, str(src), "-o", str(exe)], capture_output=True, text=True
    )
    assert r.returncode == 0, r.stderr[-800:]
    return exe


def _symbols(exe) -> dict[str, int]:
    out = subprocess.run(["nm", "--defined-only", str(exe)], capture_output=True, text=True).stdout
    syms = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[1] in "Tt":
            syms[parts[2]] = int(parts[0], 16)
    return syms


@requires_clang
class TestFunctionLocalCounts:
    @pytest.mark.parametrize("dwarf", ["-gdwarf-4", "-gdwarf-5"])
    def test_counts_match_source(self, tmp_path, dwarf):
        exe = _compile(tmp_path, dwarf)
        counts = function_local_counts(str(exe))
        syms = _symbols(exe)

        got = {name: counts.get(syms[name]) for name in EXPECTED}
        assert got == EXPECTED

    def test_no_debug_info_is_empty(self, tmp_path):
        """Falsification: without -g there is nothing to count."""
        exe = _compile(tmp_path)
        assert function_local_counts(str(exe)) == {}

    def test_garbage_file_is_empty(self, tmp_path):
        """Adversarial: a non-ELF never raises."""
        bad = tmp_path / "bad"
        bad.write_bytes(b"\x7fELF" + os.urandom(256))
        assert function_local_counts(str(bad)) == {}


class TestGuardTable:
    def test_entries_map_to_locals(self):
        entries = np.array([2, 5], dtype=np.uint32)
        vaddrs = np.array([0x1000, 0x2000], dtype=np.uint64)
        table = guard_table(entries, vaddrs, {0x1000: 3, 0x2000: 9}, size=8)

        expected = np.zeros(8, dtype=np.uint8)
        expected[2], expected[5] = 3, 9
        assert np.array_equal(table, expected)

    def test_unknown_pc_and_out_of_range_guard_skipped(self):
        """Adversarial: no DWARF match -> 0; a guard past the table is dropped."""
        entries = np.array([1, 99], dtype=np.uint32)
        vaddrs = np.array([0xDEAD, 0x1000], dtype=np.uint64)
        table = guard_table(entries, vaddrs, {0x1000: 3}, size=4)
        assert not table.any()

    def test_clamped_to_u8(self):
        table = guard_table(
            np.array([0], dtype=np.uint32), np.array([0x10], dtype=np.uint64), {0x10: 1000}, size=1
        )
        assert int(table[0]) == C11_MAX


class TestWeight:
    def test_curve_endpoints(self):
        lo, hi = 2, 10
        assert c11_weight(lo, lo, hi) == pytest.approx(1.1)
        assert c11_weight(hi, lo, hi) == pytest.approx(1.1 + 0.9)
        mid = (lo + hi) / 2
        assert c11_weight(int(mid), lo, hi) == pytest.approx(1.1 + 0.9 * 0.5**2)

    def test_monotone(self):
        """Falsification: more locals never weighs less."""
        ws = [c11_weight(c, 1, 20) for c in range(1, 21)]
        assert ws == sorted(ws)

    def test_unknown_is_neutral(self):
        """c11 0 = not measured: AFL++ applies no factor."""
        assert c11_weight(0, 1, 20) == 1.0
        assert c11_weight(5, 0, 0) == 1.0

    def test_flat_range(self):
        """lo == hi: AFL++ lowers lo by one, so every seed gets the top weight."""
        assert c11_weight(7, 7, 7) == pytest.approx(2.0)

    def test_bounds_ignore_unmeasured(self):
        assert c11_bounds([0, 4, 9, 0, 2]) == (2, 9)
        assert c11_bounds([0, 0]) == (0, 0)


@requires_clang
class TestShimMax:
    def _build(self, tmp_path):
        src = tmp_path / "s.c"
        src.write_text(_SRC)
        obj = tmp_path / "s.o"
        exe = tmp_path / "s"
        cc = [
            "clang", "-O0", "-g", "-fno-omit-frame-pointer", "-D__AFL_CTX_SENSITIVE=0",
            "-fsanitize-coverage=trace-pc-guard,pc-table", "-include", str(SHIM),
            "-c", str(src), "-o", str(obj),
        ]  # fmt: skip
        r = subprocess.run(cc, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-800:]
        r = subprocess.run(
            ["clang", str(obj), "-o", str(exe), "-ldl"], capture_output=True, text=True
        )
        assert r.returncode == 0, r.stderr[-800:]
        return exe

    @pytest.fixture
    def segs(self):
        cov = ShmCoverage(size=MAP_ENTRIES)
        blocks = BlockBitmapShm(BLOCK_BITS)
        c11 = C11Shm(BLOCK_BITS)
        tables = SancovTables()
        yield cov, blocks, c11, tables
        for seg in (cov, blocks, c11, tables):
            seg.cleanup()

    def _run(self, exe, segs, arg):
        cov, blocks, c11, tables = segs
        env = dict(
            os.environ,
            __AFL_SHM_ID=str(cov.shm_id),
            AFL_MAP_SIZE=str(MAP_ENTRIES),
            __AFL_BLOCK_BITMAP_ID=blocks.env_id,
            __AFL_C11_SHM_ID=c11.env_id,
            **{SANCOV_ENV: tables.path},
        )
        env.pop("LD_PRELOAD", None)
        r = subprocess.run([str(exe), arg], env=env, capture_output=True, timeout=30)
        assert r.returncode == 0, r.stderr[-800:]

    def _fill(self, exe, segs):
        _, _, c11, tables = segs
        rec = [r for r in tables.records() if r.entry_indices().size][0]
        entry = rec.pcs_flags() & np.uint64(1) != 0
        c11.fill(
            guard_table(
                rec.entry_indices(),
                rec.pcs_vaddrs()[entry],
                function_local_counts(str(exe)),
                BLOCK_BITS,
            )
        )

    def test_max_follows_deepest_function(self, tmp_path, segs):
        exe = self._build(tmp_path)
        cov, _, c11, _ = segs
        self._run(exe, segs, "A")  # writes the pc-table dump
        self._fill(exe, segs)

        cov.reset_edge_map()
        self._run(exe, segs, "A")
        always = max(v for k, v in EXPECTED.items() if k != "five")
        assert c11.read(cov.read_generation()) == always

        cov.reset_edge_map()
        self._run(exe, segs, "B")
        assert c11.read(cov.read_generation()) == EXPECTED["five"]

    def test_stale_generation_reads_zero(self, tmp_path, segs):
        """Adversarial: a max from an earlier exec is never reported as this one's."""
        exe = self._build(tmp_path)
        cov, _, c11, _ = segs
        self._run(exe, segs, "A")
        self._fill(exe, segs)

        cov.reset_edge_map()
        self._run(exe, segs, "B")
        cov.reset_edge_map()  # next exec, not run yet
        assert c11.read(cov.read_generation()) == 0
