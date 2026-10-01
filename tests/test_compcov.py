"""Tests for the COMPCOV layer of afl_shim.c — byte-level comparison
progress folded directly into the edge map.

Mirrors the harness in test_regression_cmplog_shim_merge.py: build the
shim into a target *executable* (not a preload .so, since COMPCOV needs
real edge machinery), attach a SysV SHM segment the way ShmCoverage does,
run the target, and read back the segment's header/table.

The metric under test is the cumulative "edge_count" field the shim
already maintains at SHM offset 16 (see afl_shim.c's metadata layout
comment): a partially-matching wide comparison should mint strictly more
distinct edges at $__AFL_COMPCOV_LEVEL=2 than at 0 or unset, for the exact
same input and the exact same real control flow.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import struct
import subprocess
from pathlib import Path

import pytest

from fuzzer_tool.adapters.shm import SHM_METADATA_SIZE, SHM_TAIL_SIZE

REPO = Path(__file__).parent.parent
SHIM = REPO / "src" / "fuzzer_tool" / "adapters" / "afl_shim.c"

needs_cc = pytest.mark.skipif(shutil.which("gcc") is None, reason="no C compiler")

NOBUILTIN = [
    "-fno-builtin-memcmp",
    "-fno-builtin-bcmp",
    "-fno-builtin-strcmp",
    "-fno-builtin-strncmp",
]

MAP_ENTRIES = 8192
SHM_HEADER = SHM_METADATA_SIZE
SHM_BYTES = MAP_ENTRIES * 8 + SHM_HEADER + SHM_TAIL_SIZE  # shim writes the distance tail

# Layer 1 (memcmp): 6 of 8 bytes match ("MAGICHD" vs "MAGICHDR"'s prefix,
# broken at index 6). Layer 2 (trace_cmp4, called directly since no clang
# in this environment to emit it from real IR): shares 2 low bytes then
# diverges at byte index 2.
_TARGET = """
#include <string.h>
#include <stdint.h>
extern void __sanitizer_cov_trace_cmp4(uint32_t a, uint32_t b);
int main(void) {
    char buf[16];
    memcpy(buf, "MAGICHDXXXXXXXXX", 16);
    volatile int r = memcmp(buf, "MAGICHDR", 8);
    __sanitizer_cov_trace_cmp4(0x00CDAB34u, 0x00EFAB34u);
    return r == 12345;
}
"""


def _cc(tmp_path, name, source, *, flags=()):
    src = tmp_path / f"{name}.c"
    src.write_text(source)
    out = tmp_path / name
    cmd = ["gcc", "-O0", "-g", *flags, "-o", str(out), str(src)]
    r = subprocess.run(cmd, capture_output=True, timeout=180)
    assert r.returncode == 0, r.stderr.decode()[:600]
    return out


def _cc_shim_target(tmp_path, name, source):
    flags = ["-D__AFL_CMPLOG=1", *NOBUILTIN, "-include", str(SHIM), "-ldl"]
    return _cc(tmp_path, name, source, flags=flags)


class _Shm:
    """A coverage segment sized the way ShmCoverage sizes it."""

    def __init__(self):
        self._libc = ctypes.CDLL("libc.so.6", use_errno=True)
        self._libc.shmget.restype = ctypes.c_int
        self._libc.shmat.restype = ctypes.c_void_p
        self._libc.shmat.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
        # shmid 0 is a legal id the shim rejects (it checks `<= 0`); take
        # two and use the second, as the cmplog shim-merge harness does.
        self._ids = [self._libc.shmget(0, SHM_BYTES, 0o1000 | 0o600) for _ in range(2)]
        self.shm_id = self._ids[-1]

    def env(self, **extra):
        e = dict(os.environ, __AFL_SHM_ID=str(self.shm_id), AFL_MAP_SIZE=str(MAP_ENTRIES))
        e.pop("LD_PRELOAD", None)
        e.update(extra)
        return e

    def edge_count(self) -> int:
        addr = self._libc.shmat(self.shm_id, None, 0)
        header = bytes((ctypes.c_ubyte * SHM_HEADER).from_address(addr))
        # offset 16: uint64 edge_count (monotonic new-slot insertion count)
        return struct.unpack_from("<Q", header, 16)[0]

    def close(self):
        for i in self._ids:
            self._libc.shmctl(i, 0, None)


@pytest.fixture
def shm():
    s = _Shm()
    yield s
    s.close()


@needs_cc
class TestCompcovEdgeSignal:
    def test_more_edges_at_level_2_than_disabled(self, tmp_path, shm):
        target = _cc_shim_target(tmp_path, "compcov_target", _TARGET)

        r_off = subprocess.run([str(target)], env=shm.env(), capture_output=True, timeout=30)
        assert r_off.returncode in (0, 1)
        edges_off = shm.edge_count()

        # Fresh segment for a fresh run: reuse would carry over the first
        # run's edges and only ever grow, not compare cleanly.
        shm.close()
        shm.__init__()

        r_on = subprocess.run(
            [str(target)],
            env=shm.env(__AFL_COMPCOV_LEVEL="2"),
            capture_output=True,
            timeout=30,
        )
        assert r_on.returncode in (0, 1)
        edges_on = shm.edge_count()

        assert edges_on > edges_off, (
            f"level=2 should mint COMPCOV marks beyond real control-flow edges: "
            f"off={edges_off} on={edges_on}"
        )

    def test_level_1_ignores_non_const_trace_cmp(self, tmp_path, shm):
        """trace_cmp4 above is the non-const callback; level 1 must not mark it."""
        target = _cc_shim_target(tmp_path, "compcov_target2", _TARGET)

        r_off = subprocess.run([str(target)], env=shm.env(), capture_output=True, timeout=30)
        assert r_off.returncode in (0, 1)
        edges_off = shm.edge_count()

        shm.close()
        shm.__init__()

        r_lvl1 = subprocess.run(
            [str(target)],
            env=shm.env(__AFL_COMPCOV_LEVEL="1"),
            capture_output=True,
            timeout=30,
        )
        assert r_lvl1.returncode in (0, 1)
        edges_lvl1 = shm.edge_count()

        # memcmp (Layer 1) only ever fires at level 2, and the harness's
        # only trace-cmp call is the non-const variant, so level 1 should
        # add nothing beyond the real control-flow edges.
        assert edges_lvl1 == edges_off

    def test_disabled_by_default(self, tmp_path, shm):
        """No $__AFL_COMPCOV_LEVEL at all must behave identically to '0'.

        This harness has no clang, so main() carries no real
        trace-pc-guard edges of its own -- every edge-table entry here
        comes from COMPCOV marks. With COMPCOV off (unset == level 0),
        the table must stay empty; that's the same "off" outcome the
        explicit __AFL_COMPCOV_LEVEL=0 case in the first test exercises.
        """
        target = _cc_shim_target(tmp_path, "compcov_target3", _TARGET)

        env = shm.env()
        assert "__AFL_COMPCOV_LEVEL" not in env
        r = subprocess.run([str(target)], env=env, capture_output=True, timeout=30)
        assert r.returncode in (0, 1)
        assert shm.edge_count() == 0


# ── Regressions: exec-stable, NUL-bounded, context-neutral marks ─────

needs_clang = pytest.mark.skipif(shutil.which("clang") is None, reason="no clang")

# Prints its own load address so a test can tell ASLR actually moved it.
_ASLR_TARGET = """
#include <stdio.h>
#include <string.h>
int main(void) {
    char buf[16];
    memcpy(buf, "MAGICHDXXXXXXXXX", 16);
    volatile int r = memcmp(buf, "MAGICHDR", 8);
    printf("%p\\n", (void *)main);
    return r == 12345;
}
"""

# a and b agree up to and including the NUL; bytes after it are argv[1][0]
# in a, 'Q' in b. strncmp must not see past the NUL.
_STRNCMP_TARGET = """
#include <string.h>
int main(int argc, char **argv) {
    char a[16], b[16];
    memset(a, argc > 1 ? argv[1][0] : 'Q', sizeof a);
    memset(b, 'Q', sizeof b);
    memcpy(a, "AB", 3);
    memcpy(b, "AB", 3);
    volatile int r = strncmp(a, b, sizeof a);
    return r == 12345;
}
"""

# Real edges after a partial memcmp: their ids must not depend on COMPCOV.
_CTX_TARGET = """
#include <string.h>
int main(void) {
    char buf[16];
    memcpy(buf, "MAGICHDXXXXXXXXX", 16);
    int r = memcmp(buf, "MAGICHDR", 8);
    if (r > 0) return 1;
    return 0;
}
"""

_CLANG_FLAGS = ["-O0", "-fPIE", "-pie", "-fno-omit-frame-pointer", "-D__AFL_CMPLOG=1", *NOBUILTIN]


def _clang(*args):
    r = subprocess.run(["clang", *args], capture_output=True, timeout=180)
    assert r.returncode == 0, r.stderr.decode()[:600]


def _clang_shim_target(tmp_path, name, source, *cov):
    """Build target + shim as separate objects.

    The shim is its own TU so ``cov`` flags (trace-pc-guard) instrument
    only the target, and the final link carries no coverage flag, so clang
    asks for no compiler-rt runtime.
    """
    src = tmp_path / f"{name}.c"
    src.write_text(source)
    obj, shim_obj, out = tmp_path / f"{name}.o", tmp_path / f"{name}_shim.o", tmp_path / name
    _clang(*_CLANG_FLAGS, *cov, "-c", str(src), "-o", str(obj))
    _clang(*_CLANG_FLAGS, "-c", str(SHIM), "-o", str(shim_obj))
    _clang("-pie", str(obj), str(shim_obj), "-o", str(out), "-ldl")
    return out


def _edge_ids(target, level, *args):
    """Run once on a fresh segment; return (edge-id set, stdout)."""
    s = _Shm()
    try:
        env = s.env(__AFL_COMPCOV_LEVEL=level, FUZZER_KEEP_ASLR="1")
        r = subprocess.run([str(target), *args], env=env, capture_output=True, timeout=30)
        assert r.returncode in (0, 1), r.stderr.decode()[:600]
        addr = s._libc.shmat(s.shm_id, None, 0)
        raw = bytes((ctypes.c_ubyte * (MAP_ENTRIES * 8)).from_address(addr + SHM_HEADER))
        s._libc.shmdt(ctypes.c_void_p(addr))
    finally:
        s.close()
    ids = {struct.unpack_from("<I", raw, i * 8)[0] for i in range(MAP_ENTRIES)} - {0}
    return ids, r.stdout


@needs_clang
class TestCompcovRegressions:
    def test_regression_compcov_ids_survive_aslr(self, tmp_path):
        target = _clang_shim_target(tmp_path, "compcov_aslr", _ASLR_TARGET)
        ids_a, out_a = _edge_ids(target, "2")
        ids_b, out_b = _edge_ids(target, "2")
        if out_a == out_b:
            pytest.skip("ASLR off: load base did not move")

        assert ids_a
        assert ids_a == ids_b

    def test_regression_compcov_strncmp_stops_at_nul(self, tmp_path):
        target = _clang_shim_target(tmp_path, "compcov_strncmp", _STRNCMP_TARGET)

        # Control: same input twice must mint the same number of marks.
        same_a, _ = _edge_ids(target, "2", "Q")
        same_b, _ = _edge_ids(target, "2", "Q")
        assert len(same_a) == len(same_b)
        assert same_a

        # Bytes after the NUL differ: strncmp's result is identical, so
        # COMPCOV's view must be too.
        diff, _ = _edge_ids(target, "2", "Z")
        assert len(same_a) == len(diff)

    def test_regression_compcov_keeps_real_edge_ids(self, tmp_path):
        target = _clang_shim_target(
            tmp_path, "compcov_ctx", _CTX_TARGET, "-fsanitize-coverage=trace-pc-guard"
        )

        # Control: two level-0 runs agree exactly.
        off_a, _ = _edge_ids(target, "0")
        off_b, _ = _edge_ids(target, "0")
        assert off_a == off_b
        assert off_a

        # Marks may only add ids, never rename the real edges after them.
        on, _ = _edge_ids(target, "2")
        assert off_a <= on
        assert len(on) > len(off_a)


# ── Regression: bit 31 is reserved for synthetic ids iff COMPCOV is on ─
#
# COMPCOV/DATAFLOW ids set bit 31. With $__AFL_COMPCOV_LEVEL > 0 real edge ids
# must never carry it, or a reader cannot tell a mark from a real edge and a
# mark can equal a real id. With COMPCOV off ids must be exactly the legacy
# ones: k=2 ids are < 2^24 by construction; k>=3 uses a full 32-bit FNV hash,
# where 27 of 52 real ids on compcov_gates carry bit 31, and saved corpora
# built from those ids must stay valid.

_LOOP_TARGET = """
#include <string.h>
int main(int argc, char **argv) {
    char buf[16];
    memcpy(buf, "MAGICHDXXXXXXXXX", 16);
    int acc = 0;
    for (int i = 0; i < 16; i++) {
        if (buf[i] > 'M') acc += 1; else if (buf[i] == 'X') acc += 2; else acc += 3;
        if (argc > 5) acc ^= i;
    }
    volatile int r = memcmp(buf, "MAGICHDR", 8);
    return (r == 12345) + (acc == -1);
}
"""

_MASK31 = 0x7FFFFFFF


def _ngram_target(tmp_path, k):
    src = tmp_path / f"loop_k{k}.c"
    src.write_text(_LOOP_TARGET)
    out = tmp_path / f"loop_k{k}"
    flags = [
        "-O2", "-fPIE", "-pie", "-fno-omit-frame-pointer", "-D__AFL_CMPLOG=1",
        f"-D__AFL_NGRAM_K={k}", *NOBUILTIN, "-fsanitize-coverage=trace-pc-guard",
        "-include", str(SHIM),
    ]
    _clang(*flags, "-o", str(out), str(src), "-ldl")
    return out


@needs_clang
class TestSyntheticIdBit:
    @pytest.mark.parametrize("k", [2, 3])
    @pytest.mark.parametrize("level", ["1", "2"])
    def test_regression_real_ids_never_carry_bit31_when_compcov_on(self, tmp_path, k, level):
        target = _ngram_target(tmp_path, k)
        on, _ = _edge_ids(target, level)
        assert len(on) >= 10, "target too small to be a meaningful census"
        marks = {i for i in on if i >> 31}
        real = on - marks
        off, _ = _edge_ids(target, "0")
        # Reserving the bit changes nothing about a real id except that bit:
        # the same set of real edges, minus bit 31.
        assert real == {i & _MASK31 for i in off}, f"k={k} level={level}: real ids moved"

    @pytest.mark.parametrize("k", [2, 3])
    def test_regression_compcov_marks_are_tagged_and_disjoint(self, tmp_path, k):
        target = _ngram_target(tmp_path, k)
        off, _ = _edge_ids(target, "0")
        on, _ = _edge_ids(target, "2")
        marks = {i for i in on if i >> 31} - {i for i in off if i >> 31}
        real_on = {i for i in on if not i >> 31}
        assert marks, "level 2 minted no COMPCOV marks"
        assert not (real_on & marks), "a real id equals a COMPCOV mark"
        assert len(on) >= len(real_on) + len(marks)

    def test_regression_compcov_off_leaves_ngram_ids_unmasked(self, tmp_path):
        """Off means legacy: the k=3 FNV hash keeps its full 32 bits."""
        target = _ngram_target(tmp_path, 3)
        off, _ = _edge_ids(target, "0")
        assert any(i >> 31 for i in off), (
            "k=3 ids never carry bit 31 with COMPCOV off -- either the hash "
            "changed width or the reservation leaked into the off path"
        )

    def test_regression_compcov_off_k2_ids_unchanged(self, tmp_path):
        target = _ngram_target(tmp_path, 2)
        off, _ = _edge_ids(target, "0")
        assert off and all(i < (1 << 24) for i in off)
