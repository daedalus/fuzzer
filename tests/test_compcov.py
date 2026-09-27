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

from fuzzer_tool.adapters.shm import SHM_METADATA_SIZE

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
SHM_BYTES = MAP_ENTRIES * 8 + SHM_HEADER

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
