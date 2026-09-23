"""Regression: distinct control-flow edges must reach Python as distinct ids.

Measured on fuzzgoat (clang 18, trace-pc-guard, 250-input corpus) with a
logging tracer standing in for the shim: 344 real (prev, cur) edges arrived
as 145 edge ids. Two independent causes, each pinned below.

* ``edge_id |= 1`` erased bit 0 of cur_loc on every edge, so a block's two
  successors numbered 2k and 2k+1 -- very often the two sides of one branch
  -- were one edge (80 of the 344 by that alone).
* Guards were numbered 1..N and used raw, so every id sat below ~2N and XORs
  of small neighbouring integers collided systematically (8 edges on one id).
  Hand-written ``__afl_map_edge(0x1100 + depth)`` ids had the same problem:
  ``prev >> 1`` merges predecessors 2k and 2k+1.

After hashing guard and manual locations: 344 of 344 on fuzzgoat, and on the
synthetic target 218/385 ids against 218/385 ground-truth edges.

The harness drives the guard callback by hand (like gen_synthetic_target's
SYNTH_MANUAL_GUARDS mode) so it builds with gcc and needs no clang.
"""

from __future__ import annotations

import os
import subprocess
import textwrap
from shutil import which

import pytest

from fuzzer_tool.adapters.shm import ShmCoverage

SHIM = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src",
    "fuzzer_tool",
    "adapters",
    "afl_shim.c",
)

pytestmark = pytest.mark.skipif(
    (which("gcc") is None and which("clang") is None) or not os.path.exists(SHIM),
    reason="needs a C compiler and afl_shim.c",
)

# argv[1] selects the path; each run records exactly one prev -> cur edge
# after a reset (prev_loc is 0 after reset, so step 0 fires first).
_HARNESS = textwrap.dedent(
    """
    #include <stdint.h>
    #include <stdlib.h>
    extern void __sanitizer_cov_trace_pc_guard(uint32_t *guard);
    extern void __sanitizer_cov_trace_pc_guard_init(uint32_t *start, uint32_t *stop);
    extern void __afl_map_edge(unsigned int cur_loc);
    static uint32_t g[64];
    int main(int argc, char **argv) {
        __sanitizer_cov_trace_pc_guard_init(g, g + 64);
        int mode = atoi(argv[1]);
        switch (mode) {
        /* guard successors 2k / 2k+1 of one predecessor (g[3] is guard 4) */
        case 0: __sanitizer_cov_trace_pc_guard(&g[3]); __sanitizer_cov_trace_pc_guard(&g[9]);  break;
        case 1: __sanitizer_cov_trace_pc_guard(&g[3]); __sanitizer_cov_trace_pc_guard(&g[10]); break;
        /* hand-written ids: predecessors 0x1102 / 0x1103 into one successor */
        case 2: __afl_map_edge(0x1102); __afl_map_edge(0x1700); break;
        case 3: __afl_map_edge(0x1103); __afl_map_edge(0x1700); break;
        /* hand-written successors 2k / 2k+1 */
        case 4: __afl_map_edge(0x1500); __afl_map_edge(0x1100); break;
        case 5: __afl_map_edge(0x1500); __afl_map_edge(0x1101); break;
        }
        return 0;
    }
    """
)


@pytest.fixture(scope="module")
def harness(tmp_path_factory):
    cc = which("gcc") or which("clang")
    d = tmp_path_factory.mktemp("edge_alias")
    src = d / "h.c"
    src.write_text(_HARNESS)
    exe = d / "h"
    r = subprocess.run(
        [cc, "-O1", "-D__AFL_CTX_SENSITIVE=0", f"-include{SHIM}", "-o", str(exe), str(src)],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr[-1500:]
    return str(exe)


def _ids(harness: str, mode: int) -> frozenset[int]:
    cov = ShmCoverage()
    try:
        env = dict(os.environ, __AFL_SHM_ID=str(cov.shm_id), AFL_MAP_SIZE=str(cov.num_entries))
        subprocess.run([harness, str(mode)], env=env, capture_output=True, check=True)
        return frozenset(cov.get_edge_ids())
    finally:
        cov.cleanup()


@pytest.mark.parametrize(
    "a,b,what",
    [
        (0, 1, "guard successors 2k/2k+1 (the |= 1 bug)"),
        (2, 3, "manual predecessors 2k/2k+1 (prev >> 1 on raw ids)"),
        (4, 5, "manual successors 2k/2k+1 (the |= 1 bug)"),
    ],
)
def test_sibling_edges_get_distinct_ids(harness, a, b, what):
    ia, ib = _ids(harness, a), _ids(harness, b)
    assert len(ia) == len(ib) == 2, (ia, ib)
    # The first edge of each run (0 -> first location) is shared by design
    # in the manual predecessor case; the second edge must differ.
    assert ia != ib and len(ia ^ ib) >= 2, f"{what}: {sorted(ia)} vs {sorted(ib)}"


def test_ids_are_exec_stable(harness):
    """Hashed locations must be deterministic, not seeded per process."""
    assert _ids(harness, 0) == _ids(harness, 0)


def test_manual_edge_survives_distance_mode_opt_out(tmp_path):
    """``__afl_map_edge`` must exist when the distance channel is compiled out.

    The hashed wrapper was first added inside the ``#if __AFL_DISTANCE_MODE``
    block, so ``-D__AFL_DISTANCE_MODE=0`` -- the documented opt-out -- left
    every harness wrapper without it: executables failed to link, and a .so
    linked with the symbol undefined. Build the same harness with the channel
    off and require both the link and the sibling separation.
    """
    cc = which("gcc") or which("clang")
    src = tmp_path / "h.c"
    src.write_text(_HARNESS)
    exe = tmp_path / "h_nodist"
    r = subprocess.run(
        [
            cc,
            "-O1",
            "-D__AFL_CTX_SENSITIVE=0",
            "-D__AFL_DISTANCE_MODE=0",
            f"-include{SHIM}",
            "-o",
            str(exe),
            str(src),
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr[-1500:]
    ia, ib = _ids(str(exe), 4), _ids(str(exe), 5)
    assert len(ia) == len(ib) == 2 and ia != ib, (sorted(ia), sorted(ib))
