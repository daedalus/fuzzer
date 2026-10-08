"""Context variants of one edge must not share a home neighbourhood.

The 8-bit call-site context is XORed into the LOW bits of edge_id, and the
home slot was ``edge_id & mask``. Every context variant of an edge therefore
homed inside one 256-slot block. A helper reached from many call sites packed
its block solid, the 64-slot probe window overflowed, and edges were dropped
at a global load of 28% (ffmpeg: 51M drops, map 28.2%). Hashing the id into
the home slot spreads the variants across the table.
"""

from __future__ import annotations

import subprocess

import numpy as np
import pytest

from fuzzer_tool.adapters.shm import _ENTRY_DTYPE, ShmCoverage
from tests.conftest import requires_clang
from tests.test_regression_shim_audit import _build, _clean_env

MAP_ENTRIES = 1 << 16
CTX_VARIANTS = 256  # 2^__AFL_CTX_BITS
CLASH_BASES = 4
GOLDEN = 0x9E3779B1

# argv[1]: "clash" -> CLASH_BASES bases sharing their low 16 bits, each with
# every context variant. "fanout" -> LCG bases with 64 variants, ~28% load.
_DRIVER = """
#include <stdlib.h>
#include <string.h>
int main(int argc, char **argv) {
    if (!strcmp(argv[1], "clash")) {
        for (uint32_t k = 1; k <= 4; k++)
            for (uint32_t c = 0; c < 256; c++)
                __afl_map_id_raw((k << 16) ^ c);
        return 0;
    }
    uint32_t x = 12345;
    for (int b = 0; b < 287; b++) {
        x = x * 1664525u + 1013904223u;
        uint32_t base = (x >> 8) | 0x100;
        for (uint32_t c = 0; c < 64; c++)
            __afl_map_id_raw(base ^ c);
    }
    return 0;
}
"""


@pytest.fixture(scope="module")
def target(tmp_path_factory):
    return _build(tmp_path_factory.mktemp("ctxhome"), _DRIVER, "-D__AFL_CTX_SENSITIVE=0")


def _run(exe, mode: str) -> tuple[int, dict[int, int]]:
    """Run the driver; return (drops, {edge_id: slot})."""
    cov = ShmCoverage(size=MAP_ENTRIES)
    try:
        env = _clean_env(__AFL_SHM_ID=cov.env_id, AFL_MAP_SIZE=str(MAP_ENTRIES))
        cov.reset_edge_map()
        r = subprocess.run([str(exe), mode], env=env, capture_output=True, text=True, timeout=30)
        assert r.returncode == 0, r.stderr

        table = np.frombuffer(cov._map, dtype=_ENTRY_DTYPE, count=cov.num_entries)
        slots = {int(e): i for i, e in enumerate(table["edge_id"]) if e}
        return cov.read_dropped_edges(), slots
    finally:
        cov.cleanup()


@requires_clang
def test_regression_ctx_variants_clash_drop_nothing(target):
    """Adversarial: 4 edges whose 256 variants all homed on block 0."""
    drops, slots = _run(target, "clash")

    assert drops == 0
    assert len(slots) == CLASH_BASES * CTX_VARIANTS


@requires_clang
def test_regression_ctx_fanout_low_load_no_drops(target):
    """Falsification: ~28% load with context fan-out must not drop."""
    drops, slots = _run(target, "fanout")

    assert drops == 0
    assert len(slots) / MAP_ENTRIES > 0.25


@requires_clang
def test_record_edge_mirrors_shim_placement(target):
    """The Python mirror must place ids where the shim does."""
    _, shim_slots = _run(target, "clash")

    cov = ShmCoverage(size=MAP_ENTRIES)
    try:
        for k in range(1, CLASH_BASES + 1):
            for c in range(CTX_VARIANTS):
                cov.record_edge((k << 16) ^ c)
        table = np.frombuffer(cov._map, dtype=_ENTRY_DTYPE, count=cov.num_entries)
        py_slots = {int(e): i for i, e in enumerate(table["edge_id"]) if e}
    finally:
        cov.cleanup()

    assert py_slots == shim_slots


def test_home_slot_reference():
    """Home slot is the high half of a golden-ratio product, scaled to size."""
    from fuzzer_tool.adapters.shm import home_slot

    for size in (1, 3, 1000, MAP_ENTRIES):
        for eid in (1, 255, 256, 0xFFFFFFFE):
            assert home_slot(eid, size) == (((eid * GOLDEN) & 0xFFFFFFFF) * size) >> 32
