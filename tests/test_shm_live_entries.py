"""The generation-first SHM scan returns exactly what the id-first scan did.

``_live_entries`` tests the generation byte before ``edge_id != 0``. Stale
entries keep their ids across ``reset_edge_map`` (it bumps the generation
instead of zeroing), so on a long campaign the table saturates and the
id-first scan selected -- and copied, via ``flatnonzero`` on a strided field
-- every slot. The old scan is kept here verbatim as the oracle.
"""

import numpy as np
import pytest

from fuzzer_tool.adapters.shm import _ENTRY_DTYPE, _live_entries


def _reference(buf, n, gen):
    arr = np.frombuffer(buf, dtype=_ENTRY_DTYPE, count=n)
    eid = arr["edge_id"]
    occupied = np.flatnonzero(eid)
    live_counts = arr["count"][occupied]
    live = ((live_counts >> 24) & 0xFF) == gen
    return occupied[live], eid[occupied][live], live_counts[live]


def _table(rng, n, occupancy, gen, live_frac):
    buf = bytearray(n * 8)
    arr = np.frombuffer(buf, dtype=_ENTRY_DTYPE)
    occ = rng.choice(n, int(n * occupancy), replace=False)
    arr["edge_id"][occ] = rng.integers(1, 2**32, occ.size, dtype=np.uint64).astype(np.uint32)
    gens = rng.integers(0, 256, occ.size).astype(np.uint32)
    arr["count"][occ] = (gens << 24) | rng.integers(0, 1 << 24, occ.size).astype(np.uint32)
    live = rng.choice(occ, int(occ.size * live_frac), replace=False) if occ.size else occ
    arr["count"][live] = (np.uint32(gen) << 24) | rng.integers(1, 1000, live.size).astype(np.uint32)
    # Torn reads: a count stamped with the live generation under a zero id.
    torn = rng.choice(n, 5, replace=False)
    arr["edge_id"][torn] = 0
    arr["count"][torn] = np.uint32(gen) << 24
    return buf


@pytest.mark.parametrize("occupancy", [0.0, 0.05, 0.5, 1.0])
@pytest.mark.parametrize("gen", [0, 1, 7, 255])
def test_matches_the_id_first_scan(occupancy, gen):
    rng = np.random.default_rng(int(occupancy * 100) + gen)
    n = 4096
    buf = _table(rng, n, occupancy, gen, 0.1)
    got = _live_entries(buf, n, gen)
    want = _reference(buf, n, gen)
    for g, w in zip(got, want, strict=True):
        assert np.array_equal(g, w)
        assert g.dtype.kind == w.dtype.kind


def test_empty_table():
    buf = bytearray(64 * 8)
    slots, ids, counts = _live_entries(buf, 64, 3)
    assert slots.size == ids.size == counts.size == 0
