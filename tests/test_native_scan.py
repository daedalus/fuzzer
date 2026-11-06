"""C scan kernels (``adapters/native_scan.c``) match the numpy paths they replace.

Each kernel is checked against an oracle derived independently of both
implementations (plain Python loops), and every oracle is first run against
a second copy of itself (Hard Rule 46). The end-to-end tests drive two
``ShmCoverage`` tables, one native and one forced onto numpy, through the
same execution sequence and compare every novelty counter.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from fuzzer_tool.adapters import native_scan
from fuzzer_tool.adapters.cmp_shm import CMP_NAMES, SITE_SLOTS, CmpCounterShm
from fuzzer_tool.adapters.shm import (
    _ENTRY_DTYPE,
    MAX_COUNT_GROWTH_FACTOR,
    SHM_EDGE_COUNT_OFFSET,
    SHM_PATH_HASH_OFFSET,
    ShmCoverage,
    _live_entries,
)
from fuzzer_tool.core.count_class import bucket_bit

KERNELS = native_scan.load()
pytestmark = pytest.mark.skipif(KERNELS is None, reason="no C compiler for native_scan")

COUNT_MASK = 0xFFFFFF


def _table(rng, n: int, gen: int) -> np.ndarray:
    """Entry table: live, stale-generation, zero-id and empty slots."""
    arr = np.zeros(n, dtype=_ENTRY_DTYPE)
    occ = rng.choice(n, n // 2, replace=False)
    arr["edge_id"][occ] = rng.integers(0, 1 << 20, occ.size).astype(np.uint32)
    gens = rng.integers(0, 256, occ.size).astype(np.uint32)
    gens[: occ.size // 2] = gen  # half live
    arr["count"][occ] = (gens << 24) | rng.integers(0, 1000, occ.size).astype(np.uint32)
    return arr


def _ref_live(arr: np.ndarray, gen: int, slots=None) -> tuple[list[int], list[int]]:
    """Oracle: loop over (candidate) slots, keep current-generation non-zero ids."""
    ids, counts = [], []
    for s in range(len(arr)) if slots is None else slots:
        eid, cnt = int(arr["edge_id"][s]), int(arr["count"][s])
        if eid and (cnt >> 24) == gen & 0xFF:
            ids.append(eid)
            counts.append(cnt)
    return ids, counts


def _ref_fold(ids, counts, virgin, maxc):
    """Oracle for one fold: candidates, transitions, old bucket move, max gain."""
    cands, trans, old_moved, gain = [], 0, False, 0
    for eid, raw in zip(ids, counts, strict=True):
        c = raw & COUNT_MASK
        bits = bucket_bit(min(c, 255))
        prev = int(virgin[eid])
        if prev == 0:
            cands.append((eid, bits))
        if bits & ~prev:
            trans += 1
            virgin[eid] = prev | bits
            old_moved |= prev != 0
        pm = int(maxc[eid])
        if pm and c >= pm * MAX_COUNT_GROWTH_FACTOR:
            gain += 1
        if c > pm:
            maxc[eid] = c
    return cands, trans, old_moved, gain


def _addr(a: np.ndarray) -> int:
    return a.ctypes.data


# ── live-entry gather ───────────────────────────────────────────────────


@pytest.mark.parametrize("gen", [0, 1, 200, 255])
def test_live_matches_oracle(gen):
    rng = np.random.default_rng(gen)
    arr = _table(rng, 4096, gen)
    want = _ref_live(arr, gen)
    assert _ref_live(arr.copy(), gen) == want  # control

    scan = native_scan.LiveScan(KERNELS)
    ids, counts = scan.live(_addr(arr), len(arr), gen)
    assert (ids.tolist(), counts.tolist()) == want

    # Same answer as the numpy scan it replaces.
    _, np_ids, np_counts = _live_entries(arr.tobytes(), len(arr), gen)
    assert np.array_equal(ids, np_ids) and np.array_equal(counts, np_counts)


def test_live_empty_table():
    arr = np.zeros(64, dtype=_ENTRY_DTYPE)
    ids, counts = native_scan.LiveScan(KERNELS).live(_addr(arr), 64, 0)
    assert ids.size == counts.size == 0


def test_live_result_survives_next_scan():
    """Adversarial: returned arrays are copies, not views of reused scratch."""
    rng = np.random.default_rng(7)
    a, b = _table(rng, 512, 3), _table(rng, 512, 3)
    scan = native_scan.LiveScan(KERNELS)
    first, _ = scan.live(_addr(a), 512, 3)
    snapshot = first.copy()
    scan.live(_addr(b), 512, 3)
    assert np.array_equal(first, snapshot)


@pytest.mark.parametrize("density", [0.01, 0.3, 1.0])
def test_touched_matches_oracle(density):
    rng = np.random.default_rng(int(density * 100))
    n, gen = 1000, 9  # n not a multiple of 64: padding bits must be dropped
    arr = _table(rng, n, gen)
    words = np.zeros((n + 63) // 64 + 1, dtype=np.uint64)
    bits = np.unpackbits(words.view(np.uint8), bitorder="little")
    bits[: words.size * 64] = rng.random(words.size * 64) < density
    words[:] = np.packbits(bits, bitorder="little").view(np.uint64)
    slots = [s for s in range(n) if (int(words[s >> 6]) >> (s & 63)) & 1]
    want = _ref_live(arr, gen, slots)
    assert _ref_live(arr.copy(), gen, list(slots)) == want  # control

    ids, counts = native_scan.LiveScan(KERNELS).live_touched(
        _addr(arr), n, gen, _addr(words), words.size
    )
    assert (ids.tolist(), counts.tolist()) == want


# ── virgin / max fold ───────────────────────────────────────────────────


def _fold_state(rng, size: int):
    virgin = np.zeros(size, dtype=np.uint8)
    maxc = np.zeros(size, dtype=np.uint32)
    known = rng.choice(size, size // 3, replace=False)
    virgin[known] = 1 << rng.integers(0, 8, known.size)
    maxc[known] = rng.integers(1, 300, known.size)
    return virgin, maxc


@pytest.mark.parametrize("seed", range(5))
def test_fold_matches_oracle(seed):
    rng = np.random.default_rng(seed)
    size = 2048
    virgin, maxc = _fold_state(rng, size)
    ids = rng.choice(size, 700, replace=False).astype(np.uint32)
    counts = (np.uint32(5) << 24) | rng.integers(0, 600, ids.size).astype(np.uint32)

    rv, rm = virgin.copy(), maxc.copy()
    want = _ref_fold(ids.tolist(), counts.tolist(), rv, rm)
    cv, cm = virgin.copy(), maxc.copy()
    assert _ref_fold(ids.tolist(), counts.tolist(), cv, cm) == want  # control

    out = native_scan.LiveScan(KERNELS).fold(ids, counts, virgin, maxc, MAX_COUNT_GROWTH_FACTOR)
    got_cands = list(zip(out.cand_ids.tolist(), out.cand_bits.tolist(), strict=True))
    assert (got_cands, out.transitions, out.old_moved, out.gain) == want
    assert np.array_equal(virgin, rv) and np.array_equal(maxc, rm)


def test_fold_reports_old_bucket_move():
    """Falsification: an old edge reaching a new bucket must be reported."""
    virgin = np.zeros(16, dtype=np.uint8)
    maxc = np.zeros(16, dtype=np.uint32)
    scan = native_scan.LiveScan(KERNELS)
    ids = np.array([3], dtype=np.uint32)

    first = scan.fold(ids, np.array([1], np.uint32), virgin, maxc, MAX_COUNT_GROWTH_FACTOR)
    assert first.cand_ids.tolist() == [3] and not first.old_moved

    same = scan.fold(ids, np.array([1], np.uint32), virgin, maxc, MAX_COUNT_GROWTH_FACTOR)
    assert same.transitions == 0 and not same.old_moved

    moved = scan.fold(ids, np.array([40], np.uint32), virgin, maxc, MAX_COUNT_GROWTH_FACTOR)
    assert moved.old_moved and moved.transitions == 1 and moved.gain == 1
    assert moved.cand_ids.size == 0


def test_fold_count_ceiling_does_not_wrap():
    """Adversarial: a saturated max must not read as growth (u32 * 1.5 wraps)."""
    virgin = np.full(4, 0x80, dtype=np.uint8)
    maxc = np.full(4, COUNT_MASK, dtype=np.uint32)
    ids = np.arange(4, dtype=np.uint32)
    counts = np.full(4, 0xFF000000 | COUNT_MASK, dtype=np.uint32)  # gen byte set
    out = native_scan.LiveScan(KERNELS).fold(ids, counts, virgin, maxc, MAX_COUNT_GROWTH_FACTOR)
    assert out.gain == 0 and out.transitions == 0
    assert maxc.tolist() == [COUNT_MASK] * 4


def test_fold_zero_count_is_candidate_without_bits():
    """Adversarial: a new id with count 0 is still a candidate, with no bucket."""
    virgin = np.zeros(8, dtype=np.uint8)
    maxc = np.zeros(8, dtype=np.uint32)
    out = native_scan.LiveScan(KERNELS).fold(
        np.array([5], np.uint32), np.array([0], np.uint32), virgin, maxc, 1.5
    )
    assert out.cand_ids.tolist() == [5] and out.cand_bits.tolist() == [0]
    assert out.transitions == 0 and virgin[5] == 0


@pytest.mark.parametrize("short", ["virgin", "maxc"])
def test_fold_refuses_ids_past_arrays(short):
    """Adversarial: an id past either array returns None and writes nothing."""
    virgin = np.zeros(4 if short == "virgin" else 64, dtype=np.uint8)
    maxc = np.zeros(4 if short == "maxc" else 64, dtype=np.uint32)
    ids = np.array([1, 40], np.uint32)
    out = native_scan.LiveScan(KERNELS).fold(ids, np.array([1, 1], np.uint32), virgin, maxc, 1.5)
    assert out is None
    assert not virgin.any() and not maxc.any()


# ── ShmCoverage end to end ──────────────────────────────────────────────


def _load_exec(cov: ShmCoverage, rng, step: int, pool: np.ndarray) -> None:
    """One synthetic execution: fresh generation, live entries, new headers."""
    cov.reset_edge_map()
    gen = cov.read_generation()
    arr = np.frombuffer(cov._map, dtype=_ENTRY_DTYPE, count=cov.num_entries)
    ids = rng.choice(pool, rng.integers(1, min(60, pool.size) + 1), replace=False)
    slots = rng.choice(cov.num_entries, ids.size, replace=False)
    counts = rng.integers(0, 400, ids.size).astype(np.uint32)
    arr["edge_id"][slots] = ids
    arr["count"][slots] = (np.uint32(gen) << 24) | counts
    ctypes.c_uint64.from_address(cov._ptr + SHM_PATH_HASH_OFFSET).value = step + 1
    ctypes.c_uint64.from_address(cov._ptr + SHM_EDGE_COUNT_OFFSET).value = step + 1


def _observe(cov: ShmCoverage, hit: bool):
    return (
        hit,
        sorted(cov.last_new_ids),
        cov.last_old_bucket_novel,
        cov.new_max_edges,
        cov.cumulative_edges,
        cov.bucket_transitions,
        cov.max_count_transitions,
        sorted(cov._last_ids),
    )


def _run(native: bool, seed: int, pool: np.ndarray) -> list:
    cov = ShmCoverage(size=1024)
    try:
        if not native:
            cov._native = None
        rng = np.random.default_rng(seed)
        seen = []
        for step in range(40):
            _load_exec(cov, rng, step, pool)
            seen.append(_observe(cov, cov.is_new_coverage()))
        seen.append((cov.get_virgin_buckets(), cov.get_max_counts()))
        return seen
    finally:
        cov.cleanup()


@pytest.mark.parametrize("seed", range(3))
def test_shm_native_matches_numpy(seed):
    pool = np.arange(1, 300, dtype=np.uint32)
    want = _run(False, seed, pool)
    assert _run(False, seed, pool) == want  # control: numpy vs numpy
    assert _run(True, seed, pool) == want


def test_shm_native_handles_wide_ids():
    """Adversarial: ids past VIRGIN_DENSE_MAX take the numpy path, same result."""
    pool = np.array([1, 2, 3, (1 << 24) + 5, (1 << 30) + 1], dtype=np.uint32)
    assert _run(True, 11, pool) == _run(False, 11, pool)


def test_shm_native_is_wired():
    cov = ShmCoverage(size=256)
    try:
        assert cov._native is not None
    finally:
        cov.cleanup()


# ── cmplog counter drains ───────────────────────────────────────────────


@pytest.fixture
def cmp_pair():
    a, b = CmpCounterShm(), CmpCounterShm()
    b._native = None
    yield a, b
    a.close()
    b.close()


def _fill_counts(shm: CmpCounterShm, rng) -> None:
    hot = rng.choice(len(CMP_NAMES), 9, replace=False)
    shm.fired[hot] = rng.integers(1, 1 << 40, hot.size, dtype=np.uint64)
    shm.hit[hot] = rng.integers(0, 1 << 20, hot.size, dtype=np.uint64)


def _fill_sites(shm: CmpCounterShm, rng) -> None:
    hot = rng.choice(SITE_SLOTS, 300, replace=False)
    shm.sites["pc"][hot] = rng.integers(1, 1 << 40, hot.size, dtype=np.uint64)
    shm.sites["id"][hot] = rng.integers(0, len(CMP_NAMES), hot.size, dtype=np.uint32)
    shm.sites["fired"][hot] = rng.integers(1, 1 << 30, hot.size, dtype=np.uint64)
    shm.sites["hit"][hot] = rng.integers(0, 1 << 30, hot.size, dtype=np.uint64)


def test_cmp_drain_counts_matches_numpy(cmp_pair):
    native, numpy_ = cmp_pair
    assert native._native is not None
    for seed in range(3):
        _fill_counts(native, np.random.default_rng(seed))
        _fill_counts(numpy_, np.random.default_rng(seed))
        assert native.drain_counts() == numpy_.drain_counts()
        assert not native.fired.any() and not native.hit.any()
    assert native.drain_counts() == []


def test_cmp_drain_sites_matches_numpy(cmp_pair):
    native, numpy_ = cmp_pair
    for seed in range(3):
        _fill_sites(native, np.random.default_rng(seed))
        _fill_sites(numpy_, np.random.default_rng(seed))
        keys = native.sites[["pc", "id"]].copy()
        got, want = native.drain_sites(), numpy_.drain_sites()
        for g, w in zip(got, want, strict=True):
            assert np.array_equal(g, w) and g.dtype == w.dtype
        assert not native.sites["fired"].any() and not native.sites["hit"].any()
        assert np.array_equal(native.sites[["pc", "id"]], keys)  # keys kept


def test_cmp_drain_sites_survive_next_drain(cmp_pair):
    """Adversarial: drained arrays are kept by the fold; scratch reuse must not alias."""
    native, _ = cmp_pair
    _fill_sites(native, np.random.default_rng(1))
    slots, fired, hit = native.drain_sites()
    snap = slots.copy(), fired.copy(), hit.copy()
    _fill_sites(native, np.random.default_rng(2))
    native.drain_sites()
    assert all(np.array_equal(a, b) for a, b in zip((slots, fired, hit), snap, strict=True))
