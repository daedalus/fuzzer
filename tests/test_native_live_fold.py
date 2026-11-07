"""``fs_live_fold``: ShmCoverage's slow-path gather and fold in one C call.

``_scan_new`` made two ctypes calls (``fs_live`` / ``fs_live_touched`` then
``fs_fold``, 5 + 12 arguments) plus four ``__array_interface__`` lookups per
exec. The fused call takes 6 arguments with the array addresses bound once.
Oracle: the two-call path, forced by disabling the fused one (control first).
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from fuzzer_tool.adapters import native_scan
from fuzzer_tool.adapters.shm import (
    _ENTRY_DTYPE,
    MAX_COUNT_GROWTH_FACTOR,
    SHM_EDGE_COUNT_OFFSET,
    SHM_PATH_HASH_OFFSET,
    SHM_TOUCHED_HEADER,
    ShmCoverage,
)
from tests.test_native_scan import _addr, _fold_state, _observe, _ref_fold, _ref_live, _table

KERNELS = native_scan.load()
pytestmark = pytest.mark.skipif(KERNELS is None, reason="no C compiler for native_scan")

_STEPS = 40
_MAX_LIVE = 60
_VSIZE = 512


def _mark_touched(cov: ShmCoverage, slots: np.ndarray) -> None:
    """Set the shim's touched bits for *slots* (what the shim does on insert)."""
    words = (ctypes.c_uint64 * cov.touched_words).from_address(
        cov._touched_ptr + SHM_TOUCHED_HEADER
    )
    for s in slots.tolist():
        words[s >> 6] |= 1 << (s & 63)


def _load_exec(cov: ShmCoverage, rng, step: int, pool: np.ndarray) -> None:
    """One synthetic execution: fresh generation, live entries, new headers."""
    cov.reset_edge_map()
    gen = cov.read_generation()
    arr = np.frombuffer(cov._map, dtype=_ENTRY_DTYPE, count=cov.num_entries)
    ids = rng.choice(pool, rng.integers(1, min(_MAX_LIVE, pool.size) + 1), replace=False)
    slots = rng.choice(cov.num_entries, ids.size, replace=False)
    counts = rng.integers(0, 400, ids.size).astype(np.uint32)
    arr["edge_id"][slots] = ids
    arr["count"][slots] = (np.uint32(gen) << 24) | counts
    if cov.touched_scan:
        _mark_touched(cov, slots)
    ctypes.c_uint64.from_address(cov._ptr + SHM_PATH_HASH_OFFSET).value = step + 1
    ctypes.c_uint64.from_address(cov._ptr + SHM_EDGE_COUNT_OFFSET).value = step + 1


def _run(fused: bool, seed: int, pool: np.ndarray, touched: bool = False, warm: bool = False):
    cov = ShmCoverage(size=1024, touched_scan=touched)
    try:
        if not fused:
            cov._gather_fold = lambda: None
        rng = np.random.default_rng(seed)
        seen = []
        for step in range(_STEPS):
            _load_exec(cov, rng, step, pool)
            if warm:
                cov.get_edge_ids()  # warms the scan memo before the fold
            seen.append(_observe(cov, cov.is_new_coverage()))
        seen.append((cov.get_virgin_buckets(), cov.get_max_counts()))
        return seen
    finally:
        cov.cleanup()


@pytest.mark.parametrize("touched", [False, True])
@pytest.mark.parametrize("seed", range(3))
def test_fused_matches_two_call_path(seed, touched):
    pool = np.arange(1, 300, dtype=np.uint32)
    want = _run(False, seed, pool, touched)
    assert _run(False, seed, pool, touched) == want  # control (Hard Rule 46)
    assert _run(True, seed, pool, touched) == want


def test_fused_grows_dense_arrays():
    """Adversarial: ids past the current virgin size force the grow-and-refold path."""
    pool = np.array([1, 2, 3, 5000, 70000, 900000], dtype=np.uint32)
    assert _run(True, 4, pool) == _run(False, 4, pool)


def test_fused_handles_wide_ids():
    """Adversarial: ids past VIRGIN_DENSE_MAX fall back to numpy, same result."""
    pool = np.array([1, 2, 3, (1 << 24) + 5, (1 << 30) + 1], dtype=np.uint32)
    assert _run(True, 11, pool) == _run(False, 11, pool)


def test_warm_memo_matches():
    """Edge: a scan memo warmed earlier in the exec is reused, same result."""
    pool = np.arange(1, 300, dtype=np.uint32)
    assert _run(True, 7, pool, warm=True) == _run(False, 7, pool, warm=True)


def test_regression_slow_path_one_native_call(monkeypatch):
    """Falsification: a cold slow-path scan crosses into C once, not twice."""
    cov = ShmCoverage(size=1024)
    try:
        calls = []
        for name in ("live", "live_touched", "fold", "live_fold"):
            real = getattr(cov._native, name)
            monkeypatch.setattr(
                cov._native, name, lambda *a, _n=name, _r=real: calls.append(_n) or _r(*a)
            )
        rng, pool = np.random.default_rng(1), np.arange(1, 50, dtype=np.uint32)
        _load_exec(cov, rng, 0, pool)
        cov.is_new_coverage()  # first exec grows the dense arrays: live_fold + fold
        assert calls == ["live_fold", "fold"]

        calls.clear()
        _load_exec(cov, rng, 1, pool)
        cov.is_new_coverage()
        assert calls == ["live_fold"]
    finally:
        cov.cleanup()


@pytest.mark.parametrize("seed", range(4))
def test_kernel_matches_oracle(seed):
    """fs_live_fold == the independent live + fold oracles on a raw table."""
    rng = np.random.default_rng(seed)
    n, gen = 1000, 5
    arr = _table(rng, n, gen)
    arr["edge_id"] %= _VSIZE  # inside the fold arrays (zero ids stay empty)
    virgin, maxc = _fold_state(rng, _VSIZE)
    want_ids, want_counts = _ref_live(arr, gen)
    assert _ref_live(arr.copy(), gen) == (want_ids, want_counts)  # control

    rv, rm = virgin.copy(), maxc.copy()
    want = _ref_fold(want_ids, want_counts, rv, rm)

    scan = native_scan.LiveScan(KERNELS)
    ids, counts, out = scan.live_fold(
        _addr(arr), n, gen, 0, 0, virgin, maxc, MAX_COUNT_GROWTH_FACTOR
    )
    assert (ids.tolist(), counts.tolist()) == (want_ids, want_counts)
    assert out is not None
    got_cands = list(zip(out.cand_ids.tolist(), out.cand_bits.tolist(), strict=True))
    assert (got_cands, out.transitions, out.old_moved, out.gain) == want
    assert np.array_equal(virgin, rv) and np.array_equal(maxc, rm)
