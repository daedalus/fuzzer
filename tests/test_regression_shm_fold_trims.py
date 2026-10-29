"""Virgin-bucket and max-count folds: one ``max()`` replaces the range check.

Both folds tested ``eids < VIRGIN_DENSE_MAX`` and ``.all()`` before taking
``max()`` anyway; the max alone answers both. Oracle: the pre-change
methods, frozen here, run on a twin ShmCoverage fed the same sequence.
"""

import numpy as np

from fuzzer_tool.adapters.shm import MAX_COUNT_GROWTH_FACTOR, VIRGIN_DENSE_MAX, ShmCoverage
from fuzzer_tool.core.count_class import bucket_bits

_SEED = 0xF01D
_N_EXECS = 300
_MAP = 8192


def _oracle_virgin(self, edge_ids, counts):
    """Pre-change _update_virgin_buckets, verbatim."""
    bits = bucket_bits(counts)
    occupied = bits != 0
    if not occupied.any():
        return False
    eids = np.asarray(edge_ids, dtype=np.uint32)[occupied]
    bits = bits[occupied]
    found = False
    in_range = eids < VIRGIN_DENSE_MAX
    if not in_range.all():
        found = self._update_virgin_overflow(eids[~in_range], bits[~in_range])
        eids, bits = eids[in_range], bits[in_range]
        if eids.size == 0:
            return found
    hi = int(eids.max()) + 1
    if hi > self._virgin.size:
        grown = np.zeros(1 << max(10, (hi - 1).bit_length()), dtype=np.uint8)
        grown[: self._virgin.size] = self._virgin
        self._virgin = grown
    prev = self._virgin[eids]
    fresh = bits & ~prev
    if not fresh.any():
        return found
    self.bucket_transitions += int(np.count_nonzero(fresh))
    self._virgin[eids] = prev | bits
    return True


def _oracle_max(self, edge_ids, counts):
    """Pre-change _update_max_counts, verbatim."""
    eids = np.asarray(edge_ids, dtype=np.uint32)
    cnts = np.asarray(counts, dtype=np.uint32)
    if eids.size == 0:
        return 0
    gained = 0
    in_range = eids < VIRGIN_DENSE_MAX
    if not in_range.all():
        gained = self._update_max_overflow(eids[~in_range], cnts[~in_range])
        eids, cnts = eids[in_range], cnts[in_range]
        if eids.size == 0:
            return gained
    hi = int(eids.max()) + 1
    if hi > self._max_counts.size:
        grown = np.zeros(1 << max(10, (hi - 1).bit_length()), dtype=np.uint32)
        grown[: self._max_counts.size] = self._max_counts
        self._max_counts = grown
    prev = self._max_counts[eids]
    fresh = (prev > 0) & (
        cnts.astype(np.float64) >= prev.astype(np.float64) * MAX_COUNT_GROWTH_FACTOR
    )
    grew = cnts > prev
    if grew.any():
        self._max_counts[eids[grew]] = cnts[grew]
    if fresh.any():
        n = int(np.count_nonzero(fresh))
        self.max_count_transitions += n
        gained += n
    return gained


_STATE = (
    "_virgin",
    "_virgin_wide",
    "bucket_transitions",
    "_max_counts",
    "_max_counts_wide",
    "max_count_transitions",
)


def _execs(rng):
    """Edge-id/count batches: dense ids, ids past VIRGIN_DENSE_MAX (alone and
    mixed), zero counts, counts past 255 and near the 24-bit ceiling, empty."""
    pool = rng.choice(np.arange(1, 1 << 16, dtype=np.uint32), 400, replace=False)
    wide = np.array([VIRGIN_DENSE_MAX, VIRGIN_DENSE_MAX + 7, (1 << 32) - 1], dtype=np.uint32)
    for i in range(_N_EXECS):
        ids = rng.choice(pool, int(rng.integers(0, 60)), replace=False)
        if i % 7 == 0:
            ids = np.concatenate([ids, wide[: 1 + i % 3]])
        if i % 11 == 0:
            ids = wide[: 1 + i % 3].copy()
        counts = rng.integers(0, 300, ids.size, dtype=np.uint32)
        if i % 5 == 0 and counts.size:
            counts[0] = 0xFFFFFF
        yield ids.astype(np.uint32), counts
    yield np.zeros(0, dtype=np.uint32), np.zeros(0, dtype=np.uint32)


def _run(virgin_fn, max_fn):
    cov = ShmCoverage(size=_MAP)
    try:
        out = [
            (virgin_fn(cov, i, c), max_fn(cov, i, c))
            for i, c in _execs(np.random.default_rng(_SEED))
        ]
        return out, {
            k: getattr(cov, k).copy() for k in _STATE if hasattr(getattr(cov, k), "copy")
        } | {k: getattr(cov, k) for k in _STATE if not hasattr(getattr(cov, k), "copy")}
    finally:
        cov.cleanup()


def _same(a, b):
    out_a, st_a = a
    out_b, st_b = b
    assert out_a == out_b
    for k in _STATE:
        x, y = st_a[k], st_b[k]
        assert np.array_equal(x, y) if isinstance(x, np.ndarray) else x == y, k


def test_control_oracle_matches_itself():
    """Rule 46: two oracle runs on one sequence agree."""
    _same(_run(_oracle_virgin, _oracle_max), _run(_oracle_virgin, _oracle_max))


def test_regression_folds_match_oracle():
    """Falsification: same verdicts, gains and state over the sequence."""
    _same(
        _run(ShmCoverage._update_virgin_buckets, ShmCoverage._update_max_counts),
        _run(_oracle_virgin, _oracle_max),
    )


def test_regression_folds_wide_only_and_zero_counts():
    """Adversarial: a batch of only past-range ids, and all-zero counts."""
    cov, ref = ShmCoverage(size=_MAP), ShmCoverage(size=_MAP)
    try:
        wide = np.array([VIRGIN_DENSE_MAX + 1, VIRGIN_DENSE_MAX + 2], dtype=np.uint32)
        for ids, cnt in (
            (wide, np.array([3, 9], dtype=np.uint32)),
            (wide, np.array([0, 0], dtype=np.uint32)),
        ):
            assert cov._update_virgin_buckets(ids, cnt) == _oracle_virgin(ref, ids, cnt)
            assert cov._update_max_counts(ids, cnt) == _oracle_max(ref, ids, cnt)
        assert cov._virgin_wide == ref._virgin_wide
        assert cov._max_counts_wide == ref._max_counts_wide
    finally:
        cov.cleanup()
        ref.cleanup()
