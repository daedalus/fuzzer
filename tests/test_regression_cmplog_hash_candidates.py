"""Regression: ``hash_candidates`` outlived pair eviction and grew forever.

``_evict_pairs`` drops an evicted pair from every pair-keyed companion map
but ``hash_candidates``. The shared discovery test in
``test_regression_cmplog_pair_eviction.py`` could not see it: its 4-byte
operands are below ``_HASH_MIN_BYTES``, so the set stayed empty. Measured on
ffmpeg (non-ASAN): +38.5 MB between 1.5k and 3.5k execs, never shrinking.
"""

from fuzzer_tool.core.cmplog import _HASH_MIN_BYTES, CmplogCollector

_CAP = 8
_DISTINCT = 40
# Complementary operands share no byte position: hash-like by construction.
_MASK = (1 << (8 * _HASH_MIN_BYTES)) - 1


def _hash_line(i: int) -> str:
    a = (0x0123456789ABCDEF + i) & _MASK
    width = 2 * _HASH_MIN_BYTES
    return f"CMP {a:0{width}x} {a ^ _MASK:0{width}x} 0 {_HASH_MIN_BYTES}"


def _feed(n: int = _DISTINCT) -> CmplogCollector:
    c = CmplogCollector(max_tokens=100_000, max_pairs=_CAP)
    c._parse_lines([_hash_line(i) for i in range(n)])
    return c


def test_regression_cmplog_hash_candidates_bounded():
    """Falsification: flagged pairs leave with their pair."""
    c = _feed()

    assert c.evicted_pair_count >= _DISTINCT - _CAP
    assert len(c.hash_candidates) <= _CAP
    assert c.hash_candidates <= c._pair_set


def test_surviving_hash_pairs_stay_flagged():
    """Adversarial: pruning must not drop flags of pairs still held."""
    c = _feed(_CAP)

    assert c.evicted_pair_count == 0
    assert len(c.hash_candidates) == _CAP
    assert all(c.is_hash_candidate(a, b) for a, b in c.pairs)
