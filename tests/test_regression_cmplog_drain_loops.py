"""Drain-loop fast paths keep the collector's results unchanged.

``detect_hash_candidates`` counts matching byte positions as zero bytes of
``a ^ b`` instead of a Python generator over ``zip``. Pair eviction, the
occurrence count and the pending-redqueen filter bind locals and filter in C.
Oracles: the pre-change bodies, frozen here.
"""

from fuzzer_tool.core.cmplog import CmplogCollector
from fuzzer_tool.core.rand_pool import RandPool

_SEED = 0xBEEF
_N_PAIRS = 3000
_WIDTHS = (1, 2, 4, 7, 8, 16, 32)
_MAX_PAIRS = 64
_DRAINS = 6


def _oracle_hash(pairs, min_bytes=8, max_match=2):
    """Pre-change detect_hash_candidates predicate, verbatim."""
    flagged = []
    for op_a, op_b in pairs:
        if len(op_a) < min_bytes or len(op_b) < min_bytes:
            continue
        if len(op_a) != len(op_b):
            continue
        matches = sum(1 for a, b in zip(op_a, op_b, strict=False) if a == b)
        if matches <= max_match:
            flagged.append((op_a, op_b))
    return flagged


def _pairs(rng: RandPool) -> list[tuple[bytes, bytes]]:
    out = []
    for _ in range(_N_PAIRS):
        w = rng.choice(_WIDTHS)
        a = rng.randbytes(w)
        # Copy k random positions of a into b: match counts around the cut.
        b = bytearray(rng.randbytes(w))
        for i in rng.randrange_list(w, rng.randrange(4)):
            b[i] = a[i]
        out.append((a, bytes(b)))
    return out


# Leading zero bytes (int.from_bytes drops them), match counts at the cut,
# unequal and too-short lengths.
_ADVERSARIAL = [
    (b"\0" * 8, b"\0" * 8),
    (b"\0" * 8, b"\0\0" + b"\1" * 6),
    (b"\0" * 8, b"\0\0\0" + b"\1" * 5),
    (b"\0\0\0\1\2\3\4\5", b"\0\0\0\6\7\x08\x09\x0a"),
    (b"\xff" * 8, b"\x00" * 8),
    (b"\1" * 7, b"\2" * 7),
    (b"\1" * 8, b"\2" * 9),
    (b"\1" * 9, b"\2" * 8),
    (b"", b""),
]


def test_control_hash_oracle_matches_itself():
    """Rule 46: the oracle is deterministic on identical input."""
    pairs = _pairs(RandPool(seed=_SEED)) + _ADVERSARIAL
    assert _oracle_hash(pairs) == _oracle_hash(list(pairs))


def test_regression_hash_matches_oracle():
    """Falsification: same count and same flagged set on random pairs."""
    pairs = _pairs(RandPool(seed=_SEED))
    expected = _oracle_hash(pairs)
    c = CmplogCollector()
    assert c.detect_hash_candidates(pairs) == len(expected)
    assert c.hash_candidates == set(expected)


def test_regression_hash_adversarial():
    """Leading zeros and the 2-vs-3 match cut decide exactly as before."""
    c = CmplogCollector()
    assert c.detect_hash_candidates(_ADVERSARIAL) == len(_oracle_hash(_ADVERSARIAL))
    assert c.hash_candidates == set(_oracle_hash(_ADVERSARIAL))
    # 2 matching leading zeros flag; 3 do not.
    assert (b"\0" * 8, b"\0\0" + b"\1" * 6) in c.hash_candidates
    assert (b"\0\0\0\1\2\3\4\5", b"\0\0\0\6\7\x08\x09\x0a") not in c.hash_candidates


def _drain_lines(rng: RandPool, n: int) -> list[str]:
    return [f"CMP {rng.randbytes(4).hex()} {rng.randbytes(4).hex()} 1 4\n" for _ in range(n)]


def test_regression_eviction_state_consistent():
    """Adversarial: drains several times the pair cap keep every companion
    map inside the held pool, occurrences equal drains seen, and pending in
    pool order."""
    rng = RandPool(seed=_SEED)
    c = CmplogCollector(max_pairs=_MAX_PAIRS)
    seen_in: dict[tuple[bytes, bytes], int] = {}
    hot = _drain_lines(rng, _MAX_PAIRS // 4)
    for _ in range(_DRAINS):
        lines = hot + _drain_lines(rng, _MAX_PAIRS * 3)
        c._parse_lines(lines)
        for p in dict.fromkeys(c.last_pairs):
            seen_in[p] = seen_in.get(p, 0) + 1

    held = c._pair_set
    assert len(c.pairs) <= _MAX_PAIRS
    assert set(c.pairs) == held
    assert set(c._pair_occurrence) <= held
    assert set(c._pair_cmp) <= held
    assert set(c._pending_redqueen) <= held
    pending = set(c._pending_redqueen)
    assert c._pending_redqueen == [p for p in c.pairs if p in pending]
    # A pair evicted and re-added restarts its count, so held counts never
    # exceed the drains it appeared in.
    assert all(c._pair_occurrence[p] <= seen_in[p] for p in c._pair_occurrence)
