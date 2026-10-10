"""scanner_for_pairs: a grown pool keeps its automaton, new tokens go to find.

The collector grows the operand pool in place, so every growth was a cache
miss and a full Aho-Corasick rebuild (~21 ms; 108 rebuilds, 2.3 s of a
2000-exec ``--hail-mary`` profile). Tokens added since the last build are now
scanned with ``bytes.find`` until they outnumber ``1/TAIL_REBUILD_RATIO`` of
the built ones, so rebuilds are geometric. Per-token offsets are identical;
consumers read the result by token (weizz) or mark idempotently (colorizer).
"""

import random

import pytest

from fuzzer_tool.core import aho_corasick as ac


@pytest.fixture(autouse=True)
def _fresh_cache():
    ac._reset_scanner_cache()
    yield
    ac._reset_scanner_cache()


@pytest.fixture
def builds(monkeypatch):
    """Count automaton constructions."""
    count = []
    real = ac.AhoCorasick.__init__

    def _counting(self, patterns):
        count.append(1)
        real(self, patterns)

    monkeypatch.setattr(ac.AhoCorasick, "__init__", _counting)
    return count


def _pairs(rnd, n):
    # Non-integer widths: 1/2/4/8 take the trie-free fast path.
    return [(rnd.randbytes(rnd.choice((3, 5, 6, 7))), rnd.randbytes(5)) for _ in range(n)]


def _fresh_scan(pairs, data, min_len):
    return ac.TokenScanner(ac.tokens_from_pairs(pairs)).scan(data, min_len=min_len)


# ---------------------------------------------------------------------------


def test_regression_scanner_absorbs_growth(builds):
    """In-place growth below the rebuild ratio builds nothing new."""
    rnd = random.Random(0)
    pool = _pairs(rnd, 600)
    ac.scanner_for_pairs(pool)
    assert len(builds) == 1

    pool.extend(_pairs(rnd, 20))
    scanner = ac.scanner_for_pairs(pool)
    assert len(builds) == 1
    assert scanner is ac.scanner_for_pairs(pool)


def test_scan_matches_fresh_build_through_growth():
    """Falsification: after every growth step, offsets equal a fresh scanner's."""
    rnd = random.Random(1)
    pool = _pairs(rnd, 300)  # starts on the find backend
    data = rnd.randbytes(4000)
    data += b"".join(a for a, _ in pool[::7])  # plant known operands
    for _ in range(25):
        pool.extend(_pairs(rnd, rnd.randrange(1, 80)))
        planted = data + b"".join(a for a, _ in pool[-5:])
        for min_len in (1, 2):
            got = ac.scanner_for_pairs(pool).scan(planted, min_len=min_len)
            assert got == _fresh_scan(pool, planted, min_len)


def test_tail_past_ratio_rebuilds(builds):
    """Adversarial: growth past 1/TAIL_REBUILD_RATIO of the built tokens rebuilds."""
    rnd = random.Random(2)
    pool = _pairs(rnd, 1000)
    ac.scanner_for_pairs(pool)
    built = len(ac.scanner_for_pairs(pool).tokens)
    pool.extend(_pairs(rnd, built // ac.TAIL_REBUILD_RATIO + 50))
    ac.scanner_for_pairs(pool)
    assert len(builds) == 2


def test_crossing_min_tokens_switches_to_automaton(builds):
    """Adversarial: a find-only scanner rebuilds once the pool earns the automaton."""
    rnd = random.Random(3)
    pool = _pairs(rnd, 100)
    assert ac.scanner_for_pairs(pool).backend == "find"
    pool.extend(_pairs(rnd, ac.AC_MIN_TOKENS))
    assert ac.scanner_for_pairs(pool).backend != "find"
    assert len(builds) == 1


def test_duplicate_and_empty_tokens_are_not_added():
    rnd = random.Random(4)
    pool = _pairs(rnd, 600)
    before = len(ac.scanner_for_pairs(pool).tokens)
    pool.extend([pool[0], (b"", pool[1][1])])
    assert len(ac.scanner_for_pairs(pool).tokens) == before


def test_rebound_pool_rebuilds(builds):
    """A new pool object (eviction rebinds) scans exactly like a fresh build.

    Light churn now retargets the cached automaton instead of rebuilding it
    (see test_regression_scanner_retarget.py); only the results are a
    contract, not the scanner's internal token list.
    """
    rnd = random.Random(5)
    pool = _pairs(rnd, 600)
    ac.scanner_for_pairs(pool)
    smaller = pool[:550]
    data = b"".join(a + b for a, b in smaller[:50])
    assert ac.scanner_for_pairs(smaller).scan(data, min_len=1) == _fresh_scan(smaller, data, 1)
