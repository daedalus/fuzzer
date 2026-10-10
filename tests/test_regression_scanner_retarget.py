"""scanner_for_pairs survives eviction: a rebound pool retargets the automaton.

Once the cmplog pool is at its cap, every drain evicts and rebinds
``collector.pairs``, so the identity-keyed cache missed on every drain and
rebuilt the automaton: 112 rebuilds at ~58 ms (6.5 s of a 3000-exec FFmpeg
profile). The fast path keeps the automaton, scans new tokens with
``bytes.find`` and filters tokens that left the pool out of the result; it
rebuilds (slow path) once dead + new tokens pass 1/TAIL_REBUILD_RATIO of
the built ones. Results must equal a fresh build's exactly.
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
    return [(rnd.randbytes(rnd.choice((2, 3, 4))), rnd.randbytes(4)) for _ in range(n)]


def _fresh_scan(pairs, data, min_len=2):
    return ac.TokenScanner(ac.tokens_from_pairs(pairs)).scan(data, min_len=min_len)


def _evict(rnd, pool, drop, add):
    """The collector's eviction: a NEW list, oldest *drop* gone, *add* appended."""
    return pool[drop:] + _pairs(rnd, add)


def _data_with(rnd, pool, n_tokens):
    """Input that contains *n_tokens* pool operands among random bytes."""
    parts = []
    for op_a, _ in rnd.sample(pool, n_tokens):
        parts += [rnd.randbytes(rnd.randint(0, 6)), op_a]
    return b"".join(parts)


def test_control_fresh_scan_matches_itself():
    """Rule 46: two fresh builds over one pool agree."""
    rnd = random.Random(1)
    pool = _pairs(rnd, 400)
    data = _data_with(rnd, pool, 40)
    assert _fresh_scan(pool, data) == _fresh_scan(pool, data)


def test_eviction_reuses_automaton(builds):
    """Falsification: a rebound pool with light churn builds nothing new and
    scans exactly like a fresh build."""
    rnd = random.Random(2)
    pool = _pairs(rnd, 600)
    first = ac.scanner_for_pairs(pool)

    pool = _evict(rnd, pool, drop=20, add=20)
    scanner = ac.scanner_for_pairs(pool)
    assert len(builds) == 1
    assert scanner is first

    data = _data_with(rnd, pool, 60)
    assert scanner.scan(data, min_len=2) == _fresh_scan(pool, data)


def test_evicted_token_not_reported():
    """Adversarial: an operand that left the pool must vanish from results
    even though it is still inside the automaton."""
    rnd = random.Random(3)
    pool = _pairs(rnd, 600)
    gone = pool[0][0]
    ac.scanner_for_pairs(pool)

    pool = _evict(rnd, pool, drop=1, add=0)
    if any(gone in p for p in pool):
        pytest.skip("random collision kept the operand in the pool")
    found = ac.scanner_for_pairs(pool).scan(b"xx" + gone + b"yy", min_len=2)
    assert gone not in found


def test_readded_token_reported_again():
    """Adversarial: evicted then re-added -> reported again."""
    rnd = random.Random(4)
    pool = _pairs(rnd, 600)
    back = pool[0]
    ac.scanner_for_pairs(pool)
    pool = _evict(rnd, pool, drop=1, add=0)
    ac.scanner_for_pairs(pool)
    pool = pool + [back]
    data = b"--" + back[0] + b"--"
    assert ac.scanner_for_pairs(pool).scan(data, min_len=2) == _fresh_scan(pool, data)


def test_heavy_churn_rebuilds(builds):
    """Adversarial: churn past the ratio takes the slow path (full rebuild)."""
    rnd = random.Random(5)
    pool = _pairs(rnd, 600)
    first = ac.scanner_for_pairs(pool)
    pool = _evict(rnd, pool, drop=400, add=400)
    scanner = ac.scanner_for_pairs(pool)
    assert scanner is not first
    assert len(builds) == 2


def test_repeated_evictions_stay_exact():
    """Falsification over a drain sequence: every step equals a fresh build,
    and dead tokens accumulating eventually force a rebuild (new scanner)."""
    rnd = random.Random(6)
    pool = _pairs(rnd, 600)
    scanners = {id(ac.scanner_for_pairs(pool))}
    for _ in range(30):
        pool = _evict(rnd, pool, drop=15, add=15)
        scanner = ac.scanner_for_pairs(pool)
        scanners.add(id(scanner))
        data = _data_with(rnd, pool, 30)
        assert scanner.scan(data, min_len=2) == _fresh_scan(pool, data)
    assert 1 < len(scanners) < 30
