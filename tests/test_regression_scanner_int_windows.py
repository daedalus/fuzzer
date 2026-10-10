"""TokenScanner scans 1/2/4/8-byte tokens as integers, without a trie.

FFmpeg's cmplog pool is all 1/2/4/8-byte operands (4673 tokens: 99/347/
2514/1713). Eviction churn between colorize calls exceeds what retarget
absorbs, so nearly every call rebuilt the Aho-Corasick trie: 7.4% of
fuzz-loop wall time. The fast path reads the seed as little-endian
integers at every offset and binary-searches a sorted token array; other
widths keep the automaton / find path (slow path). Results must equal a
plain ``bytes.find`` sweep.
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


def _ref_scan(tokens, data, min_len):
    """Oracle: one find loop per distinct token, independent of the scanner."""
    found = {}
    for tok in dict.fromkeys(tokens):
        if not tok or len(tok) < min_len:
            continue
        offs, pos = [], 0
        while (idx := data.find(tok, pos)) >= 0:
            offs.append(idx)
            pos = idx + 1
        if offs:
            found[tok] = offs
    return found


def _tokens(rnd, n, widths):
    return [rnd.randbytes(rnd.choice(widths)) for _ in range(n)]


def _data_with(rnd, tokens, k, filler=6):
    """Seed holding *k* sampled tokens among random bytes (all alignments)."""
    parts = []
    for tok in rnd.sample(tokens, k):
        parts += [rnd.randbytes(rnd.randint(0, filler)), tok]
    return b"".join(parts)


def _case(seed, n, widths, k=80):
    rnd = random.Random(seed)
    toks = _tokens(rnd, n, widths)
    return toks, _data_with(rnd, toks, min(k, n))


def test_control_oracle_matches_itself():
    """Rule 46: the reference agrees with a second run of itself."""
    toks, data = _case(1, 2000, (1, 2, 4, 8))
    assert _ref_scan(toks, data, 1) == _ref_scan(toks, data, 1)


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("min_len", [1, 2, 3, 5])
@pytest.mark.parametrize("widths", [(1, 2, 4, 8), (2, 3, 4, 8, 16), (3, 5, 7)])
@pytest.mark.parametrize("n", [50, 3000])
def test_scan_matches_find(seed, min_len, widths, n):
    """Falsification: fast, slow and mixed pools equal the find sweep,
    offsets ascending, both sides of the AC_MIN_TOKENS switch."""
    toks, data = _case(seed, n, widths)
    assert ac.TokenScanner(toks).scan(data, min_len=min_len) == _ref_scan(toks, data, min_len)


def test_fast_widths_build_no_automaton(builds):
    """Falsification: a 1/2/4/8-only pool past AC_MIN_TOKENS builds no trie."""
    toks, data = _case(4, ac.AC_MIN_TOKENS * 8, (1, 2, 4, 8))
    assert ac.TokenScanner(toks).scan(data, min_len=2) == _ref_scan(toks, data, 2)
    assert builds == []


def test_eviction_never_builds_for_fast_pool(builds):
    """Falsification: the collector's rebind-every-drain churn (any size)
    stays exact and trie-free."""
    rnd = random.Random(5)
    pool = [(rnd.randbytes(rnd.choice((1, 2, 4, 8))), rnd.randbytes(4)) for _ in range(5000)]
    for _ in range(6):
        pool = pool[2500:] + [(rnd.randbytes(4), rnd.randbytes(8)) for _ in range(2500)]
        toks = ac.tokens_from_pairs(pool)
        data = _data_with(rnd, toks, 60)
        assert ac.scanner_for_pairs(pool).scan(data, min_len=2) == _ref_scan(toks, data, 2)
    assert builds == []


def test_grown_pool_absorbs_fast_tokens():
    """Falsification: in-place growth keeps the scanner and finds new tokens."""
    rnd = random.Random(6)
    pool = [(rnd.randbytes(4), rnd.randbytes(2)) for _ in range(1000)]
    first = ac.scanner_for_pairs(pool)
    pool.extend((rnd.randbytes(8), rnd.randbytes(1)) for _ in range(1000))
    toks = ac.tokens_from_pairs(pool)
    data = _data_with(rnd, toks[2000:], 40)
    assert ac.scanner_for_pairs(pool) is first
    assert first.scan(data, min_len=1) == _ref_scan(toks, data, 1)


@pytest.mark.parametrize(
    "data",
    [b"", b"\x00", b"\x00" * 7, b"\x00" * 8, b"\xff" * 33, bytes(range(256))],
)
def test_edge_buffers(data):
    """Adversarial: buffers shorter than a width, exact width, extremes of
    the integer range (0 / all-ones), every byte value."""
    toks = [
        b"\x00",
        b"\x00\x00",
        b"\x00" * 4,
        b"\x00" * 8,
        b"\xff" * 8,
        b"\xff\xff",
        bytes(range(8)),
    ]
    toks += _tokens(random.Random(7), 3000, (1, 2, 4, 8))
    assert ac.TokenScanner(toks).scan(data, min_len=1) == _ref_scan(toks, data, 1)


def test_overlapping_and_repeated_hits():
    """Adversarial: overlapping occurrences at every alignment are all kept."""
    toks = [b"aa", b"aaaa", b"aaaaaaaa", b"ab"] + _tokens(random.Random(8), 2000, (2, 4, 8))
    data = b"a" * 37 + b"b" + b"a" * 11
    assert ac.TokenScanner(toks).scan(data, min_len=2) == _ref_scan(toks, data, 2)


@pytest.mark.parametrize("group_min", [0, 1 << 30])
@pytest.mark.parametrize("seed", [1, 2])
def test_both_hit_assemblies_match(monkeypatch, group_min, seed):
    """Falsification: numpy grouping and per-hit appends both equal find."""
    monkeypatch.setattr(ac, "GROUP_MIN_HITS", group_min)
    toks, data = _case(seed, 3000, (1, 2, 4, 8), k=400)
    data += b"\x00" * 300 + data[:200]  # repeats of the same tokens
    toks += [b"\x00" * 2, b"\x00" * 8]
    assert ac.TokenScanner(toks).scan(data, min_len=1) == _ref_scan(toks, data, 1)


def test_long_zero_run_groups_exactly():
    """Adversarial: thousands of overlapping hits of few tokens (grouped path)."""
    toks = [b"\x00", b"\x00\x00", b"\x00" * 4, b"\x00" * 8, b"\x01\x00"]
    data = b"\x00" * 5000 + b"\x01" + b"\x00" * 999
    assert ac.TokenScanner(toks).scan(data, min_len=1) == _ref_scan(toks, data, 1)
