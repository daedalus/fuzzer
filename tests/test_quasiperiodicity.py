"""Tests for quasiperiodicity.py — the string-cover-based novelty signal.

The load-bearing check is the cross-validation against a brute-force
reference (try every candidate length directly, no border theorem, no
Z-function) over many random small strings -- without it, a subtle bug in
the border-chain shortcut or the Z-array coverage sweep could silently
return a wrong-but-plausible-looking cover length forever.
"""

import random

import pytest

from fuzzer_tool.core.quasiperiodicity import (
    QP_SAMPLE_BYTES,
    QuasiperiodicityAnalyzer,
    cover_ratio,
    is_cover,
    is_quasiperiodic,
    prefix_function,
    shortest_cover_length,
    z_function,
)


def _brute_force_is_cover(s: bytes, c: int) -> bool:
    """Reference implementation: find every occurrence of s[0:c] in s by
    direct substring search, then check the union of [pos, pos+c) covers
    [0, len(s)). No Z-function, no shortcuts.
    """
    n = len(s)
    if c <= 0 or c > n:
        return False
    u = s[:c]
    covered = [False] * n
    start = 0
    while True:
        pos = s.find(u, start)
        if pos == -1:
            break
        for i in range(pos, min(pos + c, n)):
            covered[i] = True
        start = pos + 1
    return all(covered)


def _brute_force_shortest_cover(s: bytes) -> int:
    n = len(s)
    if n <= 1:
        return n
    for c in range(1, n):
        if _brute_force_is_cover(s, c):
            return c
    return n


class TestPrefixAndZFunction:
    def test_prefix_function_known_case(self):
        # "abcabcabc": pi = [0,0,0,1,2,3,4,5,6]
        assert prefix_function(b"abcabcabc") == [0, 0, 0, 1, 2, 3, 4, 5, 6]

    def test_z_function_known_case(self):
        # "aaaaa": z = [_,4,3,2,1] (z[0] unused/0 by convention)
        assert z_function(b"aaaaa") == [0, 4, 3, 2, 1]

    def test_empty_string(self):
        assert prefix_function(b"") == []
        assert z_function(b"") == []


class TestIsCoverAgainstBruteForce:
    def test_matches_brute_force_random_strings(self):
        rng = random.Random(0)
        alphabet = bytes(range(4))  # small alphabet maximizes overlap structure
        for _ in range(300):
            n = rng.randint(1, 25)
            s = bytes(rng.choice(alphabet) for _ in range(n))
            z = z_function(s)
            for c in range(1, n + 1):
                assert is_cover(s, c, z) == _brute_force_is_cover(s, c), (s, c)

    def test_matches_brute_force_wider_alphabet(self):
        rng = random.Random(1)
        for _ in range(150):
            n = rng.randint(1, 20)
            s = bytes(rng.randint(0, 255) for _ in range(n))
            z = z_function(s)
            for c in range(1, n + 1):
                assert is_cover(s, c, z) == _brute_force_is_cover(s, c), (s, c)


class TestShortestCoverAgainstBruteForce:
    def test_matches_brute_force_random_strings(self):
        rng = random.Random(2)
        alphabet = bytes(range(3))
        for _ in range(300):
            n = rng.randint(1, 30)
            s = bytes(rng.choice(alphabet) for _ in range(n))
            assert shortest_cover_length(s) == _brute_force_shortest_cover(s), s

    def test_matches_brute_force_wider_alphabet(self):
        rng = random.Random(3)
        for _ in range(150):
            n = rng.randint(1, 25)
            s = bytes(rng.randint(0, 255) for _ in range(n))
            assert shortest_cover_length(s) == _brute_force_shortest_cover(s), s


class TestKnownCases:
    def test_repeated_single_byte_covers_with_length_one(self):
        assert shortest_cover_length(b"aaaaaaaa") == 1
        assert is_quasiperiodic(b"aaaaaaaa")

    def test_overlapping_cover_ababa(self):
        # "ababa": cover "ab" (len 2) occurs at 0 and 2, covering
        # 0-2,2-4; position 4 ('a') is covered by the occurrence at
        # position 2 extending to index 3 only -- so "ab" alone leaves
        # index 4 uncovered unless an occurrence starts there. "a" (len 1)
        # trivially covers everything since 'a' appears at every other
        # position and the gaps ('b' at odd positions) are NOT covered by
        # a length-1 "a" cover. Verify against brute force rather than
        # asserting a hand-guessed answer.
        s = b"ababa"
        assert shortest_cover_length(s) == _brute_force_shortest_cover(s)

    def test_no_proper_cover_random_looking_string(self):
        # A string with no internal repetition at all has no proper cover.
        s = bytes([0, 1, 2, 3, 4, 5, 6, 7])
        assert shortest_cover_length(s) == len(s)
        assert not is_quasiperiodic(s)

    def test_empty_and_single_byte(self):
        assert shortest_cover_length(b"") == 0
        assert shortest_cover_length(b"x") == 1
        assert not is_quasiperiodic(b"")
        assert not is_quasiperiodic(b"x")

    def test_cover_ratio_direction(self):
        # Highly covered (redundant) -> low ratio; no cover (novel) -> ratio 1.0.
        assert cover_ratio(b"aaaaaaaaaaaaaaaa") < 0.2
        assert cover_ratio(bytes(range(16))) == 1.0
        assert cover_ratio(b"") == 1.0


class TestSampleCapConstant:
    def test_cap_is_positive_and_reasonable(self):
        # Documents the tradeoff rather than pinning an exact value: must be
        # small enough to bound the O(n^2) worst case, large enough to be a
        # meaningful sample.
        assert 0 < QP_SAMPLE_BYTES <= 65536


class TestQuasiperiodicityAnalyzer:
    def test_disabled_returns_neutral(self):
        qp = QuasiperiodicityAnalyzer(enabled=False)
        assert qp.compute_seed_ratio(b"aaaa") == 1.0
        assert qp.compute_seed_novelty(b"aaaa") == 1.0

    def test_empty_seed_returns_neutral(self):
        qp = QuasiperiodicityAnalyzer()
        assert qp.compute_seed_ratio(b"") == 1.0

    def test_novelty_direction_matches_ppmd_semantics(self):
        qp = QuasiperiodicityAnalyzer()
        redundant = b"ab" * 500
        rng = random.Random(4)
        novel = bytes(rng.randint(0, 255) for _ in range(1000))
        assert qp.compute_seed_novelty(redundant) < qp.compute_seed_novelty(novel)

    def test_memoized_by_digest(self):
        qp = QuasiperiodicityAnalyzer()
        data = b"abcabcabcabc"
        r1 = qp.compute_seed_ratio(data)
        assert len(qp._ratios) == 1
        r2 = qp.compute_seed_ratio(data)
        assert r1 == r2
        assert len(qp._ratios) == 1  # still one entry, not recomputed as a new one

    def test_cache_bounded(self):
        qp = QuasiperiodicityAnalyzer()
        from fuzzer_tool.core.quasiperiodicity import QP_CACHE_MAX

        for i in range(QP_CACHE_MAX + 10):
            qp.compute_seed_ratio(i.to_bytes(4, "big") * 3)
        assert len(qp._ratios) <= QP_CACHE_MAX


class TestAnalyzerRegistryWiring:
    def test_registered_and_gated_on_flag(self):
        from fuzzer_tool.core.analyzer_registry import REGISTRY as ANALYZER_REGISTRY

        assert "quasiperiodicity" in ANALYZER_REGISTRY.names()
        spec = ANALYZER_REGISTRY._specs["quasiperiodicity"]

        class _FOff:
            _corpus_quasiperiodicity_requested = False

        class _FOn:
            _corpus_quasiperiodicity_requested = True

        assert not spec.available(_FOff())
        assert spec.available(_FOn())

        f = _FOn()
        spec.activate(f)
        assert f._qp is not None and f._qp.enabled
        spec.deactivate(f)
        assert f._qp is None
