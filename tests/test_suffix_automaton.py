"""Suffix automaton: corpus-wide repeated substrings -> dictionary tokens, zero execs."""

import itertools
from types import SimpleNamespace

import pytest

from fuzzer_tool.core.auto_dict import MAX_AUTO_TOKEN, MIN_AUTO_TOKEN
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.run_mode import Switch
from fuzzer_tool.core.suffix_automaton import SuffixAutomaton, corpus_tokens
from fuzzer_tool.services.fuzzer import Fuzzer
from tests.test_global_snapshot import _fuzz_kwargs

_RNG = RandPool(seed=0x5A11)
MAGIC = b"\x89PNG\r\n\x1a\nIHDR"


def _count(seeds: list[bytes], sub: bytes) -> int:
    """Brute-force (overlapping) occurrence count: independent oracle."""
    m = len(sub)
    return sum(1 for s in seeds for i in range(len(s) - m + 1) if s[i : i + m] == sub)


def test_counts_match_brute_force():
    seeds = [b"abcabxabcd", b"xabcab", b"bcab"]
    sam = SuffixAutomaton()
    for s in seeds:
        sam.add(s)
    for sub in (b"ab", b"abc", b"bcab", b"cab", b"abcd", b"zz"):
        assert sam.count(sub) == _count(seeds, sub), sub


def test_control_oracle_against_itself():
    seeds = [_RNG.randbytes(64) for _ in range(3)]
    assert _count(seeds, seeds[0][:3]) == _count(list(seeds), bytes(seeds[0][:3]))


def test_shared_header_becomes_token():
    seeds = [_RNG.randbytes(40) + MAGIC + _RNG.randbytes(40) for _ in range(4)]
    assert MAGIC in corpus_tokens(seeds)


def test_tokens_repeat_and_fit_bounds():
    seeds = [b"GET /a HTTP/1.1\r\n", b"GET /bb HTTP/1.1\r\n", b"POST /c HTTP/1.1\r\n"]
    toks = corpus_tokens(seeds)
    assert b" HTTP/1.1\r\n" in toks
    for t in toks:
        assert MIN_AUTO_TOKEN <= len(t) <= MAX_AUTO_TOKEN
        assert _count(seeds, t) >= 2


def test_falsification_unique_bytes_yield_nothing():
    seeds = [bytes(range(i * 50, i * 50 + 50)) for i in range(5)]
    assert corpus_tokens(seeds) == []


def test_tokens_never_span_two_seeds():
    # "abcdef" exists only across the seam of "abc" + "def".
    seeds = [b"xxabc", b"defyy", b"xxabc", b"defyy"]
    assert all(b"cd" not in t for t in corpus_tokens(seeds))


def test_adversarial_uniform_fill_skipped():
    seeds = [bytes(200), bytes(300), b"\xff" * 100]
    assert corpus_tokens(seeds) == []


def test_adversarial_long_shared_blob_bounded():
    # A shared blob (distinct bytes: no inner repeats) yields its right edge
    # only, not one window per byte.
    blob = bytes(range(256))
    seeds = [b"A" + blob + b"B", b"C" + blob + b"D"]
    assert corpus_tokens(seeds) == [blob[-MAX_AUTO_TOKEN:]]


def test_adversarial_budget_and_limit():
    seeds = [MAGIC + _RNG.randbytes(100) for _ in range(50)]
    assert corpus_tokens(seeds, budget=0) == []
    assert len(corpus_tokens(seeds, limit=1)) == 1
    assert corpus_tokens([]) == []


def test_regression_repeated_prefix_keeps_counts():
    """A seed repeating a known prefix once made a duplicate state ("a" 6 vs 7)."""
    seeds = [b"abab", b"aabba", b"aa"]
    sam = SuffixAutomaton()
    for s in seeds:
        sam.add(s)
    assert sam.count(b"a") == _count(seeds, b"a")


def test_counts_exhaustive_small_pairs():
    words = [bytes(w) for n in range(5) for w in itertools.product(b"ab", repeat=n)]
    subs = [bytes(w) for n in range(1, 4) for w in itertools.product(b"ab", repeat=n)]
    for x, y in itertools.product(words, repeat=2):
        sam = SuffixAutomaton()
        sam.add(x)
        sam.add(y)
        for sub in subs:
            assert sam.count(sub) == _count([x, y], sub), (x, y, sub)


# ── wiring: --corpus-tokens -> Fuzzer -> dictionary ─────────────────────────


@pytest.mark.parametrize(("argv", "want"), [([], Switch.OFF), (["--corpus-tokens"], Switch.ON)])
def test_cli_flag_reaches_fuzzer(monkeypatch, tmp_path, argv, want):
    kw = _fuzz_kwargs(monkeypatch, tmp_path, argv)
    assert kw["corpus_tokens"] is want


def test_learn_adds_tokens_to_dictionary():
    f = SimpleNamespace(dictionary=[b"seen"], corpus=[b"x" + MAGIC + b"1", b"y" + MAGIC + b"2"])
    f._operators = SimpleNamespace(_add_auto_tokens=f.dictionary.extend)
    Fuzzer._learn_corpus_tokens(f)
    assert f.dictionary == [b"seen", *corpus_tokens(f.corpus)]
    assert MAGIC in f.dictionary
