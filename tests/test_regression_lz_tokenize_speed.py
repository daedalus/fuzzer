"""Regression: ``_lz_tokenize`` scanned every window position per byte.

100 ms per call on 4 KB ffmpeg inputs (24 s of a 574 s profile). The hash
chain visits only positions sharing the first ``min_match`` bytes; output
must stay token-for-token identical to the naive greedy parse below.
"""

import time

import pytest

from fuzzer_tool.core.mutations.structured import _lz_tokenize

_MIN_MATCH = 3


def _naive(block: bytearray, min_match: int, max_match: int) -> list[tuple]:
    """The pre-fix parser, verbatim: the oracle."""
    tokens: list[tuple] = []
    i = 0
    while i < len(block):
        best_dist, best_len = 0, 0
        search_start = max(0, i - 32768)
        for j in range(search_start, i):
            match_len = 0
            while (
                match_len < max_match
                and i + match_len < len(block)
                and block[j + match_len] == block[i + match_len]
            ):
                match_len += 1
            if match_len > best_len:
                best_dist = i - j
                best_len = match_len
        if best_len >= min_match:
            tokens.append(("ref", best_dist, best_len))
            i += best_len
        else:
            tokens.append(("lit", block[i]))
            i += 1
    return tokens


def _lcg_bytes(n: int, seed: int, alphabet: int) -> bytearray:
    out = bytearray(n)
    x = seed
    for k in range(n):
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        out[k] = (x >> 16) % alphabet
    return out


_CASES = {
    "low_entropy": _lcg_bytes(700, 1, 4),
    "text": bytearray(b"RIFF....WAVEfmt " * 20 + b"data" + bytes(range(64)) * 3),
    "all_zero": bytearray(300),
    "high_entropy": _lcg_bytes(500, 7, 256),
    "overlapping_run": bytearray(b"ab" + b"a" * 200 + b"abcabcabc" * 10),
    "tiny": bytearray(b"abcdefgh"),
}


@pytest.mark.parametrize("name", sorted(_CASES))
@pytest.mark.parametrize("max_match", [3, 4, 64])
def test_regression_lz_tokenize_matches_naive(name, max_match):
    block = _CASES[name]
    max_match = min(max_match, len(block) - 1)
    assert _lz_tokenize(bytearray(block), _MIN_MATCH, max_match) == _naive(
        bytearray(block), _MIN_MATCH, max_match
    )


def test_control_oracle_against_itself():
    """Hard Rule 46."""
    block = _CASES["low_entropy"]
    assert _naive(bytearray(block), _MIN_MATCH, 64) == _naive(bytearray(block), _MIN_MATCH, 64)


def test_does_not_mutate_block():
    block = bytearray(_CASES["text"])
    before = bytes(block)
    _lz_tokenize(block, _MIN_MATCH, 64)
    assert bytes(block) == before


def test_4k_block_is_fast():
    """Adversarial: the profiled size; naive took ~100 ms here."""
    block = _lcg_bytes(4096, 3, 16)
    t = time.perf_counter()
    _lz_tokenize(block, _MIN_MATCH, 64)
    assert time.perf_counter() - t < 0.05
