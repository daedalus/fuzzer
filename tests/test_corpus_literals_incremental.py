"""Regression tests for incremental corpus-literal extraction.

`extract_corpus_literals` was called with the entire corpus every time
the corpus grew, so a campaign paid O(corpus^2) bytes of scanning. It now
takes an optional accumulator so a caller can fold in only the seeds it
has not scanned yet, and the per-byte class tests come from a table.

The scanner's behaviour must be identical either way. In particular
underscore belongs to *both* the alpha run and the symbol run under the
original predicates, which a naive single-class table would get wrong.
"""

from __future__ import annotations

import os
import random

from fuzzer_tool.core.mutations import LiteralAccumulator, extract_corpus_literals
from fuzzer_tool.core.mutations.generic import (
    _LIT_ALPHA,
    _LIT_CLASS,
    _LIT_DIGIT,
    _LIT_SYMBOL,
)


# ---------------------------------------------------------------------------
# Oracle: the original predicates, inline.
# ---------------------------------------------------------------------------


def _old_extract(corpus):
    int_lits: list[bytes] = []
    str_lits: list[bytes] = []
    seen_int: set[bytes] = set()
    seen_str: set[bytes] = set()
    for raw in corpus:
        raw = bytes(raw)
        i = 0
        while i < len(raw):
            if 0x30 <= raw[i] <= 0x39:
                j = i + 1
                while j < len(raw) and 0x30 <= raw[j] <= 0x39:
                    j += 1
                if j - i >= 2:
                    lit = raw[i:j]
                    if lit not in seen_int:
                        seen_int.add(lit)
                        int_lits.append(lit)
                i = j
                continue
            if (
                raw[i] == 45
                and i + 2 < len(raw)
                and all(0x30 <= b <= 0x39 for b in raw[i + 1 : i + 3])
            ):
                j = i + 1
                while j < len(raw) and 0x30 <= raw[j] <= 0x39:
                    j += 1
                lit = raw[i:j]
                if lit not in seen_int:
                    seen_int.add(lit)
                    int_lits.append(lit)
                i = j
                continue
            if (0x61 <= raw[i] <= 0x7A) or (0x41 <= raw[i] <= 0x5A) or raw[i] == 0x5F:
                j = i + 1
                while j < len(raw) and (
                    (0x61 <= raw[j] <= 0x7A) or (0x41 <= raw[j] <= 0x5A) or raw[j] == 0x5F
                ):
                    j += 1
                if j - i >= 3:
                    lit = raw[i:j]
                    if lit not in seen_str:
                        seen_str.add(lit)
                        str_lits.append(lit)
                i = j
                continue
            if 0x20 <= raw[i] <= 0x7E and not (
                (0x30 <= raw[i] <= 0x39)
                or (0x61 <= raw[i] <= 0x7A)
                or (0x41 <= raw[i] <= 0x5A)
            ):
                j = i + 1
                while (
                    j < len(raw)
                    and 0x20 <= raw[j] <= 0x7E
                    and not (
                        (0x30 <= raw[j] <= 0x39)
                        or (0x61 <= raw[j] <= 0x7A)
                        or (0x41 <= raw[j] <= 0x5A)
                    )
                ):
                    j += 1
                if j - i >= 3:
                    lit = raw[i:j]
                    if lit not in seen_str:
                        seen_str.add(lit)
                        str_lits.append(lit)
                i = j
                continue
            i += 1
    return int_lits, str_lits


# ---------------------------------------------------------------------------


def test_class_table_matches_the_original_predicates():
    for b in range(256):
        flags = _LIT_CLASS[b]
        assert bool(flags & _LIT_DIGIT) == (0x30 <= b <= 0x39)
        assert bool(flags & _LIT_ALPHA) == (
            (0x61 <= b <= 0x7A) or (0x41 <= b <= 0x5A) or b == 0x5F
        )
        assert bool(flags & _LIT_SYMBOL) == (
            0x20 <= b <= 0x7E
            and not ((0x30 <= b <= 0x39) or (0x61 <= b <= 0x7A) or (0x41 <= b <= 0x5A))
        )


def test_underscore_is_in_both_the_alpha_and_symbol_classes():
    """A single-class table would break symbol runs containing '_'."""
    assert _LIT_CLASS[0x5F] & _LIT_ALPHA
    assert _LIT_CLASS[0x5F] & _LIT_SYMBOL
    # A symbol run continues through an underscore, so "=_=" is one
    # three-byte string literal, not three one-byte non-literals.
    assert extract_corpus_literals([b"=_="]) == ([], [b"=_="])
    assert extract_corpus_literals([b"=_="]) == _old_extract([b"=_="])


def test_matches_the_original_scanner_on_random_corpora():
    rnd = random.Random(0)
    alphabet = b"abcXYZ_0123456789-=.;/ \x00\xff"
    for _ in range(300):
        corpus = []
        for _ in range(rnd.randrange(1, 5)):
            n = rnd.randrange(0, 400)
            if rnd.random() < 0.5:
                corpus.append(bytes(rnd.choice(alphabet) for _ in range(n)))
            else:
                corpus.append(os.urandom(n))
        assert extract_corpus_literals(corpus) == _old_extract(corpus)


def test_incremental_equals_one_full_scan():
    """Feeding seeds one at a time must give the same lists, in order."""
    rnd = random.Random(1)
    for _ in range(150):
        corpus = [
            bytes(rnd.choice(b"abc_12-=;") for _ in range(rnd.randrange(0, 200)))
            for _ in range(6)
        ]
        acc = LiteralAccumulator()
        for seed in corpus:
            extract_corpus_literals([seed], acc)
        assert acc.result() == _old_extract(corpus)
        assert acc.scanned == len(corpus)


def test_accumulator_dedups_across_calls():
    acc = LiteralAccumulator()
    extract_corpus_literals([b"name=1234"], acc)
    first = ([*acc.int_lits], [*acc.str_lits])
    extract_corpus_literals([b"name=1234"], acc)
    assert (acc.int_lits, acc.str_lits) == first
    assert acc.scanned == 2


def test_scanned_counter_tracks_entries_not_bytes():
    acc = LiteralAccumulator()
    extract_corpus_literals([b"abc", b"def", b"ghi"], acc)
    assert acc.scanned == 3
    extract_corpus_literals([], acc)
    assert acc.scanned == 3


def test_bytearray_and_memoryview_still_accepted():
    text = b"count=1000;name=foo"
    assert extract_corpus_literals([bytearray(text)]) == extract_corpus_literals([text])
    assert extract_corpus_literals([memoryview(text)]) == extract_corpus_literals([text])


def test_empty_corpus():
    assert extract_corpus_literals([]) == ([], [])
    acc = LiteralAccumulator()
    assert extract_corpus_literals([], acc) == ([], [])
