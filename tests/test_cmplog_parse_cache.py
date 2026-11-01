"""CMP lines parsed once across drains (bounded cache keyed by the raw line).

On FFmpeg 9.0.2 71% of a drain's distinct lines were already parsed by an
earlier drain (input-invariant comparisons), and the parse was ~46% of
``collect_tokens`` (~9.4% of fuzz_one). The parse is a pure function of the
line, so a cache cannot change the output.
"""

from __future__ import annotations

import random

from fuzzer_tool.adapters import track_parser as tp
from fuzzer_tool.core.cmplog import CmplogCollector

LINES = [
    "CMP 0a0b 0c0d 0 2 0x401000",
    "CMP 0a0b 0c0d 0 2 0x401000",  # repeat inside one drain
    "  CMP 00ff 0100 1 2 0x401004",  # leading spaces still parse
    "CMP 0000 0000 0 2 0x401008",  # both empty after strip? no: zero bytes, kept
    "CMP zz 01 0 1 0x40100c",  # bad hex: skipped
    "CMP 01",  # too short: skipped
    "DIV 0300 2 0x401010",  # not a comparison
    "CMP 41424344 45464748 0 4",  # no pc
    "CMP 01 02",  # no result/width/pc
    "CMP 0a0b 0c0d 0 2 0x401014",  # same operands, other pc
]


def _drains(seed: int, n: int = 6) -> list[list[str]]:
    rng = random.Random(seed)
    return [[rng.choice(LINES) for _ in range(40)] for _ in range(n)]


def test_cached_parse_equals_uncached_across_drains():
    """Oracle: the uncached parser, drain by drain; Rule 46 control first."""
    cache: dict = {}
    for drain in _drains(1):
        ref = tp.cond_tuples_from_cmplog_text(drain)
        assert tp.cond_tuples_from_cmplog_text(drain) == ref
        assert tp.cond_tuples_from_cmplog_text(drain, cache=cache) == ref


def test_falsify_hits_skip_the_parse(monkeypatch):
    calls = []
    real = tp._parse_cmp_line
    monkeypatch.setattr(tp, "_parse_cmp_line", lambda line: calls.append(line) or real(line))
    cache: dict = {}
    distinct = list(dict.fromkeys(LINES))

    tp.cond_tuples_from_cmplog_text(distinct, cache=cache)
    first = len(calls)
    tp.cond_tuples_from_cmplog_text(distinct, cache=cache)

    assert first == len(distinct)
    assert len(calls) == first


def test_adversarial_cache_is_bounded(monkeypatch):
    monkeypatch.setattr(tp, "CMP_PARSE_CACHE_MAX", 4)
    cache: dict = {}
    lines = [f"CMP {i:04x} {i + 1:04x} 0 2 0x{i:x}" for i in range(10)]

    out = tp.cond_tuples_from_cmplog_text(lines, cache=cache)

    assert len(out) == 10
    assert len(cache) <= 4


def test_collector_keeps_one_cache_across_drains(monkeypatch):
    calls = []
    real = tp._parse_cmp_line
    monkeypatch.setattr(tp, "_parse_cmp_line", lambda line: calls.append(line) or real(line))
    c = CmplogCollector()
    lines = ["CMP 0a0b0c0d 01020304 0 4 0x401000", "CMP 11223344 55667788 0 4 0x401004"]

    c._parse_lines(list(lines))
    c._parse_lines(list(lines))

    assert len(calls) == 2
