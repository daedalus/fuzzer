"""Regression tests for finding #24 — unbounded grammar expansion product.

Per-token repeats were clamped to ``_MAX_REPEAT`` (32) but their PRODUCT was
not, and ``max_len`` truncated only *after* the whole tree had been expanded.
A grammar file of chained ``{32}`` rules -- something ``--grammar`` accepts
from disk -- therefore OOM'd or hung the fuzzer itself no matter how small a
``max_len`` the caller asked for.

The tests below pin three things:

1. the cost is bounded (a chain deep enough to have been fatal now returns
   promptly and small),
2. the *output* is unchanged for grammars that fit, byte for byte, against
   the whole-expansion-then-slice behaviour it replaces,
3. ``generate(max_len=k)`` equals ``generate()[:k]`` for the same RNG state,
   which is the identity the three call sites inside ``mutate()`` rely on.
"""

from __future__ import annotations

import logging
import random
import time

import pytest

from fuzzer_tool.core.grammar import GENERATION_BYTE_CAP, Grammar

GRAMMAR_FILES = [
    "dictionaries/png.gram",
    "dictionaries/der.gram",
    "dictionaries/jpeg.gram",
    "dictionaries/rar.gram",
]


def _chain_grammar(levels: int, repeat: int = 32, leaf: str = "AAAAAAAA") -> Grammar:
    """start = r0{N}; r0 = r1{N}; ... ; r<levels-1> = "AAAAAAAA"."""
    rules = [f"start = r0{{{repeat}}}"]
    rules += [f"r{i} = r{i + 1}{{{repeat}}}" for i in range(levels - 1)]
    rules += [f'r{levels - 1} = "{leaf}"']
    g = Grammar()
    g.parse("\n".join(rules))
    return g


@pytest.fixture(autouse=True)
def _quiet_depth_warnings():
    """The deep-chain grammars log one warning per exhausted leaf."""
    logger = logging.getLogger("fuzzer_tool.core.grammar")
    prev = logger.level
    logger.setLevel(logging.CRITICAL)
    yield
    logger.setLevel(prev)


class TestCostIsBounded:
    @pytest.mark.parametrize("levels", [4, 5, 6, 8, 10])
    def test_deep_chain_returns_promptly(self, levels):
        """Measured before the fix: 0.65s at 4 levels, 22s/560MB at 5, x32 after."""
        g = _chain_grammar(levels)
        t0 = time.monotonic()
        out = g.generate("start", max_len=16)
        elapsed = time.monotonic() - t0
        assert len(out) <= 16
        assert elapsed < 2.0, f"{levels} levels took {elapsed:.2f}s"

    def test_no_max_len_still_bounded(self):
        """max_len=0 means "no truncation", not "no ceiling"."""
        g = _chain_grammar(6)
        t0 = time.monotonic()
        out = g.generate("start")
        elapsed = time.monotonic() - t0
        assert len(out) <= GENERATION_BYTE_CAP + 64
        assert elapsed < 5.0, f"unbounded generate took {elapsed:.2f}s"

    def test_wide_single_rule_is_bounded(self):
        """The budget must span the whole tree, not reset per rule."""
        g = Grammar()
        g.parse('start = a{32}\na = b{32}\nb = c{32}\nc = d{32}\nd = "ABCDEFGH"')
        out = g.generate("start", max_len=100)
        assert len(out) == 100

    def test_mutate_paths_are_bounded(self):
        """_mutate_extend/_mutate_insert/_mutate_replace_section reach generate()."""
        g = _chain_grammar(6)
        rng = random.Random(1234)
        t0 = time.monotonic()
        for _ in range(50):
            g.mutate(b"seed data here", max_len=256, rng=rng)
        elapsed = time.monotonic() - t0
        assert elapsed < 5.0, f"50 mutations took {elapsed:.2f}s"


class TestOutputUnchanged:
    """A budget that changed the bytes would be a different bug, not a fix."""

    @pytest.mark.parametrize("path", GRAMMAR_FILES)
    @pytest.mark.parametrize("max_len", [1, 8, 64, 512, 4096])
    def test_generate_equals_expand_then_slice(self, path, max_len):
        """generate(max_len=k) == generate()[:k] for the same RNG state.

        This is exactly the substitution made at the three call sites inside
        mutate(), and it holds because expansion is a left-to-right
        concatenation: bytes already emitted cannot be changed by expansions
        that are abandoned afterwards.
        """
        g = Grammar()
        g.parse_file(path)
        for seed in range(50):
            g._rng = random.Random(seed)
            sliced = g.generate()[:max_len]
            g._rng = random.Random(seed)
            budgeted = g.generate(max_len=max_len)
            assert budgeted == sliced, f"{path} seed={seed} max_len={max_len}"

    @pytest.mark.parametrize("path", GRAMMAR_FILES)
    def test_shipped_grammars_still_produce_output(self, path):
        """A budget of zero would satisfy every cost assertion above."""
        g = Grammar()
        g.parse_file(path)
        g._rng = random.Random(7)
        outs = [g.generate(max_len=512) for _ in range(20)]
        assert any(outs), f"{path} generated nothing at all"
        assert max(len(o) for o in outs) > 1

    def test_budget_is_rearmed_per_call(self):
        """A spent budget must not leak into the next generate()."""
        g = _chain_grammar(5)
        first = g.generate("start", max_len=64)
        second = g.generate("start", max_len=64)
        assert len(first) == 64
        assert len(second) == 64
