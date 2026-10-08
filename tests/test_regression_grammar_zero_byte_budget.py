"""Regression: GENERATION_BYTE_CAP charged emitted bytes only.

Chained repeats over a zero-byte rule (empty literal) charged nothing, so
expansion cost grew as R^depth regardless of ``max_len``.
"""

from __future__ import annotations

import logging

import pytest

from fuzzer_tool.core.grammar import GENERATION_EXPANSION_CAP, Grammar

LEVELS = 8
REPEAT = 32


@pytest.fixture(autouse=True)
def _quiet_grammar_log():
    logger = logging.getLogger("fuzzer_tool.core.grammar")
    prev = logger.level
    logger.setLevel(logging.CRITICAL)
    yield
    logger.setLevel(prev)


def _chain(leaf: str) -> Grammar:
    """start = r0{32}; r0 = r1{32}; ...; r7 = <leaf>."""
    rules = [f"start = r0{{{REPEAT}}}"]
    rules += [f"r{i} = r{i + 1}{{{REPEAT}}}" for i in range(LEVELS - 1)]
    rules += [f"r{LEVELS - 1} = {leaf}"]
    g = Grammar()
    g.parse("\n".join(rules))
    return g


def _count_expansions(monkeypatch) -> list[int]:
    """Count _expand_rule calls via a wrapper (monkeypatch restores it)."""
    calls = [0]
    orig = Grammar._expand_rule

    def counted(self, name, depth):
        calls[0] += 1
        return orig(self, name, depth)

    monkeypatch.setattr(Grammar, "_expand_rule", counted)
    return calls


@pytest.mark.timeout(5)
def test_regression_empty_literal_chain_bounded(monkeypatch):
    g = _chain('""')
    calls = _count_expansions(monkeypatch)
    assert g.generate("start", max_len=16) == b""
    assert calls[0] <= GENERATION_EXPANSION_CAP + 1


@pytest.mark.timeout(5)
def test_adversarial_empty_alts_no_max_len(monkeypatch):
    """Several empty alternatives and no max_len (byte cap 1 MiB)."""
    g = _chain('"" | "" | ""')
    calls = _count_expansions(monkeypatch)
    assert g.generate("start") == b""
    assert calls[0] <= GENERATION_EXPANSION_CAP + 1


def test_falsify_fitting_output_unchanged():
    """Expansion cap must not truncate output that fits: 32*32 bytes."""
    g = Grammar()
    g.parse('start = a{32}\na = b{32}\nb = "Z"')
    assert g.generate("start") == b"Z" * (REPEAT * REPEAT)
