"""Regression: grammar follow-ups to the expansion-cap fix (PR #111).

(a) Built-in grammars referenced undefined rules (json ``number``, bare
    words in ``http_request``/``elf``); each expanded to ``b"?"``.
(b) ``_boltzmann_expand_tokens`` lacked ``GENERATION_EXPANSION_CAP``, so a
    zero-byte repeat chain cost R^depth under ``generate_boltzmann``.
"""

from __future__ import annotations

import json
import logging
import re

import pytest

from fuzzer_tool.core.grammar import GENERATION_EXPANSION_CAP, GRAMMARS, Grammar, load_grammar
from fuzzer_tool.core.rand_pool import RandPool
from tests.support.scripted_rng import ScriptedRng

# RFC 8259 section 6, written independently of the grammar under test.
RFC8259_NUMBER = re.compile(rb"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?")

SEEDS = range(200)
LEVELS = 8
REPEAT = 32


@pytest.fixture(autouse=True)
def _quiet_grammar_log():
    logger = logging.getLogger("fuzzer_tool.core.grammar")
    prev = logger.level
    logger.setLevel(logging.CRITICAL)
    yield
    logger.setLevel(prev)


def _undefined(g: Grammar) -> set[str]:
    """Names referenced by any ref/repeat token but never defined."""
    refs = {
        tok[1]
        for alts in g.rules.values()
        for alt in alts
        for tok in alt
        if tok[0] in ("ref", "repeat")
    }
    return refs - g.rules.keys()


# --- (a) undefined rule references ----------------------------------------


def test_control_undefined_ref_is_detected():
    """Hard Rule 46: the oracle must flag a known-bad grammar."""
    g = Grammar()
    g.parse('start = a missing{2}\na = "x"')
    assert _undefined(g) == {"missing"}


@pytest.mark.parametrize("name", list(GRAMMARS))
def test_regression_builtin_grammar_no_undefined_refs(name):
    assert _undefined(load_grammar(name)) == set()


def test_falsify_json_number_scripted():
    """Scripted draws spell ``-10.5E+3``; json.loads is the oracle."""
    # choice order: json, value(number), number, minus, int(onenine digit*),
    # onenine("1"), digit("0"), frac, digit("5"), exp, e("E"), sign("+"),
    # digit("3").
    rng = ScriptedRng(
        choice_idxs=[0, 3, 0, 0, 1, 0, 0, 0, 5, 0, 1, 1, 3],
        # minus?, digit*, frac?, digit+, exp?, sign?, digit+
        randints=[1, 1, 1, 1, 1, 1, 1],
    )
    out = load_grammar("json").generate(rng=rng)
    assert out == b"-10.5E+3"
    assert json.loads(out) == -10.5e3


def test_adversarial_json_output_parses_for_many_seeds():
    """Every seed: whole document parses, every number matches RFC 8259."""
    for seed in SEEDS:
        g = load_grammar("json")
        g._rng = Grammar(seed=seed)._rng
        doc = g.generate()
        num = g.generate(rule="number")
        assert b"?" not in doc, (seed, doc)
        json.loads(doc)
        assert RFC8259_NUMBER.fullmatch(num), (seed, num)


# --- (b) Boltzmann expansion cap ------------------------------------------


def _chain(leaf: str) -> Grammar:
    """start = r0{32}; r0 = r1{32}; ...; r7 = <leaf>."""
    rules = [f"start = r0{{{REPEAT}}}"]
    rules += [f"r{i} = r{i + 1}{{{REPEAT}}}" for i in range(LEVELS - 1)]
    rules += [f"r{LEVELS - 1} = {leaf}"]
    g = Grammar(seed=1)
    g.parse("\n".join(rules))
    return g


def _count_samples(monkeypatch) -> list[int]:
    """Count _boltzmann_sample_rule calls (monkeypatch restores it)."""
    calls = [0]
    orig = Grammar._boltzmann_sample_rule

    def counted(self, name, depth, x, memo):
        calls[0] += 1
        return orig(self, name, depth, x, memo)

    monkeypatch.setattr(Grammar, "_boltzmann_sample_rule", counted)
    return calls


@pytest.mark.timeout(5)
def test_regression_boltzmann_empty_chain_bounded(monkeypatch):
    g = _chain('""')
    calls = _count_samples(monkeypatch)
    assert g.generate_boltzmann("start", max_len=16) == b""
    assert calls[0] <= GENERATION_EXPANSION_CAP + 1


@pytest.mark.timeout(5)
def test_adversarial_boltzmann_empty_alts_no_max_len(monkeypatch):
    """Weighted alt draws, no max_len, then a second call: cap resets."""
    g = _chain('"" | "" | ""')
    calls = _count_samples(monkeypatch)
    assert g.generate_boltzmann("start") == b""
    assert calls[0] <= GENERATION_EXPANSION_CAP + 1

    g.parse('fits = a{32}\na = "Z"')
    assert g.generate_boltzmann("fits") == b"Z" * REPEAT


def test_falsify_boltzmann_fitting_output_unchanged():
    """Cap must not truncate output that fits: 32*32 bytes."""
    g = Grammar(seed=1)
    g.parse('start = a{32}\na = b{32}\nb = "Z"')
    assert g.generate_boltzmann("start") == b"Z" * (REPEAT * REPEAT)


ELF_IDENT_LEN = 16  # e_ident[EI_NIDENT]
ELF_MAGIC = b"\x7fELF"


def _pool(seed: int) -> RandPool:
    return RandPool(seed=seed)


def test_regression_elf_grammar_emits_full_ident():
    """The start rule was `magic`, so generation only ever emitted 4 bytes."""
    g = load_grammar("elf")
    for seed in range(40):
        out = g.generate(rng=_pool(seed))
        assert len(out) == ELF_IDENT_LEN
        assert out.startswith(ELF_MAGIC)


def test_adversarial_elf_ident_fields_in_range():
    """EI_CLASS 1/2, EI_DATA 1, EI_VERSION 1, padding all zero."""
    g = load_grammar("elf")
    for seed in range(40):
        out = g.generate(rng=_pool(seed))
        assert out[4] in (1, 2) and out[5] == 1 and out[6] == 1
        assert out[8:] == bytes(ELF_IDENT_LEN - 8)
