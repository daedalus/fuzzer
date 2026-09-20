"""Regression test: `entropy_deviation` must show up in the enabled-features
banner (`Fuzzer._print_enabled_features`), the same way its two siblings
`entropy_kl` and `entropy_zscore` already do.

Context: `entropy_deviation` is wired into the strategy (services/fuzzer.py
sets `self._entropy_deviation`), into `--hail-mary` (`entropy_deviation` is
present in cli/commands.py's `_HAIL_MARY_FLAGS`), and into the live
stats/report lines (`services/report.py::_entropy_seed_lines`,
`services/stats.py`) -- but `_print_enabled_features` never checked for it,
so a `--hail-mary` run silently omitted it from the startup banner even
though the strategy was actually running underneath.
"""

from __future__ import annotations

from types import SimpleNamespace

from fuzzer_tool.core.schedulers.seed_entropy_deviation import (
    EntropyDeviationSeedStrategy,
)
from fuzzer_tool.services.fuzzer import Fuzzer


def _make_fake_fuzzer(**overrides):
    """A minimal stand-in exposing every attribute
    `_print_enabled_features` reads unconditionally (via plain `self.x`,
    not `getattr`), all falsy/default, so the method runs without
    AttributeErrors and prints nothing unless overridden.
    """
    base = dict(
        _adaptive_timeout=False,
        _calibrate=0,
        _corpus_boost=0,
        _diff_target=None,
        _exp4=False,
        _format_learner=None,
        _inprocess_runner=False,
        _successive_elim=False,
        _tracer=None,
        branch_cov=False,
        colorize=False,
        debug=False,
        enable_arm_mutator=False,
        enable_regex_bomb=False,
        enable_x86_mutator=False,
        ga=False,
        honggfuzz=False,
        hw_perf=False,
        markov_generate=False,
        markov_trained=False,
        mc_bandit=False,
        mc_cem=False,
        persistent=False,
        pt_cov=False,
        qea=False,
        weizz_tags=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_entropy_deviation_absent_by_default(capsys):
    fake = _make_fake_fuzzer()
    Fuzzer._print_enabled_features(fake)
    out = capsys.readouterr().out
    assert "entropy-deviation" not in out


def test_entropy_deviation_appears_when_active(capsys):
    fake = _make_fake_fuzzer(
        _entropy_deviation=EntropyDeviationSeedStrategy(rng=None)
    )
    Fuzzer._print_enabled_features(fake)
    out = capsys.readouterr().out
    assert "Seed selection:" in out
    assert "entropy-deviation" in out


def test_entropy_deviation_sits_alongside_its_siblings(capsys):
    """entropy_kl, entropy_zscore and entropy_deviation are the three
    byte-entropy seed arms; if all three are active they should all three
    show up in the same Seed selection line.
    """
    fake = _make_fake_fuzzer(
        _entropy_kl=object(),
        _entropy_zscore=object(),
        _entropy_deviation=EntropyDeviationSeedStrategy(rng=None),
    )
    Fuzzer._print_enabled_features(fake)
    out = capsys.readouterr().out
    assert "entropy-kl" in out
    assert "entropy-zscore" in out
    assert "entropy-deviation" in out
