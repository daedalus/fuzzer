"""Regression test: `op_kuramoto` must show up in the enabled-features
banner (`Fuzzer._print_enabled_features`) like every other scheduler.

Context: `op_kuramoto`'s own module docstring (core/schedulers/op_kuramoto.py)
originally drew a deliberate parity with `op_katz`/`op_tang`/
`WhittleIndexScheduler`: "off by default, Elo-only, absent from
`_FALLBACK_PRECEDENCE`" -- and `_print_enabled_features` folded it into that
same "Elo-only, banner-silent" group alongside gradient/corral/whittle (see
the comment block right above the check this test exercises). That parity
was intentional for op_katz/op_tang/gradient/corral/whittle, but per an
explicit request `op_kuramoto` is the one exception: it is now shown in the
banner like every other scheduler, even though it remains off by default /
Elo-only / absent from `_FALLBACK_PRECEDENCE` in every other respect.
"""

from __future__ import annotations

from types import SimpleNamespace

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.op_kuramoto import OpKuramotoScheduler
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


def test_op_kuramoto_absent_by_default(capsys):
    fake = _make_fake_fuzzer()
    Fuzzer._print_enabled_features(fake)
    out = capsys.readouterr().out
    assert "op-kuramoto" not in out


def test_op_kuramoto_appears_when_active(capsys):
    fake = _make_fake_fuzzer(_op_kuramoto=OpKuramotoScheduler(rng=RandPool(seed=4)))
    Fuzzer._print_enabled_features(fake)
    out = capsys.readouterr().out
    assert "Scheduling:" in out
    assert "op-kuramoto" in out


def test_op_katz_and_op_tang_remain_banner_silent(capsys):
    """op_kuramoto is the deliberate exception -- op_katz/op_tang/gradient/
    corral/whittle keep the original Elo-only, banner-silent treatment.
    This only checks the seed-selection attributes that the banner
    would otherwise report under different names (`_katz_channel`/`_tang`
    are the seed-selection arms, distinct from the operator-gate versions
    that stay silent), so it just guards that adding op_kuramoto's line
    didn't accidentally also start printing something for the others.
    """
    fake = _make_fake_fuzzer(_op_kuramoto=OpKuramotoScheduler(rng=RandPool(seed=4)))
    Fuzzer._print_enabled_features(fake)
    out = capsys.readouterr().out
    assert "op-katz" not in out
    assert "op-tang" not in out
    assert "gradient" not in out
    assert "corral" not in out
    assert "whittle" not in out
