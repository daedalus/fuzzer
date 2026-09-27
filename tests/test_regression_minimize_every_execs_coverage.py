"""Regression: --minimize-every-execs must fire on every fuzz_one() exit path.

Two stacked bugs let a live --hail-mary run (or any run combining
--metropolis/--anneal-budget with --minimize-every-execs, or simply hitting
crashes) go long stretches, or an entire campaign, without ever minimizing:

1. auto_minimize_corpus() unconditionally returned early whenever --ga or
   --qea was set, even though GA feeds f.corpus/f.seed_meta identically to
   the default path and QEA does too once --elo lifts its bypass. Covered
   separately in tests/test_corpus_minimization.py.

2. The periodic "(exec_count - baseline) % minimize_every_execs == 0" check
   was duplicated inline at only two of fuzz_one()'s four exit points (the
   has_new_coverage branch and the final "boring" branch). The crash-return
   and the Metropolis-acceptance-return paths skipped it entirely -- not
   just for that call, but permanently for that exact exec_count value,
   since the check is never re-evaluated once exec_count has moved past it.
   --hail-mary force-enables both --metropolis and a 10000-exec
   --anneal-budget, making the skip path common enough that minimization
   could go long stretches, or (on a target with a high crash rate)
   effectively the entire campaign, without firing.

This file covers (2): both at the `_maybe_periodic_minimize` unit level and
by driving the real Fuzzer.fuzz_one() through the crash and Metropolis exit
points end to end.
"""

from __future__ import annotations

import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.services.fuzzer import Fuzzer


def _mk_fuzzer(**kwargs):
    tmpdir = tempfile.mkdtemp(prefix="fuzz_minimize_coverage_")
    defaults = dict(
        target="/bin/true",
        corpus_dir=f"{tmpdir}/corpus",
        crashes_dir=f"{tmpdir}/crashes",
        max_len=256,
        timeout=1,
        mutations_per_input=2,
    )
    defaults.update(kwargs)
    with (
        patch("os.path.isfile", return_value=True),
        patch("os.access", return_value=True),
    ):
        f = Fuzzer(**defaults)
    # _maybe_periodic_minimize requires len(corpus) > 1 (see
    # _auto_minimize_corpus's own guard against a trivial corpus) -- the
    # single default seed from a fresh corpus_dir isn't enough on its own.
    f.corpus.append(b"SECOND_SEED_PADDING")
    return f


def _fake_self(**overrides):
    """Bare stand-in for `self` -- exercises _maybe_periodic_minimize as a
    plain function, without constructing a real Fuzzer."""
    ns = SimpleNamespace(
        minimize_every_execs=0,
        exec_count=0,
        _exec_baseline=0,
        corpus=[b"a", b"b"],
        _auto_minimize_corpus=MagicMock(),
        _deprioritize_near_duplicates=MagicMock(),
    )
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


class TestMaybePeriodicMinimizeUnit:
    """_maybe_periodic_minimize in isolation, independent of fuzz_one()."""

    def test_fires_on_exact_multiple(self):
        f = _fake_self(minimize_every_execs=500, exec_count=500)
        Fuzzer._maybe_periodic_minimize(f)
        f._auto_minimize_corpus.assert_called_once()

    def test_does_not_fire_off_multiple(self):
        f = _fake_self(minimize_every_execs=500, exec_count=501)
        Fuzzer._maybe_periodic_minimize(f)
        f._auto_minimize_corpus.assert_not_called()

    def test_disabled_when_zero(self):
        f = _fake_self(minimize_every_execs=0, exec_count=500)
        Fuzzer._maybe_periodic_minimize(f)
        f._auto_minimize_corpus.assert_not_called()

    def test_respects_exec_baseline_offset(self):
        f = _fake_self(minimize_every_execs=500, exec_count=1500, _exec_baseline=1000)
        Fuzzer._maybe_periodic_minimize(f)
        f._auto_minimize_corpus.assert_called_once()

    def test_dedup_only_when_requested(self):
        f = _fake_self(minimize_every_execs=500, exec_count=500)
        Fuzzer._maybe_periodic_minimize(f, dedup=True)
        f._auto_minimize_corpus.assert_called_once()
        f._deprioritize_near_duplicates.assert_called_once()

    def test_no_dedup_by_default(self):
        f = _fake_self(minimize_every_execs=500, exec_count=500)
        Fuzzer._maybe_periodic_minimize(f)
        f._deprioritize_near_duplicates.assert_not_called()

    def test_skips_when_corpus_too_small(self):
        f = _fake_self(minimize_every_execs=500, exec_count=500, corpus=[b"only_one"])
        Fuzzer._maybe_periodic_minimize(f)
        f._auto_minimize_corpus.assert_not_called()


class TestFuzzOneCrashPathMinimizes:
    """The crash-return exit of fuzz_one() must still hit the periodic check.

    Before the fix, `if is_crash: ... return True` returned before either
    inline periodic-minimize block, so a crashy target could hit exact
    multiples of --minimize-every-execs purely on crash iterations and
    never minimize on those counts.
    """

    def test_crash_iteration_at_exact_multiple_minimizes(self):
        f = _mk_fuzzer(minimize_every_execs=10)
        seed = f.corpus[0]
        f.exec_count = f._exec_baseline + 9  # this call makes it +10
        with (
            patch.object(f, "_dedup_mutate", return_value=b"CRASHY01"),
            patch.object(f, "_run_target", return_value=(139, "segv")),
            patch.object(f, "_is_crash", return_value=True),
            patch.object(f, "_is_interesting", return_value=False),
            patch.object(f, "_auto_minimize_corpus") as spy,
        ):
            assert f.fuzz_one(seed) is True
        spy.assert_called_once()

    def test_crash_iteration_off_multiple_does_not_minimize(self):
        f = _mk_fuzzer(minimize_every_execs=10)
        seed = f.corpus[0]
        f.exec_count = f._exec_baseline + 3  # this call makes it +4
        with (
            patch.object(f, "_dedup_mutate", return_value=b"CRASHY02"),
            patch.object(f, "_run_target", return_value=(139, "segv")),
            patch.object(f, "_is_crash", return_value=True),
            patch.object(f, "_is_interesting", return_value=False),
            patch.object(f, "_auto_minimize_corpus") as spy,
        ):
            assert f.fuzz_one(seed) is True
        spy.assert_not_called()


class TestFuzzOneMetropolisPathMinimizes:
    """The Metropolis-acceptance exit of fuzz_one() must still hit the
    periodic check -- the specific path --hail-mary makes common via its
    forced --metropolis + --anneal-budget 10000.
    """

    def test_metropolis_accept_at_exact_multiple_minimizes(self):
        f = _mk_fuzzer(minimize_every_execs=10, metropolis=True, anneal_budget=10000)
        seed = f.corpus[0]
        f.exec_count = f._exec_baseline + 9  # this call makes it +10
        with (
            patch.object(f, "_dedup_mutate", return_value=b"BORINGXX"),
            patch.object(f, "_run_target", return_value=(0, "")),
            patch.object(f, "_is_crash", return_value=False),
            patch.object(f, "_is_interesting", return_value=False),
            patch.object(f, "_metropolis_accept_p", return_value=1.0),
            patch.object(RandPool, "random", return_value=0.0),
            patch.object(f, "_auto_minimize_corpus") as spy,
        ):
            assert f.fuzz_one(seed) is True
        spy.assert_called_once()

    def test_metropolis_accept_off_multiple_does_not_minimize(self):
        f = _mk_fuzzer(minimize_every_execs=10, metropolis=True, anneal_budget=10000)
        seed = f.corpus[0]
        f.exec_count = f._exec_baseline + 3  # this call makes it +4
        with (
            patch.object(f, "_dedup_mutate", return_value=b"BORINGYY"),
            patch.object(f, "_run_target", return_value=(0, "")),
            patch.object(f, "_is_crash", return_value=False),
            patch.object(f, "_is_interesting", return_value=False),
            patch.object(f, "_metropolis_accept_p", return_value=1.0),
            patch.object(RandPool, "random", return_value=0.0),
            patch.object(f, "_auto_minimize_corpus") as spy,
        ):
            assert f.fuzz_one(seed) is True
        spy.assert_not_called()


class TestFuzzOneBoringPathStillMinimizes:
    """Control: the pre-existing "boring iteration" exit point (never
    affected by this bug) keeps working after the refactor to a shared
    helper.
    """

    def test_boring_iteration_at_exact_multiple_minimizes(self):
        f = _mk_fuzzer(minimize_every_execs=10)
        seed = f.corpus[0]
        f.exec_count = f._exec_baseline + 9
        with (
            patch.object(f, "_dedup_mutate", return_value=b"BORINGZZ"),
            patch.object(f, "_run_target", return_value=(0, "")),
            patch.object(f, "_is_crash", return_value=False),
            patch.object(f, "_is_interesting", return_value=False),
            patch.object(f, "_auto_minimize_corpus") as spy,
        ):
            assert f.fuzz_one(seed) is False
        spy.assert_called_once()
