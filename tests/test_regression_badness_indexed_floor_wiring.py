"""Regression test: `Fuzzer._op_katz`'s `badness_fn` is wired to
`Fuzzer._current_scheduling_badness`, and that method correctly reflects
`f._regime.regime` (`core/percolation.py`'s `CoverageRegime`).

See docs/action_plan_compositional_stability.md (P1) and
docs/handover/handover_badness_indexed_floor_2026-09-21.md.
"""

import tempfile
from pathlib import Path

from fuzzer_tool.core.percolation import CoverageRegime

_TARGET = str(Path(__file__).resolve().parent.parent / "targets" / "test_target")


def _build_fuzzer(**kwargs):
    from fuzzer_tool.services.fuzzer import Fuzzer

    tmp = tempfile.TemporaryDirectory()
    corpus = Path(tmp.name) / "corpus"
    crashes = Path(tmp.name) / "crashes"
    corpus.mkdir()
    crashes.mkdir()
    kwargs.setdefault("max_len", 4096)
    f = Fuzzer(
        target=_TARGET,
        corpus_dir=str(corpus),
        crashes_dir=str(crashes),
        **kwargs,
    )
    f._test_tmp = tmp  # keep the tempdir alive for the caller's lifetime
    return f


class TestBadnessFnWiring:
    def test_op_katz_off_by_default(self):
        f = _build_fuzzer()
        assert f._op_katz is None

    def test_op_katz_badness_fn_is_the_fuzzer_method(self):
        f = _build_fuzzer(op_katz=True)
        # Bound-method access creates a fresh wrapper object each time, so
        # `is` would spuriously fail here -- compare identity of the
        # underlying function and the bound instance instead (what `==`
        # on bound methods already checks).
        assert f._op_katz.badness_fn == f._current_scheduling_badness

    def test_scheduling_badness_reflects_subcritical_regime(self):
        f = _build_fuzzer(op_katz=True)
        f._regime._regime = CoverageRegime.SUBCRITICAL
        assert f._current_scheduling_badness() == 1.0

    def test_scheduling_badness_reflects_supercritical_regime(self):
        f = _build_fuzzer(op_katz=True)
        f._regime._regime = CoverageRegime.SUPERCRITICAL
        assert f._current_scheduling_badness() == 0.0

    def test_scheduling_badness_reflects_critical_regime(self):
        f = _build_fuzzer(op_katz=True)
        f._regime._regime = CoverageRegime.CRITICAL
        assert f._current_scheduling_badness() == 0.5

    def test_scheduling_badness_defensive_when_regime_missing(self):
        f = _build_fuzzer(op_katz=True)
        del f._regime
        assert f._current_scheduling_badness() == 0.5

    def test_op_katz_select_op_uses_live_badness(self):
        """End-to-end: select_op's actual floor changes when the fuzzer's
        own regime detector flips from SUPERCRITICAL to SUBCRITICAL.
        """
        f = _build_fuzzer(op_katz=True)
        n = 10
        ops = [f"op{i}" for i in range(n)]
        f._op_katz.successes = {"op0": 1.0}
        f._op_katz.attempts = dict.fromkeys(ops, 10.0)

        f._regime._regime = CoverageRegime.SUPERCRITICAL
        low_floor_probs = f._op_katz._select_probs(ops)

        f._regime._regime = CoverageRegime.SUBCRITICAL
        high_floor_probs = f._op_katz._select_probs(ops)

        # Every never-attempted arm's floor-bound probability should be
        # at least as large under SUBCRITICAL (badness=1, higher floor)
        # as under SUPERCRITICAL (badness=0, static default floor).
        for i in range(1, n):
            assert high_floor_probs[i] >= low_floor_probs[i] - 1e-9
