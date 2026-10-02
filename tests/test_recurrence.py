"""Recurrence quantification (core/recurrence.py) and its monitor/gate wiring."""

import numpy as np
import pytest

from fuzzer_tool.core.analyzers.analyzer_recurrence import (
    Novelty,
    Recurrence,
    RecurrenceMonitor,
)
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.recurrence import rqa


def _rqa_loop(s, embed, l_min):
    """Pair-by-pair reference: no numpy, no diagonal tricks."""
    n = len(s) - embed + 1
    if n < 2:
        return 0.0, 0.0, 0

    def rec(i, j):
        return all(s[i + k] == s[j + k] for k in range(embed))

    points = 0
    in_lines = 0
    best_rate, best_lag = 0.0, 0
    for lag in range(1, n):
        diag = [rec(i, i + lag) for i in range(n - lag)]
        points += sum(diag)
        run = 0
        for v in diag + [False]:
            if v:
                run += 1
                continue
            if run >= l_min:
                in_lines += run
            run = 0
        rate = sum(diag) / len(diag)
        if lag <= n // 2 and rate > best_rate:
            best_rate, best_lag = rate, lag

    pairs = n * (n - 1) // 2
    det = in_lines / points if points else 0.0
    return points / pairs, det, best_lag


def _random(n, k, seed):
    rng = RandPool(seed=seed)
    return [rng.randint(0, k - 1) for _ in range(n)]


# --- rqa -------------------------------------------------------------------


class TestRqaLiterals:
    def test_period_two(self):
        # Lag 2: 4 pairs, lag 4: 2 pairs -> 6 of C(6,2)=15; all in lines >= 2.
        r = rqa(np.array([1, 2, 1, 2, 1, 2]), embed=1, l_min=2)
        assert r.rr == pytest.approx(6 / 15)
        assert r.det == pytest.approx(1.0)
        assert r.period == 2

    def test_embedding_shrinks_pairs(self):
        # 2-grams (1,2)(2,1)(1,2)(2,1)(1,2): lag 2 -> 3, lag 4 -> 1, of 10.
        r = rqa(np.array([1, 2, 1, 2, 1, 2]), embed=2, l_min=1)
        assert r.rr == pytest.approx(4 / 10)

    def test_single_recurrence_is_not_a_line(self):
        r = rqa(np.array([1, 2, 3, 1, 4, 5]), embed=1, l_min=2)
        assert r.rr == pytest.approx(1 / 15)
        assert r.det == 0.0

    def test_constant_is_period_one(self):
        # 38 m-grams; diagonal at lag L has 38 - L points, a line iff >= 8.
        r = rqa(np.full(40, 9), embed=3, l_min=8)
        in_lines = sum(range(8, 38))
        assert (r.rr, r.det, r.period) == (1.0, pytest.approx(in_lines / (38 * 37 / 2)), 1)


class TestRqaMatchesReference:
    def test_control_reference_against_itself(self):
        """Rule 46: the oracle must agree with a second run of itself."""
        s = _random(120, 3, seed=11)
        assert _rqa_loop(s, 3, 4) == _rqa_loop(list(s), 3, 4)

    @pytest.mark.parametrize(("k", "embed", "l_min"), [(2, 1, 2), (3, 3, 4), (5, 2, 8), (40, 3, 8)])
    def test_random_sequences(self, k, embed, l_min):
        s = _random(150, k, seed=k)
        rr, det, period = _rqa_loop(s, embed, l_min)
        r = rqa(np.array(s), embed=embed, l_min=l_min)
        assert r.rr == pytest.approx(rr)
        assert r.det == pytest.approx(det)
        assert r.period == period

    def test_noisy_cycle(self):
        s = [i % 7 for i in range(150)]
        s[40] = s[90] = 99
        rr, det, period = _rqa_loop(s, 3, 8)
        r = rqa(np.array(s), embed=3, l_min=8)
        assert (r.rr, r.det, r.period) == (pytest.approx(rr), pytest.approx(det), period)


class TestRqaAdversarial:
    @pytest.mark.parametrize("n", [0, 1, 3])
    def test_too_short_is_empty(self, n):
        r = rqa(np.arange(n), embed=3, l_min=8)
        assert (r.rr, r.det, r.period) == (0.0, 0.0, 0)

    def test_no_recurrence_has_zero_det(self):
        r = rqa(np.arange(100), embed=1, l_min=2)
        assert (r.rr, r.det) == (0.0, 0.0)

    def test_negative_and_huge_symbols(self):
        s = [-(2**63), 2**63 - 1] * 20
        assert rqa(np.array(s, dtype=np.int64), embed=2, l_min=4).period == 2


# --- RecurrenceMonitor -----------------------------------------------------

_W = 64


def _feed(mon, symbols, novel_at=()):
    for i, s in enumerate(symbols):
        mon.push(s, Novelty.NEW if i in novel_at else Novelty.NONE)


class TestMonitor:
    def test_unknown_until_window_full(self):
        mon = RecurrenceMonitor(window=_W)
        _feed(mon, [1] * (_W - 1))
        assert mon.verdict is Recurrence.UNKNOWN

    def test_limit_cycle_is_trapped(self):
        mon = RecurrenceMonitor(window=_W)
        _feed(mon, [i % 7 for i in range(_W)])
        assert mon.verdict is Recurrence.TRAPPED
        assert mon.reading().period == 7

    def test_constant_stream_is_trapped(self):
        mon = RecurrenceMonitor(window=_W)
        _feed(mon, [5] * _W)
        assert mon.verdict is Recurrence.TRAPPED

    def test_novelty_in_window_is_free(self):
        """Falsification: a cycle that still finds coverage is not trapped."""
        mon = RecurrenceMonitor(window=_W)
        _feed(mon, [i % 7 for i in range(_W)], novel_at={_W // 2})
        assert mon.verdict is Recurrence.FREE

    def test_stochastic_plateau_is_free(self):
        """Adversarial: a small alphabet recurs a lot but forms no long lines."""
        mon = RecurrenceMonitor(window=256)
        _feed(mon, _random(256, 3, seed=5))
        assert mon.verdict is Recurrence.FREE
        assert mon.reading().rr > 0.02

    def test_novelty_clears_trapped_at_once(self):
        mon = RecurrenceMonitor(window=_W)
        _feed(mon, [5] * _W)
        mon.push(5, Novelty.NEW)
        assert mon.verdict is Recurrence.FREE

    def test_ring_is_bounded(self):
        mon = RecurrenceMonitor(window=_W)
        _feed(mon, range(10 * _W))
        assert len(mon._ring) == _W

    def test_uses_most_recent_window(self):
        """Ring order: a cycle after a random prefix must still be seen."""
        mon = RecurrenceMonitor(window=_W)
        _feed(mon, _random(_W + 17, 50, seed=2))
        _feed(mon, [i % 5 for i in range(_W - 17)])
        assert mon.verdict is Recurrence.TRAPPED

    @pytest.mark.parametrize(
        "kw", [{"window": 4}, {"embed": 0}, {"l_min": 0}, {"det_trapped": 1.5}]
    )
    def test_rejects_bad_args(self, kw):
        with pytest.raises(ValueError):
            RecurrenceMonitor(**kw)

    def test_same_input_same_reading(self):
        """Rule 46 control: two monitors fed identically agree."""
        a, b = RecurrenceMonitor(window=_W), RecurrenceMonitor(window=_W)
        s = _random(3 * _W, 4, seed=9)
        _feed(a, s)
        _feed(b, s)
        assert a.reading() == b.reading()
        assert a.verdict is b.verdict


# --- Saturation gate -------------------------------------------------------


class TestSaturationGate:
    def _picker(self, verdict):
        import types

        from fuzzer_tool.core.edge_tracker import EdgeTracker
        from fuzzer_tool.services.seed_picker import SeedPicker

        f = types.SimpleNamespace(
            _edge_tracker=EdgeTracker(),
            exec_count=0,
            _last_new_edge_exec=0,
            _cached_weights={},
            _recurrence=types.SimpleNamespace(verdict=verdict),
        )
        f._edge_tracker.good_turing_estimate = lambda: {"saturation": 1.0}
        return SeedPicker(f)

    def test_free_keeps_gate(self):
        assert self._picker(Recurrence.FREE)._saturation_gate() is True

    def test_trapped_forces_gate_off(self):
        assert self._picker(Recurrence.TRAPPED)._saturation_gate() is False


# --- Wiring ----------------------------------------------------------------


class TestWiring:
    def test_flag_off_by_default(self):
        import inspect

        from fuzzer_tool.services.fuzzer import Fuzzer

        assert inspect.signature(Fuzzer.__init__).parameters["recurrence"].default is False

    def test_cli_dest_and_hail_mary(self):
        import ast
        import inspect

        from fuzzer_tool.cli import commands
        from tests.test_regression_cli_fuzzer_kwargs import _fuzz_parser_dests

        assert "recurrence" in _fuzz_parser_dests(ast.parse(inspect.getsource(commands)))
        assert "recurrence" in commands._HAIL_MARY_FLAGS

    def test_registry_activates_monitor(self):
        import types

        from fuzzer_tool.core.analyzer_registry import REGISTRY

        f = types.SimpleNamespace(_use_recurrence=True)
        spec = REGISTRY._specs["recurrence"]
        assert spec.available(f)
        spec.activate(f)
        assert isinstance(f._recurrence, RecurrenceMonitor)

    def test_stats_and_report_fragments(self):
        import types

        from fuzzer_tool.services.report import _recurrence_lines
        from fuzzer_tool.services.stats import _recurrence_str

        mon = RecurrenceMonitor(window=_W)
        _feed(mon, [i % 7 for i in range(_W)])
        f = types.SimpleNamespace(_recurrence=mon)
        assert "rqa: trapped p=7" in _recurrence_str(f)
        assert any("period 7" in line for line in _recurrence_lines(f))

    def test_fragments_empty_when_off(self):
        from unittest.mock import MagicMock

        from fuzzer_tool.services.report import _recurrence_lines
        from fuzzer_tool.services.stats import _recurrence_str

        assert _recurrence_str(MagicMock()) == ""
        assert _recurrence_lines(MagicMock()) == []


class TestFuzzRoundPush:
    def _round(self, path, novel):
        import types

        from fuzzer_tool.services.fuzz_round import FuzzRound

        r = FuzzRound.__new__(FuzzRound)
        r._f = types.SimpleNamespace(_recurrence=RecurrenceMonitor(window=_W))
        r._scanned_shm = types.SimpleNamespace(read_path_hash=lambda: path)
        r._has_new_coverage = novel
        r._data = b"seed"
        return r

    def test_pushes_seed_path_symbol(self):
        r = self._round(path=1234, novel=False)
        r._push_recurrence()
        assert r._f._recurrence._ring[0] == hash((b"seed", 1234))

    def test_find_is_novelty(self):
        r = self._round(path=1, novel=True)
        mon = r._f._recurrence
        _feed(mon, [5] * _W)
        r._push_recurrence()
        assert mon.verdict is Recurrence.FREE

    def test_off_is_noop(self):
        import types

        r = self._round(path=1, novel=False)
        r._f = types.SimpleNamespace()
        r._push_recurrence()
