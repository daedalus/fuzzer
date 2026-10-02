"""PositionFinchScheduler: quantitative hot-byte proposals (Finch).

Covers core/schedulers/pos_finch.py and the engine side that keeps the
per-byte trace-delta magnitudes the byteflip pass measures
(``OperatorEngine.effector_heat``). The effector arm only knows LIVE/INERT;
this arm weights a byte by how many edges its flip moved, and adapts the
weight from round outcomes.
"""

from array import array

import pytest

from fuzzer_tool.core.schedulers.pos_base import Outcome, PositionScheduler
from fuzzer_tool.core.schedulers.pos_finch import (
    EPSILON,
    GAIN_BONUS,
    HEAT_CAP,
    MAX_BONUS_OFFSETS,
    MAX_SEEDS,
    PositionFinchScheduler,
)

SEED = bytes(range(64))
OFFS = array("I", [3, 17, 40, 63])
MAGS = array("H", [1, 9, 0, 5])  # byte 40 moved nothing -> never proposed
NO_ESCAPE = 0.99  # random() draw above EPSILON
GRID = 1000


class ScriptedRng:
    """Scripted random(): first draw is the escape, second the CDF point."""

    def __init__(self, randoms=()):
        self._randoms = list(randoms)

    def random(self):
        return self._randoms.pop(0) if self._randoms else NO_ESCAPE


def _sched(rng=None, heat=(OFFS, MAGS), ready=True):
    return PositionFinchScheduler(
        rng or ScriptedRng(),
        heat_of=lambda d: heat,
        ready=lambda: ready,
        key_of=lambda d: bytes(d[:4]).hex(),
    )


def _pick(sched, u, buf_len=len(SEED), data=SEED):
    """Propose with a scripted CDF point *u* in [0, 1)."""
    sched._rng = ScriptedRng([NO_ESCAPE, u])
    return sched.propose(data, buf_len)


class TestProtocol:
    def test_satisfies_the_protocol(self):
        assert isinstance(_sched(), PositionScheduler)

    def test_active_follows_ready(self):
        assert _sched(ready=True).active()
        assert not _sched(ready=False).active()


class TestDeclines:
    def test_no_map(self):
        assert _sched(heat=None).propose(SEED, len(SEED)) is None

    def test_empty_map(self):
        assert _sched(heat=(array("I"), array("H"))).propose(SEED, len(SEED)) is None

    def test_all_inert_map(self):
        heat = (array("I", [1, 2]), array("H", [0, 0]))
        assert _sched(heat=heat).propose(SEED, len(SEED)) is None

    def test_empty_data_or_buffer(self):
        assert _sched().propose(b"", 10) is None
        assert _sched().propose(SEED, 0) is None

    def test_epsilon_escape(self):
        assert _sched(ScriptedRng([EPSILON / 2])).propose(SEED, len(SEED)) is None

    def test_every_hot_byte_past_a_shrunk_buffer(self):
        heat = (array("I", [50, 60]), array("H", [3, 3]))
        assert _sched(heat=heat).propose(SEED, 50) is None


class TestQuantitativePicks:
    def test_cdf_points_map_to_bytes_by_weight(self):
        # weights 1, 9, 5 over bytes 3, 17, 63: total 15.
        s = _sched()
        assert _pick(s, 0.0) == 3
        assert _pick(s, 1.5 / 15) == 17
        assert _pick(s, 9.5 / 15) == 17
        assert _pick(s, 10.5 / 15) == 63
        assert _pick(s, 0.999) == 63

    def test_share_is_proportional_to_heat(self):
        s = _sched()
        picks = [_pick(s, (i + 0.5) / GRID) for i in range(GRID)]
        total = sum(MAGS)
        for off, mag in zip(OFFS, MAGS, strict=True):
            assert picks.count(off) / GRID == pytest.approx(mag / total, abs=0.01)

    def test_falsification_never_lands_on_a_zero_heat_byte(self):
        s = _sched()
        picks = {_pick(s, (i + 0.5) / GRID) for i in range(GRID)}
        assert 40 not in picks
        assert picks == {3, 17, 63}

    def test_falsification_not_uniform_over_hot_bytes(self):
        # A uniform-over-live arm (effector) would give 1/3 each.
        s = _sched()
        picks = [_pick(s, (i + 0.5) / GRID) for i in range(GRID)]
        assert picks.count(17) > 2 * picks.count(3)

    def test_shrunk_buffer_renormalizes(self):
        # buf_len 18 drops 40 and 63; weights 1 and 9 remain.
        s = _sched()
        assert _pick(s, 0.05, buf_len=18) == 3
        assert _pick(s, 0.5, buf_len=18) == 17


class TestHeatCap:
    def test_adversarial_one_huge_byte_cannot_starve_the_rest(self):
        heat = (array("I", [1, 2]), array("H", [65535, HEAT_CAP]))
        s = _sched(heat=heat)
        picks = [_pick(s, (i + 0.5) / GRID) for i in range(GRID)]
        # Both capped to HEAT_CAP: an even split.
        assert picks.count(1) / GRID == pytest.approx(0.5, abs=0.01)

    def test_cap_leaves_small_values_alone(self):
        heat = (array("I", [1, 2]), array("H", [1, 3]))
        s = _sched(heat=heat)
        picks = [_pick(s, (i + 0.5) / GRID) for i in range(GRID)]
        assert picks.count(2) / GRID == pytest.approx(0.75, abs=0.01)


class TestAdaptiveBonus:
    def test_gain_heats_the_offsets_that_produced_it(self):
        heat = (array("I", [3, 17]), array("H", [1, 1]))
        s = _sched(heat=heat)
        s.record(SEED, [3], Outcome.GAIN, 1.0)
        picks = [_pick(s, (i + 0.5) / GRID) for i in range(GRID)]
        want = (1 + GAIN_BONUS) / (2 + GAIN_BONUS)
        assert picks.count(3) / GRID == pytest.approx(want, abs=0.01)

    def test_gain_makes_a_cold_byte_proposable(self):
        s = _sched(heat=(array("I", [3]), array("H", [1])))
        s.record(SEED, [50], Outcome.GAIN, 1.0)
        picks = {_pick(s, (i + 0.5) / GRID) for i in range(GRID)}
        assert 50 in picks

    def test_a_gain_without_a_static_map_still_proposes(self):
        s = _sched(heat=None)
        s.record(SEED, [9], Outcome.GAIN, 1.0)
        assert _pick(s, 0.3) == 9

    def test_miss_decays_the_bonus(self):
        heat = (array("I", [3, 17]), array("H", [1, 1]))
        s = _sched(heat=heat)
        s.record(SEED, [3], Outcome.GAIN, 1.0)
        s.record(SEED, [3], Outcome.MISS, 1.0)
        picks = [_pick(s, (i + 0.5) / GRID) for i in range(GRID)]
        assert picks.count(3) / GRID < 0.5 + 0.01 + 0.0
        assert picks.count(3) / GRID > 0.5 - 0.01

    def test_falsification_miss_alone_adds_nothing(self):
        heat = (array("I", [3, 17]), array("H", [1, 1]))
        s = _sched(heat=heat)
        s.record(SEED, [3], Outcome.MISS, 1.0)
        picks = [_pick(s, (i + 0.5) / GRID) for i in range(GRID)]
        assert picks.count(3) / GRID == pytest.approx(0.5, abs=0.01)

    def test_bonus_is_per_seed(self):
        s = _sched(heat=None)
        other = b"\xff" * 64
        s.record(SEED, [9], Outcome.GAIN, 1.0)
        assert _pick(s, 0.3, data=other) is None

    def test_repeated_gain_saturates(self):
        s = _sched(heat=(array("I", [3]), array("H", [1])))
        for _ in range(10 * HEAT_CAP):
            s.record(SEED, [50], Outcome.GAIN, 1.0)
        picks = [_pick(s, (i + 0.5) / GRID) for i in range(GRID)]
        # Bonus capped at HEAT_CAP against heat 1.
        assert picks.count(50) / GRID == pytest.approx(HEAT_CAP / (HEAT_CAP + 1), abs=0.01)

    def test_adversarial_bad_offsets_are_ignored(self):
        s = _sched(heat=None)
        s.record(SEED, [-1, len(SEED), 10**9], Outcome.GAIN, 1.0)
        assert s.propose(SEED, len(SEED)) is None

    def test_adversarial_empty_data_record(self):
        s = _sched(heat=None)
        s.record(b"", [0], Outcome.GAIN, 1.0)
        assert s.propose(b"", 4) is None


class TestBounds:
    def test_bonus_offsets_per_seed_are_bounded(self):
        s = _sched(heat=None)
        n = len(SEED)
        big = bytes(n)
        s.record(big, list(range(n)), Outcome.GAIN, 1.0)
        assert s.bonus_size(big) <= MAX_BONUS_OFFSETS

    def test_bonus_seeds_are_bounded(self):
        s = _sched(heat=None)
        for i in range(MAX_SEEDS + 5):
            s.record(i.to_bytes(4, "big") + bytes(60), [1], Outcome.GAIN, 1.0)
        assert s.seeds_tracked() == MAX_SEEDS


class TestEngineKeepsTheHeat:
    """Magnitudes outlive the deterministic queue so the arm can read them."""

    @staticmethod
    def _fuzzer(tmp_path, seed):
        from fuzzer_tool.services.fuzzer import Fuzzer

        corpus = tmp_path / "corpus"
        (corpus / "seeds").mkdir(parents=True)
        (tmp_path / "crashes").mkdir()
        (corpus / "seeds" / "seed1").write_bytes(seed)
        f = Fuzzer(
            "/bin/true",
            corpus_dir=str(corpus),
            crashes_dir=str(tmp_path / "crashes"),
            max_len=4096,
            deterministic=True,
        )
        key = f._seed_key(seed)
        f._favored = {key}
        f._edge_tracker.seed_edges[key] = {1, 2, 3}
        return f

    @staticmethod
    def _drain(f, data, mags):
        eng = f._operators
        while eng.maybe_deterministic_mutation(data) is not None:
            pending = eng._det_pending
            if pending is None:
                continue
            m = mags.get(pending[1], 0)
            eng.note_deterministic_result(m > 0, 7 if m else 0, m)

    def test_magnitudes_survive_the_drain(self, tmp_path):
        data = bytes((0x41 + i) & 0xFF for i in range(16))
        f = self._fuzzer(tmp_path, data)
        assert f._operators.effector_heat(data) is None
        self._drain(f, data, {2: 4, 11: 1, 15: 70000})
        offs, mags = f._operators.effector_heat(data)
        assert list(offs) == [2, 11, 15]
        assert list(mags) == [4, 1, 65535]  # saturates at the uint16 ceiling

    def test_live_map_is_unchanged_by_magnitudes(self, tmp_path):
        data = bytes((0x41 + i) & 0xFF for i in range(16))
        f = self._fuzzer(tmp_path, data)
        self._drain(f, data, {2: 4, 11: 1})
        assert list(f._operators.effector_live(data)) == [2, 11]

    def test_adversarial_all_inert_is_empty_not_none(self, tmp_path):
        data = bytes((0x41 + i) & 0xFF for i in range(8))
        f = self._fuzzer(tmp_path, data)
        self._drain(f, data, {})
        offs, mags = f._operators.effector_heat(data)
        assert len(offs) == 0 and len(mags) == 0

    def test_legacy_caller_without_magnitude_counts_one(self, tmp_path):
        data = bytes((0x41 + i) & 0xFF for i in range(8))
        f = self._fuzzer(tmp_path, data)
        eng = f._operators
        while eng.maybe_deterministic_mutation(data) is not None:
            pending = eng._det_pending
            if pending is not None:
                eng.note_deterministic_result(pending[1] == 3, 7)
        offs, mags = eng.effector_heat(data)
        assert list(offs) == [3] and list(mags) == [1]

    def test_retention_is_bounded(self, tmp_path):
        from fuzzer_tool.services.operators import MAX_EFF_SEEDS

        data = bytes((0x41 + i) & 0xFF for i in range(4))
        f = self._fuzzer(tmp_path, data)
        eng = f._operators
        for i in range(MAX_EFF_SEEDS + 5):
            eng._keep_effector(f"k{i}", bytearray(4), array("H", [1, 0, 0, 0]))
        assert len(eng._det_heat) == MAX_EFF_SEEDS
        assert "k0" not in eng._det_heat


@pytest.mark.parametrize("n", [1, 7])
def test_regression_propose_stays_in_buffer(n):
    heat = (array("I", range(64)), array("H", [1] * 64))
    s = _sched(heat=heat)
    off = _pick(s, 0.99, buf_len=n)
    assert 0 <= off < n
