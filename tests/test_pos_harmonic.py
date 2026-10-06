"""PositionHarmonicScheduler (core/schedulers/pos_harmonic.py).

Draw order in ``propose`` that the scripted RNG relies on: spark check
(random), phase draw (random), record index (randint).
"""

import numpy as np
import pytest

from fuzzer_tool.core.schedulers.pos_base import Outcome, PositionScheduler
from fuzzer_tool.core.schedulers.pos_harmonic import (
    FLOOR,
    HARMONICS,
    MIN_GAINS,
    PositionHarmonicScheduler,
    phase_density,
)

L = 16
REC = bytes(range(L)) * 64  # strongly periodic, stride 16, 1024 bytes
NO_SPARK = 0.99


class ScriptedRng:
    def __init__(self, randoms=()):
        self._randoms = list(randoms)

    def random(self):
        return self._randoms.pop(0) if self._randoms else NO_SPARK

    def randint(self, a, b):
        return a


def _trained(field_offset=5, rounds=MIN_GAINS):
    s = PositionHarmonicScheduler(ScriptedRng())
    for k in range(rounds):
        s.record(REC, [k * L + field_offset], Outcome.GAIN)
    return s


def test_protocol_and_stride():
    s = PositionHarmonicScheduler(ScriptedRng())
    assert isinstance(s, PositionScheduler)
    s.record(REC, [5], Outcome.GAIN)
    assert s.stride(REC) == L


def test_declines_until_enough_gains_or_without_stride():
    s = _trained(rounds=MIN_GAINS - 1)
    assert s.propose(REC, len(REC)) is None
    flat = bytes(1024)  # no periodicity -> no stride
    s2 = PositionHarmonicScheduler(ScriptedRng())
    for _ in range(MIN_GAINS):
        s2.record(flat, [7], Outcome.GAIN)
    assert s2.propose(flat, len(flat)) is None
    assert s2.propose(REC, 0) is None


def test_density_peaks_at_the_productive_phase():
    s = _trained(field_offset=5, rounds=10)
    st = s._seeds[s._key(REC)]
    p = phase_density(st.coeffs, st.total, L)
    assert p.argmax() == 5
    assert p.sum() == pytest.approx(1.0)
    # Every raw bin is in [FLOOR, 1 + 2K + FLOOR], so no phase can be starved.
    assert p.min() >= FLOOR / (L * (1.0 + 2 * HARMONICS + FLOOR))


def test_two_fields_both_get_mass():
    s = PositionHarmonicScheduler(ScriptedRng())
    for k in range(8):
        s.record(REC, [k * L + 3], Outcome.GAIN)
        s.record(REC, [k * L + 11], Outcome.GAIN)
    st = s._seeds[s._key(REC)]
    p = phase_density(st.coeffs, st.total, L)
    top2 = set(np.argsort(p)[-2:].tolist())
    assert top2 == {3, 11}


def test_propose_follows_the_density_deterministically():
    s = _trained(field_offset=5, rounds=10)
    s._rng = ScriptedRng([NO_SPARK, 0.0])  # cdf draw 0.0 -> first non-zero bin
    pos = s.propose(REC, len(REC))
    assert pos is not None and pos % L == 0  # u=0 lands on phase 0, record 0 (randint->a)
    st = s._seeds[s._key(REC)]
    cdf = np.cumsum(phase_density(st.coeffs, st.total, L))
    u = float(cdf[4]) + 1e-9  # just past bin 4 -> bin 5
    s._rng = ScriptedRng([NO_SPARK, u])
    assert s.propose(REC, len(REC)) == 5


def test_proposals_stay_in_bounds_on_short_buffers():
    s = _trained()
    for buf_len in (1, 3, 17, 100):
        s._rng = ScriptedRng([NO_SPARK, 0.999999])
        pos = s.propose(REC, buf_len)
        assert 0 <= pos < buf_len


def test_miss_and_empty_rounds_do_not_create_state():
    s = PositionHarmonicScheduler(ScriptedRng())
    s.record(REC, [5], Outcome.MISS)
    s.record(REC, [], Outcome.GAIN)
    s.record(REC, [-1], Outcome.GAIN)
    assert s.seed_count() == 0


def test_state_round_trips_and_behaves_the_same():
    s = _trained(field_offset=5, rounds=6)
    blob = s.to_dict()
    t = PositionHarmonicScheduler(ScriptedRng())
    t.from_dict(blob)
    assert t.to_dict() == blob
    assert t.stride(REC) == L
    s._rng = ScriptedRng([NO_SPARK, 0.3])
    t._rng = ScriptedRng([NO_SPARK, 0.3])
    assert s.propose(REC, len(REC)) == t.propose(REC, len(REC))


@pytest.mark.parametrize(
    "bad",
    [
        {"version": 99, "seeds": {}},
        {"version": 1, "seeds": {"1": {"stride": 1, "coeffs": [[0, 0]] * 6, "total": 1, "gains": 1}}},
        {"version": 1, "seeds": {"1": {"stride": 16, "coeffs": [[0, 0]] * 5, "total": 1, "gains": 1}}},
        {"version": 1, "seeds": {"1": {"stride": 16, "coeffs": [[float("nan"), 0]] * 6, "total": 1, "gains": 1}}},
        {"version": 1, "seeds": {"1": {"stride": 16, "coeffs": [[0, 0]] * 6, "total": -1, "gains": 1}}},
        {"version": 1, "seeds": [1]},
        "junk",
    ],
)
def test_malformed_state_clears_instead_of_raising(bad):
    s = _trained()
    s.from_dict(bad)
    assert s.seed_count() == 0


def test_empty_state_clears():
    s = _trained()
    s.from_dict({})
    assert s.seed_count() == 0
