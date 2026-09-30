"""PositionEffectorScheduler: land mutations on bytes the byteflip pass saw live.

Covers core/schedulers/pos_effector.py and the engine side that keeps the
effector map after the deterministic queue drains
(``OperatorEngine.effector_live``).
"""

from array import array

import pytest

from fuzzer_tool.core.schedulers.pos_base import Outcome, PositionScheduler
from fuzzer_tool.core.schedulers.pos_effector import EPSILON, PositionEffectorScheduler

SEED = bytes(range(64))
LIVE = array("I", [3, 17, 40, 63])
NO_ESCAPE = 0.99  # random() draw above EPSILON


class ScriptedRng:
    """Scripted random()/randint(); randint asserts its bounds."""

    def __init__(self, randoms=(), ints=()):
        self._randoms = list(randoms)
        self._ints = list(ints)
        self.bounds = []

    def random(self):
        return self._randoms.pop(0) if self._randoms else NO_ESCAPE

    def randint(self, a, b):
        self.bounds.append((a, b))
        v = self._ints.pop(0) if self._ints else a
        assert a <= v <= b, f"scripted randint {v} outside [{a}, {b}]"
        return v


def _sched(rng=None, live=LIVE, ready=True):
    return PositionEffectorScheduler(
        rng or ScriptedRng(), live_of=lambda d: live, ready=lambda: ready
    )


class TestProtocol:
    def test_satisfies_the_protocol(self):
        assert isinstance(_sched(), PositionScheduler)

    def test_record_is_a_noop(self):
        s = _sched(ScriptedRng(ints=[1]))
        s.record(SEED, [5], Outcome.GAIN, 1.0)
        assert s.propose(SEED, len(SEED)) == LIVE[1]

    def test_active_follows_ready(self):
        assert _sched(ready=True).active()
        assert not _sched(ready=False).active()


class TestDeclines:
    def test_no_map(self):
        assert _sched(live=None).propose(SEED, len(SEED)) is None

    def test_no_live_byte(self):
        assert _sched(live=array("I")).propose(SEED, len(SEED)) is None

    def test_empty_data_or_buffer(self):
        assert _sched().propose(b"", 10) is None
        assert _sched().propose(SEED, 0) is None

    def test_epsilon_escape(self):
        rng = ScriptedRng(randoms=[EPSILON / 2])
        assert _sched(rng).propose(SEED, len(SEED)) is None

    def test_every_live_byte_past_a_shrunk_buffer(self):
        assert _sched(live=array("I", [50, 60])).propose(SEED, 50) is None


class TestPicks:
    def test_scripted_index_picks_that_live_byte(self):
        for i, off in enumerate(LIVE):
            assert _sched(ScriptedRng(ints=[i])).propose(SEED, len(SEED)) == off

    def test_falsification_never_lands_on_an_inert_byte(self):
        # Every scripted index maps into LIVE: the arm is not uniform.
        picks = {_sched(ScriptedRng(ints=[i])).propose(SEED, len(SEED)) for i in range(len(LIVE))}
        assert picks == set(LIVE)

    def test_adversarial_shrunk_buffer_bounds_the_draw(self):
        # buf_len 41 keeps 3, 17, 40 and drops 63: the draw is over 3 entries.
        rng = ScriptedRng(ints=[2])
        assert _sched(rng).propose(SEED, 41) == 40
        assert rng.bounds == [(0, 2)]

    def test_adversarial_one_byte_buffer(self):
        assert _sched(live=array("I", [0, 9])).propose(SEED, 1) == 0


class TestEngineKeepsTheMap:
    """The map outlives the deterministic queue so the arm can read it."""

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
    def _drain(f, data, live):
        while f._operators.maybe_deterministic_mutation(data) is not None:
            pending = f._operators._det_pending
            if pending is not None:
                f._operators.note_deterministic_result(pending[1] in live)

    def test_live_offsets_survive_the_drain(self, tmp_path):
        data = bytes((0x41 + i) & 0xFF for i in range(16))
        f = self._fuzzer(tmp_path, data)
        assert f._operators.effector_live(data) is None
        live = {2, 11, 15}
        self._drain(f, data, live)
        assert list(f._operators.effector_live(data)) == sorted(live)

    def test_adversarial_all_inert_is_an_empty_map_not_none(self, tmp_path):
        data = bytes((0x41 + i) & 0xFF for i in range(8))
        f = self._fuzzer(tmp_path, data)
        self._drain(f, data, set())
        got = f._operators.effector_live(data)
        assert got is not None and len(got) == 0

    def test_retention_is_bounded(self, tmp_path):
        from fuzzer_tool.services.operators import MAX_EFF_SEEDS

        data = bytes((0x41 + i) & 0xFF for i in range(4))
        f = self._fuzzer(tmp_path, data)
        eng = f._operators
        for i in range(MAX_EFF_SEEDS + 5):
            eng._keep_effector(f"k{i}", bytearray(4))
        assert len(eng._det_live) == MAX_EFF_SEEDS
        assert "k0" not in eng._det_live


@pytest.mark.parametrize("n", [1, 7])
def test_regression_propose_stays_in_buffer(n):
    off = _sched(live=array("I", range(64))).propose(SEED, n)
    assert 0 <= off < n
