"""``--bloat-reward``: scale operator reward by child/parent edges per bit."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from fuzzer_tool.core.edge_density import BLOAT_FLOOR, bloat_factor, edges_per_bit
from fuzzer_tool.core.run_mode import Switch
from fuzzer_tool.services.corpus_manager import TRACE_EDGES
from fuzzer_tool.services.fuzz_round import FuzzRound
from fuzzer_tool.services.fuzzer import Fuzzer

PARENT = b"P" * 10
CHILD = b"C" * 40


def test_factor_is_density_ratio():
    """Docstring example: 40 edges/10 B -> 44 edges/40 B."""
    want = edges_per_bit(44, 40) / edges_per_bit(40, 10)
    assert bloat_factor((40, 10), (44, 40)) == pytest.approx(want)


def test_denser_child_capped_at_one():
    assert bloat_factor((10, 10), (100, 20)) == 1.0


def test_extreme_bloat_floored():
    assert bloat_factor((100, 1), (101, 10_000)) == BLOAT_FLOOR


def test_child_not_longer_is_neutral():
    """Falsification: a sparser child that did not grow is not bloat."""
    assert bloat_factor((40, 10), (5, 10)) is None
    assert bloat_factor((40, 10), (5, 8)) is None


@pytest.mark.parametrize(
    "parent, child", [((0, 10), (5, 40)), ((40, 10), (0, 40)), ((5, 0), (5, 4))]
)
def test_unknown_side_is_neutral(parent, child):
    """Adversarial: missing trace size or empty parent -> None, never a cut."""
    assert bloat_factor(parent, child) is None


def _shaper(on: Switch, new_ids, trace, parent_trace: int | None):
    f = Fuzzer.__new__(Fuzzer)
    f._bloat_reward = on
    f._bloat_reward_rounds = 0
    f._bloat_reward_neutral = 0
    f._bloat_reward_factor_sum = 0.0
    f._last_new_edge_ids = new_ids
    f._last_trace_edges = trace
    f.seed_meta = {PARENT: {} if parent_trace is None else {TRACE_EDGES: parent_trace}}
    return f


def test_shape_applies_and_counts():
    trace = tuple(range(44))
    f = _shaper(Switch.ON, [1], trace, parent_trace=40)
    got = f._bloat_reward_shape(PARENT, CHILD)

    assert got == pytest.approx(bloat_factor((40, len(PARENT)), (len(trace), len(CHILD))))
    assert f.bloat_reward_stats()["bloat_reward_rounds"] == 1


def test_shape_off_or_no_discovery():
    assert _shaper(Switch.OFF, [1], (1, 2), 40)._bloat_reward_shape(PARENT, CHILD) is None
    assert _shaper(Switch.ON, [], (1, 2), 40)._bloat_reward_shape(PARENT, CHILD) is None


def test_shape_unknown_parent_counted_neutral():
    """Adversarial: parent never traced (e.g. imported mid-run) -> neutral, counted."""
    f = _shaper(Switch.ON, [1], (1, 2, 3), parent_trace=None)
    assert f._bloat_reward_shape(PARENT, CHILD) is None
    assert f.bloat_reward_stats()["bloat_reward_neutral_rounds"] == 1


def test_op_rewards_compose_bloat_shape():
    """Wiring: _op_rewards multiplies successful weights by the bloat factor."""
    factor = 0.5
    f = SimpleNamespace(
        _last_ops_used=["a", "b"],
        _cost_adjusted_weight=lambda op, w: w,
        _credit_reward_shape=lambda: None,
        _continuum_reward_shape=lambda: None,
        _bloat_reward_shape=lambda parent, child: factor,
    )
    r = FuzzRound.__new__(FuzzRound)
    r._f, r._data, r._mutated = f, PARENT, CHILD
    r._success, r._effective, r._surprisal_weight = True, None, 1.0

    assert r._op_rewards() == [("a", True, factor), ("b", True, factor)]


def test_admit_records_child_trace():
    child_meta: dict = {}
    r = FuzzRound.__new__(FuzzRound)
    r._f, r._trace_n = SimpleNamespace(seed_meta={CHILD: child_meta}), 3
    r._note_trace(CHILD)
    assert child_meta[TRACE_EDGES] == 3


def test_admit_without_trace_records_nothing():
    """Adversarial: a round that read no trace stores no 0."""
    child_meta: dict = {}
    r = FuzzRound.__new__(FuzzRound)
    r._f, r._trace_n = SimpleNamespace(seed_meta={CHILD: child_meta}), 0
    r._note_trace(CHILD)
    assert TRACE_EDGES not in child_meta


class TestResume:
    SEED = b"bloat-resume-seed"

    def _fuzzer(self, tmp_path, **kw):
        from fuzzer_tool.adapters.filesystem import save_to_corpus

        corpus, crashes = tmp_path / "c", tmp_path / "k"
        (corpus / "seeds").mkdir(parents=True, exist_ok=True)
        crashes.mkdir(parents=True, exist_ok=True)
        save_to_corpus(self.SEED, corpus, set(), None)
        return Fuzzer(target="/bin/true", corpus_dir=str(corpus), crashes_dir=str(crashes), **kw)

    def test_trace_survives_resume(self, tmp_path):
        f = self._fuzzer(tmp_path)
        f.seed_meta[self.SEED][TRACE_EDGES] = 7
        f._save_state()
        assert self._fuzzer(tmp_path, resume=True).seed_meta[self.SEED][TRACE_EDGES] == 7

    def test_flag_wires(self, tmp_path):
        assert self._fuzzer(tmp_path, bloat_reward=Switch.ON)._bloat_reward is Switch.ON
