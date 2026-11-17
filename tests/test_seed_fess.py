"""SeedFessScheduler: FESS (Shoham) feature-space seed scheduling.

Seeds project to cells of a small feature grid; the least-explored occupied
cell is fuzzed next, then its least-fuzzed seed. Effort is charged to the
cell a seed was in when picked. Every draw goes through the injected RNG, so
each test scripts the exact sequence (Hard Rule 39); an unscripted draw
raises StopIteration.
"""

from __future__ import annotations

import io
import pickle
from types import SimpleNamespace

import pytest

from fuzzer_tool.core.schedulers.seed_fess import (
    FESS_BUCKET_CAP,
    SeedFessScheduler,
    fess_cell,
)
from fuzzer_tool.core.state_store import safe_load
from tests.support.scripted_rng import ScriptedRng

X = (1, 0, 0, 0)
Y = (2, 0, 0, 0)
N_AXES = 4


def _sched(**rng_kw) -> SeedFessScheduler:
    return SeedFessScheduler(rng=ScriptedRng(**rng_kw))


def _effort(s: SeedFessScheduler) -> dict:
    return s.to_dict()["effort"]


class TestPolicy:
    def test_tied_cells_drawn_then_lone_seed_without_draw(self):
        s = _sched(choice_idxs=[1])
        cells = {"a": X, "b": X, "c": Y}

        # Cells tie at 0 effort; idx 1 -> Y, whose lone seed costs no draw.
        assert s.select_seed(["a", "b", "c"], cells.__getitem__) == "c"

    def test_least_explored_cell_wins_without_draw(self):
        s = _sched(choice_idxs=[1, 0])
        cells = {"a": X, "b": X, "c": Y}
        s.select_seed(["a", "b", "c"], cells.__getitem__)
        s.record("c", success=False)

        # Y has effort 1, X has 0: X is unique; a/b tie -> idx 0.
        assert s.select_seed(["a", "b", "c"], cells.__getitem__) == "a"

    def test_least_fuzzed_seed_inside_cell(self):
        s = _sched()
        same = {"a": X, "b": X}
        s.select_seed(["a"], same.__getitem__)
        s.record("a", success=True)

        # One cell, b has fewer execs: no draw at all.
        assert s.select_seed(["a", "b"], same.__getitem__) == "b"

    def test_migration_keeps_history_in_old_cell(self):
        s = _sched(choice_idxs=[0, 0])
        before = {"a": X, "b": Y}
        s.select_seed(["a", "b"], before.__getitem__)
        s.record("a", success=False)

        # "a" migrated to Y: its past exec stays X's, the next one is Y's.
        after = {"a": Y, "b": Y}
        s.select_seed(["a", "b"], after.__getitem__)
        s.record("a", success=False)

        assert _effort(s) == {X: 1, Y: 1}


class TestFalsification:
    """Degenerate grids reduce to the least-executed seed policy."""

    EXECS = {"a": 3, "b": 1, "c": 2}

    def _reference(self) -> str:
        return min(self.EXECS, key=self.EXECS.__getitem__)

    def _drive(self, cell) -> str:
        s = _sched()
        ids = list(self.EXECS)
        for key, n in self.EXECS.items():
            s.select_seed([key], cell)
            for _ in range(n):
                s.record(key, success=False)
        return s.select_seed(ids, cell)

    def test_control_reference_is_unique_minimum(self):
        """Control: the oracle must not tie, else a draw would be scripted."""
        low = self.EXECS[self._reference()]

        assert sum(v == low for v in self.EXECS.values()) == 1

    def test_one_cell_is_least_executed(self):
        assert self._drive(lambda _k: X) == self._reference()

    def test_cell_per_seed_is_least_executed(self):
        own = {"a": (1, 0, 0, 0), "b": (2, 0, 0, 0), "c": (3, 0, 0, 0)}

        assert self._drive(own.__getitem__) == self._reference()


class TestAdversarial:
    def test_empty_returns_blank(self):
        assert _sched().select_seed([], lambda _k: X) == ""

    def test_single_seed_draws_nothing(self):
        assert _sched().select_seed(["a"], lambda _k: X) == "a"

    def test_requires_rng(self):
        with pytest.raises(ValueError):
            SeedFessScheduler(rng=None)

    def test_declares_no_priors(self):
        assert SeedFessScheduler.supports_priors is False

    def test_record_of_unseen_key_charges_nothing(self):
        s = _sched()
        s.record("ghost", success=True)

        assert _effort(s) == {}
        assert s.bandit_stats() == {"ghost": (1.0, 0.0)}

    def test_departed_seeds_leave_cell_memory(self):
        s = _sched()
        ids = [str(i) for i in range(64)]
        for key in ids:
            s.select_seed([key], lambda _k: X)

        # "5" left long ago: the ledger trim must have dropped its cell too.
        s.record("5", success=False)

        assert len(s._cell_of) <= len(s._counts)
        assert _effort(s) == {}


class TestCell:
    def test_zero_is_origin(self):
        assert fess_cell(0, 0, 0, 0) == (0,) * N_AXES

    def test_log2_buckets(self):
        # 1 -> 1, 3 -> 2, 4 -> 3 (bit length); size halves: 255 has 8 bits -> 4.
        assert fess_cell(1, 3, 4, 255) == (1, 2, 3, 4)

    def test_huge_values_cap(self):
        big = 1 << 64

        assert fess_cell(big, big, big, big) == (FESS_BUCKET_CAP,) * N_AXES

    def test_negative_values_floor_at_zero(self):
        assert fess_cell(-5, -1, -9, -100) == (0,) * N_AXES

    def test_grid_is_bounded(self):
        values = [0, 1, 2, 3, 7, 100, 1 << 20, 1 << 40]
        cells = {
            fess_cell(e, r, d, z) for e in values for r in values for d in values for z in values
        }

        assert len(cells) <= (FESS_BUCKET_CAP + 1) ** N_AXES


class TestPersistence:
    def test_round_trip_through_safe_unpickler(self):
        s = _sched()
        s.select_seed(["a"], lambda _k: X)
        s.record("a", success=False)

        blob = pickle.dumps(s.to_dict())
        t = _sched()
        t.from_dict(safe_load(io.BytesIO(blob)))

        assert _effort(t) == {X: 1}

    @pytest.mark.parametrize(
        "data",
        [
            None,
            [],
            {"effort": "junk"},
            {"effort": {"bad": 1}},
            {"effort": {(1, 2): 1}},
            {"effort": {(1, 0, 0, FESS_BUCKET_CAP + 1): 1}},
            {"effort": {X: -3}},
            {"effort": {X: "3"}},
        ],
    )
    def test_malformed_payload_starts_fresh(self, data):
        t = _sched()
        t.select_seed(["a"], lambda _k: Y)
        t.record("a", success=False)
        t.from_dict(data)

        assert _effort(t) == {}


# --- SeedPicker cell signal --------------------------------------------------


class _Tracker:
    def __init__(self, edges: dict[str, int], rare: dict[str, int]):
        self._edges, self._rare = edges, rare
        self.calls = 0

    def get_seed_edge_count(self, key: str) -> int:
        self.calls += 1
        return self._edges.get(key, 0)

    def rare_edge_count(self, key: str) -> int:
        return self._rare.get(key, 0)


def _picker(corpus, meta, tracker=None, exec_count=0):
    from fuzzer_tool.services.seed_picker import SeedPicker

    f = SimpleNamespace(
        corpus=list(corpus),
        seed_meta=meta,
        exec_count=exec_count,
        mean_exec_time=lambda: 0.0,
        _favored=set(),
        _edge_tracker=tracker,
        _seed_key=lambda d: d.decode(),
    )
    sp = SeedPicker.__new__(SeedPicker)
    sp.f = f
    return sp


class TestCellSignal:
    def test_reads_tracker_and_meta(self):
        tracker = _Tracker({"a": 9}, {"a": 2})
        meta = {b"a": {"lineage_depth": 5, "input_size": 300}}
        sp = _picker([b"a"], meta, tracker)

        cell = sp._seed_signals({"a": b"a"})["cell"]

        assert cell("a") == fess_cell(9, 2, 5, 300)

    def test_no_tracker_no_meta_uses_length(self):
        sp = _picker([b"abc"], {})

        cell = sp._seed_signals({"abc": b"abc"})["cell"]

        assert cell("abc") == fess_cell(0, 0, 0, len(b"abc"))

    def test_memo_reused_until_drift(self):
        tracker = _Tracker({"a": 1, "b": 1}, {})
        sp = _picker([b"a", b"b"], {}, tracker)
        key_to_seed = {"a": b"a", "b": b"b"}

        sp._seed_signals(key_to_seed)["cell"]("a")
        first = tracker.calls
        sp._seed_signals(key_to_seed)["cell"]("b")

        assert first == len(key_to_seed)
        assert tracker.calls == first

    def test_seed_admitted_after_pass_is_projected(self):
        tracker = _Tracker({"a": 1, "new": 4}, {})
        sp = _picker([b"a"], {}, tracker)
        sp._seed_signals({"a": b"a"})["cell"]("a")

        cell = sp._seed_signals({"a": b"a", "new": b"new"})["cell"]

        assert cell("new") == fess_cell(4, 0, 0, len(b"new"))


# --- Fuzzer wiring -----------------------------------------------------------


def _real_fuzzer(tmp_path, **kw):
    from fuzzer_tool.services.fuzzer import Fuzzer

    corpus, crashes = tmp_path / "c", tmp_path / "k"
    corpus.mkdir(parents=True, exist_ok=True)
    crashes.mkdir(parents=True, exist_ok=True)
    return Fuzzer(
        target="targets/test_target",
        corpus_dir=str(corpus),
        crashes_dir=str(crashes),
        max_len=4096,
        **kw,
    )


class TestFuzzerWiring:
    def test_resume_restores_effort(self, tmp_path):
        f = _real_fuzzer(tmp_path, seed_fess_scheduler=True)
        f._seed_fess.select_seed(["a"], lambda _k: X)
        f._seed_fess.record("a", success=False)
        f._save_learned()

        g = _real_fuzzer(tmp_path, seed_fess_scheduler=True)
        g.resume = True
        g._state_store = f._state_store
        g._load_learned()

        assert _effort(g._seed_fess) == {X: 1}
        assert g._seed_fess in g._seed_os_arms

    def test_fresh_run_ignores_saved_effort(self, tmp_path):
        f = _real_fuzzer(tmp_path, seed_fess_scheduler=True)
        f._seed_fess.select_seed(["a"], lambda _k: X)
        f._seed_fess.record("a", success=False)
        f._save_learned()

        g = _real_fuzzer(tmp_path, seed_fess_scheduler=True)
        g._state_store = f._state_store
        g._load_learned()

        assert _effort(g._seed_fess) == {}

    def test_off_writes_no_section(self, tmp_path):
        f = _real_fuzzer(tmp_path)
        f._save_learned()

        assert "seed_fess" not in f._state_store
