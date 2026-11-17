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

from fuzzer_tool.core.run_mode import Switch
from fuzzer_tool.core.schedulers.seed_fess import (
    FESS_BUCKET_CAP,
    FESS_CELL_EDGE_CAP,
    FESS_HOT_WINDOW,
    FESS_NOVELTY_HALF,
    FESS_RANDOM_EVERY,
    FessMode,
    SeedFessScheduler,
    cell_better,
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


# --- Parent-cell inheritance + new-cell jump (--fess-inherit) ---------------


class TestCellBetter:
    """Order: more edges, then more rare edges, then smaller size, then shallower."""

    def test_more_edges_wins_over_everything_else(self):
        assert cell_better((2, 0, 7, 7), (1, 7, 0, 0))

    def test_rare_breaks_edge_tie(self):
        assert cell_better((1, 2, 0, 0), (1, 1, 0, 0))

    def test_smaller_size_breaks_tie(self):
        assert cell_better((1, 1, 0, 2), (1, 1, 0, 3))

    def test_shallower_breaks_last_tie(self):
        assert cell_better((1, 1, 1, 2), (1, 1, 2, 2))

    def test_equal_is_not_strictly_better(self):
        """Falsification: an equal child must inherit, never self-allocate."""
        assert not cell_better(X, X)

    def test_order_is_antisymmetric(self):
        a, b = (1, 1, 0, 2), (1, 1, 0, 3)

        assert cell_better(a, b) != cell_better(b, a)


def _inherit_picker(parents: dict, edges: dict, inherit=Switch.ON):
    """Picker over *parents* (key -> parent key); every seed is 8 bytes, depth 0."""
    meta = {k.encode(): {"parent_key": p, "input_size": 8} for k, p in parents.items()}
    sp = _picker([k.encode() for k in parents], meta, _Tracker(edges, {}))
    sp.f._fess_inherit = inherit
    return sp, {k: k.encode() for k in parents}


class TestInherit:
    PARENTS = {"p": None, "worse": "p", "better": "p"}
    EDGES = {"p": 4, "worse": 1, "better": 16}

    def _cells(self, inherit=Switch.ON, parents=None, edges=None):
        sp, key_to_seed = _inherit_picker(parents or self.PARENTS, edges or self.EDGES, inherit)
        return sp._fess_cells(key_to_seed)

    def test_non_improving_child_takes_parent_cell(self):
        cells = self._cells()

        assert cells["worse"] == cells["p"] == fess_cell(4, 0, 0, 8)

    def test_improving_child_keeps_own_cell(self):
        cells = self._cells()

        assert cells["better"] == fess_cell(16, 0, 0, 8)

    def test_flag_off_is_own_cell(self):
        """Falsification: without --fess-inherit every seed keeps its own cell."""
        cells = self._cells(inherit=Switch.OFF)

        assert cells["worse"] == fess_cell(1, 0, 0, 8)

    def test_evicted_parent_means_own_cell(self):
        cells = self._cells(parents={"worse": "gone"}, edges={"worse": 1})

        assert cells["worse"] == fess_cell(1, 0, 0, 8)

    def test_chain_projects_to_root_in_any_order(self):
        # Children listed before parents: g -> c -> p, neither beats p.
        parents = {"g": "c", "c": "p", "p": None}
        cells = self._cells(parents=parents, edges={"p": 64, "c": 4, "g": 2})

        assert cells["g"] == cells["c"] == cells["p"] == fess_cell(64, 0, 0, 8)

    def test_child_compared_with_projected_parent(self):
        # c inherits p's cell; g beats c's own cell but not the cell c
        # competes in, so g inherits too.
        parents = {"p": None, "c": "p", "g": "c"}
        cells = self._cells(parents=parents, edges={"p": 64, "c": 2, "g": 8})

        assert cells["g"] == cells["p"]

    @pytest.mark.parametrize(
        "parents",
        [{"a": "b", "b": "a"}, {"a": "a"}, {"a": "b", "b": "c", "c": "a"}],
    )
    def test_cycles_terminate(self, parents):
        cells = self._cells(parents=parents, edges=dict.fromkeys(parents, 1))

        assert set(cells) == set(parents)

    def test_seed_admitted_between_refreshes_inherits(self):
        sp, _ = _inherit_picker({"p": None, "new": "p"}, {"p": 4, "new": 1})
        sp._fess_cells({"p": b"p"})

        cells = sp._fess_cells({"p": b"p", "new": b"new"})

        assert cells["new"] == cells["p"]


class TestNewCellJump:
    Z = (3, 0, 0, 0)

    def _primed(self, jump=Switch.ON, choice_idxs=(0,)):
        """X charged once, Y never: two cells known before Z appears."""
        s = SeedFessScheduler(rng=ScriptedRng(choice_idxs=choice_idxs), jump=jump)
        cells = {"a": X, "b": Y}
        s.select_seed(["a", "b"], cells.__getitem__)
        s.record("a", success=False)
        return s

    def test_new_cell_picked_without_draw(self):
        s = self._primed()
        cells = {"a": X, "b": Y, "z": self.Z}

        # Y and Z tie at effort 0; the jump takes Z and draws nothing.
        assert s.select_seed(["a", "b", "z"], cells.__getitem__) == "z"

    def test_jump_fires_once(self):
        s = self._primed()
        cells = {"a": X, "b": Y, "z": self.Z}
        s.select_seed(["a", "b", "z"], cells.__getitem__)
        s.record("z", success=False)

        # Back to least effort: Y is the unique minimum.
        assert s.select_seed(["a", "b", "z"], cells.__getitem__) == "b"

    def test_off_ties_are_drawn(self):
        """Falsification: without the jump Y/Z tie and the pool draws (idx 0 -> Y)."""
        s = self._primed(jump=Switch.OFF, choice_idxs=(0, 0))
        cells = {"a": X, "b": Y, "z": self.Z}

        assert s.select_seed(["a", "b", "z"], cells.__getitem__) == "b"

    def test_first_select_primes_without_jump(self):
        # Every cell is new on the first select: least effort decides
        # (X/Y tie -> scripted idx 1 -> Y), not a jump.
        s = SeedFessScheduler(rng=ScriptedRng(choice_idxs=[1]), jump=Switch.ON)
        cells = {"a": X, "b": Y}

        assert s.select_seed(["a", "b"], cells.__getitem__) == "b"

    def test_vanished_new_cell_is_skipped(self):
        s = self._primed()
        s.select_seed(["z"], lambda _k: self.Z)

        # Z left before a multi-seed select: no jump to an empty cell.
        cells = {"a": X, "b": Y}
        assert s.select_seed(["a", "b"], cells.__getitem__) == "b"

    def test_new_cells_jump_in_arrival_order(self):
        s = self._primed()
        cells = {"a": X, "b": Y, "z": self.Z, "w": (4, 0, 0, 0)}
        ids = list(cells)

        assert s.select_seed(ids, cells.__getitem__) == "z"
        assert s.select_seed(ids, cells.__getitem__) == "w"


class TestInheritWiring:
    def test_flag_reaches_scheduler_and_picker(self, tmp_path):
        f = _real_fuzzer(tmp_path, seed_fess_scheduler=True, fess_inherit=Switch.ON)

        assert f._fess_inherit is Switch.ON
        assert f._seed_fess._jump_on is Switch.ON

    def test_without_arm_is_inert(self, tmp_path, caplog):
        f = _real_fuzzer(tmp_path, fess_inherit=Switch.ON)

        assert f._fess_inherit is Switch.OFF
        assert "--fess-inherit" in caplog.text


# --- Three-way cell schedule (--seed-fess-mode triple) ------------------------

Z = (3, 0, 0, 0)
TRIO = {"a": X, "b": Y, "c": Z}


def _triple(**rng_kw) -> SeedFessScheduler:
    return SeedFessScheduler(rng=ScriptedRng(**rng_kw), mode=FessMode.TRIPLE)


def _turn_kinds(turns: int) -> list[str]:
    """Schedule from its definition: every Nth random, else even best, odd round robin."""
    kinds = []
    for t in range(1, turns + 1):
        if t % FESS_RANDOM_EVERY == 0:
            kinds.append("random")
        elif t % 2 == 0:
            kinds.append("best")
        else:
            kinds.append("rr")
    return kinds


def _picks(s: SeedFessScheduler, n: int, cells=TRIO) -> list[str]:
    return [s.select_seed(list(cells), cells.__getitem__) for _ in range(n)]


class TestTriple:
    def test_schedule_interleaves_rr_best_random(self):
        turns = FESS_RANDOM_EVERY + 1
        random_idx = 2
        s = _triple(choice_idxs=[random_idx])
        s.record("b", success=True)  # Y: rate 1.0; X, Z untried: 0.0

        order = list(TRIO)  # creation order = first-select member order
        expected, rr = [], 0
        for kind in _turn_kinds(turns):
            if kind == "rr":
                expected.append(order[rr % len(order)])
                rr += 1
            elif kind == "best":
                expected.append("b")
            else:
                expected.append(order[random_idx])

        assert _picks(s, turns) == expected

    def test_best_rate_aggregates_over_cell_seeds(self):
        # X: a 1/1 but a2 0/3 -> 1/4. Y: b 1/2. Per-seed max would say X.
        s = _triple()
        cells = {"a": X, "a2": X, "b": Y}
        s.record("a", success=True)
        for _ in range(3):
            s.record("a2", success=False)
        s.record("b", success=True)
        s.record("b", success=False)

        picks = _picks(s, 2, cells)

        assert picks[1] == "b"

    def test_best_ties_drawn_from_pool(self):
        s = _triple(choice_idxs=[1])
        s.record("a", success=True)
        s.record("b", success=True)

        # Turn 2: X and Y tie at 1.0 -> idx 1 -> Y.
        assert _picks(s, 2)[1] == "b"

    def test_round_robin_skips_empty_cells(self):
        s = _triple(choice_idxs=[0])
        _picks(s, 2)  # rr X, best: all 0.0 tie -> idx 0 -> X

        # Y left: turn 3 round robin moves past it to Z.
        cells = {"a": X, "c": Z}
        assert s.select_seed(list(cells), cells.__getitem__) == "c"

    def test_least_fuzzed_seed_inside_chosen_cell(self):
        s = _triple()
        cells = {"a": X, "a2": X, "b": Y}
        s.record("a", success=False)

        # Turn 1 round robin -> X; a2 has fewer execs.
        assert _picks(s, 1, cells) == ["a2"]

    def test_one_cell_never_draws(self):
        """Falsification: one occupied cell reduces every turn to least-fuzzed."""
        s = _triple()
        cells = {"a": X, "a2": X}
        s.record("a", success=False)

        assert _picks(s, FESS_RANDOM_EVERY, cells) == ["a2"] * FESS_RANDOM_EVERY

    def test_rejects_non_enum_mode(self):
        with pytest.raises(ValueError):
            SeedFessScheduler(rng=ScriptedRng(), mode="triple")

    def test_jump_preempts_without_spending_a_turn(self):
        s = SeedFessScheduler(rng=ScriptedRng(), mode=FessMode.TRIPLE, jump=Switch.ON)
        two = {"a": X, "b": Y}
        _picks(s, 1, two)  # turn 1: rr -> X (priming select, no jump)

        jumped = _picks(s, 1)[0]  # Z is new: jump, turn stays 1
        s.record("c", success=True)

        assert jumped == "c"
        assert _picks(s, 2) == ["c", "b"]  # turn 2 best (Z 1.0), turn 3 rr -> Y


class TestTriplePersistence:
    def _driven(self) -> SeedFessScheduler:
        s = _triple()
        s.record("b", success=True)
        _picks(s, 3)  # rr X, best Y, rr Y
        return s

    def test_round_trip_continues_schedule(self):
        s = self._driven()
        blob = pickle.dumps(s.to_dict())
        t = _triple()
        t.from_dict(safe_load(io.BytesIO(blob)))
        t.record("b", success=True)  # seed ledgers are not persisted

        # Turn 4 best (Y), turn 5 round robin resumes after Y -> Z.
        assert _picks(t, 2) == _picks(s, 2) == ["b", "c"]

    @pytest.mark.parametrize(
        "bad",
        [
            {"turn": -1},
            {"turn": True},
            {"turn": "3"},
            {"cursor": 9},
            {"cursor": -1},
            {"order": [X, X]},
            {"order": [(1, 2)]},
            {"order": "junk"},
        ],
    )
    def test_malformed_schedule_starts_fresh(self, bad):
        data = {**self._driven().to_dict(), **bad}
        t = _triple()
        t.from_dict(data)

        assert t.to_dict()["turn"] == 0
        assert t.to_dict()["cursor"] == 0
        assert t.to_dict()["order"] == []

    def test_malformed_schedule_keeps_effort(self):
        s = _sched(choice_idxs=[0])
        s.select_seed(["a"], lambda _k: X)
        s.record("a", success=False)
        t = _triple()
        t.from_dict({**s.to_dict(), "turn": -5})

        assert _effort(t) == {X: 1}


class TestModeWiring:
    def test_mode_reaches_scheduler_and_resumes(self, tmp_path):
        f = _real_fuzzer(tmp_path, seed_fess_scheduler=True, seed_fess_mode=FessMode.TRIPLE)
        _picks(f._seed_fess, 3)
        f._save_learned()

        g = _real_fuzzer(tmp_path, seed_fess_scheduler=True, seed_fess_mode=FessMode.TRIPLE)
        g.resume = True
        g._state_store = f._state_store
        g._load_learned()

        assert g._seed_fess._mode is FessMode.TRIPLE
        assert g._seed_fess.to_dict()["turn"] == 3

    def test_default_is_least_effort(self, tmp_path):
        f = _real_fuzzer(tmp_path, seed_fess_scheduler=True)

        assert f._seed_fess._mode is FessMode.LEAST_EFFORT

    @pytest.mark.parametrize("mode", list(FessMode))
    def test_cli_values_map_to_enum(self, monkeypatch, mode):
        args = _cli_args(monkeypatch, "--seed-fess-mode", mode.value)

        assert FessMode(args.seed_fess_mode) is mode


def _cli_args(monkeypatch, *extra: str):
    """Parse ``fuzz /bin/true *extra`` through the real CLI, cmd_fuzz stubbed."""
    import sys

    from fuzzer_tool.cli import commands

    captured: dict = {}

    def _spy(args):
        captured["args"] = args
        return 0

    monkeypatch.setattr(commands, "cmd_fuzz", _spy)
    monkeypatch.setattr(sys, "argv", ["fuzzer-tool", "fuzz", "/bin/true", *extra])
    assert commands.main() == 0
    return captured["args"]


# --- Per-cell scoped novelty (--fess-cell-novelty) -----------------------------

OWN = {"a": frozenset({1, 2}), "b": frozenset({1, 2}), "c": frozenset({7})}


def _novel(rng=None, own=OWN, mode=FessMode.LEAST_EFFORT) -> SeedFessScheduler:
    return SeedFessScheduler(
        rng=rng or ScriptedRng(),
        mode=mode,
        novelty=Switch.ON,
        seed_edges=lambda k: own.get(k, frozenset()),
    )


def _credit(n: int) -> float:
    """Novelty credit from its definition: n / (n + half)."""
    return n / (n + FESS_NOVELTY_HALF)


def _exec(s: SeedFessScheduler, key: str, edges, success=False, weight=1.0) -> None:
    s.observe_edges(key, frozenset(edges))
    s.record(key, success=success, weight=weight)


class TestCellNovelty:
    def test_globally_known_edge_new_in_cell_is_credited(self):
        s = _novel()
        s.select_seed(["a"], lambda _k: X)

        _exec(s, "a", {1, 2, 3})  # 3 is new to X, not to the parent's own run

        assert s.bandit_stats()["a"] == (_credit(1), 1.0 - _credit(1))

    def test_parent_own_edges_earn_nothing(self):
        s = _novel()
        s.select_seed(["a"], lambda _k: X)

        _exec(s, "a", OWN["a"])

        assert s.bandit_stats()["a"] == (0.0, 1.0)

    def test_edge_credited_once_per_cell(self):
        s = _novel(rng=ScriptedRng(choice_idxs=[0]))  # a/b tie inside X
        same = {"a": X, "b": X}
        s.select_seed(["a", "b"], same.__getitem__)
        _exec(s, "a", {1, 2, 3})

        _exec(s, "b", {1, 2, 3})  # X already holds 3

        assert s.bandit_stats()["b"] == (0.0, 1.0)

    def test_novelty_is_scoped_to_the_cell(self):
        s = _novel(rng=ScriptedRng(choice_idxs=[0]))
        cells = {"a": X, "c": Y}
        s.select_seed(["a", "c"], cells.__getitem__)
        _exec(s, "a", {1, 2, 3})

        _exec(s, "c", {7, 3})  # 3 known in X, new in Y

        assert s.bandit_stats()["c"] == (_credit(1), 1.0 - _credit(1))

    def test_global_find_keeps_larger_weight(self):
        s = _novel()
        s.select_seed(["a"], lambda _k: X)

        _exec(s, "a", {1, 2, 3}, success=True, weight=0.9)

        assert s.bandit_stats()["a"] == pytest.approx((0.9, 0.1))

    def test_off_ignores_edges(self):
        """Falsification: without the flag the same exec earns nothing."""
        s = SeedFessScheduler(rng=ScriptedRng())
        s.select_seed(["a"], lambda _k: X)

        _exec(s, "a", {1, 2, 3})

        assert s.bandit_stats()["a"] == (0.0, 1.0)


class TestHotCells:
    """Hot = never charged, or fewer than FESS_HOT_WINDOW charges since in-cell novelty."""

    def _setup(self, novelty=Switch.ON):
        # X: 3 charges, the last novel. Y: 1 charge, never novel.
        s = SeedFessScheduler(
            rng=ScriptedRng(choice_idxs=[0]),
            novelty=novelty,
            seed_edges=lambda k: OWN.get(k, frozenset()),
        )
        cells = {"a": X, "c": Y}
        s.select_seed(["a", "c"], cells.__getitem__)
        _exec(s, "c", OWN["c"])
        for edges in ({1, 2}, {1, 2}, {1, 2, 3}):
            _exec(s, "a", edges)
        return s, cells

    def test_recent_novelty_beats_least_effort(self):
        s, cells = self._setup()

        assert s.select_seed(list(cells), cells.__getitem__) == "a"

    def test_off_is_least_effort(self):
        """Falsification: same charges, flag off -> Y (least effort)."""
        s, cells = self._setup(novelty=Switch.OFF)

        assert s.select_seed(list(cells), cells.__getitem__) == "c"

    def test_cell_cools_after_window(self):
        s, cells = self._setup()
        for _ in range(FESS_HOT_WINDOW):
            _exec(s, "a", {1, 2})

        assert s.select_seed(list(cells), cells.__getitem__) == "c"

    def test_one_charge_before_window_still_hot(self):
        s, cells = self._setup()
        for _ in range(FESS_HOT_WINDOW - 1):
            _exec(s, "a", {1, 2})

        assert s.select_seed(list(cells), cells.__getitem__) == "a"

    def test_no_novelty_matches_off_sequence(self):
        """Falsification: with no in-cell novelty ever, on and off pick alike."""
        cells = {"a": X, "b": X, "c": Y}
        runs = []
        for flag in (Switch.OFF, Switch.ON):
            s = SeedFessScheduler(
                rng=ScriptedRng(choice_idxs=[0] * 8),
                novelty=flag,
                seed_edges=lambda k: OWN.get(k, frozenset()),
            )
            picks = []
            for _ in range(6):
                k = s.select_seed(list(cells), cells.__getitem__)
                _exec(s, k, OWN[k])
                picks.append(k)
            runs.append(picks)

        assert runs[0] == runs[1]

    def test_triple_random_turn_ignores_hot_filter(self):
        # X gets novelty on every exec and Y is charged but cold, so best /
        # round robin stay on X; turn 8 (random) still draws over both
        # cells: idx 1 -> Y. Single-seed selects spend no turn.
        s = _novel(rng=ScriptedRng(choice_idxs=[1]), mode=FessMode.TRIPLE)
        cells = {"a": X, "c": Y}
        s.select_seed(["a"], lambda _k: X)
        _exec(s, "a", {1, 2, 100})
        s.select_seed(["c"], lambda _k: Y)
        _exec(s, "c", OWN["c"])
        for t in range(1, FESS_RANDOM_EVERY):
            assert s.select_seed(list(cells), cells.__getitem__) == "a"
            _exec(s, "a", {1, 2, 100 + t})

        assert s.select_seed(list(cells), cells.__getitem__) == "c"


class TestNoveltyAdversarial:
    def test_unknown_key_earns_nothing(self):
        s = _novel()
        _exec(s, "ghost", {1, 2, 3})

        assert s.bandit_stats()["ghost"] == (0.0, 1.0)

    def test_stash_not_applied_to_another_key(self):
        s = _novel(rng=ScriptedRng(choice_idxs=[0]))  # a/b tie inside X
        same = {"a": X, "b": X}
        s.select_seed(["a", "b"], same.__getitem__)
        s.observe_edges("a", frozenset({1, 2, 3}))

        s.record("b", success=False)

        assert s.bandit_stats()["b"] == (0.0, 1.0)

    def test_empty_edges(self):
        s = _novel()
        s.select_seed(["a"], lambda _k: X)

        _exec(s, "a", set())

        assert s.bandit_stats()["a"] == (0.0, 1.0)

    def test_cell_cap_bounds_memory_and_credit(self):
        s = _novel(own={})
        s.select_seed(["a"], lambda _k: X)
        _exec(s, "a", range(FESS_CELL_EDGE_CAP + 10))

        _exec(s, "a", {FESS_CELL_EDGE_CAP + 100})  # full cell: no room, no credit

        assert s._stored == FESS_CELL_EDGE_CAP
        assert s.bandit_stats()["a"][0] == pytest.approx(_credit(FESS_CELL_EDGE_CAP))

    def test_global_budget_bounds_all_cells(self, monkeypatch):
        import fuzzer_tool.core.schedulers.seed_fess as mod

        budget = 5
        monkeypatch.setattr(mod, "FESS_NOVELTY_BUDGET", budget)
        s = _novel(rng=ScriptedRng(choice_idxs=[0]), own={})
        cells = {"a": X, "c": Y}
        s.select_seed(["a", "c"], cells.__getitem__)
        _exec(s, "a", range(4))
        _exec(s, "c", range(10, 20))

        assert s._stored == budget
        assert s.bandit_stats()["c"][0] == pytest.approx(_credit(budget - 4))


class TestNoveltyWiring:
    def test_flag_binds_observer(self, tmp_path):
        f = _real_fuzzer(tmp_path, seed_fess_scheduler=True, fess_cell_novelty=Switch.ON)

        assert f._fess_observe is not None
        assert f._seed_fess._novelty is Switch.ON

    def test_without_arm_is_inert(self, tmp_path, caplog):
        f = _real_fuzzer(tmp_path, fess_cell_novelty=Switch.ON)

        assert f._fess_observe is None
        assert "--fess-cell-novelty" in caplog.text

    def test_off_binds_nothing(self, tmp_path):
        f = _real_fuzzer(tmp_path, seed_fess_scheduler=True)

        assert f._fess_observe is None

    def _round(self, edges, timeout=False):
        from fuzzer_tool.services.fuzz_round import FuzzRound

        seen = []
        f = SimpleNamespace(
            _lineage_exhaust=None,
            _seed_quality=None,
            _seed_canary=None,
            _seed_round_robin=None,
            _seed_drr=None,
            _seed_os_arms=(object(),),
            _seed_key=lambda d: d.decode(),
            _edge_tracker=SimpleNamespace(estimate_distinct_edges_f0=lambda: None),
            _record_seed_floor_arms=lambda *a: None,
            _record_seed_os_arms=lambda *a: seen.append(("record", a[0])),
            _fess_observe=lambda d, e: seen.append(("observe", d, e)),
            _current_edges_cache=edges,
        )
        r = FuzzRound(f, b"a")
        r._is_timeout = timeout
        r._credit_seed()
        return seen

    def test_round_feeds_exec_edges_before_record(self):
        edges = {5, 6}

        assert self._round(edges) == [("observe", b"a", edges), ("record", b"a")]

    def test_timeout_feeds_no_edges(self):
        assert self._round({5, 6}, timeout=True)[0] == ("observe", b"a", ())
