"""Tests for tools/sim_funnel_probe.py.

Linear probe placement is checked against ``adapters/shm.displacements`` (the
shim mirror), never against the simulator's own bookkeeping.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from fuzzer_tool.adapters.shm import displacements, home_slot
from fuzzer_tool.core.rand_pool import RandPool

TOOLS = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(TOOLS))

import sim_funnel_probe as sim  # noqa: E402

SIZE = 8192
WINDOW = 64
DELTA = 1 / 8
SEED = 7


def _ids_homed_at(slot: int, size: int, count: int) -> list[int]:
    """First *count* ids (1, 2, ...) whose home slot is *slot*: deterministic."""
    out = []
    eid = 1
    while len(out) < count:
        if home_slot(eid, size) == slot:
            out.append(eid)
        eid += 1
    return out


def _fill(table, load: float) -> list[int]:
    ids = sim.unique_ids(RandPool(seed=SEED), int(load * table.size))
    for eid in ids:
        table.insert(eid)
    return ids


class TestLinear:
    def test_hit_probes_match_shm_displacement(self):
        table = sim.LinearTable(SIZE, WINDOW)
        ids = [e for e in _fill(table, 0.85) if table.lookup(e).found]
        slots = np.array([table.slot_of(e) for e in ids], dtype=np.uint32)
        expect = displacements(slots, np.array(ids, dtype=np.uint32), SIZE) + 1
        got = np.array([table.lookup(e).probes for e in ids])
        assert np.array_equal(got, expect)

    def test_full_window_drops(self):
        size, window = 16, 4
        table = sim.LinearTable(size, window)
        ids = _ids_homed_at(0, size, window + 1)
        placed = [table.insert(e) for e in ids]
        assert placed[:window] == [sim.Insert.PLACED] * window
        assert placed[window] is sim.Insert.DROPPED

    def test_probe_crossing_line_touches_two_lines(self):
        size = 16
        last = next(s for s in range(size - 1) if sim.line_of(s) != sim.line_of(s + 1))
        table = sim.LinearTable(size, WINDOW)
        first, second = _ids_homed_at(last, size, 2)
        table.insert(first)
        table.insert(second)
        hit = table.lookup(second)
        assert (hit.probes, hit.lines) == (2, 2)


class TestFunnel:
    def test_levels_partition_table(self):
        table = sim.FunnelTable(SIZE, DELTA)
        assert table.layout_size() == SIZE

    def test_search_cost_equals_insert_cost(self):
        # Greedy, no reordering: a key's search probes equal its insert probes.
        table = sim.FunnelTable(SIZE, DELTA)
        ids = sim.unique_ids(RandPool(seed=SEED), int(0.85 * SIZE))
        costs = {e: table.insert_probes(e) for e in ids}
        for eid, cost in costs.items():
            if cost is None:
                continue
            assert table.lookup(eid).probes == cost

    @pytest.mark.parametrize("delta", [0.25, 0.1, 0.0])
    def test_rejects_bad_delta(self, delta):
        with pytest.raises(ValueError):
            sim.FunnelTable(SIZE, delta)


class TestAdversarial:
    @pytest.mark.parametrize("kind", [sim.Kind.LINEAR, sim.Kind.FUNNEL])
    def test_overfull_table_terminates(self, kind):
        size = 256
        table = sim.make_table(kind, size, WINDOW, DELTA)
        ids = sim.unique_ids(RandPool(seed=SEED), 2 * size)
        outcomes = [table.insert(e) for e in ids]
        assert outcomes.count(sim.Insert.PLACED) <= size
        assert sim.Insert.DROPPED in outcomes

        absent = sim.unique_ids(RandPool(seed=SEED + 1), 64, exclude=set(ids))
        assert not any(table.lookup(e).found for e in absent)

    def test_unique_ids_excludes_and_is_distinct(self):
        taken = set(range(1, 1000))
        ids = sim.unique_ids(RandPool(seed=SEED), 500, exclude=taken)
        assert len(set(ids)) == 500
        assert not taken & set(ids)
        assert 0 not in ids


class TestControl:
    def test_same_seed_same_stats(self):
        # A/A: identical configs must report identical stats, else the harness leaks state.
        args = (sim.Kind.FUNNEL, SIZE // 8, WINDOW, DELTA, 0.8, 200)
        a = sim.run_trial(*args, RandPool(seed=SEED))
        b = sim.run_trial(*args, RandPool(seed=SEED))
        assert a == b

    @pytest.mark.parametrize("kind", [sim.Kind.LINEAR, sim.Kind.FUNNEL])
    def test_every_placed_key_is_found(self, kind):
        table = sim.make_table(kind, SIZE, WINDOW, DELTA)
        ids = sim.unique_ids(RandPool(seed=SEED), int(0.9 * SIZE))
        placed = [e for e in ids if table.insert(e) is sim.Insert.PLACED]
        assert all(table.lookup(e).found for e in placed)
