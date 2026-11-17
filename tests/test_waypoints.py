"""core/waypoints.WaypointRace and TargetDistance.waypoint_stages (--waypoints)."""

import math

import numpy as np
import pytest

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.waypoints import (
    BUDGET_CAP,
    BUDGET_START,
    MAX_STAGES,
    NO_STRATUM,
    WaypointRace,
    waypoint_stages,
)

NUM_RANDOM_CASES = 150

WINDOW = 4


def _mask(*bits: int) -> bytes:
    out = bytearray(2)
    for b in bits:
        out[b >> 3] |= 1 << (b & 7)
    return bytes(out)


def _window(race: WaypointRace, dist: float | None) -> None:
    """One full window whose minimum distance is *dist*."""
    for _ in range(WINDOW):
        race.note(dist)


# --- signature ---------------------------------------------------------------


def test_stratum_is_deepest_waypoint_bit():
    race = WaypointRace(4)
    assert race.stratum_of(_mask()) == NO_STRATUM
    assert race.stratum_of(_mask(0)) == 0
    assert race.stratum_of(_mask(0, 2)) == 2
    assert race.stratum_of(_mask(1, 9)) == 1  # bit past n_stages ignored


# --- trial order ----------------------------------------------------------------


def test_never_tried_strata_first_deepest_first():
    race = WaypointRace(4, window=WINDOW)
    race.admit(0)
    assert race.active == 0
    race.admit(2)  # deeper, untried: taken at the next trial boundary
    _window(race, 5.0)  # first window: shrinks from inf, trial goes on
    _window(race, 5.0)  # no shrink: trial over
    assert race.active == 2


def test_budget_doubles_while_min_shrinks_and_caps():
    race = WaypointRace(2, window=WINDOW)
    race.admit(1)
    assert race.multiplier(1) == BUDGET_START
    want = BUDGET_START
    d = 100.0
    for _ in range(int(math.log2(BUDGET_CAP)) + 3):
        d -= 1.0
        _window(race, d)
        want = min(want * 2, BUDGET_CAP)
        assert race.multiplier(1) == want
    assert race.multiplier(1) == BUDGET_CAP


def test_falsification_flat_distance_resets_budget():
    race = WaypointRace(2, window=WINDOW)
    race.admit(1)
    _window(race, 9.0)
    assert race.multiplier(1) == 2 * BUDGET_START
    _window(race, 9.0)  # equal is not shrinking
    assert race.multiplier(1) == BUDGET_START
    assert race.active == 1  # the only stratum is retried, not pruned


def test_inactive_stratum_gets_neutral_multiplier():
    race = WaypointRace(3, window=WINDOW)
    race.admit(2)
    assert race.multiplier(0) == 1.0
    assert race.multiplier(NO_STRATUM) == 1.0


def test_failed_stratum_deprioritized_not_pruned():
    race = WaypointRace(3, window=WINDOW)
    race.admit(1)
    race.admit(0)  # untried, queued behind the running trial
    assert race.active == 1
    _window(race, 5.0)  # 1 shrinks once (prog 1)
    _window(race, 5.0)  # 1 fails -> untried 0
    assert race.active == 0
    _window(race, None)  # 0: no data, fails without progress
    # Both tried: 1 scores (1 + eps) / 2 > 0's eps / 2.
    assert race.active == 1
    _window(race, 6.0)  # no new record (best 5): 1 now (1+eps)/3, 0 still eps/2
    assert race.active == 1


def test_new_seed_in_tried_stratum_doubles_its_score():
    race = WaypointRace(3, window=WINDOW)
    race.admit(1)
    race.admit(0)
    _window(race, 5.0)
    _window(race, 5.0)  # 1 fails with prog 1, 0 active
    race.admit(0)
    _window(race, 4.0)  # 0 shrinks: prog 1
    _window(race, 4.0)  # 0 fails: both prog 1, tries 1
    # Tie on score: the deeper stratum wins.
    assert race.active == 1
    race.admit(0)  # fresh seed in tried stratum 0
    _window(race, 4.0)  # 1 fails (tries 2)
    assert race.active == 0


def test_no_seeds_no_active():
    race = WaypointRace(3, window=WINDOW)
    _window(race, 1.0)
    assert race.active == NO_STRATUM
    assert race.multiplier(0) == 1.0


def test_admit_out_of_range_ignored():
    race = WaypointRace(2, window=WINDOW)
    race.admit(NO_STRATUM)
    race.admit(7)
    assert race.active == NO_STRATUM


def test_state_round_trip_and_mismatch_ignored():
    race = WaypointRace(3, window=WINDOW)
    race.admit(2)
    _window(race, 3.0)
    _window(race, 3.0)
    back = WaypointRace(3, window=WINDOW)
    back.load_state_dict(race.state_dict())
    assert back.state_dict() == race.state_dict()

    other = WaypointRace(5, window=WINDOW)
    other.load_state_dict(race.state_dict())  # different chain: fresh
    assert other.state_dict() == WaypointRace(5, window=WINDOW).state_dict()


def test_zero_stages_rejected():
    with pytest.raises(ValueError):
        WaypointRace(0)


# --- stages from the interprocedural dominator chain ------------------------


def _stages(n: int, edges, entry: int, targets: set[int]) -> list[frozenset[int]]:
    src = np.array([u for u, _ in edges], dtype=np.int64)
    dst = np.array([v for _, v in edges], dtype=np.int64)
    return waypoint_stages(n, src, dst, entry, targets)


def _brute_dominators(n: int, edges, entry: int, t: int) -> set[int]:
    """Blocks whose removal disconnects entry from t (plus entry and t)."""
    out = set()
    for v in range(n):
        if v in (entry, t):
            out.add(v)
            continue
        seen, todo = {entry}, [entry]
        while todo:
            x = todo.pop()
            for a, b in edges:
                if a == x and b != v and b not in seen:
                    seen.add(b)
                    todo.append(b)
        if t not in seen:
            out.add(v)
    return out


def test_stages_follow_dominator_chain():
    # 0 -> 1 -> {2, 3} -> 4 -> 5(target). Dominators of 5: 0, 1, 4.
    edges = [(0, 1), (1, 2), (1, 3), (2, 4), (3, 4), (4, 5)]
    want = [frozenset({0}), frozenset({1}), frozenset({4}), frozenset({5})]
    assert _stages(6, edges, 0, {5}) == want


def test_stages_cross_call_edges():
    # main 0 -> 1 calls f entry 3 (call edge 1->3); f: 3 -> {4, 5} -> 6(target).
    # Off-path block 2 and the sibling branch never dominate.
    edges = [(0, 1), (1, 2), (1, 3), (3, 4), (3, 5), (4, 6), (5, 6)]
    assert _stages(7, edges, 0, {6}) == [
        frozenset({0}),
        frozenset({1}),
        frozenset({3}),
        frozenset({6}),
    ]


def test_matches_brute_force_dominators_on_random():
    rng = RandPool(seed=21)
    checked = 0
    for _ in range(NUM_RANDOM_CASES):
        n = rng.randint(2, 8)
        edges = [(rng.randint(0, n - 1), rng.randint(0, n - 1)) for _ in range(rng.randint(1, 16))]
        t = rng.randint(1, n - 1)
        stages = _stages(n, edges, 0, {t})
        if not _reaches(n, edges, 0, t):
            assert stages == []
            continue
        checked += 1
        assert set().union(*stages) == _brute_dominators(n, edges, 0, t)
        assert stages[0] == frozenset({0}) and stages[-1] == frozenset({t})
    assert checked >= NUM_RANDOM_CASES // 4


def test_brute_force_dominators_control():
    # A/A control (Hard Rule 46): the oracle on a known diamond.
    edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
    assert _brute_dominators(4, edges, 0, 3) == {0, 3}


def _reaches(n, edges, a, b) -> bool:
    seen, todo = {a}, [a]
    while todo:
        x = todo.pop()
        for u, v in edges:
            if u == x and v not in seen:
                seen.add(v)
                todo.append(v)
    return b in seen


def test_multiple_targets_share_common_prefix():
    # 0 -> 1 -> {2, 3}; targets 2 and 3: common dominators 0, 1; last stage both.
    edges = [(0, 1), (1, 2), (1, 3)]
    assert _stages(4, edges, 0, {2, 3}) == [frozenset({0}), frozenset({1}), frozenset({2, 3})]


def test_unreachable_target_no_stages():
    assert _stages(3, [(0, 1)], 0, {2}) == []
    assert _stages(3, [(0, 1)], 0, set()) == []


def test_deep_chain_capped_at_max_stages_nearest_target():
    n = MAX_STAGES + 10
    edges = [(i, i + 1) for i in range(n - 1)]
    stages = _stages(n, edges, 0, {n - 1})
    assert len(stages) == MAX_STAGES
    assert stages[-1] == frozenset({n - 1})
    assert stages[0] == frozenset({n - MAX_STAGES})


def test_very_deep_chain_no_recursion_limit():
    n = 20_000
    stages = _stages(n, [(i, i + 1) for i in range(n - 1)], 0, {n - 1})
    assert stages[-1] == frozenset({n - 1})


# --- wiring ------------------------------------------------------------------

STUB = """\
static volatile int sink;
__attribute__((no_sanitize("coverage"))) void __sanitizer_cov_trace_pc(void) { sink++; }
__attribute__((noinline)) int target_fn(int a) {
    if (a > 3) { sink = a; if (a > 7) return a * 3; }
    return a + 1;
}
int main(int argc, char **argv) { if (argc > 2) return 0; return target_fn(argc); }
"""


@pytest.fixture(scope="module")
def stub_target(tmp_path_factory):
    import shutil
    import subprocess

    if not shutil.which("clang"):
        pytest.skip("clang not available")
    d = tmp_path_factory.mktemp("waypoints")
    (d / "wp.c").write_text(STUB)
    # Compile and link apart: linking with -fsanitize-coverage pulls a runtime.
    cc = ["clang", "-O0", "-g", "-fsanitize-coverage=trace-pc", "-c", "-o", str(d / "wp.o")]
    assert subprocess.run([*cc, str(d / "wp.c")], capture_output=True).returncode == 0
    r = subprocess.run(["clang", "-o", str(d / "wp"), str(d / "wp.o")], capture_output=True)
    assert r.returncode == 0, r.stderr.decode()
    return str(d / "wp")


def test_plan_resolves_main_to_target_sites(stub_target):
    from fuzzer_tool.core.analyzers.analyzer_distance import TargetDistance
    from fuzzer_tool.core.waypoints import plan_waypoints

    td = TargetDistance(stub_target, targets=["target_fn"])
    assert td.load()
    plan = plan_waypoints(td)
    assert plan is not None and plan.n_stages >= 2
    stages = set(plan.node_of.values())
    assert 0 in stages  # main's entry block
    assert plan.n_stages - 1 in stages  # the target function's entry
    assert set(plan.dist) == set(plan.node_of)


def test_plan_none_for_unknown_target(stub_target):
    from fuzzer_tool.core.analyzers.analyzer_distance import TargetDistance
    from fuzzer_tool.core.waypoints import plan_waypoints

    td = TargetDistance(stub_target, targets=["no_such_fn"])
    td.load()
    assert plan_waypoints(td) is None


def _round(stratum_bits: bytes, distance: float | None):
    from types import SimpleNamespace

    from fuzzer_tool.services.fuzz_round import FuzzRound

    race = WaypointRace(3, window=WINDOW)
    bmp = SimpleNamespace(read_and_clear=lambda: stratum_bits)
    meta: dict = {}
    f = SimpleNamespace(_waypoints=race, _waypoint_bmp=bmp, seed_meta={b"m": meta})
    r = FuzzRound.__new__(FuzzRound)
    r._f, r._mutated, r._distance, r._stratum = f, b"m", distance, None
    return r, race, meta


def test_round_tags_admitted_mutant_with_stratum():
    r, race, meta = _round(_mask(0, 2), 4.0)
    r._note_waypoint()
    r._tag_distance()
    assert meta == {"avg_distance": 4.0, "waypoint": 2}
    assert race.active == 2


def test_round_no_waypoint_bit_is_no_stratum():
    r, race, meta = _round(_mask(), 4.0)
    r._note_waypoint()
    r._tag_distance()
    assert meta["waypoint"] == NO_STRATUM
    assert race.active == NO_STRATUM


def test_plan_off_unless_requested():
    from types import SimpleNamespace

    from fuzzer_tool.core.analyzer_registry import _plan_waypoints

    assert _plan_waypoints(SimpleNamespace(_waypoints_requested=False)) is None


def test_energy_multiplier_and_state_wired_in_fuzzer():
    # Fuzzer() needs a live target; the hooks are checked in source.
    import inspect

    from fuzzer_tool.services import fuzzer

    src = inspect.getsource(fuzzer)
    assert 'self._waypoints.multiplier(meta.get("waypoint", NO_STRATUM))' in src
    assert 'self._state_store.set("waypoints", self._waypoints.state_dict())' in src
    assert 'self._waypoints.load_state_dict(self._state_store.get("waypoints", {}))' in src
