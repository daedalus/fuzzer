"""core/schedulers/seed_branch.BranchFrontier and its 'branch' seed-arm wiring."""

from types import SimpleNamespace

import numpy as np
import pytest

from fuzzer_tool.core.icfg import InterproceduralCFG
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.seed_branch import GAMMA, MAX_HOPS, BranchFrontier
from fuzzer_tool.services.katz_channel import KatzChannel
from tests.support.scripted_rng import ScriptedRng
from tests.test_line_graph import _brute_dist, _floyd_dist

NUM_RANDOM_CASES = 60


def _icfg(n: int, edges: list[tuple[int, int]]) -> InterproceduralCFG:
    src = np.array([u for u, _ in edges], dtype=np.int64)
    dst = np.array([v for _, v in edges], dtype=np.int64)
    return InterproceduralCFG([0x100 * i for i in range(n)], ["f"] * n, src, dst, {})


def _mask(n: int, nodes) -> bytes:
    bits = np.zeros(n, dtype=bool)
    bits[list(nodes)] = True
    return np.packbits(bits, bitorder="little").tobytes()


def _visited(n: int, nodes) -> np.ndarray:
    v = np.zeros(n, dtype=bool)
    v[list(nodes)] = True
    return v


# Chain 0->1->2->3 with branches 1->4 and 3->5. Visited {0,1,2,3}: frontier
# branches are 1->4 (edge 3) and 3->5 (edge 4).
CHAIN = [(0, 1), (1, 2), (2, 3), (1, 4), (3, 5)]


def test_least_tried_branch_is_the_target():
    bf = BranchFrontier(_icfg(6, CHAIN))
    hits = np.array([9, 9, 9, 1, 0, 0], dtype=np.float64)  # block 3 rarely run
    bf.refresh(_visited(6, {0, 1, 2, 3}), hits)
    assert bf.targets.tolist() == [4]  # 3->5: tail 3 has the fewest hits


def test_energy_decays_with_line_graph_hops():
    bf = BranchFrontier(_icfg(6, CHAIN))
    bf.refresh(_visited(6, {0, 1, 2, 3}), np.array([9, 9, 9, 1, 0, 0], dtype=np.float64))
    # Target edge 4 (3->5). Block 3 is its tail: d 0. Block 2: edge 2->3 then
    # 3->5, d 1. Block 0: 0->1->2->3->5, d 3.
    assert bf.energy(_mask(6, {3})) == 1.0
    assert bf.energy(_mask(6, {2})) == GAMMA
    assert bf.energy(_mask(6, {0})) == GAMMA**3
    assert bf.energy(_mask(6, {0, 2})) == GAMMA  # nearest block wins


def test_falsification_unrelated_seed_scores_zero():
    # Block 5 has no out-edge: it reaches no target.
    bf = BranchFrontier(_icfg(6, CHAIN))
    bf.refresh(_visited(6, {0, 1, 2, 3}), np.array([9, 9, 9, 1, 0, 0], dtype=np.float64))
    assert bf.energy(_mask(6, {5})) == 0.0


def test_matches_brute_force_on_random():
    rng = RandPool(seed=99)
    for _ in range(NUM_RANDOM_CASES):
        n = rng.randint(2, 8)
        edges = [(rng.randint(0, n - 1), rng.randint(0, n - 1)) for _ in range(rng.randint(1, 14))]
        visited = {v for v in range(n) if rng.randint(0, 1)}
        hits = np.array([float(rng.randint(0, 5)) for _ in range(n)])
        bf = BranchFrontier(_icfg(n, edges))
        bf.refresh(_visited(n, visited), hits)
        if not bf.ready:
            continue

        # Oracles agree before judging (control, Hard Rule 46).
        src = [u for u, _ in edges]
        dst = [v for _, v in edges]
        targets = set(bf.targets.tolist())
        dist = _brute_dist(src, dst, targets)
        assert dist == _floyd_dist(n, src, dst, targets)

        seed = {v for v in range(n) if rng.randint(0, 1)}
        best = min(
            (dist[e] for e in range(len(edges)) if src[e] in seed and 0 <= dist[e] <= MAX_HOPS),
            default=None,
        )
        want = 0.0 if best is None else GAMMA**best
        assert bf.energy(_mask(n, seed)) == pytest.approx(want)


# --- adversarial ---------------------------------------------------------


def test_no_frontier_is_not_ready():
    bf = BranchFrontier(_icfg(3, [(0, 1), (1, 2)]))
    bf.refresh(_visited(3, {0, 1, 2}), np.zeros(3))
    assert not bf.ready
    assert bf.energy(_mask(3, {0})) == 0.0


def test_before_refresh_scores_zero():
    assert BranchFrontier(_icfg(3, [(0, 1)])).energy(_mask(3, {0})) == 0.0


def test_beyond_max_hops_scores_zero():
    n = MAX_HOPS + 3
    edges = [(i, i + 1) for i in range(n - 1)]
    bf = BranchFrontier(_icfg(n, edges))
    bf.refresh(_visited(n, range(n - 1)), np.zeros(n))
    assert bf.targets.tolist() == [n - 2]
    assert bf.energy(_mask(n, {n - 2 - MAX_HOPS})) == GAMMA**MAX_HOPS
    assert bf.energy(_mask(n, {n - 3 - MAX_HOPS})) == 0.0


def test_short_mask_raises():
    bf = BranchFrontier(_icfg(16, [(0, 1)]))
    bf.refresh(_visited(16, {0}), np.zeros(16))
    with pytest.raises(ValueError):
        bf.energy(b"\x01")


# --- KatzChannel wiring ----------------------------------------------------


def _channel() -> KatzChannel:
    ch = KatzChannel(_icfg(6, CHAIN), {k: k for k in range(6)})
    bits = np.zeros(6, dtype=bool)
    bits[[0, 1, 2]] = True
    ch.record(bits, seed_key="near")
    for _ in range(4):
        ch.record(bits, seed_key=None)  # blocks 0-2 run often
    bits[:] = False
    bits[[0, 1, 2, 3]] = True
    ch.record(bits, seed_key="tail")
    return ch


def test_channel_branch_off_by_default():
    ch = _channel()
    assert not ch.branch_ready
    assert ch.branch_energy("tail") == 0.0


def test_channel_branch_energy_after_rebuild():
    ch = _channel()
    ch.enable_branch()
    ch.ensure_scores(force=True)
    assert ch.branch_ready
    assert ch.branch_energy("tail") == 1.0  # covers block 3, tail of 3->5
    assert ch.branch_energy("near") == GAMMA  # block 2, one hop from 3->5
    assert ch.branch_energy("unknown") == 0.0


# --- seed picker wiring ----------------------------------------------------


def _fuzzer(energy: dict[bytes, float], ready=True):
    ch = SimpleNamespace(branch_ready=ready, branch_energy=lambda k: energy[k])
    return SimpleNamespace(corpus=list(energy), _seed_key=lambda s: s, _katz_channel=ch)


def test_branch_arm_listed_only_when_ready():
    from fuzzer_tool.services.seed_picker import SeedPicker

    avail: list[str] = []
    SeedPicker._elo_flag_arms(_fuzzer({b"a": 1.0}), avail)
    assert "branch" in avail
    avail = []
    SeedPicker._elo_flag_arms(_fuzzer({b"a": 1.0}, ready=False), avail)
    assert "branch" not in avail


def test_branch_pick_is_energy_roulette():
    from fuzzer_tool.services.seed_picker import SeedPicker

    f = _fuzzer({b"a": 0.0, b"b": 1.0, b"c": 0.5})
    f._rng = ScriptedRng(randoms=[0.6])
    # Weights: a floor, b 1, c 0.5; 0.6 of the total lands in b.
    assert SeedPicker(f)._pick_branch_seed() == b"b"


def test_branch_pick_declines_when_all_zero():
    from fuzzer_tool.services.seed_picker import SeedPicker

    f = _fuzzer({b"a": 0.0, b"b": 0.0})
    f._rng = ScriptedRng()
    assert SeedPicker(f)._pick_branch_seed() is None


def test_strategy_name_registered():
    from fuzzer_tool.services.fuzzer import _SEED_STRATEGY_NAMES

    assert "branch" in _SEED_STRATEGY_NAMES


def test_cli_flag_registered(monkeypatch, capsys):
    import sys

    from fuzzer_tool.cli import commands

    monkeypatch.setattr(sys, "argv", ["fuzzer-tool", "fuzz", "--help"])
    with pytest.raises(SystemExit):
        commands.main()
    assert "--seed-branch-scheduler" in capsys.readouterr().out
