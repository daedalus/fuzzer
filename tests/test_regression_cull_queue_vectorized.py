"""Vectorized ``_cull_queue`` keeps the per-edge loop's favored set.

Reference: the loop ``Fuzzer._cull_queue`` ran before vectorizing (copied
below). ffmpeg --lightweight spent 77 s of 459 s in it (45 calls).
"""

import tempfile
from unittest.mock import patch

from fuzzer_tool.core.cost_ledger import seed_exec_us
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.services.fuzzer import Fuzzer

SEEDS = 60
EDGE_SPACE = 400


def _make_fuzzer() -> Fuzzer:
    tmpdir = tempfile.mkdtemp(prefix="fuzz_test_")
    with (
        patch("os.path.isfile", return_value=True),
        patch("os.access", return_value=True),
    ):
        return Fuzzer(
            target="/bin/true",
            corpus_dir=f"{tmpdir}/corpus",
            crashes_dir=f"{tmpdir}/crashes",
            max_len=256,
            timeout=1,
            mutations_per_input=2,
        )


def _reference_favored(f: Fuzzer) -> set[str]:
    # The pre-vectorization loop, verbatim in behaviour.
    top_rated: dict[int, tuple[str, float]] = {}
    mean_us = f.mean_exec_time() * 1_000_000
    for key, edges in f._edge_tracker.seed_edges.items():
        m = f.seed_meta.get(key) or {}
        cost = seed_exec_us(m, mean_us) * max(1, m.get("input_size", 1))
        for e in edges:
            cur = top_rated.get(e)
            if cur is None or cost < cur[1]:
                top_rated[e] = (key, cost)

    et = f._edge_tracker
    covered: set[int] = set()
    favored: set[str] = set()
    for e in sorted(top_rated, key=lambda e: (et.edge_owner_count(e), e)):
        if e in covered:
            continue
        k = top_rated[e][0]
        favored.add(k)
        covered |= et.seed_edges[k]
    return favored


def _populate(f: Fuzzer, seed: int, sizes: int) -> None:
    # *sizes* distinct input sizes: few -> many cost ties.
    rng = RandPool(seed=seed)
    for i in range(SEEDS):
        n = rng.randint(0, 40)
        edges = set(rng.randint_list(1, EDGE_SPACE, n)) if n else set()
        key = f"s{i}"
        if edges:
            f._edge_tracker.record_edges(key, edges)
        f._edge_tracker.seed_edges.setdefault(key, edges)
        f.seed_meta[key] = {
            "total_time": 0.001 * rng.randint(1, 3),
            "cost_samples": rng.randint(0, 2),
            "fuzz_count": 1,
            "input_size": rng.randint(1, sizes),
        }


def _favored_of(seed: int, sizes: int) -> tuple[set[str], set[str]]:
    f = _make_fuzzer()
    _populate(f, seed, sizes)
    f._cull_queue()
    return f._favored, _reference_favored(f)


def test_regression_cull_queue_control_matches_itself():
    # Control (Hard Rule 46): the reference agrees with a second run of itself.
    f = _make_fuzzer()
    _populate(f, 7, 50)
    assert _reference_favored(f) == _reference_favored(f)


def test_regression_cull_queue_matches_loop():
    for seed in range(8):
        got, want = _favored_of(seed, 50)
        assert got == want, seed


def test_regression_cull_queue_cost_ties_first_seed_wins():
    # Adversarial: one input size and time class -> heavy ties; the loop's
    # strict `<` keeps the earliest seed in iteration order.
    for seed in range(8):
        got, want = _favored_of(seed, 1)
        assert got == want, seed


def test_regression_cull_queue_untracked_owners():
    # Falsification: a hand-built seed_edges with no owner counts (every
    # edge reads 0) orders edges by id alone.
    f = _make_fuzzer()
    f._edge_tracker.seed_edges = {"a": {5, 1}, "b": {1, 2}, "c": set(), "d": {9}}
    f.seed_meta = {k: {"input_size": 1} for k in "abcd"}
    f._cull_queue()
    assert f._favored == _reference_favored(f) == {"a", "b", "d"}


def test_regression_cull_queue_all_empty_sets():
    f = _make_fuzzer()
    f._edge_tracker.seed_edges = {"a": set(), "b": set()}
    f._cull_queue()
    assert f._favored == set()
