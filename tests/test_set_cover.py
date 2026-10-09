"""core/set_cover.min_cover and its wiring into minimize / corpus_manager."""

from itertools import combinations

import pytest

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.set_cover import Reduce, Tie, _Cover, _to_masks, min_cover

NUM_RANDOM_CASES = 40
MAX_SEEDS = 12
MAX_EDGES = 30
BRUTE_FORCE_MAX_SEEDS = 10

# A greedy trap: A has the biggest gain, but B and C are the only
# coverers of edges 5 and 6, so every cover contains B and C, and A is
# redundant once they are in. Plain greedy returns [A, B, C].
TRAP = {"A": {1, 2, 3, 4}, "B": {1, 2, 5}, "C": {3, 4, 6}}
TRAP_SIZES = {"A": 10, "B": 10, "C": 10}


# Size tie-break + reductions alone return 3 seeds here; first-wins greedy
# plus the redundancy pass returns 2. Found by search; guards the multi-start in min_cover.
TIE_TRAP = {
    "s0": {2, 3, 4, 5, 6},
    "s1": {0, 3, 4, 5, 6},
    "s2": {0, 2, 4},
    "s3": {2, 4, 5, 7},
    "s4": {3, 4, 5, 6, 7},
}
TIE_TRAP_SIZES = {"s0": 2, "s1": 8, "s2": 2, "s3": 2, "s4": 2}


def _cover_run(seed_edges, sizes, tie, reduce):
    order = {k: i for i, k in enumerate(seed_edges)}
    full = {k: frozenset(e) for k, e in seed_edges.items() if e}
    return _Cover(order, _to_masks(full), sizes, tie, reduce).solve()


def _ref_greedy(seed_edges):
    """Verbatim old rule: max new edges, first wins ties, no size, no reductions."""
    covered, chosen, remaining = set(), [], list(seed_edges)
    while remaining:
        best, best_new = None, 0
        for key in remaining:
            new = len(seed_edges[key] - covered)
            if new > best_new:
                best, best_new = key, new
        if best is None:
            break
        chosen.append(best)
        covered |= seed_edges[best]
        remaining.remove(best)
    return chosen


def _random_instance(rp):
    n_seeds = rp.randint(1, MAX_SEEDS)
    n_edges = rp.randint(1, MAX_EDGES)
    seed_edges = {}
    for i in range(n_seeds):
        k = rp.randint(1, min(8, n_edges))
        seed_edges[f"s{i}"] = set(rp.sample(range(n_edges), k))
    sizes = {k: rp.randint(1, 500) for k in seed_edges}
    return seed_edges, sizes


def _covered(seed_edges, keys):
    return set().union(*(seed_edges[k] for k in keys)) if keys else set()


def _coverable(seed_edges):
    return set().union(*seed_edges.values()) if seed_edges else set()


def _cases():
    rp = RandPool(seed=1234)
    return [_random_instance(rp) for _ in range(NUM_RANDOM_CASES)]


class TestMinCover:
    def test_trap_returns_forced_pair_only(self):
        assert sorted(min_cover(TRAP, TRAP_SIZES)) == ["B", "C"]

    def test_control_reference_greedy_falls_in_trap(self):
        # The oracle must be wrong on the trap, and identical to itself.
        assert _ref_greedy(TRAP) == _ref_greedy(TRAP)
        assert len(_ref_greedy(TRAP)) == 3

    def test_size_tiebreak_alone_loses_on_tie_trap(self):
        alone = _cover_run(TIE_TRAP, TIE_TRAP_SIZES, Tie.SIZE, Reduce.ON)
        assert len(alone) == 3
        plain = _cover_run(TIE_TRAP, TIE_TRAP_SIZES, Tie.ORDER, Reduce.OFF)
        assert len(plain) == 2

    def test_multi_start_recovers_tie_trap(self):
        assert len(min_cover(TIE_TRAP, TIE_TRAP_SIZES)) == 2

    def test_equal_gain_prefers_smaller_file(self):
        seed_edges = {"big": {1, 2}, "small": {1, 2}}
        assert min_cover(seed_edges, {"big": 100, "small": 10}) == ["small"]

    def test_size_tiebreak_keeps_count(self):
        seed_edges = {"a": {1, 2, 3}, "b": {1, 2, 3}, "c": {4, 5}, "d": {4, 5}}
        out = min_cover(seed_edges, {"a": 90, "b": 5, "c": 7, "d": 70})
        assert sorted(out) == ["b", "c"]

    def test_covers_every_coverable_edge(self):
        for seed_edges, sizes in _cases():
            out = min_cover(seed_edges, sizes)
            assert _covered(seed_edges, out) == _coverable(seed_edges)

    def test_result_is_irredundant(self):
        for seed_edges, sizes in _cases():
            out = min_cover(seed_edges, sizes)
            for dropped in out:
                rest = [k for k in out if k != dropped]
                assert _covered(seed_edges, rest) != _coverable(seed_edges)

    def test_never_more_seeds_than_plain_greedy(self):
        for seed_edges, sizes in _cases():
            assert len(min_cover(seed_edges, sizes)) <= len(_ref_greedy(seed_edges))

    def test_never_below_brute_force_optimum(self):
        for seed_edges, sizes in _cases():
            if len(seed_edges) > BRUTE_FORCE_MAX_SEEDS:
                continue
            target = _coverable(seed_edges)
            opt = next(
                r
                for r in range(1, len(seed_edges) + 1)
                if any(_covered(seed_edges, c) == target for c in combinations(seed_edges, r))
            )
            assert opt <= len(min_cover(seed_edges, sizes))

    def test_deterministic(self):
        for seed_edges, sizes in _cases():
            assert min_cover(seed_edges, sizes) == min_cover(seed_edges, sizes)

    def test_empty_inputs(self):
        assert min_cover({}, {}) == []
        assert min_cover({"a": set(), "b": set()}, {"a": 1, "b": 1}) == []

    def test_empty_edge_seed_is_ignored(self):
        assert min_cover({"a": set(), "b": {1}}, {"a": 1, "b": 1}) == ["b"]

    def test_missing_size_fails_loudly(self):
        with pytest.raises(KeyError):
            min_cover({"a": {1}}, {})

    def test_zero_size_is_accepted(self):
        assert min_cover({"a": {1}, "b": {1}}, {"a": 0, "b": 3}) == ["a"]


class TestAdversarial:
    def test_all_unique_edges_all_forced(self):
        n = 5000
        seed_edges = {i: {i} for i in range(n)}
        assert sorted(min_cover(seed_edges, dict.fromkeys(seed_edges, 1))) == list(range(n))

    def test_many_identical_seeds_collapse_to_smallest(self):
        n = 20000
        seed_edges = {i: {1, 2, 3} for i in range(n)}
        sizes = {i: 100 + i for i in range(n)}
        sizes[777] = 1
        assert min_cover(seed_edges, sizes) == [777]

    def test_nested_chain_picks_the_top(self):
        n = 300
        seed_edges = {i: set(range(i + 1)) for i in range(n)}
        assert min_cover(seed_edges, dict.fromkeys(seed_edges, 1)) == [n - 1]

    def test_one_edge_shared_by_all_plus_private_edges(self):
        n = 200
        seed_edges = {i: {-1, i} for i in range(n)}
        assert sorted(min_cover(seed_edges, dict.fromkeys(seed_edges, 1))) == list(range(n))

    def test_forced_seed_makes_larger_seed_redundant(self):
        # F is the only coverer of edge 9; G is a superset of F's other edges.
        seed_edges = {"F": {1, 9}, "G": {1, 2, 3}, "H": {2, 3}}
        out = min_cover(seed_edges, dict.fromkeys(seed_edges, 10))
        assert sorted(out) == ["F", "G"] or sorted(out) == ["F", "H"]
        assert len(out) == 2


class _FakeShm:
    """Stands in for ShmCoverage: edges come from the data last 'run'."""

    current: bytes = b""
    table: dict = {}

    def __init__(self):
        self.env_id = "0"
        self.num_entries = 16

    def reset_edge_map(self):
        pass

    def get_edge_ids(self):
        return set(type(self).table[type(self).current])

    def cleanup(self):
        pass


def _fake_run(target, data, timeout, env=None, **_):
    _FakeShm.current = data
    return 0, "", 0


class TestMinimizeWiring:
    def _run(self, tmp_path, monkeypatch, files):
        from fuzzer_tool.adapters import process
        from fuzzer_tool.services import minimize

        corpus = tmp_path / "corpus"
        corpus.mkdir()
        paths = []
        _FakeShm.table = {}
        for name, data, edges in files:
            p = corpus / name
            p.write_bytes(data)
            _FakeShm.table[data] = edges
            paths.append(p)
        monkeypatch.setattr(minimize, "ShmCoverage", _FakeShm)
        monkeypatch.setattr(process, "run_target_stdin", _fake_run)
        minimize._minimize_with_coverage(paths, "t", 1.0, False, None, None, corpus)
        return sorted(p.name for p in corpus.iterdir() if p.is_file())

    def test_trap_corpus_keeps_forced_pair(self, tmp_path, monkeypatch):
        files = [
            ("a.bin", b"A" * 40, {1, 2, 3, 4}),
            ("b.bin", b"B" * 40, {1, 2, 5}),
            ("c.bin", b"C" * 40, {3, 4, 6}),
        ]
        assert self._run(tmp_path, monkeypatch, files) == ["b.bin", "c.bin"]

    def test_equal_coverage_keeps_smaller_file(self, tmp_path, monkeypatch):
        files = [
            ("big.bin", b"x" * 500, {1, 2}),
            ("small.bin", b"y" * 5, {1, 2}),
        ]
        assert self._run(tmp_path, monkeypatch, files) == ["small.bin"]


class TestCorpusManagerWiring:
    def test_greedy_cover_uses_forced_reduction(self):
        from fuzzer_tool.services.corpus_manager import CorpusManager

        a, b, c = b"A" * 40, b"B" * 40, b"C" * 40
        edge_map = {id(a): TRAP["A"], id(b): TRAP["B"], id(c): TRAP["C"]}
        out = CorpusManager._greedy_cover(object(), [a, b, c], edge_map)
        assert out == {id(b), id(c)}

    def test_greedy_cover_prefers_smaller_on_ties(self):
        from fuzzer_tool.services.corpus_manager import CorpusManager

        big, small = b"x" * 500, b"y" * 5
        edge_map = {id(big): {1, 2}, id(small): {1, 2}}
        out = CorpusManager._greedy_cover(object(), [big, small], edge_map)
        assert out == {id(small)}

    def test_greedy_cover_empty(self):
        from fuzzer_tool.services.corpus_manager import CorpusManager

        assert CorpusManager._greedy_cover(object(), [], {}) == set()
