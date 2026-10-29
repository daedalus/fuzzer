"""core/sdr.distinct_reps (Hall / Hopcroft-Karp) and its ``minimize --hall`` wiring."""

from itertools import combinations

import numpy as np
import pytest

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.sdr import distinct_reps, reps_report

NUM_RANDOM_CASES = 40
MAX_SEEDS = 7
MAX_EDGES = 9
CHAIN_LEN = 5000


def _u32(*ids: int) -> np.ndarray:
    return np.array(sorted(ids), dtype=np.uint32)


def _book(spec: dict[str, set[int]]) -> dict[str, np.ndarray]:
    return {k: _u32(*v) for k, v in spec.items()}


def _owners(book: dict[str, np.ndarray]) -> dict[int, set[str]]:
    out: dict[int, set[str]] = {}
    for k, e in book.items():
        for x in e.tolist():
            out.setdefault(x, set()).add(k)
    return out


def _deficiency(adj: dict[int, set[str]]) -> int:
    """Kőnig-Ore: max over edge subsets S of |S| - |N(S)|, by brute force."""
    lefts = list(adj)
    best = 0
    for r in range(1, len(lefts) + 1):
        for sub in combinations(lefts, r):
            nbrs = set().union(*(adj[e] for e in sub))
            best = max(best, r - len(nbrs))
    return best


def _check(book: dict[str, np.ndarray], rare_owners: int) -> None:
    owners = _owners(book)
    rare = {e: s for e, s in owners.items() if len(s) <= rare_owners}
    reps = distinct_reps(book, rare_owners=rare_owners)

    # Valid and distinct: each matched edge goes to a seed that covers it.
    assert len(set(reps.owner.values())) == len(reps.owner)
    for e, k in reps.owner.items():
        assert k in rare[e]

    # Maximum: unmatched count equals the Hall deficiency.
    assert reps.n_rare == len(rare)
    assert set(reps.owner) | set(reps.deficient) == set(rare)
    assert len(reps.deficient) == _deficiency(rare)

    # Witness: N(S) is exactly the witness seeds, |S| - |N(S)| = deficiency.
    nbrs = set().union(*(rare[e] for e in reps.violator)) if reps.violator else set()
    assert nbrs == set(reps.witness)
    assert len(reps.violator) - len(reps.witness) == len(reps.deficient)


def test_oracle_control_on_hand_derived_instance():
    # Edges 1, 2 have only seed a; the oracle must see deficiency 1.
    adj = {1: {"a"}, 2: {"a"}, 3: {"a", "b"}}
    assert _deficiency(adj) == 1
    assert _deficiency({1: {"a"}, 2: {"b"}}) == 0


def test_perfect_matching_needs_augmenting():
    # Greedy edge 1 -> a strands edge 2; the matching must swap to 1 -> b.
    book = _book({"a": {1, 2}, "b": {1}})
    reps = distinct_reps(book)
    assert reps.owner == {1: "b", 2: "a"}
    assert reps.deficient == ()


def test_hall_violation_reports_witness():
    book = _book({"a": {1, 2, 3}, "b": {3}})
    reps = distinct_reps(book)
    assert len(reps.deficient) == 1
    assert set(reps.violator) == {1, 2}
    assert reps.witness == ("a",)
    _check(book, 3)


def test_common_edges_are_not_rare():
    book = _book({"a": {1, 9}, "b": {9}, "c": {9}, "d": {9}})
    reps = distinct_reps(book, rare_owners=3)
    assert set(reps.owner) == {1}
    assert reps.n_rare == 1


def test_prefer_order_picks_kept_seed():
    book = _book({"a": {1}, "b": {1}})
    assert distinct_reps(book, prefer=["b"]).owner == {1: "b"}
    assert distinct_reps(book, prefer=["a"]).owner == {1: "a"}


def test_matches_brute_force_on_random_books():
    rng = RandPool(seed=1234)
    for _ in range(NUM_RANDOM_CASES):
        n_seeds = rng.randint(1, MAX_SEEDS)
        book = {
            f"s{i}": _u32(*{rng.randint(0, MAX_EDGES) for _ in range(rng.randint(0, 4))})
            for i in range(n_seeds)
        }
        _check(book, rare_owners=rng.randint(1, 3))


def test_deterministic():
    book = _book({"a": {1, 2}, "b": {2, 3}, "c": {3, 1}})
    assert distinct_reps(book) == distinct_reps(book)


# --- adversarial ---------------------------------------------------------


def test_empty_book_and_edgeless_seeds():
    assert distinct_reps({}).n_rare == 0
    reps = distinct_reps({"a": _u32(), "b": _u32()})
    assert reps.owner == {} and reps.deficient == () and reps.violator == ()


def test_long_augmenting_chain_does_not_recurse():
    # Edge i is held by seeds i and i+1; the last edge only by seed n.
    # Matching edge i -> i+1 first leaves an augmenting path of length n.
    book: dict[str, set[int]] = {f"s{i:05d}": set() for i in range(CHAIN_LEN + 1)}
    for i in range(CHAIN_LEN):
        book[f"s{i:05d}"].add(i)
        book[f"s{i + 1:05d}"].add(i)
    book[f"s{CHAIN_LEN:05d}"].add(CHAIN_LEN)
    reps = distinct_reps(_book(book), rare_owners=2)
    assert len(reps.owner) == CHAIN_LEN + 1
    assert reps.deficient == ()


def test_duplicate_ids_in_a_seed_count_once():
    book = {"a": np.array([1, 1, 1], dtype=np.uint32), "b": _u32(1), "c": _u32(1)}
    assert distinct_reps(book, rare_owners=3).n_rare == 1


def test_report_lines():
    text = reps_report(distinct_reps(_book({"a": {1, 2}, "b": {3}})), added=1)
    for label in (
        "Distinct Representatives",
        "Rare edges:",
        "Matched:",
        "Deficiency:",
        "Seeds added:",
    ):
        assert label in text
    assert "No rare edges" in reps_report(distinct_reps({}), added=0)


# --- wiring --------------------------------------------------------------


class _FakeShm:
    """Stands in for ShmCoverage: edges come from the data last 'run'."""

    current: bytes = b""
    table: dict = {}

    def __init__(self):
        self.env_id = "0"
        self.num_entries = 16

    def reset_edge_map(self):
        pass

    def edge_id_array(self):
        return _u32(*type(self).table[type(self).current])

    def cleanup(self):
        pass


def _fake_run(target, data, timeout, env=None, **_):
    _FakeShm.current = data
    return 0, "", 0


class TestMinimizeWiring:
    # Cover keeps a alone; rare edges 1 and 2 need two distinct seeds.
    FILES = [
        ("a.bin", b"A" * 10, {1, 2}),
        ("b.bin", b"B" * 40, {1}),
        ("c.bin", b"C" * 40, {2}),
    ]

    def _run(self, tmp_path, monkeypatch, hall):
        from fuzzer_tool.adapters import process
        from fuzzer_tool.services import minimize

        corpus = tmp_path / "corpus"
        corpus.mkdir()
        paths = []
        _FakeShm.table = {}
        for name, data, edges in self.FILES:
            p = corpus / name
            p.write_bytes(data)
            _FakeShm.table[data] = edges
            paths.append(p)
        monkeypatch.setattr(minimize, "ShmCoverage", _FakeShm)
        monkeypatch.setattr(process, "run_target_stdin", _fake_run)
        minimize._minimize_with_coverage(paths, "t", 1.0, False, None, None, corpus, hall=hall)
        return sorted(p.name for p in corpus.iterdir() if p.is_file())

    def test_off_keeps_cover_only(self, tmp_path, monkeypatch):
        from fuzzer_tool.core.run_mode import Switch

        assert self._run(tmp_path, monkeypatch, Switch.OFF) == ["a.bin"]

    def test_on_adds_distinct_rep(self, tmp_path, monkeypatch, capsys):
        from fuzzer_tool.core.run_mode import Switch

        kept = self._run(tmp_path, monkeypatch, Switch.ON)
        assert len(kept) == 2 and "a.bin" in kept
        assert "Distinct Representatives" in capsys.readouterr().out


def test_minimize_flag_registered(monkeypatch, capsys):
    import sys

    from fuzzer_tool.cli import commands

    monkeypatch.setattr(sys, "argv", ["fuzzer-tool", "minimize", "--help"])
    with pytest.raises(SystemExit):
        commands.main()
    assert "--hall" in capsys.readouterr().out
