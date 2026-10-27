"""Inverse lottery: randomized eviction, P(victim) proportional to 1 / ticket."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from fuzzer_tool.core.inverse_lottery import inverse_lottery_victims
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.services.corpus_manager import CorpusManager
from tests.test_corpus_minimization import MockFuzzer, _cm_seed_key

NAN = float("nan")
INF = float("inf")


def test_equal_tickets_is_a_uniform_cull():
    """Falsification: flat tickets give every flow the same chance of being first out."""
    rng = random.Random(1)
    first = [inverse_lottery_victims([5.0] * 4, 1, rng)[0] for _ in range(8000)]

    assert all(first.count(i) == pytest.approx(2000, abs=200) for i in range(4))


def test_victim_probability_is_inverse_to_tickets():
    """9:1 tickets -> the low holder is the first victim 90% of the time."""
    rng = random.Random(2)
    first = [inverse_lottery_victims([9.0, 1.0], 1, rng)[0] for _ in range(8000)]

    assert first.count(1) / 8000 == pytest.approx(0.9, abs=0.02)


def test_high_ticket_can_still_lose():
    """Randomized, not a top-K cut: the best-scored flow is sometimes the victim."""
    rng = random.Random(3)
    first = [inverse_lottery_victims([100.0, 1.0, 1.0], 1, rng)[0] for _ in range(20000)]

    assert first.count(0) > 0
    assert first.count(0) / 20000 < 0.02


def test_victims_are_distinct_and_counted():
    rng = random.Random(4)
    victims = inverse_lottery_victims([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 4, rng)

    assert len(victims) == 4
    assert len(set(victims)) == 4
    assert all(0 <= v < 6 for v in victims)


def test_k_bounds():
    rng = random.Random(5)

    assert inverse_lottery_victims([1.0, 2.0], 0, rng) == []
    assert inverse_lottery_victims([1.0, 2.0], -3, rng) == []
    assert inverse_lottery_victims([], 2, rng) == []
    assert sorted(inverse_lottery_victims([1.0, 2.0, 3.0], 3, rng)) == [0, 1, 2]
    assert sorted(inverse_lottery_victims([1.0, 2.0, 3.0], 99, rng)) == [0, 1, 2]


def test_deterministic_under_a_fixed_seed():
    t = [1.0, 4.0, 2.0, 8.0, 3.0]

    assert inverse_lottery_victims(t, 3, random.Random(9)) == inverse_lottery_victims(
        t, 3, random.Random(9)
    )


@pytest.mark.parametrize("bad", [0.0, -1.0, NAN, INF, -5.0])
def test_garbage_ticket_is_the_likeliest_victim(bad):
    """Adversarial: garbage floors to ~0, so it goes first; never raises, never ties."""
    rng = random.Random(6)
    first = [inverse_lottery_victims([bad, 1.0, 1.0, 1.0], 1, rng)[0] for _ in range(2000)]

    assert first.count(0) == 2000


def test_all_tickets_garbage_still_returns_k_distinct():
    rng = random.Random(7)
    victims = inverse_lottery_victims([0.0, NAN, -1.0, INF], 2, rng)

    assert len(set(victims)) == 2


# --- corpus_manager wiring ----------------------------------------------------


def _meta(coverage_edges: int) -> dict:
    return {
        "fuzz_count": 1,
        "coverage_edges": coverage_edges,
        "added_at": 100.0,
        "edge_bitmap": bytearray(0),
        "redqueen_offsets": [],
        "momentum": 0.0,
        "lineage_depth": 0,
        "hamming_distance": 0,
    }


def _corpus(tmp_path: Path, lottery: bool) -> tuple[MockFuzzer, list[bytes]]:
    """Eight seeds, max_corpus 4. Seed 0 alone holds edge 2, so it is the (deterministic)
    cover; the other seven share edge 1, are optional, and four of them must go."""
    f = MockFuzzer(tmp_path)
    f.max_corpus = 4
    f._use_inverse_lottery_select = lottery
    seeds = [bytes([65 + i]) * 40 for i in range(8)]
    f.corpus = list(seeds)
    f.seed_meta = {s: _meta(10 + i) for i, s in enumerate(seeds)}
    f._edge_tracker.record_edges(_cm_seed_key(seeds[0]), {1, 2})
    for s in seeds[1:]:
        f._edge_tracker.record_edges(_cm_seed_key(s), {1})
    return f, seeds


def test_select_optional_keeps_exactly_the_budget(tmp_path):
    f, seeds = _corpus(tmp_path, lottery=True)
    scored = [(float(i + 1), s) for i, s in enumerate(seeds)]

    kept = CorpusManager(f)._inverse_lottery_select_optional(scored, 3)

    assert len(kept) == 3
    assert [s for _, s in scored if s in kept] == kept


def test_select_optional_zero_budget_evicts_all(tmp_path):
    f, seeds = _corpus(tmp_path, lottery=True)
    scored = [(1.0, s) for s in seeds]

    assert CorpusManager(f)._inverse_lottery_select_optional(scored, 0) == []
    assert CorpusManager(f)._inverse_lottery_select_optional(scored, -2) == []


def test_select_optional_budget_over_size_keeps_all(tmp_path):
    f, seeds = _corpus(tmp_path, lottery=True)
    scored = [(1.0, s) for s in seeds]

    assert CorpusManager(f)._inverse_lottery_select_optional(scored, 50) == seeds


def test_flag_off_is_the_deterministic_top_k(tmp_path):
    """Control: without the flag the same fixture keeps the same seeds every run."""
    runs = []
    for _ in range(3):
        f, _seeds = _corpus(tmp_path, lottery=False)
        CorpusManager(f).auto_minimize_corpus()
        runs.append(sorted(f.corpus))

    assert runs[0] == runs[1] == runs[2]


def test_flag_on_culls_to_budget_but_varies_with_the_rng(tmp_path):
    """The lottery path is reached (budget held) and is randomized, not a fixed cut."""
    survivors = set()
    for n in range(12):
        f, seeds = _corpus(tmp_path, lottery=True)
        f._rng = RandPool(seed=n)
        CorpusManager(f).auto_minimize_corpus()
        assert len(f.corpus) == 4
        assert seeds[0] in f.corpus
        survivors.add(tuple(sorted(f.corpus)))

    assert len(survivors) > 1
