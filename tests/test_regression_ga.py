"""Regressions: GA seed-key scheme and unmutated clones."""

from fuzzer_tool.core.edge_tracker import EdgeTracker
from fuzzer_tool.core.ga import GALifecycle, Individual
from fuzzer_tool.core.qea import QEALifecycle
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.services.corpus_manager import seed_key

CORPUS = [b"seed_%d" % i for i in range(4)]
EDGES_PER_SEED = 5


def _tracker() -> EdgeTracker:
    """Tracker keyed the way the fuzzer keys it (corpus_manager.seed_key)."""
    et = EdgeTracker()
    for i, data in enumerate(CORPUS):
        et.seed_edges[seed_key(data)] = {i * 10 + j for j in range(EDGES_PER_SEED)}
    et.cumulative_edges = set(range(10 * len(CORPUS)))
    return et


# --- seed key -----------------------------------------------------------


def test_regression_ga_seed_key_mismatch():
    """GA keyed by sha256 while the tracker keys by XXH64: every lookup missed."""
    ga = GALifecycle(pop_size=10)
    ga.initialize(CORPUS, _tracker())
    assert [i.edge_count for i in ga.population] == [EDGES_PER_SEED] * len(CORPUS)


def test_regression_qea_seed_key_mismatch():
    qea = QEALifecycle(pop_size=10)
    qea.initialize(CORPUS, _tracker())
    assert [i.edge_count for i in qea.population] == [EDGES_PER_SEED] * len(CORPUS)


def test_individual_key_matches_corpus_scheme():
    assert Individual(data=b"abc").seed_key == seed_key(b"abc")


def test_regression_ga_resume_drops_stale_key():
    """A state.json from before the fix carries sha256 keys; recompute on load."""
    stale = {"data": b"abc".hex(), "seed_key": "ba7816bf8f01cfea"}
    assert Individual.from_dict(stale).seed_key == seed_key(b"abc")


def test_unknown_seed_has_no_edges():
    """Falsification: a seed the tracker never saw must not borrow edges."""
    ga = GALifecycle(pop_size=10)
    ga.initialize([b"never_run"], _tracker())
    assert ga.population[0].edge_count == 0


def test_empty_seed_key():
    """Adversarial: empty input hashes like any other."""
    assert Individual(data=b"").seed_key == seed_key(b"")


# --- clone mutation -----------------------------------------------------

PARENT = b"parent"
MARK = b"!"


def _breeder(crossover_rate: float) -> GALifecycle:
    """GA with fixed parent, marking mutate/crossover so offspring are traceable."""
    ga = GALifecycle(
        pop_size=4, elite_fraction=0.25, crossover_rate=crossover_rate, mutation_rate=0.0
    )
    ga.population = [Individual(data=PARENT, fitness=1.0) for _ in range(4)]
    ga.select_parent = lambda: ga.population[0]
    ga._mutate = lambda d: d + MARK
    ga._crossover = lambda a, b: a + b"X"
    return ga


def test_regression_ga_clone_not_mutated():
    """No crossover + no mutation roll used to copy the parent verbatim."""
    ga = _breeder(crossover_rate=0.0)
    ga._evolve(EdgeTracker())
    offspring = ga.population[1:]
    assert [o.data for o in offspring] == [PARENT + MARK] * len(offspring)


def test_crossed_child_not_force_mutated():
    """Falsification: forcing applies to clones only, not crossover children."""
    ga = _breeder(crossover_rate=1.0)
    ga._evolve(EdgeTracker())
    offspring = ga.population[1:]
    assert [o.data for o in offspring] == [PARENT + b"X"] * len(offspring)


def test_clone_of_empty_parent():
    """Adversarial: real ops on an empty parent must not raise."""
    ga = GALifecycle(pop_size=3, elite_fraction=0.34, crossover_rate=0.0, mutation_rate=0.0)
    ga._rng = RandPool(seed=7)
    ga.population = [Individual(data=b"", fitness=1.0) for _ in range(3)]
    ga._evolve(EdgeTracker())
    assert all(isinstance(o.data, bytes) for o in ga.population)
