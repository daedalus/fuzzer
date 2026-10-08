"""Tests for --cuckoo-seed-filter: pruned seeds tracked in a CuckooFilter.

At startup, all seeds under corpus/seeds/pruned/ are added to the filter.
When a seed is pruned during minimization, its hash is added.
In _dedup_mutate(), if a mutation's hash matches a pruned seed, the
mutation is skipped (original data returned).
"""

from pathlib import Path

import pytest

from fuzzer_tool.services.fuzzer import Fuzzer


def _write_seed(corpus_dir: Path, seed: bytes) -> str:
    """Write a seed to corpus/seeds/ and return its 16-char hash."""
    from fuzzer_tool.adapters.filesystem import hash_data

    h = hash_data(seed)
    sub = corpus_dir / "seeds" / h[:2]
    sub.mkdir(parents=True, exist_ok=True)
    (sub / f"id_{h}").write_bytes(seed)
    return h


def _make_corpus_dir(tmp_path: Path) -> Path:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir(parents=True)
    (corpus_dir / "seeds").mkdir()
    return corpus_dir


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    return _make_corpus_dir(tmp_path)


def test_pruned_seed_added_to_cuckoo_at_startup(corpus_dir: Path) -> None:
    """Pruned seeds under corpus/seeds/pruned/ are loaded into the cuckoo filter at startup."""
    seed = b"pruned_seed_data"
    pruned_dir = corpus_dir / "seeds" / "pruned"
    pruned_dir.mkdir(parents=True)
    _write_seed(corpus_dir, seed)
    # Move seed to pruned
    from fuzzer_tool.adapters.filesystem import hash_data

    h = hash_data(seed)
    (corpus_dir / "seeds" / h[:2] / f"id_{h}").unlink()
    pruned_sub = pruned_dir / h[:2]
    pruned_sub.mkdir(parents=True)
    (pruned_sub / f"id_{h}").write_bytes(seed)

    f = Fuzzer(
        target="nonexistent",
        corpus_dir=str(corpus_dir),
        crashes_dir=str(corpus_dir / "crashes"),
        cuckoo_seed_filter=True,
    )
    assert f.cuckoo_seed_filter is not None
    assert f.cuckoo_seed_filter.contains(h)


def test_non_pruned_seed_not_in_cuckoo_filter(corpus_dir: Path) -> None:
    """Seeds not pruned should not be in the cuckoo filter."""
    seed = b"active_seed_data"
    _write_seed(corpus_dir, seed)

    f = Fuzzer(
        target="nonexistent",
        corpus_dir=str(corpus_dir),
        crashes_dir=str(corpus_dir / "crashes"),
        cuckoo_seed_filter=True,
    )
    h = f._seed_key(seed)
    # Active seed should NOT be in the cuckoo filter
    assert not f.cuckoo_seed_filter.contains(h)


def test_pruned_seed_mutation_is_skipped(corpus_dir: Path) -> None:
    """A pruned parent is not returned unmutated (was a wasted exec)."""
    seed = b"seed_to_prune"
    _write_seed(corpus_dir, seed)

    f = Fuzzer(
        target="nonexistent",
        corpus_dir=str(corpus_dir),
        crashes_dir=str(corpus_dir / "crashes"),
        cuckoo_seed_filter=True,
    )
    # Add the seed's hash to the cuckoo filter (simulating pruning)
    h = f._seed_key(seed)
    f.cuckoo_seed_filter.add(h)

    # A pruned parent is mutated, never executed verbatim.
    result = f._dedup_mutate(seed)
    assert result != seed


def test_non_pruned_seed_mutates_normally(corpus_dir: Path) -> None:
    """Seeds not in the cuckoo filter should mutate normally."""
    seed = b"active_seed"
    _write_seed(corpus_dir, seed)

    f = Fuzzer(
        target="nonexistent",
        corpus_dir=str(corpus_dir),
        crashes_dir=str(corpus_dir / "crashes"),
        cuckoo_seed_filter=True,
    )
    h = f._seed_key(seed)
    # Ensure seed is NOT in the filter
    assert not f.cuckoo_seed_filter.contains(h)

    # Mutation should proceed normally (return mutated data)
    result = f._dedup_mutate(seed)
    # Result should be different from original (mutation happened)
    assert result != seed


def test_cuckoo_filter_disabled_by_default(corpus_dir: Path) -> None:
    """When --cuckoo-seed-filter is not set, the filter is None."""
    seed = b"test_seed"
    _write_seed(corpus_dir, seed)

    f = Fuzzer(
        target="nonexistent",
        corpus_dir=str(corpus_dir),
        crashes_dir=str(corpus_dir / "crashes"),
    )
    assert f.cuckoo_seed_filter is None


def test_filter_capacity_scaled_to_corpus_size(corpus_dir: Path) -> None:
    """Filter capacity is max(10 * len(corpus), 100_000)."""
    # Add 5 seeds
    for i in range(5):
        _write_seed(corpus_dir, f"seed_{i}".encode())

    f = Fuzzer(
        target="nonexistent",
        corpus_dir=str(corpus_dir),
        crashes_dir=str(corpus_dir / "crashes"),
        cuckoo_seed_filter=True,
    )
    assert f.cuckoo_seed_filter is not None
    # Capacity should be at least 100_000 (min) since 5 * 10 = 50 < 100_000
    assert f.cuckoo_seed_filter.capacity == 100_000


def test_empty_corpus_filter_has_minimum_capacity(corpus_dir: Path) -> None:
    """Filter capacity is at least 100_000 even with empty corpus."""
    f = Fuzzer(
        target="nonexistent",
        corpus_dir=str(corpus_dir),
        crashes_dir=str(corpus_dir / "crashes"),
        cuckoo_seed_filter=True,
    )
    assert f.cuckoo_seed_filter is not None
    assert f.cuckoo_seed_filter.capacity == 100_000


def test_pruned_seeds_added_to_filter_at_startup(corpus_dir: Path) -> None:
    """All seeds under corpus/seeds/pruned/ are added to the filter at startup."""
    pruned_dir = corpus_dir / "seeds" / "pruned"
    pruned_dir.mkdir(parents=True)

    # Create multiple pruned seeds
    seeds = [b"pruned_1", b"pruned_2", b"pruned_3"]
    hashes = []
    for seed in seeds:
        h = _write_seed(corpus_dir, seed)
        # Move to pruned
        from fuzzer_tool.adapters.filesystem import hash_data

        hash_data(seed)
        sub = pruned_dir / h[:2]
        sub.mkdir(parents=True, exist_ok=True)
        (sub / f"id_{h}").write_bytes(seed)
        hashes.append(h)

    f = Fuzzer(
        target="nonexistent",
        corpus_dir=str(corpus_dir),
        crashes_dir=str(corpus_dir / "crashes"),
        cuckoo_seed_filter=True,
    )
    for h in hashes:
        assert f.cuckoo_seed_filter.contains(h), f"Pruned seed {h} should be in cuckoo filter"
