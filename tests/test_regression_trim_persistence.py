"""Regression tests for finding #27 — corpus trim was memory-only.

``trim_new_coverage`` replaced ``f.corpus[idx]`` with the shorter input and
rewrote ``seed_meta``, but never touched disk. ``auto_minimize_corpus`` builds
its kept-set from ``f.corpus``, so on the next minimize pass the ORIGINAL
file — whose hash is no longer in that set — was moved to ``pruned/`` while
the trimmed bytes had never been written anywhere. After a resume the corpus
held neither: the seed was lost outright.

These tests drive ``trim_new_coverage`` against a fake fuzzer with a real
corpus directory and then reload from disk, which is the state a ``--resume``
sees.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fuzzer_tool.adapters.filesystem import (
    hash_data,
    rehydrate_by_hash,
    save_to_corpus,
)
from fuzzer_tool.services.corpus_manager import CorpusManager, _retire_seed_file


class _FakeShm:
    """Returns a fixed edge set; the trimmed run must be a subset to be kept."""

    def __init__(self, edges):
        self._edges = set(edges)

    def get_edge_ids(self):
        return set(self._edges)


class _FakeRunner:
    def __init__(self, rc=0):
        self.rc = rc
        self.calls = []

    def run_target(self, data):
        self.calls.append(data)
        return self.rc, ""


class _FakeFuzzer:
    def __init__(self, corpus_dir: Path):
        self.corpus_dir = corpus_dir
        self.corpus: list[bytes] = []
        self.seed_meta: dict[bytes, dict] = {}
        self.seen_hashes: set[str] = set()
        self.irreplaceable_hashes: set[str] = set()
        self.bloom = None
        self.shm_cov = _FakeShm({1, 2, 3})
        self.ptrace_cov = None
        self._runner = _FakeRunner()
        self._agg_cache_valid = True
        self._use_lineage = False
        self._edge_tracker = self
        self.exec_count = 0

    # stand-in for EdgeTracker.get_seed_edge_count
    def get_seed_edge_count(self, key):
        return 3


@pytest.fixture
def fuzzer(tmp_path):
    corpus_dir = tmp_path / "corpus"
    (corpus_dir / "seeds").mkdir(parents=True)
    return _FakeFuzzer(corpus_dir)


ORIGINAL = b"A" * 64
TRIMMED = ORIGINAL[:32]


def _seed_the_corpus(f: _FakeFuzzer) -> None:
    save_to_corpus(ORIGINAL, f.corpus_dir, f.seen_hashes, f.bloom)
    f.corpus.append(ORIGINAL)
    f.seed_meta[ORIGINAL] = {
        "fuzz_count": 3,
        "coverage_edges": 3,
        "momentum": 0.0,
        "edge_bitmap": bytearray(0),
        "redqueen_offsets": [],
        "added_at": 0.0,
        "lineage_depth": 0,
    }


class TestTrimIsPersisted:
    def test_trimmed_bytes_reach_disk(self, fuzzer):
        _seed_the_corpus(fuzzer)
        CorpusManager(fuzzer).trim_new_coverage(ORIGINAL, ORIGINAL)

        assert fuzzer.corpus == [TRIMMED], "in-memory swap did not happen"
        recovered = rehydrate_by_hash(hash_data(TRIMMED), fuzzer.corpus_dir)
        assert recovered == TRIMMED, "trimmed seed is not on disk; lost on resume"

    def test_original_is_retired_not_left_live(self, fuzzer):
        _seed_the_corpus(fuzzer)
        live = fuzzer.corpus_dir / "seeds" / hash_data(ORIGINAL)[:2] / f"id_{hash_data(ORIGINAL)}"
        assert live.is_file()

        CorpusManager(fuzzer).trim_new_coverage(ORIGINAL, ORIGINAL)

        assert not live.is_file(), "original still occupies a live corpus slot"

    def test_original_is_recoverable_from_pruned(self, fuzzer):
        """Retire, not unlink: delta children must still rehydrate."""
        _seed_the_corpus(fuzzer)
        CorpusManager(fuzzer).trim_new_coverage(ORIGINAL, ORIGINAL)

        assert rehydrate_by_hash(hash_data(ORIGINAL), fuzzer.corpus_dir) == ORIGINAL

    def test_resume_sees_exactly_one_of_the_two(self, fuzzer):
        """The state a --resume loads: the trimmed seed present, live, once."""
        _seed_the_corpus(fuzzer)
        CorpusManager(fuzzer).trim_new_coverage(ORIGINAL, ORIGINAL)

        seeds_dir = fuzzer.corpus_dir / "seeds"
        live_files = [p for p in seeds_dir.rglob("id_*") if p.is_file() and "pruned" not in p.parts]
        assert len(live_files) == 1
        assert live_files[0].read_bytes() == TRIMMED

    def test_original_hash_stays_seen(self, fuzzer):
        """Deliberate: re-admitting the original would undo the trim."""
        _seed_the_corpus(fuzzer)
        CorpusManager(fuzzer).trim_new_coverage(ORIGINAL, ORIGINAL)
        assert hash_data(ORIGINAL) in fuzzer.seen_hashes
        assert hash_data(TRIMMED) in fuzzer.seen_hashes


class TestTrimDeclineIsUnchanged:
    """Nothing may be written or retired when the trim is rejected."""

    def test_irreplaceable_seed_untouched(self, fuzzer):
        _seed_the_corpus(fuzzer)
        fuzzer.irreplaceable_hashes.add(hash_data(ORIGINAL))
        CorpusManager(fuzzer).trim_new_coverage(ORIGINAL, ORIGINAL)

        assert fuzzer.corpus == [ORIGINAL]
        assert rehydrate_by_hash(hash_data(TRIMMED), fuzzer.corpus_dir) is None

    def test_coverage_loss_rejects_trim(self, fuzzer):
        _seed_the_corpus(fuzzer)
        # Trimmed run reports an edge the full input never hit.
        fuzzer.shm_cov = _FakeShm({1, 2, 3})
        cm = CorpusManager(fuzzer)
        calls = {"n": 0}
        original_get = fuzzer.shm_cov.get_edge_ids

        def alternating():
            calls["n"] += 1
            return original_get() if calls["n"] == 1 else {1, 2, 3, 99}

        fuzzer.shm_cov.get_edge_ids = alternating
        cm.trim_new_coverage(ORIGINAL, ORIGINAL)

        assert fuzzer.corpus == [ORIGINAL]
        assert rehydrate_by_hash(hash_data(TRIMMED), fuzzer.corpus_dir) is None

    def test_failed_execution_rejects_trim(self, fuzzer):
        _seed_the_corpus(fuzzer)
        fuzzer._runner = _FakeRunner(rc=-1)
        CorpusManager(fuzzer).trim_new_coverage(ORIGINAL, ORIGINAL)

        assert fuzzer.corpus == [ORIGINAL]
        assert rehydrate_by_hash(hash_data(TRIMMED), fuzzer.corpus_dir) is None

    def test_short_seed_rejects_trim(self, fuzzer):
        tiny = b"AB"
        save_to_corpus(tiny, fuzzer.corpus_dir, fuzzer.seen_hashes, fuzzer.bloom)
        fuzzer.corpus.append(tiny)
        CorpusManager(fuzzer).trim_new_coverage(tiny, tiny)
        assert fuzzer.corpus == [tiny]


class TestRetireHelper:
    def test_moves_full_seed(self, fuzzer):
        save_to_corpus(ORIGINAL, fuzzer.corpus_dir, fuzzer.seen_hashes, fuzzer.bloom)
        h = hash_data(ORIGINAL)
        assert _retire_seed_file(fuzzer.corpus_dir, h) is True
        assert not (fuzzer.corpus_dir / "seeds" / h[:2] / f"id_{h}").is_file()
        assert (fuzzer.corpus_dir / "seeds" / "pruned" / h[:2] / f"id_{h}").is_file()

    def test_missing_hash_is_a_noop(self, fuzzer):
        assert _retire_seed_file(fuzzer.corpus_dir, "0" * 16) is False

    def test_no_corpus_dir_is_a_noop(self):
        assert _retire_seed_file(None, "0" * 16) is False
