"""Regression: the trim exec threw away its own finds.

``trim_new_coverage`` runs ``data[:len // 2]`` once. A crash, or an edge set
different from the original's, made it ``return``: the crash was never saved,
the new edges never admitted. AFL++ fixed the same bug in 5.03c (``op:trim``).

The trim stays a trim: a crashing or differently-covering cut never replaces
the original seed.
"""

from __future__ import annotations

from pathlib import Path

from fuzzer_tool.adapters.filesystem import save_to_corpus
from fuzzer_tool.services.corpus_manager import CorpusManager

CRASH_RC = -11
TIMEOUT_RC = -1
ORIGINAL = b"A" * 64
TRIMMED = ORIGINAL[: len(ORIGINAL) // 2]
BASE_EDGES = frozenset({1, 2, 3})
NEW_EDGE = 99


class _Shm:
    """First read is the original's trace, the second the trimmed run's."""

    def __init__(self, trimmed_edges, new_ids):
        self._reads = [set(BASE_EDGES), set(trimmed_edges)]
        self._new_ids = set(new_ids)

    def get_edge_ids(self):
        return self._reads.pop(0) if len(self._reads) > 1 else set(self._reads[0])

    def is_new_coverage_with_edges(self):
        # Novelty is reported once, as the real virgin map does.
        has_new, self._new_ids = bool(self._new_ids), set()
        return has_new, set(self._reads[-1])

    def get_edge_counts(self):
        return {e: 1 for e in self._reads[-1]}


class _Runner:
    def __init__(self, rc):
        self.rc = rc

    def run_target(self, data):
        return self.rc, "stderr"


class _Tracker:
    def __init__(self):
        self.recorded: list[tuple[str, set[int]]] = []

    def record_edges(self, seed_key, hit_edges, hit_counts=None):
        self.recorded.append((seed_key, set(hit_edges)))
        return set(hit_edges)

    def get_seed_edge_count(self, key):
        return len(BASE_EDGES)


class _Fuzzer:
    def __init__(self, corpus_dir: Path, rc: int, trimmed_edges, new_ids):
        self.corpus_dir = corpus_dir
        self.corpus: list[bytes] = []
        self.seed_meta: dict[bytes, dict] = {}
        self.seen_hashes: set[str] = set()
        self.irreplaceable_hashes: set[str] = set()
        self.bloom = None
        self.shm_cov = _Shm(trimmed_edges, new_ids)
        self.ptrace_cov = None
        self._runner = _Runner(rc)
        self._edge_tracker = _Tracker()
        self._agg_cache_valid = True
        self._use_lineage = False
        self.crash_count = 0
        self.crashes: list[tuple[bytes, int]] = []
        self.exec_count = 10**9  # fuzz execs: the trim exec share never binds
        self.admitted: list[tuple[bytes, bytes | None]] = []
        self.pruned = 0

    def _is_crash(self, rc, stderr):
        return rc == CRASH_RC

    def save_crash(self, data, rc, stderr):
        self.crashes.append((data, rc))
        return "crash_name"

    def _prune_crash_data(self):
        self.pruned += 1

    def _confirm_new_coverage(self, data, shm, has_new, edge_ids):
        return has_new, edge_ids

    def _only_confirmed(self, counts):
        return counts

    def save_to_corpus(self, data, parent=None):
        self.admitted.append((data, parent))

    def _exec_distance(self):
        return None


def _make(tmp_path, rc=0, trimmed_edges=BASE_EDGES, new_ids=()):
    corpus_dir = tmp_path / "corpus"
    (corpus_dir / "seeds").mkdir(parents=True)
    f = _Fuzzer(corpus_dir, rc, trimmed_edges, new_ids)
    save_to_corpus(ORIGINAL, f.corpus_dir, f.seen_hashes, f.bloom)
    f.corpus.append(ORIGINAL)
    f.seed_meta[ORIGINAL] = {"fuzz_count": 0, "lineage_depth": 0}
    return f


class TestTrimKeepsFinds:
    def test_regression_trim_crash_saved(self, tmp_path):
        f = _make(tmp_path, rc=CRASH_RC)
        CorpusManager(f).trim_new_coverage(ORIGINAL, ORIGINAL)

        assert f.crashes == [(TRIMMED, CRASH_RC)]
        assert f.crash_count == 1
        assert f.pruned == 1
        assert f.corpus == [ORIGINAL], "a crashing cut replaced the seed"

    def test_regression_trim_new_edges_admitted(self, tmp_path):
        trimmed_edges = BASE_EDGES | {NEW_EDGE}
        f = _make(tmp_path, trimmed_edges=trimmed_edges, new_ids={NEW_EDGE})
        cm = CorpusManager(f)
        cm.trim_new_coverage(ORIGINAL, ORIGINAL)

        assert f.admitted == [(TRIMMED, ORIGINAL)]
        assert f._edge_tracker.recorded == [(cm.seed_key(TRIMMED), set(trimmed_edges))]
        assert f.corpus == [ORIGINAL], "a differently-covering cut replaced the seed"


class TestTrimFalsification:
    def test_changed_trace_without_novelty_not_admitted(self, tmp_path):
        """Lost edges, nothing new: neither kept nor swapped in."""
        f = _make(tmp_path, trimmed_edges={1, 2})
        CorpusManager(f).trim_new_coverage(ORIGINAL, ORIGINAL)

        assert f.admitted == []
        assert f._edge_tracker.recorded == []
        assert f.crashes == []
        assert f.corpus == [ORIGINAL]


class TestTrimAdversarial:
    def test_timeout_is_not_a_crash_or_a_find(self, tmp_path):
        f = _make(
            tmp_path, rc=TIMEOUT_RC, trimmed_edges=BASE_EDGES | {NEW_EDGE}, new_ids={NEW_EDGE}
        )
        CorpusManager(f).trim_new_coverage(ORIGINAL, ORIGINAL)

        assert f.crashes == []
        assert f.admitted == []
        assert f.corpus == [ORIGINAL]

    def test_crash_with_identical_trace_never_swapped_in(self, tmp_path):
        """Same edges as the original, but crashing: save it, don't trim to it."""
        f = _make(tmp_path, rc=CRASH_RC, trimmed_edges=BASE_EDGES)
        CorpusManager(f).trim_new_coverage(ORIGINAL, ORIGINAL)

        assert f.crashes == [(TRIMMED, CRASH_RC)]
        assert f.corpus == [ORIGINAL]
