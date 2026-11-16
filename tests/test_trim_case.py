"""AFL ``trim_case`` chunk trimming in ``CorpusManager.trim_new_coverage``.

Chunks of len/2, len/4, ... len/1024 (floor ``TRIM_MIN_BYTES``) are cut,
sliding from ``pos = chunk``; a cut is kept when the trace equals the
original's. The trace here is a pure function of the bytes, so every
expected result is derived by hand from the AFL walk in the docstrings.
"""

from __future__ import annotations

from pathlib import Path

from fuzzer_tool.adapters.filesystem import save_to_corpus
from fuzzer_tool.services.corpus_manager import _TRIM_EXEC_SHARE, _TRIM_MAX_EXECS, CorpusManager

CRASH_RC = -11
TIMEOUT_RC = -1
# Enough fuzz execs that the trim share never binds (single-walk tests).
FUZZ_EXECS = 10**9
KEPT = frozenset({1})
CHANGED = frozenset({2})


class _Shm:
    """Trace of the last run; starts as the original's (KEPT)."""

    def __init__(self):
        self.edges = set(KEPT)

    def get_edge_ids(self):
        return set(self.edges)

    def is_new_coverage_with_edges(self):
        return False, set(self.edges)


class _Runner:
    """Runs ``target(data) -> (rc, edges)`` and feeds the shm."""

    def __init__(self, shm, target):
        self._shm = shm
        self._target = target
        self.calls: list[bytes] = []

    def run_target(self, data):
        self.calls.append(data)
        rc, edges = self._target(data)
        self._shm.edges = set(edges)
        return rc, ""


class _Fuzzer:
    def __init__(self, corpus_dir: Path, target):
        self.corpus_dir = corpus_dir
        self.corpus: list[bytes] = []
        self.seed_meta: dict[bytes, dict] = {}
        self.seen_hashes: set[str] = set()
        self.irreplaceable_hashes: set[str] = set()
        self.bloom = None
        self.shm_cov = _Shm()
        self.ptrace_cov = None
        self._runner = _Runner(self.shm_cov, target)
        self._edge_tracker = self
        self._agg_cache_valid = True
        self._use_lineage = False
        self.crash_count = 0
        self.crashes: list[bytes] = []
        self.exec_count = FUZZ_EXECS

    def get_seed_edge_count(self, key):
        return len(KEPT)

    def _is_crash(self, rc, stderr):
        return rc == CRASH_RC

    def save_crash(self, data, rc, stderr):
        self.crashes.append(data)

    def _prune_crash_data(self):
        return None

    def _confirm_new_coverage(self, data, shm, has_new, edge_ids):
        return has_new, edge_ids

    def _exec_distance(self):
        return None


def _admit(f: _Fuzzer, original: bytes) -> None:
    save_to_corpus(original, f.corpus_dir, f.seen_hashes, f.bloom)
    f.corpus.append(original)
    f.seed_meta[original] = {"fuzz_count": 0, "lineage_depth": 0}


def _trim(tmp_path, original: bytes, target, execs: int = FUZZ_EXECS) -> _Fuzzer:
    corpus_dir = tmp_path / "corpus"
    (corpus_dir / "seeds").mkdir(parents=True)
    f = _Fuzzer(corpus_dir, target)
    f.exec_count = execs
    _admit(f, original)
    CorpusManager(f).trim_new_coverage(original, original)
    return f


def _head_tail(data):
    ok = data.startswith(b"HEAD") and data.endswith(b"TAIL")
    return 0, KEPT if ok else CHANGED


def test_middle_filler_is_cut_out(tmp_path):
    """HEAD + 56x + TAIL (64 B). Chunk 32 at pos 32 loses TAIL; 16 at pos 16
    twice -> 32 B; 8 at pos 8 twice -> 16 B; 4 at pos 4 twice -> HEADTAIL.
    The old single ``data[:len//2]`` cut lost TAIL and kept nothing."""
    original = b"HEAD" + b"x" * 56 + b"TAIL"
    f = _trim(tmp_path, original, _head_tail)

    assert f.corpus == [b"HEADTAIL"]


def test_tail_cut_down_to_prefix(tmp_path):
    """MAGIC + 59x: chunks 32, 16, 8 each drop the tail (-> 8 B); chunk 4 at
    pos 4 cuts into MAGIC and is rejected."""
    original = b"MAGIC" + b"x" * 59
    f = _trim(tmp_path, original, lambda d: (0, KEPT if d.startswith(b"MAGIC") else CHANGED))

    assert f.corpus == [b"MAGICxxx"]


class TestFalsification:
    def test_changed_trace_keeps_seed(self, tmp_path):
        """Every cut changes the trace: the seed stays byte-identical."""
        original = b"Z" * 64
        f = _trim(tmp_path, original, lambda d: (0, KEPT if d == original else CHANGED))

        assert f.corpus == [original]
        assert f.crashes == []


class TestAdversarial:
    def test_exec_budget_bounded(self, tmp_path):
        """Max-size input, nothing ever kept: the walk stops at the budget."""
        original = bytes(range(256)) * 256
        f = _trim(tmp_path, original, lambda d: (0, CHANGED))

        assert 0 < len(f._runner.calls) <= _TRIM_MAX_EXECS
        assert f.corpus == [original]

    def test_crash_stops_walk_keeps_progress(self, tmp_path):
        """MAGIC + 59x, crash below 16 B: 32 and 16 B kept, the 8 B cut
        crashes -> saved, walk stops, the 16 B seed is committed."""
        original = b"MAGIC" + b"x" * 59

        def target(d):
            if len(d) < 16:
                return CRASH_RC, CHANGED
            return 0, KEPT if d.startswith(b"MAGIC") else CHANGED

        f = _trim(tmp_path, original, target)

        assert f.crashes == [original[:8]]
        assert f.corpus == [original[:16]]

    def test_timeout_cut_skipped_walk_continues(self, tmp_path):
        """HEAD + 56 distinct bytes + TAIL; the first 16 B cut at pos 16 times
        out: skipped, pos -> 32, cutting there is kept, the walk reaches HEADTAIL."""
        original = b"HEAD" + bytes(range(0x80, 0x80 + 56)) + b"TAIL"
        timed_out = original[:16] + original[32:]

        def target(d):
            if d == timed_out:
                return TIMEOUT_RC, CHANGED
            return _head_tail(d)

        f = _trim(tmp_path, original, target)

        assert f.corpus == [b"HEADTAIL"]
        assert f.crashes == []


class TestExecShare:
    """Trim execs are capped at a share of fuzz execs, across walks.

    On FFmpeg 9.0.2 ~99% of cuts change the trace, so every admitted 2.7 KB
    seed spent the full 512-exec budget: 9,920 -> 65 fuzz execs in 90 s.
    """

    def test_regression_trim_exec_share(self, tmp_path):
        """40 fuzz execs allow int(40 * share) trim execs, no more."""
        execs = 40
        original = bytes(range(256)) * 16
        f = _trim(tmp_path, original, lambda d: (0, CHANGED), execs=execs)

        assert len(f._runner.calls) == int(execs * _TRIM_EXEC_SHARE)
        assert f.corpus == [original]

    def test_share_is_cumulative(self, tmp_path):
        """Falsification: a second walk with no new fuzz execs runs nothing."""
        corpus_dir = tmp_path / "corpus"
        (corpus_dir / "seeds").mkdir(parents=True)
        f = _Fuzzer(corpus_dir, lambda d: (0, CHANGED))
        f.exec_count = 40
        cm = CorpusManager(f)
        first, second = bytes(range(256)) * 16, bytes(range(255, -1, -1)) * 16

        _admit(f, first)
        cm.trim_new_coverage(first, first)
        spent = len(f._runner.calls)
        _admit(f, second)
        cm.trim_new_coverage(second, second)

        assert spent == int(f.exec_count * _TRIM_EXEC_SHARE)
        assert len(f._runner.calls) == spent

    def test_no_fuzz_execs_no_trim(self, tmp_path):
        """Adversarial: zero fuzz execs -> zero trim execs, seed untouched."""
        original = b"HEAD" + b"x" * 56 + b"TAIL"
        f = _trim(tmp_path, original, _head_tail, execs=0)

        assert f._runner.calls == []
        assert f.corpus == [original]
