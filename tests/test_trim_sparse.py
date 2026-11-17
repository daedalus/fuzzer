"""``--trim-sparse``: trim the least dense untried seed, within the trim budget."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from fuzzer_tool.adapters.filesystem import hash_data
from fuzzer_tool.services.corpus_manager import TRIM_TRIED, CorpusManager
from tests.test_corpus_minimization import MockFuzzer

MAGIC = b"KEEP"
EDGES = {1, 2}
DENSE = MAGIC + b"d" * 20
SPARSE = MAGIC + b"s" * 400
MIDDLE = MAGIC + b"m" * 100


class _Target:
    """Coverage depends on the 4-byte prefix only; every exec is recorded."""

    def __init__(self):
        self.runs: list[bytes] = []
        self.edges: set[int] = set()

    def run_target(self, data: bytes):
        self.runs.append(data)
        self.edges = set(EDGES) if data.startswith(MAGIC) else {99}
        return 0, ""

    def get_edge_ids(self) -> set[int]:
        return set(self.edges)


@pytest.fixture
def setup():
    tmp = Path(tempfile.mkdtemp(prefix="trim_sparse_"))
    f = MockFuzzer(tmp)
    target = _Target()
    f._runner = target
    f.shm_cov = target
    f.exec_count = 10**6
    f._afl_custom = None
    cm = CorpusManager(f)
    yield f, cm, target
    shutil.rmtree(tmp, ignore_errors=True)


def _load(f, cm, seeds, tracked=None):
    for s in seeds:
        f.corpus.append(s)
        f.seed_meta[s] = {"fuzz_count": 1, "coverage_edges": 0}
        if s in (seeds if tracked is None else tracked):
            f._edge_tracker.record_edges(cm.seed_key(s), set(EDGES))


def test_trims_least_dense_first(setup):
    """Falsification of corpus order: SPARSE sits last but goes first."""
    f, cm, target = setup
    _load(f, cm, [DENSE, MIDDLE, SPARSE])

    assert cm.trim_sparsest() is True
    assert target.runs[0] == SPARSE
    assert SPARSE not in f.corpus
    assert DENSE in f.corpus and MIDDLE in f.corpus
    assert min(map(len, f.corpus)) < len(DENSE)


def test_each_seed_tried_once(setup):
    """Trimmed and untrimmable seeds both stay tried; then nothing is left."""
    f, cm, target = setup
    _load(f, cm, [DENSE, SPARSE])

    assert cm.trim_sparsest() and cm.trim_sparsest()
    probes = len(target.runs)
    assert cm.trim_sparsest() is False
    assert len(target.runs) == probes


def test_no_budget_no_exec(setup):
    f, cm, target = setup
    _load(f, cm, [SPARSE])
    f.exec_count = 0
    assert cm.trim_sparsest() is False
    assert target.runs == []


def test_skips_irreplaceable_short_untracked(setup):
    """Adversarial: crash seeds, tiny seeds and untracked seeds are never probed."""
    f, cm, target = setup
    tiny = MAGIC + b"t"
    _load(f, cm, [SPARSE, tiny, MIDDLE], tracked=[SPARSE, tiny])
    f.irreplaceable_hashes.add(hash_data(SPARSE))

    assert cm.trim_sparsest() is False
    assert target.runs == []


def test_probe_with_changed_trace_not_trimmed(setup):
    """Adversarial: a probe whose trace is gone (flaky seed) leaves it alone."""
    f, cm, target = setup
    _load(f, cm, [SPARSE])
    target.run_target = lambda data: (target.runs.append(data), (-1, ""))[1]

    assert cm.trim_sparsest() is True
    assert f.corpus == [SPARSE]


def test_fuzzer_gate():
    """Off by default; on only with the switch and auto-minimize."""
    from fuzzer_tool.core.run_mode import Switch
    from fuzzer_tool.services.fuzzer import Fuzzer

    calls = []
    f = Fuzzer.__new__(Fuzzer)
    f._corpus_manager = SimpleNamespace(trim_sparsest=lambda: calls.append(1))
    f._auto_minimize = Switch.ON

    f._trim_sparse_on = Switch.OFF
    f._maybe_trim_sparse()
    f._trim_sparse_on = Switch.ON
    f._maybe_trim_sparse()
    f._auto_minimize = Switch.OFF
    f._maybe_trim_sparse()
    assert calls == [1]


class TestResume:
    """trim_tried rides the seed_meta whitelist, so --resume does not retry."""

    SEED = b"trim-sparse-resume-seed"

    def _fuzzer(self, tmp_path, **kw):
        from fuzzer_tool.adapters.filesystem import save_to_corpus
        from fuzzer_tool.services.fuzzer import Fuzzer

        corpus, crashes = tmp_path / "c", tmp_path / "k"
        (corpus / "seeds").mkdir(parents=True, exist_ok=True)
        crashes.mkdir(parents=True, exist_ok=True)
        save_to_corpus(self.SEED, corpus, set(), None)
        return Fuzzer(
            target="/bin/true",
            corpus_dir=str(corpus),
            crashes_dir=str(crashes),
            max_len=4096,
            **kw,
        )

    def test_trim_tried_survives_resume(self, tmp_path):
        f = self._fuzzer(tmp_path)
        f.seed_meta[self.SEED][TRIM_TRIED] = True
        f._save_state()

        resumed = self._fuzzer(tmp_path, resume=True)
        assert resumed.seed_meta[self.SEED][TRIM_TRIED] is True

    def test_untried_seed_resumes_untried(self, tmp_path):
        """Adversarial: an untried seed does not resume as tried."""
        f = self._fuzzer(tmp_path)
        f._save_state()

        resumed = self._fuzzer(tmp_path, resume=True)
        assert resumed.seed_meta[self.SEED].get(TRIM_TRIED, False) is False
