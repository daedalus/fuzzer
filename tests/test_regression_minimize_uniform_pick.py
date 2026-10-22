"""--max-seeds took the first N files in disk order, not a uniform sample.

Disk order is seeds/<hh>/ by hash prefix, so batch minification only ever
saw low-prefix seeds. The subset is now k-of-n uniform via RandPool.
"""

import random

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.services import minimize
from fuzzer_tool.services.minimize import _pick_seeds

_N = 20
_K = 5
_ROUNDS = 4000
_SIGMAS = 5


class _ScriptedSample:
    """Fake RandPool: each sample() returns the next scripted index list."""

    def __init__(self, picks):
        self._picks = iter(picks)
        self.calls = []

    def sample(self, population, k):
        self.calls.append((population, k))
        return next(self._picks)


def _inclusion(sampler) -> list[int]:
    """Per-file inclusion counts over _ROUNDS k-of-n draws."""
    files = list(range(_N))
    counts = [0] * _N
    for _ in range(_ROUNDS):
        for f in sampler(files):
            counts[f] += 1
    return counts


def _uniform(counts) -> bool:
    """Every file within _SIGMAS binomial sd of k/n inclusion."""
    p = _K / _N
    mean = _ROUNDS * p
    sd = (_ROUNDS * p * (1 - p)) ** 0.5
    return all(abs(c - mean) <= _SIGMAS * sd for c in counts)


def test_regression_pick_uses_scripted_indices():
    """Falsification: exact files from scripted indices, disk order kept."""
    files = [f"f{i}" for i in range(_N)]
    rng = _ScriptedSample([[17, 3, 9]])

    picked = _pick_seeds(files, 3, rng)

    assert picked == [files[3], files[9], files[17]]
    assert rng.calls == [(_N, 3)]


def test_pick_uncapped_draws_nothing():
    """Adversarial: 0 or >= n returns every file without touching the RNG."""
    files = [f"f{i}" for i in range(_N)]
    rng = _ScriptedSample([])

    assert _pick_seeds(files, 0, rng) == files
    assert _pick_seeds(files, _N, rng) == files
    assert _pick_seeds(files, _N + 1, rng) == files
    assert rng.calls == []


def test_pick_is_uniform():
    """Inclusion ~ k/n per file; control: stdlib random.sample passes first."""
    ref = random.Random(1)
    assert _uniform(_inclusion(lambda f: ref.sample(f, _K)))

    pool = RandPool(seed=1)
    assert _uniform(_inclusion(lambda f: _pick_seeds(f, _K, pool)))


def test_head_slice_is_not_uniform():
    """The old [:k] fails the same oracle, so the oracle can fail."""
    assert not _uniform(_inclusion(lambda f: f[:_K]))


def test_minimize_corpus_wires_pick(tmp_path, monkeypatch):
    """minimize_corpus hands the scripted subset to the pruner."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for i in range(_N):
        (corpus / f"id_{i:02d}").write_bytes(bytes([i]))
    seen = {}

    def fake_hash(files, *_a, **_k):
        seen["files"] = [p.name for p in files]
        return len(files), 0

    monkeypatch.setattr(minimize, "_minimize_by_hash", fake_hash)
    files = minimize._discover_corpus_files(corpus)
    rng = _ScriptedSample([[4, 1]])

    minimize.minimize_corpus("/bin/true", str(corpus), max_seeds=2, rng=rng)

    assert seen["files"] == [files[1].name, files[4].name]
