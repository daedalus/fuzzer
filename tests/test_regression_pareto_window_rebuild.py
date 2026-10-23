"""``_rebuild_pareto`` scores only the window its front reads.

The front is taken over the last ``window`` scores, yet every rebuild (one
per admission) scored the whole corpus: ffmpeg --lightweight, 4k seeds,
4,192 rebuilds, 39 s of a 459 s profile. Reference: the full loop, copied.
"""

import tempfile
from unittest.mock import patch

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.services.fuzzer import Fuzzer
from fuzzer_tool.services.seed_picker import SeedPicker

NOW = 1000.0


def _make_fuzzer() -> Fuzzer:
    tmpdir = tempfile.mkdtemp(prefix="fuzz_test_")
    with (
        patch("os.path.isfile", return_value=True),
        patch("os.access", return_value=True),
    ):
        return Fuzzer(
            target="/bin/true",
            corpus_dir=f"{tmpdir}/corpus",
            crashes_dir=f"{tmpdir}/crashes",
            max_len=256,
            timeout=1,
            mutations_per_input=2,
        )


def _reference_front(f: Fuzzer, now: float) -> set[int]:
    # Pre-change loop: score every seed, front over the last 100.
    scores = []
    for seed in f.corpus:
        meta = f.seed_meta.get(seed)
        if meta is None:
            scores.append((1.0, 1.0, 1.0))
            continue
        sub, _div, spa, _cov = f._cached_weights.get(f._seed_key(seed), (1.0, 1.0, 1.0, 0.5))
        age = now - meta["added_at"]
        burst = max(1.0, 1.0 + f._temperature * 4.0 - (age / 60.0) * f._temperature)
        scores.append((sub, burst, spa))
    return SeedPicker._pareto_front(scores, window=100)


def _populate(f: Fuzzer, n: int, seed: int) -> None:
    rng = RandPool(seed=seed)
    f.corpus = [b"s%d" % i for i in range(n)]
    f.seed_meta = {}
    f._cached_weights = {}
    for i, s in enumerate(f.corpus):
        if i % 7 == 3:
            continue  # no meta -> placeholder score
        f.seed_meta[s] = {"added_at": NOW - rng.randint(0, 600)}
        # Quantized: ties exercise non-strict dominance.
        sub, spa = rng.randint(0, 4) / 4, rng.randint(0, 4) / 4
        f._cached_weights[f._seed_key(s)] = (sub, 1.0, spa, 0.5)
    f._temperature = 0.5


def _front(f: Fuzzer) -> set[int]:
    f._seed_picker._rebuild_pareto(NOW, len(f.corpus), False)
    return f._pareto_front_cache


def test_regression_pareto_control_matches_itself():
    # Control (Hard Rule 46).
    f = _make_fuzzer()
    _populate(f, 250, 1)
    assert _reference_front(f, NOW) == _reference_front(f, NOW)


def test_regression_pareto_window_matches_full_scan():
    for seed in range(6):
        f = _make_fuzzer()
        _populate(f, 250, seed)
        assert _front(f) == _reference_front(f, NOW), seed


def test_regression_pareto_small_corpus():
    # Adversarial: corpus shorter than the window, and exactly the window.
    for n in (3, 99, 100, 101):
        f = _make_fuzzer()
        _populate(f, n, n)
        assert _front(f) == _reference_front(f, NOW), n


def test_regression_pareto_scores_only_window():
    # Falsification: the rebuild must not touch seeds outside the window.
    f = _make_fuzzer()
    _populate(f, 250, 2)
    looked = []
    real = f._seed_key
    f._seed_key = lambda s: looked.append(s) or real(s)
    _front(f)
    assert looked
    assert set(looked) <= set(f.corpus[-100:])
