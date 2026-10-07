"""KatzChannel.seed_energy memoizes its per-score-vector lookups.

The seed-name position (two linear scans of the horizon's seed list) and the
peak score (a max over every graph node) were recomputed on every pick; they
only change when ensure_scores() replaces the scores or the horizon.
"""

import types

import numpy as np

from fuzzer_tool.services.katz_channel import KatzChannel


def _channel(names, scores):
    ch = KatzChannel.__new__(KatzChannel)
    ch._horizon = types.SimpleNamespace(n_u=2, seed_names=list(names))
    res = types.SimpleNamespace(scores=np.asarray(scores, dtype=float))
    ch.ensure_scores = lambda: res
    return ch, res


def _reference(ch, res, key):
    names = ch._horizon.seed_names
    if key not in names:
        return 0.0
    idx = ch._horizon.n_u + names.index(key)
    peak = float(res.scores.max())
    return 0.0 if peak <= 0 else min(float(res.scores[idx]) / peak, 1.0)


def test_matches_the_unmemoized_formula():
    ch, res = _channel(["a", "b", "a", "c"], [0.0, 9.0, 3.0, 4.5, 1.0, 2.0])
    for key in ("a", "b", "c", "zz"):
        assert ch.seed_energy(key) == _reference(ch, res, key)


def test_new_scores_invalidate_the_memo():
    ch, res = _channel(["a", "b"], [0.0, 0.0, 1.0, 2.0])
    assert ch.seed_energy("a") == 0.5
    res2 = types.SimpleNamespace(scores=np.asarray([0.0, 0.0, 4.0, 2.0]))
    ch._horizon = types.SimpleNamespace(n_u=2, seed_names=["b", "a"])
    ch.ensure_scores = lambda: res2
    assert ch.seed_energy("a") == 0.5
    assert ch.seed_energy("b") == 1.0
