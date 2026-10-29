"""``CrashMITracker.weighted_position`` bisects Python lists.

Each call ran two ``np.searchsorted`` on scalars plus numpy scalar reads:
~8 us in-run, ~4 calls per exec on fuzzgoat. The sampler lists are built
once per ``_refresh``. Oracle: the pre-change method, frozen here; both
sides must also leave the RNG in the same state.
"""

import numpy as np

from fuzzer_tool.core.analyzers import analyzer_crash_eta as m
from fuzzer_tool.core.analyzers.analyzer_crash_eta import CrashMITracker
from fuzzer_tool.core.rand_pool import RandPool

_SEED = 0xB15
_RNG_SEED = 99
_N_RECORDS = 400
_MAX_LEN = 300
_CRASH_RATE = 0.08
_N_DRAWS = 1500


def _oracle(t: CrashMITracker, input_length: int):
    """Pre-change weighted_position, verbatim."""
    t._refresh()
    positions = t._pos_arr
    if not positions.size:
        return None
    idx = int(np.searchsorted(positions, input_length, side="left"))
    if idx == 0:
        return None
    prefix = t._prefix_arr
    eligible_mass = float(prefix[idx - 1])
    if eligible_mass <= 0.0:
        return int(positions[t._rng.randrange(idx)])
    r = t._rng.random() * eligible_mass
    i = int(np.searchsorted(prefix[:idx], r, side="left"))
    if i >= idx:
        i = idx - 1
    return int(positions[i])


def _feed(t: CrashMITracker, rng: RandPool, n: int) -> None:
    for _ in range(n):
        t.record(rng.randbytes(rng.randrange(_MAX_LEN)), rng.random() < _CRASH_RATE)


def _pair():
    a = CrashMITracker(rng=RandPool(seed=_RNG_SEED))
    b = CrashMITracker(rng=RandPool(seed=_RNG_SEED))
    _feed(a, RandPool(seed=_SEED), _N_RECORDS)
    _feed(b, RandPool(seed=_SEED), _N_RECORDS)
    return a, b


def _lengths():
    rng = RandPool(seed=_SEED + 1)
    # 0 and 1 sit below every observed position; _MAX_LEN + 50 above all.
    return [0, 1, _MAX_LEN + 50] + [rng.randrange(_MAX_LEN + 50) for _ in range(_N_DRAWS)]


def test_control_oracle_matches_itself():
    """Rule 46: two oracle trackers on one stream draw identically."""
    a, b = _pair()
    lens = _lengths()
    assert [_oracle(a, n) for n in lens] == [_oracle(b, n) for n in lens]


def test_regression_wpos_matches_oracle():
    """Falsification: same positions, same RNG consumption."""
    a, b = _pair()
    lens = _lengths()
    assert [b.weighted_position(n) for n in lens] == [_oracle(a, n) for n in lens]
    assert a._rng.random() == b._rng.random()


def test_regression_wpos_no_searchsorted(monkeypatch):
    """Falsification: the draw path makes no numpy searchsorted call."""
    t, _ = _pair()
    t._refresh()
    calls = {"n": 0}
    real = np.searchsorted

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(m._np, "searchsorted", counting)
    for n in _lengths()[:50]:
        t.weighted_position(n)
    assert calls["n"] == 0


def test_regression_wpos_tracks_new_records():
    """Adversarial: records between draws (crashes invalidate the sampler)
    must not leave stale lists; interleaved draws still match the oracle."""
    a, b = _pair()
    ra, rb = RandPool(seed=_SEED + 2), RandPool(seed=_SEED + 2)
    for n in _lengths()[:300]:
        assert b.weighted_position(n) == _oracle(a, n)
        _feed(a, ra, 3)
        _feed(b, rb, 3)


def test_regression_wpos_empty_and_loaded():
    """Adversarial: no observed positions returns None; a loaded tracker
    samples like the one it was saved from."""
    assert CrashMITracker().weighted_position(10) is None

    a, _ = _pair()
    b = CrashMITracker(rng=RandPool(seed=_RNG_SEED))
    b.load(a.save())
    c = CrashMITracker(rng=RandPool(seed=_RNG_SEED))
    c.load(a.save())
    lens = _lengths()[:200]
    assert [b.weighted_position(n) for n in lens] == [_oracle(c, n) for n in lens]
