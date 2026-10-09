"""MI profile from running sums: no per-cell pass per rebuild.

``mi_profile`` called ``mi(pos)`` for every position, walking every
(position, byte, edge) cell: 17 ms per rebuild at 72k cells, 2.2 s of a
2000-exec ``--hail-mary`` profile. With f(k) = k*log2(k), n_p the position's
observations, T the edge-marginal total, b the byte marginal and m the edge
marginal::

    n_p * MI_p = sum_cells f(c) + C_p*log2(T) - sum_x J_px*log2(b_px) - sum_e K_pe*log2(m_e)

with C_p = sum c, J_px = sum_e c, K_pe = sum_x c. record() keeps sum f(c)
and K; the profile adds one pass over (position, byte) pairs and one bincount
over K. Values equal the per-cell sum to rounding.
"""

import math
import random

import pytest

from fuzzer_tool.core.mi import MutualInformationTracker

REL = 1e-9
ABS = 1e-12


def _tracker(seed: int, n: int = 400, min_obs: int = 5) -> MutualInformationTracker:
    rnd = random.Random(seed)
    t = MutualInformationTracker(max_positions=64, min_observations=min_obs)
    for _ in range(n):
        data = bytes(rnd.randrange(rnd.choice((2, 4, 256))) for _ in range(rnd.randrange(1, 48)))
        edges = {rnd.randrange(40) for _ in range(rnd.randrange(0, 12))}
        if data and data[0] == 0:
            edges.add(1000)  # an edge correlated with byte 0
        t.record(data, edges, map_size=65536)
    return t


def _direct(t: MutualInformationTracker, input_length: int) -> dict[int, float]:
    """Oracle: the per-position, per-cell sum (``mi``)."""
    return {p: t.mi(p) for p in range(input_length) if p in t.position_counts}


def _close(got: dict, want: dict) -> bool:
    return got.keys() == want.keys() and all(
        math.isclose(got[p], want[p], rel_tol=REL, abs_tol=ABS) for p in want
    )


# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(5))
def test_regression_mi_profile_closed_form(seed):
    """Profile equals the per-cell oracle at every checkpoint while recording."""
    rnd = random.Random(100 + seed)
    t = MutualInformationTracker(max_positions=64, min_observations=5)
    for step in range(300):
        data = bytes(rnd.randrange(8) for _ in range(rnd.randrange(1, 40)))
        t.record(data, {rnd.randrange(30) for _ in range(rnd.randrange(0, 10))})
        if step % 37 == 0:
            want = _direct(t, 64)
            # Control (Hard Rule 46): the oracle against a second run of itself.
            assert _direct(t, 64) == want
            assert _close(t.mi_profile(64), want)


def test_profile_does_not_walk_cells(monkeypatch):
    """The profile reads the running sums; mi() (the cell walk) is not called."""
    t = _tracker(5)
    calls = []
    monkeypatch.setattr(t, "mi", lambda p: calls.append(p) or 0.0)
    t.mi_profile(64)
    assert calls == []


def test_profile_detects_the_planted_dependency():
    """Falsification: the byte that picks the edge carries the most MI."""
    rnd = random.Random(1)
    t = MutualInformationTracker(max_positions=8, min_observations=5)
    for _ in range(400):
        data = bytes(rnd.randrange(2) for _ in range(8))
        t.record(data, {1000 if data[3] else 2000})
    profile = t.mi_profile(8)
    assert max(profile, key=profile.get) == 3
    assert profile[3] > 10 * max(v for p, v in profile.items() if p != 3)


def test_profile_after_save_load_round_trip():
    """Adversarial: sums are rebuilt from the loaded joint."""
    t = _tracker(2)
    restored = MutualInformationTracker()
    restored.from_dict(t.to_dict())
    assert _close(restored.mi_profile(64), _direct(t, 64))
    restored.record(b"\x00\x01\x02", {1000, 3})
    assert _close(restored.mi_profile(64), _direct(restored, 64))


def test_profile_with_input_length_and_unseen_positions():
    """Adversarial: truncated length, positions below min_observations, empty tracker."""
    t = _tracker(3, n=60, min_obs=40)
    for length in (0, 1, 7, 64, 200):
        assert _close(t.mi_profile(length), _direct(t, length))
    assert MutualInformationTracker().mi_profile() == {}


def test_inconsistent_loaded_marginal_falls_back():
    """Adversarial: an edge whose marginal is 0 (hand-edited state) matches mi()."""
    t = _tracker(4)
    state = t.to_dict()
    edge = next(iter(next(iter(next(iter(state["joint"].values())).values()))))
    state["edge_marginal"][int(edge)] = 0
    restored = MutualInformationTracker()
    restored.from_dict(state)
    assert _close(restored.mi_profile(64), _direct(restored, 64))
