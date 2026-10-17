"""core/edge_stats.py: hit profile, plateau, rarefaction, seed bootstrap."""

import itertools

import numpy as np
import pytest

from fuzzer_tool.core.edge_stats import (
    CI_HIGH_PCT,
    CI_LOW_PCT,
    KEXEC,
    MIN_SEGMENT,
    PLATEAU_RATIO,
    CurveShape,
    boot_ci,
    hit_profile,
    plateau,
    rarefy,
)
from tests.support.scripted_rng import ScriptedRng

# Seed -> edges; small enough to enumerate every subset.
INCIDENCE = [{1, 2, 3}, {2, 3}, {3, 4}, {5}, {1, 5, 6}, {3}]


def _owners(incidence) -> np.ndarray:
    edges = sorted(set().union(*incidence))
    return np.array([sum(e in s for s in incidence) for e in edges])


def _brute_rarefy(incidence, t: int) -> float:
    """Mean union size over every t-subset of seeds."""
    subsets = list(itertools.combinations(incidence, t))
    return sum(len(set().union(*c)) for c in subsets) / len(subsets)


def _survival(owners: np.ndarray) -> float:
    return float((owners > 0).mean())


# ── hit_profile ─────────────────────────────────────────────────────────────


def test_hit_profile_shares_and_decades():
    hits = [100, 10, 1, 1]
    prof = hit_profile(hits)
    total = sum(hits)

    assert prof.edges == len(hits)
    assert prof.total == total
    assert prof.top_share[1] == pytest.approx(100 / total)
    assert prof.top_share[10] == pytest.approx(1.0)
    assert prof.quantiles[50] == pytest.approx(np.median(hits))
    assert prof.decades[:3] == (2, 1, 1)


def test_hit_profile_decade_boundaries():
    # Adversarial: float log10 rounding at exact powers of ten.
    prof = hit_profile([9, 10, 999, 1000, 2**32 - 1])
    assert prof.decades[:4] == (1, 1, 1, 1)
    assert prof.decades[9] == 1


def test_hit_profile_empty_and_zero():
    assert hit_profile([]) is None
    assert hit_profile([0, 0]) is None
    assert hit_profile([0, 5]).edges == 1


# ── plateau ─────────────────────────────────────────────────────────────────


def _two_lines(head_rate: int, tail_rate: int, n: int, brk: int):
    x = [i * KEXEC for i in range(n)]
    y = [head_rate * i if i < brk else head_rate * brk + tail_rate * (i - brk) for i in range(n)]
    return list(zip(x, y, strict=True))


def test_plateau_finds_exact_break():
    brk, n = 10, 20
    p = plateau(_two_lines(10, 1, n, brk))

    assert p.exec_at == brk * KEXEC
    assert p.head_rate == pytest.approx(10)
    assert p.tail_rate == pytest.approx(1)
    assert p.tail_edges == (n - 1 - brk) * 1


def test_plateau_large_exec_counts():
    # Adversarial: exec counts near 1e9 must not lose the break to cancellation.
    base = 10**9
    pts = [(x + base, y) for x, y in _two_lines(10, 0, 40, 25)]
    p = plateau(pts)
    assert p.exec_at == base + 25 * KEXEC
    assert p.tail_rate == pytest.approx(0, abs=1e-6)


def test_plateau_degenerate():
    assert plateau([]) is None
    assert plateau([(i, i) for i in range(2 * MIN_SEGMENT - 1)]) is None
    # Every snapshot at one exec count: no slope to fit.
    assert plateau([(5, i) for i in range(10)]) is None


def test_plateau_flat_line():
    p = plateau([(i * KEXEC, 7) for i in range(12)])
    assert p.head_rate == 0 and p.tail_rate == 0 and p.tail_edges == 0
    assert p.shape is CurveShape.PLATEAU


def test_shape_by_rate_ratio():
    # FFmpeg 5k: 5171 -> 1004 edges/kexec is still growing, not a plateau.
    head = 100
    cut = int(head * PLATEAU_RATIO)
    assert plateau(_two_lines(head, cut - 1, 20, 10)).shape is CurveShape.PLATEAU
    assert plateau(_two_lines(head, cut + 1, 20, 10)).shape is CurveShape.SLOWDOWN
    assert plateau(_two_lines(head, head * 2, 20, 10)).shape is CurveShape.SPEEDUP


# ── rarefy ──────────────────────────────────────────────────────────────────


def test_rarefy_matches_subset_enumeration():
    # Falsification: exact expectation against brute force over all subsets.
    m = len(INCIDENCE)
    ts = list(range(m + 1))
    got = rarefy(_owners(INCIDENCE), m, ts)
    want = [0.0] + [_brute_rarefy(INCIDENCE, t) for t in ts[1:]]
    assert got == pytest.approx(want)


def test_rarefy_last_step_is_singletons_over_m():
    # Identity: E[S(m)] - E[S(m-1)] = Q1 / m.
    owners = _owners(INCIDENCE)
    m = len(INCIDENCE)
    a, b = rarefy(owners, m, [m - 1, m])
    assert b - a == pytest.approx((owners == 1).sum() / m)


def test_rarefy_clips_out_of_range():
    # Adversarial: owner counts above m (stale after prune) and t outside [0, m].
    got = rarefy(np.array([1, 9]), 3, [-1, 3, 7])
    assert got == pytest.approx([0.0, 2.0, 2.0])
    assert rarefy(np.array([], dtype=int), 3, [2]) == pytest.approx([0.0])


# ── boot_ci ─────────────────────────────────────────────────────────────────


def test_boot_ci_scripted_resamples():
    incidence = [{1, 2}, {2}, {3}]
    draws = [[0, 0, 0], [1, 2, 2], [0, 1, 2], [2, 2, 2]]
    rng = ScriptedRng(randrange_lists=draws)

    low, high = boot_ci(incidence, _survival, len(draws), rng)

    edges = set().union(*incidence)
    reps = [len(set().union(*(incidence[i] for i in d))) / len(edges) for d in draws]
    want = np.percentile(reps, [CI_LOW_PCT, CI_HIGH_PCT])
    assert (low, high) == pytest.approx(tuple(want))


def test_boot_ci_draws_exactly_resamples():
    # Adversarial: one extra draw would exhaust the script and raise.
    rng = ScriptedRng(randrange_lists=[[0, 1]])
    assert boot_ci([{1}, {2}], _survival, 1, rng) == pytest.approx((1.0, 1.0))


def test_boot_ci_degenerate():
    rng = ScriptedRng()
    assert boot_ci([{1}], _survival, 10, rng) is None
    assert boot_ci([{1}, {2}], _survival, 0, rng) is None
    assert boot_ci([set(), set()], _survival, 10, rng) is None
