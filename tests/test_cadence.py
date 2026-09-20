"""Tests for per-site phase offsets on the fuzzer's periodic cadences."""

from __future__ import annotations

import math
from collections import Counter

import pytest

from fuzzer_tool.core.cadence import bucket, due, phase_of

SITE = "seed_picker.weights"


# ── phase ──────────────────────────────────────────────────────────────


def test_phase_is_inside_the_period():
    for period in (2, 8, 50, 100, 128, 200, 500, 1000, 2000):
        assert 0 <= phase_of(SITE, period) < period


def test_phase_is_stable_across_calls():
    assert phase_of(SITE, 200) == phase_of(SITE, 200)


def test_phase_does_not_use_the_randomized_str_hash():
    # FALSIFICATION: `hash("x")` is salted per interpreter run, so a resumed
    # campaign would fire on a different schedule than the one it saved.
    # Derived from the digest rather than an echoed literal (Hard Rule 39).
    import zlib

    for site in ("seed_picker.weights", "fuzzer.rss_eps", "markov.snapshot"):
        for period in (50, 100, 200):
            assert phase_of(site, period) == zlib.crc32(site.encode()) % period


def test_phase_survives_a_subprocess():
    # The salt only differs between interpreters, so a same-process check
    # cannot see the bug this guards against.
    import subprocess
    import sys

    code = (
        "from fuzzer_tool.core.cadence import phase_of;print(phase_of('seed_picker.weights', 200))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert int(out.stdout) == phase_of("seed_picker.weights", 200)


def test_distinct_sites_get_distinct_phases():
    sites = [f"module.site_{i}" for i in range(40)]
    phases = [phase_of(s, 100) for s in sites]
    # Collisions are possible in 100 slots; a pile-up is not.
    assert max(Counter(phases).values()) <= 3


def test_period_one_has_no_phase():
    assert phase_of(SITE, 1) == 0


@pytest.mark.parametrize("bad", [0, -1, -100])
def test_non_positive_period_is_rejected(bad):
    with pytest.raises(ValueError):
        phase_of(SITE, bad)


# ── due ────────────────────────────────────────────────────────────────


def test_fires_exactly_once_per_period():
    period = 50
    hits = [c for c in range(1000) if due(c, period, SITE)]
    assert len(hits) == 1000 // period
    gaps = {b - a for a, b in zip(hits[:-1], hits[1:], strict=True)}
    assert gaps == {period}


def test_rate_is_unchanged_by_the_offset():
    # The point of the fix: same frequency, different phase. A change in
    # rate would silently retune every subsystem it is applied to.
    for period in (8, 50, 100, 128, 200):
        n = sum(due(c, period, SITE) for c in range(100_000))
        assert n == 100_000 // period


def test_offset_actually_shifts_the_firing_instant():
    # FALSIFICATION: a no-op `due` that ignored the site would fire at the
    # same counters as the bare modulo, and decohere nothing.
    period = 100
    bare = {c for c in range(1000) if c % period == 0}
    shifted = {c for c in range(1000) if due(c, period, SITE)}
    assert bare != shifted


def test_period_one_fires_every_counter():
    assert all(due(c, 1, SITE) for c in range(20))


def test_co_firing_collapses_across_the_real_cadences():
    # The measured defect: 19 periodic subsystems whose 10 distinct periods
    # have 0 coprime pairs out of 45, co-firing up to 19-deep. Shifting the
    # phases cannot change that the periods share divisors, but it spreads
    # which counters the firings land on.
    sites = {
        "execution_time.crps": 8,
        "monte_carlo.draw_refresh": 16,
        "filesystem.snapshot": 20,
        "crash_eta.sample": 50,
        "katz.recompute": 50,
        "markov.snapshot": 50,
        "fuzzer.dict_eps": 100,
        "fuzzer.rss_eps": 100,
        "fuzzer.ablation_flush": 100,
        "elo.decay": 100,
        "seed_picker.classify": 100,
        "garch.refit": 128,
        "seed_picker.weights": 200,
        "seed_picker.corpus_history": 500,
        "stats.report": 1000,
        "monte_carlo.refit": 1000,
        "fuzzer.drop_resize": 1000,
        "tang.refit": 2000,
        "seed_picker.saturation": 2000,
    }
    n = 20_000

    before = Counter(sum(c % p == 0 for p in sites.values()) for c in range(1, n + 1))
    after = Counter(sum(due(c, p, s) for s, p in sites.items()) for c in range(1, n + 1))

    assert max(before) >= 18
    assert max(after) < max(before)
    # Total firings are conserved: this redistributes, it does not skip.
    assert sum(k * v for k, v in before.items()) == sum(k * v for k, v in after.items())


# ── bucket ─────────────────────────────────────────────────────────────


def test_bucket_is_constant_within_a_period():
    period = 200
    phase = phase_of(SITE, period)
    start = period - phase  # first counter of a fresh bucket
    values = {bucket(start + k, period, SITE) for k in range(period)}
    assert len(values) == 1


def test_bucket_advances_exactly_at_the_firing_counter():
    period = 200
    fires = [c for c in range(2000) if due(c, period, SITE)]
    for c in fires[1:]:
        assert bucket(c, period, SITE) == bucket(c - 1, period, SITE) + 1


def test_bucket_is_monotone():
    vals = [bucket(c, 200, SITE) for c in range(5000)]
    assert vals == sorted(vals)


def test_bucket_matches_due_for_every_site_and_period():
    # ADVERSARIAL: the two helpers must not drift apart -- a cache keyed on
    # `bucket` while a sibling recompute is gated on `due` would rebuild
    # from a stale key for one counter.
    for site in ("a.b", "seed_picker.weights", "zzz"):
        for period in (2, 7, 50, 200):
            for c in range(1, 600):
                advanced = bucket(c, period, site) != bucket(c - 1, period, site)
                assert advanced == due(c, period, site), (site, period, c)


def test_bucket_of_zero_is_not_negative():
    # ADVERSARIAL: exec_count starts at 0, and a negative bucket would make
    # the initial cache key collide with the one after the first wrap.
    for period in (50, 100, 200, 2000):
        assert bucket(0, period, SITE) >= 0


def test_phases_are_spread_over_the_real_site_names():
    sites = [
        "seed_picker.weights",
        "seed_picker.classify",
        "seed_picker.corpus_history",
        "fuzzer.dict_eps",
        "fuzzer.rss_eps",
        "crash_eta.sample",
        "markov.snapshot",
    ]
    fractions = [phase_of(s, 1000) / 1000 for s in sites]
    r = abs(sum(complex(math.cos(2 * math.pi * f), math.sin(2 * math.pi * f)) for f in fractions))
    assert r / len(fractions) < 0.6
