"""Tests for the on-disk phase exchange behind sync desynchronization."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from fuzzer_tool.core.desync import PHASE_FILE
from fuzzer_tool.services.parallel import _publish_phase, _sibling_phases, _sync_delay

PERIOD = 30.0


@pytest.fixture
def parent():
    root = Path(tempfile.mkdtemp(prefix="desync_test_"))
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _worker(parent: Path, wid: int) -> Path:
    d = parent / f".w{wid}"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── publish ────────────────────────────────────────────────────────────


def test_publish_writes_the_firing_time(parent):
    own = _worker(parent, 0)
    _publish_phase(own, 1234.5)
    assert float((own / PHASE_FILE).read_text()) == pytest.approx(1234.5)


def test_publish_overwrites_rather_than_appends(parent):
    own = _worker(parent, 0)
    _publish_phase(own, 1.0)
    _publish_phase(own, 2.0)
    assert float((own / PHASE_FILE).read_text()) == pytest.approx(2.0)


def test_publish_survives_an_unwritable_directory(parent):
    # ADVERSARIAL: a phase write failing must not take the worker down --
    # it only costs this round's coupling.
    _publish_phase(parent / ".w9" / "missing", 1.0)


def test_phase_file_stays_out_of_the_synced_subtree(parent):
    # The corpus sync walks `seeds/`; a phase file landing there would be
    # offered to siblings as a seed.
    own = _worker(parent, 0)
    _publish_phase(own, 1.0)
    assert not list(own.glob("seeds/**/*"))
    assert PHASE_FILE.startswith(".")


# ── read ───────────────────────────────────────────────────────────────


def test_reads_siblings_and_skips_self(parent):
    own = _worker(parent, 0)
    _publish_phase(own, 100.0)
    _publish_phase(_worker(parent, 1), 110.0)
    _publish_phase(_worker(parent, 2), 120.0)

    phases = _sibling_phases(parent, own, now=120.0, period=PERIOD)
    assert sorted(phases) == pytest.approx(sorted([110.0 % PERIOD, 120.0 % PERIOD]))


def test_stale_workers_are_ignored(parent):
    # FALSIFICATION: a dead worker's file would otherwise pin a phase
    # forever and the survivors would spread around a ghost.
    own = _worker(parent, 0)
    _publish_phase(own, 1000.0)
    _publish_phase(_worker(parent, 1), 1000.0 - 10 * PERIOD)
    _publish_phase(_worker(parent, 2), 1000.0 - 1.0)

    phases = _sibling_phases(parent, own, now=1000.0, period=PERIOD)
    assert phases == pytest.approx([999.0 % PERIOD])


def test_future_timestamps_are_ignored(parent):
    # ADVERSARIAL: clock skew between processes, or a file from a previous
    # campaign in the same directory.
    own = _worker(parent, 0)
    _publish_phase(_worker(parent, 1), 5000.0)
    assert _sibling_phases(parent, own, now=1000.0, period=PERIOD) == []


def test_unparseable_file_is_skipped(parent):
    own = _worker(parent, 0)
    sib = _worker(parent, 1)
    (sib / PHASE_FILE).write_text("not a float")
    _publish_phase(_worker(parent, 2), 999.0)
    assert _sibling_phases(parent, own, now=1000.0, period=PERIOD) == pytest.approx(
        [999.0 % PERIOD]
    )


def test_partial_write_is_skipped(parent):
    # ADVERSARIAL: the write is not atomic, so an empty read is reachable.
    own = _worker(parent, 0)
    (_worker(parent, 1) / PHASE_FILE).write_text("")
    assert _sibling_phases(parent, own, now=1000.0, period=PERIOD) == []


def test_non_worker_directories_are_ignored(parent):
    own = _worker(parent, 0)
    other = parent / "seeds"
    other.mkdir()
    (other / PHASE_FILE).write_text("999.0")
    assert _sibling_phases(parent, own, now=1000.0, period=PERIOD) == []


def test_missing_parent_yields_nothing(parent):
    assert _sibling_phases(parent / "gone", parent / ".w0", now=1.0, period=PERIOD) == []


# ── delay ──────────────────────────────────────────────────────────────


def test_lone_worker_keeps_the_nominal_period(parent):
    own = _worker(parent, 0)
    assert _sync_delay(parent, own, now=1000.0, period=PERIOD) == pytest.approx(PERIOD)


def test_delay_moves_away_from_a_crowded_sibling(parent):
    own = _worker(parent, 0)
    _publish_phase(_worker(parent, 1), 999.0)  # fired 1s ago
    assert _sync_delay(parent, own, now=1000.0, period=PERIOD) > PERIOD


def test_delay_publishes_this_firing(parent):
    own = _worker(parent, 0)
    _sync_delay(parent, own, now=1000.0, period=PERIOD)
    assert float((own / PHASE_FILE).read_text()) == pytest.approx(1000.0)


def test_delay_is_always_positive(parent):
    own = _worker(parent, 0)
    for offset in (-0.001, 0.001, PERIOD / 2, PERIOD - 0.001):
        _publish_phase(_worker(parent, 1), 1000.0 + offset - PERIOD)
        assert _sync_delay(parent, own, now=1000.0, period=PERIOD) > 0.0


def _fleet_coherence(parent: Path, n: int, damping: float, fires: int, jitter: float, seed: int):
    """Run *n* workers through the real file exchange; return final coherence."""
    import math
    import random

    from fuzzer_tool.core.circular_stats import order_parameter
    from fuzzer_tool.core.desync import initial_offset, phase_shift

    rnd = random.Random(seed)
    dirs = [_worker(parent, i) for i in range(n)]
    fire = [1000.0 + initial_offset(i, n, PERIOD) + rnd.uniform(0, 0.05) for i in range(n)]

    for _ in range(fires):
        i = min(range(n), key=lambda k: fire[k])
        now = fire[i]
        _publish_phase(dirs[i], now)
        shift = phase_shift(
            now % PERIOD, _sibling_phases(parent, dirs[i], now, PERIOD), PERIOD, damping
        )
        # Per-round drift: workers overshoot the period by one fuzz_one
        # iteration, and by a different amount each.
        fire[i] = now + PERIOD + shift + rnd.uniform(-jitter, jitter)

    return order_parameter([2 * math.pi * (t % PERIOD) / PERIOD for t in fire])[0]


def test_coupling_holds_the_fleet_apart_against_drift(parent):
    """The operating case: staggered at startup, then drifting.

    FALSIFICATION: the control is the same fleet with the coupling
    neutralised, which is the pre-existing behaviour. Measured over 12 seeds
    x n in {4, 8, 16}, 600 fires, 1.5 s of per-round drift: coupled median
    0.113 / p90 0.204 / max 0.237, uncoupled median 0.250 / p90 0.848 /
    max 0.973. The worst case is the point -- an uncoupled fleet reaches
    near-total coherence, which is the thundering herd this exists to stop.
    """
    coupled = _fleet_coherence(parent, 8, 0.5, 600, 1.5, seed=3)
    shutil.rmtree(parent, ignore_errors=True)
    parent.mkdir(parents=True)
    uncoupled = _fleet_coherence(parent, 8, 1e-9, 600, 1.5, seed=3)

    assert coupled < 0.30
    assert uncoupled > coupled


def test_cold_cluster_does_not_converge(parent):
    """LIMITATION, pinned in the style of the scheduler STUCK tests.

    DESYNC's midpoint rule has zero gradient in the interior of a tight
    cluster -- a node already halfway between two neighbours 0.1 s away does
    not move -- so the cluster can only peel from its edges, and under the
    asynchronous file exchange the peeled nodes leapfrog as a group instead
    of spreading. Measured at 600 fires from a 1 s window, n=6: r stays at
    0.62 (damping 0.5) and does not improve at 2000 fires.

    This is why `initial_offset` stakes out the spacing at startup rather
    than trusting convergence. Asserting the bad behaviour so that a future
    change to the rule has to come here and say what it fixed.
    """
    import math

    from fuzzer_tool.core.circular_stats import order_parameter
    from fuzzer_tool.core.desync import phase_shift

    n = 6
    dirs = [_worker(parent, i) for i in range(n)]
    fire = [1000.0 + 0.2 * i for i in range(n)]

    for _ in range(600):
        i = min(range(n), key=lambda k: fire[k])
        now = fire[i]
        _publish_phase(dirs[i], now)
        nb = _sibling_phases(parent, dirs[i], now, PERIOD)
        fire[i] = now + PERIOD + phase_shift(now % PERIOD, nb, PERIOD)

    r = order_parameter([2 * math.pi * (t % PERIOD) / PERIOD for t in fire])[0]
    assert r > 0.4, f"cold-cluster convergence improved to {r:.3f} -- update this test"
