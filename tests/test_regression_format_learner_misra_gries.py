"""Format-learner signature counts are a Misra–Gries summary.

The old cap sorted all counts and dropped the lowest quarter, and it counted
exactly inside that window. Uniform garbage prefixes then recur by chance
(birthday) and get promoted to their own clusters. Measured over 60k
observations: ~850 garbage promotions vs ~40 under Misra–Gries, and
14 us vs 0.9 us per observation.

Misra–Gries keeps K counters; a newcomer arriving when they are full
decrements every counter and is not stored. Any signature seen more than
n/(K+1) times survives.
"""

from __future__ import annotations

import pytest

from fuzzer_tool.core.analyzers import analyzer_format_learner as afl
from fuzzer_tool.core.analyzers.analyzer_format_learner import DEFAULT_SIGNATURE, FormatLearner

K = 8  # shrunk capacity so overflow is reachable in a few observations
REAL = b"\xab\xcd"


@pytest.fixture(autouse=True)
def _small_cap(monkeypatch):
    monkeypatch.setattr(afl, "MAX_TRACKED_SIGNATURES", K)


def _input(prefix: bytes) -> bytes:
    return prefix + b"payload"


def _garbage(i: int) -> bytes:
    """Distinct 2-byte prefix per i, never REAL."""
    return (0x1000 + i).to_bytes(2, "big")


def _observe(fl: FormatLearner, prefix: bytes) -> None:
    fl.record_transition(_input(prefix), "op", 0, 1, 0, 0, set(), set())


def _reference_mg(stream: list[str], k: int, promote: int) -> dict[str, int]:
    """Textbook Misra–Gries with promotion; promoted keys stop being counted."""
    counts: dict[str, int] = {}
    promoted: set[str] = set()
    for s in stream:
        if s in promoted:
            continue
        if s in counts:
            counts[s] += 1
        elif len(counts) < k:
            counts[s] = 1
        else:
            counts = {key: c - 1 for key, c in counts.items() if c > 1}
            continue
        if counts[s] >= promote:
            promoted.add(s)
    return counts


def _stream() -> list[bytes]:
    """Garbage flood with REAL every 3rd observation and one garbage repeat."""
    out: list[bytes] = []
    for i in range(40):
        out.append(REAL if i % 3 == 0 else _garbage(i))
    out.insert(5, _garbage(1))
    return out


def test_counts_match_reference():
    prefixes = _stream()
    keys = [FormatLearner.format_signature(_input(p)) for p in prefixes]

    # Control: the oracle agrees with a second run of itself.
    assert _reference_mg(keys, K, 3) == _reference_mg(list(keys), K, 3)

    fl = FormatLearner()
    for p in prefixes:
        _observe(fl, p)

    assert fl._signature_counts == _reference_mg(keys, K, fl.promote_threshold)


def test_heavy_hitter_promoted_under_flood():
    """REAL at 1/3 of the stream exceeds n/(K+1): it must earn a cluster."""
    fl = FormatLearner()
    for p in _stream():
        _observe(fl, p)

    assert REAL.hex() in fl.clusters


def test_falsify_single_shot_never_promoted():
    fl = FormatLearner()
    for i in range(10 * K):
        _observe(fl, _garbage(i))

    assert set(fl.clusters) == {DEFAULT_SIGNATURE}


def test_adversarial_flood_stays_bounded():
    """Each counter and pending buffer stays within its cap at every step."""
    fl = FormatLearner()
    for i in range(20 * K):
        _observe(fl, _garbage(i % (2 * K)))
        assert len(fl._signature_counts) <= K
        assert set(fl._pending) <= set(fl._signature_counts)
        assert all(len(v) <= fl.promote_threshold for v in fl._pending.values())


def test_overflow_observation_is_not_lost():
    """A newcomer dropped by overflow still lands in the default cluster."""
    fl = FormatLearner()
    n = 3 * K
    for i in range(n):
        _observe(fl, _garbage(i))

    assert fl.clusters[DEFAULT_SIGNATURE].total_observations == n
