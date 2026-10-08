"""MI weighted_position rebuilt on inputs ending at an ineligible position.

``_drop_stale_wp`` cleared the cache whenever an input's last position was
missing from it, but positions below ``min_observations`` are never in it, so
every such input forced a full MI profile rebuild: 126 of 183 rebuilds over
800 execs under --hail-mary.
"""

from fuzzer_tool.core.mi import MutualInformationTracker

MIN_OBS = 3
SHORT = 8


def _warm() -> MutualInformationTracker:
    t = MutualInformationTracker(max_positions=64, min_observations=MIN_OBS)
    for i in range(MIN_OBS + 1):
        t.record(bytes([i]) * SHORT, {1 + i % 2, 7}, map_size=64)
    assert t.weighted_position(SHORT) is not None
    return t


def test_regression_mi_wp_rebuild():
    """A longer input whose tail is still below min_observations keeps the cache."""
    t = _warm()
    cached = t._wp_sorted_pos
    t.record(b"\x05" * (SHORT + 4), {1}, map_size=64)
    assert t._wp_sorted_pos is cached


def test_eligible_new_tail_rebuilds():
    """Falsification: once the tail position is eligible, the cache drops."""
    t = _warm()
    for _ in range(MIN_OBS):
        t.record(b"\x05" * (SHORT + 4), {1}, map_size=64)
    t.weighted_position(SHORT + 4)
    t.record(b"\x06" * (SHORT + 6), {2}, map_size=64)  # tail SHORT+5 not eligible
    cached = t._wp_sorted_pos
    for _ in range(MIN_OBS):
        t.record(b"\x06" * (SHORT + 6), {2}, map_size=64)
    assert cached is not None
    assert t._wp_sorted_pos is None


def test_known_tail_keeps_cache():
    """Adversarial: an input ending on a cached position never rebuilds."""
    t = _warm()
    cached = t._wp_sorted_pos
    t.record(b"\x09" * SHORT, {1}, map_size=64)
    assert t._wp_sorted_pos is cached
