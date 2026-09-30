"""``colorize(mode=PATH_POOLED)``: group-testing colorization."""

import pytest

from fuzzer_tool.core.colorization import ColorMode, colorize
from fuzzer_tool.core.rand_pool import RandPool


def _oracle(live):
    """Path checksum depends only on the bytes at ``live``."""
    return lambda data: hash(tuple(data[i] for i in live)) or 1


def _tainted(result):
    return {i for t in result.taints for i in range(t.start, t.end + 1)}


def _run(data, live, *, max_execs=0, seed=1, mode=ColorMode.POOLED):
    return colorize(
        data,
        _oracle(live),
        use_type_aware=False,
        max_execs=max_execs,
        rng=RandPool(seed=seed),
        mode=mode,
    )


@pytest.mark.parametrize("live", [{3}, {0, 40}, set(range(0, 64, 8)), {63}])
def test_never_taints_a_live_byte(live):
    data = bytes(range(64))
    for seed in range(5):
        assert not _tainted(_run(data, live, seed=seed)) & live


def test_finds_most_dead_bytes():
    data = bytes(range(128))
    tainted = _tainted(_run(data, {5, 77}))
    assert len(tainted) >= 0.9 * 126


def test_control_bisect_also_sound_on_same_oracle():
    data = bytes(range(64))
    live = {3, 30}
    res = _run(data, live, mode=ColorMode.BISECT)
    assert not _tainted(res) & live


def test_all_live_no_taints():
    data = bytes(range(32))
    res = _run(data, set(range(32)))
    assert res.taints == []
    assert res.colorized == data


def test_underestimated_d_stays_sound():
    data = bytes(range(64))
    live = set(range(0, 64, 2))
    assert not _tainted(_run(data, live)) & live


def test_budget_truncation_stays_sound():
    data = bytes(range(64))
    live = {9, 50}
    res = _run(data, live, max_execs=5)
    assert res.exec_count <= 5
    assert not _tainted(res) & live


def test_empty_and_single():
    assert _run(b"", set()).taints == []
    assert not _tainted(_run(b"x", {0}))


def test_colorized_keeps_path():
    data = bytes(range(64))
    live = {1, 2, 40}
    res = _run(data, live)
    assert _oracle(live)(res.colorized) == _oracle(live)(data)


def test_deterministic_per_seed():
    data = bytes(range(64))
    a, b = _run(data, {7}, seed=3), _run(data, {7}, seed=3)
    assert a.taints == b.taints and a.colorized == b.colorized
