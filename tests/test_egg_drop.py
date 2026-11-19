"""Tests for the egg-drop threshold probe (core/egg_drop.py)."""

from functools import cache

import pytest

from fuzzer_tool.core.egg_drop import find_threshold, min_drops, reach


class FakeOracle:
    """Monotone predicate failing at x >= threshold; counts probes and failures."""

    def __init__(self, threshold):
        self.threshold = threshold
        self.probes = 0
        self.failures = 0

    def __call__(self, x):
        self.probes += 1
        failed = self.threshold is not None and x >= self.threshold
        self.failures += failed
        return failed


@cache
def _worst(eggs, floors):
    """Independent oracle: textbook min-max DP over the first drop floor."""
    if floors == 0:
        return 0
    if eggs == 1:
        return floors
    return 1 + min(
        max(_worst(eggs - 1, x - 1), _worst(eggs, floors - x)) for x in range(1, floors + 1)
    )


def _outcomes(lo, hi):
    return [*range(lo, hi + 1), None]


def test_oracle_control_textbook_values():
    """Hard Rule 46: the reference must reproduce known answers first."""
    assert _worst(2, 100) == 14
    assert _worst(2, 36) == 8
    assert _worst(3, 14) == 4
    assert _worst(1, 7) == 7


@pytest.mark.parametrize("eggs", [1, 2, 3, 5])
@pytest.mark.parametrize("floors", [1, 2, 7, 36, 60])
def test_min_drops_matches_dp(eggs, floors):
    assert min_drops(eggs, floors) == _worst(eggs, floors)


def test_reach_matches_recurrence():
    """Closed form vs f(k,m) = f(k-1,m-1) + f(k,m-1) + 1, built independently."""
    eggs, drops = 6, 40
    f = [[0] * (drops + 1) for _ in range(eggs + 1)]
    for k in range(1, eggs + 1):
        for m in range(1, drops + 1):
            f[k][m] = f[k - 1][m - 1] + f[k][m - 1] + 1
    for k in range(eggs + 1):
        for m in range(drops + 1):
            assert reach(k, m) == f[k][m]


@pytest.mark.parametrize("eggs", [1, 2, 3, 4])
@pytest.mark.parametrize("lo,hi", [(0, 0), (0, 9), (1, 36), (5, 40)])
def test_exact_threshold_within_budget(eggs, lo, hi):
    worst = 0
    for t in _outcomes(lo, hi):
        oracle = FakeOracle(t)
        res = find_threshold(oracle, lo, hi, eggs)
        assert res.threshold == t
        assert oracle.failures <= eggs
        assert res.failures == oracle.failures and res.probes == oracle.probes
        worst = max(worst, oracle.probes)
    assert worst == _worst(eggs, hi - lo + 1)


def test_falsification_plain_binary_search_overspends():
    """The k-failure bound is not vacuous: bisection breaks it with k=1."""
    oracle = FakeOracle(1)
    lo, hi = 0, 63
    while lo < hi:  # bisection for the first failing x
        mid = (lo + hi) // 2
        if oracle(mid):
            hi = mid
        else:
            lo = mid + 1
    assert oracle.failures > 1

    capped = FakeOracle(1)
    assert find_threshold(capped, 0, 63, 1).threshold == 1
    assert capped.failures == 1


# ── adversarial ────────────────────────────────────────────────────────


@pytest.mark.parametrize("t", [0, 1000, None])
def test_threshold_at_edges(t):
    """Threshold at lo, at hi, or never: exact answer, k bound, DP bound."""
    oracle = FakeOracle(t)
    res = find_threshold(oracle, 0, 1000, 2)
    assert res.threshold == t
    assert oracle.failures <= 2
    assert oracle.probes <= min_drops(2, 1001)


def test_non_monotone_oracle_stays_in_budget():
    """A lying target cannot drive more failures or probes than budgeted."""
    calls = []

    def flaky(x):
        calls.append(x)
        return len(calls) % 2 == 1  # alternate regardless of x

    find_threshold(flaky, 0, 500, 3)
    assert sum(1 for i in range(len(calls)) if i % 2 == 0) <= 3
    assert len(calls) <= min_drops(3, 501)
    assert all(0 <= x <= 500 for x in calls)


def test_huge_range_bounded():
    """2^40 floors: few probes, small table, no overflow."""
    hi = 1 << 40
    oracle = FakeOracle(hi - 12345)
    res = find_threshold(oracle, 1, hi, 8)
    assert res.threshold == hi - 12345
    assert oracle.probes == res.probes <= min_drops(8, hi)


@pytest.mark.parametrize("args", [(0, 10, 0), (5, 4, 1), (0, 10, -1)])
def test_rejects_bad_arguments(args):
    with pytest.raises(ValueError):
        find_threshold(FakeOracle(None), *args)


def test_min_drops_degenerate():
    assert min_drops(3, 0) == 0
    assert min_drops(50, 1) == 1
    assert min_drops(1, 10**9) == 10**9


# ── CLI: probe-size ────────────────────────────────────────────────────

SEED = b"0123456789"


def _target(tmp_path, body):
    path = tmp_path / "target.sh"
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(0o755)
    return str(path)


def _probe_args(target, seed, **kw):
    import argparse

    base = dict(
        target=target,
        seed=str(seed),
        max_repeat=64,
        failures=2,
        timeout=0.5,
        file_mode=False,
        target_args=None,
    )
    base.update(kw)
    return argparse.Namespace(**base)


def test_cli_probe_size_fake_exec(monkeypatch, tmp_path, capsys):
    """Timeout once input reaches 23 repeats; exact answer, <= 2 timeouts."""
    from fuzzer_tool.cli import commands

    seed = tmp_path / "seed"
    seed.write_bytes(SEED)
    sizes = []

    def fake_exec(_args, data, _env):
        sizes.append(len(data))
        return (-1, "") if len(data) >= 23 * len(SEED) else (0, "")

    monkeypatch.setattr(commands, "_replay_exec", fake_exec)
    rc = commands.cmd_probe_size(_probe_args(_target(tmp_path, "cat >/dev/null"), seed))

    assert rc == 0
    assert "threshold: 23 repeats" in capsys.readouterr().out
    assert sum(1 for s in sizes if s >= 23 * len(SEED)) <= 2
    assert len(sizes) <= min_drops(2, 64)


def test_cli_probe_size_real_target(tmp_path, capsys):
    """Real process: a target that hangs on >= 300 bytes (30 repeats)."""
    from fuzzer_tool.cli.commands import cmd_probe_size

    seed = tmp_path / "seed"
    seed.write_bytes(SEED)
    body = 'n=$(wc -c); [ "$n" -ge 300 ] && sleep 5; exit 0'
    rc = cmd_probe_size(_probe_args(_target(tmp_path, body), seed, max_repeat=40, timeout=0.3))
    assert rc == 0
    assert "threshold: 30 repeats" in capsys.readouterr().out


def test_cli_probe_size_never_fails(tmp_path, capsys):
    from fuzzer_tool.cli.commands import cmd_probe_size

    seed = tmp_path / "seed"
    seed.write_bytes(SEED)
    rc = cmd_probe_size(_probe_args(_target(tmp_path, "cat >/dev/null"), seed, max_repeat=8))
    assert rc == 1
    assert "no failure" in capsys.readouterr().out


@pytest.mark.parametrize("content", [None, b""])
def test_cli_probe_size_bad_seed(tmp_path, content):
    from fuzzer_tool.cli.commands import cmd_probe_size

    seed = tmp_path / "seed"
    if content is not None:
        seed.write_bytes(content)
    target = _target(tmp_path, "cat >/dev/null")
    assert cmd_probe_size(_probe_args(target, seed)) == 1


@pytest.mark.parametrize("kw", [{"failures": 0}, {"max_repeat": 0}, {"max_repeat": 1 << 40}])
def test_cli_probe_size_bad_bounds(tmp_path, kw):
    from fuzzer_tool.cli.commands import cmd_probe_size

    seed = tmp_path / "seed"
    seed.write_bytes(SEED)
    target = _target(tmp_path, "cat >/dev/null")
    assert cmd_probe_size(_probe_args(target, seed, **kw)) == 1


def test_cli_probe_size_registered(monkeypatch):
    """`fuzzer-tool probe-size` parses and dispatches."""
    from fuzzer_tool.cli import commands

    seen = {}

    def fake(a):
        seen["a"] = a
        return 0

    monkeypatch.setattr(commands, "cmd_probe_size", fake)
    argv = ["fuzzer-tool", "probe-size", "/bin/true", "s", "--max-repeat", "9", "-k", "3"]
    monkeypatch.setattr("sys.argv", argv)
    assert commands.main() == 0
    assert seen["a"].max_repeat == 9 and seen["a"].failures == 3
