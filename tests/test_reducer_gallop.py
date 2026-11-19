"""Galloping prefix truncation (core/reducer.py ``PrefixPass``).

Fake oracles count execs: the pass must find the exact shortest crashing
prefix in O(log n) runs, spend one run when truncation cannot help, and
never accept a prefix the oracle rejected (non-monotone adversary).
"""

import math

import pytest

from fuzzer_tool.core.reducer import (
    ChunkPass,
    Oracle,
    Phase,
    Prefix,
    PrefixPass,
    Reducer,
    Verdict,
    shrink_passes,
)


def _min_len(need: int):
    """Monotone oracle: any prefix of at least ``need`` bytes crashes."""
    return lambda d: Verdict.PASS if len(d) >= need else Verdict.FAIL


def _gallop(data: bytes, test) -> tuple[bytes, Oracle]:
    oracle = Oracle(test)
    out = Reducer(oracle, [(Phase.FIRST, PrefixPass())]).run(data)
    return out, oracle


def _probe_bound(n: int) -> int:
    """Doubling probes + bisection probes, each <= ceil(log2 n) + 1."""
    return 2 * (math.ceil(math.log2(n)) + 1)


def _linear_min(n: int, test) -> int:
    """Reference: shortest passing prefix length by linear scan."""
    return next(k for k in range(1, n + 1) if test(bytes(k)) is Verdict.PASS)


@pytest.mark.parametrize("n", [1, 2, 3, 7, 64, 100, 1000, 4097])
@pytest.mark.parametrize("frac", [0.0, 0.01, 0.5, 0.99, 1.0])
def test_exact_min_prefix(n, frac):
    need = max(1, min(n, round(n * frac)))
    data = (bytes(range(256)) * (n // 256 + 1))[:n]
    out, oracle = _gallop(data, _min_len(need))
    assert len(out) == need
    assert oracle.runs <= _probe_bound(max(n, 2))


def test_matches_linear_reference_exhaustive():
    # Control first (Hard Rule 46): the reference agrees with itself.
    for n in range(1, 41):
        for need in range(1, n + 1):
            ref = _linear_min(n, _min_len(need))
            assert ref == _linear_min(n, _min_len(need)) == need

            out, _ = _gallop(bytes(n), _min_len(need))
            assert len(out) == ref


def test_falsification_no_gain_costs_one_run():
    """Truncation cannot help (last byte needed): one probe, input kept."""
    data = b"A" + b"." * 500 + b"Z"
    out, oracle = _gallop(data, lambda d: Verdict.PASS if d.endswith(b"Z") else Verdict.FAIL)
    assert out == data
    assert oracle.runs == 1


def test_adversarial_non_monotone():
    """Passing lengths are scattered: result is oracle-accepted, runs bounded."""
    n = 1000
    ok = {n, 999, 997, 640, 3}
    seen: list[int] = []

    def test(d):
        seen.append(len(d))
        return Verdict.PASS if len(d) in ok else Verdict.FAIL

    out, oracle = _gallop(bytes(n), test)
    assert len(out) in ok
    assert len(out) < n
    assert oracle.runs <= _probe_bound(n)
    assert 0 not in seen


def test_adversarial_also_verdict_not_accepted():
    """A different bug on a prefix is reported, never kept."""
    also: list[bytes] = []

    def test(d):
        if len(d) < 10:
            return Verdict.ALSO
        return Verdict.PASS

    oracle = Oracle(test, on_also=also.append)
    out = Reducer(oracle, [(Phase.FIRST, PrefixPass())]).run(bytes(100))
    assert len(out) == 10
    assert also and all(len(a) < 10 for a in also)


def test_empty_and_single_byte_untouched():
    for data in (b"", b"x"):
        out, oracle = _gallop(data, lambda d: Verdict.PASS)
        assert out == data
        assert oracle.runs == 0


def test_shrink_passes_gallop_opt_in():
    """Default keeps the old passes (gallop loses on mid-buffer bugs)."""
    assert not any(isinstance(p, PrefixPass) for _, p in shrink_passes())
    first = [type(p) for phase, p in shrink_passes(Prefix.GALLOP) if phase is Phase.FIRST]
    assert first == [PrefixPass]


def test_tmin_gallop_wired(tmp_path):
    """tmin(prefix=GALLOP) truncates a long tail in fewer target runs."""
    from unittest.mock import patch

    from fuzzer_tool.services.tmin import tmin

    crash = tmp_path / "crash.bin"
    crash.write_bytes(b"HDRBUG" + bytes(range(256)) * 32)
    runs = {}

    for mode in Prefix:
        calls: list[bytes] = []

        def fake_run(target, data, timeout, env=None, calls=calls):
            calls.append(data)
            return (-11, "", 1) if b"BUG" in data else (0, "", 1)

        with patch("fuzzer_tool.adapters.process.run_target_stdin", side_effect=fake_run):
            out = tmin("/bin/true", str(crash), max_stages=10_000, prefix=mode)
        assert out == b"BUG"
        runs[mode] = len(calls)

    assert runs[Prefix.GALLOP] < runs[Prefix.KEEP]


def test_cli_flag_reaches_tmin(tmp_path, monkeypatch):
    import sys

    from fuzzer_tool.cli.commands import main

    seen = {}
    monkeypatch.setattr("fuzzer_tool.services.tmin.tmin", lambda **kw: seen.update(kw) or b"x")
    out = str(tmp_path / "out.bin")
    argv = ["fuzzer-tool", "tmin", "/bin/true", str(tmp_path / "c.bin"), "-O", out]
    monkeypatch.setattr(sys, "argv", argv)
    assert main() == 0
    assert seen["prefix"] is Prefix.KEEP

    monkeypatch.setattr(sys, "argv", [*argv, "--gallop-prefix"])
    assert main() == 0
    assert seen["prefix"] is Prefix.GALLOP


def test_fewer_runs_than_chunk_only_on_long_tail():
    """Crash in a short prefix of a long input: gallop saves ChunkPass work."""
    data = b"HDR" + b"BUG" + bytes(range(256)) * 16
    pred = lambda d: Verdict.PASS if b"BUG" in d else Verdict.FAIL  # noqa: E731

    base = Oracle(pred)
    base_out = Reducer(base, [(Phase.MAIN, ChunkPass())]).run(data)
    new = Oracle(pred)
    new_out = Reducer(new, [(Phase.FIRST, PrefixPass()), (Phase.MAIN, ChunkPass())]).run(data)

    assert base_out == new_out == b"BUG"
    assert new.runs < base.runs


def test_never_worse_by_more_than_one_probe():
    """Truncation useless (both ends needed): at most one extra run."""
    data = b"A" + bytes(range(256)) * 4 + b"Z"
    pred = lambda d: Verdict.PASS if d[:1] == b"A" and d[-1:] == b"Z" else Verdict.FAIL  # noqa: E731

    base = Oracle(pred)
    Reducer(base, [(Phase.MAIN, ChunkPass())]).run(data)
    new = Oracle(pred)
    Reducer(new, [(Phase.FIRST, PrefixPass()), (Phase.MAIN, ChunkPass())]).run(data)

    assert new.runs <= base.runs + 1
