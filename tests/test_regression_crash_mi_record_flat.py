"""``CrashMITracker.record`` gathers once through flat indices.

The 2-D fancy index ran twice per exec (read in ``_bump_sums``, then the
``+=``), ``arange(n)`` was rebuilt each call, and ``_f_step`` took a full
``max()`` pass to pick its table path. Now one flat gather feeds both the
sums and the scatter, row offsets are cached, and a running count bound
picks the table path. Oracle: the pre-change record, frozen here.
"""

import numpy as np

from fuzzer_tool.core.analyzers import analyzer_crash_eta as m
from fuzzer_tool.core.analyzers.analyzer_crash_eta import CrashMITracker
from fuzzer_tool.core.rand_pool import RandPool

_SEED = 0xE7A
_N_INPUTS = 300
_MAX_LEN = 600
_CRASH_RATE = 0.1
_STATE = (
    "_byte_total_arr",
    "_joint_crash_arr",
    "_position_counts_arr",
    "_s_total",
    "_s_crash",
    "_s_clean",
    "_crash_rows",
)


def _oracle_record(t: CrashMITracker, input_bytes: bytes, is_crash: bool) -> None:
    """Pre-change record() + _bump_sums(), verbatim."""
    t.total_execs += 1
    if is_crash:
        t.total_crashes += 1
        t._cache_valid = False
    n = min(len(input_bytes), t.max_positions, t.DENSE_POSITION_CAP)
    if not n:
        return
    t._grow(n)
    values = np.frombuffer(bytes(input_bytes), dtype=np.uint8, count=n)
    rows = np.arange(n)
    total = t._byte_total_arr[rows, values]
    crash = t._joint_crash_arr[rows, values]
    t._s_total[:n] += m._f_step(total)
    if is_crash:
        t._s_crash[:n] += m._f_step(crash)
        t._crash_rows[:n] += 1
    else:
        t._s_clean[:n] += m._f_step(total - crash)
    t._byte_total_arr[rows, values] += 1
    t._position_counts_arr[:n] += 1
    if is_crash:
        t._joint_crash_arr[rows, values] += 1


def _stream(rng: RandPool) -> list[tuple[bytes, bool]]:
    out = [
        (rng.randbytes(rng.randrange(_MAX_LEN)), rng.random() < _CRASH_RATE)
        for _ in range(_N_INPUTS)
    ]
    # Repeats push single cells up; empty input records nothing.
    return out + [(b"\x00" * 50, False)] * 20 + [(b"", True)]


def _assert_same(a: CrashMITracker, b: CrashMITracker) -> None:
    for name in _STATE:
        assert np.array_equal(getattr(a, name), getattr(b, name)), name
    assert (a.total_execs, a.total_crashes) == (b.total_execs, b.total_crashes)


def test_control_oracle_matches_itself():
    """Rule 46: two oracle runs on one stream agree."""
    stream = _stream(RandPool(seed=_SEED))
    a, b = CrashMITracker(), CrashMITracker()
    for x, c in stream:
        _oracle_record(a, x, c)
        _oracle_record(b, x, c)
    _assert_same(a, b)


def test_regression_record_matches_oracle():
    """Falsification: identical histograms and closed-form sums."""
    stream = _stream(RandPool(seed=_SEED))
    a, b = CrashMITracker(), CrashMITracker()
    for x, c in stream:
        _oracle_record(a, x, c)
        b.record(x, c)
    _assert_same(a, b)


def test_regression_record_skips_max_pass(monkeypatch):
    """Falsification: small counts index the step table without calling
    _f_step (whose max() pass was the cost)."""
    calls = {"n": 0}
    real = m._f_step

    def counting(k):
        calls["n"] += 1
        return real(k)

    monkeypatch.setattr(m, "_f_step", counting)
    t = CrashMITracker()
    for x, c in _stream(RandPool(seed=_SEED))[:20]:
        t.record(x, c)
    assert calls["n"] == 0


def test_regression_record_large_loaded_counts():
    """Adversarial: loaded counts at and past the table size take the
    generic path and still match the oracle across the boundary."""
    size = m._STEP_TABLE_SIZE
    state = CrashMITracker()
    state.record(b"\x07\x08", False)
    saved = state.save()
    saved["total_execs"] = size + 5

    a, b = CrashMITracker(), CrashMITracker()
    for t in (a, b):
        t.load(saved)
        # One cell just under the table, one at it.
        t._byte_total_arr[0, 7] = size - 1
        t._byte_total_arr[1, 8] = size
        t._rebuild_sums()
    b._count_bound = size  # what load() derives from these cells

    for x, c in [(b"\x07\x08", False), (b"\x07\x08", True), (b"\x07\x09", False)]:
        _oracle_record(a, x, c)
        b.record(x, c)
    _assert_same(a, b)


def test_regression_load_sets_count_bound():
    """Adversarial: load() derives the bound from the restored cells, so a
    saved count past the table cannot index it."""
    t = CrashMITracker()
    for _ in range(3):
        t.record(b"\x01\x02\x03", True)
    u = CrashMITracker()
    u.load(t.save())
    assert u._count_bound >= int(u._byte_total_arr.max())
    assert u._count_bound >= int(u._joint_crash_arr.max())
