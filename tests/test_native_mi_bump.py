"""``fs_mi_bump``: CrashMITracker.record's per-exec update in one C pass.

Oracle: the numpy path (``_bump_sums``), forced by hiding the kernel at
construction. Same step table and same per-element addition order, so the
table path must match bit for bit.
"""

from unittest.mock import patch

import numpy as np
import pytest

from fuzzer_tool.adapters import native_scan
from fuzzer_tool.core.analyzers import analyzer_crash_eta as m
from fuzzer_tool.core.analyzers.analyzer_crash_eta import CrashMITracker
from fuzzer_tool.core.rand_pool import RandPool

_SEED = 0x3B1
_N_INPUTS = 400
_MAX_LEN = 700
_CRASH_RATE = 0.15
_STATE = (
    "_byte_total_arr",
    "_joint_crash_arr",
    "_position_counts_arr",
    "_s_total",
    "_s_crash",
    "_s_clean",
    "_crash_rows",
)

pytestmark = pytest.mark.skipif(native_scan.load() is None, reason="no C compiler")


def _numpy_tracker(**kw) -> CrashMITracker:
    with patch.object(native_scan, "load", return_value=None):
        return CrashMITracker(**kw)


def _stream(seed: int) -> list[tuple[bytes, bool]]:
    rng = RandPool(seed)
    return [
        (rng.randbytes(rng.randrange(_MAX_LEN)), rng.random() < _CRASH_RATE)
        for _ in range(_N_INPUTS)
    ]


def _feed(t: CrashMITracker, stream) -> None:
    for data, crash in stream:
        t.record(data, crash)


def _assert_same(a: CrashMITracker, b: CrashMITracker) -> None:
    assert a._rows == b._rows
    for name in _STATE:
        assert np.array_equal(getattr(a, name), getattr(b, name)), name
    assert (a.total_execs, a.total_crashes) == (b.total_execs, b.total_crashes)


def test_control_numpy_matches_itself():
    """A/A control (Hard Rule 46): the oracle agrees with a second run of itself."""
    stream = _stream(_SEED)
    a, b = _numpy_tracker(), _numpy_tracker()
    _feed(a, stream)
    _feed(b, stream)
    _assert_same(a, b)


def test_kernel_is_bound():
    """Falsification: the default tracker really takes the C path."""
    t = CrashMITracker()
    assert t._kernel is not None
    assert t._bump == t._kernel.bump

    ref = _numpy_tracker()
    assert ref._kernel is None
    assert ref._bump == ref._bump_np


def test_kernel_matches_numpy_bitwise():
    stream = _stream(_SEED)
    native, ref = CrashMITracker(), _numpy_tracker()
    _feed(native, stream)
    _feed(ref, stream)
    _assert_same(native, ref)
    assert native.all_mi() == ref.all_mi()


def test_clamped_to_max_positions():
    """Edge: inputs longer than max_positions write only the tracked rows."""
    stream = [(b"\xff" * 1000, True), (bytes(range(256)) * 4, False), (b"", True)]
    native, ref = CrashMITracker(max_positions=64), _numpy_tracker(max_positions=64)
    _feed(native, stream)
    _feed(ref, stream)
    _assert_same(native, ref)
    assert native._rows == 64


def test_bytearray_input():
    """Edge: callers may pass bytearray / memoryview."""
    native, ref = CrashMITracker(), _numpy_tracker()
    for t in (native, ref):
        t.record(bytearray(b"abc\x00def"), True)
        t.record(memoryview(b"xyz"), False)
    _assert_same(native, ref)


def test_rebinds_after_load_and_growth():
    """Adversarial: load() and _grow() replace every array; no stale pointers."""
    first, second = _stream(_SEED), _stream(_SEED + 1)
    src = _numpy_tracker()
    _feed(src, first)
    state = src.save()

    native, ref = CrashMITracker(), _numpy_tracker()
    native.load(state)
    ref.load(state)
    _feed(native, second)
    _feed(ref, second)
    _assert_same(native, ref)


def test_step_fallback_past_table():
    """Adversarial: counts >= the step table take the log2 formula in C."""
    big = m._STEP_TABLE_SIZE + 5
    native, ref = CrashMITracker(), _numpy_tracker()
    for t in (native, ref):
        t.record(b"\x07\x07", False)
        t._byte_total_arr[0, 7] = big
        t._joint_crash_arr[0, 7] = big - 3
        t._count_bound = big
        t.record(b"\x07\x07", True)
        t.record(b"\x07", False)

    expect = (big + 1) * np.log2(big + 1) - big * np.log2(big)
    assert native._s_total[0] - ref._s_total[0] == pytest.approx(0.0, abs=1e-9)
    assert native._s_crash[0] == pytest.approx(ref._s_crash[0], rel=1e-12)
    assert native._s_clean[0] == pytest.approx(ref._s_clean[0], rel=1e-12)
    assert ref._s_total[0] > expect  # the formula path really ran
    for name in ("_byte_total_arr", "_joint_crash_arr", "_position_counts_arr", "_crash_rows"):
        assert np.array_equal(getattr(native, name), getattr(ref, name)), name
