"""Windowed region operators: ``_region(window=)``, the two operators that take
a window, and the engine hand-off.

Covers core/mutations/structured.py (_in_window, _region, cusum_bias_run,
monotone_fill) and OperatorEngine._regularity_windowed.
"""

from types import SimpleNamespace

from fuzzer_tool.core.mutations.structured import (
    MAX_REGION,
    _in_window,
    _region,
    cusum_bias_run,
    monotone_fill,
)
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.pos_kadane import WINDOW_P
from fuzzer_tool.services.operators import OperatorEngine


class ScriptedRng:
    """randint -> low bound, random() -> fixed, choice -> first."""

    def __init__(self, r=0.0):
        self.r = r
        self.draws = 0

    def randint(self, a, b):
        self.draws += 1
        return a

    def random(self):
        return self.r

    def choice(self, seq):
        return seq[0]


# -- _in_window / _region -----------------------------------------------------


def test_window_wider_than_min_is_used_as_is():
    assert _in_window(1000, (500, 100), 32, 1, None) == (500, 100)


def test_window_narrower_than_min_is_widened_around_its_centre():
    off, length = _in_window(1000, (500, 21), 32, 1, None)
    assert length == 32 and off <= 500 and off + length >= 521


def test_window_near_the_edges_is_shifted_not_truncated():
    assert _in_window(1000, (0, 4), 32, 1, None) == (0, 32)
    assert _in_window(1000, (996, 4), 32, 1, None) == (968, 32)


def test_window_longer_than_max_region_is_capped():
    off, length = _in_window(100_000, (0, 50_000), 32, 1, None)
    assert length == MAX_REGION and off + length <= 100_000


def test_window_respects_align_and_max_len():
    off, length = _in_window(1000, (501, 60), 8, 4, 8)
    assert off % 4 == 0 and length % 8 == 0 and length >= 8
    assert off + length <= 1000


def test_unusable_windows_return_none():
    assert _in_window(1000, (1000, 10), 32, 1, None) is None  # starts past the end
    assert _in_window(1000, (-1, 10), 32, 1, None) is None
    assert _in_window(1000, (10, 0), 32, 1, None) is None
    assert _in_window(16, (0, 8), 32, 1, None) is None  # buffer shorter than min_len


def test_region_uses_the_window_and_consumes_no_draw():
    rng = ScriptedRng()
    assert _region(1000, rng, min_len=32, window=(500, 100)) == (500, 100)
    assert rng.draws == 0


def test_region_falls_back_to_random_on_an_unusable_window():
    rng = ScriptedRng()
    off, length = _region(1000, rng, min_len=32, window=(5000, 10))
    assert rng.draws == 2 and length >= 32


def test_region_without_window_is_unchanged():
    a = _region(1000, RandPool(seed=3), min_len=8, align=2)
    b = _region(1000, RandPool(seed=3), min_len=8, align=2, window=None)
    assert a == b


# -- operators ------------------------------------------------------------------


def test_cusum_bias_run_overwrites_the_window():
    data = bytes(range(256)) * 4  # 1024 bytes, no run
    out = cusum_bias_run(data, ScriptedRng(), window=(500, 64))
    assert out[500:564] == bytes([out[500]]) * 64
    assert out[:500] == data[:500] and out[564:] == data[564:]
    assert len(out) == len(data)


def test_cusum_bias_run_without_window_still_works():
    out = cusum_bias_run(bytes(range(256)) * 4, RandPool(seed=1))
    assert len(out) == 1024


def test_monotone_fill_window_is_aligned_and_confined():
    data = bytes(range(256)) * 4
    out = monotone_fill(data, RandPool(seed=2), window=(500, 64))
    assert len(out) == len(data)
    changed = [i for i in range(len(data)) if out[i] != data[i]]
    if changed:  # a monotone fill can coincide with the data on some bytes
        assert min(changed) >= 496 and max(changed) < 500 + 64 + 8


# -- engine hand-off -------------------------------------------------------------


class _Kadane:
    def __init__(self, win):
        self.win = win
        self.calls = []

    def window(self, data, buf_len):
        self.calls.append((data, buf_len))
        return self.win


def _engine(kadane, r):
    e = OperatorEngine.__new__(OperatorEngine)
    e.f = SimpleNamespace(_pos_kadane=kadane)
    e._ctx_cache = SimpleNamespace(_rng=ScriptedRng(r), max_len=10_000)
    return e


def _spy():
    seen = {}

    def fn(data, rng, window=None):
        seen["window"] = window
        return data

    return fn, seen


def test_engine_passes_the_kadane_window_when_the_draw_is_below_window_p():
    k = _Kadane((500, 21))
    fn, seen = _spy()
    parent = bytes(1000)
    _engine(k, WINDOW_P - 0.01)._regularity_windowed(fn, bytearray(900), parent)
    assert seen["window"] == (500, 21)
    assert k.calls == [(parent, 900)]


def test_engine_stays_random_when_the_draw_is_above_window_p():
    k = _Kadane((500, 21))
    fn, seen = _spy()
    _engine(k, WINDOW_P + 0.01)._regularity_windowed(fn, bytearray(900), bytes(1000))
    assert seen["window"] is None and k.calls == []


def test_engine_stays_random_when_the_arm_is_off():
    fn, seen = _spy()
    _engine(None, 0.0)._regularity_windowed(fn, bytearray(900), bytes(1000))
    assert seen["window"] is None


def test_engine_handles_an_arm_with_no_run_yet():
    fn, seen = _spy()
    _engine(_Kadane(None), 0.0)._regularity_windowed(fn, bytearray(900), bytes(1000))
    assert seen["window"] is None


def test_engine_clamps_to_max_len():
    e = _engine(None, 0.0)
    e._ctx_cache.max_len = 10
    out = e._regularity_windowed(lambda d, rng, window=None: d, bytearray(100), bytes(100))
    assert isinstance(out, bytearray) and len(out) == 10
