"""Regressions for single-line fixes from docs/bugreport_2026-08-21_merged.md.

Each bug here was confirmed by the audit ([verified]/[corroborated]) and fixed
with a change no larger than one line. Each test fails against the pre-fix
source.
"""

from __future__ import annotations

import ctypes
import signal
from unittest.mock import patch

from fuzzer_tool.adapters.persistent import PersistentRunner
from fuzzer_tool.core.gf2_common import GF2n
from fuzzer_tool.core.mutations.generic import _NumNode
from fuzzer_tool.services.te_position import get_te_weighted_position


class TestGF2nPowZeroBase:
    """#85 gf2_common.py:192 -- `e %= self.m if a != 0 else e` parses as
    `e %= (self.m if a != 0 else e)`, so pow(0, e) reduces e mod itself
    instead of leaving it untouched, e.g. pow(0, e>0) wrongly returns 1."""

    def test_pow_zero_base_nonzero_exp_is_zero(self):
        f = GF2n(8)
        assert f.pow(0, 5) == 0

    def test_pow_zero_base_zero_exp_is_one(self):
        f = GF2n(8)
        assert f.pow(0, 0) == 1


class TestTEWeightedPosition:
    """#86 te_position.py:48 -- returned the highest byte POSITION instead of
    the position with the highest TE-weighted edge influence."""

    def test_returns_highest_weight_position_not_highest_index(self):
        byte_edges = {
            1: {20: 100, 21: 50},
            5: {10: 1},
        }
        assert get_te_weighted_position(byte_edges, input_length=64) == 1


class TestJpegScanLenClamp:
    """#74 jpeg.py:669 -- randint(1, min(256, max_len - len(buf) - 2)) raises
    ValueError when the upper bound drops below 1 for small max_len."""

    def test_small_max_len_does_not_raise(self):
        from fuzzer_tool.core.mutations.jpeg import JpegMutator

        m = JpegMutator()
        # A max_len tight enough that the pre-fix bound goes non-positive.
        out = m._generate_random_jpeg(max_len=30)
        assert isinstance(out, (bytes, bytearray))


class TestVersifierDecimalDigits:
    """#75 generic.py:1723-1728 -- _NumNode.Generate had no branch appending
    the generated digits to buf when base == 10, so decimal numbers were
    silently dropped."""

    def test_base_10_emits_digits(self):
        """Drive the RNG deterministically so base == 10 is selected with a
        known length and known digits, then assert buf equals exactly those
        digits (not empty, not octal/hex-prefixed)."""

        class _FakeRand:
            def __init__(self, randints, randoms):
                self._randints = iter(randints)
                self._randoms = iter(randoms)

            def random(self):
                return next(self._randoms)

            def randint(self, _a, _b):
                return next(self._randints)

        class _V:
            pass

        # Call order in _NumNode.Generate: random() [skip sample branch],
        # randint(0,2)->1 [base=10], randint(0,3), randint(0,15),
        # randint(0,39) [candidate lengths], randint(0,2)->0 [pick the
        # randint(0,3) candidate as length=2], then two digit draws, then a
        # final random() [skip trailing '-'].
        v = _V()
        v._rand = _FakeRand(randints=[1, 2, 0, 0, 0, 5, 3], randoms=[1, 1])

        node = _NumNode(samples=[])
        buf = bytearray()
        node.Generate(v, buf)

        assert bytes(buf) == b"53"


class TestPersistentSigcont:
    """#36 persistent.py:136-161 -- protocol docstring promises the runner
    resumes the target with SIGCONT after reading the result, but the
    SIGSTOP branch of run_one() never sent it, wedging the target forever
    after the first iteration."""

    def test_run_one_sends_sigcont_after_stop(self):
        pr = PersistentRunner(target="/bin/true", timeout=1)
        pr._started = True
        pr.pid = 4242
        pr.map_size = 4096
        buf = ctypes.create_string_buffer(4096)
        pr.shm_ptr = ctypes.addressof(buf)

        stopped_status = 0x7F | (signal.SIGSTOP << 8)
        kill_calls = []

        with (
            patch("os.kill", side_effect=lambda pid, sig: kill_calls.append((pid, sig))),
            patch("os.waitpid", return_value=(4242, stopped_status)),
        ):
            rc, _ = pr.run_one(b"x")

        assert rc == 0
        assert (4242, signal.SIGCONT) in kill_calls
