"""Regression: havoc and swap_regions must never grow buf past max_len.

``_apply_single_mutation``'s ``swap regions`` branch (and the standalone
``_op_swap_regions`` handler, used outside havoc via the scheduler) bounded
the swapped region's ``size`` by ``j - i`` but not by ``len(buf) - j``. When
``j + size`` exceeded ``len(buf)``, ``buf[j : j + size]`` came back shorter
than ``size``, and the mismatched-length slice assignment changed buf's
total length instead of swapping two equal-sized regions -- observed
overshooting max_len by up to several bytes in a single havoc call.

``_op_swap_regions`` mutates in place and returns None, so nothing
downstream re-clamped it: unlike every other operator in this module, which
clamps its replacement to ``[: f.max_len]``, an in-place operator's output
size was never checked at all before this fix.
"""

from unittest.mock import patch

from fuzzer_tool.services.fuzzer import Fuzzer
from fuzzer_tool.services.operators import OperatorEngine
from tests.test_new_operators import _make_minimal_fuzzer


def _make_real_fuzzer(**kwargs):
    """A real Fuzzer instance, for exercising the full mutate() orchestrator
    (the swap_regions-specific in-place-clamp fix lives there, not in the
    lightweight mock fixture)."""
    import tempfile

    tmpdir = tempfile.mkdtemp(prefix="fuzz_test_")
    defaults = dict(
        target="/bin/true",
        corpus_dir=f"{tmpdir}/corpus",
        crashes_dir=f"{tmpdir}/crashes",
        max_len=10,
        timeout=1,
        mutations_per_input=1,
    )
    defaults.update(kwargs)
    with (
        patch("os.path.isfile", return_value=True),
        patch("os.access", return_value=True),
    ):
        return Fuzzer(**defaults)


class TestHavocMaxLen:
    def setup_method(self):
        self.fuzzer = _make_minimal_fuzzer()
        self.engine = OperatorEngine(self.fuzzer)

    def test_havoc_never_exceeds_max_len_at_boundary(self):
        # Start buffers already at max_len, so any 1-byte insert bug would
        # have to be caught by the per-op guard, and any swap-regions bug
        # would have to be caught by the growth it causes.
        for max_len in (1, 2, 3, 4, 8, 20, 65536):
            self.fuzzer.max_len = max_len
            for _ in range(200):
                buf = bytearray((i * 37) % 256 for i in range(max_len))
                self.engine.havoc_mutate(buf)
                assert len(buf) <= max_len

    def test_havoc_never_exceeds_max_len_stall_recovery(self):
        # Stall recovery runs 8-16 mutations per call instead of 2-8,
        # compounding any per-iteration overshoot.
        self.fuzzer._stall_recovery_active = True
        for max_len in (1, 4, 20):
            self.fuzzer.max_len = max_len
            for _ in range(200):
                buf = bytearray((i * 37) % 256 for i in range(max_len))
                self.engine.havoc_mutate(buf)
                assert len(buf) <= max_len

    def test_op_swap_regions_never_grows_buffer(self):
        for length in (4, 5, 8, 10, 17, 64):
            for _ in range(200):
                buf = bytearray((i * 53) % 256 for i in range(length))
                self.engine._op_swap_regions(buf, 0, b"")
                assert len(buf) == length

    def test_mutate_never_exceeds_max_len_via_swap_regions_only(self):
        # Force every round to select swap_regions specifically, and drive
        # the full mutate() orchestrator (the path with no clamp for
        # in-place handlers before this fix).
        f = _make_real_fuzzer(max_len=10)
        engine = OperatorEngine(f)
        engine.select_op = lambda _ops: "swap_regions"  # noqa: E731
        data = bytes((i * 41) % 256 for i in range(10))
        for _ in range(300):
            result = engine.mutate(data)
            assert len(result) <= f.max_len
