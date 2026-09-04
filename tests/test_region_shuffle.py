"""`region_shuffle`: reorder records inside one region, keep the offset true.

The operator exists so that an ordering mutation can be attributed to a
region honestly (see `_DELOCALISED_OPS` in services/operators.py). These are
the operator-level invariants; the end-to-end verdict measurement against a
target with known ground truth lives in tests/test_region_order_attribution.py.

Falsification, checked by actually breaking each one: widening the write past
the region fails `test_only_the_selected_region_changes`; dropping the
`hi = min(hi, len(buf))` clamp fails `test_stride_is_derived_from_the_clamped_span`.

The identity-draw early return has NO behavioural difference to assert on --
the handler returns None either way and the effectiveness tracker hashes the
buffer -- so it is pinned at source level and labelled as such rather than
wrapped in a test that would pass with the guard removed.
"""

import inspect
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from fuzzer_tool.services.operators import OperatorEngine  # noqa: E402
from support.operator_env import make_minimal_fuzzer  # noqa: E402

SEED_LEN = 16384
REGION = 4096


def _engine(seed=7):
    """Engine with its ``ctx`` cache primed.

    ``OperatorEngine.ctx`` is a property that rebuilds on every access when
    the cache is cold, so assigning to ``eng.ctx.<attr>`` would write to a
    throwaway object and the test would pass for the wrong reason. Priming
    ``_ctx_cache`` is what makes those assignments stick.
    """
    eng = OperatorEngine(make_minimal_fuzzer(seed=seed))
    eng._ctx_cache = eng.ctx
    return eng


def _seed_bytes(n=SEED_LEN):
    # Structured, not random: profile_buffer must return a real profile, and a
    # reordering has to be observable as a byte diff.
    return bytes(((i // 512) * 7 + (i % 251)) % 256 for i in range(n))


class TestRegionShuffle:
    def test_regions_exist_for_this_seed_length(self):
        """Guard the premise the rest of the file rests on."""
        eng = _engine()
        entry = eng.region_weights(_seed_bytes())
        assert entry is not None
        _cum, bounds, _tot = entry
        assert len(bounds) >= 2, bounds

    def test_only_the_selected_region_changes(self):
        eng = _engine()
        data = _seed_bytes()
        _cum, bounds, _tot = eng.region_weights(data)
        for idx, (lo, _hi) in enumerate(bounds):
            buf = bytearray(data)
            changed = False
            for _ in range(40):
                buf = bytearray(data)
                eng._op_region_shuffle(buf, lo + 1, data)
                if buf != bytearray(data):
                    changed = True
                    break
            assert changed, f"region {idx} never mutated in 40 draws"
            for j, (o_lo, o_hi) in enumerate(bounds):
                if j == idx:
                    continue
                assert bytes(buf[o_lo:o_hi]) == data[o_lo:o_hi], f"region {j} touched"

    def test_length_is_preserved(self):
        eng = _engine()
        data = _seed_bytes()
        for _ in range(50):
            buf = bytearray(data)
            eng._op_region_shuffle(buf, 100, data)
            assert len(buf) == len(data)

    def test_output_is_a_permutation_of_the_region(self):
        """A reordering, not a rewrite: the multiset of bytes is invariant."""
        eng = _engine()
        data = _seed_bytes()
        for _ in range(30):
            buf = bytearray(data)
            eng._op_region_shuffle(buf, 10, data)
            assert sorted(buf) == sorted(data)

    def test_identity_draw_is_skipped_at_source(self):
        """Pinned at source: there is no behavioural difference to assert.

        Both paths return None and leave the buffer unchanged, so a
        behavioural test here would pass with the early return deleted --
        checked by deleting it. The value of the guard is the skipped write.
        """
        assert "if order == list(range(n_records)):" in inspect.getsource(
            OperatorEngine._op_region_shuffle
        )

    def test_identity_draw_leaves_the_buffer_alone(self):
        """An identity permutation must return None and change nothing."""
        eng = _engine()
        data = _seed_bytes()

        original = eng._ctx_cache.rand_pool

        class _IdentityPool:
            def shuffle(self, seq):
                return None

            def __getattr__(self, name):
                return getattr(original, name)

        try:
            eng._ctx_cache.rand_pool = _IdentityPool()
            buf = bytearray(data)
            assert eng._op_region_shuffle(buf, 10, data) is None
            assert buf == bytearray(data)
        finally:
            eng._ctx_cache.rand_pool = original

    def test_short_seed_is_a_no_op(self):
        """Below the region-profile minimum there is nothing to confine to."""
        eng = _engine()
        data = b"\x01\x02\x03\x04" * 8
        buf = bytearray(data)
        assert eng._op_region_shuffle(buf, 2, data) is None
        assert buf == bytearray(data)

    def test_offset_outside_every_region_is_a_no_op(self):
        eng = _engine()
        data = _seed_bytes()
        buf = bytearray(data)
        assert eng._op_region_shuffle(buf, len(data) * 4, data) is None
        assert buf == bytearray(data)

    def test_stride_is_derived_from_the_clamped_span(self):
        """Bounds come from the parent; an earlier operator may have truncated.

        Adversarial: the last region of the parent profile extends well past
        the end of the buffer handed to the operator. Clamped, the 100 live
        bytes are cut into 8 records of 12; unclamped, stride would be 512 and
        every record after the first would be empty, so the "reordering" would
        be the identity on the only non-empty slice. Asserting that the region
        actually changes is what separates the two.
        """
        eng = _engine()
        data = _seed_bytes()
        _cum, bounds, _tot = eng.region_weights(data)
        last_lo, _last_hi = bounds[-1]
        tail = 100
        changed = False
        for _ in range(60):
            buf = bytearray(data[: last_lo + tail])
            eng._op_region_shuffle(buf, last_lo + 1, data)
            assert len(buf) == last_lo + tail
            assert bytes(buf[:last_lo]) == data[:last_lo]
            if bytes(buf[last_lo:]) != data[last_lo : last_lo + tail]:
                changed = True
                assert sorted(buf[last_lo:]) == sorted(data[last_lo : last_lo + tail])
        assert changed, "clamped stride should let the surviving tail be reordered"

    @pytest.mark.parametrize("stride", [64, 512, 1024])
    def test_inferred_record_stride_is_honoured(self, stride):
        """With a parent stride the region is a permutation of stride blocks.

        Asserted on the block multiset rather than on the first differing
        byte: a permutation is free to leave the leading records in place, so
        "the first diff sits on a boundary" is not an invariant.
        """
        eng = _engine()
        data = _seed_bytes()
        eng._ctx_cache.seed_meta = {data: {"record_stride": stride}}
        _cum, bounds, _tot = eng.region_weights(data)
        lo, hi = bounds[0]
        n = (hi - lo) // stride
        original = [data[lo + r * stride : lo + (r + 1) * stride] for r in range(n)]
        for _ in range(40):
            buf = bytearray(data)
            eng._op_region_shuffle(buf, lo + 1, data)
            if buf == bytearray(data):
                continue
            got = [bytes(buf[lo + r * stride : lo + (r + 1) * stride]) for r in range(n)]
            assert sorted(got) == sorted(original)
            assert got != original
            return
        pytest.fail("no mutation observed in 40 draws")
