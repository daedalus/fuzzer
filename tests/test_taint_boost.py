"""Revizor input boosting: path-equivalent siblings from colorization taints.

Colorization proves a set of bytes path-irrelevant (``TaintRegion``s) by
replacing them with same-class values and checking the path held. ``boost``
redraws exactly those bytes from the same classes and keeps the rest, so
every sibling should take the seed's path (Revizor's ``generate_boosted``).
"""

from __future__ import annotations

import tempfile
from unittest.mock import patch

from fuzzer_tool.core.colorization import TaintRegion, _taint_index, boost, colorize
from fuzzer_tool.core.mutations import type_replace_byte
from fuzzer_tool.core.mutations.generic import _SWAP_MAP, _in_class
from fuzzer_tool.core.operator_registry import REGISTRY
from fuzzer_tool.core.rand_pool import RandPool
from tests.support.scripted_rng import ScriptedRng

_UPPER = (0x47, 0x5A)  # G-Z: a 20-member class in type_replace_byte


def _boost_loop(data: bytes, taints, rng) -> bytes:
    """Reference oracle: one ``type_replace_byte`` per tainted byte."""
    out = bytearray(data)
    for i in _taint_index(len(data), taints).tolist():
        out[i] = type_replace_byte(out[i], rng)
    return bytes(out)


def _u16(*values: int) -> bytes:
    return b"".join(v.to_bytes(2, "little") for v in values)


class TestBoostExact:
    def test_class_byte_follows_the_shift_arithmetic(self):
        # 'M' (0x4D) in G-Z: size = 0x5A - 0x47 = 19, u = (r * 19) >> 16.
        start, end = _UPPER
        size = end - start
        r = 0x8000
        u = (r * size) >> 16
        expected = start + (0x4D - start + 1 + u) % (size + 1)

        rng = ScriptedRng(randbytes=[_u16(r)])
        assert boost(b"aMb", [TaintRegion(1, 1)], rng) == bytes([0x61, expected, 0x62])

    def test_fixed_bytes_match_type_replace_byte(self):
        # Swap-map and out-of-class bytes draw nothing in type_replace_byte.
        data = bytes([0x00, 0x0A, 0x2B, 0x10, 0x90, 0xFF])
        out = boost(data, [TaintRegion(0, len(data) - 1)], ScriptedRng(randbytes=[b""]))
        assert out == bytes(type_replace_byte(b, None) for b in data)

    def test_bytes_outside_taints_are_untouched(self):
        data = bytes(range(64, 96))
        out = boost(data, [TaintRegion(4, 7), TaintRegion(20, 21)], RandPool(1))
        keep = [i for i in range(len(data)) if not (4 <= i <= 7 or 20 <= i <= 21)]
        assert all(out[i] == data[i] for i in keep)

    def test_every_tainted_byte_changes_within_its_class(self):
        data = b"GHIJKLMNOPQRSTUVWXYZabcdef234567"
        out = boost(data, [TaintRegion(0, len(data) - 1)], RandPool(7))
        assert all(o != d for o, d in zip(out, data, strict=True))
        assert all(
            (0x47 <= o <= 0x5A) == (0x47 <= d <= 0x5A) for o, d in zip(out, data, strict=True)
        )

    def test_same_seed_same_sibling(self):
        data = b"hello world 12345"
        taints = [TaintRegion(0, 4), TaintRegion(12, 16)]
        assert boost(data, taints, RandPool(3)) == boost(data, taints, RandPool(3))

    def test_vectorized_agrees_with_the_loop_on_classes(self):
        # The loop is the type_replace_byte reference; the draws differ, the
        # class and the "always changes" contract must not.
        data = bytes(range(256))
        taints = [TaintRegion(0, 255)]
        fast, slow = boost(data, taints, RandPool(5)), _boost_loop(data, taints, RandPool(5))
        assert all(f != d and s != d for f, s, d in zip(fast, slow, data, strict=True))
        for b in data:
            if b in _SWAP_MAP or _in_class(b) is None:
                assert fast[b] == slow[b]
                continue
            lo, hi = _in_class(b)
            assert lo <= fast[b] <= hi and lo <= slow[b] <= hi


class TestBoostAdversarial:
    def test_empty_input(self):
        assert boost(b"", [TaintRegion(0, 3)], RandPool(1)) == b""

    def test_no_taints_is_identity(self):
        assert boost(b"abc", [], RandPool(1)) == b"abc"

    def test_regions_past_the_end_are_clipped(self):
        out = boost(b"abcd", [TaintRegion(2, 99), TaintRegion(50, 60)], RandPool(1))
        assert len(out) == 4 and out[:2] == b"ab" and out[2:] != b"cd"

    def test_overlapping_regions_redraw_once(self):
        # Overlap must not apply the shift twice (which could land on the original).
        data = b"MMMMMMMM"
        out = boost(data, [TaintRegion(0, 5), TaintRegion(3, 7)], RandPool(4))
        assert all(o != 0x4D for o in out)

    def test_inverted_region_is_ignored(self):
        assert boost(b"abcd", [TaintRegion(3, 1)], RandPool(1)) == b"abcd"


def _magic_digit_path(data: bytes) -> int:
    """Path of a fake target: magic header, then a digit check on byte 10."""
    if data[:4] != b"FUZZ":
        return 1
    return 2 if len(data) > 10 and 0x30 <= data[10] <= 0x39 else 3


class TestFalsification:
    """Hard Rule 23: boosting must preserve the path, and the test must be able to fail."""

    SEED = b"FUZZ abcde5 tail bytes here"

    def _taints(self):
        return colorize(self.SEED, _magic_digit_path, max_execs=256, rng=RandPool(2)).taints

    def test_boosted_siblings_keep_the_path(self):
        taints = self._taints()
        assert taints
        want = _magic_digit_path(self.SEED)
        for seed in range(8):
            assert _magic_digit_path(boost(self.SEED, taints, RandPool(seed))) == want

    def test_uniform_redraw_of_the_same_bytes_breaks_the_path(self):
        # Control: a class-blind redraw of byte 10 leaves the digit class.
        taints = self._taints()
        assert any(t.start <= 10 <= t.end for t in taints)
        broken = bytearray(self.SEED)
        broken[10] = 0x41
        assert _magic_digit_path(bytes(broken)) != _magic_digit_path(self.SEED)


def _fuzzer():
    from fuzzer_tool.services.fuzzer import Fuzzer

    tmp = tempfile.mkdtemp(prefix="boost_")
    with patch("os.path.isfile", return_value=True), patch("os.access", return_value=True):
        return Fuzzer(
            target="/bin/true",
            corpus_dir=f"{tmp}/c",
            crashes_dir=f"{tmp}/x",
            max_len=64,
            timeout=1,
            mutations_per_input=2,
        )


class TestOperator:
    SEED = b"header:ABCDEFGH"

    def test_registered_as_adaptive(self):
        assert REGISTRY.category_of("taint_boost") == "adaptive"

    def test_unavailable_without_cached_taints(self):
        f = _fuzzer()
        assert "taint_boost" not in REGISTRY.available(f, self.SEED)

    def test_available_once_the_seed_was_colorized(self):
        f = _fuzzer()
        f._colorize_taint_cache[hash(self.SEED)] = [TaintRegion(7, 14)]
        assert "taint_boost" in REGISTRY.available(f, self.SEED)

    def test_failed_colorization_keeps_it_unavailable(self):
        # _colorize_seed caches None for "no usable answer".
        f = _fuzzer()
        f._colorize_taint_cache[hash(self.SEED)] = None
        assert "taint_boost" not in REGISTRY.available(f, self.SEED)

    def test_handler_redraws_only_the_taints(self):
        f = _fuzzer()
        f._colorize_taint_cache[hash(self.SEED)] = [TaintRegion(7, 14)]
        buf = bytearray(self.SEED)
        f._operators._op_taint_boost(buf, 0, self.SEED)
        assert buf[:7] == self.SEED[:7]
        assert all(
            b != s and 0x41 <= b <= 0x5A for b, s in zip(buf[7:], self.SEED[7:], strict=True)
        )

    def test_handler_without_taints_is_a_noop(self):
        f = _fuzzer()
        buf = bytearray(self.SEED)
        f._operators._op_taint_boost(buf, 0, self.SEED)
        assert buf == self.SEED
