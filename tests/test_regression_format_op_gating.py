"""Regression tests: format-operator relevance gating.

The format operators (png/jpeg/webm/...) parse the input and, when it is NOT
that format, fall back to generating a whole random file of that format from
scratch. On a target that has nothing to do with those formats that fallback
burned ~50% of the execution budget building files the target rejects
instantly — gating the operators roughly doubled throughput (2213 -> ~4100
eps on targets/test_target.so).

These tests pin the three behaviors the gate must preserve:
  1. a real file of format F keeps F available (never break real mutation),
  2. once F has been seen it stays live for the rest of the run,
  3. an unseen F is only offered on a small fraction of selections, so
     bootstrap-from-garbage-corpus still works without dominating.
"""

from __future__ import annotations

import pytest

from fuzzer_tool.core.operator_registry import (
    _FORMAT_BOOTSTRAP_RATE,
    _FORMAT_SNIFFERS,
    REGISTRY,
)
from fuzzer_tool.core.rand_pool import RandPool

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 40
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 40
GZIP = b"\x1f\x8b" + b"\x00" * 40
PLAIN = b"hello world, definitely not an image"


class _Fuzzer:
    """Minimal stand-in — the gate only touches these two attributes."""

    def __init__(self):
        self._rand_pool = RandPool()


class TestFormatOpGating:
    @pytest.mark.parametrize(
        "data,op",
        [
            (PNG, "png_chunk_mutate"),
            (JPEG, "jpeg_chunk_mutate"),
            (GZIP, "gzip_chunk_mutate"),
        ],
    )
    def test_real_format_file_keeps_op_available(self, data, op):
        """A genuine file of format F must never have F gated away."""
        f = _Fuzzer()
        assert op in REGISTRY.available(f, data)

    def test_format_stays_live_after_first_sighting(self):
        """Once a real PNG is seen, png ops stay available for later
        non-PNG inputs — a run that handles PNGs must keep mutating them."""
        f = _Fuzzer()
        REGISTRY.available(f, PNG)  # first sighting
        for _ in range(50):
            assert "png_chunk_mutate" in REGISTRY.available(f, PLAIN)

    def test_unseen_format_is_throttled_not_disabled(self):
        """An unseen format is still offered occasionally (bootstrap), but
        far below the every-selection rate that caused the slowdown."""
        hits = 0
        trials = 4000
        f = _Fuzzer()
        for _ in range(trials):
            f._live_formats = set()  # fresh run each time: never seen
            if "jpeg_chunk_mutate" in REGISTRY.available(f, PLAIN):
                hits += 1
        rate = hits / trials
        # Not disabled outright...
        assert hits > 0
        # ...but nowhere near always-on (the pre-gate behavior).
        assert rate < _FORMAT_BOOTSTRAP_RATE * 4
        assert rate < 0.2

    def test_plain_input_excludes_most_format_ops(self):
        """The actual throughput win: a plain-text input should not pull in
        the whole format operator family on every selection."""
        f = _Fuzzer()
        avail = set(REGISTRY.available(f, PLAIN))
        gated = set(_FORMAT_SNIFFERS)
        # Allow the bootstrap trickle to admit a couple, but the bulk must
        # be excluded — before the gate, all of them were always present.
        assert len(gated & avail) < len(gated) // 2

    def test_none_fuzzer_is_tolerated(self):
        """REGISTRY.available(None, ...) is used by tests/tools; the format
        predicate must not raise on it."""
        assert isinstance(REGISTRY.available(None, b""), list)
