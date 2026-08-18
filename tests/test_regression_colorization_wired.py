"""Colorization must actually run, and must actually filter.

Two defects, both closed here:

* ``core/colorization.py::colorize()`` -- the real AFL++/Redqueen pass, which
  holds the execution path fixed while replacing everything it can -- was
  never called from ``src/``. It had tests and no caller.
* The ``colorization`` *operator* was gated on ``operator_registry._never``,
  justified only as "keeps historic build_ops behavior". That behaviour was
  an accident: the operator sat in the dispatch table and in no op list, so
  nothing could ever draw it.

The reason the pass matters is in the redqueen match loop, which accepts
every literal occurrence of a comparison operand as an input-to-state
candidate, filtered only by ``len(op_a) >= 2``. Short operands hit a large
input many times by chance. Colorization is what separates the occurrence the
target read from the ones it did not.
"""

from __future__ import annotations

import types

from fuzzer_tool.core.colorization import TaintRegion, colorize
from fuzzer_tool.core.operator_registry import REGISTRY
from fuzzer_tool.services.fuzzer import _in_taint


class TestInTaint:
    def test_occurrence_wholly_inside_a_taint_is_rejected(self):
        taints = [TaintRegion(start=10, end=40)]
        assert _in_taint(taints, 20, 4) is True

    def test_occurrence_outside_every_taint_is_kept(self):
        taints = [TaintRegion(start=10, end=40)]
        assert _in_taint(taints, 50, 4) is False

    def test_partial_overlap_is_kept(self):
        """If any byte of the occurrence is path-relevant, keep the match --
        dropping it would discard a real input-to-state candidate."""
        taints = [TaintRegion(start=10, end=40)]
        assert _in_taint(taints, 38, 6) is False  # runs past the taint end
        assert _in_taint(taints, 8, 6) is False  # starts before the taint

    def test_no_taints_filters_nothing(self):
        """None/empty is the disabled path and must behave as before."""
        assert _in_taint(None, 0, 4) is False
        assert _in_taint([], 0, 4) is False

    def test_boundaries_are_inclusive(self):
        taints = [TaintRegion(start=10, end=13)]
        assert _in_taint(taints, 10, 4) is True  # exactly fills [10, 13]
        assert _in_taint(taints, 10, 5) is False  # one byte past


class TestColorizeSeparatesRealOperandsFromCoincidence:
    def test_the_byte_the_target_reads_is_not_tainted(self):
        """A target that branches on one offset should leave that offset out
        of the taint set, and everything it ignores inside it."""
        data = b"AB" + bytes(60) + b"AB" + bytes(60)
        watched = 0

        def exec_fn(candidate: bytes) -> int:
            # Path depends only on byte 0: everything else is inert.
            return 1 if candidate[watched] == data[watched] else 2

        result = colorize(data, exec_fn, use_type_aware=False)

        assert not _in_taint(result.taints, watched, 1), (
            "the one byte the target branches on was marked path-irrelevant"
        )
        # The second, coincidental "AB" is in dead space and should be taintable.
        assert result.taints, "nothing was found replaceable in an almost-inert input"

    def test_a_fully_inert_input_is_all_taint(self):
        data = bytes(range(64))

        def exec_fn(_candidate: bytes) -> int:
            return 7  # path never moves

        result = colorize(data, exec_fn, use_type_aware=False)
        assert result.taints
        assert (
            _in_taint(result.taints, 0, len(data))
            or sum(r.end - r.start + 1 for r in result.taints) >= len(data) // 2
        )


class TestColorizationOperatorIsSelectable:
    def test_gated_on_cmplog_pairs_not_permanently_off(self):
        fuzzer = types.SimpleNamespace(_cmplog=types.SimpleNamespace(pairs=[(b"IHDR", b"IDAT")]))
        assert "colorization" in REGISTRY.available(fuzzer, b"some input bytes")

    def test_unavailable_without_pairs(self):
        """Without cmplog it degrades to random offset selection, which havoc
        already covers -- so it should not take a selection slot."""
        fuzzer = types.SimpleNamespace(_cmplog=types.SimpleNamespace(pairs=[]))
        assert "colorization" not in REGISTRY.available(fuzzer, b"some input bytes")
