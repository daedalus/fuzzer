"""Binary-search input colorization for Redqueen-style taint tracking.

Port of Redqueen's colorize.py (NDSS 2019).  The core idea is to
determine which bytes of a fuzz input "participate in" (affect the
outcome of) comparison instructions.  The result is a per-byte
classification:

- ``COLORABLE`` (1): this byte influences a comparison — it is valuable
  for input-to-state matching.
- ``FIXED`` (-1): this byte does not influence any comparison.
- ``UNKNOWN`` (0): not yet classified.

The colorizer uses binary search over byte ranges:  for each range it
tests whether zeroing the bytes outside the range (while keeping the
range intact) still triggers the same comparison results.  If so, the
bytes inside the range are *colorable* — they carry comparison-relevant
data.

In the fuzzer context, the ``checker`` callback runs the target with a
modified input and returns True if the same comparison results are
observed.

Usage:
    c = Colorizer(len(input_data), my_checker)
    while c.step():
        pass
    mask = c.color_info  # per-byte classification
"""

import array
import logging

log = logging.getLogger(__name__)

# State constants
COLORABLE = 1
UNKNOWN = 0
FIXED = -1


class Colorizer:
    """Binary-search input colorizer.

    Attributes:
        color_info: ``array('b')`` of per-byte state.
        unknown_ranges: Set of ``(lo, hi)`` ranges still to process.
    """

    def __init__(self, data_length: int, checker):
        """Initialize colorizer.

        Args:
            data_length: Length of the fuzz input in bytes.
            checker: Callable ``(lo, hi) -> bool`` that runs the target
                with bytes outside ``[lo, hi)`` zeroed and returns True
                if comparison-relevant execution is unchanged.
        """
        self.color_info = array.array("b", [UNKNOWN] * data_length)
        self.unknown_ranges: set[tuple[int, int]] = set()
        self.checker = checker
        if data_length > 0:
            self.unknown_ranges.add((0, data_length))

    # ── Public API ─────────────────────────────────────────────────

    def step(self) -> bool:
        """Perform one binary-search iteration.

        Picks the largest remaining unknown range and tests whether it
        is colorable. If so, all bytes in the range are marked COLORABLE.
        Otherwise the range is split and each half is tested in a
        subsequent step.

        Returns:
            True if there are more unknown ranges to process; False when
            all bytes are classified.
        """
        if not self.unknown_ranges:
            return False

        # Pick the largest unknown range.
        lo, hi = max(self.unknown_ranges, key=lambda r: r[1] - r[0])
        self.unknown_ranges.remove((lo, hi))
        self._bin_search(lo, hi)

        return bool(self.unknown_ranges)

    def classify_all(self, max_steps: int | None = None) -> None:
        """Run the colorizer to completion (or up to *max_steps*)."""
        steps = 0
        while self.step():
            steps += 1
            if max_steps is not None and steps >= max_steps:
                log.debug("Colorizer stopped after %d steps (max_steps=%d)", steps, max_steps)
                break

    def colorable_bytes(self) -> list[int]:
        """Return list of byte indices marked COLORABLE."""
        return [i for i, v in enumerate(self.color_info) if v == COLORABLE]

    def fixed_bytes(self) -> list[int]:
        """Return list of byte indices marked FIXED."""
        return [i for i, v in enumerate(self.color_info) if v == FIXED]

    def color_mask(self) -> bytes:
        """Return a mask where COLORABLE bytes are 0xFF, others are 0x00."""
        return bytes(
            0xFF if v == COLORABLE else 0x00 for v in self.color_info
        )

    def fraction_classified(self) -> float:
        """Return the fraction of bytes that are classified (not UNKNOWN)."""
        if not self.color_info:
            return 1.0
        n_classified = sum(1 for v in self.color_info if v != UNKNOWN)
        return n_classified / len(self.color_info)

    # ── Internal ───────────────────────────────────────────────────

    def _is_range_colorable(self, lo: int, hi: int) -> bool:
        """Test whether bytes ``[lo, hi)`` affect any comparison.

        Delegates to ``self.checker(lo, hi)``.  On success, marks all
        bytes in the range as COLORABLE.
        """
        if self.checker(lo, hi):
            for i in range(lo, hi):
                self.color_info[i] = COLORABLE
            return True
        # If this is a single byte that is NOT colorable, mark FIXED.
        if lo + 1 == hi:
            self.color_info[lo] = FIXED
        return False

    def _bin_search(self, lo: int, hi: int) -> None:
        """Binary search within a range: test, split, or mark."""
        if self._is_range_colorable(lo, hi) or lo + 1 == hi:
            return
        mid = lo + (hi - lo) // 2
        self._add_unknown(lo, mid)
        self._add_unknown(mid, hi)

    def _add_unknown(self, lo: int, hi: int) -> None:
        """Register a range as needing classification."""
        if lo < hi:
            self.unknown_ranges.add((lo, hi))


# ── Integration helper ───────────────────────────────────────────────


class CmplogColorizer:
    """Colorize input bytes based on cmplog comparison data.

    Instead of running the target repeatedly (which requires VM-level
    instrumentation), this simpler heuristic marks bytes as COLORABLE
    if they appear in any cmplog comparison operand.
    """

    def __init__(self):
        self.color_info: array.array | None = None

    def colorize_from_cmplog(
        self, input_data: bytes, cmplog_pairs: list[tuple[bytes, bytes]]
    ) -> bytes:
        """Produce a color mask from cmplog operand data.

        Every byte position in *input_data* that falls within a span of
        cmplog-token bytes is marked COLORABLE (0xFF in the mask).

        Args:
            input_data: The fuzz input.
            cmplog_pairs: List of ``(operand_a, operand_b)`` from cmplog.

        Returns:
            Mask bytes of ``len(input_data)`` with 0xFF for colorable
            positions and 0x00 elsewhere.
        """
        n = len(input_data)
        self.color_info = array.array("b", [UNKNOWN] * n)

        for op_a, op_b in cmplog_pairs:
            for token in (op_a, op_b):
                if len(token) < 2:
                    continue
                pos = 0
                while pos <= n - len(token):
                    idx = input_data.find(token, pos)
                    if idx == -1:
                        break
                    for i in range(idx, idx + len(token)):
                        if i < n:
                            self.color_info[i] = COLORABLE
                    pos = idx + 1

        return self.color_mask()

    def color_mask(self) -> bytes:
        """Return a mask where COLORABLE bytes are 0xFF, others are 0x00."""
        if self.color_info is None:
            return b""
        return bytes(
            0xFF if v == COLORABLE else 0x00 for v in self.color_info
        )
