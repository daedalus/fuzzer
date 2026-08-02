"""Checksum learner: recovers unknown linear checksum polynomials from
observed ``(data, checksum)`` pairs and exposes them to the fuzzer's
mutation operators.

Attaches to the fuzzer instance as ``f.checksum_learner``.  Feeds on
three pair sources:

1. **Format-aware extraction** — deterministic extraction from known
   container formats (PNG, ZIP, GZIP) where the checksum field location
   and coverage are specified by the format.
2. **cmplog pair heuristics** — scans the global ``f._cmplog.pairs`` pool
   for 4-byte operands that appear near data in the input buffer,
   yielding candidate checksum pairs.
3. **Seed-meta / redqueen** — reuses existing ``seed_meta`` entries
   where one operand is a plausible 4-byte checksum.

Recovery uses either Berlekamp-Massey (sequential outputs) or
GCD-of-syndromes (independent pairs), whichever the evidence supports.
The recovered polynomial is cached on ``f.checksum_poly`` and persisted
to ``state.json``.
"""

from __future__ import annotations

import contextlib
import struct
from collections.abc import Callable
from typing import Any

from fuzzer_tool.core.berlekamp_massey import (
    compute_checksum as _compute_checksum,
)
from fuzzer_tool.core.berlekamp_massey import (
    recover_lfsr,
    recover_polynomial_gcd,
)
from fuzzer_tool.core.crc32 import _STANDARD_POLY, crc32, set_active_model


class ChecksumLearner:
    """Learns and caches checksum polynomials from observed pairs."""

    def __init__(self, fuzzer: Any, min_pairs: int = 64, poly_width: int = 32):
        self.f = fuzzer
        self.min_pairs = min_pairs
        self._pairs: list[tuple[bytes, int]] = []
        self._poly: int | None = None  # connection polynomial (model form)
        self._poly_width: int = poly_width
        self._reflect: bool = False  # True when recovered via BM (reflected domain)
        self._format_extractors: list[Callable[[bytes], list[tuple[bytes, int]]]] = [
            self._extract_png_pairs,
            self._extract_zip_pairs,
            self._extract_gzip_pairs,
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def has_enough_pairs(self) -> bool:
        """Return True when enough pairs have been collected to attempt recovery."""
        return len(self._pairs) >= self.min_pairs

    def add_pairs(self, pairs: list[tuple[bytes, int]]) -> None:
        """Add newly-observed (data, checksum) pairs and trigger recovery."""
        if not pairs:
            return
        self._pairs.extend(pairs)
        if self._poly is None and self.has_enough_pairs():
            self._recover()

    def extract_format_pairs(self, data: bytes) -> list[tuple[bytes, int]]:
        """Run all format-aware extractors on *data*."""
        result: list[tuple[bytes, int]] = []
        for extractor in self._format_extractors:
            with contextlib.suppress(Exception):
                result.extend(extractor(data))
        return result

    def ensure_poly(self) -> int | None:
        """Return the recovered polynomial, attempting recovery if needed."""
        if self._poly is None and self.has_enough_pairs():
            self._recover()
        return self._poly

    @property
    def pair_count(self) -> int:
        return len(self._pairs)

    # ------------------------------------------------------------------
    # Format-aware extractors
    # ------------------------------------------------------------------

    def _extract_png_pairs(self, data: bytes) -> list[tuple[bytes, int]]:
        """Extract (chunk_type+data, crc) pairs from a PNG file."""
        if len(data) < 8 or data[:8] != b"\x89PNG\r\n\x1a\n":
            return []
        result: list[tuple[bytes, int]] = []
        pos = 8
        n = len(data)
        while pos + 12 <= n:
            length = struct.unpack_from(">I", data, pos)[0]
            chunk_type = data[pos + 4 : pos + 8]
            chunk_data = data[pos + 8 : pos + 8 + length]
            crc_bytes = data[pos + 8 + length : pos + 12 + length]
            if len(crc_bytes) != 4:
                break
            crc = struct.unpack(">I", crc_bytes)[0]
            result.append((bytes(chunk_type) + bytes(chunk_data), crc))
            pos += 12 + length
            if chunk_type == b"IEND":
                break
        return result

    def _extract_zip_pairs(self, data: bytes) -> list[tuple[bytes, int]]:
        """Extract (filename+data, crc32) pairs from a ZIP local file header."""
        result: list[tuple[bytes, int]] = []
        pos = 0
        n = len(data)
        while pos + 30 <= n:
            sig = struct.unpack_from("<I", data, pos)[0]
            if sig != 0x04034B50:  # PK\\x03\\x04
                break
            crc = struct.unpack_from("<I", data, pos + 14)[0]
            fname_len = struct.unpack_from("<H", data, pos + 26)[0]
            extra_len = struct.unpack_from("<H", data, pos + 28)[0]
            fname = data[pos + 30 : pos + 30 + fname_len]
            # Use filename as the "data" for pair extraction
            result.append((bytes(fname), crc))
            comp_size = struct.unpack_from("<I", data, pos + 18)[0]
            pos += 30 + fname_len + extra_len + comp_size
        return result

    def _extract_gzip_pairs(self, data: bytes) -> list[tuple[bytes, int]]:
        """Extract (payload, original_crc) pairs from a GZIP member.

        Known limitation: the compressed payload is not decompressed, so
        pairs carry an empty payload placeholder — a gzip member's real
        (payload, crc) relationship requires inflate, which is out of
        scope for format extraction.  The empty-payload pairs never pass
        ``_verify`` (they reproduce no observed checksum), so they cannot
        corrupt recovery.
        """
        if len(data) < 18 or data[:2] != b"\x1f\x8b":
            return []
        # Trailer: CRC32(4) + ISIZE(4) at the end of the member.
        crc = struct.unpack_from("<I", data, len(data) - 8)[0]
        return [(b"", crc)]

    # ------------------------------------------------------------------
    # cmplog pair extraction (heuristic)
    # ------------------------------------------------------------------

    def extract_cmplog_pairs(self, input_data: bytes) -> list[tuple[bytes, int]]:
        """Heuristic extraction of (data, checksum) pairs from cmplog.

        Scans ``f._cmplog.pairs`` for 4-byte operands that appear after
        data of a plausible length in the input buffer.
        """
        pairs: list[tuple[bytes, int]] = []
        cmplog = getattr(self.f, "_cmplog", None)
        if not cmplog or not cmplog.pairs:
            return pairs

        for op_a, op_b in cmplog.pairs:
            len_a = len(op_a)
            len_b = len(op_b)
            if len_a == 4 and len_b == 4:
                # The checksum operand appears AFTER the data operand in
                # the input buffer; the earlier one is the "data".
                pos_a = input_data.find(op_a)
                pos_b = input_data.find(op_b)
                if pos_a >= 0 and pos_b >= 0 and pos_a != pos_b:
                    data_op = op_a if pos_a < pos_b else op_b
                    checksum_op = op_b if pos_a < pos_b else op_a
                    pairs.append((data_op, int.from_bytes(checksum_op, "big")))
        return pairs

    # ------------------------------------------------------------------
    # Recovery
    # ------------------------------------------------------------------

    def _recover(self) -> None:
        """Attempt polynomial recovery from collected pairs.

        A candidate polynomial is only accepted when it reproduces the
        observed checksums for at least two distinct pairs — this rejects
        the GCD residue that independent pairs with a mismatched
        ``(init, final_xor)`` configuration produce (e.g. standard PNG
        CRCs, which use ``final_xor=0xFFFFFFFF``), preventing a garbage
        polynomial from ever being activated.
        """
        if not self._pairs:
            return

        # Deduplicate pairs
        unique = list({(d, c) for d, c in self._pairs})

        # GCD-of-syndromes first: works for independent (data, checksum)
        # pairs (different files/chunks).  Recovers in the non-reflected
        # domain (normal-form polynomial).
        poly = recover_polynomial_gcd(unique, width=self._poly_width)
        if poly and poly != 0 and self._verify(poly, unique, reflect=False):
            self._reflect = False
            self._set_model(poly)
            return

        # BM fallback: recovers sequential LFSR output streams (reflected
        # domain, reversed-form polynomial).  Requires the checksum values
        # to be one-step-apart register states, which realistic corpora
        # rarely provide — hence the strict verification below.
        checksums = sorted({c for _, c in unique})
        poly = recover_lfsr(checksums, width=self._poly_width)
        if poly and poly != 0 and self._verify(poly, unique, reflect=True):
            self._reflect = True
            self._set_model(poly)

    def _verify(self, poly: int, pairs: list[tuple[bytes, int]], reflect: bool) -> bool:
        """True when *poly* reproduces the checksum of >= 2 distinct pairs."""
        matches = 0
        for data, crc in pairs:
            if reflect:
                got = _compute_checksum(
                    data,
                    poly=poly,
                    width=self._poly_width,
                    init=0,
                    final_xor=0,
                    reflect_in=True,
                    reflect_out=False,
                )
            else:
                got = _compute_checksum(data, poly=poly, width=self._poly_width)
            if got == crc:
                matches += 1
                if matches >= 2:
                    return True
        return False

    def _set_model(self, poly: int) -> None:
        """Cache the recovered polynomial and activate it in the crc32 module."""
        self._poly = poly
        if poly == _STANDARD_POLY:
            return  # the crc32 module already handles the standard poly
        set_active_model(
            poly,
            width=self._poly_width,
            init=0,
            final_xor=0,
            reflect_in=self._reflect,
            reflect_out=False,
        )

    def compute_checksum(self, data: bytes) -> int:
        """Compute a checksum for *data* using the recovered model.

        Falls back to the standard ``crc32`` wrapper when no polynomial
        has been recovered yet.
        """
        poly = self.ensure_poly()
        if poly is None:
            return crc32(data)
        return _compute_checksum(
            data,
            poly=poly,
            width=self._poly_width,
            init=0,
            final_xor=0,
            reflect_in=self._reflect,
            reflect_out=False,
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "poly": self._poly,
            "poly_width": self._poly_width,
            "reflect": self._reflect,
            "pair_count": self.pair_count,
        }

    @classmethod
    def from_dict(cls, fuzzer: Any, data: dict[str, Any] | None) -> ChecksumLearner:
        learner = cls(fuzzer)
        if data:
            learner._poly = data.get("poly")
            learner._poly_width = data.get("poly_width", 32)
            learner._reflect = data.get("reflect", False)
            if learner._poly and learner._poly != _STANDARD_POLY:
                # Re-activate the recovered model in the crc32 module.
                set_active_model(
                    learner._poly,
                    width=learner._poly_width,
                    init=0,
                    final_xor=0,
                    reflect_in=learner._reflect,
                    reflect_out=False,
                )
        return learner
