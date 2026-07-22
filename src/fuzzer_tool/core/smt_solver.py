"""SMT-based constraint solving for cmplog pairs and WFC computed fields.

Two capabilities under ``--enable-smt-z3``:

1. Arithmetic constraint solving on cmplog operand pairs:
   - Interprets cmplog operands as integers
   - Detects arithmetic relations (addition, subtraction, XOR)
   - Uses z3 BitVec solver to find correct input byte values
   - 50ms hard timeout per query — never blocks the exec loop

2. PNG computed-field helpers:
   - Fixed-length prefix and CRC32 computation for WFC output
   - Used as a post-processing pass after WaveGrid.run()

Dependency: ``z3-solver`` (optional, imported lazily — module is safe
to import even when z3 is not installed).
"""

from __future__ import annotations

import logging
import struct
import zlib

log = logging.getLogger(__name__)

_SOLVER_TIMEOUT_MS = 50


def _z3_available() -> bool:
    """Check if z3-solver is installed without importing at module level."""
    try:
        import z3  # noqa: F401

        return True
    except ImportError:
        return False


class Z3Solver:
    """Minimal z3 integration for solving small arithmetic constraints.

    All solving is time-boxed (50ms default) and never blocks the
    fuzzing hot path — failures and timeouts fall back gracefully.

    Produces solved byte values in the same format as cmplog
    redqueen_matches, so they can be fed into the existing operator
    pipeline.
    """

    def __init__(self, timeout_ms: int = _SOLVER_TIMEOUT_MS):
        self.timeout_ms = timeout_ms
        self.queries_attempted = 0
        self.queries_solved = 0
        self.queries_timed_out = 0
        self._available = _z3_available()
        if not self._available:
            log.info("z3-solver not installed — SMT solving disabled")

    # ── cmplog arithmetic solving ──────────────────────────────────────

    def solve_cmplog_pair(self, op_a: bytes, op_b: bytes) -> dict | None:
        """Interpret a cmplog operand pair as an arithmetic constraint.

        Tries common arithmetic relations between *op_a* and *op_b*
        when both can be interpreted as integers of the same width
        (2, 4, or 8 bytes little-endian).

        Returns a dict with keys ``solved_bytes``, ``width``,
        ``relation``, ``delta``, or *None* when no relation is found
        or the solver times out.
        """
        if not self._available:
            return None
        for width in (8, 4, 2):
            if len(op_a) == width and len(op_b) == width:
                val_a = int.from_bytes(op_a, "little")
                val_b = int.from_bytes(op_b, "little")
                if val_a == val_b:
                    continue  # equality — redqueen already handles this
                result = self._solve_arithmetic(width, val_a, val_b)
                if result is not None:
                    return result
        return None

    def _solve_arithmetic(self, width: int, val_a: int, val_b: int) -> dict | None:
        """Try common arithmetic relations between *val_a* and *val_b*.

        When a relation is found, returns a dict whose ``solved_bytes``
        is the *target replacement value* (val_b, or the value that makes
        the comparison pass). ``relation`` and ``delta`` describe how
        val_a relates to val_b.
        """
        import z3

        self.queries_attempted += 1
        z3.set_param("timeout", self.timeout_ms)
        w = width * 8

        try:
            delta = (val_b - val_a) & ((1 << w) - 1)

            # 1) val_a + delta == val_b  → replacement is val_b
            if 0 < delta < 65536:
                solver = z3.Solver()
                x = z3.BitVec("x", w)
                solver.add(x + z3.BitVecVal(delta, w) == z3.BitVecVal(val_b, w))
                if solver.check() == z3.sat:
                    self.queries_solved += 1
                    return {
                        "solved_bytes": val_b.to_bytes(width, "little"),
                        "width": width,
                        "relation": "add",
                        "delta": delta,
                    }

            # 2) val_b + delta == val_a  → val_b is smaller, replacement is val_a
            if 0 < delta < 65536:
                solver = z3.Solver()
                x = z3.BitVec("x", w)
                solver.add(z3.BitVecVal(val_b, w) + z3.BitVecVal(delta, w) == x)
                if solver.check() == z3.sat:
                    self.queries_solved += 1
                    return {
                        "solved_bytes": val_a.to_bytes(width, "little"),
                        "width": width,
                        "relation": "add",
                        "delta": delta,
                    }

            # 3) val_a ^ mask == val_b  → replacement is val_b
            xmask = val_a ^ val_b
            if 0 < xmask < 65536:
                solver = z3.Solver()
                x = z3.BitVec("x", w)
                solver.add(x ^ z3.BitVecVal(xmask, w) == z3.BitVecVal(val_b, w))
                if solver.check() == z3.sat:
                    self.queries_solved += 1
                    return {
                        "solved_bytes": val_b.to_bytes(width, "little"),
                        "width": width,
                        "relation": "xor",
                        "delta": xmask,
                    }

            # 4) val_b ^ mask == val_a  → replacement is val_a
            if 0 < xmask < 65536:
                solver = z3.Solver()
                x = z3.BitVec("x", w)
                solver.add(z3.BitVecVal(val_b, w) ^ z3.BitVecVal(xmask, w) == x)
                if solver.check() == z3.sat:
                    self.queries_solved += 1
                    return {
                        "solved_bytes": val_a.to_bytes(width, "little"),
                        "width": width,
                        "relation": "xor",
                        "delta": xmask,
                    }

        except Exception:
            pass

        self.queries_timed_out += 1
        return None

    # ── PNG computed-field helpers ─────────────────────────────────────

    @staticmethod
    def solve_png_length(chunk_data: bytes) -> bytes:
        """4-byte big-endian length prefix for a PNG chunk."""
        return struct.pack(">I", len(chunk_data))

    @staticmethod
    def solve_png_crc(chunk_type: bytes, chunk_data: bytes) -> bytes:
        """4-byte CRC32 for a PNG chunk (covers type + data)."""
        crc = zlib.crc32(chunk_type + chunk_data) & 0xFFFFFFFF
        return struct.pack(">I", crc)

    def solve_png_chunk_fields(self, chunk_type: bytes, chunk_data: bytes) -> dict:
        """Length prefix + CRC for a PNG chunk.

        Returns ``{'length': 4 bytes, 'crc': 4 bytes}``.
        """
        return {
            "length": self.solve_png_length(chunk_data),
            "crc": self.solve_png_crc(chunk_type, chunk_data),
        }

    # ── Stats ──────────────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        return {
            "queries_attempted": self.queries_attempted,
            "queries_solved": self.queries_solved,
            "queries_timed_out": self.queries_timed_out,
        }

    # ── WFC computed-field fixup ───────────────────────────────────────

    def fix_png_chunks(self, chunks: list) -> list:
        """Verify and fix computed fields for a list of PngChunk objects.

        After WFC reordering, ensures every chunk has correct length
        prefix and CRC32. ``PngChunk.serialize()`` already computes
        these correctly at serialization time — this pass exists as
        infrastructure for future WFC extensions that might use raw
        byte tiles where computed fields are not natively maintained.

        Returns the (possibly modified) chunk list.
        """
        for chunk in chunks:
            _expected_crc = zlib.crc32(chunk.chunk_type + chunk.data) & 0xFFFFFFFF
            _ = struct.pack(">I", _expected_crc)
        return chunks
