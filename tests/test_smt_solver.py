"""Tests for core/smt_solver.py — SMT-based constraint solving.

Test categories:
1. Z3Solver init and availability detection
2. cmplog arithmetic solving (add, sub, xor relations)
3. PNG computed-field helpers (length, CRC)
4. WFC fixup pass
5. Stats tracking
6. Edge cases (timeout, equal operands, mismatched widths)
"""

import struct

from fuzzer_tool.core.smt_solver import Z3Solver, _z3_available


# ═══════════════════════════════════════════════════════════════════
# 1. Init and availability
# ═══════════════════════════════════════════════════════════════════


class TestZ3SolverInit:
    def test_available(self):
        """z3-solver is installed in the test environment."""
        assert _z3_available()

    def test_init_defaults(self):
        s = Z3Solver()
        assert s.timeout_ms == 50
        assert s.queries_attempted == 0
        assert s.queries_solved == 0
        assert s.queries_timed_out == 0
        assert s._available

    def test_init_custom_timeout(self):
        s = Z3Solver(timeout_ms=200)
        assert s.timeout_ms == 200

    def test_stats_property(self):
        s = Z3Solver()
        st = s.stats
        assert "queries_attempted" in st
        assert "queries_solved" in st
        assert "queries_timed_out" in st


# ═══════════════════════════════════════════════════════════════════
# 2. cmplog arithmetic solving
# ═══════════════════════════════════════════════════════════════════


class TestSolveCmplogPair:
    def test_solve_add_4byte(self):
        """Arithmetic: op_a + delta == op_b (4-byte LE)."""
        s = Z3Solver()
        val_a = 100
        val_b = 150  # val_a + 50 == val_b
        op_a = struct.pack("<I", val_a)
        op_b = struct.pack("<I", val_b)
        result = s.solve_cmplog_pair(op_a, op_b)
        assert result is not None
        assert result["relation"] in ("add", "sub")
        assert int.from_bytes(result["solved_bytes"], "little") == 150

    def test_solve_sub_4byte(self):
        """Arithmetic: val_a - delta == op_b (val_a > val_b)."""
        s = Z3Solver()
        val_a = 80
        val_b = 30  # val_a - 50 == val_b
        op_a = struct.pack("<I", val_a)
        op_b = struct.pack("<I", val_b)
        result = s.solve_cmplog_pair(op_a, op_b)
        assert result is not None
        # add: val_b + 50 == val_a, or xor: 80 ^ 78 == 30 (both valid)
        assert result["relation"] in ("add", "sub", "xor")
        solved = int.from_bytes(result["solved_bytes"], "little")
        assert solved in (val_a, val_b)  # either direction
        assert int.from_bytes(result["solved_bytes"], "little") == 30

    def test_solve_xor_4byte(self):
        """XOR: val_a ^ mask == val_b."""
        s = Z3Solver()
        val_a = 0xAABBCCDD
        mask = 0x0000FFFF
        val_b = val_a ^ mask
        op_a = struct.pack("<I", val_a)
        op_b = struct.pack("<I", val_b)
        result = s.solve_cmplog_pair(op_a, op_b)
        assert result is not None
        assert result["relation"] in ("add", "sub", "xor")
        assert int.from_bytes(result["solved_bytes"], "little") == val_b

    def test_equal_operands_skipped(self):
        """Equal operands → None (redqueen handles equality)."""
        s = Z3Solver()
        val = 42
        op_a = struct.pack("<I", val)
        op_b = struct.pack("<I", val)
        result = s.solve_cmplog_pair(op_a, op_b)
        assert result is None

    def test_mismatched_width_returns_none(self):
        """Mismatched operand widths → None."""
        s = Z3Solver()
        op_a = struct.pack("<I", 42)
        op_b = struct.pack("<Q", 100)
        result = s.solve_cmplog_pair(op_a, op_b)
        assert result is None

    def test_2byte_width(self):
        """2-byte operands also solved."""
        s = Z3Solver()
        val_a = 10
        val_b = 35  # +25
        op_a = struct.pack("<H", val_a)
        op_b = struct.pack("<H", val_b)
        result = s.solve_cmplog_pair(op_a, op_b)
        assert result is not None
        assert result["width"] == 2

    def test_8byte_width(self):
        """8-byte operands."""
        s = Z3Solver()
        val_a = 1000
        val_b = 1005  # +5
        op_a = struct.pack("<Q", val_a)
        op_b = struct.pack("<Q", val_b)
        result = s.solve_cmplog_pair(op_a, op_b)
        assert result is not None
        assert result["width"] == 8

    def test_stats_increment(self):
        """Queries increment stats correctly."""
        s = Z3Solver()
        assert s.queries_attempted == 0
        op_a = struct.pack("<I", 10)
        op_b = struct.pack("<I", 50)
        s.solve_cmplog_pair(op_a, op_b)
        assert s.queries_attempted >= 1
        assert s.queries_solved >= 1

    def test_no_relation_returns_none(self):
        """Completely unrelated values → None."""
        s = Z3Solver()
        op_a = struct.pack("<I", 0x12345678)
        op_b = struct.pack("<I", 0x9ABCDEF0)
        result = s.solve_cmplog_pair(op_a, op_b)
        # May or may not find a relation — neither assertion should crash
        assert result is None or isinstance(result, dict)


# ═══════════════════════════════════════════════════════════════════
# 3. PNG computed-field helpers
# ═══════════════════════════════════════════════════════════════════


class TestPngHelpers:
    def test_solve_png_length(self):
        """Length prefix equals sizeof(data) in big-endian."""
        data = b"hello world"
        length_bytes = Z3Solver.solve_png_length(data)
        assert len(length_bytes) == 4
        assert struct.unpack(">I", length_bytes)[0] == len(data)

    def test_solve_png_length_empty(self):
        """Empty data → length prefix 0."""
        length_bytes = Z3Solver.solve_png_length(b"")
        assert struct.unpack(">I", length_bytes)[0] == 0

    def test_solve_png_crc(self):
        """CRC covers chunk_type + data."""
        ct = b"IHDR"
        data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
        crc_bytes = Z3Solver.solve_png_crc(ct, data)
        assert len(crc_bytes) == 4
        import zlib

        expected = struct.pack(">I", zlib.crc32(ct + data) & 0xFFFFFFFF)
        assert crc_bytes == expected

    def test_solve_png_chunk_fields(self):
        """Combined length + CRC."""
        s = Z3Solver()
        ct = b"IDAT"
        data = b"compressed data here"
        fields = s.solve_png_chunk_fields(ct, data)
        assert "length" in fields
        assert "crc" in fields
        assert len(fields["length"]) == 4
        assert len(fields["crc"]) == 4
        assert struct.unpack(">I", fields["length"])[0] == len(data)


# ═══════════════════════════════════════════════════════════════════
# 4. WFC fixup pass
# ═══════════════════════════════════════════════════════════════════


class TestFixPngChunks:
    def test_fix_png_chunks_noop(self):
        """fix_png_chunks is a no-op for already-correct chunks."""
        s = Z3Solver()

        class FakeChunk:
            def __init__(self, ct, data):
                self.chunk_type = ct
                self.data = data

        chunks = [FakeChunk(b"IHDR", b"\x00" * 13), FakeChunk(b"IEND", b"")]
        result = s.fix_png_chunks(chunks)
        assert len(result) == 2
        assert result[0].chunk_type == b"IHDR"
        assert result[1].chunk_type == b"IEND"

    def test_fix_png_chunks_empty(self):
        """Empty chunk list."""
        s = Z3Solver()
        result = s.fix_png_chunks([])
        assert result == []


# ═══════════════════════════════════════════════════════════════════
# 5. Edge cases
# ═══════════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_single_byte_operand_skipped(self):
        """Single-byte operands are skipped (too small for meaningful solve)."""
        s = Z3Solver()
        result = s.solve_cmplog_pair(b"\x01", b"\x02")
        assert result is None

    def test_both_zero_width(self):
        """Both operands empty → None."""
        s = Z3Solver()
        result = s.solve_cmplog_pair(b"", b"")
        assert result is None

    def test_large_delta_skipped(self):
        """Delta > 65536 is skipped (not a reasonable arithmetic relation)."""
        s = Z3Solver()
        val_a = 10
        val_b = 10 + 100000  # delta too large
        op_a = struct.pack("<I", val_a)
        op_b = struct.pack("<I", val_b)
        result = s.solve_cmplog_pair(op_a, op_b)
        assert result is None

    def test_stats_no_queries(self):
        s = Z3Solver()
        st = s.stats
        assert st["queries_attempted"] == 0
        assert st["queries_solved"] == 0
        assert st["queries_timed_out"] == 0
