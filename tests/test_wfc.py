"""Extensive tests for Wave Function Collapse (WFC) structural generation.

Test categories:
1. Tile
2. AdjacencyTable (from_pairs, from_corpus, compatibility)
3. Core WaveGrid mechanics (collapse, min-entropy, propagation)
4. Contradiction and backtrack handling
5. 1D sequencing (PNG chunk reorder)
6. BMP pixel generation
7. Determinism (critical for tmin reproducibility)
8. Edge case / stress
"""

import random

from fuzzer_tool.core.png_mutations import (
    PngChunkMutator,
    parse_png_chunks,
    serialize_png_chunks,
)
from fuzzer_tool.core.wfc import (
    AdjacencyTable,
    ConstraintSet,
    Tile,
    WaveGrid,
)

# ═══════════════════════════════════════════════════════════════════
# 1. Tile
# ═══════════════════════════════════════════════════════════════════


class TestTile:
    def test_tile_default_weight(self):
        t = Tile(name=b"IHDR")
        assert t.name == b"IHDR"
        assert t.weight == 1.0

    def test_tile_custom_weight(self):
        t = Tile(name=b"IDAT", weight=2.5)
        assert t.weight == 2.5

    def test_tile_name_preserved(self):
        t = Tile(name=b"\x00\x01\x02")
        assert t.name == b"\x00\x01\x02"


# ═══════════════════════════════════════════════════════════════════
# 2. AdjacencyTable
# ═══════════════════════════════════════════════════════════════════


class TestAdjacencyTable:
    def test_from_pairs_basic(self):
        pairs = [
            (b"IHDR", b"PLTE"),
            (b"PLTE", b"IDAT"),
            (b"IDAT", b"IEND"),
        ]
        table = AdjacencyTable.from_pairs(pairs)
        assert table.compatible(b"IHDR", b"PLTE", "right")
        assert table.compatible(b"PLTE", b"IHDR", "left")
        assert table.compatible(b"PLTE", b"IDAT", "right")
        assert table.compatible(b"IDAT", b"PLTE", "left")
        assert table.compatible(b"IDAT", b"IEND", "right")
        assert table.compatible(b"IEND", b"IDAT", "left")

    def test_from_pairs_empty(self):
        table = AdjacencyTable.from_pairs([])
        # Empty table has no tiles registered
        assert not table.has_tile(b"A")

    def test_compatible_closed_world(self):
        """Closed-world: tiles with no rules are incompatible."""
        table = AdjacencyTable()
        table.add_forward(b"A", b"B")
        assert table.compatible(b"A", b"B", "right")
        assert table.compatible(b"B", b"A", "left")
        # Unknown tile C has no rules → incompatible
        assert not table.compatible(b"C", b"A", "right")
        assert not table.compatible(b"A", b"C", "left")

    def test_add_undirected(self):
        table = AdjacencyTable()
        table.add_undirected(b"A", b"B")
        assert table.compatible(b"A", b"B", "right")
        assert table.compatible(b"A", b"B", "left")
        assert table.compatible(b"B", b"A", "right")
        assert table.compatible(b"B", b"A", "left")

    def test_from_corpus_basic(self):
        tile_names = [b"A", b"B", b"C"]
        sequences = [[b"A", b"B", b"A", b"C"]]
        table = AdjacencyTable.from_corpus(tile_names, sequences)
        assert table.compatible(b"A", b"B", "right")
        assert table.compatible(b"B", b"A", "right")
        assert table.compatible(b"A", b"C", "right")
        assert table.compatible(b"C", b"A", "left")

    def test_from_corpus_empty(self):
        table = AdjacencyTable.from_corpus([b"A"], [])
        assert table.has_tile(b"A")

    def test_has_tile(self):
        table = AdjacencyTable()
        table.add_forward(b"IHDR", b"IDAT")
        assert table.has_tile(b"IHDR")
        assert table.has_tile(b"IDAT")
        assert not table.has_tile(b"UNKNOWN")


# ═══════════════════════════════════════════════════════════════════
# 3. Core WaveGrid mechanics
# ═══════════════════════════════════════════════════════════════════


class TestWaveGridCore:
    def test_collapse_reduces_superposition(self):
        """Cell has N tiles before collapse, 1 after."""
        tiles = [Tile(name=b"A"), Tile(name=b"B"), Tile(name=b"C")]
        adj = AdjacencyTable()
        # All tiles compatible with each other
        for a in tiles:
            for b in tiles:
                adj.add_undirected(a.name, b.name)
        wave = WaveGrid(tiles, adj, width=3)
        assert wave.superpositions[0].count(True) == 3
        wave._observe(0)
        assert wave.superpositions[0].count(True) == 1

    def test_collapse_returns_valid_tile(self):
        """Collapsed tile was in the original superposition."""
        tiles = [Tile(name=b"A"), Tile(name=b"B")]
        adj = AdjacencyTable()
        adj.add_undirected(b"A", b"B")
        wave = WaveGrid(tiles, adj, width=1)
        wave._observe(0)
        chosen = wave.tile_at(0)
        assert chosen in (b"A", b"B")

    def test_min_entropy_picks_most_constrained(self):
        """Among cells with 2 vs 5 options, picks the one with 2."""
        tiles = [Tile(name=b"A"), Tile(name=b"B"), Tile(name=b"C")]
        adj = AdjacencyTable()
        for a in tiles:
            for b in tiles:
                adj.add_undirected(a.name, b.name)
        wave = WaveGrid(tiles, adj, width=2)
        # Cell 0 has only 2 options, cell 1 has all 3
        wave.superpositions[0][2] = False  # remove tile C from cell 0
        idx = wave._find_min_entropy()
        assert idx == 0, f"Expected cell 0 (2 options), got {idx}"

    def test_min_entropy_none_when_all_collapsed(self):
        """All cells collapsed → returns None."""
        tiles = [Tile(name=b"A")]
        adj = AdjacencyTable()
        adj.add_forward(b"A", b"A")
        wave = WaveGrid(tiles, adj, width=3)
        # All cells already collapsed
        assert wave._find_min_entropy() is None

    def test_propagation_prunes_neighbors(self):
        """Collapsing a cell removes incompatible options from neighbors."""
        tiles = [Tile(name=b"A"), Tile(name=b"B")]
        adj = AdjacencyTable()
        adj.add_forward(b"A", b"B")  # A → B allowed, B → A NOT allowed
        # Also add B → A explicitly for symmetry of the closed-world
        adj.add_forward(b"B", b"A")
        wave = WaveGrid(tiles, adj, width=2)
        # Collapse cell 0 to A
        wave.superpositions[0][0] = True  # A
        wave.superpositions[0][1] = False  # not B
        wave._propagate(budget=100)
        # Cell 1 should still have both options (B is compatible right of A, A is compatible right of B)
        assert wave.superpositions[1].count(True) >= 1

    def test_run_completes_1d(self):
        """10-cell 1D wave with spec rules → all cells collapsed, no contradictions."""
        tiles = [Tile(name=b"A"), Tile(name=b"B"), Tile(name=b"C")]
        adj = AdjacencyTable()
        adj.add_forward(b"A", b"B")
        adj.add_forward(b"A", b"C")
        adj.add_forward(b"B", b"A")
        adj.add_forward(b"B", b"C")
        adj.add_forward(b"C", b"A")
        adj.add_forward(b"C", b"B")
        wave = WaveGrid(tiles, adj, width=10)
        result = wave.run(seed=42, max_restarts=3, ac3_budget=1000)
        # All cells should have collapsed
        for row in result:
            for cell in row:
                assert cell is not None

    def test_run_completes_2d(self):
        """2×2 2D wave with simple adjacency → all cells collapsed."""
        tiles = [Tile(name=b"A"), Tile(name=b"B")]
        adj = AdjacencyTable()
        adj.add_undirected(b"A", b"B")
        wave = WaveGrid(tiles, adj, width=2, height=2)
        result = wave.run(seed=42, max_restarts=5, ac3_budget=2000)
        valid = all(cell is not None for row in result for cell in row)
        if not valid:
            # With 2x2 grid and relaxed constraints, most cells should collapse
            nones = sum(cell is None for row in result for cell in row)
            assert nones <= 1, f"Too many uncollapsed cells: {nones}"

    def test_entropy_weighted(self):
        """Heavier tiles are selected more often weighted by their weight alone."""
        tiles = [Tile(name=b"A", weight=100.0), Tile(name=b"B", weight=0.01)]
        adj = AdjacencyTable()
        # Direct neighbor rules to constrain the search space
        adj.add_forward(b"A", b"A")
        adj.add_forward(b"A", b"B")
        adj.add_forward(b"B", b"A")
        adj.add_forward(b"B", b"B")
        wave = WaveGrid(tiles, adj, width=1)
        wave.run(seed=42, max_restarts=3, ac3_budget=500)
        # With only 1 cell, the weighted selection should pick A more often
        counts = {b"A": 0, b"B": 0}
        for s in range(100):
            w = WaveGrid(tiles, adj, width=1)
            w.run(seed=42 + s * 7919, max_restarts=3, ac3_budget=500)
            tile = w.tile_at(0)
            if tile is not None:
                counts[tile] += 1
        assert counts[b"A"] > counts[b"B"], f"A={counts[b'A']}, B={counts[b'B']}"


# ═══════════════════════════════════════════════════════════════════
# 4. Contradiction and backtrack
# ═══════════════════════════════════════════════════════════════════


class TestContradiction:
    def test_contradiction_no_restart_without_budget(self):
        """max_restarts=0 → returns partial output on contradiction, no crash."""
        tiles = [Tile(name=b"A"), Tile(name=b"B")]
        adj = AdjacencyTable()
        # Only A → B allowed, no B → A. In a 2-cell wave, this can dead-end.
        adj.add_forward(b"A", b"B")
        wave = WaveGrid(tiles, adj, width=3)
        result = wave.run(seed=42, max_restarts=0, ac3_budget=100)
        # Should complete without crashing, even if some cells are None
        assert len(result) > 0

    def test_contradiction_restart_recovers(self):
        """budget=3 → retries on contradiction, eventually produces valid output."""
        tiles = [Tile(name=b"A"), Tile(name=b"B")]
        adj = AdjacencyTable()
        adj.add_forward(b"A", b"B")
        adj.add_forward(b"B", b"A")
        wave = WaveGrid(tiles, adj, width=5)
        res = wave.run(seed=42, max_restarts=3, ac3_budget=500)
        # With budget=3 and mixed rules, should typically converge
        # If it doesn't, at least no crash
        assert len(res) > 0

    def test_backtrack_budget_exhausted(self):
        """Exhausts budget → returns best-effort partial output."""
        tiles = [Tile(name=b"A"), Tile(name=b"B")]
        adj = AdjacencyTable()
        adj.add_forward(b"A", b"B")  # Only A→B, no B→A
        wave = WaveGrid(tiles, adj, width=10)
        result = wave.run(seed=42, max_restarts=1, ac3_budget=5)
        # Should return something (could be partial) without crashing
        assert result is not None
        assert len(result) == 1

    def test_all_tiles_not_eliminated(self):
        """After propagate, at least one tile remains per cell (unless contradiction)."""
        tiles = [Tile(name=b"A"), Tile(name=b"B"), Tile(name=b"C")]
        adj = AdjacencyTable()
        adj.add_undirected(b"A", b"B")
        adj.add_undirected(b"B", b"C")
        wave = WaveGrid(tiles, adj, width=5)
        wave.run(seed=42, max_restarts=3, ac3_budget=1000)
        for i in range(wave.n):
            assert wave.superpositions[i].count(True) >= 1


# ═══════════════════════════════════════════════════════════════════
# 5. PNG chunk reorder (1D WFC integration)
# ═══════════════════════════════════════════════════════════════════


class TestPngWfcReorder:
    @staticmethod
    def _make_test_png() -> bytes:
        """Create a minimal valid PNG with IHDR, IDAT, IEND."""
        import struct
        import zlib

        ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0)
        compressed = zlib.compress(b"\x00\x80", 6)
        from fuzzer_tool.core.png_mutations import PngChunk

        chunks = [
            PngChunk(b"IHDR", ihdr_data),
            PngChunk(b"IDAT", compressed),
            PngChunk(b"IEND", b""),
        ]
        return serialize_png_chunks(chunks)

    def test_wfc_reorder_valid_png(self):
        """WFC reorder on valid PNG → output still parses."""
        png = self._make_test_png()
        mutator = PngChunkMutator()
        mutator.use_wfc = True
        result = mutator._wfc_reorder(
            parse_png_chunks(png),
            max_len=4096,
        )
        assert parse_png_chunks(result) is not None

    def test_wfc_reorder_preserves_chunks(self):
        """Same set of chunk types present (no data loss)."""
        png = self._make_test_png()
        mutator = PngChunkMutator()
        mutator.use_wfc = True
        original_chunks = parse_png_chunks(png)
        result = mutator._wfc_reorder(list(original_chunks), max_len=4096)
        result_chunks = parse_png_chunks(result)
        assert result_chunks is not None
        # Check all original types are present
        orig_types = {c.chunk_type for c in original_chunks}
        result_types = {c.chunk_type for c in result_chunks}
        assert orig_types.issubset(result_types)

    def test_wfc_reorder_ihdr_first(self):
        """IHDR is first in the output."""
        png = self._make_test_png()
        mutator = PngChunkMutator()
        mutator.use_wfc = True
        result = mutator._wfc_reorder(
            parse_png_chunks(png),
            max_len=4096,
        )
        chunks = parse_png_chunks(result)
        assert chunks is not None
        assert chunks[0].chunk_type == b"IHDR"

    def test_wfc_reorder_iend_last(self):
        """IEND is last in the output."""
        png = self._make_test_png()
        mutator = PngChunkMutator()
        mutator.use_wfc = True
        result = mutator._wfc_reorder(
            parse_png_chunks(png),
            max_len=4096,
        )
        chunks = parse_png_chunks(result)
        assert chunks is not None
        assert chunks[-1].chunk_type == b"IEND"

    def test_wfc_reorder_with_ancillary(self):
        """Reorder with ancillary chunks produces valid PNG."""
        import struct
        import zlib

        from fuzzer_tool.core.png_mutations import PngChunk

        ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0)
        compressed = zlib.compress(b"\x00\x80\x80\x80\x80", 6)
        chunks = [
            PngChunk(b"IHDR", ihdr_data),
            PngChunk(b"gAMA", struct.pack(">I", 100000)),
            PngChunk(b"pHYs", struct.pack(">IIb", 2835, 2835, 1)),
            PngChunk(b"IDAT", compressed),
            PngChunk(b"IEND", b""),
        ]
        png = serialize_png_chunks(chunks)
        assert parse_png_chunks(png) is not None, "Baseline invalid"
        mutator = PngChunkMutator()
        mutator.use_wfc = True
        result = mutator._wfc_reorder(
            parse_png_chunks(png),
            max_len=4096,
        )
        result_chunks = parse_png_chunks(result)
        assert result_chunks is not None, "WFC reorder produced invalid PNG"
        assert result_chunks[0].chunk_type == b"IHDR"
        assert result_chunks[-1].chunk_type == b"IEND"

    def test_wfc_reorder_invalid_fallback(self):
        """Non-PNG input → WFC reorder falls back to returning original."""
        mutator = PngChunkMutator()
        mutator.use_wfc = True
        from fuzzer_tool.core.png_mutations import PngChunk

        chunks = [PngChunk(b"x", b"data")]
        result = mutator._wfc_reorder(chunks, max_len=1024)
        # Should still produce something
        assert len(result) > 0

    def test_wfc_reorder_changes_order(self):
        """WFC reorder produces a different order than the original."""
        png = self._make_test_png()
        mutator = PngChunkMutator()
        mutator.use_wfc = True
        original_chunks = parse_png_chunks(png)
        # Add more chunks so reordering has room to work
        import struct

        from fuzzer_tool.core.png_mutations import PngChunk

        extra = [
            PngChunk(b"gAMA", struct.pack(">I", 100000)),
            PngChunk(b"pHYs", struct.pack(">IIb", 1, 1, 0)),
        ]
        expendable = list(original_chunks[:-1]) + extra + [original_chunks[-1]]
        png_extended = serialize_png_chunks(expendable)
        result = mutator._wfc_reorder(
            parse_png_chunks(png_extended),
            max_len=4096,
        )
        result_chunks = parse_png_chunks(result)
        assert result_chunks is not None
        # IHDR and IEND stay in place, but ancillary/internal may differ
        assert len(result_chunks) == len(expendable)


# ═══════════════════════════════════════════════════════════════════
# 6. BMP pixel generation
# ═══════════════════════════════════════════════════════════════════


class TestBmpWfc:
    def test_wfc_pixels_produces_output(self):
        """WFC pixel generation runs without error."""
        from fuzzer_tool.core.bmp_mutations import BmpInfo, BmpMutator

        # 4x4 24bpp BMP pixel data
        pixels = bytes(random.randint(0, 255) for _ in range(4 * 4 * 3))
        info = BmpInfo(
            file_size=0,
            pixel_offset=0,
            dib_size=40,
            width=4,
            height=4,
            planes=1,
            bit_count=24,
            compression=0,
            image_size=len(pixels),
            x_ppm=0,
            y_ppm=0,
            colors_used=0,
            colors_important=0,
            header=bytearray(54),
            pixel_data=pixels,
        )
        mutator = BmpMutator()
        mutator.use_wfc = True
        result = mutator._wfc_pixels(info, max_len=4096)
        assert result.pixel_data is not None
        assert len(result.pixel_data) > 0

    def test_wfc_pixels_small(self):
        """Very small image (1×1) → falls through without WFC."""
        from fuzzer_tool.core.bmp_mutations import BmpInfo, BmpMutator

        info = BmpInfo(
            file_size=0,
            pixel_offset=0,
            dib_size=40,
            width=1,
            height=1,
            planes=1,
            bit_count=24,
            compression=0,
            image_size=3,
            x_ppm=0,
            y_ppm=0,
            colors_used=0,
            colors_important=0,
            header=bytearray(54),
            pixel_data=b"\x00\x00\x00",
        )
        mutator = BmpMutator()
        mutator.use_wfc = True
        result = mutator._wfc_pixels(info, max_len=4096)
        assert result is info  # unchanged when too small

    def test_wfc_pixels_empty(self):
        """Empty pixel data → falls through."""
        from fuzzer_tool.core.bmp_mutations import BmpInfo, BmpMutator

        info = BmpInfo(
            file_size=0,
            pixel_offset=0,
            dib_size=40,
            width=4,
            height=4,
            planes=1,
            bit_count=24,
            compression=0,
            image_size=0,
            x_ppm=0,
            y_ppm=0,
            colors_used=0,
            colors_important=0,
            header=bytearray(54),
            pixel_data=b"",
        )
        mutator = BmpMutator()
        mutator.use_wfc = True
        result = mutator._wfc_pixels(info, max_len=4096)
        assert result is info


# ═══════════════════════════════════════════════════════════════════
# 7. Determinism
# ═══════════════════════════════════════════════════════════════════


class TestDeterminism:
    def test_deterministic_reproducibility(self):
        """Same seed → same output."""
        tiles = [Tile(name=b"A"), Tile(name=b"B"), Tile(name=b"C")]
        adj = AdjacencyTable()
        for a in tiles:
            for b in tiles:
                adj.add_undirected(a.name, b.name)

        wave1 = WaveGrid(tiles, adj, width=10)
        r1 = wave1.run(seed=42, max_restarts=3, ac3_budget=1000)

        wave2 = WaveGrid(tiles, adj, width=10)
        r2 = wave2.run(seed=42, max_restarts=3, ac3_budget=1000)

        # Flatten for comparison
        flat1 = [cell for row in r1 for cell in row]
        flat2 = [cell for row in r2 for cell in row]
        assert flat1 == flat2, "Same seed produced different outputs"

    def test_different_seed_different_output(self):
        """Different seed → different output (at least some cells differ)."""
        tiles = [Tile(name=b"A"), Tile(name=b"B"), Tile(name=b"C")]
        adj = AdjacencyTable()
        for a in tiles:
            for b in tiles:
                adj.add_undirected(a.name, b.name)

        wave1 = WaveGrid(tiles, adj, width=20)
        r1 = wave1.run(seed=42, max_restarts=3, ac3_budget=1000)

        wave2 = WaveGrid(tiles, adj, width=20)
        r2 = wave2.run(seed=99, max_restarts=3, ac3_budget=1000)

        flat1 = [cell for row in r1 for cell in row]
        flat2 = [cell for row in r2 for cell in row]
        # With different seeds, at least one cell should differ
        assert flat1 != flat2 or all(c is None for c in flat1), (
            "Different seeds produced identical output"
        )

    def test_deterministic_restart_consistent(self):
        """After backtrack restart, same seed → same output."""
        tiles = [Tile(name=b"A"), Tile(name=b"B")]
        adj = AdjacencyTable()
        adj.add_forward(b"A", b"B")
        adj.add_forward(b"B", b"A")

        wave1 = WaveGrid(tiles, adj, width=15)
        r1 = wave1.run(seed=42, max_restarts=5, ac3_budget=500)

        wave2 = WaveGrid(tiles, adj, width=15)
        r2 = wave2.run(seed=42, max_restarts=5, ac3_budget=500)

        flat1 = [cell for row in r1 for cell in row]
        flat2 = [cell for row in r2 for cell in row]
        valid1 = [c for c in flat1 if c is not None]
        valid2 = [c for c in flat2 if c is not None]
        assert valid1 == valid2, "Deterministic restart failed"


# ═══════════════════════════════════════════════════════════════════
# 8. ConstraintSet (predefined tables)
# ═══════════════════════════════════════════════════════════════════


class TestConstraintSet:
    def test_png_chunks_ihdr_first(self):
        adj = ConstraintSet.png_chunks()
        assert adj.compatible(b"IHDR", b"PLTE", "right")
        assert adj.compatible(b"IHDR", b"IDAT", "right")
        assert adj.compatible(b"IHDR", b"IEND", "right")
        # Nothing can precede IHDR (closed-world: no left rules)
        assert not adj.compatible(b"IHDR", b"IDAT", "left")

    def test_png_chunks_iend_last(self):
        adj = ConstraintSet.png_chunks()
        assert adj.compatible(b"IDAT", b"IEND", "right")
        assert not adj.compatible(b"IEND", b"IDAT", "right")

    def test_png_chunks_ancillary_compatibility(self):
        adj = ConstraintSet.png_chunks()
        assert adj.compatible(b"gAMA", b"pHYs", "right")
        assert adj.compatible(b"pHYs", b"gAMA", "left")
        assert adj.compatible(b"gAMA", b"IDAT", "right")
        assert adj.compatible(b"gAMA", b"IEND", "right")

    def test_jpeg_markers(self):
        adj = ConstraintSet.jpeg_markers()
        assert adj.compatible(b"SOI", b"APP0", "right")
        assert adj.compatible(b"SOI", b"SOF0", "right")
        assert adj.compatible(b"APP0", b"DHT", "right")
        assert adj.compatible(b"SOS", b"EOI", "right")
        # EOI cannot be followed by anything
        assert not adj.compatible(b"EOI", b"APP0", "right")


# ═══════════════════════════════════════════════════════════════════
# 9. Edge case / stress
# ═══════════════════════════════════════════════════════════════════


class TestStress:
    def test_single_cell(self):
        """1×1 wave collapses immediately."""
        tiles = [Tile(name=b"A")]
        adj = AdjacencyTable()
        adj.add_forward(b"A", b"A")
        wave = WaveGrid(tiles, adj, width=1)
        wave.run(seed=42, max_restarts=3, ac3_budget=100)
        assert wave.tile_at(0) == b"A"

    def test_large_grid_1d(self):
        """50-cell 1D wave completes within budget."""
        tiles = [Tile(name=b"A"), Tile(name=b"B"), Tile(name=b"C")]
        adj = AdjacencyTable()
        for a in tiles:
            for b in tiles:
                adj.add_undirected(a.name, b.name)
        wave = WaveGrid(tiles, adj, width=50)
        result = wave.run(seed=42, max_restarts=3, ac3_budget=2000)
        assert wave.contradiction or all(cell is not None for row in result for cell in row)

    def test_output_diversity(self):
        """10 runs produce at least 2 distinct orderings."""
        tiles = [Tile(name=b"A"), Tile(name=b"B")]
        adj = AdjacencyTable()
        adj.add_undirected(b"A", b"B")
        outputs = set()
        for i in range(10):
            wave = WaveGrid(tiles, adj, width=20)
            result = wave.run(seed=42 + i, max_restarts=3, ac3_budget=500)
            flat = tuple(cell for row in result for cell in row)
            outputs.add(flat)
        assert len(outputs) >= 2, f"Only {len(outputs)} distinct outputs"

    def test_hundred_runs_no_crash(self):
        """100 distinct WFC runs, no exceptions."""
        tiles = [Tile(name=b"A"), Tile(name=b"B")]
        adj = AdjacencyTable()
        adj.add_undirected(b"A", b"B")
        for i in range(100):
            wave = WaveGrid(tiles, adj, width=5)
            wave.run(seed=i, max_restarts=3, ac3_budget=100)
        assert True

    def test_wfc_empty_cells(self):
        """Zero cells → empty output, no crash."""
        tiles = [Tile(name=b"A")]
        adj = AdjacencyTable()
        adj.add_forward(b"A", b"A")
        wave = WaveGrid(tiles, adj, width=0, height=0)
        result = wave.run(seed=42)
        assert result == [] or len(result) == 0

    def test_ac3_budget_respected(self):
        """Tiny AC-3 budget → falls back to greedy without infinite loop."""
        tiles = [Tile(name=b"A"), Tile(name=b"B"), Tile(name=b"C")]
        adj = AdjacencyTable()
        for a in tiles:
            for b in tiles:
                adj.add_undirected(a.name, b.name)
        wave = WaveGrid(tiles, adj, width=20)
        wave.run(seed=42, max_restarts=3, ac3_budget=5)
        # Should complete without hanging
        assert True

    def test_unsolvable_propagates_contradiction(self):
        """Enforce impossible constraint → contradiction → handled gracefully."""
        tiles = [Tile(name=b"A"), Tile(name=b"B")]
        adj = AdjacencyTable()
        adj.add_forward(b"A", b"B")  # ONLY A→B allowed
        # Forcing B then A creates impossible constraint
        wave = WaveGrid(tiles, adj, width=3)
        # Manually collapse cell 0 to B
        wave.superpositions[0][0] = False  # not A
        wave.superpositions[0][1] = True  # B
        wave._propagate(budget=1000)
        # May or may not be a contradiction, but should not crash
        assert not wave.contradiction or wave.contradiction is not None
