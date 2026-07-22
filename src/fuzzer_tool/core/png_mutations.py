"""PNG-aware mutations for structure-aware fuzzing of PNG files.

Provides PNG chunk parsing/serialization and a comprehensive set of
mutation operators that maintain (or intentionally break) PNG structural
validity. These operators target specific PNG code paths:

  - IHDR mutations (dimensions, color type, bit depth, interlace)
  - IDAT mutations (compression level, filter type, data corruption)
  - PLTE mutations (palette manipulation)
  - Chunk operations (add, delete, duplicate, reorder, split)
  - CRC operations (fix, corrupt, repair)
  - Ancillary chunks (tRNS, gAMA, pHYs, cHRM, sBIT, iCCP, tEXt)
  - Multi-IDAT splitting for decompression path coverage

Usage:
    from fuzzer_tool.core.png_mutations import PngChunkMutator, parse_png_chunks

    mutator = PngChunkMutator()
    mutated = mutator.mutate(original_png, max_len=4096)
"""

import random
import struct
import zlib


class PngChunk:
    """A single PNG chunk: type(4) + data(length) + crc(4)."""

    __slots__ = ("chunk_type", "data")

    def __init__(self, chunk_type: bytes, data: bytes):
        self.chunk_type = chunk_type
        self.data = data

    def serialize(self) -> bytes:
        length = struct.pack(">I", len(self.data))
        crc = struct.pack(">I", self._compute_crc())
        return length + self.chunk_type + self.data + crc

    def _compute_crc(self) -> int:
        return zlib.crc32(self.chunk_type + self.data) & 0xFFFFFFFF


def parse_png_chunks(data: bytes) -> list[PngChunk] | None:
    """Parse PNG data into a list of chunks. Returns None if invalid."""
    if len(data) < 8 or data[:8] != b"\x89PNG\r\n\x1a\n":
        return None

    chunks = []
    pos = 8
    while pos + 8 <= len(data):
        if pos + 8 > len(data):
            break
        length = struct.unpack_from(">I", data, pos)[0]
        chunk_type = data[pos + 4 : pos + 8]

        if pos + 12 + length > len(data):
            break  # truncated

        chunk_data = data[pos + 8 : pos + 8 + length]
        chunks.append(PngChunk(chunk_type, chunk_data))
        pos += 12 + length  # 4(len) + 4(type) + data + 4(crc)

        if chunk_type == b"IEND":
            break

    return chunks if chunks else None


def serialize_png_chunks(chunks: list[PngChunk]) -> bytes:
    """Serialize chunks back to PNG bytes."""
    result = b"\x89PNG\r\n\x1a\n"
    for chunk in chunks:
        result += chunk.serialize()
    return result


class PngChunkMutator:
    """Structure-aware PNG fuzzer with chunk-level mutation operators.

    When ``use_wfc=True``, the reorder operator uses Wave Function Collapse
    to generate novel-but-valid chunk orderings instead of random swap.

    Mutations target specific PNG code paths while maintaining (or
    intentionally breaking) structural validity:

    1. IHDR corruption — test dimension/color validation
    2. IDAT mutation — test decompression paths
    3. PLTE manipulation — test palette validation
    4. Chunk add/delete/reorder — test chunk ordering rules
    5. CRC corruption — break CRC validation paths
    6. Length field mutation — test length validation
    7. Multi-IDAT split — test multi-chunk decompression
    8. Filter type mutation — test filter processing
    9. Interlace mutation — test interlace deinterlacing
    10. Ancillary chunk injection — test optional chunk handling
    11. tRNS/gAMA/pHYs/cHRM/sBIT/iCCP/tEXt injection
    12. Micro IDAT — test minimal decompression
    13. Duplicate IHDR — test header validation
    14. Move-after-IEND — test trailer validation
    15. Empty chunk injection — test zero-length handling
    """

    use_wfc: bool = False  # set to True by Fuzzer when --wfc is active
    smt_solver: object | None = None  # Z3Solver instance (set by Fuzzer when --enable-smt-z3)

    def mutate(self, data: bytes, max_len: int = 4096) -> bytes:
        """Apply a random PNG-aware mutation."""
        chunks = parse_png_chunks(data)
        if not chunks:
            # Not valid PNG — generate a minimal one
            return self._generate_random_png(max_len)

        op = random.randint(0, 23)
        if op == 0:
            return self._mutate_ihdr(chunks, max_len)
        elif op == 1:
            return self._mutate_idat(chunks, max_len)
        elif op == 2:
            return self._duplicate_chunk(chunks, max_len)
        elif op == 3:
            return self._delete_chunk(chunks, max_len)
        elif op == 4:
            if self.use_wfc:
                return self._wfc_reorder(chunks, max_len)
            return self._reorder_chunks(chunks, max_len)
        elif op == 5:
            return self._corrupt_crc(chunks, max_len)
        elif op == 6:
            return self._mutate_length(chunks, max_len)
        elif op == 7:
            return self._split_idat(chunks, max_len)
        elif op == 8:
            return self._mutate_filter(chunks, max_len)
        elif op == 9:
            return self._mutate_interlace(chunks, max_len)
        elif op == 10:
            return self._add_empty_chunks(chunks, max_len)
        elif op == 11:
            return self._mutate_idat_multi(chunks, max_len)
        elif op == 12:
            return self._generate_random_png(max_len)
        elif op == 13:
            return self._duplicate_ihdr(chunks, max_len)
        elif op == 14:
            return self._move_after_iend(chunks, max_len)
        elif op == 15:
            return self._mutate_plte(chunks, max_len)
        elif op == 16:
            return self._mutate_chrm(chunks, max_len)
        elif op == 17:
            return self._mutate_sbit(chunks, max_len)
        elif op == 18:
            return self._mutate_iccp(chunks, max_len)
        elif op == 19:
            return self._mutate_trns(chunks, max_len)
        elif op == 20:
            return self._mutate_ancillary(chunks, max_len)
        elif op == 21:
            return self._micro_idat(chunks, max_len)
        elif op == 22:
            return self._corrupt_signature(max_len)
        else:
            return self._swap_idat_chunks(chunks, max_len)

    def _mutate_ihdr(self, chunks: list[PngChunk], max_len: int) -> bytes:
        """Corrupt IHDR — test dimension/color validation."""
        ihdr = self._find_chunk(chunks, b"IHDR")
        if not ihdr or len(ihdr.data) < 13:
            return serialize_png_chunks(chunks)[:max_len]

        data = bytearray(ihdr.data)
        field = random.randint(0, 4)
        if field == 0:  # width
            struct.pack_into(">I", data, 0, random.randint(0, 0xFFFFFFFF))
        elif field == 1:  # height
            struct.pack_into(">I", data, 4, random.randint(0, 0xFFFFFFFF))
        elif field == 2:  # bit depth
            data[8] = random.choice([0, 1, 2, 4, 8, 16, 32, 64, 128, 255])
        elif field == 3:  # color type
            data[9] = random.randint(0, 255)
        elif field == 4:  # interlace
            data[12] = random.choice([0, 1, 42, 255])
        ihdr.data = bytes(data)
        return serialize_png_chunks(chunks)[:max_len]

    def _mutate_plte(self, chunks: list[PngChunk], max_len: int) -> bytes:
        """Manipulate palette — test palette validation."""
        plte = self._find_chunk(chunks, b"PLTE")
        if plte:
            # Corrupt existing palette
            if plte.data:
                data = bytearray(plte.data)
                idx = random.randint(0, len(data) - 1)
                data[idx] = random.randint(0, 255)
                plte.data = bytes(data)
        else:
            # Add a palette where none exists
            idat_idx = self._find_chunk_index(chunks, b"IDAT")
            if idat_idx >= 0:
                plte_data = bytes(random.randint(0, 255) for _ in range(768))
                chunks.insert(idat_idx, PngChunk(b"PLTE", plte_data))
        return serialize_png_chunks(chunks)[:max_len]

    def _mutate_idat(self, chunks: list[PngChunk], max_len: int) -> bytes:
        """Mutate IDAT data — test decompression paths."""
        idat = self._find_chunk(chunks, b"IDAT")
        if not idat or not idat.data:
            return serialize_png_chunks(chunks)[:max_len]

        data = bytearray(idat.data)
        # Flip bytes in the compressed stream
        for _ in range(random.randint(1, min(8, len(data)))):
            idx = random.randint(0, len(data) - 1)
            data[idx] ^= 1 << random.randint(0, 7)
        idat.data = bytes(data)
        return serialize_png_chunks(chunks)[:max_len]

    def _duplicate_chunk(self, chunks: list[PngChunk], max_len: int) -> bytes:
        """Duplicate a random chunk."""
        if len(chunks) < 2:
            return serialize_png_chunks(chunks)[:max_len]
        idx = random.randint(0, len(chunks) - 1)
        clone = PngChunk(chunks[idx].chunk_type, chunks[idx].data)
        insert_at = random.randint(0, len(chunks))
        chunks.insert(insert_at, clone)
        return serialize_png_chunks(chunks)[:max_len]

    def _delete_chunk(self, chunks: list[PngChunk], max_len: int) -> bytes:
        """Delete a random chunk (except IHDR and IEND)."""
        if len(chunks) < 3:
            return serialize_png_chunks(chunks)[:max_len]
        deletable = [i for i, c in enumerate(chunks) if c.chunk_type not in (b"IHDR", b"IEND")]
        if deletable:
            del chunks[random.choice(deletable)]
        return serialize_png_chunks(chunks)[:max_len]

    def _reorder_chunks(self, chunks: list[PngChunk], max_len: int) -> bytes:
        """Swap two random chunks (tests ordering rules)."""
        if len(chunks) < 3:
            return serialize_png_chunks(chunks)[:max_len]
        i = random.randint(0, len(chunks) - 2)
        j = random.randint(i + 1, len(chunks) - 1)
        chunks[i], chunks[j] = chunks[j], chunks[i]
        return serialize_png_chunks(chunks)[:max_len]

    def _wfc_reorder(self, chunks: list[PngChunk], max_len: int) -> bytes:
        """Reorder chunks using 1D Wave Function Collapse.

        Extracts the chunk-type sequence, runs WFC to generate a new
        valid ordering, preserves original chunk data in the new order.
        Falls back to random shuffle if WFC fails or no valid ordering
        found. Always preserves IHDR-first and IEND-last.
        """
        if len(chunks) < 3:
            return serialize_png_chunks(chunks)[:max_len]

        from fuzzer_tool.core.wfc import ConstraintSet, Tile, WaveGrid

        # Extract unique chunk types present, preserving critical anchors
        type_names = list(dict.fromkeys(c.chunk_type for c in chunks))
        has_ihdr = b"IHDR" in type_names
        has_iend = b"IEND" in type_names

        # Build adjacency and tiles
        adjacency = ConstraintSet.png_chunks()
        tiles = [Tile(name=t) for t in type_names]

        # Create 1D wave with same number of cells as we have chunks
        wave = WaveGrid(tiles, adjacency, width=len(chunks), height=1)

        # Seed the first cell to IHDR and last cell to IEND (if present)
        if has_ihdr:
            ihdr_tid = next(i for i, t in enumerate(tiles) if t.name == b"IHDR")
            for j in range(len(tiles)):
                wave.superpositions[0][j] = j == ihdr_tid
        if has_iend:
            iend_tid = next(i for i, t in enumerate(tiles) if t.name == b"IEND")
            for j in range(len(tiles)):
                wave.superpositions[-1][j] = j == iend_tid

        result = wave.run(seed=random.randint(0, 2**31), max_restarts=3, ac3_budget=2000)

        if result is None or result[0] is None:
            random.shuffle(chunks)
            # Ensure IHDR-first and IEND-last invariant
            self._ensure_invariants(chunks)
            return serialize_png_chunks(chunks)[:max_len]

        new_order = result[0]

        # Build chunk lookup by type (preserving IDAT ordering)
        by_type: dict[bytes, list[PngChunk]] = {}
        for c in chunks:
            by_type.setdefault(c.chunk_type, []).append(c)

        reordered: list[PngChunk] = []
        for tile_name in new_order:
            if tile_name is None:
                continue
            pool = by_type.get(tile_name, [])
            if pool:
                reordered.append(pool.pop(0))

        # If WFC produced a shorter sequence (shouldn't happen), append remaining chunks
        placed = {c for c in chunks if c not in sum(by_type.values(), [])}
        remaining = [c for c in chunks if c not in placed]
        reordered.extend(remaining)

        if not reordered:
            return serialize_png_chunks(chunks)[:max_len]

        # Final invariant enforcement
        self._ensure_invariants(reordered)

        # SMT fixup: ensure computed fields (length, CRC) are correct
        if self.smt_solver is not None and hasattr(self.smt_solver, "fix_png_chunks"):
            self.smt_solver.fix_png_chunks(reordered)

        return serialize_png_chunks(reordered)[:max_len]

    @staticmethod
    def _ensure_invariants(chunks: list[PngChunk]):
        """Move IHDR to first position and IEND to last position."""
        # Find and move IHDR to front
        for i, c in enumerate(chunks):
            if c.chunk_type == b"IHDR":
                if i != 0:
                    chunks.insert(0, chunks.pop(i))
                break
        # Find and move IEND to end
        for i in range(len(chunks) - 1, -1, -1):
            if chunks[i].chunk_type == b"IEND":
                if i != len(chunks) - 1:
                    chunks.append(chunks.pop(i))
                break

    def _corrupt_crc(self, chunks: list[PngChunk], max_len: int) -> bytes:
        """Corrupt a chunk's CRC (tests CRC validation paths)."""
        if len(chunks) < 2:
            return serialize_png_chunks(chunks)[:max_len]

        result = b"\x89PNG\r\n\x1a\n"
        target = random.randint(0, len(chunks) - 1)
        for i, chunk in enumerate(chunks):
            length = struct.pack(">I", len(chunk.data))
            if i == target:
                crc = struct.pack(">I", random.randint(0, 0xFFFFFFFF))
            else:
                crc = struct.pack(">I", chunk._compute_crc())
            result += length + chunk.chunk_type + chunk.data + crc
        return result[:max_len]

    def _mutate_length(self, chunks: list[PngChunk], max_len: int) -> bytes:
        """Corrupt a chunk's length field."""
        if len(chunks) < 2:
            return serialize_png_chunks(chunks)[:max_len]

        result = b"\x89PNG\r\n\x1a\n"
        target = random.randint(0, len(chunks) - 1)
        for i, chunk in enumerate(chunks):
            if i == target:
                length = struct.pack(">I", random.randint(0, 0xFFFFFFFF))
            else:
                length = struct.pack(">I", len(chunk.data))
            crc = struct.pack(">I", chunk._compute_crc())
            result += length + chunk.chunk_type + chunk.data + crc
        return result[:max_len]

    def _split_idat(self, chunks: list[PngChunk], max_len: int) -> bytes:
        """Split one IDAT into multiple (tests multi-chunk decompression)."""
        idat = self._find_chunk(chunks, b"IDAT")
        if not idat or len(idat.data) < 4:
            return serialize_png_chunks(chunks)[:max_len]

        idx = self._find_chunk_index(chunks, b"IDAT")
        if idx < 0:
            return serialize_png_chunks(chunks)[:max_len]

        data = idat.data
        pieces = []
        num_pieces = random.randint(2, min(4, len(data)))
        chunk_size = len(data) // num_pieces
        for i in range(num_pieces):
            start = i * chunk_size
            end = start + chunk_size if i < num_pieces - 1 else len(data)
            pieces.append(PngChunk(b"IDAT", data[start:end]))

        chunks[idx : idx + 1] = pieces
        return serialize_png_chunks(chunks)[:max_len]

    def _mutate_filter(self, chunks: list[PngChunk], max_len: int) -> bytes:
        """Mutate the filter byte at the start of IDAT data."""
        idat = self._find_chunk(chunks, b"IDAT")
        if not idat or not idat.data:
            return serialize_png_chunks(chunks)[:max_len]

        ihdr = self._find_chunk(chunks, b"IHDR")
        if not ihdr or len(ihdr.data) < 13:
            return serialize_png_chunks(chunks)[:max_len]

        width = struct.unpack_from(">I", ihdr.data, 0)[0]
        if width == 0:
            return serialize_png_chunks(chunks)[:max_len]

        data = bytearray(idat.data)
        if data:
            # Flip filter type at start of a row
            row_idx = random.randint(0, min(width - 1, len(data) - 1))
            if row_idx < len(data):
                data[row_idx] = random.randint(0, 4)
        idat.data = bytes(data)
        return serialize_png_chunks(chunks)[:max_len]

    def _mutate_interlace(self, chunks: list[PngChunk], max_len: int) -> bytes:
        """Mutate the interlace field in IHDR."""
        ihdr = self._find_chunk(chunks, b"IHDR")
        if not ihdr or len(ihdr.data) < 13:
            return serialize_png_chunks(chunks)[:max_len]

        data = bytearray(ihdr.data)
        data[12] = random.choice([0, 1, 42, 255])
        ihdr.data = bytes(data)
        return serialize_png_chunks(chunks)[:max_len]

    def _add_empty_chunks(self, chunks: list[PngChunk], max_len: int) -> bytes:
        """Inject empty (zero-length) chunks."""
        chunk_types = [b"tEXt", b"zTXt", b"iTXt", b"gAMA", b"pHYs"]
        ct = random.choice(chunk_types)
        chunks.insert(random.randint(1, len(chunks)), PngChunk(ct, b""))
        return serialize_png_chunks(chunks)[:max_len]

    def _mutate_idat_multi(self, chunks: list[PngChunk], max_len: int) -> bytes:
        """Replace IDAT with multiple small compressed chunks."""
        idat = self._find_chunk(chunks, b"IDAT")
        if not idat:
            return serialize_png_chunks(chunks)[:max_len]

        idx = self._find_chunk_index(chunks, b"IDAT")
        if idx < 0:
            return serialize_png_chunks(chunks)[:max_len]

        # Generate multiple small compressed blocks
        new_chunks = []
        for _ in range(random.randint(2, 5)):
            block = bytes(random.randint(0, 255) for _ in range(random.randint(8, 64)))
            compressed = zlib.compress(block, 6)
            new_chunks.append(PngChunk(b"IDAT", compressed))

        chunks[idx : idx + 1] = new_chunks
        return serialize_png_chunks(chunks)[:max_len]

    def _generate_random_png(self, max_len: int) -> bytes:
        """Generate a random valid PNG."""
        w = random.randint(1, 64)
        h = random.randint(1, 64)
        ct = random.choice([0, 2, 3, 4, 6])
        bd = random.choice([1, 2, 4, 8, 16])
        channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(ct, 1)
        # Build raw scanlines: one zero byte (filter=None) + pixel data per row
        row_len = w * channels
        raw = bytearray(h * (1 + row_len))
        raw[0 :: 1 + row_len] = b"\x00" * h  # filter bytes
        for i in range(h):
            start = i * (1 + row_len) + 1
            raw[start : start + row_len] = random.randbytes(row_len)

        ihdr_data = struct.pack(">IIBBBBB", w, h, bd, ct, 0, 0, 0)
        ihdr = PngChunk(b"IHDR", ihdr_data)
        compressed = zlib.compress(bytes(raw), 6)
        idat = PngChunk(b"IDAT", compressed)
        iend = PngChunk(b"IEND", b"")
        return serialize_png_chunks([ihdr, idat, iend])[:max_len]

    def _duplicate_ihdr(self, chunks: list[PngChunk], max_len: int) -> bytes:
        """Duplicate IHDR (test header validation)."""
        ihdr = self._find_chunk(chunks, b"IHDR")
        if not ihdr:
            return serialize_png_chunks(chunks)[:max_len]
        clone = PngChunk(b"IHDR", ihdr.data)
        chunks.insert(1, clone)
        return serialize_png_chunks(chunks)[:max_len]

    def _move_after_iend(self, chunks: list[PngChunk], max_len: int) -> bytes:
        """Move a chunk after IEND (test trailer validation)."""
        if len(chunks) < 3:
            return serialize_png_chunks(chunks)[:max_len]
        iend_idx = self._find_chunk_index(chunks, b"IEND")
        if iend_idx < 0:
            return serialize_png_chunks(chunks)[:max_len]
        # Pick a chunk before IEND
        movable = [i for i in range(iend_idx) if chunks[i].chunk_type != b"IHDR"]
        if not movable:
            return serialize_png_chunks(chunks)[:max_len]
        src = random.choice(movable)
        chunk = chunks.pop(src)
        # Insert after IEND (which may have shifted)
        new_iend = self._find_chunk_index(chunks, b"IEND")
        chunks.insert(new_iend + 1, chunk)
        return serialize_png_chunks(chunks)[:max_len]

    def _mutate_chrm(self, chunks: list[PngChunk], max_len: int) -> bytes:
        """Inject/mutate cHRM chunk (chromaticity)."""
        chrm = self._find_chunk(chunks, b"cHRM")
        if chrm:
            data = bytearray(chrm.data)
            if len(data) >= 32:
                idx = random.randint(0, 7) * 4
                struct.pack_into(">I", data, idx, random.randint(0, 0xFFFFFFFF))
                chrm.data = bytes(data)
        else:
            chrm_data = bytes(random.randint(0, 255) for _ in range(32))
            chunks.insert(random.randint(1, len(chunks)), PngChunk(b"cHRM", chrm_data))
        return serialize_png_chunks(chunks)[:max_len]

    def _mutate_sbit(self, chunks: list[PngChunk], max_len: int) -> bytes:
        """Inject/mutate sBIT chunk (significant bits)."""
        sbit = self._find_chunk(chunks, b"sBIT")
        if sbit:
            data = bytearray(sbit.data)
            if data:
                idx = random.randint(0, len(data) - 1)
                data[idx] = random.randint(0, 255)
                sbit.data = bytes(data)
        else:
            ihdr = self._find_chunk(chunks, b"IHDR")
            if ihdr and len(ihdr.data) >= 10:
                ct = ihdr.data[9]
                channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(ct, 1)
                sbit_data = bytes(random.randint(0, 16) for _ in range(channels))
                chunks.insert(random.randint(1, len(chunks)), PngChunk(b"sBIT", sbit_data))
        return serialize_png_chunks(chunks)[:max_len]

    def _mutate_iccp(self, chunks: list[PngChunk], max_len: int) -> bytes:
        """Inject/mutate iCCP chunk (ICC profile)."""
        iccp = self._find_chunk(chunks, b"iCCP")
        if iccp:
            data = bytearray(iccp.data)
            if data:
                idx = random.randint(0, len(data) - 1)
                data[idx] = random.randint(0, 255)
                iccp.data = bytes(data)
        else:
            # Minimal ICC profile: profile name + null + compression method + compressed data
            name = b"test\x00"
            compressed = zlib.compress(bytes(random.randint(0, 255) for _ in range(64)), 6)
            iccp_data = name + b"\x00" + compressed
            chunks.insert(random.randint(1, len(chunks)), PngChunk(b"iCCP", iccp_data))
        return serialize_png_chunks(chunks)[:max_len]

    def _mutate_trns(self, chunks: list[PngChunk], max_len: int) -> bytes:
        """Inject/mutate tRNS chunk (transparency)."""
        trns = self._find_chunk(chunks, b"tRNS")
        if trns:
            data = bytearray(trns.data)
            if data:
                idx = random.randint(0, len(data) - 1)
                data[idx] = random.randint(0, 255)
                trns.data = bytes(data)
        else:
            ihdr = self._find_chunk(chunks, b"IHDR")
            if ihdr and len(ihdr.data) >= 10:
                ct = ihdr.data[9]
                if ct in (0, 2, 3):  # gray, rgb, palette
                    if ct == 3:
                        trns_data = bytes(
                            random.randint(0, 255) for _ in range(random.randint(1, 256))
                        )
                    elif ct == 0:
                        trns_data = struct.pack(">H", random.randint(0, 65535))
                    else:
                        trns_data = bytes(random.randint(0, 255) for _ in range(6))
                    chunks.insert(random.randint(1, len(chunks)), PngChunk(b"tRNS", trns_data))
        return serialize_png_chunks(chunks)[:max_len]

    def _mutate_ancillary(self, chunks: list[PngChunk], max_len: int) -> bytes:
        """Inject ancillary chunks (gAMA, pHYs, tIME)."""
        choice = random.randint(0, 2)
        if choice == 0:  # gAMA
            gama = self._find_chunk(chunks, b"gAMA")
            if gama:
                data = bytearray(gama.data)
                if len(data) >= 4:
                    struct.pack_into(">I", data, 0, random.randint(0, 0xFFFFFFFF))
                    gama.data = bytes(data)
            else:
                gama_data = struct.pack(">I", random.randint(0, 0xFFFFFFFF))
                chunks.insert(random.randint(1, len(chunks)), PngChunk(b"gAMA", gama_data))
        elif choice == 1:  # pHYs
            phys = self._find_chunk(chunks, b"pHYs")
            if phys:
                data = bytearray(phys.data)
                if len(data) >= 9:
                    struct.pack_into(
                        ">II", data, 0, random.randint(0, 0xFFFFFFFF), random.randint(0, 0xFFFFFFFF)
                    )
                    phys.data = bytes(data)
            else:
                phys_data = struct.pack(
                    ">IIb",
                    random.randint(0, 0xFFFFFFFF),
                    random.randint(0, 0xFFFFFFFF),
                    random.choice([0, 1]),
                )
                chunks.insert(random.randint(1, len(chunks)), PngChunk(b"pHYs", phys_data))
        else:  # tIME
            tyme = self._find_chunk(chunks, b"tIME")
            if tyme and len(tyme.data) >= 7:
                data = bytearray(tyme.data)
                struct.pack_into(
                    ">HBBBBB",
                    data,
                    0,
                    random.randint(1990, 2030),
                    random.randint(1, 12),
                    random.randint(1, 31),
                    random.randint(0, 23),
                    random.randint(0, 59),
                    random.randint(0, 59),
                )
                tyme.data = bytes(data)
            else:
                tyme_data = struct.pack(
                    ">HBBBBB",
                    random.randint(1990, 2030),
                    random.randint(1, 12),
                    random.randint(1, 31),
                    random.randint(0, 23),
                    random.randint(0, 59),
                    random.randint(0, 59),
                )
                chunks.insert(random.randint(1, len(chunks)), PngChunk(b"tIME", tyme_data))
        return serialize_png_chunks(chunks)[:max_len]

    def _micro_idat(self, chunks: list[PngChunk], max_len: int) -> bytes:
        """Replace IDAT with minimal compressed data."""
        idat = self._find_chunk(chunks, b"IDAT")
        if not idat:
            return serialize_png_chunks(chunks)[:max_len]
        compressed = zlib.compress(b"\x00", 0)
        idat.data = compressed
        return serialize_png_chunks(chunks)[:max_len]

    def _corrupt_signature(self, max_len: int) -> bytes:
        """Corrupt the 8-byte PNG signature."""
        sig = bytearray(b"\x89PNG\r\n\x1a\n")
        idx = random.randint(0, 7)
        sig[idx] = random.randint(0, 255)
        return (
            bytes(sig)
            + b"".join(
                c.serialize()
                for c in [
                    PngChunk(b"IHDR", b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02"),
                    PngChunk(b"IEND", b""),
                ]
            )[:max_len]
        )

    def _swap_idat_chunks(self, chunks: list[PngChunk], max_len: int) -> bytes:
        """Swap two IDAT chunks to test decompressor ordering tolerance."""
        idats = [i for i, c in enumerate(chunks) if c.chunk_type == b"IDAT"]
        if len(idats) < 2:
            return self._generate_random_png(max_len)
        a, b = random.sample(idats, 2)
        chunks[a], chunks[b] = chunks[b], chunks[a]
        return serialize_png_chunks(chunks)[:max_len]

    @staticmethod
    def _find_chunk(chunks: list[PngChunk], chunk_type: bytes) -> PngChunk | None:
        for chunk in chunks:
            if chunk.chunk_type == chunk_type:
                return chunk
        return None

    @staticmethod
    def _find_chunk_index(chunks: list[PngChunk], chunk_type: bytes) -> int:
        for i, chunk in enumerate(chunks):
            if chunk.chunk_type == chunk_type:
                return i
        return -1
