"""Structure-aware SQLite database file mutator.

SQLite file layout (see https://www.sqlite.org/fileformat2.html):

  Page 1 (fixed 100-byte database header, then that page's own b-tree page):
    [0:16]   magic: "SQLite format 3\\0"
    [16:18]  page_size, u16 BE (1 means 65536, the one special-cased value)
    [18]     file format write version
    [19]     file format read version
    [20]     bytes of unused "reserved space" at the end of each page
    [21]     max embedded payload fraction (must be 64)
    [22]     min embedded payload fraction (must be 32)
    [23]     leaf payload fraction (must be 32)
    [24:28]  file change counter, u32 BE
    [28:32]  size of the database in pages, u32 BE
    [32:36]  first freelist trunk page, u32 BE
    [36:40]  total freelist pages, u32 BE
    [40:44]  schema cookie, u32 BE
    [44:48]  schema format number (1-4), u32 BE
    [48:52]  default page cache size, u32 BE
    [52:56]  largest root b-tree page (vacuum), u32 BE
    [56:60]  text encoding (1=utf8, 2=utf16le, 3=utf16be), u32 BE
    [60:64]  user version, u32 BE
    [64:68]  incremental-vacuum mode, u32 BE
    [68:72]  application ID, u32 BE
    [72:92]  reserved, must be zero
    [92:96]  version-valid-for number, u32 BE
    [96:100] SQLITE_VERSION_NUMBER, u32 BE

  Every page (page 1's b-tree page starts at byte 100, not 0) begins with a
  b-tree page header:
    [0]      page type: 0x02 interior-index, 0x05 interior-table,
                        0x0a leaf-index, 0x0d leaf-table
    [1:3]    first freeblock offset, u16 BE (0 = none)
    [3:5]    number of cells on the page, u16 BE
    [5:7]    start of the cell content area, u16 BE (0 means 65536)
    [7]      number of fragmented free bytes
    [8:12]   right-most child page number, u32 BE -- interior pages only
  followed by the cell pointer array (num_cells * u16 BE offsets, each
  pointing at a cell elsewhere in the page) and the cells themselves,
  which grow from the end of the page backward.

Field-level mutations of the header and of a page's b-tree header / cell
pointer array are applied by overwriting bytes at their known offsets, the
same convention ``bmp.py``/``webp.py`` use — full record/varint decoding is
not needed for a corruption probe: corrupting a cell pointer to fall outside
the page, or the content-area/freeblock offsets, or the page type byte, all
probe the b-tree walker without understanding the record format inside a
cell.
"""

from __future__ import annotations

import random
import struct

MAGIC = b"SQLite format 3\x00"

# (offset, size, signed) for each mutable header field after the magic.
# size is in bytes; all fields are big-endian.
HEADER_FIELDS: list[tuple[str, int, int]] = [
    ("page_size", 16, 2),
    ("file_format_write_version", 18, 1),
    ("file_format_read_version", 19, 1),
    ("reserved_space", 20, 1),
    ("max_payload_fraction", 21, 1),
    ("min_payload_fraction", 22, 1),
    ("leaf_payload_fraction", 23, 1),
    ("file_change_counter", 24, 4),
    ("db_size_pages", 28, 4),
    ("freelist_trunk_page", 32, 4),
    ("freelist_page_count", 36, 4),
    ("schema_cookie", 40, 4),
    ("schema_format", 44, 4),
    ("default_page_cache_size", 48, 4),
    ("largest_root_page", 52, 4),
    ("text_encoding", 56, 4),
    ("user_version", 60, 4),
    ("incremental_vacuum_mode", 64, 4),
    ("application_id", 68, 4),
    ("version_valid_for", 92, 4),
    ("sqlite_version_number", 96, 4),
]

# Interesting values, sized down per field width by _write_be's masking.
INT_VALUES = [0, 1, 2, 0x7F, 0xFF, 0x7FFF, 0xFFFF, 0x7FFFFFFF, 0xFFFFFFFF]

# Valid page sizes are powers of two from 512 to 32768, plus the 1 => 65536
# special case; also include a few invalid ones as corruption probes.
PAGE_SIZE_VALUES = [1, 512, 1024, 2048, 4096, 8192, 16384, 32768, 0, 3, 65535]

# B-tree page types; 0x02/0x05 are interior (carry a right-pointer), the
# rest plus a few invalid bytes are corruption probes.
PAGE_TYPES = [0x02, 0x05, 0x0A, 0x0D, 0x00, 0x01, 0xFF]

HEADER_SIZE = 100
BTREE_HEADER_LEAF = 8
BTREE_HEADER_INTERIOR = 12


def _page_size_from_header(data: bytes) -> int | None:
    raw = struct.unpack_from(">H", data, 16)[0]
    if raw == 1:
        return 65536
    if raw < 512 or (raw & (raw - 1)) != 0:
        return None
    return raw


def _write_be(data: bytearray, off: int, size: int, value: int) -> None:
    value &= (1 << (size * 8)) - 1
    fmt = {1: ">B", 2: ">H", 4: ">I", 8: ">Q"}.get(size)
    if fmt is None:
        return
    struct.pack_into(fmt, data, off, value)


class SqliteDoc:
    """A parsed SQLite file: the 100-byte header plus a flat page array.

    ``pages[0]`` is page 1's *body* (page_size - 100 bytes: its own b-tree
    page header, cell pointer array, and cells) so that every page after
    the header is a plain, uniformly-sized chunk -- serialization is just
    concatenation.
    """

    __slots__ = ("header", "page_size", "pages")

    def __init__(self, header: bytearray, page_size: int, pages: list[bytearray]) -> None:
        self.header = header
        self.page_size = page_size
        self.pages = pages


def parse_sqlite(data: bytes) -> SqliteDoc | None:
    """Parse a SQLite database file. Returns None unless the layout is sane
    enough to slice into whole pages (magic present, valid page size, file
    length a multiple of the page size)."""
    if len(data) < HEADER_SIZE or data[:16] != MAGIC:
        return None
    page_size = _page_size_from_header(data)
    if page_size is None or len(data) < page_size or len(data) % page_size != 0:
        return None

    header = bytearray(data[:HEADER_SIZE])
    page1_body = bytearray(data[HEADER_SIZE:page_size])
    other_pages = [
        bytearray(data[i : i + page_size]) for i in range(page_size, len(data), page_size)
    ]
    return SqliteDoc(header=header, page_size=page_size, pages=[page1_body, *other_pages])


def serialize_sqlite(doc: SqliteDoc) -> bytes:
    """Concatenate the header and every page body back into file bytes."""
    out = bytearray(doc.header)
    for page in doc.pages:
        out += page
    return bytes(out)


def _btree_header_len(page: bytearray, is_page1: bool) -> int:
    if not page:
        return 0
    page_type = page[0]
    return BTREE_HEADER_INTERIOR if page_type in (0x02, 0x05) else BTREE_HEADER_LEAF


def _cell_pointer_offsets(page: bytearray) -> list[int]:
    """Byte offsets (within *page*) of each entry in the cell pointer array."""
    if len(page) < BTREE_HEADER_LEAF:
        return []
    hdr_len = _btree_header_len(page, is_page1=False)
    ncells = struct.unpack_from(">H", page, 3)[0]
    ptrs = []
    pos = hdr_len
    for _ in range(min(ncells, 8192)):
        if pos + 2 > len(page):
            break
        ptrs.append(pos)
        pos += 2
    return ptrs


class SqliteMutator:
    """Structure-aware SQLite database file mutator."""

    _rng = random

    def mutate(self, data: bytes, max_len: int = 65536, rng=None) -> bytes:
        self._rng = rng or random
        doc = parse_sqlite(data)
        if doc is None:
            return self._generate_random_sqlite(max_len, rng=self._rng)

        mutators = [
            self._mutate_header_field,
            self._mutate_page_size_mismatch,
            self._mutate_page_type,
            self._mutate_cell_count,
            self._mutate_content_area_start,
            self._mutate_freeblock_offset,
            self._mutate_cell_pointer,
            self._mutate_rightmost_pointer,
            self._flip_cell_byte,
            self._swap_pages,
            self._duplicate_page,
            self._delete_page,
            self._truncate_page,
        ]
        op = self._rng.randint(0, len(mutators) - 1)
        result = mutators[op](doc, max_len)
        return serialize_sqlite(result)[:max_len]

    # ── header mutations ─────────────────────────────────────────────

    def _mutate_header_field(self, doc: SqliteDoc, max_len: int) -> SqliteDoc:
        _name, off, size = self._rng.choice(HEADER_FIELDS)
        value = self._rng.choice(INT_VALUES + [self._rng.randint(0, (1 << (size * 8)) - 1)])
        _write_be(doc.header, off, size, value)
        return doc

    def _mutate_page_size_mismatch(self, doc: SqliteDoc, max_len: int) -> SqliteDoc:
        """Corrupt the declared page_size without resizing pages -- a
        declared-size-vs-actual-layout mismatch, same probe as the
        isobmff/webp box-size mutators."""
        _write_be(doc.header, 16, 2, self._rng.choice(PAGE_SIZE_VALUES))
        return doc

    # ── b-tree page mutations ───────────────────────────────────────

    def _mutate_page_type(self, doc: SqliteDoc, max_len: int) -> SqliteDoc:
        page = self._rng.choice(doc.pages)
        if not page:
            return doc
        page[0] = self._rng.choice(PAGE_TYPES)
        return doc

    def _mutate_cell_count(self, doc: SqliteDoc, max_len: int) -> SqliteDoc:
        page = self._rng.choice(doc.pages)
        if len(page) < BTREE_HEADER_LEAF:
            return doc
        current = struct.unpack_from(">H", page, 3)[0]
        value = self._rng.choice(
            [0, 1, current + 1, current * 2 + 1, 0xFFFF, self._rng.randint(0, 0xFFFF)]
        )
        _write_be(page, 3, 2, value)
        return doc

    def _mutate_content_area_start(self, doc: SqliteDoc, max_len: int) -> SqliteDoc:
        page = self._rng.choice(doc.pages)
        if len(page) < BTREE_HEADER_LEAF:
            return doc
        value = self._rng.choice([0, 1, len(page), doc.page_size, self._rng.randint(0, 0xFFFF)])
        _write_be(page, 5, 2, value)
        return doc

    def _mutate_freeblock_offset(self, doc: SqliteDoc, max_len: int) -> SqliteDoc:
        page = self._rng.choice(doc.pages)
        if len(page) < BTREE_HEADER_LEAF:
            return doc
        value = self._rng.choice([0, 1, len(page), self._rng.randint(0, 0xFFFF)])
        _write_be(page, 1, 2, value)
        return doc

    def _mutate_rightmost_pointer(self, doc: SqliteDoc, max_len: int) -> SqliteDoc:
        """Corrupt the interior-page right-most child pointer (no-op on a
        leaf page -- there is nothing at that offset to mean anything)."""
        interior = [p for p in doc.pages if p and p[0] in (0x02, 0x05) and len(p) >= 12]
        if not interior:
            return doc
        page = self._rng.choice(interior)
        _write_be(page, 8, 4, self._rng.choice(INT_VALUES))
        return doc

    def _mutate_cell_pointer(self, doc: SqliteDoc, max_len: int) -> SqliteDoc:
        """Corrupt one entry of a page's cell pointer array so it points
        outside the page, into the header, or onto another pointer -- the
        classic b-tree-walker OOB-read probe."""
        candidates = [(p, _cell_pointer_offsets(p)) for p in doc.pages]
        candidates = [(p, offs) for p, offs in candidates if offs]
        if not candidates:
            return doc
        page, offsets = self._rng.choice(candidates)
        off = self._rng.choice(offsets)
        value = self._rng.choice(
            [0, 1, len(page), len(page) + 1, 0xFFFF, self._rng.randint(0, 0xFFFF)]
        )
        _write_be(page, off, 2, value)
        return doc

    def _flip_cell_byte(self, doc: SqliteDoc, max_len: int) -> SqliteDoc:
        """Flip a byte somewhere past the b-tree header / pointer array --
        i.e. in the cell content area (payload lengths, rowid varints,
        record header, or column data)."""
        page = self._rng.choice(doc.pages)
        hdr_len = _btree_header_len(page, is_page1=False)
        ncells = struct.unpack_from(">H", page, 3)[0] if len(page) >= 5 else 0
        start = hdr_len + 2 * min(ncells, 8192)
        if start >= len(page):
            return doc
        pos = self._rng.randint(start, len(page) - 1)
        page[pos] ^= 1 << self._rng.randint(0, 7)
        return doc

    # ── page-level structural mutations ─────────────────────────────

    def _swap_pages(self, doc: SqliteDoc, max_len: int) -> SqliteDoc:
        """Swap two non-page-1 pages (page 1 carries the file header and
        its b-tree header offsets are relied on by _generate_random_sqlite's
        callers elsewhere, so it is left in place)."""
        if len(doc.pages) >= 3:
            i, j = self._rng.sample(range(1, len(doc.pages)), 2)
            doc.pages[i], doc.pages[j] = doc.pages[j], doc.pages[i]
        return doc

    def _duplicate_page(self, doc: SqliteDoc, max_len: int) -> SqliteDoc:
        if len(doc.pages) >= 2:
            i = self._rng.randint(1, len(doc.pages) - 1)
            doc.pages.insert(i, bytearray(doc.pages[i]))
        return doc

    def _delete_page(self, doc: SqliteDoc, max_len: int) -> SqliteDoc:
        if len(doc.pages) >= 3:
            i = self._rng.randint(1, len(doc.pages) - 1)
            doc.pages.pop(i)
        return doc

    def _truncate_page(self, doc: SqliteDoc, max_len: int) -> SqliteDoc:
        """Zero-pad a page's tail to simulate a torn/short write."""
        if len(doc.pages) >= 2:
            i = self._rng.randint(1, len(doc.pages) - 1)
            page = doc.pages[i]
            cut = self._rng.randint(1, max(1, len(page) - 1))
            doc.pages[i] = bytearray(page[:cut]) + bytearray(len(page) - cut)
        return doc

    # ── generation ───────────────────────────────────────────────────

    def _generate_random_sqlite(self, _doc=None, max_len: int = 65536, rng=None) -> bytes:
        """Generate a minimal single-table-page SQLite database: header +
        page 1 (leaf table b-tree, one cell holding one integer column)."""
        self._rng = rng or self._rng or random
        r = self._rng

        page_size = 4096 if max_len >= 4096 else 512
        header = bytearray(HEADER_SIZE)
        header[0:16] = MAGIC
        struct.pack_into(">H", header, 16, page_size)
        header[18] = 1  # file format write version: legacy
        header[19] = 1  # file format read version: legacy
        header[20] = 0  # reserved space
        header[21] = 64  # max payload fraction
        header[22] = 32  # min payload fraction
        header[23] = 32  # leaf payload fraction
        struct.pack_into(">I", header, 24, 1)  # file change counter
        struct.pack_into(">I", header, 28, 1)  # db size in pages
        struct.pack_into(">I", header, 44, 4)  # schema format number
        struct.pack_into(">I", header, 56, 1)  # text encoding: utf8
        struct.pack_into(">I", header, 92, 1)  # version-valid-for
        struct.pack_into(">I", header, 96, 3045000)  # sqlite_version_number

        # One cell: table-leaf layout is [varint payload_len][varint rowid]
        # [record: header_len byte, serial_type varint, payload bytes].
        value = r.randint(0, 127)
        record = bytes([0x02, 0x01, value])  # header_len=2, serial_type=1 (8-bit int), payload
        cell = bytes([len(record)]) + bytes([1]) + record  # payload_len, rowid=1, record

        page1_body = bytearray(page_size - HEADER_SIZE)
        cell_off = len(page1_body) - len(cell)
        page1_body[cell_off : cell_off + len(cell)] = cell

        page1_body[0] = 0x0D  # leaf table b-tree page
        struct.pack_into(">H", page1_body, 1, 0)  # no freeblocks
        struct.pack_into(">H", page1_body, 3, 1)  # one cell
        struct.pack_into(">H", page1_body, 5, cell_off)  # content area start
        page1_body[7] = 0  # no fragmented bytes
        struct.pack_into(">H", page1_body, 8, cell_off)  # cell pointer array[0]

        return bytes(header) + bytes(page1_body)
