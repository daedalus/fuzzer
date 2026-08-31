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
from typing import Any

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

# Schema-text substitutions for the CREATE statement stored in
# sqlite_master. Same length class as what they replace so the surrounding
# record's declared payload length stays plausible; overlong ones are
# truncated at the record boundary by the mutator.
SQL_TOKENS = [
    b"CREATE TABLE",
    b"CREATE INDEX",
    b"CREATE VIEW",
    b"CREATE TRIGGER",
    b"CREATE VIRTUAL TABLE",
    b"PRIMARY KEY",
    b"WITHOUT ROWID",
    b"sqlite_master",
    b"sqlite_sequence",
]

HEADER_SIZE = 100
BTREE_HEADER_LEAF = 8
BTREE_HEADER_INTERIOR = 12


def _page_size_from_header(data: bytes) -> int | None:
    raw = int(struct.unpack_from(">H", data, 16)[0])
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


def _varint(value: int) -> bytes:
    """Encode SQLite's big-endian base-128 variable-length integer.

    1-8 bytes carry 7 payload bits each with the high bit as a
    continuation flag; the 9th byte, when reached, carries a full 8 bits.
    """
    value &= 0xFFFFFFFFFFFFFFFF
    if value <= 0x7F:
        return bytes([value])
    if value > 0x00FFFFFFFFFFFFFF:
        out = bytearray([value & 0xFF])
        value >>= 8
        for _ in range(8):
            out.append((value & 0x7F) | 0x80)
            value >>= 7
        return bytes(reversed(out))
    chunks = []
    while value:
        chunks.append(value & 0x7F)
        value >>= 7
    return bytes([c | 0x80 for c in reversed(chunks[1:])] + [chunks[0]])


def _serial(value: int | bytes) -> tuple[int, bytes]:
    """Return (serial_type, body_bytes) for one record column value."""
    if isinstance(value, bytes):
        return 13 + 2 * len(value), value  # TEXT (utf8)
    if -128 <= value <= 127:
        return 1, struct.pack(">b", value)
    if -32768 <= value <= 32767:
        return 2, struct.pack(">h", value)
    return 4, struct.pack(">i", value)


def _record(values: list[int | bytes]) -> bytes:
    """Encode a SQLite record ("serial type" header + column bodies)."""
    types = bytearray()
    body = bytearray()
    for value in values:
        serial_type, raw = _serial(value)
        types += _varint(serial_type)
        body += raw
    # The header length varint counts itself, so its own width has to be
    # settled before it can be written; one retry always converges for any
    # header this code produces.
    header_len = len(types) + 1
    if len(_varint(header_len)) != 1:
        header_len = len(types) + len(_varint(len(types) + 2))
    return _varint(header_len) + bytes(types) + bytes(body)


def _table_leaf_cell(rowid: int, values: list[int | bytes]) -> bytes:
    """Encode one table-leaf b-tree cell: payload length, rowid, record."""
    payload = _record(values)
    return _varint(len(payload)) + _varint(rowid) + payload


def _write_leaf_page(page: bytearray, hdr_off: int, page_size: int, cells: list[bytes]) -> None:
    """Lay out a table-leaf b-tree page in place.

    ``hdr_off`` is where the b-tree header starts (0 for an ordinary page,
    100 for page 1, which carries the file header first). Every *offset*
    the page stores -- the cell pointers and the content-area start -- is
    measured from byte 0 of the page regardless, which is the detail that
    makes page 1 easy to get wrong.
    """
    content_start = page_size
    pointers = []
    for cell in cells:
        content_start -= len(cell)
        page[content_start : content_start + len(cell)] = cell
        pointers.append(content_start)

    page[hdr_off] = 0x0D  # leaf table b-tree page
    struct.pack_into(">H", page, hdr_off + 1, 0)  # no freeblocks
    struct.pack_into(">H", page, hdr_off + 3, len(cells))
    # 0 encodes 65536; a full-size empty page is the only case that hits it.
    struct.pack_into(">H", page, hdr_off + 5, content_start & 0xFFFF)
    page[hdr_off + 7] = 0  # no fragmented free bytes
    for i, ptr in enumerate(pointers):
        struct.pack_into(">H", page, hdr_off + 8 + 2 * i, ptr)


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


def _btree_header_len(page: bytearray) -> int:
    """Length of the b-tree page header: interior pages carry a right-most
    child pointer that leaf pages do not."""
    if not page:
        return 0
    return BTREE_HEADER_INTERIOR if page[0] in (0x02, 0x05) else BTREE_HEADER_LEAF


def _cell_pointer_offsets(page: bytearray) -> list[int]:
    """Byte offsets (within *page*) of each entry in the cell pointer array."""
    if len(page) < BTREE_HEADER_LEAF:
        return []
    hdr_len = _btree_header_len(page)
    ncells = struct.unpack_from(">H", page, 3)[0]
    ptrs = []
    pos = hdr_len
    for _ in range(min(ncells, 8192)):
        if pos + 2 > len(page):
            break
        ptrs.append(pos)
        pos += 2
    return ptrs


def _cell_starts(page: bytearray, is_page1: bool) -> list[int]:
    """Where each cell begins, as an offset into *page*.

    Cell pointers are stored relative to the start of the **page**, but
    ``SqliteDoc.pages[0]`` holds page 1's *body* -- the 100-byte file
    header is sliced off -- so page 1's pointers have to be rebased by
    ``HEADER_SIZE`` before they index into it. Pointers that land outside
    the buffer (already-corrupted input) are dropped rather than clamped.
    """
    base = HEADER_SIZE if is_page1 else 0
    starts = []
    for ptr_off in _cell_pointer_offsets(page):
        ptr = struct.unpack_from(">H", page, ptr_off)[0]
        rel = ptr - base
        if 0 <= rel < len(page):
            starts.append(rel)
    return starts


class SqliteMutator:
    """Structure-aware SQLite database file mutator."""

    _rng = random

    def mutate(self, data: bytes, max_len: int = 65536, rng: Any = None) -> bytes:
        self._rng = rng or random
        doc = parse_sqlite(data)
        if doc is None:
            return self._generate_random_sqlite(max_len=max_len, rng=self._rng)

        mutators = [
            self._mutate_header_field,
            self._mutate_page_size_mismatch,
            self._mutate_page_type,
            self._mutate_cell_count,
            self._mutate_content_area_start,
            self._mutate_freeblock_offset,
            self._mutate_cell_pointer,
            self._mutate_rightmost_pointer,
            self._mutate_cell_header,
            self._mutate_schema_sql,
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
        hdr_len = _btree_header_len(page)
        ncells = struct.unpack_from(">H", page, 3)[0] if len(page) >= 5 else 0
        start = hdr_len + 2 * min(ncells, 8192)
        if start >= len(page):
            return doc
        pos = self._rng.randint(start, len(page) - 1)
        page[pos] ^= 1 << self._rng.randint(0, 7)
        return doc

    def _mutate_cell_header(self, doc: SqliteDoc, max_len: int) -> SqliteDoc:
        """Corrupt a cell's leading varints -- payload length, rowid, or the
        record's own header length / first serial type.

        These drive the record decoder rather than the b-tree walker: a
        payload length that disagrees with the space actually available on
        the page is what pushes SQLite onto its overflow-page path, and an
        out-of-range serial type is what pushes it off the column type map.
        Writing a byte < 0x80 over the first byte of a multi-byte varint
        also truncates it, which shifts every field after it -- a
        re-synchronization probe the fixed-offset header mutators cannot
        reach.
        """
        candidates = [
            (page, _cell_starts(page, is_page1=(i == 0))) for i, page in enumerate(doc.pages)
        ]
        candidates = [(p, starts) for p, starts in candidates if starts]
        if not candidates:
            return doc
        page, starts = self._rng.choice(candidates)
        start = self._rng.choice(starts)
        # Offsets 0/1 are the payload-length and rowid varints; 2/3 land in
        # the record header (header length, then the first serial type).
        pos = start + self._rng.randint(0, 3)
        if pos >= len(page):
            return doc
        page[pos] = self._rng.choice([0x00, 0x01, 0x7F, 0x80, 0xFF, page[pos] ^ 0x80])
        return doc

    def _mutate_schema_sql(self, doc: SqliteDoc, max_len: int) -> SqliteDoc:
        """Corrupt the CREATE statement text stored in ``sqlite_master``.

        This is the only field in the file that SQLite feeds back through
        its SQL parser, and it does so at open time before any query runs,
        so it reaches an entirely different subsystem from every other
        mutation here.
        """
        page = doc.pages[0]
        idx = page.find(b"CREATE")
        if idx < 0:
            return doc
        end = min(len(page), idx + 256)
        if self._rng.random() < 0.5:
            token = self._rng.choice(SQL_TOKENS)
            page[idx : idx + len(token)] = token[: max(0, end - idx)]
        else:
            pos = self._rng.randint(idx, end - 1)
            page[pos] = self._rng.choice([0x00, 0x28, 0x29, 0x22, 0x27, 0xFF])
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

    def _generate_random_sqlite(self, max_len: int = 65536, rng: Any = None) -> bytes:
        """Generate a real, openable two-page SQLite database.

        Page 1 is ``sqlite_master`` (one row describing one table), page 2
        is that table's leaf page with a couple of rows. The output passes
        ``PRAGMA integrity_check`` -- which matters more here than in the
        other format generators: SQLite validates the file header and the
        schema row before it walks anything else, so a seed that is merely
        header-shaped gets rejected at the door and every mutation derived
        from it explores the same three rejection paths. Falls back to a
        valid *empty* single-page database when ``max_len`` is too small to
        hold two pages.
        """
        self._rng = rng or self._rng or random
        r = self._rng

        # Largest legal page size that leaves room for both pages.
        page_size = 512
        for candidate in (1024, 2048, 4096):
            if 2 * candidate <= max_len:
                page_size = candidate
        two_pages = 2 * page_size <= max_len

        table = f"t{r.randint(0, 999)}"
        sql = f"CREATE TABLE {table}(a)"

        # ── page 1: the schema table ────────────────────────────────
        # sqlite_master columns: type, name, tbl_name, rootpage, sql
        page1 = bytearray(page_size)
        page1[0:HEADER_SIZE] = self._build_header(page_size, 2 if two_pages else 1)
        if two_pages:
            schema_cell = _table_leaf_cell(
                rowid=1,
                values=[b"table", table.encode(), table.encode(), 2, sql.encode()],
            )
            _write_leaf_page(page1, HEADER_SIZE, page_size, [schema_cell])
        else:
            # Zero rows: a schema-less but entirely valid database.
            _write_leaf_page(page1, HEADER_SIZE, page_size, [])
            return bytes(page1)[:max_len]

        # ── page 2: the table's own rows ────────────────────────────
        page2 = bytearray(page_size)
        rows = [
            _table_leaf_cell(rowid=i + 1, values=[r.randint(0, 0x7FFF)])
            for i in range(r.randint(1, 3))
        ]
        _write_leaf_page(page2, 0, page_size, rows)

        return (bytes(page1) + bytes(page2))[:max_len]

    @staticmethod
    def _build_header(page_size: int, page_count: int) -> bytearray:
        header = bytearray(HEADER_SIZE)
        header[0:16] = MAGIC
        # page_size is stored as 1 for the 65536 special case; every other
        # legal value fits the u16 directly.
        struct.pack_into(">H", header, 16, 1 if page_size == 65536 else page_size)
        header[18] = 1  # file format write version: legacy
        header[19] = 1  # file format read version: legacy
        header[20] = 0  # reserved space
        header[21] = 64  # max payload fraction (must be 64)
        header[22] = 32  # min payload fraction (must be 32)
        header[23] = 32  # leaf payload fraction (must be 32)
        struct.pack_into(">I", header, 24, 1)  # file change counter
        struct.pack_into(">I", header, 28, page_count)  # db size in pages
        struct.pack_into(">I", header, 40, 1)  # schema cookie
        struct.pack_into(">I", header, 44, 4)  # schema format number
        struct.pack_into(">I", header, 56, 1)  # text encoding: utf8
        struct.pack_into(">I", header, 92, 1)  # version-valid-for == change ctr
        struct.pack_into(">I", header, 96, 3045000)  # sqlite_version_number
        return header
