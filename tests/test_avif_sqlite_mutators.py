"""Tests for the structure-aware AVIF and SQLite mutators.

Follows the convention in ``test_new_format_mutators.py``: parse/serialize
round trip, a declared-size-reaches-the-wire regression, mutate() diversity
via a fixed-seed deterministic sweep (never a retry-until-hit loop), and
degenerate-input safety. Field-level mutation correctness is checked with a
scripted deterministic RNG stand-in rather than a real ``random.Random``, so
the exact byte written is asserted rather than merely "changed".
"""

from __future__ import annotations

import random
import struct

import pytest


def _rng(seed: int = 42) -> random.Random:
    return random.Random(seed)


def _diversity(mutator, data: bytes, rounds: int = 60, max_len: int = 4096) -> int:
    rng = _rng()
    return len({mutator.mutate(data, max_len=max_len, rng=rng) for _ in range(rounds)})


class _FixedRng:
    """Deterministic stand-in: always picks the first/lowest option.

    Lets a test drive a mutation method through an exact, scripted sequence
    of choices instead of sampling `random.Random` and hoping for a hit
    (see AGENTS.md #39) -- every call is reproducible and the resulting byte
    offset/value is asserted exactly.
    """

    def choice(self, seq):
        seq = list(seq)
        return seq[0]

    def randint(self, a, b):
        return a

    def randrange(self, *args):
        return args[0]

    def random(self):
        return 0.0

    def sample(self, population, k):
        return list(population)[:k]


class TestAvifMutations:
    def test_roundtrip_generated(self):
        from fuzzer_tool.core.mutations.avif import AvifMutator, parse_avif, serialize_avif

        data = AvifMutator()._generate_random_avif(max_len=4096, rng=_rng())
        doc = parse_avif(data)
        assert doc is not None
        assert serialize_avif(doc) == data

    def test_sniff_requires_avif_brand(self):
        from fuzzer_tool.core.mutations.avif import sniff_avif

        # A generic (non-AVIF) ISO-BMFF ftyp must not sniff as AVIF.
        mp4 = struct.pack(">I", 16) + b"ftyp" + b"isom" + struct.pack(">I", 0)
        assert sniff_avif(mp4) is False
        assert sniff_avif(b"\x00" * 20) is False

    def test_sniff_accepts_compatible_brand(self):
        from fuzzer_tool.core.mutations.avif import sniff_avif

        # major brand "mif1" but "avif" present in the compatible-brands list
        ftyp = struct.pack(">I", 24) + b"ftyp" + b"mif1" + struct.pack(">I", 0) + b"mif1avif"
        assert sniff_avif(ftyp) is True

    def test_parse_requires_meta_box(self):
        from fuzzer_tool.core.mutations.avif import parse_avif

        ftyp_only = struct.pack(">I", 16) + b"ftyp" + b"avif" + struct.pack(">I", 0)
        assert parse_avif(ftyp_only) is None

    def test_diversity(self):
        from fuzzer_tool.core.mutations.avif import AvifMutator

        data = AvifMutator()._generate_random_avif(max_len=4096, rng=_rng())
        assert _diversity(AvifMutator(), data) > 1

    def test_mutate_never_raises_on_degenerate_input(self):
        from fuzzer_tool.core.mutations.avif import AvifMutator

        mut = AvifMutator()
        for data in (b"", b"f", b"\x00\x00\x00\x10ftypavif", b"\x00" * 40):
            mut.mutate(data, max_len=4096, rng=_rng())

    def test_mutate_respects_max_len(self):
        from fuzzer_tool.core.mutations.avif import AvifMutator

        data = AvifMutator()._generate_random_avif(max_len=4096, rng=_rng())
        for i in range(20):
            out = AvifMutator().mutate(data, max_len=64, rng=_rng(i))
            assert len(out) <= 64

    def test_box_size_written_verbatim(self):
        """Regression: a mutated meta-child box size must reach the wire,
        not be recomputed from the payload -- same probe as isobmff/webp."""
        from fuzzer_tool.core.mutations.avif import AvifMutator, parse_avif

        data = AvifMutator()._generate_random_avif(max_len=4096, rng=_rng())
        doc = parse_avif(data)
        mutator = AvifMutator()
        mutator._rng = _FixedRng()
        mutator._mutate_box_size(doc, max_len=4096)
        # _FixedRng.choice always returns the first candidate: the first
        # meta child, and the first entry of the size-value list (0).
        assert doc.meta_children[0].size_orig == 0

    def test_mutate_ispe_writes_chosen_width(self):
        """Falsification test: ispe's declared width must become the exact
        value the (scripted) RNG picked, at the exact field offset."""
        from fuzzer_tool.core.mutations.avif import AvifMutator, parse_avif

        data = AvifMutator()._generate_random_avif(max_len=4096, rng=_rng())
        doc = parse_avif(data)
        mutator = AvifMutator()
        mutator._rng = _FixedRng()
        mutator._mutate_ispe(doc, max_len=4096)

        ispe = next(b for b in doc.meta_children[4].children[0].children if b.box_type == b"ispe")
        width = struct.unpack_from(">I", ispe.data, 4)[0]
        assert width == 0  # INT_VALUES[0]

    def test_iloc_huge_item_count_does_not_hang_or_crash(self):
        """Adversarial: a malformed iloc claiming a huge item_count must be
        bounded, not walked in full or crash on out-of-range reads."""
        from fuzzer_tool.core.mutations.avif import _iloc_item_offsets

        payload = bytes([0]) + bytes(3) + bytes([0x44, 0x00]) + struct.pack(">H", 0xFFFF)
        items = _iloc_item_offsets(payload)
        assert items == []  # too short to hold even one declared item


class TestSqliteMutations:
    def test_roundtrip_generated(self):
        from fuzzer_tool.core.mutations.sqlite import SqliteMutator, parse_sqlite, serialize_sqlite

        data = SqliteMutator()._generate_random_sqlite(max_len=4096, rng=_rng())
        doc = parse_sqlite(data)
        assert doc is not None
        assert serialize_sqlite(doc) == data

    def test_parse_requires_magic(self):
        from fuzzer_tool.core.mutations.sqlite import parse_sqlite

        assert parse_sqlite(b"\x00" * 200) is None
        assert parse_sqlite(b"not a database" + b"\x00" * 100) is None

    def test_parse_requires_page_size_multiple(self):
        from fuzzer_tool.core.mutations.sqlite import MAGIC, parse_sqlite

        header = bytearray(100)
        header[0:16] = MAGIC
        struct.pack_into(">H", header, 16, 4096)
        # File length not a multiple of the declared page size.
        assert parse_sqlite(bytes(header) + b"\x00" * 50) is None

    def test_diversity(self):
        from fuzzer_tool.core.mutations.sqlite import SqliteMutator

        data = SqliteMutator()._generate_random_sqlite(max_len=4096, rng=_rng())
        assert _diversity(SqliteMutator(), data) > 1

    def test_mutate_never_raises_on_degenerate_input(self):
        from fuzzer_tool.core.mutations.sqlite import SqliteMutator

        mut = SqliteMutator()
        for data in (b"", b"S", b"SQLite format 3\x00", b"\x00" * 100):
            mut.mutate(data, max_len=4096, rng=_rng())

    def test_mutate_respects_max_len(self):
        from fuzzer_tool.core.mutations.sqlite import SqliteMutator

        data = SqliteMutator()._generate_random_sqlite(max_len=4096, rng=_rng())
        for i in range(20):
            out = SqliteMutator().mutate(data, max_len=64, rng=_rng(i))
            assert len(out) <= 64

    def test_page_size_written_verbatim(self):
        """Regression: a mutated page_size field must reach the wire without
        the page array being resized to match -- a declared-size-vs-actual-
        layout mismatch probe, same convention as the isobmff/webp mutators."""
        from fuzzer_tool.core.mutations.sqlite import SqliteMutator, parse_sqlite

        data = SqliteMutator()._generate_random_sqlite(max_len=4096, rng=_rng())
        doc = parse_sqlite(data)
        mutator = SqliteMutator()
        mutator._rng = _FixedRng()
        mutator._mutate_page_size_mismatch(doc, max_len=4096)
        written = struct.unpack_from(">H", doc.header, 16)[0]
        assert written == 1  # PAGE_SIZE_VALUES[0]
        # The page array itself must be untouched: its size still reflects
        # the page size the file was *built* with, not the one now declared.
        # pages[0] is page 1's *body*, i.e. page_size minus the file header.
        assert len(doc.pages[0]) == doc.page_size - 100

    def test_cell_pointer_mutation_writes_chosen_value(self):
        """Falsification test: the first cell pointer entry must become the
        exact value the (scripted) RNG picked."""
        from fuzzer_tool.core.mutations.sqlite import SqliteMutator, parse_sqlite

        data = SqliteMutator()._generate_random_sqlite(max_len=4096, rng=_rng())
        doc = parse_sqlite(data)
        mutator = SqliteMutator()
        mutator._rng = _FixedRng()
        mutator._mutate_cell_pointer(doc, max_len=4096)
        page = doc.pages[0]
        pointer = struct.unpack_from(">H", page, 8)[0]  # leaf header is 8 bytes
        assert pointer == 0  # first candidate value list entry

    def test_delete_page_never_touches_header(self):
        """Adversarial: repeated page deletion must never remove page 1's
        body (which carries the schema's root b-tree page) or the header."""
        from fuzzer_tool.core.mutations.sqlite import SqliteMutator, parse_sqlite

        data = SqliteMutator()._generate_random_sqlite(max_len=4096, rng=_rng())
        doc = parse_sqlite(data)
        mutator = SqliteMutator()
        for i in range(50):
            mutator._delete_page(doc, max_len=4096)
            mutator._rng = _rng(i)
        assert doc.header[:16] == b"SQLite format 3\x00"
        assert len(doc.pages) >= 1


class TestGeneratorRegressions:
    """Regressions for the WIP-era generator defects.

    Both generators carried a ``_generate_random_X(self, _doc=None,
    max_len=...)`` signature (copied from ``zip.py``/``isobmff.py``, where
    the generator doubles as a mutation op and so takes a parsed doc first)
    but were called positionally as ``(max_len, rng=...)``. ``max_len``
    landed in ``_doc``, was never read, and the generator silently used its
    default -- so ``mutate()`` on unparseable input returned output larger
    than the caller asked for.
    """

    def test_avif_generate_honours_max_len(self):
        from fuzzer_tool.core.mutations.avif import AvifMutator

        for limit in (32, 64, 128, 4096):
            out = AvifMutator()._generate_random_avif(max_len=limit, rng=_rng())
            assert len(out) <= limit

    def test_sqlite_generate_honours_max_len(self):
        from fuzzer_tool.core.mutations.sqlite import SqliteMutator

        for limit in (64, 512, 1024, 4096):
            out = SqliteMutator()._generate_random_sqlite(max_len=limit, rng=_rng())
            assert len(out) <= limit

    def test_avif_mutate_on_unparseable_input_honours_max_len(self):
        """The generator path is the one that used to overshoot."""
        from fuzzer_tool.core.mutations.avif import AvifMutator

        for limit in (16, 64, 200):
            out = AvifMutator().mutate(b"not an avif file", max_len=limit, rng=_rng())
            assert len(out) <= limit

    def test_sqlite_mutate_on_unparseable_input_honours_max_len(self):
        from fuzzer_tool.core.mutations.sqlite import SqliteMutator

        for limit in (16, 64, 600):
            out = SqliteMutator().mutate(b"not a database", max_len=limit, rng=_rng())
            assert len(out) <= limit

    def test_avif_generated_container_sizes_are_real(self):
        """Regression: ``ipco``/``iprp`` were built with ``size_orig=0``.

        ``serialize_boxes`` writes ``size_orig`` verbatim, and
        ``_parse_boxes`` stops at any box declaring ``size < 8`` -- so a
        zero-sized container silently truncated every box after it on the
        way back in, losing the entire property tree.
        """
        from fuzzer_tool.core.mutations.avif import AvifMutator, parse_avif

        data = AvifMutator()._generate_random_avif(max_len=4096, rng=_rng())
        doc = parse_avif(data)
        assert [b.box_type for b in doc.meta_children] == [
            b"hdlr",
            b"pitm",
            b"iloc",
            b"iinf",
            b"iprp",
        ]
        iprp = doc.meta_children[4]
        assert [b.box_type for b in iprp.children] == [b"ipco", b"ipma"]
        assert [b.box_type for b in iprp.children[0].children] == [b"ispe", b"av1C"]

    def test_avif_generated_iinf_keeps_its_fullbox_prefix(self):
        """Regression: ``iinf`` was built with both ``data`` and
        ``children``, but ``serialize_boxes`` drops ``data`` whenever
        ``children`` is non-empty -- so the version/flags + entry_count
        prefix never reached the wire and ``_infe_entries`` found nothing.
        """
        from fuzzer_tool.core.mutations.avif import (
            AvifMutator,
            _find_leaf,
            _infe_entries,
            parse_avif,
        )

        data = AvifMutator()._generate_random_avif(max_len=4096, rng=_rng())
        doc = parse_avif(data)
        parent, idx = _find_leaf(doc.meta_children, b"iinf")[0]
        payload = parent[idx].data
        assert payload[:4] == b"\x00\x00\x00\x00"  # version 0, flags 0
        assert struct.unpack_from(">H", payload, 4)[0] == 1  # entry_count
        assert len(_infe_entries(payload)) == 1


class TestSqliteGeneratedDatabaseIsReal:
    """The generated seed must be a database SQLite will actually open.

    This matters more here than for the other format generators: SQLite
    validates the file header and parses the ``sqlite_master`` schema row
    before it walks anything else, so a merely header-shaped seed is
    rejected at the door and every mutation derived from it re-explores the
    same handful of rejection paths. The WIP generator wrote page 1's cell
    pointers and content-area start relative to the page *body* rather than
    the page, leaving them 100 bytes short, and real SQLite refused the
    file with "malformed database schema".
    """

    @staticmethod
    def _open(data: bytes):
        import sqlite3
        import tempfile

        path = tempfile.mktemp(suffix=".db")
        with open(path, "wb") as fh:
            fh.write(data)
        return sqlite3.connect(path)

    def test_integrity_check_passes(self):
        from fuzzer_tool.core.mutations.sqlite import SqliteMutator

        for seed in range(8):
            data = SqliteMutator()._generate_random_sqlite(max_len=4096, rng=_rng(seed))
            conn = self._open(data)
            try:
                assert conn.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
            finally:
                conn.close()

    def test_schema_row_is_queryable(self):
        from fuzzer_tool.core.mutations.sqlite import SqliteMutator

        data = SqliteMutator()._generate_random_sqlite(max_len=4096, rng=_rng())
        conn = self._open(data)
        try:
            rows = conn.execute("SELECT type, name, rootpage, sql FROM sqlite_master").fetchall()
            assert len(rows) == 1
            kind, name, rootpage, sql = rows[0]
            assert kind == "table"
            assert rootpage == 2
            assert sql == f"CREATE TABLE {name}(a)"
            # The table's own page must be readable too, not just declared.
            assert conn.execute(f'SELECT a FROM "{name}"').fetchall()
        finally:
            conn.close()

    def test_single_page_fallback_is_still_valid(self):
        """Below two pages the generator emits an empty-but-valid database
        rather than a truncated two-page one."""
        from fuzzer_tool.core.mutations.sqlite import SqliteMutator

        data = SqliteMutator()._generate_random_sqlite(max_len=600, rng=_rng())
        assert len(data) == 512
        conn = self._open(data)
        try:
            assert conn.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
            assert conn.execute("SELECT * FROM sqlite_master").fetchall() == []
        finally:
            conn.close()


class TestSqliteRecordEncoding:
    """The varint/record encoders the generator is built on."""

    @staticmethod
    def _decode_varint(data: bytes) -> tuple[int, int]:
        value = 0
        for i in range(8):
            byte = data[i]
            if byte & 0x80:
                value = (value << 7) | (byte & 0x7F)
            else:
                return (value << 7) | byte, i + 1
        return (value << 8) | data[8], 9

    @pytest.mark.parametrize(
        "value",
        [0, 1, 0x7F, 0x80, 0x81, 0x3FFF, 0x4000, 0xFFFF, 0xFFFFFF, 0x00FFFFFFFFFFFFFF, 2**63],
    )
    def test_varint_round_trips(self, value):
        from fuzzer_tool.core.mutations.sqlite import _varint

        encoded = _varint(value)
        assert 1 <= len(encoded) <= 9
        decoded, consumed = self._decode_varint(encoded)
        assert decoded == value
        assert consumed == len(encoded)

    def test_varint_is_minimal_width(self):
        from fuzzer_tool.core.mutations.sqlite import _varint

        assert len(_varint(0x7F)) == 1
        assert len(_varint(0x80)) == 2
        assert len(_varint(0x3FFF)) == 2
        assert len(_varint(0x4000)) == 3

    def test_record_header_length_counts_itself(self):
        from fuzzer_tool.core.mutations.sqlite import _record

        record = _record([b"table", 2])
        header_len, consumed = self._decode_varint(record)
        # The declared header length spans the length varint plus every
        # serial-type varint -- and nothing else.
        assert header_len == consumed + 2

    def test_record_serial_types_match_values(self):
        from fuzzer_tool.core.mutations.sqlite import _record

        record = _record([b"abc", 5])
        header_len, consumed = self._decode_varint(record)
        types = record[consumed:header_len]
        assert types[0] == 13 + 2 * 3  # TEXT of length 3
        assert types[1] == 1  # 8-bit int
        assert record[header_len:] == b"abc" + bytes([5])


class TestOperatorWiring:
    """Both mutators were unreachable dead code before this: no sniffer, no
    category membership, no handler. Same gap the ogg/flv/asf/riff wiring
    commit closed."""

    def test_operators_are_registered(self):
        from fuzzer_tool.core.operator_registry import REGISTRY

        names = REGISTRY.names()
        assert "avif_chunk_mutate" in names
        assert "sqlite_chunk_mutate" in names

    def test_handlers_exist_on_the_engine(self):
        from fuzzer_tool.services.operators import OperatorEngine

        assert hasattr(OperatorEngine, "_op_avif_chunk_mutate")
        assert hasattr(OperatorEngine, "_op_sqlite_chunk_mutate")

    def test_avif_gate_accepts_generated_avif(self):
        from fuzzer_tool.core.mutations.avif import AvifMutator
        from fuzzer_tool.core.operator_registry import format_gate_matches

        data = AvifMutator()._generate_random_avif(max_len=4096, rng=_rng())
        assert format_gate_matches("avif_chunk_mutate", data) is True

    def test_avif_gate_declines_generic_isobmff(self):
        """A plain MP4 must not pull in the AVIF-specific operator, even
        though both are ftyp-led ISO-BMFF."""
        from fuzzer_tool.core.operator_registry import format_gate_matches

        mp4 = struct.pack(">I", 16) + b"ftyp" + b"isom" + struct.pack(">I", 0)
        assert format_gate_matches("avif_chunk_mutate", mp4) is False

    def test_avif_gate_accepts_compatible_brand_form(self):
        from fuzzer_tool.core.operator_registry import format_gate_matches

        ftyp = struct.pack(">I", 24) + b"ftyp" + b"mif1" + struct.pack(">I", 0) + b"mif1avif"
        assert format_gate_matches("avif_chunk_mutate", ftyp) is True

    def test_avif_gate_is_bounded_on_hostile_ftyp_size(self):
        """Adversarial: an ftyp declaring a 4 GiB size must not make the
        compatible-brands scan walk the whole buffer."""
        from fuzzer_tool.core.operator_registry import format_gate_matches

        hostile = (
            struct.pack(">I", 0xFFFFFFFF) + b"ftyp" + b"isom" + struct.pack(">I", 0) + b"x" * 200000
        )
        assert format_gate_matches("avif_chunk_mutate", hostile) is False

    def test_sqlite_gate_accepts_generated_database(self):
        from fuzzer_tool.core.mutations.sqlite import SqliteMutator
        from fuzzer_tool.core.operator_registry import format_gate_matches

        data = SqliteMutator()._generate_random_sqlite(max_len=4096, rng=_rng())
        assert format_gate_matches("sqlite_chunk_mutate", data) is True

    def test_sqlite_gate_declines_magic_without_a_full_header(self):
        from fuzzer_tool.core.operator_registry import format_gate_matches

        assert format_gate_matches("sqlite_chunk_mutate", b"SQLite format 3\x00") is False

    def test_gates_do_not_cross_match(self):
        from fuzzer_tool.core.mutations.avif import AvifMutator
        from fuzzer_tool.core.mutations.sqlite import SqliteMutator
        from fuzzer_tool.core.operator_registry import format_gate_matches

        avif = AvifMutator()._generate_random_avif(max_len=4096, rng=_rng())
        db = SqliteMutator()._generate_random_sqlite(max_len=4096, rng=_rng())
        assert format_gate_matches("sqlite_chunk_mutate", avif) is False
        assert format_gate_matches("avif_chunk_mutate", db) is False


class TestNewMutationOperators:
    """The operators added while completing the two modules."""

    def test_every_avif_operator_survives_the_generated_seed(self):
        """Sweep, not sample: each op runs against a freshly parsed doc so
        one op's corruption can't mask another's crash."""
        from fuzzer_tool.core.mutations.avif import AvifMutator, parse_avif, serialize_avif

        data = AvifMutator()._generate_random_avif(max_len=4096, rng=_rng())
        mutator = AvifMutator()
        ops = [
            name
            for name in dir(mutator)
            if name.startswith(("_mutate_", "_swap_", "_delete_", "_duplicate_"))
        ]
        assert len(ops) >= 16
        for name in ops:
            for seed in range(20):
                mutator._rng = _rng(seed)
                doc = parse_avif(data)
                serialize_avif(getattr(mutator, name)(doc, 4096))

    def test_avif_mdat_obu_retypes_the_leading_obu(self):
        from fuzzer_tool.core.mutations.avif import AvifMutator, parse_avif

        data = AvifMutator()._generate_random_avif(max_len=4096, rng=_rng())
        doc = parse_avif(data)
        original = doc.top_boxes[doc.mdat_idx].data
        mutator = AvifMutator()
        mutator._rng = _FixedRng()  # randint -> 0 -> the retype branch
        mutator._mutate_mdat_obu(doc, max_len=4096)
        mutated = doc.top_boxes[doc.mdat_idx].data
        # obu_type is bits 6..3; _FixedRng picks the first type in the pool
        # (1, the sequence header) and leaves the surrounding bits alone.
        assert (mutated[0] >> 3) & 0x0F == 1
        assert mutated[0] & 0x81 == original[0] & 0x81
        assert mutated[1:] == original[1:]

    def test_avif_hdlr_type_is_swapped_in_place(self):
        from fuzzer_tool.core.mutations.avif import AvifMutator, _find_leaf, parse_avif

        data = AvifMutator()._generate_random_avif(max_len=4096, rng=_rng())
        doc = parse_avif(data)
        parent, idx = _find_leaf(doc.meta_children, b"hdlr")[0]
        before = len(parent[idx].data)
        mutator = AvifMutator()
        mutator._rng = _FixedRng()
        mutator._mutate_hdlr_type(doc, max_len=4096)
        parent, idx = _find_leaf(doc.meta_children, b"hdlr")[0]
        assert parent[idx].data[8:12] == b"pict"  # first candidate
        assert len(parent[idx].data) == before  # swapped in place, not resized

    def test_avif_iinf_entry_count_mismatch_keeps_entries(self):
        from fuzzer_tool.core.mutations.avif import (
            AvifMutator,
            _find_leaf,
            _infe_entries,
            parse_avif,
        )

        data = AvifMutator()._generate_random_avif(max_len=4096, rng=_rng())
        doc = parse_avif(data)
        mutator = AvifMutator()
        mutator._rng = _FixedRng()
        mutator._mutate_iinf_entry_count(doc, max_len=4096)
        parent, idx = _find_leaf(doc.meta_children, b"iinf")[0]
        payload = parent[idx].data
        assert struct.unpack_from(">H", payload, 4)[0] == 0  # INT_VALUES[0]
        assert len(_infe_entries(payload)) == 1  # the entry itself survives

    def test_every_sqlite_operator_survives_the_generated_seed(self):
        from fuzzer_tool.core.mutations.sqlite import (
            SqliteMutator,
            parse_sqlite,
            serialize_sqlite,
        )

        data = SqliteMutator()._generate_random_sqlite(max_len=4096, rng=_rng())
        mutator = SqliteMutator()
        ops = [
            name
            for name in dir(mutator)
            if name.startswith(
                ("_mutate_", "_flip_", "_swap_", "_delete_", "_duplicate_", "_truncate_")
            )
        ]
        assert len(ops) >= 15
        for name in ops:
            for seed in range(20):
                mutator._rng = _rng(seed)
                doc = parse_sqlite(data)
                serialize_sqlite(getattr(mutator, name)(doc, 4096))

    def test_sqlite_schema_sql_mutation_targets_the_create_statement(self):
        from fuzzer_tool.core.mutations.sqlite import SqliteMutator, parse_sqlite

        data = SqliteMutator()._generate_random_sqlite(max_len=4096, rng=_rng())
        doc = parse_sqlite(data)
        before = bytes(doc.pages[0])
        idx = before.find(b"CREATE")
        assert idx > 0
        mutator = SqliteMutator()
        mutator._rng = _FixedRng()  # random() -> 0.0 -> the token branch
        mutator._mutate_schema_sql(doc, max_len=4096)
        after = bytes(doc.pages[0])
        assert after[idx : idx + 12] == b"CREATE TABLE"  # SQL_TOKENS[0]
        assert len(after) == len(before)  # page length is never disturbed
        assert after[:idx] == before[:idx]  # nothing before the statement moved

    def test_sqlite_schema_sql_mutation_is_a_no_op_without_a_schema(self):
        """The single-page fallback database has no CREATE text; the
        operator must decline rather than corrupt an arbitrary offset."""
        from fuzzer_tool.core.mutations.sqlite import SqliteMutator, parse_sqlite

        data = SqliteMutator()._generate_random_sqlite(max_len=600, rng=_rng())
        doc = parse_sqlite(data)
        before = bytes(doc.pages[0])
        mutator = SqliteMutator()
        mutator._rng = _FixedRng()
        mutator._mutate_schema_sql(doc, max_len=4096)
        assert bytes(doc.pages[0]) == before

    def test_sqlite_cell_header_rebases_page1_pointers(self):
        """Page 1's cell pointers are page-relative but ``pages[0]`` holds
        the page *body*, so they need rebasing by the 100-byte file header
        before they index into it. Without that, the mutation lands 100
        bytes early -- inside the cell pointer array, not the cell."""
        from fuzzer_tool.core.mutations.sqlite import SqliteMutator, _cell_starts, parse_sqlite

        data = SqliteMutator()._generate_random_sqlite(max_len=4096, rng=_rng())
        doc = parse_sqlite(data)
        page1 = doc.pages[0]
        starts = _cell_starts(page1, is_page1=True)
        assert len(starts) == 1
        pointer = struct.unpack_from(">H", page1, 8)[0]
        assert starts[0] == pointer - 100
        # The rebased offset must actually land on the schema cell, whose
        # record contains the table name.
        assert b"CREATE" in bytes(page1[starts[0] : starts[0] + 64])

    def test_sqlite_cell_header_writes_chosen_byte(self):
        from fuzzer_tool.core.mutations.sqlite import SqliteMutator, _cell_starts, parse_sqlite

        data = SqliteMutator()._generate_random_sqlite(max_len=4096, rng=_rng())
        doc = parse_sqlite(data)
        start = _cell_starts(doc.pages[0], is_page1=True)[0]
        mutator = SqliteMutator()
        mutator._rng = _FixedRng()  # choice -> first page, first cell, 0x00
        mutator._mutate_cell_header(doc, max_len=4096)
        assert doc.pages[0][start] == 0x00

    def test_sqlite_cell_starts_drops_out_of_range_pointers(self):
        """Adversarial: an already-corrupted pointer must be dropped, not
        clamped into a bogus in-range offset."""
        from fuzzer_tool.core.mutations.sqlite import SqliteMutator, _cell_starts, parse_sqlite

        data = SqliteMutator()._generate_random_sqlite(max_len=4096, rng=_rng())
        doc = parse_sqlite(data)
        struct.pack_into(">H", doc.pages[0], 8, 0xFFFF)
        assert _cell_starts(doc.pages[0], is_page1=True) == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
