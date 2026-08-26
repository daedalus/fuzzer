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

        ispe = next(
            b for b in doc.meta_children[4].children[0].children if b.box_type == b"ispe"
        )
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
        assert len(doc.pages[0]) == 4096 - 100  # page layout itself untouched

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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
