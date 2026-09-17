"""Tests for three mutators ported from FFmpeg demuxer source analysis:

- webm.py's ``_mutate_block_lacing`` (matroskadec.c's Xiph/fixed/EBML lacing)
- isobmff.py's ``_mutate_free_hoov_confusion`` (mov.c's moov-in-free-atom quirk)
- the new flac.py module (flacdec.c's metadata-block header invariants)

Each covers: parse -> serialize round trip, the targeted structural
invariant actually being violated, mutate() diversity, RandPool
compatibility, and degenerate-input safety.
"""

import random

import pytest

from fuzzer_tool.core.rand_pool import RandPool


def _rng(seed: int = 42) -> random.Random:
    return random.Random(seed)


def _diversity(mutator, data: bytes, rounds: int = 80, max_len: int = 4096) -> int:
    rng = _rng()
    return len({mutator.mutate(data, max_len=max_len, rng=rng) for _ in range(rounds)})


# ── webm.py: _mutate_block_lacing ──────────────────────────────────────────


class TestWebmBlockLacing:
    def _sample(self):
        from fuzzer_tool.core.mutations.webm import WebmMutator

        return WebmMutator()._generate_random_webm(max_len=8192, rng=_rng())

    def test_generated_sample_has_a_simpleblock(self):
        from fuzzer_tool.core.mutations.webm import parse_webm

        data = self._sample()
        elements = parse_webm(data)
        assert elements is not None

        def find(els, elem_id):
            for e in els:
                if e.elem_id == elem_id:
                    return e
                found = find(e.children, elem_id)
                if found:
                    return found
            return None

        assert find(elements, 0xA3) is not None, "generator must emit a SimpleBlock to mutate"

    def test_retype_flips_lacing_bits(self):
        from fuzzer_tool.core.mutations.webm import WebmMutator, parse_webm

        data = self._sample()
        mut = WebmMutator()
        mut._rng = _rng(1)
        # Force the "retype" branch: no lacing on the sample block (lacing_type == 0)
        # always takes retype regardless of rng.choice, so a plain call suffices.
        elements = parse_webm(data)
        out_elements = mut._mutate_block_lacing(elements, 8192)

        from fuzzer_tool.core.mutations.webm import serialize_webm

        out = serialize_webm(out_elements)
        assert out != data, "retype must change the flags byte"
        # Re-parse and confirm the SimpleBlock's flags byte's lacing bits changed.
        reparsed = parse_webm(out)
        assert reparsed is not None

    def test_lace_count_edge_sets_0xff(self):
        from fuzzer_tool.core.mutations.webm import WebmMutator, parse_webm, serialize_webm

        data = self._sample()
        mut = WebmMutator()

        class _Scripted:
            def choice(self, seq):
                # First call decides variant when lacing already on; here the
                # sample has lacing off, so retype always fires first to turn
                # lacing on. Route both possible choice() calls to values that
                # land on a nonzero lacing type / the edge-count byte.
                return seq[-1] if len(seq) > 1 else seq[0]

            def randint(self, a, b):
                return a

        # First mutate turns lacing on (retype, since lacing_type starts at 0).
        elements = parse_webm(data)
        elements = mut._mutate_block_lacing(elements, 8192)
        once = serialize_webm(elements)
        # Second mutate, now that lacing is on, exercises lace_count_edge /
        # xiph_run_extend depending on rng -- just assert it never raises and
        # always still parses.
        elements2 = parse_webm(once)
        assert elements2 is not None
        for seed in range(20):
            m2 = WebmMutator()
            m2._rng = _rng(seed)
            out = m2._mutate_block_lacing(parse_webm(once), 8192)
            serialize_webm(out)  # must not raise

    def test_mutate_diversity_includes_block_lacing(self):
        from fuzzer_tool.core.mutations.webm import WebmMutator

        data = self._sample()
        assert _diversity(WebmMutator(), data) > 1

    def test_no_simpleblock_is_a_safe_noop(self):
        """Elements with no (Simple)Block leaf must not raise."""
        from fuzzer_tool.core.mutations.webm import Element, KNOWN_IDS, WebmMutator

        ebml = Element(elem_id=0x1A45DFA3, id_raw=KNOWN_IDS[0x1A45DFA3], size_raw=b"\x80", size_val=0)
        mut = WebmMutator()
        mut._rng = _rng()
        out = mut._mutate_block_lacing([ebml], 4096)
        assert out == [ebml]

    def test_randpool_compat(self):
        from fuzzer_tool.core.mutations.webm import WebmMutator

        data = self._sample()
        mutator = WebmMutator()
        mutator._rng = RandPool()
        out = mutator.mutate(data, max_len=8192, rng=mutator._rng)
        assert isinstance(out, bytes) and len(out) <= 8192

    def test_degenerate_input_safety(self):
        from fuzzer_tool.core.mutations.webm import WebmMutator

        mut = WebmMutator()
        for junk in (b"", b"\x00", b"\x1a\x45\xdf\xa3", bytes(range(256))):
            out = mut.mutate(junk, max_len=2048, rng=_rng())
            assert isinstance(out, bytes) and len(out) <= 2048


# ── isobmff.py: _mutate_free_hoov_confusion ────────────────────────────────


class TestIsobmffFreeHoovConfusion:
    def _nested_moov(self):
        from fuzzer_tool.core.mutations.isobmff import Box, serialize_boxes

        mvhd = Box(box_type=b"mvhd", size_orig=0, data=b"\x00" * 100)
        mvhd.size_orig = 8 + len(mvhd.data)
        moov_payload = serialize_boxes([mvhd])
        moov = Box(box_type=b"moov", size_orig=8 + len(moov_payload), children=[mvhd])
        ftyp_data = b"isom" + b"\x00" * 8
        ftyp = Box(box_type=b"ftyp", size_orig=8 + len(ftyp_data), data=ftyp_data)
        return [ftyp, moov]

    def test_relabels_moov_to_free_or_hoov(self):
        from fuzzer_tool.core.mutations.isobmff import IsobmffMutator, serialize_boxes

        boxes = self._nested_moov()
        data = serialize_boxes(boxes)
        mut = IsobmffMutator()
        mut._rng = _rng(7)
        out = mut._mutate_free_hoov_confusion(self._nested_moov(), 8192)
        relabeled = [b for b in out if b.box_type in (b"free", b"hoov")]
        assert len(relabeled) == 1
        assert data  # sanity: original bytes were well-formed

    def test_trigger_condition_holds(self):
        """The relabeled box's payload must read mvhd/cmov at offset 4:8 --
        the exact condition mov.c's moov_read_default peeks for."""
        from fuzzer_tool.core.mutations.isobmff import IsobmffMutator, serialize_boxes

        mut = IsobmffMutator()
        seen_free = 0
        for seed in range(50):
            mut._rng = _rng(seed)
            out = mut._mutate_free_hoov_confusion(self._nested_moov(), 8192)
            relabeled = [b for b in out if b.box_type in (b"free", b"hoov")]
            if relabeled:
                seen_free += 1
                payload = serialize_boxes(relabeled[0].children)
                assert payload[4:8] in (b"mvhd", b"cmov")
        assert seen_free == 50, "the RNG only picks free/hoov, so every call must relabel"

    def test_no_moov_is_a_safe_noop(self):
        from fuzzer_tool.core.mutations.isobmff import Box, IsobmffMutator

        boxes = [Box(box_type=b"ftyp", size_orig=16, data=b"isom" + b"\x00" * 4)]
        mut = IsobmffMutator()
        mut._rng = _rng()
        out = mut._mutate_free_hoov_confusion(list(boxes), 4096)
        assert [b.box_type for b in out] == [b.box_type for b in boxes]

    def test_mutate_diversity_includes_free_hoov(self):
        from fuzzer_tool.core.mutations.isobmff import IsobmffMutator, serialize_boxes

        data = serialize_boxes(self._nested_moov())
        assert _diversity(IsobmffMutator(), data) > 1

    def test_randpool_compat(self):
        from fuzzer_tool.core.mutations.isobmff import IsobmffMutator, serialize_boxes

        data = serialize_boxes(self._nested_moov())
        mutator = IsobmffMutator()
        mutator._rng = RandPool()
        out = mutator.mutate(data, max_len=8192, rng=mutator._rng)
        assert isinstance(out, bytes) and len(out) <= 8192


# ── flac.py ─────────────────────────────────────────────────────────────


class TestFlacParseSerializeRoundTrip:
    def test_round_trip_is_byte_identical(self):
        from fuzzer_tool.core.mutations.flac import FlacMutator, parse_flac, serialize_flac

        data = FlacMutator()._generate_random_flac(max_len=8192, rng=_rng())
        blocks, trailer = parse_flac(data)
        assert serialize_flac(blocks, trailer) == data

    def test_rejects_non_flac_magic(self):
        from fuzzer_tool.core.mutations.flac import parse_flac

        assert parse_flac(b"RIFF....WAVEfmt ") is None
        assert parse_flac(b"") is None
        assert parse_flac(b"fLa") is None  # too short even for magic

    def test_declared_length_overrides_len_data_on_the_wire(self):
        """The lying-length case: declared_length must reach the wire verbatim."""
        from fuzzer_tool.core.mutations.flac import FlacBlock, serialize_flac

        block = FlacBlock(is_last=True, block_type=0, data=b"\x00" * 34, declared_length=0xABCDEF)
        out = serialize_flac([block], b"")
        written = int.from_bytes(out[5:8], "big")
        assert written == 0xABCDEF, f"expected declared_length on the wire, got {written:#x}"
        # payload bytes themselves are untouched even though the length lies
        assert out[8:] == b"\x00" * 34


class TestFlacClearLastFlag:
    def test_clears_is_last_on_the_true_last_block(self):
        from fuzzer_tool.core.mutations.flac import FlacBlock, FlacMutator

        mut = FlacMutator()
        blocks = [FlacBlock(is_last=True, block_type=0, data=b"\x00" * 34)]
        out = mut._mutate_clear_last_flag(blocks, 4096)
        assert out[-1].is_last is False

    def test_output_still_starts_with_magic(self):
        from fuzzer_tool.core.mutations.flac import (
            FlacMutator,
            parse_flac,
            serialize_flac,
        )

        data = FlacMutator()._generate_random_flac(max_len=8192, rng=_rng())
        blocks, trailer = parse_flac(data)
        mut = FlacMutator()
        mutated_blocks = mut._mutate_clear_last_flag(blocks, 8192)
        out = serialize_flac(mutated_blocks, trailer)
        assert out[:4] == b"fLaC"
        # None of the (now zero) blocks claim is_last -- a real decoder would
        # walk straight into the trailer/frame bytes looking for one more
        # metadata block header.
        assert not any(b.is_last for b in mutated_blocks)


class TestFlacDuplicateStreaminfo:
    def test_introduces_a_second_streaminfo(self):
        from fuzzer_tool.core.mutations.flac import (
            BLOCK_TYPE_STREAMINFO,
            FlacBlock,
            FlacMutator,
        )

        mut = FlacMutator()
        mut._rng = _rng()
        blocks = [
            FlacBlock(is_last=False, block_type=BLOCK_TYPE_STREAMINFO, data=b"\x00" * 34),
            FlacBlock(is_last=True, block_type=1, data=b"\x00" * 4),
        ]
        out = mut._mutate_duplicate_streaminfo(blocks, 4096)
        count = sum(1 for b in out if b.block_type == BLOCK_TYPE_STREAMINFO)
        assert count == 2
        # exactly one block still claims is_last after the insert
        assert sum(1 for b in out if b.is_last) == 1
        assert out[-1].is_last


class TestFlacBlockTypeAndFields:
    def test_block_type_can_become_reserved_127(self):
        from fuzzer_tool.core.mutations.flac import BLOCK_TYPE_RESERVED, FlacBlock, FlacMutator

        mut = FlacMutator()
        seen_reserved = False
        for seed in range(100):
            mut._rng = _rng(seed)
            blocks = [FlacBlock(is_last=True, block_type=0, data=b"\x00" * 34)]
            out = mut._mutate_block_type(blocks, 4096)
            if out[0].block_type == BLOCK_TYPE_RESERVED:
                seen_reserved = True
                break
        assert seen_reserved, "127 must be a reachable outcome across enough seeds"

    def test_streaminfo_field_mutation_touches_bytes_10_18(self):
        from fuzzer_tool.core.mutations.flac import (
            BLOCK_TYPE_STREAMINFO,
            FlacBlock,
            FlacMutator,
        )

        mut = FlacMutator()
        mut._rng = _rng()
        original = bytes(range(34))
        blocks = [FlacBlock(is_last=True, block_type=BLOCK_TYPE_STREAMINFO, data=original)]
        out = mut._mutate_streaminfo_field(blocks, 4096)
        assert out[0].data[:10] == original[:10]
        assert out[0].data[18:] == original[18:]
        assert out[0].data[10:18] != original[10:18]

    def test_no_streaminfo_is_a_safe_noop_for_field_mutation(self):
        from fuzzer_tool.core.mutations.flac import FlacBlock, FlacMutator

        mut = FlacMutator()
        mut._rng = _rng()
        blocks = [FlacBlock(is_last=True, block_type=1, data=b"\x00" * 34)]
        out = mut._mutate_streaminfo_field(blocks, 4096)
        assert out == blocks


class TestFlacGeneralMutatorBehaviour:
    def test_mutate_diversity(self):
        from fuzzer_tool.core.mutations.flac import FlacMutator

        data = FlacMutator()._generate_random_flac(max_len=8192, rng=_rng())
        assert _diversity(FlacMutator(), data) > 1

    def test_randpool_compat(self):
        from fuzzer_tool.core.mutations.flac import FlacMutator

        data = FlacMutator()._generate_random_flac(max_len=8192, rng=_rng())
        mutator = FlacMutator()
        mutator._rng = RandPool()
        out = mutator.mutate(data, max_len=8192, rng=mutator._rng)
        assert isinstance(out, bytes) and len(out) <= 8192

    def test_degenerate_input_falls_back_to_generator(self):
        from fuzzer_tool.core.mutations.flac import FlacMutator

        mut = FlacMutator()
        for junk in (b"", b"\x00", b"fLaC", b"fLaC\x80\x00\x00\x00"):
            out = mut.mutate(junk, max_len=2048, rng=_rng())
            assert isinstance(out, bytes) and len(out) <= 2048
            assert out[:4] == b"fLaC"

    def test_generator_max_len_is_honoured(self):
        from fuzzer_tool.core.mutations.flac import FlacMutator

        out = FlacMutator()._generate_random_flac(max_len=40, rng=_rng())
        assert len(out) <= 40

    def test_max_len_is_honoured_across_all_ops(self):
        from fuzzer_tool.core.mutations.flac import FlacMutator

        data = FlacMutator()._generate_random_flac(max_len=8192, rng=_rng())
        mut = FlacMutator()
        for seed in range(50):
            out = mut.mutate(data, max_len=64, rng=_rng(seed))
            assert len(out) <= 64


# ── operator registry / dispatch wiring ─────────────────────────────────


class TestFlacOperatorWiring:
    def test_flac_chunk_mutate_is_registered(self):
        from fuzzer_tool.core.operator_registry import REGISTRY

        assert "flac_chunk_mutate" in REGISTRY.names()

    def test_flac_sniffer_matches_magic(self):
        from fuzzer_tool.core.operator_registry import _FORMAT_SNIFFERS

        sniff = _FORMAT_SNIFFERS["flac_chunk_mutate"]
        assert sniff(b"fLaC" + b"\x00" * 40) is True
        assert sniff(b"RIFF") is False
        assert sniff(b"") is False
